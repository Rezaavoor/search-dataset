#!/usr/bin/env python3
"""Visualize document spread in embedding space with PCA.

This script reads page-level embeddings from the SQLite corpus, aggregates them
into one centroid per document, runs PCA on the document centroids, and writes:

  - an interactive-ish HTML scatter plot (SVG with native hover tooltips)
  - a CSV file with document coordinates and page counts
  - a short Markdown summary with explained variance and corpus stats

It also overlays a sampled set of individual page embeddings in the same PCA
projection so you can compare document-level spread against the page cloud.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "processed" / "pdf_page_store.sqlite"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_EMBED_COLUMN = "emb__text_embedding_3_large"
DEFAULT_EMBED_DIMS_COLUMN = "emb_dims__text_embedding_3_large"


@dataclass
class DocumentRecord:
    rel_path: str
    filename: str
    page_count: int
    centroid: np.ndarray


@dataclass
class SampledPage:
    rel_path: str
    filename: str
    page_number: int
    embedding: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PCA visualization of document embeddings from SQLite."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample-pages",
        type=int,
        default=8000,
        help="Number of page embeddings to reservoir-sample for the page cloud.",
    )
    parser.add_argument(
        "--embedding-column",
        default=DEFAULT_EMBED_COLUMN,
        help=f"Embedding BLOB column to read (default: {DEFAULT_EMBED_COLUMN})",
    )
    parser.add_argument(
        "--embedding-dims-column",
        default=DEFAULT_EMBED_DIMS_COLUMN,
        help=(
            "Embedding dimension column paired with --embedding-column "
            f"(default: {DEFAULT_EMBED_DIMS_COLUMN})"
        ),
    )
    return parser.parse_args()


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec64 = np.asarray(vec, dtype=np.float64)
    if not np.isfinite(vec64).all():
        raise ValueError("Embedding contains non-finite values.")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        norm = float(np.sqrt(np.dot(vec64, vec64)))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("Embedding has zero or invalid norm.")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        normalized = (vec64 / norm).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embedding contains non-finite values.")
    return normalized


def reservoir_append(
    samples: List[SampledPage],
    sample: SampledPage,
    sample_cap: int,
    seen: int,
    rng: np.random.Generator,
) -> None:
    if sample_cap <= 0:
        return
    if len(samples) < sample_cap:
        samples.append(sample)
        return
    replace_idx = int(rng.integers(0, seen))
    if replace_idx < sample_cap:
        samples[replace_idx] = sample


def load_document_centroids(
    db_path: Path,
    embedding_column: str,
    embedding_dims_column: str,
    sample_pages: int,
) -> tuple[List[DocumentRecord], List[SampledPage], int, int]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(
        f"""
        SELECT rel_path, filename, page_number, {embedding_column} AS embedding_blob,
               {embedding_dims_column} AS embedding_dims
        FROM pdf_page_store
        WHERE {embedding_column} IS NOT NULL
        ORDER BY rel_path, page_number
        """
    )

    rng = np.random.default_rng(42)
    docs: List[DocumentRecord] = []
    page_samples: List[SampledPage] = []

    current_rel_path: str | None = None
    current_filename: str | None = None
    current_sum: np.ndarray | None = None
    current_count = 0
    dim: int | None = None
    skipped_rows = 0
    seen_pages = 0

    def flush_current() -> None:
        nonlocal current_rel_path, current_filename, current_sum, current_count
        if current_rel_path is None or current_filename is None or current_sum is None:
            return
        centroid = current_sum / max(current_count, 1)
        docs.append(
            DocumentRecord(
                rel_path=current_rel_path,
                filename=current_filename,
                page_count=current_count,
                centroid=_normalize(centroid.astype(np.float32, copy=False)),
            )
        )
        current_rel_path = None
        current_filename = None
        current_sum = None
        current_count = 0

    for row in cursor:
        blob = row["embedding_blob"]
        dims = row["embedding_dims"]
        if blob is None or dims is None:
            skipped_rows += 1
            continue

        raw_vec = np.frombuffer(blob, dtype=np.float32)
        if len(raw_vec) != int(dims):
            skipped_rows += 1
            continue
        try:
            vec = _normalize(raw_vec)
        except ValueError:
            skipped_rows += 1
            continue

        if dim is None:
            dim = int(dims)

        rel_path = str(row["rel_path"])
        filename = str(row["filename"])
        page_number = int(row["page_number"])

        if current_rel_path != rel_path:
            flush_current()
            current_rel_path = rel_path
            current_filename = filename
            current_sum = np.zeros(dim, dtype=np.float64)
            current_count = 0

        current_sum += vec
        current_count += 1
        seen_pages += 1

        reservoir_append(
            page_samples,
            SampledPage(
                rel_path=rel_path,
                filename=filename,
                page_number=page_number,
                embedding=vec.copy(),
            ),
            sample_cap=sample_pages,
            seen=seen_pages,
            rng=rng,
        )

    flush_current()
    conn.close()
    return docs, page_samples, seen_pages, skipped_rows


def run_pca(x: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for PCA, got shape {x.shape}")
    if x.shape[0] < n_components:
        raise ValueError(
            f"Need at least {n_components} rows for PCA, got {x.shape[0]}"
        )

    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0, keepdims=True)
    centered = x64 - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        projected = centered @ components
    if not np.isfinite(projected).all():
        raise ValueError("PCA projection produced non-finite values.")

    eigenvalues = (singular_values ** 2) / max(x.shape[0] - 1, 1)
    explained_ratio = eigenvalues[:n_components] / np.maximum(eigenvalues.sum(), 1e-12)
    return projected.astype(np.float32), components.astype(np.float32), explained_ratio.astype(np.float32)


def project_with_pca(
    x: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    mean64 = np.asarray(mean, dtype=np.float64)
    components64 = np.asarray(components, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        projected = ((x64 - mean64) @ components64).astype(np.float32)
    if not np.isfinite(projected).all():
        raise ValueError("Page projection produced non-finite values.")
    return projected


def scale_points(
    coords: np.ndarray,
    width: int,
    height: int,
    padding: int,
) -> np.ndarray:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    x = padding + (coords[:, 0] - mins[0]) / spans[0] * (width - 2 * padding)
    y = height - padding - (coords[:, 1] - mins[1]) / spans[1] * (height - 2 * padding)
    return np.column_stack([x, y]).astype(np.float32)


def color_for_page_count(page_count: int, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.5
    else:
        t = (math.log1p(page_count) - lo) / (hi - lo)
    t = min(max(t, 0.0), 1.0)

    stops = [
        (40, 120, 181),
        (98, 190, 154),
        (247, 203, 77),
        (230, 126, 34),
        (192, 57, 43),
    ]
    scaled = t * (len(stops) - 1)
    idx = min(int(scaled), len(stops) - 2)
    frac = scaled - idx
    a = stops[idx]
    b = stops[idx + 1]
    rgb = tuple(int(round(a[i] + (b[i] - a[i]) * frac)) for i in range(3))
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def build_svg(
    page_points: np.ndarray,
    doc_points: np.ndarray,
    docs: Sequence[DocumentRecord],
    width: int = 1200,
    height: int = 900,
    padding: int = 70,
) -> str:
    all_points = doc_points if len(page_points) == 0 else np.vstack([doc_points, page_points])
    scaled_all = scale_points(all_points, width=width, height=height, padding=padding)

    doc_scaled = scaled_all[: len(doc_points)]
    page_scaled = scaled_all[len(doc_points) :]

    page_counts = [doc.page_count for doc in docs]
    log_counts = [math.log1p(c) for c in page_counts]
    lo = min(log_counts) if log_counts else 0.0
    hi = max(log_counts) if log_counts else 1.0

    page_cloud_parts = []
    for x, y in page_scaled:
        page_cloud_parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.7" fill="#9aa4b2" fill-opacity="0.16" />'
        )

    doc_parts = []
    for idx, (doc, (x, y)) in enumerate(zip(docs, doc_scaled, strict=True), start=1):
        radius = 3.0 + min(9.0, math.sqrt(doc.page_count) * 0.18)
        fill = color_for_page_count(doc.page_count, lo=lo, hi=hi)
        title = html.escape(
            f"{idx}. {doc.filename}\n"
            f"rel_path: {doc.rel_path}\n"
            f"pages: {doc.page_count}"
        )
        doc_parts.append(
            (
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                f'fill="{fill}" fill-opacity="0.82" stroke="#0f172a" stroke-opacity="0.25" '
                f'stroke-width="0.8"><title>{title}</title></circle>'
            )
        )

    axes = (
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" '
        f'y2="{height - padding}" stroke="#334155" stroke-width="1.2" />'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" '
        f'y2="{height - padding}" stroke="#334155" stroke-width="1.2" />'
    )

    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="PCA scatter plot of document embeddings">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" />
  <g>{axes}</g>
  <g>{''.join(page_cloud_parts)}</g>
  <g>{''.join(doc_parts)}</g>
  <text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" fill="#0f172a" font-size="18">PC1</text>
  <text x="24" y="{height / 2:.1f}" text-anchor="middle" fill="#0f172a" font-size="18"
        transform="rotate(-90 24 {height / 2:.1f})">PC2</text>
</svg>
""".strip()


def write_csv(output_path: Path, docs: Sequence[DocumentRecord], coords: np.ndarray) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_path", "filename", "page_count", "pc1", "pc2"])
        for doc, (pc1, pc2) in zip(docs, coords, strict=True):
            writer.writerow([doc.rel_path, doc.filename, doc.page_count, float(pc1), float(pc2)])


def write_summary(
    output_path: Path,
    docs: Sequence[DocumentRecord],
    total_pages: int,
    sampled_pages: int,
    skipped_rows: int,
    explained_ratio: np.ndarray,
) -> None:
    page_counts = np.array([doc.page_count for doc in docs], dtype=np.int32)
    median_pages = float(np.median(page_counts)) if len(page_counts) else 0.0
    p90_pages = float(np.percentile(page_counts, 90)) if len(page_counts) else 0.0

    output_path.write_text(
        "\n".join(
            [
                "# Document Embedding PCA",
                "",
                f"- Documents projected: {len(docs):,}",
                f"- Embedded pages processed: {total_pages:,}",
                f"- Sampled page points shown in background: {sampled_pages:,}",
                f"- Rows skipped due to malformed/missing embeddings: {skipped_rows:,}",
                f"- Median pages per document: {median_pages:.1f}",
                f"- 90th percentile pages per document: {p90_pages:.1f}",
                f"- PCA explained variance: PC1={explained_ratio[0] * 100:.2f}% | PC2={explained_ratio[1] * 100:.2f}%",
                "",
                "Interpretation:",
                "- Each colored point is one document centroid computed as the mean of its L2-normalized page embeddings.",
                "- Point size and color both increase with document page count.",
                "- Gray points are a random sample of individual pages projected into the same PCA basis.",
                "- PCA is a lossy 2D projection, so use it for coarse distribution structure rather than exact distance judgments.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_html(
    output_path: Path,
    svg_markup: str,
    docs: Sequence[DocumentRecord],
    total_pages: int,
    sampled_pages: int,
    explained_ratio: np.ndarray,
    embedding_column: str,
) -> None:
    page_counts = [doc.page_count for doc in docs]
    median_pages = np.median(page_counts) if page_counts else 0.0
    max_pages = max(page_counts) if page_counts else 0

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Document Embedding PCA</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #e2e8f0;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --border: #cbd5e1;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 32px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}
    .card {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    }}
    .label {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #64748b;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .plot {{
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    .notes {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.55;
    }}
    code {{
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    @media (max-width: 900px) {{
      .stats, .notes {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 640px) {{
      .stats, .notes {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Document Embedding PCA</h1>
    <p>
      Each colored point is a document centroid built from page-level embeddings in
      <code>{html.escape(embedding_column)}</code>. Hover points to inspect filenames.
      The gray background cloud is a random sample of individual pages projected into
      the same PCA space.
    </p>
    <div class="stats">
      <div class="card"><span class="label">Documents</span><span class="value">{len(docs):,}</span></div>
      <div class="card"><span class="label">Pages</span><span class="value">{total_pages:,}</span></div>
      <div class="card"><span class="label">Sampled Pages</span><span class="value">{sampled_pages:,}</span></div>
      <div class="card"><span class="label">Median Pages / Doc</span><span class="value">{median_pages:.1f}</span></div>
      <div class="card"><span class="label">Max Pages / Doc</span><span class="value">{max_pages:,}</span></div>
      <div class="card"><span class="label">PC1 Variance</span><span class="value">{explained_ratio[0] * 100:.2f}%</span></div>
      <div class="card"><span class="label">PC2 Variance</span><span class="value">{explained_ratio[1] * 100:.2f}%</span></div>
      <div class="card"><span class="label">What To Look For</span><span class="value" style="font-size:18px">clusters, outliers, density</span></div>
    </div>
    <div class="plot">
      {svg_markup}
    </div>
    <div class="notes">
      <div class="card">
        <span class="label">Interpretation</span>
        <ul>
          <li>Documents close together have similar centroid embeddings in this 2D projection.</li>
          <li>Larger, warmer points correspond to longer documents.</li>
          <li>The gray cloud shows where individual pages sit relative to the document centroids.</li>
        </ul>
      </div>
      <div class="card">
        <span class="label">Method</span>
        <ul>
          <li>Each page embedding is L2-normalized before document aggregation.</li>
          <li>A document centroid is the mean of its normalized page embeddings.</li>
          <li>PCA is fit on document centroids, then sampled pages are projected with the same basis.</li>
        </ul>
      </div>
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from: {args.db}")
    docs, page_samples, total_pages, skipped_rows = load_document_centroids(
        db_path=args.db,
        embedding_column=args.embedding_column,
        embedding_dims_column=args.embedding_dims_column,
        sample_pages=args.sample_pages,
    )
    if len(docs) < 2:
        raise RuntimeError("Need at least two documents with embeddings to run PCA.")

    doc_matrix = np.stack([doc.centroid for doc in docs]).astype(np.float32)
    doc_mean = doc_matrix.mean(axis=0, keepdims=True)
    doc_coords, components, explained_ratio = run_pca(doc_matrix, n_components=2)

    if page_samples:
        sample_matrix = np.stack([page.embedding for page in page_samples]).astype(np.float32)
        page_coords = project_with_pca(sample_matrix, mean=doc_mean, components=components)
    else:
        page_coords = np.empty((0, 2), dtype=np.float32)

    html_path = args.output_dir / "document_embedding_pca.html"
    csv_path = args.output_dir / "document_embedding_pca_points.csv"
    summary_path = args.output_dir / "document_embedding_pca_summary.md"

    svg_markup = build_svg(page_coords, doc_coords, docs)
    write_html(
        output_path=html_path,
        svg_markup=svg_markup,
        docs=docs,
        total_pages=total_pages,
        sampled_pages=len(page_samples),
        explained_ratio=explained_ratio,
        embedding_column=args.embedding_column,
    )
    write_csv(csv_path, docs, doc_coords)
    write_summary(
        output_path=summary_path,
        docs=docs,
        total_pages=total_pages,
        sampled_pages=len(page_samples),
        skipped_rows=skipped_rows,
        explained_ratio=explained_ratio,
    )

    print(f"Wrote HTML visualization: {html_path}")
    print(f"Wrote PCA points CSV   : {csv_path}")
    print(f"Wrote summary          : {summary_path}")
    print(
        "Explained variance: "
        f"PC1={explained_ratio[0] * 100:.2f}% | PC2={explained_ratio[1] * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
