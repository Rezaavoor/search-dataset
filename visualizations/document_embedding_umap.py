#!/usr/bin/env python3
"""Visualize document spread in embedding space with UMAP.

This script mirrors the PCA visualization pipeline but uses UMAP instead of a
linear projection. It aggregates page-level embeddings into one centroid per
document, fits UMAP on those document centroids, and overlays a sampled set of
page embeddings transformed into the same 2D space.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from document_embedding_pca import (
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_COLUMN,
    DEFAULT_EMBED_DIMS_COLUMN,
    DEFAULT_OUTPUT_DIR,
    DocumentRecord,
    load_document_centroids,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a UMAP visualization of document embeddings from SQLite."
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
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=25,
        help="UMAP neighborhood size. Larger values emphasize broader structure.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.08,
        help="UMAP minimum distance. Lower values create tighter clusters.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Distance metric used by UMAP (default: cosine).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic UMAP output.",
    )
    return parser.parse_args()


def run_umap(
    doc_matrix: np.ndarray,
    page_matrix: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import umap
    except ImportError as exc:
        raise SystemExit(
            "UMAP requires the optional 'umap-learn' package. "
            "Install it with: python -m pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        transform_seed=random_state,
        low_memory=True,
        verbose=True,
    )
    doc_coords = reducer.fit_transform(doc_matrix)
    if page_matrix.size == 0:
        page_coords = np.empty((0, 2), dtype=np.float32)
    else:
        page_coords = reducer.transform(page_matrix)
    return doc_coords.astype(np.float32), page_coords.astype(np.float32)


def reduce_for_browser(doc_matrix: np.ndarray, n_components: int = 20) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    n_components = max(2, min(n_components, doc_matrix.shape[0], doc_matrix.shape[1]))
    reducer = PCA(n_components=n_components, random_state=42)
    reduced = reducer.fit_transform(doc_matrix)
    return reduced.astype(np.float32), reducer.explained_variance_ratio_.astype(np.float32)


def write_summary(
    output_path: Path,
    docs: Sequence[DocumentRecord],
    total_pages: int,
    sampled_pages: int,
    skipped_rows: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
) -> None:
    page_counts = np.array([doc.page_count for doc in docs], dtype=np.int32)
    median_pages = float(np.median(page_counts)) if len(page_counts) else 0.0
    p90_pages = float(np.percentile(page_counts, 90)) if len(page_counts) else 0.0

    output_path.write_text(
        "\n".join(
            [
                "# Document Embedding UMAP",
                "",
                f"- Documents projected: {len(docs):,}",
                f"- Embedded pages processed: {total_pages:,}",
                f"- Sampled page points shown in background: {sampled_pages:,}",
                f"- Rows skipped due to malformed/missing embeddings: {skipped_rows:,}",
                f"- Median pages per document: {median_pages:.1f}",
                f"- 90th percentile pages per document: {p90_pages:.1f}",
                f"- UMAP metric: {metric}",
                f"- UMAP n_neighbors: {n_neighbors}",
                f"- UMAP min_dist: {min_dist}",
                "",
                "Interpretation:",
                "- Each colored point is one document centroid computed as the mean of its L2-normalized page embeddings.",
                "- UMAP emphasizes local neighborhoods, so nearby points are usually more meaningful than large global distances.",
                "- Gaps and island-like clusters can indicate real local structure, but axis values themselves have no direct semantic meaning.",
                "- The gray points are sampled pages transformed with the same fitted UMAP model.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_html(
    output_path: Path,
    docs: Sequence[DocumentRecord],
    total_pages: int,
    sampled_pages: int,
    embedding_column: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    initial_coords: np.ndarray,
    browser_features: np.ndarray,
    browser_feature_variance: np.ndarray,
) -> None:
    page_counts = [doc.page_count for doc in docs]
    median_pages = np.median(page_counts) if page_counts else 0.0
    max_pages = max(page_counts) if page_counts else 0
    feature_dims = int(browser_features.shape[1]) if browser_features.ndim == 2 else 0

    doc_payload = [
        {
            "filename": doc.filename,
            "rel_path": doc.rel_path,
            "page_count": int(doc.page_count),
        }
        for doc in docs
    ]
    data_payload = {
        "docs": doc_payload,
        "initialCoords": np.round(initial_coords, 6).tolist(),
        "browserFeatures": np.round(browser_features, 6).tolist(),
        "featureVariance": np.round(browser_feature_variance, 6).tolist(),
        "defaults": {
            "nNeighbors": int(n_neighbors),
            "minDist": float(min_dist),
            "metric": metric,
        },
    }
    data_json = json.dumps(data_payload, separators=(",", ":")).replace("</", "<\\/")

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Document Embedding UMAP</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #ecfeff;
      --panel: #ffffff;
      --ink: #082f49;
      --muted: #155e75;
      --border: #a5f3fc;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #ecfeff 0%, #f8fafc 100%);
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
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 12px 30px rgba(8, 47, 73, 0.07);
    }}
    .label {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #0e7490;
      margin-bottom: 6px;
    }}
    .value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .plot {{
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 18px 40px rgba(8, 47, 73, 0.08);
    }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin: 0 0 18px;
    }}
    .control {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .control-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }}
    .control-value {{
      font-weight: 700;
      color: #0f172a;
      min-width: 52px;
      text-align: right;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: #0891b2;
    }}
    .status {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .status strong {{
      color: var(--ink);
    }}
    .plot-shell {{
      position: relative;
    }}
    .plot-shell svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      display: none;
      max-width: 320px;
      background: rgba(8, 47, 73, 0.94);
      color: #f8fafc;
      padding: 10px 12px;
      border-radius: 10px;
      font-size: 13px;
      line-height: 1.45;
      box-shadow: 0 18px 40px rgba(8, 47, 73, 0.18);
      white-space: pre-line;
      z-index: 3;
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
      background: #ecfeff;
      padding: 2px 6px;
      border-radius: 6px;
    }}
    @media (max-width: 900px) {{
      .stats, .notes, .controls {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 640px) {{
      .stats, .notes, .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Document Embedding UMAP</h1>
    <p>
      Each colored point is a document centroid built from page-level embeddings in
      <code>{html.escape(embedding_column)}</code>. Hover points to inspect filenames.
      This interactive page recomputes UMAP in the browser as you adjust
      <code>n_neighbors</code> and <code>min_dist</code>, so it focuses on document
      centroids only to stay responsive.
    </p>
    <div class="stats">
      <div class="card"><span class="label">Documents</span><span class="value">{len(docs):,}</span></div>
      <div class="card"><span class="label">Pages</span><span class="value">{total_pages:,}</span></div>
      <div class="card"><span class="label">Sampled Pages</span><span class="value">{sampled_pages:,}</span></div>
      <div class="card"><span class="label">Median Pages / Doc</span><span class="value">{median_pages:.1f}</span></div>
      <div class="card"><span class="label">Max Pages / Doc</span><span class="value">{max_pages:,}</span></div>
      <div class="card"><span class="label">n_neighbors</span><span class="value">{n_neighbors}</span></div>
      <div class="card"><span class="label">min_dist</span><span class="value">{min_dist:.2f}</span></div>
      <div class="card"><span class="label">metric</span><span class="value" style="font-size:18px">{html.escape(metric)}</span></div>
    </div>
    <div class="plot">
      <div class="controls">
        <div class="card control">
          <span class="label">n_neighbors</span>
          <div class="control-row"><span>local vs global</span><span id="n-neighbors-value" class="control-value">{n_neighbors}</span></div>
          <input id="n-neighbors" type="range" min="2" max="200" step="1" value="{n_neighbors}" />
        </div>
        <div class="card control">
          <span class="label">min_dist</span>
          <div class="control-row"><span>cluster tightness</span><span id="min-dist-value" class="control-value">{min_dist:.2f}</span></div>
          <input id="min-dist" type="range" min="0" max="0.99" step="0.01" value="{min_dist}" />
        </div>
        <div class="card control">
          <span class="label">Browser Features</span>
          <div class="value" style="font-size:20px">{feature_dims}</div>
          <div style="color: var(--muted);">UMAP runs on PCA-reduced document centroids for speed.</div>
        </div>
        <div class="card control">
          <span class="label">Live Behavior</span>
          <div class="value" style="font-size:20px">auto-rerun</div>
          <div style="color: var(--muted);">Changing a slider schedules a fresh layout immediately after the current run finishes.</div>
        </div>
      </div>
      <div class="plot-shell">
        <div id="tooltip" class="tooltip"></div>
        <svg id="plot-svg" viewBox="0 0 1200 900" role="img" aria-label="Interactive UMAP scatter plot of document embeddings"></svg>
      </div>
      <div class="status">
        <div><strong id="status-label">Ready.</strong> <span id="status-detail">Showing the default layout.</span></div>
        <div id="run-meta">Reduced dimensions capture {(float(browser_feature_variance.sum()) * 100):.2f}% of variance before browser-side UMAP.</div>
      </div>
    </div>
    <div class="notes">
      <div class="card">
        <span class="label">Interpretation</span>
        <ul>
          <li>Nearby points usually indicate documents with similar local neighborhoods in embedding space.</li>
          <li>UMAP often separates clusters more clearly than PCA, but the axes themselves are arbitrary.</li>
          <li>Large distances between islands can be suggestive, but local grouping is the main thing to trust.</li>
        </ul>
      </div>
      <div class="card">
        <span class="label">Method</span>
        <ul>
          <li>Each page embedding is L2-normalized before document aggregation.</li>
          <li>A document centroid is the mean of its normalized page embeddings.</li>
          <li>This page first reduces document centroids to {feature_dims} dimensions with PCA, then reruns UMAP in your browser.</li>
        </ul>
      </div>
    </div>
  </div>
  <script id="umap-data" type="application/json">{data_json}</script>
  <script src="vendor/umap-js.min.js"></script>
  <script>
    const data = JSON.parse(document.getElementById('umap-data').textContent);
    const docs = data.docs;
    const features = data.browserFeatures;
    const defaults = data.defaults;
    const UMAPCtor = window.UMAP && window.UMAP.UMAP ? window.UMAP.UMAP : null;

    const svg = document.getElementById('plot-svg');
    const tooltip = document.getElementById('tooltip');
    const neighborsInput = document.getElementById('n-neighbors');
    const minDistInput = document.getElementById('min-dist');
    const neighborsValue = document.getElementById('n-neighbors-value');
    const minDistValue = document.getElementById('min-dist-value');
    const statusLabel = document.getElementById('status-label');
    const statusDetail = document.getElementById('status-detail');
    const runMeta = document.getElementById('run-meta');

    const WIDTH = 1200;
    const HEIGHT = 900;
    const PADDING = 70;

    let currentToken = 0;
    let running = false;
    let pending = null;

    function mulberry32(seed) {{
      let t = seed >>> 0;
      return function() {{
        t += 0x6D2B79F5;
        let r = Math.imul(t ^ (t >>> 15), 1 | t);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
      }};
    }}

    function colorForPageCount(pageCount, lo, hi) {{
      let t = 0.5;
      if (hi > lo) {{
        t = (Math.log1p(pageCount) - lo) / (hi - lo);
      }}
      t = Math.max(0, Math.min(1, t));
      const stops = [
        [40, 120, 181],
        [98, 190, 154],
        [247, 203, 77],
        [230, 126, 34],
        [192, 57, 43],
      ];
      const scaled = t * (stops.length - 1);
      const idx = Math.min(Math.floor(scaled), stops.length - 2);
      const frac = scaled - idx;
      const a = stops[idx];
      const b = stops[idx + 1];
      const rgb = a.map((v, i) => Math.round(v + (b[i] - v) * frac));
      return `rgb(${{rgb[0]}}, ${{rgb[1]}}, ${{rgb[2]}})`;
    }}

    function radiusForPageCount(pageCount) {{
      return 3 + Math.min(9, Math.sqrt(pageCount) * 0.18);
    }}

    function scalePoints(coords) {{
      const xs = coords.map((d) => d[0]);
      const ys = coords.map((d) => d[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const spanX = Math.max(maxX - minX, 1e-9);
      const spanY = Math.max(maxY - minY, 1e-9);
      return coords.map(([x, y]) => {{
        const sx = PADDING + ((x - minX) / spanX) * (WIDTH - 2 * PADDING);
        const sy = HEIGHT - PADDING - ((y - minY) / spanY) * (HEIGHT - 2 * PADDING);
        return [sx, sy];
      }});
    }}

    function render(coords) {{
      const scaled = scalePoints(coords);
      const logCounts = docs.map((d) => Math.log1p(d.page_count));
      const lo = Math.min(...logCounts);
      const hi = Math.max(...logCounts);

      const parts = [];
      parts.push(`<rect x="0" y="0" width="${{WIDTH}}" height="${{HEIGHT}}" fill="#f8fafc"></rect>`);
      parts.push(`<line x1="${{PADDING}}" y1="${{HEIGHT - PADDING}}" x2="${{WIDTH - PADDING}}" y2="${{HEIGHT - PADDING}}" stroke="#334155" stroke-width="1.2"></line>`);
      parts.push(`<line x1="${{PADDING}}" y1="${{PADDING}}" x2="${{PADDING}}" y2="${{HEIGHT - PADDING}}" stroke="#334155" stroke-width="1.2"></line>`);

      scaled.forEach(([x, y], idx) => {{
        const doc = docs[idx];
        const fill = colorForPageCount(doc.page_count, lo, hi);
        const radius = radiusForPageCount(doc.page_count);
        const safeTitle = `${{doc.filename}}\\nrel_path: ${{doc.rel_path}}\\npages: ${{doc.page_count}}`;
        parts.push(
          `<circle class="doc-point" data-idx="${{idx}}" cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="${{radius.toFixed(2)}}" fill="${{fill}}" fill-opacity="0.84" stroke="#082f49" stroke-opacity="0.28" stroke-width="0.8"><title>${{safeTitle}}</title></circle>`
        );
      }});

      parts.push(`<text x="${{WIDTH / 2}}" y="${{HEIGHT - 18}}" text-anchor="middle" fill="#082f49" font-size="18">UMAP-1</text>`);
      parts.push(`<text x="24" y="${{HEIGHT / 2}}" text-anchor="middle" fill="#082f49" font-size="18" transform="rotate(-90 24 ${{HEIGHT / 2}})">UMAP-2</text>`);
      svg.innerHTML = parts.join('');

      svg.querySelectorAll('.doc-point').forEach((node) => {{
        node.addEventListener('mousemove', (event) => {{
          const idx = Number(node.dataset.idx);
          const doc = docs[idx];
          tooltip.style.display = 'block';
          tooltip.style.left = `${{event.offsetX + 20}}px`;
          tooltip.style.top = `${{event.offsetY + 20}}px`;
          tooltip.textContent = `${{doc.filename}}\\nrel_path: ${{doc.rel_path}}\\npages: ${{doc.page_count}}`;
        }});
        node.addEventListener('mouseleave', () => {{
          tooltip.style.display = 'none';
        }});
      }});
    }}

    async function runLayout(nNeighbors, minDist) {{
      const token = ++currentToken;
      running = true;
      statusLabel.textContent = 'Running UMAP...';
      statusDetail.textContent = `Recomputing document layout with n_neighbors=${{nNeighbors}} and min_dist=${{minDist.toFixed(2)}}.`;

      if (!UMAPCtor) {{
        statusLabel.textContent = 'UMAP failed to load.';
        statusDetail.textContent = 'The local browser bundle was not available, so live controls are disabled.';
        running = false;
        return;
      }}

      const umap = new UMAPCtor({{
        nComponents: 2,
        nNeighbors,
        minDist,
        random: mulberry32(42),
      }});

      const startedAt = performance.now();
      const coords = await umap.fitAsync(features, (epoch) => {{
        if (token !== currentToken) {{
          return false;
        }}
        if (epoch % 20 === 0) {{
          statusDetail.textContent = `Epoch ${{epoch}} in progress for n_neighbors=${{nNeighbors}}, min_dist=${{minDist.toFixed(2)}}.`;
        }}
        return true;
      }});

      if (token !== currentToken) {{
        return;
      }}

      render(coords);
      const seconds = ((performance.now() - startedAt) / 1000).toFixed(2);
      statusLabel.textContent = 'Ready.';
      statusDetail.textContent = `Rendered in ${{seconds}}s with n_neighbors=${{nNeighbors}} and min_dist=${{minDist.toFixed(2)}}.`;
      runMeta.textContent = `Browser-side UMAP on ${{docs.length.toLocaleString()}} documents using ${{features[0].length}} PCA-reduced features.`;
      running = false;

      if (pending) {{
        const next = pending;
        pending = null;
        runLayout(next.nNeighbors, next.minDist);
      }}
    }}

    function scheduleRun() {{
      const nNeighbors = Number(neighborsInput.value);
      const minDist = Number(minDistInput.value);
      neighborsValue.textContent = String(nNeighbors);
      minDistValue.textContent = minDist.toFixed(2);

      if (running) {{
        pending = {{ nNeighbors, minDist }};
        statusLabel.textContent = 'Queued...';
        statusDetail.textContent = `Will rerun after the current layout finishes with n_neighbors=${{nNeighbors}} and min_dist=${{minDist.toFixed(2)}}.`;
        currentToken += 1;
        return;
      }}
      runLayout(nNeighbors, minDist);
    }}

    neighborsInput.addEventListener('input', scheduleRun);
    minDistInput.addEventListener('input', scheduleRun);

    render(data.initialCoords);
    neighborsValue.textContent = String(defaults.nNeighbors);
    minDistValue.textContent = Number(defaults.minDist).toFixed(2);
    runMeta.textContent = `Browser-side UMAP on ${{docs.length.toLocaleString()}} documents using ${{features[0].length}} PCA-reduced features.`;
    if (!UMAPCtor) {{
      statusLabel.textContent = 'UMAP failed to load.';
      statusDetail.textContent = 'The page rendered, but the local UMAP bundle is missing so the controls will not rerun the layout.';
    }}
  </script>
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
        raise RuntimeError("Need at least two documents with embeddings to run UMAP.")

    doc_matrix = np.stack([doc.centroid for doc in docs]).astype(np.float32)
    page_matrix = (
        np.stack([page.embedding for page in page_samples]).astype(np.float32)
        if page_samples
        else np.empty((0, doc_matrix.shape[1]), dtype=np.float32)
    )

    doc_coords, _page_coords = run_umap(
        doc_matrix=doc_matrix,
        page_matrix=page_matrix,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    browser_features, browser_feature_variance = reduce_for_browser(doc_matrix, n_components=20)

    html_path = args.output_dir / "document_embedding_umap.html"
    csv_path = args.output_dir / "document_embedding_umap_points.csv"
    summary_path = args.output_dir / "document_embedding_umap_summary.md"

    write_html(
        output_path=html_path,
        docs=docs,
        total_pages=total_pages,
        sampled_pages=len(page_samples),
        embedding_column=args.embedding_column,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        initial_coords=doc_coords,
        browser_features=browser_features,
        browser_feature_variance=browser_feature_variance,
    )
    write_csv(csv_path, docs, doc_coords)
    write_summary(
        output_path=summary_path,
        docs=docs,
        total_pages=total_pages,
        sampled_pages=len(page_samples),
        skipped_rows=skipped_rows,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
    )

    print(f"Wrote HTML visualization: {html_path}")
    print(f"Wrote UMAP points CSV  : {csv_path}")
    print(f"Wrote summary          : {summary_path}")
    print(
        "UMAP parameters: "
        f"n_neighbors={args.n_neighbors} | min_dist={args.min_dist} | metric={args.metric}"
    )


if __name__ == "__main__":
    main()
