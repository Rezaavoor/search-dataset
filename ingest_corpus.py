#!/usr/bin/env python3
"""Ingest an entire document corpus into the SQLite page store.

Walks a corpus directory, extracts pages from every supported file format
(PDF, DOCX, XLSX, PPTX, TXT, CSV, JSON), and stores them in the SQLite
page store.  Embeddings are NOT computed here — that is handled separately
by ``embed_corpus.py`` for better resume safety.

Features:
  - Resume-safe: skips files whose size + mtime haven't changed.
  - Error-isolated: failures are logged per-file; ingestion continues.
  - Progress tracking with ETA.
  - Configurable file-type filter, dry-run mode.
  - Handles both upper- and lower-case extensions (.PDF, .Docx, etc.).

Usage examples:
    # Ingest everything (default DB at processed/pdf_page_store.sqlite)
    python ingest_corpus.py --input-dir search-dataset/

    # Only PDFs and DOCX files
    python ingest_corpus.py --input-dir search-dataset/ --file-types pdf,docx

    # Dry run — just count files, don't write
    python ingest_corpus.py --input-dir search-dataset/ --dry-run

    # Force re-process all files (delete + re-insert)
    python ingest_corpus.py --input-dir search-dataset/ --reprocess
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from modules.config import DEFAULT_PDF_STORE_DB_NAME
from modules.db import (
    init_pdf_page_store,
    open_pdf_page_store,
    pdf_store_needs_refresh,
    upsert_file_into_store,
)
from modules.loaders import SUPPORTED_EXTENSIONS, file_type_from_path
from modules.utils import compute_rel_path_for_store

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# File extensions to always skip (regardless of --file-types)
# ---------------------------------------------------------------------------
ALWAYS_SKIP_EXTENSIONS: Set[str] = {
    ".ds_store", ".gitkeep", ".gitignore",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".svg", ".webp",
    ".ico", ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
    ".pyc", ".pyo", ".class",
    ".eml", ".mbox", ".ics", ".pages", ".step",
}

# Filename patterns to always skip
ALWAYS_SKIP_NAMES: Set[str] = {
    ".ds_store", "thumbs.db", "desktop.ini",
}


# ---------------------------------------------------------------------------
# Collect files
# ---------------------------------------------------------------------------
def collect_files(
    input_dir: Path,
    *,
    file_types: Optional[Set[str]] = None,
) -> List[Path]:
    """Walk *input_dir* recursively and return supported files.

    Args:
        input_dir: Root corpus directory.
        file_types: If provided, only include these file types (e.g. {"pdf", "docx"}).
                    If None, include all supported types.
    """
    files: List[Path] = []
    skipped_ext: Dict[str, int] = {}

    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue

        # Skip hidden files / known junk
        if p.name.lower() in ALWAYS_SKIP_NAMES:
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in ALWAYS_SKIP_EXTENSIONS:
            skipped_ext[p.suffix.lower()] = skipped_ext.get(p.suffix.lower(), 0) + 1
            continue

        ft = file_type_from_path(p)
        if ft is None:
            skipped_ext[p.suffix.lower()] = skipped_ext.get(p.suffix.lower(), 0) + 1
            continue

        if file_types and ft not in file_types:
            continue

        files.append(p)

    if skipped_ext:
        top = sorted(skipped_ext.items(), key=lambda x: -x[1])[:10]
        skip_summary = ", ".join(f"{ext}({n})" for ext, n in top)
        log.info("Skipped extensions: %s", skip_summary)

    return files


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _fmt_count(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------
def ingest(
    *,
    input_dir: Path,
    db_path: Path,
    file_types: Optional[Set[str]] = None,
    dry_run: bool = False,
    reprocess: bool = False,
    log_every: int = 50,
) -> Dict[str, int]:
    """Ingest all supported files from *input_dir* into the SQLite store.

    Returns a stats dict with keys: total, ingested, skipped, errors, pages.
    """
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # --- Collect files ---
    log.info("Scanning %s ...", input_dir)
    files = collect_files(input_dir, file_types=file_types)
    log.info("Found %d supported files", len(files))

    if not files:
        return {"total": 0, "ingested": 0, "skipped": 0, "errors": 0, "pages": 0}

    # --- File type breakdown ---
    type_counts: Dict[str, int] = {}
    for f in files:
        ft = file_type_from_path(f) or "?"
        type_counts[ft] = type_counts.get(ft, 0) + 1
    breakdown = ", ".join(f"{ft}: {n}" for ft, n in sorted(type_counts.items()))
    log.info("Breakdown: %s", breakdown)

    if dry_run:
        log.info("DRY RUN — no files will be ingested")
        return {"total": len(files), "ingested": 0, "skipped": 0, "errors": 0, "pages": 0}

    # --- Open DB ---
    conn = open_pdf_page_store(db_path)
    init_pdf_page_store(conn)
    log.info("SQLite store: %s", db_path)

    # --- Check which files already exist ---
    existing_count = conn.execute("SELECT COUNT(*) FROM pdf_page_store").fetchone()[0]
    log.info("Existing rows in store: %s", _fmt_count(existing_count))

    # --- Ingest ---
    stats = {"total": len(files), "ingested": 0, "skipped": 0, "errors": 0, "pages": 0}
    error_files: List[Tuple[str, str]] = []  # (rel_path, error_msg)
    t0 = time.monotonic()

    for idx, fpath in enumerate(files):
        rel_path = compute_rel_path_for_store(fpath, input_dir)

        # --- Progress logging ---
        if idx > 0 and idx % log_every == 0:
            elapsed = time.monotonic() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (len(files) - idx) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] %s  (%.1f files/s, ETA %s)  ingested=%d skipped=%d errors=%d pages=%d",
                idx, len(files), rel_path[:60],
                rate, _fmt_duration(remaining),
                stats["ingested"], stats["skipped"], stats["errors"], stats["pages"],
            )

        # --- Freshness check (skip if unchanged) ---
        if not reprocess:
            try:
                st = fpath.stat()
                needs = pdf_store_needs_refresh(
                    conn,
                    rel_path=rel_path,
                    size_bytes=int(st.st_size),
                    mtime_ns=int(st.st_mtime_ns),
                )
                if not needs:
                    stats["skipped"] += 1
                    continue
            except Exception:
                pass  # If stat fails, try to ingest anyway

        # --- Ingest file ---
        try:
            pages_stored = upsert_file_into_store(
                conn,
                file_path=fpath,
                base_input_dir=input_dir,
                reprocess=reprocess,
            )
            stats["ingested"] += 1
            stats["pages"] += pages_stored
        except Exception as exc:
            stats["errors"] += 1
            err_msg = f"{type(exc).__name__}: {exc}"
            error_files.append((rel_path, err_msg))
            log.warning("FAILED [%s]: %s", rel_path[:80], err_msg)

    elapsed = time.monotonic() - t0
    conn.close()

    # --- Summary ---
    log.info("=" * 80)
    log.info("INGESTION COMPLETE in %s", _fmt_duration(elapsed))
    log.info(
        "  total=%d  ingested=%d  skipped=%d  errors=%d  pages=%d",
        stats["total"], stats["ingested"], stats["skipped"],
        stats["errors"], stats["pages"],
    )

    final_count_msg = ""
    try:
        conn2 = open_pdf_page_store(db_path)
        final_count = conn2.execute("SELECT COUNT(*) FROM pdf_page_store").fetchone()[0]
        by_type = conn2.execute(
            "SELECT file_type, COUNT(*), SUM(content_chars) "
            "FROM pdf_page_store GROUP BY file_type ORDER BY COUNT(*) DESC"
        ).fetchall()
        conn2.close()
        log.info("  Total rows in store: %s", _fmt_count(final_count))
        for ft, count, chars in by_type:
            log.info("    %-6s  %6d rows  %12s chars", ft, count, f"{chars:,}")
    except Exception:
        pass

    # --- Error report ---
    if error_files:
        log.info("")
        log.info("Files with errors (%d):", len(error_files))
        for rel_path, err_msg in error_files[:50]:
            log.info("  %s — %s", rel_path[:80], err_msg)
        if len(error_files) > 50:
            log.info("  ... and %d more", len(error_files) - 50)

        # Also write errors to a log file
        err_log_path = db_path.parent / "ingest_errors.log"
        try:
            with open(err_log_path, "w") as f:
                for rel_path, err_msg in error_files:
                    f.write(f"{rel_path}\t{err_msg}\n")
            log.info("  Error log saved: %s", err_log_path)
        except Exception:
            pass

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest a document corpus into the SQLite page store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir",
        type=str,
        default=str(SCRIPT_DIR / "search-dataset"),
        help="Root corpus directory (default: search-dataset/)",
    )
    p.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite DB path (default: processed/pdf_page_store.sqlite)",
    )
    p.add_argument(
        "--file-types",
        type=str,
        default=None,
        help="Comma-separated file types to include (e.g. 'pdf,docx,xlsx'). Default: all supported.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files without ingesting.",
    )
    p.add_argument(
        "--reprocess",
        action="store_true",
        help="Force re-process all files (delete + re-insert).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N files (default: 50).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (SCRIPT_DIR / "processed" / DEFAULT_PDF_STORE_DB_NAME).resolve()

    file_types: Optional[Set[str]] = None
    if args.file_types:
        file_types = {ft.strip().lower() for ft in args.file_types.split(",")}
        log.info("Filtering to file types: %s", file_types)

    ingest(
        input_dir=input_dir,
        db_path=db_path,
        file_types=file_types,
        dry_run=args.dry_run,
        reprocess=args.reprocess,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
