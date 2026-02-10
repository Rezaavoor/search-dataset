#!/usr/bin/env python3
"""Generate document profiles for all files in the SQLite page store.

Profiles are compact LLM-generated metadata per FILE (not per page):
title_guess, doc_type, summary, topics, key_entities, likely_user_intents.
They are stored in `pdf_profile_json` on page 1 of each file and reused
by the query generation pipeline for realistic search queries.

Features:
  - Resume-safe: skips files that already have a profile (with matching model tag).
  - Multi-endpoint parallelism for Azure OpenAI (up to N concurrent workers).
  - Exponential backoff with jitter on rate-limit / throttle errors.
  - Graceful shutdown on Ctrl+C (finishes current files, commits, exits).
  - Progress tracking with ETA.

Environment Variables (Azure OpenAI):
    AZURE_OPENAI_API_KEY      + AZURE_OPENAI_ENDPOINT       (primary)
    AZURE_OPENAI_API_KEY_2    + AZURE_OPENAI_ENDPOINT_2     (optional)
    AZURE_OPENAI_API_KEY_3    + AZURE_OPENAI_ENDPOINT_3     (optional)
    AZURE_OPENAI_DEPLOYMENT_NAME  - Chat model deployment (default: gpt-4o)
    AZURE_OPENAI_API_VERSION      - API version

Usage:
    # Profile all files (auto-detects Azure, uses all 3 endpoints)
    python profile_corpus.py

    # Dry run — count files needing profiles
    python profile_corpus.py --dry-run

    # Force regenerate all profiles
    python profile_corpus.py --reprocess

    # Use specific model
    python profile_corpus.py --model gpt-4o --max-pages 3
"""

import argparse
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(SCRIPT_DIR / ".env")

from modules.config import (
    DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
    DEFAULT_PDF_PROFILE_MAX_PAGES,
    DEFAULT_PDF_STORE_DB_NAME,
    PDF_PROFILE_SCHEMA_VERSION,
)
from modules.db import (
    init_pdf_page_store,
    open_pdf_page_store,
    store_set_pdf_profile,
)
from modules.profiles import _generate_pdf_profile_with_llm
from modules.utils import (
    compute_rel_path_for_store,
    extract_json_object,
    truncate_for_profile,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("profile")

# Suppress noisy HTTP request logs from httpx/openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 8
BASE_DELAY = 3.0
MAX_DELAY = 120.0
JITTER_FACTOR = 0.3

RETRIABLE_SUBSTRINGS = (
    "throttl", "rate", "too many", "429",
    "timeout", "timed out",
    "serviceunavailable", "service unavailable",
    "server error", "internal error", "502", "503",
    "connection", "reset by peer",
)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()


def _signal_handler(sig, frame):
    if _shutdown_event.is_set():
        log.warning("Double Ctrl+C — forcing exit")
        sys.exit(1)
    log.warning("Ctrl+C received — finishing current files and exiting cleanly...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Azure credential discovery
# ---------------------------------------------------------------------------
def _discover_azure_chat_clients(model: str) -> List[Any]:
    """Create AzureChatOpenAI clients for all available endpoints."""
    from langchain_openai import AzureChatOpenAI

    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", model)

    clients = []

    # Primary
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if key and endpoint:
        clients.append(AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=0.3,
            request_timeout=90,
            max_retries=0,
        ))

    # Numbered (2, 3, ...)
    for i in range(2, 10):
        key = os.environ.get(f"AZURE_OPENAI_API_KEY_{i}", "").strip()
        endpoint = os.environ.get(f"AZURE_OPENAI_ENDPOINT_{i}", "").strip()
        if key and endpoint:
            clients.append(AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version=api_version,
                azure_deployment=deployment,
                temperature=0.3,
                request_timeout=90,
                max_retries=0,
            ))

    return clients


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_retriable(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(s in err for s in RETRIABLE_SUBSTRINGS)


def _backoff_delay(attempt: int) -> float:
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = delay * JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.5, delay + jitter)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# Profile generation with retry (wraps existing logic)
# ---------------------------------------------------------------------------
def _generate_profile_with_retry(
    llm,
    *,
    rel_path: str,
    base_input_dir: Path,
    excerpt: str,
    provider: str,
    llm_id: str,
    max_pages: int,
    max_chars_per_page: int,
    worker_id: int = 0,
) -> Optional[Dict[str, Any]]:
    """Generate a profile with retry logic on rate-limit errors."""
    # Build the path from rel_path
    file_path = (base_input_dir / rel_path).resolve()

    for attempt in range(MAX_RETRIES):
        if _shutdown_event.is_set():
            return None

        try:
            prof = _generate_pdf_profile_with_llm(
                llm,
                pdf_path=file_path,
                base_input_dir=base_input_dir,
                excerpt=excerpt,
                provider=provider,
                llm_id=llm_id,
                max_pages=max_pages,
                max_chars_per_page=max_chars_per_page,
            )
            return prof
        except Exception as exc:
            if _is_retriable(exc) and attempt < MAX_RETRIES - 1:
                delay = _backoff_delay(attempt)
                log.warning(
                    "Worker %d: retriable error (attempt %d/%d), retrying in %.1fs: %s",
                    worker_id, attempt + 1, MAX_RETRIES, delay,
                    str(exc)[:120],
                )
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not _shutdown_event.is_set():
                    time.sleep(min(1.0, deadline - time.monotonic()))
                continue
            else:
                log.warning(
                    "Worker %d: profile failed for %s: %s",
                    worker_id, rel_path[:60], str(exc)[:120],
                )
                return None

    return None


# ---------------------------------------------------------------------------
# Worker function (runs in thread)
# ---------------------------------------------------------------------------
def _worker_process_files(
    *,
    worker_id: int,
    llm,
    files: List[Dict[str, Any]],  # list of {rel_path, ...}
    db_path: Path,
    base_input_dir: Path,
    provider: str,
    llm_id: str,
    model_tag: str,
    max_pages: int,
    max_chars_per_page: int,
    stats: Dict[str, int],
    stats_lock: threading.Lock,
) -> int:
    """Process assigned files: generate profiles + write to DB."""
    conn = open_pdf_page_store(db_path)
    profiled = 0

    for file_info in files:
        if _shutdown_event.is_set():
            break

        rel_path = file_info["rel_path"]

        # Build excerpt from stored pages
        rows = conn.execute(
            """
            SELECT page_number, doc_content FROM pdf_page_store
            WHERE rel_path = ? ORDER BY page_number ASC LIMIT ?
            """,
            (rel_path, int(max_pages)),
        ).fetchall()

        parts = []
        for page_number, doc_content in rows:
            page_label = f"{int(page_number)}" if page_number is not None else "?"
            snippet = truncate_for_profile(
                str(doc_content or ""), max_chars=int(max_chars_per_page)
            )
            if snippet.strip():
                parts.append(f"--- PAGE {page_label} ---\n{snippet}")
        excerpt = "\n\n".join(parts).strip()

        if not excerpt:
            # No content to profile
            with stats_lock:
                stats["skipped_empty"] += 1
            continue

        prof = _generate_profile_with_retry(
            llm,
            rel_path=rel_path,
            base_input_dir=base_input_dir,
            excerpt=excerpt,
            provider=provider,
            llm_id=llm_id,
            max_pages=max_pages,
            max_chars_per_page=max_chars_per_page,
            worker_id=worker_id,
        )

        if prof is not None:
            try:
                store_set_pdf_profile(
                    conn,
                    rel_path=rel_path,
                    profile=prof,
                    pdf_profile_model=model_tag,
                )
                conn.commit()
                profiled += 1
            except Exception as exc:
                log.warning("Worker %d: DB write failed for %s: %s",
                            worker_id, rel_path[:60], str(exc)[:80])

        with stats_lock:
            stats["processed"] += 1
            stats["profiled"] = stats.get("profiled", 0) + (1 if prof else 0)

    conn.close()
    return profiled


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def profile_corpus(
    *,
    db_path: Path,
    base_input_dir: Path,
    model: str = "gpt-4o",
    max_pages: int = DEFAULT_PDF_PROFILE_MAX_PAGES,
    max_chars_per_page: int = DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
    max_workers: Optional[int] = None,
    reprocess: bool = False,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Generate profiles for all files in the store that are missing one."""
    conn = open_pdf_page_store(db_path)
    init_pdf_page_store(conn)

    provider = "azure"
    llm_id = model
    model_tag = f"{provider}:{llm_id}:p{int(max_pages)}:c{int(max_chars_per_page)}"

    # --- Find files needing profiles ---
    log.info("Querying files needing profiles (model_tag=%s) ...", model_tag)

    # Get all unique files (rel_path) and whether they have a profile
    if reprocess:
        rows = conn.execute(
            """
            SELECT DISTINCT rel_path FROM pdf_page_store
            ORDER BY rel_path
            """
        ).fetchall()
        need_profile = [{"rel_path": r[0]} for r in rows]
        already_done = 0
    else:
        # Files WITH ANY profile (regardless of model tag)
        rows_with = conn.execute(
            """
            SELECT DISTINCT rel_path FROM pdf_page_store
            WHERE pdf_profile_json IS NOT NULL
            """
        ).fetchall()
        done_paths = {r[0] for r in rows_with}

        # All unique files
        rows_all = conn.execute(
            "SELECT DISTINCT rel_path FROM pdf_page_store ORDER BY rel_path"
        ).fetchall()
        all_paths = [r[0] for r in rows_all]

        need_profile = [{"rel_path": rp} for rp in all_paths if rp not in done_paths]
        already_done = len(done_paths)

    total_files = conn.execute(
        "SELECT COUNT(DISTINCT rel_path) FROM pdf_page_store"
    ).fetchone()[0]
    conn.close()

    log.info("Total unique files in store: %d", total_files)
    log.info("Already profiled:            %d", already_done)
    log.info("Files needing profiles:      %d", len(need_profile))

    if not need_profile:
        log.info("Nothing to do — all files have profiles")
        return {"total": total_files, "processed": 0, "profiled": 0}

    if dry_run:
        log.info("DRY RUN — no profiles will be generated")
        return {"total": total_files, "processed": 0, "to_profile": len(need_profile)}

    # --- Create LLM clients ---
    clients = _discover_azure_chat_clients(model)
    if not clients:
        log.error("No Azure OpenAI credentials found. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT.")
        sys.exit(1)

    num_workers = max_workers if max_workers else len(clients)
    num_workers = min(num_workers, len(clients))

    for i, c in enumerate(clients[:num_workers]):
        ep = getattr(c, "azure_endpoint", "?")
        region = ep.split("//")[-1].split(".")[0] if "//" in ep else ep[:30]
        log.info("  Azure LLM client %d: endpoint=%s  deployment=%s", i + 1, region, model)

    log.info("Workers: %d  |  Files to profile: %d", num_workers, len(need_profile))

    # --- Distribute files across workers (round-robin) ---
    worker_files: List[List[Dict[str, Any]]] = [[] for _ in range(num_workers)]
    for idx, f in enumerate(need_profile):
        worker_files[idx % num_workers].append(f)

    for i in range(num_workers):
        log.info("  Worker %d: %d files", i + 1, len(worker_files[i]))

    # --- Shared stats ---
    stats: Dict[str, int] = {"processed": 0, "profiled": 0, "skipped_empty": 0}
    stats_lock = threading.Lock()

    # --- Progress reporter ---
    t0 = time.monotonic()

    def _progress_reporter():
        while not _shutdown_event.is_set():
            _shutdown_event.wait(20)
            with stats_lock:
                processed = stats["processed"]
                profiled = stats["profiled"]
            elapsed = time.monotonic() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(need_profile) - processed) / rate if rate > 0 else 0
            pct = (processed / len(need_profile)) * 100 if need_profile else 0
            log.info(
                "Progress: %d/%d (%.1f%%)  profiled=%d  rate=%.1f files/s  elapsed=%s  ETA=%s",
                processed, len(need_profile), pct, profiled,
                rate, _fmt_duration(elapsed), _fmt_duration(remaining),
            )
            if processed >= len(need_profile):
                break

    progress_thread = threading.Thread(target=_progress_reporter, daemon=True)
    progress_thread.start()

    # --- Launch workers ---
    log.info("Starting profile generation (Ctrl+C to stop gracefully)...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            future = executor.submit(
                _worker_process_files,
                worker_id=i + 1,
                llm=clients[i],
                files=worker_files[i],
                db_path=db_path,
                base_input_dir=base_input_dir,
                provider=provider,
                llm_id=llm_id,
                model_tag=model_tag,
                max_pages=max_pages,
                max_chars_per_page=max_chars_per_page,
                stats=stats,
                stats_lock=stats_lock,
            )
            futures[future] = i + 1

        total_profiled = 0
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                worker_profiled = future.result()
                total_profiled += worker_profiled
                log.info("Worker %d finished: %d profiles generated", worker_id, worker_profiled)
            except Exception as exc:
                log.error("Worker %d failed: %s", worker_id, exc)

    _shutdown_event.set()
    progress_thread.join(timeout=5)

    elapsed = time.monotonic() - t0

    # --- Final summary ---
    log.info("=" * 80)
    log.info("PROFILING COMPLETE in %s", _fmt_duration(elapsed))
    log.info("  Files processed: %d / %d", stats["processed"], len(need_profile))
    log.info("  Profiles generated: %d", total_profiled)
    log.info("  Skipped (empty): %d", stats.get("skipped_empty", 0))
    log.info("  Rate: %.1f files/s", stats["processed"] / elapsed if elapsed > 0 else 0)

    # Verify from DB
    try:
        conn2 = open_pdf_page_store(db_path)
        done_now = conn2.execute(
            "SELECT COUNT(DISTINCT rel_path) FROM pdf_page_store "
            "WHERE pdf_profile_json IS NOT NULL"
        ).fetchone()[0]
        conn2.close()
        log.info("  DB verify: %d files with profiles", done_now)
    except Exception:
        pass

    if _shutdown_event.is_set():
        log.info("  (Stopped early — re-run to resume)")

    return {
        "total": len(need_profile),
        "processed": stats["processed"],
        "profiled": total_profiled,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate document profiles for all files in the SQLite page store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", type=str, default="gpt-4o",
        help="Azure chat model deployment name (default: gpt-4o)",
    )
    p.add_argument(
        "--db-path", type=str, default=None,
        help="SQLite DB path (default: processed/pdf_page_store.sqlite)",
    )
    p.add_argument(
        "--input-dir", type=str, default=str(SCRIPT_DIR / "search-dataset"),
        help="Corpus root directory (for computing rel_paths -> absolute paths)",
    )
    p.add_argument(
        "--max-pages", type=int, default=DEFAULT_PDF_PROFILE_MAX_PAGES,
        help=f"Max pages to excerpt per file (default: {DEFAULT_PDF_PROFILE_MAX_PAGES})",
    )
    p.add_argument(
        "--max-chars-per-page", type=int, default=DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
        help=f"Max chars per page in excerpt (default: {DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE})",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: auto, one per Azure endpoint)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Count files needing profiles without generating anything.",
    )
    p.add_argument(
        "--reprocess", action="store_true",
        help="Regenerate profiles for all files (even if already profiled).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (SCRIPT_DIR / "processed" / DEFAULT_PDF_STORE_DB_NAME).resolve()

    base_input_dir = Path(args.input_dir).expanduser().resolve()

    log.info("Model:     %s", args.model)
    log.info("DB:        %s", db_path)
    log.info("Input dir: %s", base_input_dir)
    log.info("Excerpt:   first %d pages, up to %d chars each",
             args.max_pages, args.max_chars_per_page)

    profile_corpus(
        db_path=db_path,
        base_input_dir=base_input_dir,
        model=args.model,
        max_pages=args.max_pages,
        max_chars_per_page=args.max_chars_per_page,
        max_workers=args.workers,
        reprocess=args.reprocess,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
