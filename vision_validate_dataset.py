#!/usr/bin/env python3
"""Vision-based post-hoc validation for the search dataset.

For each row in the dataset, renders the labeled source PDF page(s) as
single-page PDF bytes and sends them to Gemini 2.5 Pro via Vertex AI alongside
the query and expected answer.  The model rates two dimensions:

  1. query_quality        — "good" | "mediocre" | "bad"
  2. source_answerability — "answerable" | "partial" | "not_answerable"

Two filtered CSVs are written:
  vision_strict_<ts>.csv   — query_quality="good"  AND source_answerability="answerable"
  vision_relaxed_<ts>.csv  — query_quality in {good,mediocre}
                             AND source_answerability in {answerable,partial}

A full JSON report with per-row verdicts and aggregate stats is also saved.

The script is resume-safe: each completed row is appended to a JSONL
checkpoint file.  Re-running with the same --checkpoint path skips rows
that have already been evaluated.

Usage:
    python vision_validate_dataset.py \\
        --dataset output/full_10_000.csv \\
        --db-path processed/pdf_page_store.sqlite \\
        --pdf-root search-dataset/ \\
        --leya-env /Users/reza/Documents/GitHub/leya/.local.env \\
        --concurrency 5 \\
        --checkpoint processed/vision_validation_cache.jsonl \\
        --output output/vision_validated \\
        --max-rows 50

    # Dry-run (parses CSV and resolves paths, no LLM calls):
    python vision_validate_dataset.py --dry-run
"""

import argparse
import ast
import base64
import json
import logging
import random
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # pymupdf
import pandas as pd
from dotenv import dotenv_values

SCRIPT_DIR = Path(__file__).resolve().parent

LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("vision_validate")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = threading.Event()


def _signal_handler(sig, frame):
    if _shutdown.is_set():
        log.warning("Double Ctrl+C — forcing exit")
        sys.exit(1)
    log.warning("Ctrl+C received — finishing in-flight calls and saving checkpoint…")
    _shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------
MAX_RETRIES = 8

# Substrings that indicate a transient error worth retrying
RETRIABLE_SUBSTRINGS = (
    "throttl", "rate", "too many", "429",
    "quota", "resource exhausted",
    "timeout", "timed out", "deadline",
    "unavailable", "service unavailable",
    "internal", "502", "503",
    "connection", "reset by peer",
)

# Substrings that indicate a rate-limit specifically (warrant longer backoff)
RATE_LIMIT_SUBSTRINGS = (
    "429", "quota", "resource exhausted", "throttl", "rate", "too many",
)


def _is_retriable(exc: Exception) -> bool:
    return any(s in str(exc).lower() for s in RETRIABLE_SUBSTRINGS)


def _is_rate_limit(exc: Exception) -> bool:
    return any(s in str(exc).lower() for s in RATE_LIMIT_SUBSTRINGS)


def _retry_delay(attempt: int, exc: Exception) -> float:
    """Exponential backoff with jitter.

    Rate-limit errors (429 / quota exhausted) start at 10s and double up to
    120s.  Other transient errors start at 2s and double up to 30s.
    """
    if _is_rate_limit(exc):
        base, cap = 10.0, 120.0
    else:
        base, cap = 2.0, 30.0
    delay = min(base * (2 ** (attempt - 1)), cap)
    jitter = delay * 0.2 * (2 * random.random() - 1)
    return max(1.0, delay + jitter)


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# GCP credential loading from leya .local.env → google.genai.Client
# ---------------------------------------------------------------------------
def _build_genai_client(leya_env_path: Path, model: str = "gemini-2.5-pro"):
    """Parse leya .local.env, extract GCP service-account creds, return
    a google.genai.Client configured for Vertex AI.

    Looks up the best region for the requested model from AI_REGIONS_V3.
    Falls back to europe-central2 if the model isn't listed (e.g. Flash).

    Returns (client, project_id, location, model_name).
    """
    from google import genai
    from google.oauth2 import service_account

    env = dotenv_values(str(leya_env_path))
    raw = env.get("AI_REGIONS_V3")
    if not raw:
        raise ValueError(f"AI_REGIONS_V3 not found in {leya_env_path}")

    regions = json.loads(raw)
    gcp = next((r for r in regions if r.get("provider") == "gcp"), None)
    if gcp is None:
        raise ValueError("No 'gcp' provider entry found in AI_REGIONS_V3")

    sa_info = json.loads(base64.b64decode(gcp["base64EncodedKey"]))
    creds = service_account.Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    project_id = gcp["projectId"]

    # Find region that lists the requested model; fall back to europe-central2
    location = "europe-central2"
    for reg in gcp.get("regions", []):
        for m in reg.get("models", []):
            if m.get("model") == model or m.get("id") == model:
                location = reg["region"]
                break

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=creds,
    )
    return client, project_id, location, model


# ---------------------------------------------------------------------------
# PDF page extraction (single-page PDF bytes — mirrors pdf-lib in TS)
# ---------------------------------------------------------------------------
_pdf_doc_cache: Dict[str, fitz.Document] = {}
_pdf_cache_lock = threading.Lock()


def _open_pdf(path: str) -> fitz.Document:
    with _pdf_cache_lock:
        if path not in _pdf_doc_cache:
            _pdf_doc_cache[path] = fitz.open(path)
        return _pdf_doc_cache[path]


def extract_page_as_pdf_bytes(pdf_path: str, page_number: int) -> bytes:
    """Extract a single page as a self-contained PDF buffer.

    Page numbers are 1-indexed, matching the dataset's page_numbers field.
    Mirrors pdf-utils.ts extractPageAsPdf().
    """
    src = _open_pdf(pdf_path)
    if page_number < 1 or page_number > src.page_count:
        raise ValueError(
            f"Page {page_number} out of range for {pdf_path} ({src.page_count} pages)"
        )
    single = fitz.open()
    single.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)
    return single.tobytes()


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
_db_local = threading.local()


def _get_conn(db_path: str) -> sqlite3.Connection:
    if not hasattr(_db_local, "conn"):
        _db_local.conn = sqlite3.connect(db_path)
        _db_local.conn.execute("PRAGMA journal_mode=WAL")
    return _db_local.conn


def _lookup_rel_path(db_path: str, filename: str) -> Optional[str]:
    """Resolve a bare filename to its rel_path via SQLite lookup."""
    conn = _get_conn(db_path)
    row = conn.execute(
        "SELECT rel_path FROM pdf_page_store WHERE filename = ? LIMIT 1",
        (filename,),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


# ---------------------------------------------------------------------------
# Source page resolution
# ---------------------------------------------------------------------------
def _parse_source_files_with_pages(row: Dict[str, Any]) -> List[Tuple[str, int]]:
    """Parse source_files_with_pages into list of (filename, page_number) tuples."""
    raw = row.get("source_files_with_pages", "")
    items: List[str] = []

    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items = [str(x) for x in parsed]
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    items = [str(x) for x in parsed]
                else:
                    items = [raw.strip()]
            except Exception:
                items = [raw.strip()]
    elif isinstance(raw, list):
        items = [str(x) for x in raw]

    # Also include source_file_with_page for single-hop rows
    sfwp = row.get("source_file_with_page", "")
    if isinstance(sfwp, str) and sfwp.strip() and sfwp.strip().lower() != "nan":
        if sfwp.strip() not in items:
            items.insert(0, sfwp.strip())

    results: List[Tuple[str, int]] = []
    for item in items:
        s = str(item).strip()
        if " (page " in s and s.endswith(")"):
            idx = s.rfind(" (page ")
            fname = s[:idx].strip()
            try:
                page_num = int(s[idx + 7:-1])
                results.append((fname, page_num))
            except ValueError:
                pass
    return results


def resolve_source_pages(
    row: Dict[str, Any],
    pdf_root: Path,
    db_path: str,
) -> List[Tuple[Path, int]]:
    """Resolve all source pages to (abs_path, page_number) pairs via SQLite lookup.

    Each filename from source_files_with_pages is resolved to its rel_path by
    querying pdf_page_store, then combined with pdf_root to get the absolute path.

    Raises FileNotFoundError if any resolved path does not exist on disk.
    """
    sources = _parse_source_files_with_pages(row)
    if not sources:
        raise ValueError("No source pages found in row")

    resolved: List[Tuple[Path, int]] = []
    for filename, page_number in sources:
        rel = _lookup_rel_path(db_path, filename)
        if rel is None:
            raise FileNotFoundError(
                f"rel_path not found in SQLite for filename: {filename!r}"
            )

        abs_path = pdf_root / rel
        if not abs_path.exists():
            raise FileNotFoundError(f"PDF not found on disk: {abs_path}")

        resolved.append((abs_path, page_number))

    return resolved


# ---------------------------------------------------------------------------
# Prompts (adapted from validate-dataset/prompt.ts — general, no categories)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_BASE = """\
You are a rigorous QA reviewer for a document search benchmark dataset.

CONTEXT:
This dataset evaluates semantic search over ~7,000 documents (171,860 pages total).
Each query is submitted as a search input against all pages, and the system checks
whether the correct source page is retrieved at rank 1.

You will receive the actual PDF page(s) that are labeled as the source for a given query.
Read the page content carefully — it may contain tables, charts, scanned text,
multi-column layouts, handwriting, or other challenging formats.

Your job is to validate each query + source pair on two dimensions:

────────────────────────────────────────────────────
1. query_quality — Is the query standalone and realistic as a search query over
   these ~7,000 documents?

Think of it this way: if someone typed this query into a search bar over 7,000
documents, would it retrieve the right page? A good query must be:
- Self-contained: understandable without any prior context or conversation history.
- Specific: contains enough distinguishing detail (names, dates, document titles,
  numbers, entities) to narrow the search to the correct page rather than matching
  many unrelated pages.
- No assumed context: must NOT use definite references ("the agreement", "the policy",
  "the report", "the contract", "the company") that assume the reader already knows
  which specific document is being discussed. Generic document-type references without
  identifying context are NOT standalone.
- Realistic: resembles a query a real user would type when searching a document
  repository.

Rate as:
- "good"     → standalone, specific, uniquely identifying search query
- "mediocre" → understandable but relies on assumed context or is too generic to
               reliably retrieve the right page among 7,000
- "bad"      → not standalone, nonsensical, references internal metadata (filenames,
               page numbers), or so generic it would match hundreds of pages

────────────────────────────────────────────────────
2. source_answerability — Does the attached PDF page actually contain the answer?

Look at the actual page content. Judge whether that page contains enough information
to answer the query.

Rate as:
- "answerable"     → the page clearly contains the information needed to answer
- "partial"        → only part of the answer is present, or significant inference
                     is required
- "not_answerable" → the page does not contain the information needed\
"""

_SYSTEM_PROMPT_MULTI_PAGE_SUFFIX = """

Note: multiple pages are attached (from one or more documents). Judge whether all
the attached pages combined contain a complete answer to the query for
source_answerability.\
"""

_SYSTEM_PROMPT_FOOTER = """

────────────────────────────────────────────────────

For each dimension provide a short reasoning (1–2 sentences).

Respond with ONLY a raw JSON object, no markdown fences, no explanation:
{
  "query_quality": "good" | "mediocre" | "bad",
  "query_quality_reasoning": "Brief explanation",
  "source_answerability": "answerable" | "partial" | "not_answerable",
  "source_answerability_reasoning": "Brief explanation",
  "confidence": between 0 and 1.0
}\
"""


def build_system_prompt(multi_page: bool) -> str:
    suffix = _SYSTEM_PROMPT_MULTI_PAGE_SUFFIX if multi_page else ""
    return _SYSTEM_PROMPT_BASE + suffix + _SYSTEM_PROMPT_FOOTER


def build_user_text(row: Dict[str, Any]) -> str:
    source = row.get("source_files_with_pages_readable", "") or row.get("source_file_with_page", "")
    query = row.get("user_input", "")
    answer = row.get("reference", "")
    return (
        f"Source: {source}\n\n"
        f"Query:\n{query}\n\n"
        f"Expected answer:\n{answer}\n\n"
        "The attached PDF(s) are the actual source page(s). "
        "Judge the content as-is for source_answerability."
    )


# ---------------------------------------------------------------------------
# Gemini call (google.genai SDK — Vertex AI)
# ---------------------------------------------------------------------------
def _call_gemini(
    client,
    model_name: str,
    system_prompt: str,
    pdf_parts_bytes: List[bytes],
    user_text: str,
    label: str = "",
) -> Dict[str, Any]:
    """Call Gemini with inline PDF parts + text, return parsed JSON dict.

    Uses the google.genai SDK (v1+). PDF bytes are sent as inline blobs with
    mime_type='application/pdf', matching the TypeScript validate-dataset approach.
    """
    from google.genai import types

    parts = []
    for pdf_bytes in pdf_parts_bytes:
        parts.append(types.Part(inline_data=types.Blob(data=pdf_bytes, mime_type="application/pdf")))
    parts.append(types.Part(text=user_text))

    cfg = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,
        max_output_tokens=4096,  # thinking tokens count against budget; 600 is too low
    )

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        if _shutdown.is_set():
            return {"error": "shutdown"}
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=parts,
                config=cfg,
            )
            raw_text = resp.text.strip()
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
            return json.loads(raw_text)
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            if _is_retriable(exc) and attempt < MAX_RETRIES:
                delay = _retry_delay(attempt, exc)
                kind = "rate-limit" if _is_rate_limit(exc) else "transient"
                log.warning(
                    "%s error%s (attempt %d/%d), retrying in %.0fs: %s",
                    kind, f" [{label}]" if label else "",
                    attempt, MAX_RETRIES, delay, msg[:100],
                )
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not _shutdown.is_set():
                    time.sleep(min(1.0, deadline - time.monotonic()))
            else:
                log.error(
                    "Gemini call failed%s after %d attempt(s): %s",
                    f" [{label}]" if label else "", attempt, msg[:200],
                )
                return {"error": msg[:300]}

    return {"error": str(last_exc)[:300] if last_exc else "unknown"}


# ---------------------------------------------------------------------------
# Per-row evaluator
# ---------------------------------------------------------------------------
def evaluate_row(
    row: Dict[str, Any],
    row_index: int,
    client,
    model_name: str,
    pdf_root: Path,
    db_path: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single dataset row.  Returns a result dict ready for JSONL."""
    query = str(row.get("user_input", ""))
    answer = str(row.get("reference", ""))

    base = {
        "row_index": row_index,
        "query": query,
        "answer": answer[:200],
        "source": row.get("source_files_with_pages_readable", "") or row.get("source_file_with_page", ""),
        "synthesizer_name": str(row.get("synthesizer_name", "")),
    }

    # Resolve source pages
    try:
        pages = resolve_source_pages(row, pdf_root, db_path)
    except Exception as exc:
        log.warning("[%d] Source resolution failed: %s", row_index, str(exc)[:120])
        return {**base, "query_quality": "error", "source_answerability": "error",
                "confidence": 0.0, "error": f"source_resolution: {str(exc)[:200]}",
                "query_quality_reasoning": "", "source_answerability_reasoning": ""}

    # Extract all source pages as single-page PDFs
    pdf_bytes_list: List[bytes] = []
    try:
        for abs_path, page_num in pages:
            pdf_bytes_list.append(extract_page_as_pdf_bytes(str(abs_path), page_num))
    except Exception as exc:
        log.warning("[%d] PDF extraction failed: %s", row_index, str(exc)[:120])
        return {**base, "query_quality": "error", "source_answerability": "error",
                "confidence": 0.0, "error": f"pdf_extraction: {str(exc)[:200]}",
                "query_quality_reasoning": "", "source_answerability_reasoning": ""}

    if dry_run:
        return {
            **base,
            "query_quality": "dry_run",
            "source_answerability": "dry_run",
            "confidence": 0.0,
            "error": "",
            "query_quality_reasoning": f"{len(pdf_bytes_list)} page(s) resolved OK",
            "source_answerability_reasoning": "",
            "pages_resolved": [f"{p}:{n}" for p, n in pages],
        }

    # Build prompts
    multi_page = len(pdf_bytes_list) > 1
    system_prompt = build_system_prompt(multi_page)
    user_text = build_user_text(row)

    # Call Gemini
    result = _call_gemini(
        client,
        model_name=model_name,
        system_prompt=system_prompt,
        pdf_parts_bytes=pdf_bytes_list,
        user_text=user_text,
        label=str(row_index),
    )

    if "error" in result and "query_quality" not in result:
        return {**base, "query_quality": "error", "source_answerability": "error",
                "confidence": 0.0, "error": result.get("error", ""),
                "query_quality_reasoning": "", "source_answerability_reasoning": ""}

    return {
        **base,
        "query_quality": result.get("query_quality", "error"),
        "query_quality_reasoning": result.get("query_quality_reasoning", ""),
        "source_answerability": result.get("source_answerability", "error"),
        "source_answerability_reasoning": result.get("source_answerability_reasoning", ""),
        "confidence": float(result.get("confidence", 0.0)),
        "error": result.get("error", ""),
        "page_count": len(pdf_bytes_list),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _load_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load already-evaluated rows from JSONL checkpoint."""
    results: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results[int(obj["row_index"])] = obj
            except Exception:
                pass
    return results


_checkpoint_lock = threading.Lock()


def _append_checkpoint(path: Path, result: Dict[str, Any]) -> None:
    with _checkpoint_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def _passes_strict(r: Dict[str, Any]) -> bool:
    return r.get("query_quality") == "good" and r.get("source_answerability") == "answerable"


def _passes_relaxed(r: Dict[str, Any]) -> bool:
    return (
        r.get("query_quality") in ("good", "mediocre")
        and r.get("source_answerability") in ("answerable", "partial")
    )


def _write_outputs(
    df: pd.DataFrame,
    results: Dict[int, Dict[str, Any]],
    output_base: Path,
) -> Tuple[Path, Path, Path]:
    """Write strict CSV, relaxed CSV, and JSON report."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    all_results = list(results.values())
    evaluated = len(all_results)

    # Build a verdicts dataframe to join onto the original dataset columns
    verdict_rows = []
    for r in all_results:
        verdict_rows.append({
            "_row_index": r["row_index"],
            "vision_query_quality": r.get("query_quality", ""),
            "vision_query_quality_reasoning": r.get("query_quality_reasoning", ""),
            "vision_source_answerability": r.get("source_answerability", ""),
            "vision_source_answerability_reasoning": r.get("source_answerability_reasoning", ""),
            "vision_confidence": r.get("confidence", 0.0),
            "vision_page_count": r.get("page_count", ""),
            "vision_error": r.get("error", ""),
        })
    verdicts_df = pd.DataFrame(verdict_rows).set_index("_row_index")

    # Join verdicts onto the original dataset rows, preserving original index
    df_with_verdicts = df.join(verdicts_df, how="left")

    # --- Filtered CSVs ---
    strict_indices = [r["row_index"] for r in all_results if _passes_strict(r)]
    relaxed_indices = [r["row_index"] for r in all_results if _passes_relaxed(r)]

    strict_df = df_with_verdicts[df_with_verdicts.index.isin(strict_indices)].reset_index(drop=True)
    relaxed_df = df_with_verdicts[df_with_verdicts.index.isin(relaxed_indices)].reset_index(drop=True)

    strict_path = output_base.parent / f"{output_base.name}_strict_{ts}.csv"
    relaxed_path = output_base.parent / f"{output_base.name}_relaxed_{ts}.csv"
    strict_df.to_csv(strict_path, index=False)
    relaxed_df.to_csv(relaxed_path, index=False)

    # --- Aggregate stats ---
    def _count(field, value):
        return sum(1 for r in all_results if r.get(field) == value)

    qc_good = _count("query_quality", "good")
    qc_mediocre = _count("query_quality", "mediocre")
    qc_bad = _count("query_quality", "bad")
    qc_error = _count("query_quality", "error")

    sa_answerable = _count("source_answerability", "answerable")
    sa_partial = _count("source_answerability", "partial")
    sa_not = _count("source_answerability", "not_answerable")
    sa_error = _count("source_answerability", "error")

    # Per synthesizer breakdown
    synth_types = sorted(set(r.get("synthesizer_name", "") for r in all_results))
    synth_breakdown: Dict[str, Any] = {}
    for st in synth_types:
        sub = [r for r in all_results if r.get("synthesizer_name", "") == st]
        synth_breakdown[st or "(unknown)"] = {
            "count": len(sub),
            "qc_good": sum(1 for r in sub if r.get("query_quality") == "good"),
            "qc_mediocre": sum(1 for r in sub if r.get("query_quality") == "mediocre"),
            "qc_bad": sum(1 for r in sub if r.get("query_quality") == "bad"),
            "sa_answerable": sum(1 for r in sub if r.get("source_answerability") == "answerable"),
            "sa_partial": sum(1 for r in sub if r.get("source_answerability") == "partial"),
            "sa_not_answerable": sum(1 for r in sub if r.get("source_answerability") == "not_answerable"),
        }

    report = {
        "metadata": {
            "dataset": str(df.shape),
            "evaluated_rows": evaluated,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "query_quality": {
                "good": qc_good,
                "mediocre": qc_mediocre,
                "bad": qc_bad,
                "error": qc_error,
                "good_rate": round(qc_good / max(evaluated, 1) * 100, 1),
                "good_or_mediocre_rate": round((qc_good + qc_mediocre) / max(evaluated, 1) * 100, 1),
            },
            "source_answerability": {
                "answerable": sa_answerable,
                "partial": sa_partial,
                "not_answerable": sa_not,
                "error": sa_error,
                "answerable_rate": round(sa_answerable / max(evaluated, 1) * 100, 1),
                "answerable_or_partial_rate": round((sa_answerable + sa_partial) / max(evaluated, 1) * 100, 1),
            },
            "filtered": {
                "strict_kept": len(strict_indices),
                "strict_rate": round(len(strict_indices) / max(evaluated, 1) * 100, 1),
                "relaxed_kept": len(relaxed_indices),
                "relaxed_rate": round(len(relaxed_indices) / max(evaluated, 1) * 100, 1),
            },
            "by_synthesizer": synth_breakdown,
        },
        "per_row_results": sorted(all_results, key=lambda r: r["row_index"]),
    }

    report_path = output_base.parent / f"{output_base.name}_report_{ts}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return strict_path, relaxed_path, report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vision-based post-hoc dataset validation using Gemini 2.5 Pro.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default=str(SCRIPT_DIR / "output" / "full_10_000.csv"),
        help="Path to dataset CSV (default: output/full_10_000.csv)",
    )
    parser.add_argument(
        "--db-path", default=str(SCRIPT_DIR / "processed" / "pdf_page_store.sqlite"),
        help="SQLite page store for rel_path lookups",
    )
    parser.add_argument(
        "--pdf-root", default=str(SCRIPT_DIR / "search-dataset"),
        help="Root directory containing the PDF corpus",
    )
    parser.add_argument(
        "--leya-env",
        default=str(Path.home() / "Documents" / "GitHub" / "leya" / ".local.env"),
        help="Path to leya .local.env (contains AI_REGIONS_V3 with GCP credentials)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Number of parallel Gemini calls (default: 5)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(SCRIPT_DIR / "processed" / "vision_validation_cache.jsonl"),
        help="JSONL resume checkpoint path",
    )
    parser.add_argument(
        "--output", default=str(SCRIPT_DIR / "output" / "vision_validated"),
        help="Output base path (timestamp + suffix appended)",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Process only first N rows (for testing)",
    )
    parser.add_argument(
        "--rows", type=str, default=None,
        help="Comma-separated row indices to evaluate (e.g. 0,1,5000,5001)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve paths and report page counts, skip LLM calls",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    pdf_root = Path(args.pdf_root).expanduser().resolve()
    leya_env_path = Path(args.leya_env).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_base = Path(args.output).expanduser().resolve()

    log.info("=" * 70)
    log.info("Vision Dataset Validation (Gemini 2.5 Pro)")
    log.info("=" * 70)
    log.info("  Dataset:     %s", dataset_path)
    log.info("  DB:          %s", db_path)
    log.info("  PDF root:    %s", pdf_root)
    log.info("  Leya env:    %s", leya_env_path)
    log.info("  Checkpoint:  %s", checkpoint_path)
    log.info("  Output:      %s.*", output_base)
    log.info("  Concurrency: %d", args.concurrency)
    log.info("  Dry run:     %s", args.dry_run)

    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        return 1
    if not db_path.exists():
        log.error("SQLite DB not found: %s", db_path)
        return 1
    if not pdf_root.exists():
        log.error("PDF root not found: %s", pdf_root)
        return 1

    # --- Load dataset ---
    df = pd.read_csv(dataset_path)
    if args.rows:
        row_indices = [int(x.strip()) for x in args.rows.split(",")]
        df = df.loc[row_indices]
        log.info("  Rows:        %d (selected by --rows)", len(df))
    elif args.max_rows:
        df = df.head(args.max_rows)
        log.info("  Rows:        %d", len(df))
    else:
        log.info("  Rows:        %d", len(df))

    # --- Load checkpoint ---
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = _load_checkpoint(checkpoint_path)
    skip_count = len(already_done)
    todo_indices = [i for i in df.index if i not in already_done]
    log.info("  Already evaluated: %d, remaining: %d", skip_count, len(todo_indices))

    if not args.dry_run and not todo_indices:
        log.info("All rows already evaluated — writing outputs from checkpoint.")
        strict_p, relaxed_p, report_p = _write_outputs(df, already_done, output_base)
        log.info("  Strict:  %s", strict_p)
        log.info("  Relaxed: %s", relaxed_p)
        log.info("  Report:  %s", report_p)
        return 0

    # --- Set up Gemini (google.genai client) ---
    client = None
    model_name = "gemini-2.5-pro"
    if not args.dry_run:
        if not leya_env_path.exists():
            log.error("Leya env not found: %s", leya_env_path)
            return 1
        try:
            client, project_id, location, model_name = _build_genai_client(leya_env_path, args.model)
            log.info("  Gemini:      %s @ %s / %s", model_name, location, project_id)
        except Exception as exc:
            log.error("Failed to initialise Gemini client: %s", exc)
            return 1

    # --- Parallel evaluation ---
    t0 = time.monotonic()
    completed = 0
    errors = 0
    lock = threading.Lock()
    SUMMARY_EVERY = 100  # print running totals every N completions

    def process(idx: int) -> Optional[Dict[str, Any]]:
        if _shutdown.is_set():
            return None
        row = df.loc[idx].to_dict()
        result = evaluate_row(
            row, idx, client, model_name, pdf_root, str(db_path), dry_run=args.dry_run
        )
        if not args.dry_run:
            _append_checkpoint(checkpoint_path, result)
        return result

    results_map: Dict[int, Dict[str, Any]] = dict(already_done)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_idx = {executor.submit(process, idx): idx for idx in todo_indices}
        for future in as_completed(future_to_idx):
            if _shutdown.is_set():
                break
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result is None:
                    continue
                results_map[idx] = result
                with lock:
                    completed += 1
                    if result.get("error"):
                        errors += 1

                # Per-row progress line
                elapsed = time.monotonic() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(todo_indices) - completed) / rate if rate > 0 else 0
                log.info(
                    "[%d/%d] row=%d  QC=%s  SA=%s  conf=%.2f  [%.1f/s  ETA %s]%s",
                    completed, len(todo_indices), idx,
                    result.get("query_quality", "?")[:1].upper(),
                    result.get("source_answerability", "?")[:2],
                    result.get("confidence", 0.0),
                    rate, _fmt_duration(remaining),
                    f"  ERR: {result['error'][:60]}" if result.get("error") else "",
                )

                # Periodic running-totals summary
                if completed % SUMMARY_EVERY == 0:
                    snap = list(results_map.values())
                    n = len(snap)
                    qc_g = sum(1 for r in snap if r.get("query_quality") == "good")
                    qc_m = sum(1 for r in snap if r.get("query_quality") == "mediocre")
                    qc_b = sum(1 for r in snap if r.get("query_quality") == "bad")
                    sa_a = sum(1 for r in snap if r.get("source_answerability") == "answerable")
                    sa_p = sum(1 for r in snap if r.get("source_answerability") == "partial")
                    sa_n = sum(1 for r in snap if r.get("source_answerability") == "not_answerable")
                    strict = sum(1 for r in snap if _passes_strict(r))
                    relaxed = sum(1 for r in snap if _passes_relaxed(r))
                    log.info(
                        "── SUMMARY (%d evaluated, %d errors) ──────────────────────────",
                        n, errors,
                    )
                    log.info(
                        "   Query quality:  good=%d (%.0f%%)  mediocre=%d (%.0f%%)  bad=%d (%.0f%%)",
                        qc_g, qc_g/max(n,1)*100, qc_m, qc_m/max(n,1)*100, qc_b, qc_b/max(n,1)*100,
                    )
                    log.info(
                        "   Answerability:  answerable=%d (%.0f%%)  partial=%d (%.0f%%)  not=%d (%.0f%%)",
                        sa_a, sa_a/max(n,1)*100, sa_p, sa_p/max(n,1)*100, sa_n, sa_n/max(n,1)*100,
                    )
                    log.info(
                        "   Filtered:       strict=%d (%.0f%%)  relaxed=%d (%.0f%%)",
                        strict, strict/max(n,1)*100, relaxed, relaxed/max(n,1)*100,
                    )
            except Exception as exc:
                log.error("Future failed for row %d: %s", idx, exc)
                errors += 1

    elapsed_total = time.monotonic() - t0
    log.info("")
    log.info("=" * 70)
    log.info("Evaluation complete: %d rows in %s (%d errors)",
             completed, _fmt_duration(elapsed_total), errors)

    if args.dry_run:
        log.info("Dry run complete — no outputs written.")
        return 0

    # --- Write outputs ---
    strict_p, relaxed_p, report_p = _write_outputs(df, results_map, output_base)

    total_eval = len(results_map)
    strict_kept = sum(1 for r in results_map.values() if _passes_strict(r))
    relaxed_kept = sum(1 for r in results_map.values() if _passes_relaxed(r))

    print("\n" + "=" * 70)
    print("VISION VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Evaluated:         {total_eval}")
    print(f"  Errors:            {errors}")

    qc_good = sum(1 for r in results_map.values() if r.get("query_quality") == "good")
    qc_med = sum(1 for r in results_map.values() if r.get("query_quality") == "mediocre")
    qc_bad = sum(1 for r in results_map.values() if r.get("query_quality") == "bad")
    sa_ans = sum(1 for r in results_map.values() if r.get("source_answerability") == "answerable")
    sa_par = sum(1 for r in results_map.values() if r.get("source_answerability") == "partial")
    sa_no = sum(1 for r in results_map.values() if r.get("source_answerability") == "not_answerable")

    print("\n  Query quality:")
    print(f"    good:     {qc_good:>5}  ({qc_good/max(total_eval,1)*100:5.1f}%)")
    print(f"    mediocre: {qc_med:>5}  ({qc_med/max(total_eval,1)*100:5.1f}%)")
    print(f"    bad:      {qc_bad:>5}  ({qc_bad/max(total_eval,1)*100:5.1f}%)")

    print("\n  Source answerability:")
    print(f"    answerable:     {sa_ans:>5}  ({sa_ans/max(total_eval,1)*100:5.1f}%)")
    print(f"    partial:        {sa_par:>5}  ({sa_par/max(total_eval,1)*100:5.1f}%)")
    print(f"    not_answerable: {sa_no:>5}  ({sa_no/max(total_eval,1)*100:5.1f}%)")

    print("\n  Filtered output:")
    print(f"    Strict  (good + answerable):          {strict_kept:>5}  ({strict_kept/max(total_eval,1)*100:5.1f}%)")
    print(f"    Relaxed (good/mediocre + ans/partial): {relaxed_kept:>5}  ({relaxed_kept/max(total_eval,1)*100:5.1f}%)")

    print("\n  Outputs:")
    print(f"    Strict CSV:  {strict_p}")
    print(f"    Relaxed CSV: {relaxed_p}")
    print(f"    Report JSON: {report_p}")
    print(f"    Checkpoint:  {checkpoint_path}")
    print("=" * 70)

    return 0


# ============================================================================
# Public API — importable by generation scripts
# ============================================================================
def vision_filter_dataset(
    df: pd.DataFrame,
    leya_env_path: Path,
    db_path: Path,
    pdf_root: Path,
    model: str = "gemini-2.5-flash",
    concurrency: int = 5,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Vision-based quality filter for a generated dataset DataFrame.

    Evaluates every row by sending its source PDF page(s) to Gemini and rating:
      - query_quality        → "good" | "mediocre" | "bad"
      - source_answerability → "answerable" | "partial" | "not_answerable"

    Returns:
        strict_df  — rows where query_quality="good" AND source_answerability="answerable"
        relaxed_df — rows where query_quality in {good,mediocre}
                     AND source_answerability in {answerable,partial}
        stats      — dict with summary counts (total, strict_kept, relaxed_kept, errors, …)
    """
    if checkpoint_path is None:
        checkpoint_path = SCRIPT_DIR / "processed" / "vision_filter_cache.jsonl"

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = _load_checkpoint(checkpoint_path)
    todo_indices = [i for i in df.index if i not in already_done]

    log.info(
        "vision_filter_dataset: %d rows total, %d already done, %d to evaluate",
        len(df), len(already_done), len(todo_indices),
    )

    # Build Gemini client
    client, project_id, location, model_name = _build_genai_client(
        Path(leya_env_path), model
    )
    log.info("  Vision model: %s @ %s / %s", model_name, location, project_id)

    t0 = time.monotonic()
    completed = 0
    errors = 0
    lock = threading.Lock()
    SUMMARY_EVERY = 100

    results_map: Dict[int, Dict[str, Any]] = dict(already_done)

    def _process(idx: int) -> Optional[Dict[str, Any]]:
        if _shutdown.is_set():
            return None
        row = df.loc[idx].to_dict()
        result = evaluate_row(row, idx, client, model_name, pdf_root, str(db_path))
        _append_checkpoint(checkpoint_path, result)
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {executor.submit(_process, idx): idx for idx in todo_indices}
        for future in as_completed(future_to_idx):
            if _shutdown.is_set():
                break
            idx = future_to_idx[future]
            try:
                result = future.result()
                if result is None:
                    continue
                results_map[idx] = result
                with lock:
                    completed += 1
                    if result.get("error"):
                        errors += 1

                elapsed = time.monotonic() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(todo_indices) - completed) / rate if rate > 0 else 0
                log.info(
                    "  [%d/%d] row=%d  QC=%s  SA=%s  conf=%.2f  [%.1f/s  ETA %s]%s",
                    completed, len(todo_indices), idx,
                    result.get("query_quality", "?")[:1].upper(),
                    result.get("source_answerability", "?")[:2],
                    result.get("confidence", 0.0),
                    rate, _fmt_duration(remaining),
                    f"  ERR: {result['error'][:60]}" if result.get("error") else "",
                )

                if completed % SUMMARY_EVERY == 0:
                    snap = list(results_map.values())
                    n = len(snap)
                    qc_g = sum(1 for r in snap if r.get("query_quality") == "good")
                    qc_m = sum(1 for r in snap if r.get("query_quality") == "mediocre")
                    qc_b = sum(1 for r in snap if r.get("query_quality") == "bad")
                    sa_a = sum(1 for r in snap if r.get("source_answerability") == "answerable")
                    sa_p = sum(1 for r in snap if r.get("source_answerability") == "partial")
                    sa_n = sum(1 for r in snap if r.get("source_answerability") == "not_answerable")
                    strict_n = sum(1 for r in snap if _passes_strict(r))
                    relaxed_n = sum(1 for r in snap if _passes_relaxed(r))
                    log.info(
                        "── SUMMARY (%d evaluated, %d errors) ──────────────────────────",
                        n, errors,
                    )
                    log.info(
                        "   QC:  good=%d (%.0f%%)  mediocre=%d (%.0f%%)  bad=%d (%.0f%%)",
                        qc_g, qc_g/max(n,1)*100, qc_m, qc_m/max(n,1)*100, qc_b, qc_b/max(n,1)*100,
                    )
                    log.info(
                        "   SA:  answerable=%d (%.0f%%)  partial=%d (%.0f%%)  not=%d (%.0f%%)",
                        sa_a, sa_a/max(n,1)*100, sa_p, sa_p/max(n,1)*100, sa_n, sa_n/max(n,1)*100,
                    )
                    log.info(
                        "   Filtered:  strict=%d (%.0f%%)  relaxed=%d (%.0f%%)",
                        strict_n, strict_n/max(n,1)*100, relaxed_n, relaxed_n/max(n,1)*100,
                    )
            except Exception as exc:
                log.error("Future failed for row %d: %s", idx, exc)
                errors += 1

    elapsed_total = time.monotonic() - t0
    log.info(
        "vision_filter_dataset: done in %s — %d rows, %d errors",
        _fmt_duration(elapsed_total), len(results_map), errors,
    )

    # Build verdict columns DataFrame and join back to original
    verdict_rows = []
    for r in results_map.values():
        verdict_rows.append({
            "_row_index": r["row_index"],
            "vision_query_quality": r.get("query_quality", ""),
            "vision_query_quality_reasoning": r.get("query_quality_reasoning", ""),
            "vision_source_answerability": r.get("source_answerability", ""),
            "vision_source_answerability_reasoning": r.get("source_answerability_reasoning", ""),
            "vision_confidence": r.get("confidence", 0.0),
            "vision_page_count": r.get("page_count", ""),
            "vision_error": r.get("error", ""),
        })
    verdicts_df = pd.DataFrame(verdict_rows).set_index("_row_index")
    df_with_verdicts = df.join(verdicts_df, how="left")

    strict_indices = [r["row_index"] for r in results_map.values() if _passes_strict(r)]
    relaxed_indices = [r["row_index"] for r in results_map.values() if _passes_relaxed(r)]

    strict_df = df_with_verdicts[df_with_verdicts.index.isin(strict_indices)].reset_index(drop=True)
    relaxed_df = df_with_verdicts[df_with_verdicts.index.isin(relaxed_indices)].reset_index(drop=True)

    all_results = list(results_map.values())
    n = len(all_results)
    stats: Dict[str, Any] = {
        "total": n,
        "strict_kept": len(strict_indices),
        "relaxed_kept": len(relaxed_indices),
        "errors": errors,
        "elapsed_seconds": round(elapsed_total, 1),
        "query_quality": {
            "good": sum(1 for r in all_results if r.get("query_quality") == "good"),
            "mediocre": sum(1 for r in all_results if r.get("query_quality") == "mediocre"),
            "bad": sum(1 for r in all_results if r.get("query_quality") == "bad"),
        },
        "source_answerability": {
            "answerable": sum(1 for r in all_results if r.get("source_answerability") == "answerable"),
            "partial": sum(1 for r in all_results if r.get("source_answerability") == "partial"),
            "not_answerable": sum(1 for r in all_results if r.get("source_answerability") == "not_answerable"),
        },
    }
    return strict_df, relaxed_df, stats


if __name__ == "__main__":
    sys.exit(main())
