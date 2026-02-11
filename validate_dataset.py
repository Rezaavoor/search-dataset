#!/usr/bin/env python3
"""Validate a generated single-hop dataset using LLM-as-a-judge.

For each query in the dataset, evaluates:
  1. Query Quality  — Is the query appropriate for a law-focused IR dataset?
  2. Source Answerability — Does the source page truly and properly answer the query?
  3. Hard Negative Quality — Is each hard negative a true hard negative?
     (similar to source but does NOT answer the question)

Hard negative page contents are fetched from the SQLite pdf_page_store.

Usage:
    python validate_dataset.py \
        --dataset output/single_hop_dataset_100.csv \
        --db-path processed/pdf_page_store.sqlite

    # Skip hard negative validation (faster)
    python validate_dataset.py \
        --dataset output/single_hop_dataset_100.csv \
        --skip-hard-negatives

    # Custom model / output
    python validate_dataset.py \
        --dataset output/single_hop_dataset_100.csv \
        --model gpt-4o-mini \
        --output output/validation_report
"""

import argparse
import ast
import json
import logging
import os
import random
import signal
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(SCRIPT_DIR / ".env")

from modules.utils import extract_json_object, normalize_ynu, truncate_for_judge

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("validate_dataset")

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
    log.warning("Ctrl+C received — finishing current evaluation and saving partial results...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)


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


def _invoke_with_retry(llm, messages: list, *, label: str = "") -> Optional[str]:
    """Invoke LLM with exponential backoff retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        if _shutdown_event.is_set():
            return None
        try:
            resp = llm.invoke(messages)
            content = getattr(resp, "content", None)
            return content if isinstance(content, str) else str(resp)
        except Exception as exc:
            if _is_retriable(exc) and attempt < MAX_RETRIES - 1:
                delay = _backoff_delay(attempt)
                log.warning(
                    "Retriable error%s (attempt %d/%d), retrying in %.1fs: %s",
                    f" [{label}]" if label else "",
                    attempt + 1, MAX_RETRIES, delay, str(exc)[:120],
                )
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not _shutdown_event.is_set():
                    time.sleep(min(1.0, deadline - time.monotonic()))
                continue
            else:
                log.error("LLM call failed%s: %s", f" [{label}]" if label else "", str(exc)[:200])
                return None
    return None


# ---------------------------------------------------------------------------
# Azure credential discovery (reuses pattern from generate_single_hop.py)
# ---------------------------------------------------------------------------
def _discover_llm_client(model: str):
    """Create a single LLM client (Azure or OpenAI)."""
    # Try Azure first
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if key and endpoint:
        from langchain_openai import AzureChatOpenAI
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", model)
        log.info("  LLM provider: Azure OpenAI (deployment: %s)", deployment)
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=0.1,
            request_timeout=90,
            max_retries=0,
        )

    # Fallback to OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        from langchain_openai import ChatOpenAI
        log.info("  LLM provider: OpenAI (model: %s)", model)
        return ChatOpenAI(model=model, temperature=0.1)

    return None


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------
def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _fetch_page_content(conn: sqlite3.Connection, filename: str, page_number: int) -> Optional[str]:
    """Fetch page content from SQLite by filename and page number."""
    row = conn.execute(
        "SELECT doc_content FROM pdf_page_store WHERE filename = ? AND page_number = ? LIMIT 1",
        (filename, page_number),
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    return None


def _parse_hard_negatives(value) -> List[Tuple[str, int]]:
    """Parse hard_negatives column into list of (filename, page) tuples."""
    if pd.isna(value) or not value:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value or value == "[]":
            return []
        try:
            items = json.loads(value)
        except Exception:
            try:
                items = ast.literal_eval(value)
            except Exception:
                return []
    elif isinstance(value, list):
        items = value
    else:
        return []

    results = []
    for item in items:
        s = str(item).strip()
        # Parse "filename.pdf (page N)"
        if " (page " in s and s.endswith(")"):
            idx = s.rfind(" (page ")
            fname = s[:idx].strip()
            try:
                page = int(s[idx + 7:-1])
                results.append((fname, page))
            except ValueError:
                pass
    return results


def _parse_reference_contexts(value) -> List[str]:
    """Parse reference_contexts column into a list of strings."""
    if pd.isna(value) or not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            return [str(v)]
        except Exception:
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(x) for x in v if x is not None]
                return [str(v)]
            except Exception:
                return [s]
    return [str(value)]


# ============================================================================
# LLM Judge Prompts
# ============================================================================

# --- 1. Query Quality ---
QUERY_QUALITY_SYSTEM = (
    "You are an expert evaluator for Information Retrieval (IR) datasets in the legal domain.\n"
    "Your task is to assess whether a search query is appropriate for a law-focused IR dataset.\n\n"
    "A GOOD query should:\n"
    "- Be a standalone, self-contained question (no deictic references like 'this case', 'the above')\n"
    "- Be relevant to law, legal proceedings, regulations, contracts, compliance, or legal research\n"
    "- Contain concrete identifiers (case names, party names, statute numbers, docket numbers, etc.)\n"
    "- Be the kind of query a legal professional would plausibly search for\n"
    "- Not reference filenames, page numbers, or internal document identifiers\n\n"
    "A BAD query might:\n"
    "- Be too vague or generic (no legal specificity)\n"
    "- Contain deictic references ('this document', 'herein', 'the above')\n"
    "- Reference filenames or internal metadata\n"
    "- Be nonsensical or unanswerable\n"
    "- Not be relevant to the legal domain at all\n\n"
    "Return ONLY valid JSON."
)

QUERY_QUALITY_USER_TEMPLATE = (
    "QUERY:\n{query}\n\n"
    "Evaluate this query for a law-focused IR dataset.\n\n"
    "Return JSON:\n"
    '{{"verdict": "pass|fail|borderline", '
    '"standalone": "yes|no", '
    '"legal_relevance": "high|medium|low|none", '
    '"has_concrete_identifiers": "yes|no", '
    '"issues": ["<list any issues found>"], '
    '"reasoning": "<brief explanation>"}}'
)

# --- 2. Source Answerability ---
SOURCE_ANSWER_SYSTEM = (
    "You are a strict evaluator for IR dataset quality.\n"
    "Your task is to determine whether a given SOURCE PAGE truly and properly answers the QUERY.\n\n"
    "Criteria:\n"
    "- The source page must contain information that DIRECTLY answers the query.\n"
    "- The provided ANSWER must be grounded in and supported by the source page content.\n"
    "- If the source page only tangentially relates to the query but doesn't contain the answer, "
    "that is a FAIL.\n"
    "- If the answer adds information NOT found in the source page, that is a FAIL.\n\n"
    "Use ONLY the provided source page content. Do not use outside knowledge.\n"
    "Return ONLY valid JSON."
)

SOURCE_ANSWER_USER_TEMPLATE = (
    "QUERY:\n{query}\n\n"
    "GENERATED ANSWER:\n{answer}\n\n"
    "SOURCE PAGE CONTENT:\n{context}\n\n"
    "Evaluate whether the source page properly answers the query and whether the "
    "generated answer is faithful to the source page.\n\n"
    "Return JSON:\n"
    '{{"verdict": "pass|fail|borderline", '
    '"source_answers_query": "yes|partially|no", '
    '"answer_faithful_to_source": "yes|partially|no", '
    '"answer_adds_unsupported_info": "yes|no", '
    '"issues": ["<list any issues found>"], '
    '"reasoning": "<brief explanation>"}}'
)

# --- 3. Hard Negative Quality ---
HARD_NEG_SYSTEM = (
    "You are a strict evaluator for IR hard negatives.\n"
    "A GOOD hard negative is a passage that:\n"
    "- Is topically SIMILAR to the query (shares domain, entities, or legal concepts)\n"
    "- Does NOT contain enough information to answer the query\n"
    "- Would be misleading to a retrieval system (looks relevant but isn't answerable)\n\n"
    "A BAD hard negative is one that:\n"
    "- Actually answers the query (false negative — this is the worst failure)\n"
    "- Is completely unrelated to the query (too easy to distinguish)\n\n"
    "Use ONLY the provided passage content. Do not use outside knowledge.\n"
    "Return ONLY valid JSON."
)

HARD_NEG_USER_TEMPLATE = (
    "QUERY:\n{query}\n\n"
    "HARD NEGATIVE PASSAGE:\n{passage}\n\n"
    "Evaluate whether this passage is a valid hard negative for the query.\n\n"
    "Return JSON:\n"
    '{{"verdict": "pass|fail|borderline", '
    '"topical_similarity": "high|medium|low|none", '
    '"answers_the_query": "yes|partially|no", '
    '"reasoning": "<brief explanation>"}}'
)


# ============================================================================
# Evaluation Functions
# ============================================================================
def evaluate_query_quality(llm, query: str, idx: int) -> Dict[str, Any]:
    """Evaluate whether a query is appropriate for a law IR dataset."""
    user_msg = QUERY_QUALITY_USER_TEMPLATE.format(query=query)
    messages = [SystemMessage(content=QUERY_QUALITY_SYSTEM), HumanMessage(content=user_msg)]

    content = _invoke_with_retry(llm, messages, label=f"query_quality_{idx}")
    if content is None:
        return {"verdict": "error", "reasoning": "LLM call failed"}

    data = extract_json_object(content)
    if not isinstance(data, dict) or "verdict" not in data:
        return {"verdict": "error", "reasoning": f"Bad LLM response: {content[:200]}"}

    return data


def evaluate_source_answerability(
    llm, query: str, answer: str, context: str, idx: int,
) -> Dict[str, Any]:
    """Evaluate whether the source page properly answers the query."""
    context_snippet = truncate_for_judge(context, max_chars=8000)
    user_msg = SOURCE_ANSWER_USER_TEMPLATE.format(
        query=query, answer=answer, context=context_snippet,
    )
    messages = [SystemMessage(content=SOURCE_ANSWER_SYSTEM), HumanMessage(content=user_msg)]

    content = _invoke_with_retry(llm, messages, label=f"source_answer_{idx}")
    if content is None:
        return {"verdict": "error", "reasoning": "LLM call failed"}

    data = extract_json_object(content)
    if not isinstance(data, dict) or "verdict" not in data:
        return {"verdict": "error", "reasoning": f"Bad LLM response: {content[:200]}"}

    return data


def evaluate_hard_negative(
    llm, query: str, passage: str, idx: int, hn_idx: int,
) -> Dict[str, Any]:
    """Evaluate whether a hard negative is a valid hard negative."""
    passage_snippet = truncate_for_judge(passage, max_chars=8000)
    user_msg = HARD_NEG_USER_TEMPLATE.format(query=query, passage=passage_snippet)
    messages = [SystemMessage(content=HARD_NEG_SYSTEM), HumanMessage(content=user_msg)]

    content = _invoke_with_retry(llm, messages, label=f"hard_neg_{idx}_{hn_idx}")
    if content is None:
        return {"verdict": "error", "reasoning": "LLM call failed"}

    data = extract_json_object(content)
    if not isinstance(data, dict) or "verdict" not in data:
        return {"verdict": "error", "reasoning": f"Bad LLM response: {content[:200]}"}

    return data


# ============================================================================
# Main Pipeline
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a generated dataset using LLM-as-a-judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str,
        default=str(SCRIPT_DIR / "output" / "single_hop_dataset_100.csv"),
        help="Path to dataset CSV (default: output/single_hop_dataset_100.csv)",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="SQLite DB path for hard negative page content (default: processed/pdf_page_store.sqlite)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="LLM model / deployment name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output base name (default: output/validation_report_<timestamp>)",
    )
    parser.add_argument(
        "--skip-hard-negatives", action="store_true",
        help="Skip hard negative evaluation",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Evaluate only the first N rows (for testing)",
    )
    args = parser.parse_args()

    # --- Resolve paths ---
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        log.error("Dataset not found: %s", dataset_path)
        return 1

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (SCRIPT_DIR / "processed" / "pdf_page_store.sqlite").resolve()

    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_base = Path(args.output).expanduser().resolve()
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_base = output_dir / f"validation_report_{ts}"

    log.info("=" * 70)
    log.info("Dataset Validation (LLM-as-a-Judge)")
    log.info("=" * 70)
    log.info("  Dataset:   %s", dataset_path)
    log.info("  DB:        %s", db_path)
    log.info("  Model:     %s", args.model)
    log.info("  Output:    %s.*", output_base)

    # --- Load dataset ---
    df = pd.read_csv(dataset_path)
    total_rows = len(df)
    if args.max_rows:
        df = df.head(args.max_rows)
    log.info("  Rows:      %d (evaluating %d)", total_rows, len(df))

    # --- Setup LLM ---
    llm = _discover_llm_client(args.model)
    if llm is None:
        log.error("No LLM credentials found. Set AZURE_OPENAI_API_KEY/ENDPOINT or OPENAI_API_KEY.")
        return 1

    # --- Open SQLite DB (for hard negative page lookups) ---
    conn = None
    if db_path.exists():
        conn = _open_db(db_path)
        log.info("  SQLite DB opened: %s", db_path)
    else:
        log.warning("  SQLite DB not found: %s — hard negative content lookup disabled", db_path)

    # --- Evaluate each row ---
    log.info("")
    log.info("Starting evaluation...")
    t0 = time.monotonic()

    row_results: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        if _shutdown_event.is_set():
            log.warning("Shutdown requested — saving partial results")
            break

        query = str(row.get("user_input", ""))
        answer = str(row.get("reference", ""))
        contexts = _parse_reference_contexts(row.get("reference_contexts"))
        context_text = "\n\n".join(contexts) if contexts else ""
        source_file = str(row.get("source_file", ""))
        source_with_page = str(row.get("source_file_with_page", ""))
        hard_negs_raw = row.get("hard_negatives", "[]")

        log.info("[%d/%d] Evaluating: %.80s...", idx + 1, len(df), query)

        # --- 1. Query Quality ---
        qc = evaluate_query_quality(llm, query, idx)

        # --- 2. Source Answerability ---
        sa = evaluate_source_answerability(llm, query, answer, context_text, idx)

        # --- 3. Hard Negative Quality ---
        hn_evaluations: List[Dict[str, Any]] = []
        hard_neg_pairs = _parse_hard_negatives(hard_negs_raw)

        if hard_neg_pairs and not args.skip_hard_negatives:
            for hn_idx, (hn_file, hn_page) in enumerate(hard_neg_pairs):
                if _shutdown_event.is_set():
                    break
                # Fetch page content from SQLite
                hn_content = None
                if conn is not None:
                    hn_content = _fetch_page_content(conn, hn_file, hn_page)

                if hn_content is None:
                    hn_evaluations.append({
                        "file": hn_file,
                        "page": hn_page,
                        "verdict": "skipped",
                        "reasoning": "Page content not found in SQLite DB",
                    })
                    continue

                hn_eval = evaluate_hard_negative(llm, query, hn_content, idx, hn_idx)
                hn_eval["file"] = hn_file
                hn_eval["page"] = hn_page
                hn_evaluations.append(hn_eval)

        row_result = {
            "row_index": int(idx),
            "query": query,
            "answer": answer[:200],
            "source_file_with_page": source_with_page,
            "query_quality": qc,
            "source_answerability": sa,
            "hard_negative_count": len(hard_neg_pairs),
            "hard_negative_evaluations": hn_evaluations,
        }
        row_results.append(row_result)

        # Progress
        elapsed = time.monotonic() - t0
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(df) - idx - 1) / rate if rate > 0 else 0
        log.info(
            "  QC=%s  SA=%s  HN=%d/%d  [%.1f/s, ETA %s]",
            qc.get("verdict", "?"),
            sa.get("verdict", "?"),
            sum(1 for h in hn_evaluations if h.get("verdict") == "pass"),
            len(hn_evaluations),
            rate,
            _fmt_duration(remaining),
        )

    elapsed_total = time.monotonic() - t0

    if conn:
        conn.close()

    # ======================================================================
    # Compute aggregate statistics
    # ======================================================================
    log.info("")
    log.info("=" * 70)
    log.info("Computing statistics...")

    evaluated = len(row_results)

    # --- Query Quality stats ---
    qc_verdicts = [r["query_quality"].get("verdict", "error") for r in row_results]
    qc_pass = qc_verdicts.count("pass")
    qc_fail = qc_verdicts.count("fail")
    qc_borderline = qc_verdicts.count("borderline")
    qc_error = qc_verdicts.count("error")

    qc_standalone = sum(
        1 for r in row_results
        if r["query_quality"].get("standalone") in ("yes", "Yes")
    )
    qc_legal_high = sum(
        1 for r in row_results
        if r["query_quality"].get("legal_relevance") in ("high", "High")
    )
    qc_legal_medium = sum(
        1 for r in row_results
        if r["query_quality"].get("legal_relevance") in ("medium", "Medium")
    )
    qc_legal_low = sum(
        1 for r in row_results
        if r["query_quality"].get("legal_relevance") in ("low", "Low")
    )
    qc_identifiers = sum(
        1 for r in row_results
        if r["query_quality"].get("has_concrete_identifiers") in ("yes", "Yes")
    )

    # --- Source Answerability stats ---
    sa_verdicts = [r["source_answerability"].get("verdict", "error") for r in row_results]
    sa_pass = sa_verdicts.count("pass")
    sa_fail = sa_verdicts.count("fail")
    sa_borderline = sa_verdicts.count("borderline")
    sa_error = sa_verdicts.count("error")

    sa_source_yes = sum(
        1 for r in row_results
        if r["source_answerability"].get("source_answers_query") in ("yes", "Yes")
    )
    sa_faithful_yes = sum(
        1 for r in row_results
        if r["source_answerability"].get("answer_faithful_to_source") in ("yes", "Yes")
    )
    sa_unsupported = sum(
        1 for r in row_results
        if r["source_answerability"].get("answer_adds_unsupported_info") in ("yes", "Yes")
    )

    # --- Hard Negative stats ---
    total_hn_count = sum(r["hard_negative_count"] for r in row_results)
    all_hn_evals = [
        hn for r in row_results for hn in r["hard_negative_evaluations"]
    ]
    hn_evaluated = [h for h in all_hn_evals if h.get("verdict") != "skipped"]
    hn_pass = sum(1 for h in hn_evaluated if h.get("verdict") == "pass")
    hn_fail = sum(1 for h in hn_evaluated if h.get("verdict") == "fail")
    hn_borderline = sum(1 for h in hn_evaluated if h.get("verdict") == "borderline")
    hn_error = sum(1 for h in hn_evaluated if h.get("verdict") == "error")
    hn_skipped = sum(1 for h in all_hn_evals if h.get("verdict") == "skipped")

    # Check for false negatives (hard negatives that actually answer the query)
    hn_false_negatives = [
        h for h in hn_evaluated
        if h.get("answers_the_query") in ("yes", "Yes")
    ]

    # --- Rows with ALL checks passing ---
    fully_passing = sum(
        1 for r in row_results
        if r["query_quality"].get("verdict") == "pass"
        and r["source_answerability"].get("verdict") == "pass"
        and all(
            h.get("verdict") in ("pass", "skipped")
            for h in r["hard_negative_evaluations"]
        )
    )

    # --- Collect failed rows for the report ---
    query_quality_failures = [
        {
            "row": r["row_index"],
            "query": r["query"],
            "verdict": r["query_quality"].get("verdict"),
            "issues": r["query_quality"].get("issues", []),
            "reasoning": r["query_quality"].get("reasoning", ""),
        }
        for r in row_results
        if r["query_quality"].get("verdict") in ("fail", "borderline")
    ]

    source_answer_failures = [
        {
            "row": r["row_index"],
            "query": r["query"],
            "source": r["source_file_with_page"],
            "verdict": r["source_answerability"].get("verdict"),
            "issues": r["source_answerability"].get("issues", []),
            "reasoning": r["source_answerability"].get("reasoning", ""),
        }
        for r in row_results
        if r["source_answerability"].get("verdict") in ("fail", "borderline")
    ]

    hard_neg_failures = [
        {
            "row": r["row_index"],
            "query": r["query"],
            "hard_negative": f"{h.get('file', '?')} (page {h.get('page', '?')})",
            "verdict": h.get("verdict"),
            "answers_the_query": h.get("answers_the_query"),
            "reasoning": h.get("reasoning", ""),
        }
        for r in row_results
        for h in r["hard_negative_evaluations"]
        if h.get("verdict") in ("fail", "borderline")
    ]

    # ======================================================================
    # Build report
    # ======================================================================
    report = {
        "metadata": {
            "dataset": str(dataset_path),
            "model": args.model,
            "evaluated_rows": evaluated,
            "total_rows": total_rows,
            "elapsed_seconds": round(elapsed_total, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "query_quality": {
                "pass": qc_pass,
                "fail": qc_fail,
                "borderline": qc_borderline,
                "error": qc_error,
                "pass_rate": round(qc_pass / max(evaluated, 1) * 100, 1),
                "standalone_queries": qc_standalone,
                "legal_relevance_high": qc_legal_high,
                "legal_relevance_medium": qc_legal_medium,
                "legal_relevance_low": qc_legal_low,
                "has_concrete_identifiers": qc_identifiers,
            },
            "source_answerability": {
                "pass": sa_pass,
                "fail": sa_fail,
                "borderline": sa_borderline,
                "error": sa_error,
                "pass_rate": round(sa_pass / max(evaluated, 1) * 100, 1),
                "source_answers_query_yes": sa_source_yes,
                "answer_faithful_to_source_yes": sa_faithful_yes,
                "answer_adds_unsupported_info": sa_unsupported,
            },
            "hard_negatives": {
                "total_hard_negatives": total_hn_count,
                "evaluated": len(hn_evaluated),
                "skipped": hn_skipped,
                "pass": hn_pass,
                "fail": hn_fail,
                "borderline": hn_borderline,
                "error": hn_error,
                "pass_rate": round(hn_pass / max(len(hn_evaluated), 1) * 100, 1),
                "false_negatives_count": len(hn_false_negatives),
            },
            "overall": {
                "fully_passing_rows": fully_passing,
                "fully_passing_rate": round(fully_passing / max(evaluated, 1) * 100, 1),
            },
        },
        "failures": {
            "query_quality_issues": query_quality_failures,
            "source_answerability_issues": source_answer_failures,
            "hard_negative_issues": hard_neg_failures,
        },
        "per_row_results": row_results,
    }

    # ======================================================================
    # Save report
    # ======================================================================
    json_path = f"{output_base}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info("  Full report: %s", json_path)

    # Save a concise CSV summary
    csv_rows = []
    for r in row_results:
        csv_rows.append({
            "row_index": r["row_index"],
            "query": r["query"][:120],
            "source": r["source_file_with_page"],
            "query_quality_verdict": r["query_quality"].get("verdict"),
            "query_standalone": r["query_quality"].get("standalone"),
            "query_legal_relevance": r["query_quality"].get("legal_relevance"),
            "query_identifiers": r["query_quality"].get("has_concrete_identifiers"),
            "source_verdict": r["source_answerability"].get("verdict"),
            "source_answers": r["source_answerability"].get("source_answers_query"),
            "answer_faithful": r["source_answerability"].get("answer_faithful_to_source"),
            "answer_unsupported": r["source_answerability"].get("answer_adds_unsupported_info"),
            "hard_neg_count": r["hard_negative_count"],
            "hard_neg_pass": sum(
                1 for h in r["hard_negative_evaluations"] if h.get("verdict") == "pass"
            ),
            "hard_neg_fail": sum(
                1 for h in r["hard_negative_evaluations"] if h.get("verdict") == "fail"
            ),
        })
    csv_df = pd.DataFrame(csv_rows)
    csv_path = f"{output_base}.csv"
    csv_df.to_csv(csv_path, index=False)
    log.info("  CSV summary: %s", csv_path)

    # ======================================================================
    # Print analysis to stdout
    # ======================================================================
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    print(f"\nDataset: {dataset_path.name}")
    print(f"Rows evaluated: {evaluated}/{total_rows}")
    print(f"Elapsed: {_fmt_duration(elapsed_total)}")

    print(f"\n{'─' * 70}")
    print("1. QUERY QUALITY")
    print(f"{'─' * 70}")
    print(f"  Pass:        {qc_pass:>4}  ({qc_pass/max(evaluated,1)*100:5.1f}%)")
    print(f"  Borderline:  {qc_borderline:>4}  ({qc_borderline/max(evaluated,1)*100:5.1f}%)")
    print(f"  Fail:        {qc_fail:>4}  ({qc_fail/max(evaluated,1)*100:5.1f}%)")
    print(f"  Error:       {qc_error:>4}")
    print(f"  ---")
    print(f"  Standalone:            {qc_standalone:>4}/{evaluated}")
    print(f"  Legal relevance high:  {qc_legal_high:>4}/{evaluated}")
    print(f"  Legal relevance med:   {qc_legal_medium:>4}/{evaluated}")
    print(f"  Legal relevance low:   {qc_legal_low:>4}/{evaluated}")
    print(f"  Has identifiers:       {qc_identifiers:>4}/{evaluated}")

    print(f"\n{'─' * 70}")
    print("2. SOURCE ANSWERABILITY")
    print(f"{'─' * 70}")
    print(f"  Pass:        {sa_pass:>4}  ({sa_pass/max(evaluated,1)*100:5.1f}%)")
    print(f"  Borderline:  {sa_borderline:>4}  ({sa_borderline/max(evaluated,1)*100:5.1f}%)")
    print(f"  Fail:        {sa_fail:>4}  ({sa_fail/max(evaluated,1)*100:5.1f}%)")
    print(f"  Error:       {sa_error:>4}")
    print(f"  ---")
    print(f"  Source answers query:          {sa_source_yes:>4}/{evaluated}")
    print(f"  Answer faithful to source:     {sa_faithful_yes:>4}/{evaluated}")
    print(f"  Answer adds unsupported info:  {sa_unsupported:>4}/{evaluated}")

    print(f"\n{'─' * 70}")
    print("3. HARD NEGATIVES")
    print(f"{'─' * 70}")
    print(f"  Total hard negatives in dataset:  {total_hn_count}")
    print(f"  Evaluated:   {len(hn_evaluated):>4}")
    print(f"  Skipped:     {hn_skipped:>4}")
    if hn_evaluated:
        print(f"  Pass:        {hn_pass:>4}  ({hn_pass/max(len(hn_evaluated),1)*100:5.1f}%)")
        print(f"  Borderline:  {hn_borderline:>4}  ({hn_borderline/max(len(hn_evaluated),1)*100:5.1f}%)")
        print(f"  Fail:        {hn_fail:>4}  ({hn_fail/max(len(hn_evaluated),1)*100:5.1f}%)")
        print(f"  Error:       {hn_error:>4}")
        print(f"  ---")
        print(f"  FALSE NEGATIVES (actually answer the query): {len(hn_false_negatives)}")
    else:
        print(f"  (No hard negatives to evaluate)")

    rows_with_hn = sum(1 for r in row_results if r["hard_negative_count"] > 0)
    rows_without_hn = evaluated - rows_with_hn
    print(f"  ---")
    print(f"  Rows with hard negatives:     {rows_with_hn}/{evaluated}")
    print(f"  Rows without hard negatives:  {rows_without_hn}/{evaluated}")

    print(f"\n{'─' * 70}")
    print("4. OVERALL")
    print(f"{'─' * 70}")
    print(f"  Fully passing rows:  {fully_passing:>4}/{evaluated}  ({fully_passing/max(evaluated,1)*100:5.1f}%)")

    # --- Top issues ---
    if query_quality_failures:
        print(f"\n{'─' * 70}")
        print(f"QUERY QUALITY ISSUES ({len(query_quality_failures)} queries)")
        print(f"{'─' * 70}")
        for f in query_quality_failures[:10]:
            print(f"  Row {f['row']:>3}: [{f['verdict']}] {f['query'][:80]}")
            if f.get("issues"):
                for issue in f["issues"][:2]:
                    print(f"           - {issue}")
            if f.get("reasoning"):
                print(f"           Reason: {f['reasoning'][:120]}")

    if source_answer_failures:
        print(f"\n{'─' * 70}")
        print(f"SOURCE ANSWERABILITY ISSUES ({len(source_answer_failures)} queries)")
        print(f"{'─' * 70}")
        for f in source_answer_failures[:10]:
            print(f"  Row {f['row']:>3}: [{f['verdict']}] {f['query'][:80]}")
            print(f"           Source: {f['source'][:60]}")
            if f.get("issues"):
                for issue in f["issues"][:2]:
                    print(f"           - {issue}")
            if f.get("reasoning"):
                print(f"           Reason: {f['reasoning'][:120]}")

    if hard_neg_failures:
        print(f"\n{'─' * 70}")
        print(f"HARD NEGATIVE ISSUES ({len(hard_neg_failures)} items)")
        print(f"{'─' * 70}")
        for f in hard_neg_failures[:10]:
            print(f"  Row {f['row']:>3}: [{f['verdict']}] {f['query'][:60]}")
            print(f"           HN: {f['hard_negative'][:60]}")
            if f.get("answers_the_query") in ("yes", "Yes"):
                print(f"           *** FALSE NEGATIVE: actually answers the query! ***")
            if f.get("reasoning"):
                print(f"           Reason: {f['reasoning'][:120]}")

    print(f"\n{'=' * 70}")
    print(f"Reports saved to: {output_base}.*")
    print(f"{'=' * 70}\n")

    return 0


if __name__ == "__main__":
    exit(main())
