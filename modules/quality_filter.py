"""LLM-as-a-judge quality filtering for generated datasets.

Evaluates each row on:
  1. Query Quality (QC) — Is the query standalone, specific, and appropriate?
  2. Source Answerability (SA) — Does the labeled source actually answer the query?

Rows failing either check are dropped.  Rows with unknown source_files are
also dropped.

Also provides ``expand_hard_negatives_columns`` to split the single
``hard_negatives`` JSON column into three columns that mirror the
source-file column format.
"""

import ast
import json
import logging
import random
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from .utils import extract_json_object, strip_hop_prefix, truncate_for_judge

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helpers
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


def _is_retriable(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(s in err for s in RETRIABLE_SUBSTRINGS)


def _backoff_delay(attempt: int) -> float:
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = delay * JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.5, delay + jitter)


def _invoke_with_retry(llm, messages: list, *, label: str = "") -> Optional[str]:
    """Invoke LLM with exponential backoff retry on transient errors."""
    for attempt in range(MAX_RETRIES):
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
                time.sleep(delay)
                continue
            log.error(
                "LLM call failed%s: %s",
                f" [{label}]" if label else "", str(exc)[:200],
            )
            return None
    return None


# ============================================================================
# LLM Judge Prompts
# ============================================================================

# --- 1. Query Quality (domain-agnostic — no legal-relevance check) ---
QUERY_QUALITY_SYSTEM = (
    "You are an expert evaluator for Information Retrieval (IR) datasets.\n"
    "Your task is to assess whether a search query is appropriate for an IR "
    "dataset.\n\n"
    "A GOOD query should:\n"
    "- Be a standalone, self-contained question (no deictic references like "
    "'this case', 'the above')\n"
    "- Contain concrete identifiers (names, numbers, specific terms, etc.)\n"
    "- Be the kind of query someone would plausibly search for\n"
    "- Not reference filenames, page numbers, or internal document "
    "identifiers\n\n"
    "A BAD query might:\n"
    "- Be too vague or generic (no specificity)\n"
    "- Contain deictic references ('this document', 'herein', 'the above')\n"
    "- Reference filenames or internal metadata\n"
    "- Be nonsensical or unanswerable\n\n"
    "Return ONLY valid JSON."
)

QUERY_QUALITY_USER_TEMPLATE = (
    "QUERY:\n{query}\n\n"
    "Evaluate this query for an IR dataset.\n\n"
    "Return JSON:\n"
    '{{"verdict": "pass|fail|borderline", '
    '"standalone": "yes|no", '
    '"has_concrete_identifiers": "yes|no", '
    '"issues": ["<list any issues found>"], '
    '"reasoning": "<brief explanation>"}}'
)

# --- 2. Source Answerability ---
SOURCE_ANSWER_SYSTEM = (
    "You are a strict evaluator for IR dataset quality.\n"
    "Your task is to determine whether a given SOURCE PAGE truly and properly "
    "answers the QUERY.\n\n"
    "Criteria:\n"
    "- The source page must contain information that DIRECTLY answers the "
    "query.\n"
    "- The provided ANSWER must be grounded in and supported by the source "
    "page content.\n"
    "- If the source page only tangentially relates to the query but doesn't "
    "contain the answer, that is a FAIL.\n"
    "- If the answer adds information NOT found in the source page, that is "
    "a FAIL.\n\n"
    "Use ONLY the provided source page content. Do not use outside "
    "knowledge.\n"
    "Return ONLY valid JSON."
)

SOURCE_ANSWER_USER_TEMPLATE = (
    "QUERY:\n{query}\n\n"
    "GENERATED ANSWER:\n{answer}\n\n"
    "SOURCE PAGE CONTENT:\n{context}\n\n"
    "Evaluate whether the source page properly answers the query and whether "
    "the generated answer is faithful to the source page.\n\n"
    "Return JSON:\n"
    '{{"verdict": "pass|fail|borderline", '
    '"source_answers_query": "yes|partially|no", '
    '"answer_faithful_to_source": "yes|partially|no", '
    '"answer_adds_unsupported_info": "yes|no", '
    '"issues": ["<list any issues found>"], '
    '"reasoning": "<brief explanation>"}}'
)


# ============================================================================
# Evaluation functions
# ============================================================================
def evaluate_query_quality(llm, query: str, idx: int) -> Dict[str, Any]:
    """Evaluate whether a query is appropriate for an IR dataset."""
    user_msg = QUERY_QUALITY_USER_TEMPLATE.format(query=query)
    messages = [
        SystemMessage(content=QUERY_QUALITY_SYSTEM),
        HumanMessage(content=user_msg),
    ]
    content = _invoke_with_retry(llm, messages, label=f"qc_{idx}")
    if content is None:
        return {"verdict": "error", "reasoning": "LLM call failed"}
    data = extract_json_object(content)
    if not isinstance(data, dict) or "verdict" not in data:
        return {"verdict": "error", "reasoning": f"Bad response: {content[:200]}"}
    return data


def evaluate_source_answerability(
    llm, query: str, answer: str, context: str, idx: int,
) -> Dict[str, Any]:
    """Evaluate whether the source page properly answers the query."""
    context_snippet = truncate_for_judge(context, max_chars=8000)
    user_msg = SOURCE_ANSWER_USER_TEMPLATE.format(
        query=query, answer=answer, context=context_snippet,
    )
    messages = [
        SystemMessage(content=SOURCE_ANSWER_SYSTEM),
        HumanMessage(content=user_msg),
    ]
    content = _invoke_with_retry(llm, messages, label=f"sa_{idx}")
    if content is None:
        return {"verdict": "error", "reasoning": "LLM call failed"}
    data = extract_json_object(content)
    if not isinstance(data, dict) or "verdict" not in data:
        return {"verdict": "error", "reasoning": f"Bad response: {content[:200]}"}
    return data


# ============================================================================
# SQLite helpers (for fetching positive page content)
# ============================================================================
def _fetch_page_content(
    conn: sqlite3.Connection, filename: str, page_number: int,
) -> Optional[str]:
    """Fetch page content from SQLite by filename and page number."""
    row = conn.execute(
        "SELECT doc_content FROM pdf_page_store "
        "WHERE filename = ? AND page_number = ? LIMIT 1",
        (filename, page_number),
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def _parse_page_ref(ref: str) -> Optional[Tuple[str, int]]:
    """Parse ``'filename.pdf (page N)'`` → ``(filename, page_number)``."""
    ref = ref.strip()
    if " (page " in ref and ref.endswith(")"):
        idx = ref.rfind(" (page ")
        fname = ref[:idx].strip()
        try:
            return (fname, int(ref[idx + 7:-1]))
        except ValueError:
            return None
    return None


def _fetch_positive_page_content(
    row, conn: Optional[sqlite3.Connection],
) -> Optional[str]:
    """Fetch actual page content for the labeled positive page(s)."""
    if conn is None:
        return None

    items: List[str] = []

    # Try source_files_with_pages (JSON array — works for both hops)
    raw = row.get("source_files_with_pages")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                items.extend(str(x) for x in parsed)
        except Exception:
            pass

    # Also try the single-hop column
    sfwp = row.get("source_file_with_page")
    if isinstance(sfwp, str) and sfwp.strip() and sfwp.strip().lower() != "nan":
        if sfwp.strip() not in items:
            items.insert(0, sfwp.strip())

    if not items:
        return None

    page_texts: List[str] = []
    for item in items:
        parsed = _parse_page_ref(str(item))
        if parsed is None:
            continue
        fname, page_num = parsed
        content = _fetch_page_content(conn, fname, page_num)
        if content:
            page_texts.append(content)

    return "\n\n".join(page_texts) if page_texts else None


def _parse_reference_contexts(value) -> List[str]:
    """Parse reference_contexts column into a list of strings."""
    if pd.isna(value) or not value:
        return []
    if isinstance(value, list):
        return [strip_hop_prefix(str(v)) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [strip_hop_prefix(str(x)) for x in v if x is not None]
            return [strip_hop_prefix(str(v))]
        except Exception:
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [strip_hop_prefix(str(x)) for x in v if x is not None]
                return [strip_hop_prefix(str(v))]
            except Exception:
                return [strip_hop_prefix(s)]
    return [strip_hop_prefix(str(value))]


# ============================================================================
# Hard-negatives column expansion
# ============================================================================
def _parse_hn_items(value) -> List[str]:
    """Parse a hard_negatives cell into a flat list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x]
    if isinstance(value, str):
        s = value.strip()
        if not s or s == "[]":
            return []
        try:
            items = json.loads(s)
            if isinstance(items, list):
                return [str(x) for x in items if x]
        except Exception:
            try:
                items = ast.literal_eval(s)
                if isinstance(items, list):
                    return [str(x) for x in items if x]
            except Exception:
                pass
    return []


def expand_hard_negatives_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``hard_negative_files`` and ``hard_negative_pages`` columns.

    Parses the existing ``hard_negatives`` column (JSON array of
    ``"filename (page N)"`` strings) into two additional columns that
    mirror the ``source_files`` / ``page_numbers`` format.

    The original ``hard_negatives`` column is kept as-is.
    """
    if "hard_negatives" not in df.columns:
        return df

    hn_files_col: List[str] = []
    hn_pages_col: List[str] = []

    for _, row in df.iterrows():
        items = _parse_hn_items(row.get("hard_negatives"))
        files: List[str] = []
        pages: List[Optional[int]] = []
        for item in items:
            parsed = _parse_page_ref(str(item))
            if parsed:
                files.append(parsed[0])
                pages.append(parsed[1])
        hn_files_col.append(json.dumps(files))
        hn_pages_col.append(json.dumps(pages))

    df["hard_negative_files"] = hn_files_col
    df["hard_negative_pages"] = hn_pages_col
    return df


# ============================================================================
# Main filter function
# ============================================================================
def filter_dataset(
    df: pd.DataFrame,
    llm,
    conn: Optional[sqlite3.Connection] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter dataset using LLM-as-a-judge for QC and SA.

    Drops rows that:
      1. Have ``"unknown"`` in source_files
      2. Fail query quality (QC verdict != ``"pass"``)
      3. Fail source answerability (SA verdict != ``"pass"``)

    Args:
        df: Generated dataset DataFrame.
        llm: Raw LangChain chat model (NOT RAGAS-wrapped).
        conn: Optional SQLite connection for fetching page content.

    Returns:
        ``(filtered_df, stats_dict)``
    """
    total = len(df)
    if total == 0:
        return df.copy(), {
            "total": 0, "kept": 0, "dropped": 0,
            "unknown_source_dropped": 0, "qc_dropped": 0, "sa_dropped": 0,
        }

    print(f"  Running quality filter on {total} rows...")
    t0 = time.monotonic()

    keep_mask = [True] * total
    rows = list(df.iterrows())

    # --- Phase 0: Drop unknown source_files ---
    unknown_dropped = 0
    for i, (_, row) in enumerate(rows):
        sf = str(row.get("source_files", "")).lower()
        sfwp = str(row.get("source_files_with_pages", "")).lower()
        if "unknown" in sf or "unknown" in sfwp:
            keep_mask[i] = False
            unknown_dropped += 1

    if unknown_dropped:
        print(
            f"  Dropped {unknown_dropped}/{total} rows with unknown source files"
        )

    # --- Phase 1 & 2: QC + SA ---
    evaluated = 0
    qc_dropped = 0
    sa_dropped = 0

    for i, (_, row) in enumerate(rows):
        if not keep_mask[i]:
            continue

        query = str(row.get("user_input", ""))
        answer = str(row.get("reference", ""))

        # --- QC ---
        qc = evaluate_query_quality(llm, query, i)
        if qc.get("verdict") != "pass":
            keep_mask[i] = False
            qc_dropped += 1
            evaluated += 1
            if evaluated % 10 == 0:
                _print_progress(evaluated, total, t0)
            continue

        # --- SA ---
        positive_text = _fetch_positive_page_content(row, conn)
        if positive_text is None:
            contexts = _parse_reference_contexts(row.get("reference_contexts"))
            positive_text = "\n\n".join(contexts) if contexts else ""

        sa = evaluate_source_answerability(llm, query, answer, positive_text, i)
        if sa.get("verdict") != "pass":
            keep_mask[i] = False
            sa_dropped += 1

        evaluated += 1
        if evaluated % 10 == 0:
            _print_progress(evaluated, total, t0)

    elapsed = time.monotonic() - t0
    filtered_df = df[keep_mask].reset_index(drop=True)
    kept = len(filtered_df)

    stats = {
        "total": total,
        "kept": kept,
        "dropped": total - kept,
        "unknown_source_dropped": unknown_dropped,
        "qc_dropped": qc_dropped,
        "sa_dropped": sa_dropped,
        "elapsed_seconds": round(elapsed, 1),
    }

    print(f"\n  Quality filter complete ({elapsed:.1f}s):")
    print(f"    Total rows:            {total}")
    print(f"    Unknown sources:       -{unknown_dropped}")
    print(f"    Query quality fail:    -{qc_dropped}")
    print(f"    Source answer fail:    -{sa_dropped}")
    print(
        f"    Kept:                  {kept} "
        f"({kept / max(total, 1) * 100:.1f}%)"
    )

    return filtered_df, stats


def _print_progress(evaluated: int, total: int, t0: float) -> None:
    elapsed = time.monotonic() - t0
    rate = evaluated / elapsed if elapsed > 0 else 0
    remaining = (total - evaluated) / rate if rate > 0 else 0
    print(
        f"  Filter progress: {evaluated}/{total} "
        f"({rate:.1f}/s, ETA {remaining:.0f}s)"
    )
