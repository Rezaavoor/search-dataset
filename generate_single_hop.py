#!/usr/bin/env python3
"""Generate single-hop Q&A queries WITHOUT building a RAGAS Knowledge Graph.

Uses pre-computed PDF profiles (from profile_corpus.py) and page content
from SQLite to generate standalone search queries with answers.  Each query
is generated from one randomly-selected PDF file and its best page.

Features:
  - No KG required — uses profiles for entities/topics and pages for context.
  - Multi-endpoint Azure parallelism (up to N concurrent workers).
  - Exponential backoff with jitter on rate-limit / throttle errors.
  - Graceful Ctrl+C shutdown (finishes current queries, saves partial results).
  - Hard negative mining (BM25 + embedding + LLM judge) as post-processing.
  - Direct source tracking — no fuzzy text matching.
  - Resume-safe via --append to add to existing output.

Environment Variables (Azure OpenAI):
    AZURE_OPENAI_API_KEY      + AZURE_OPENAI_ENDPOINT       (primary)
    AZURE_OPENAI_API_KEY_2    + AZURE_OPENAI_ENDPOINT_2     (optional)
    AZURE_OPENAI_API_KEY_3    + AZURE_OPENAI_ENDPOINT_3     (optional)
    AZURE_OPENAI_DEPLOYMENT_NAME  - Chat model deployment (default: gpt-4o-mini)
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME - Embedding deployment
    AZURE_OPENAI_API_VERSION      - API version

Usage:
    # Generate 100 queries from random PDF files
    python generate_single_hop.py --num-queries 100

    # Generate 500 queries, reproducible, no hard negatives
    python generate_single_hop.py --num-queries 500 --seed 42 --skip-hard-negatives

    # Quick test run
    python generate_single_hop.py --num-queries 5 --skip-hard-negatives

    # Dry run — show what would be selected
    python generate_single_hop.py --num-queries 50 --dry-run
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

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(SCRIPT_DIR / ".env")

from modules.config import (
    CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION,
    DEFAULT_PDF_STORE_DB_NAME,
    REFERENTIAL_QUERY_RE,
)
from modules.db import (
    _emb_col,
    _emb_dims_col,
    init_pdf_page_store,
    load_page_embeddings,
    open_pdf_page_store,
)
from modules.hard_negatives import (
    build_bm25_index,
    expand_pages_with_proximity,
    find_bm25_hard_negative_pages,
    find_embedding_hard_negative_pages,
    llm_is_hard_negative,
    reciprocal_rank_fusion,
    _top_indices_desc,
)
from modules.utils import (
    extract_json_object,
    truncate_for_judge,
    utc_now_iso,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("single_hop")

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
# Default personas (legal domain)
# ---------------------------------------------------------------------------
DEFAULT_PERSONAS = [
    {"name": "Legal Researcher", "role_description": "Researches case law, statutes, and legal precedents for litigation support."},
    {"name": "Contract Analyst", "role_description": "Reviews and analyzes contracts, agreements, and commercial terms."},
    {"name": "Compliance Officer", "role_description": "Ensures regulatory compliance and reviews policy documents."},
    {"name": "Paralegal", "role_description": "Assists attorneys with document review, case preparation, and legal research."},
    {"name": "Corporate Counsel", "role_description": "Advises on corporate transactions, governance, and risk management."},
    {"name": "Litigation Associate", "role_description": "Prepares legal briefs, motions, and discovery documents for court proceedings."},
]

QUERY_STYLES = ["Perfect grammar", "Web search like", "Poor grammar"]
QUERY_LENGTHS = ["short", "medium", "long"]

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_event = threading.Event()


def _signal_handler(sig, frame):
    if _shutdown_event.is_set():
        log.warning("Double Ctrl+C — forcing exit")
        sys.exit(1)
    log.warning("Ctrl+C received — finishing current queries and saving partial results...")
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

    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if key and endpoint:
        clients.append(AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_version,
            azure_deployment=deployment,
            temperature=0.4,
            request_timeout=90,
            max_retries=0,
        ))

    for i in range(2, 10):
        key = os.environ.get(f"AZURE_OPENAI_API_KEY_{i}", "").strip()
        endpoint = os.environ.get(f"AZURE_OPENAI_ENDPOINT_{i}", "").strip()
        if key and endpoint:
            clients.append(AzureChatOpenAI(
                azure_endpoint=endpoint,
                api_key=key,
                api_version=api_version,
                azure_deployment=deployment,
                temperature=0.4,
                request_timeout=90,
                max_retries=0,
            ))

    return clients


def _discover_azure_embedding_client(model: str) -> Optional[Any]:
    """Create an AzureOpenAIEmbeddings client for hard negative embedding."""
    from langchain_openai import AzureOpenAIEmbeddings

    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", model)

    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if key and endpoint:
        return AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=key,
            api_version=api_version,
            azure_deployment=deployment,
        )
    return None


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
# Page selection: pick the best page from a file
# ---------------------------------------------------------------------------
def select_best_page(
    conn: sqlite3.Connection,
    rel_path: str,
    *,
    profile_entities: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Pick the best page from a file for query generation.

    Strategy:
    1. Fetch first 10 pages (avoids appendices/indexes)
    2. Filter out pages with < 200 chars
    3. Prefer pages mentioning a profile entity (if available)
    4. Among candidates, pick the longest page
    """
    rows = conn.execute(
        """
        SELECT page_number, doc_content, content_chars
        FROM pdf_page_store
        WHERE rel_path = ? AND content_chars >= 200
        ORDER BY page_number ASC
        LIMIT 10
        """,
        (rel_path,),
    ).fetchall()

    if not rows:
        return None

    candidates = [
        {"page_number": int(r[0]), "content": str(r[1] or ""), "chars": int(r[2])}
        for r in rows
    ]

    # Prefer pages mentioning a profile entity
    if profile_entities:
        entity_pages = []
        for c in candidates:
            content_lower = c["content"].lower()
            for ent in profile_entities:
                if ent.lower() in content_lower:
                    entity_pages.append(c)
                    break
        if entity_pages:
            candidates = entity_pages

    # Pick the longest page among candidates
    best = max(candidates, key=lambda c: c["chars"])
    return best


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------
def load_profile(conn: sqlite3.Connection, rel_path: str) -> Optional[Dict[str, Any]]:
    """Load the PDF profile for a file from SQLite."""
    row = conn.execute(
        "SELECT pdf_profile_json FROM pdf_page_store WHERE rel_path = ? AND pdf_profile_json IS NOT NULL LIMIT 1",
        (rel_path,),
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        prof = json.loads(row[0])
        return prof if isinstance(prof, dict) else None
    except Exception:
        return None


def _profile_entities(profile: Optional[Dict[str, Any]]) -> List[str]:
    """Extract key_entities from a profile."""
    if not profile:
        return []
    llm_prof = profile.get("llm_profile") or {}
    entities = llm_prof.get("key_entities") or []
    return [str(e) for e in entities if isinstance(e, str) and e.strip()]


def _profile_topics(profile: Optional[Dict[str, Any]]) -> List[str]:
    """Extract topics from a profile."""
    if not profile:
        return []
    llm_prof = profile.get("llm_profile") or {}
    topics = llm_prof.get("topics") or []
    return [str(t) for t in topics if isinstance(t, str) and t.strip()]


def _build_llm_context(profile: Optional[Dict[str, Any]]) -> str:
    """Build LLM context string from a profile for query generation guidance."""
    base = (
        "You are generating queries for a retrieval/search/RAG system.\n"
        "The user is searching across a large corpus of ~7,000+ PDF documents.\n\n"
        "CRITICAL: Each query must be a standalone, first-turn query.\n"
        "- Do NOT assume a document/case has already been selected.\n"
        '- Avoid deictic references like "this case", "this document", "the above", '
        '"herein", etc.\n'
        "- Include concrete identifiers from the provided context to disambiguate.\n"
        "- Do NOT mention filenames or page numbers.\n"
    )

    if not profile:
        return base

    llm_prof = profile.get("llm_profile") or {}
    doc_type = llm_prof.get("doc_type", "")
    title = llm_prof.get("title_guess", "")
    summary = llm_prof.get("summary", "")
    intents = llm_prof.get("likely_user_intents") or []

    doc_hint_parts = []
    if doc_type:
        doc_hint_parts.append(f"Document type: {doc_type}")
    if title:
        doc_hint_parts.append(f"Title: {title}")
    if summary:
        doc_hint_parts.append(f"Summary: {summary[:300]}")
    if intents:
        doc_hint_parts.append(f"Likely user intents: {', '.join(str(i) for i in intents[:4])}")

    if doc_hint_parts:
        base += "\nDocument context:\n" + "\n".join(f"  - {p}" for p in doc_hint_parts) + "\n"

    return base


# ---------------------------------------------------------------------------
# Query generation prompt
# ---------------------------------------------------------------------------
def _generate_query_prompt(
    *,
    persona_name: str,
    persona_role: str,
    term: str,
    context: str,
    query_style: str,
    query_length: str,
    llm_context: str,
) -> List[Any]:
    """Build the system + user messages for single-hop query generation."""
    system_msg = CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION

    user_msg = (
        f"persona:\n"
        f"  name: {persona_name}\n"
        f"  role_description: {persona_role}\n\n"
        f"term: {term}\n\n"
        f"query_style: {query_style}\n"
        f"query_length: {query_length}\n\n"
        f"context:\n{context}\n\n"
    )
    if llm_context:
        user_msg += f"llm_context:\n{llm_context}\n\n"

    user_msg += (
        "Return ONLY valid JSON in this exact schema:\n"
        '{"query": "<the generated query>", "answer": "<the generated answer>"}'
    )

    return [SystemMessage(content=system_msg), HumanMessage(content=user_msg)]


# ---------------------------------------------------------------------------
# Single query generation with retry
# ---------------------------------------------------------------------------
def _generate_one_query(
    llm,
    *,
    rel_path: str,
    page_content: str,
    page_number: int,
    filename: str,
    profile: Optional[Dict[str, Any]],
    rng: random.Random,
    worker_id: int = 0,
) -> Optional[Dict[str, Any]]:
    """Generate a single Q&A pair for one file+page. Returns a result dict or None."""
    entities = _profile_entities(profile)
    topics = _profile_topics(profile)

    # Pick a term (entity or topic) to focus the query on
    term_pool = entities + topics
    if not term_pool:
        # No profile data — use a generic term
        term_pool = ["this document's subject matter"]
    term = rng.choice(term_pool)

    persona = rng.choice(DEFAULT_PERSONAS)
    style = rng.choice(QUERY_STYLES)
    length = rng.choice(QUERY_LENGTHS)
    llm_context = _build_llm_context(profile)

    messages = _generate_query_prompt(
        persona_name=persona["name"],
        persona_role=persona["role_description"],
        term=term,
        context=page_content,
        query_style=style,
        query_length=length,
        llm_context=llm_context,
    )

    for attempt in range(MAX_RETRIES):
        if _shutdown_event.is_set():
            return None
        try:
            resp = llm.invoke(messages)
            content = getattr(resp, "content", None)
            content = content if isinstance(content, str) else str(resp)
            data = extract_json_object(content)

            if not isinstance(data, dict) or "query" not in data or "answer" not in data:
                log.warning(
                    "Worker %d: bad LLM response for %s (attempt %d), retrying",
                    worker_id, rel_path[:40], attempt + 1,
                )
                continue

            source_with_page = f"{filename} (page {page_number})"
            return {
                "user_input": str(data["query"]).strip(),
                "reference": str(data["answer"]).strip(),
                "reference_contexts": json.dumps([page_content]),
                "source_file": filename,
                "source_file_with_page": source_with_page,
                "source_files_with_pages": json.dumps([source_with_page]),
                "page_number": page_number,
                "rel_path": rel_path,
                "entity_term": term,
                "persona_name": persona["name"],
                "query_style": style,
                "query_length": length,
                "synthesizer_name": "single_hop_direct",
            }

        except Exception as exc:
            if _is_retriable(exc) and attempt < MAX_RETRIES - 1:
                delay = _backoff_delay(attempt)
                log.warning(
                    "Worker %d: retriable error (attempt %d/%d), retrying in %.1fs: %s",
                    worker_id, attempt + 1, MAX_RETRIES, delay, str(exc)[:120],
                )
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not _shutdown_event.is_set():
                    time.sleep(min(1.0, deadline - time.monotonic()))
                continue
            else:
                log.warning(
                    "Worker %d: query gen failed for %s: %s",
                    worker_id, rel_path[:40], str(exc)[:120],
                )
                return None

    return None


# ---------------------------------------------------------------------------
# Worker function (runs in thread)
# ---------------------------------------------------------------------------
def _worker_generate(
    *,
    worker_id: int,
    llm,
    tasks: List[Dict[str, Any]],
    db_path: Path,
    results: List[Optional[Dict[str, Any]]],
    result_indices: List[int],
    stats: Dict[str, int],
    stats_lock: threading.Lock,
    seed: Optional[int],
) -> int:
    """Process assigned tasks: select page, generate query."""
    conn = open_pdf_page_store(db_path)
    rng = random.Random(seed + worker_id if seed is not None else None)
    generated = 0

    for task_idx, task in zip(result_indices, tasks):
        if _shutdown_event.is_set():
            break

        rel_path = task["rel_path"]
        filename = task["filename"]
        profile = load_profile(conn, rel_path)
        entities = _profile_entities(profile)

        page = select_best_page(conn, rel_path, profile_entities=entities)
        if not page:
            log.debug("Worker %d: no suitable page for %s", worker_id, rel_path[:40])
            with stats_lock:
                stats["skipped_no_page"] += 1
                stats["processed"] += 1
            continue

        result = _generate_one_query(
            llm,
            rel_path=rel_path,
            page_content=page["content"],
            page_number=page["page_number"],
            filename=filename,
            profile=profile,
            rng=rng,
            worker_id=worker_id,
        )

        results[task_idx] = result

        with stats_lock:
            stats["processed"] += 1
            if result:
                stats["generated"] += 1
                generated += 1

    conn.close()
    return generated


# ---------------------------------------------------------------------------
# Hard negative mining (adapted from modules/hard_negatives.py for no-KG)
# ---------------------------------------------------------------------------
def _load_all_pages_from_store(
    conn: sqlite3.Connection,
    embedding_model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load all pages from SQLite as hard-negative candidates (replaces _extract_pages_from_kg)."""
    log.info("  Loading all pages from SQLite for hard negative candidates...")

    emb_select = ""
    emb_col = None
    dims_col = None
    if embedding_model_name:
        emb_col = _emb_col(embedding_model_name)
        dims_col = _emb_dims_col(embedding_model_name)
        # Check if columns exist
        try:
            existing = {
                str(r[1])
                for r in conn.execute("PRAGMA table_info(pdf_page_store)").fetchall()
            }
            if emb_col in existing and dims_col in existing:
                emb_select = f", {emb_col}, {dims_col}"
            else:
                emb_col = None
                dims_col = None
        except Exception:
            emb_col = None
            dims_col = None

    rows = conn.execute(
        f"""
        SELECT rel_path, page_number, filename, doc_content{emb_select}
        FROM pdf_page_store
        WHERE content_chars >= 100
        ORDER BY rel_path, page_number
        """
    ).fetchall()

    pages = []
    for row in rows:
        rel_path = str(row[0])
        page_number = int(row[1])
        filename = str(row[2])
        content = str(row[3] or "")

        embedding = None
        if emb_col and dims_col and len(row) > 5:
            blob = row[4]
            dims = row[5]
            if isinstance(blob, bytes) and isinstance(dims, int) and dims > 0:
                try:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if len(vec) == dims:
                        embedding = vec.tolist()
                except Exception:
                    pass

        pages.append({
            "file": filename,
            "page": page_number,
            "source": rel_path,
            "page_content": content,
            "embedding": embedding,
        })

    return pages


def mine_hard_negatives_no_kg(
    results_df: pd.DataFrame,
    conn: sqlite3.Connection,
    embedding_model: Any,
    judge_llm: Any,
    *,
    embedding_model_name: Optional[str] = None,
    num_bm25_negatives: int = 5,
    num_embedding_negatives: int = 5,
    max_judge_calls_per_query: int = 12,
) -> List[List[str]]:
    """Mine hard negatives for generated queries using SQLite pages directly (no KG)."""
    log.info("Mining hard negatives...")

    pages = _load_all_pages_from_store(conn, embedding_model_name)
    log.info("  Candidate pages: %d", len(pages))
    pages_with_embeddings = sum(1 for p in pages if p.get("embedding") is not None)
    log.info("  Pages with embeddings: %d/%d", pages_with_embeddings, len(pages))

    # Build BM25 index
    bm25 = None
    if num_bm25_negatives > 0:
        try:
            bm25, _ = build_bm25_index(pages)
            log.info("  Built BM25 index")
        except ImportError as e:
            log.warning("  %s — skipping BM25 hard negatives", e)

    # Pre-compute normalized embeddings for positive pages
    page_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {
        (p.get("file"), int(p.get("page"))): p
        for p in pages
        if p.get("file") and p.get("page")
    }
    page_emb_norm_by_key: Dict[Tuple[str, int], Optional[np.ndarray]] = {}
    for k, p in page_by_key.items():
        emb = p.get("embedding")
        if emb is None:
            page_emb_norm_by_key[k] = None
            continue
        try:
            v = np.asarray(emb, dtype=np.float32)
        except Exception:
            page_emb_norm_by_key[k] = None
            continue
        if v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)):
            page_emb_norm_by_key[k] = None
            continue
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n == 0:
            page_emb_norm_by_key[k] = None
            continue
        page_emb_norm_by_key[k] = v / n

    near_duplicate_cosine_threshold = 0.92

    # Pre-compute query embeddings
    query_embeddings: Dict[int, Optional[List[float]]] = {}
    if num_embedding_negatives > 0 and pages_with_embeddings > 0 and embedding_model is not None:
        query_texts = results_df["user_input"].tolist()
        try:
            batch_embs = embedding_model.embed_documents(query_texts)
            if isinstance(batch_embs, list) and len(batch_embs) == len(query_texts):
                for i, emb in enumerate(batch_embs):
                    query_embeddings[i] = emb if isinstance(emb, list) else None
                log.info("  Pre-embedded %d queries (batched)", len(query_texts))
        except Exception as e:
            log.warning("  Batch query embedding failed: %s", str(e)[:120])

    hard_negatives_list: List[List[str]] = []

    for idx, row in results_df.iterrows():
        if _shutdown_event.is_set():
            hard_negatives_list.append([])
            continue

        query = row["user_input"]
        source_file = row.get("source_file", "")
        page_number = row.get("page_number")

        desired_count = max(num_bm25_negatives, num_embedding_negatives)
        if desired_count <= 0:
            hard_negatives_list.append([])
            continue

        # Positive set: just the one source file + page we know about
        pos_pages = set()
        if source_file and page_number:
            pos_pages.add((source_file, int(page_number)))

        if not pos_pages:
            hard_negatives_list.append([])
            continue

        exclude_pages = expand_pages_with_proximity(pos_pages)

        pos_embs = [
            page_emb_norm_by_key.get(k)
            for k in pos_pages
            if page_emb_norm_by_key.get(k) is not None
        ]
        pos_embs = [v for v in pos_embs if v is not None]

        # BM25 candidates
        bm25_negs: List[Tuple[str, int]] = []
        if bm25 is not None and num_bm25_negatives > 0:
            bm25_negs = find_bm25_hard_negative_pages(
                query, pages, bm25,
                exclude_pages=exclude_pages,
                top_k=max(50, num_bm25_negatives * 10),
            )

        # Embedding candidates
        emb_negs: List[Tuple[str, int]] = []
        if num_embedding_negatives > 0 and pages_with_embeddings > 0:
            q_emb = query_embeddings.get(idx)
            if q_emb is None and embedding_model is not None:
                try:
                    q_emb = embedding_model.embed_query(query)
                except Exception:
                    pass
            if q_emb is not None:
                cand_idxs: Optional[List[int]] = None
                if bm25 is not None:
                    scores = bm25.get_scores(query.lower().split())
                    cand_idxs = _top_indices_desc(scores, top_n=300)
                emb_negs = find_embedding_hard_negative_pages(
                    q_emb, pages,
                    candidate_indices=cand_idxs,
                    exclude_pages=exclude_pages,
                    top_k=max(50, num_embedding_negatives * 10),
                )

        # RRF merge
        if num_bm25_negatives > 0 and num_embedding_negatives > 0:
            if bm25 is not None and bm25_negs and emb_negs:
                base_keys = reciprocal_rank_fusion(bm25_negs, emb_negs)
            else:
                base_keys = emb_negs or bm25_negs
        elif num_embedding_negatives > 0:
            base_keys = emb_negs
        else:
            base_keys = bm25_negs

        # Tiered filtering with LLM judge
        tier1_keys: List[Tuple[str, int]] = []
        tier2_keys: List[Tuple[str, int]] = []
        file_neg_count: Dict[str, int] = {}
        judged = 0
        for key in base_keys:
            f, p = key
            if key in exclude_pages:
                continue
            page_rec = page_by_key.get(key)
            if not page_rec:
                continue

            cand_text = str(page_rec.get("page_content") or "")

            # Near-duplicate check
            cand_v = page_emb_norm_by_key.get(key)
            if cand_v is not None and pos_embs:
                max_sim = max(float(np.dot(cand_v, pv)) for pv in pos_embs)
                if max_sim >= near_duplicate_cosine_threshold:
                    continue

            if file_neg_count.get(f, 0) >= 2:
                continue

            verdict = llm_is_hard_negative(judge_llm, question=query, passage=cand_text)
            judged += 1
            if verdict == 1:
                file_neg_count[f] = file_neg_count.get(f, 0) + 1
                tier1_keys.append(key)
            elif verdict == 2:
                file_neg_count[f] = file_neg_count.get(f, 0) + 1
                tier2_keys.append(key)

            if len(tier1_keys) + len(tier2_keys) >= desired_count * 2:
                break
            if judged >= max_judge_calls_per_query:
                break

        # Select tier 1 first, fill remaining from tier 2
        filtered_keys = tier1_keys[:desired_count]
        remaining = desired_count - len(filtered_keys)
        if remaining > 0:
            filtered_keys.extend(tier2_keys[:remaining])

        hard_negatives_list.append([f"{f} (page {p})" for (f, p) in filtered_keys])

        if (idx + 1) % 10 == 0:
            total_so_far = sum(len(x) for x in hard_negatives_list)
            log.info("  Hard neg progress: %d/%d queries, %d negatives mined",
                     idx + 1, len(results_df), total_so_far)

    total_negs = sum(len(x) for x in hard_negatives_list)
    log.info("  Mined %d hard negative(s) for %d queries", total_negs, len(results_df))
    return hard_negatives_list


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
OUTPUT_DIR = SCRIPT_DIR / "output"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate single-hop Q&A queries from PDF files without building a KG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-queries", type=int, required=True,
        help="Number of queries to generate (each picks a random PDF file)",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="SQLite DB path (default: processed/pdf_page_store.sqlite)",
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(SCRIPT_DIR / "search-dataset"),
        help="Corpus root directory",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Azure chat model deployment (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="text-embedding-3-large",
        help="Embedding model for hard neg mining (default: text-embedding-3-large)",
    )
    parser.add_argument(
        "--output", type=str, default="single_hop_dataset",
        help="Output base name (saved to output/ folder, default: single_hop_dataset)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--personas-path", type=str, default=None,
        help="Path to JSON file with custom personas [{name, role_description}, ...]",
    )
    parser.add_argument(
        "--num-bm25-negatives", type=int, default=5,
        help="BM25 hard negatives per query (default: 5)",
    )
    parser.add_argument(
        "--num-embedding-negatives", type=int, default=5,
        help="Embedding hard negatives per query (default: 5)",
    )
    parser.add_argument(
        "--skip-hard-negatives", action="store_true",
        help="Skip the hard negative mining step entirely",
    )
    parser.add_argument(
        "--filter", action=argparse.BooleanOptionalAction, default=True,
        help="Run vision quality filter after generation (default: enabled)",
    )
    parser.add_argument(
        "--vision-model", type=str, default="gemini-2.5-flash",
        help="Gemini model for vision filtering (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--leya-env",
        type=str,
        default=str(Path.home() / "Documents" / "GitHub" / "leya" / ".local.env"),
        help="Path to leya .local.env containing GCP credentials for vision filter",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show selected files without generating queries",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: auto, one per Azure endpoint)",
    )
    args = parser.parse_args()

    if args.num_queries <= 0:
        log.error("--num-queries must be a positive integer")
        return 1

    # --- Resolve paths ---
    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (SCRIPT_DIR / "processed" / DEFAULT_PDF_STORE_DB_NAME).resolve()

    base_input_dir = Path(args.input_dir).expanduser().resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Single-Hop Query Generator (No KG)")
    log.info("=" * 60)
    log.info("  Queries:    %d", args.num_queries)
    log.info("  Model:      %s", args.model)
    log.info("  DB:         %s", db_path)
    log.info("  Seed:       %s", args.seed if args.seed is not None else "random")

    # --- Load custom personas if provided ---
    global DEFAULT_PERSONAS
    if args.personas_path:
        pp = Path(args.personas_path).expanduser().resolve()
        if pp.exists():
            with open(pp, "r") as f:
                custom = json.load(f)
            if isinstance(custom, list) and len(custom) >= 1:
                DEFAULT_PERSONAS = custom
                log.info("  Personas:   %d loaded from %s", len(DEFAULT_PERSONAS), pp.name)
        else:
            log.warning("  Personas file not found: %s", pp)

    # --- Open DB and find eligible PDF files ---
    conn = open_pdf_page_store(db_path)
    init_pdf_page_store(conn)

    rows = conn.execute(
        """
        SELECT DISTINCT rel_path, filename
        FROM pdf_page_store
        WHERE file_type = 'pdf'
          AND pdf_profile_json IS NOT NULL
          AND content_chars >= 200
        ORDER BY rel_path
        """
    ).fetchall()

    eligible_files = [{"rel_path": r[0], "filename": r[1]} for r in rows]
    conn.close()

    log.info("  Eligible PDF files (with profiles, >=200 chars): %d", len(eligible_files))

    if not eligible_files:
        log.error("No eligible PDF files found. Run profile_corpus.py first.")
        return 1

    # --- Random sampling ---
    rng = random.Random(args.seed)
    n = args.num_queries

    if n <= len(eligible_files):
        selected = rng.sample(eligible_files, n)
    else:
        selected = rng.choices(eligible_files, k=n)
        log.info("  Note: --num-queries (%d) > eligible files (%d), some files will repeat",
                 n, len(eligible_files))

    log.info("  Selected %d files for query generation", len(selected))

    if args.dry_run:
        log.info("DRY RUN — selected files:")
        for i, f in enumerate(selected[:20]):
            log.info("  %3d. %s", i + 1, f["rel_path"])
        if len(selected) > 20:
            log.info("  ... and %d more", len(selected) - 20)
        return 0

    # --- Create LLM clients ---
    clients = _discover_azure_chat_clients(args.model)
    if not clients:
        log.error("No Azure OpenAI credentials found. Set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT.")
        return 1

    num_workers = args.workers if args.workers else len(clients)
    num_workers = min(num_workers, len(clients), len(selected))

    for i, c in enumerate(clients[:num_workers]):
        ep = getattr(c, "azure_endpoint", "?")
        region = ep.split("//")[-1].split(".")[0] if "//" in ep else ep[:30]
        log.info("  Azure LLM client %d: %s", i + 1, region)

    log.info("  Workers: %d", num_workers)

    # --- Distribute tasks across workers ---
    results: List[Optional[Dict[str, Any]]] = [None] * len(selected)
    worker_tasks: List[List[Dict[str, Any]]] = [[] for _ in range(num_workers)]
    worker_indices: List[List[int]] = [[] for _ in range(num_workers)]

    for idx, task in enumerate(selected):
        w = idx % num_workers
        worker_tasks[w].append(task)
        worker_indices[w].append(idx)

    # --- Shared stats ---
    stats: Dict[str, int] = {"processed": 0, "generated": 0, "skipped_no_page": 0}
    stats_lock = threading.Lock()

    # --- Progress reporter ---
    t0 = time.monotonic()

    def _progress_reporter():
        while not _shutdown_event.is_set():
            _shutdown_event.wait(15)
            with stats_lock:
                processed = stats["processed"]
                generated = stats["generated"]
            elapsed = time.monotonic() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(selected) - processed) / rate if rate > 0 else 0
            pct = (processed / len(selected)) * 100 if selected else 0
            log.info(
                "Progress: %d/%d (%.1f%%)  generated=%d  rate=%.1f/s  elapsed=%s  ETA=%s",
                processed, len(selected), pct, generated,
                rate, _fmt_duration(elapsed), _fmt_duration(remaining),
            )
            if processed >= len(selected):
                break

    progress_thread = threading.Thread(target=_progress_reporter, daemon=True)
    progress_thread.start()

    # --- Launch workers ---
    log.info("Starting query generation (Ctrl+C to stop gracefully)...")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            future = executor.submit(
                _worker_generate,
                worker_id=i + 1,
                llm=clients[i],
                tasks=worker_tasks[i],
                db_path=db_path,
                results=results,
                result_indices=worker_indices[i],
                stats=stats,
                stats_lock=stats_lock,
                seed=args.seed,
            )
            futures[future] = i + 1

        total_generated = 0
        for future in as_completed(futures):
            wid = futures[future]
            try:
                count = future.result()
                total_generated += count
                log.info("Worker %d finished: %d queries generated", wid, count)
            except Exception as exc:
                log.error("Worker %d failed: %s", wid, exc)

    _shutdown_event.set()
    progress_thread.join(timeout=5)

    elapsed_gen = time.monotonic() - t0

    # --- Collect results ---
    valid_results = [r for r in results if r is not None]
    log.info("Query generation complete in %s", _fmt_duration(elapsed_gen))
    log.info("  Generated: %d / %d", len(valid_results), len(selected))
    log.info("  Skipped (no page): %d", stats.get("skipped_no_page", 0))

    if not valid_results:
        log.error("No queries generated. Check your LLM configuration.")
        return 1

    df = pd.DataFrame(valid_results)

    # --- Warn on referential queries ---
    ref_count = 0
    for q in df["user_input"]:
        if REFERENTIAL_QUERY_RE.search(str(q)):
            ref_count += 1
    if ref_count > 0:
        log.warning(
            "  %d/%d queries look referential (e.g. 'this case', 'the above')",
            ref_count, len(df),
        )

    # --- Hard negative mining ---
    if not args.skip_hard_negatives and len(df) > 0:
        log.info("=" * 60)
        log.info("Hard Negative Mining")
        log.info("=" * 60)

        _shutdown_event.clear()
        signal.signal(signal.SIGINT, _signal_handler)

        hn_conn = open_pdf_page_store(db_path)

        embedding_client = _discover_azure_embedding_client(args.embedding_model)
        judge_llm = clients[0] if clients else None

        hard_negs = mine_hard_negatives_no_kg(
            df,
            hn_conn,
            embedding_model=embedding_client,
            judge_llm=judge_llm,
            embedding_model_name=args.embedding_model,
            num_bm25_negatives=args.num_bm25_negatives,
            num_embedding_negatives=args.num_embedding_negatives,
        )

        hn_conn.close()

        df["hard_negatives"] = [json.dumps(negs, ensure_ascii=False) for negs in hard_negs]
        neg_count = sum(len(x) for x in hard_negs)
        log.info("  Added %d hard negatives (%.1f avg per query)",
                 neg_count, neg_count / max(len(df), 1))

    # --- Save output ---
    log.info("=" * 60)
    log.info("Saving results")
    log.info("=" * 60)

    output_base = str(OUTPUT_DIR / args.output)

    csv_path = f"{output_base}.csv"
    df.to_csv(csv_path, index=False)
    log.info("  Saved: %s", csv_path)

    json_path = f"{output_base}.json"
    df.to_json(json_path, orient="records", indent=2)
    log.info("  Saved: %s", json_path)

    # --- Vision quality filter (optional) ---
    if args.filter:
        log.info("=" * 60)
        log.info("Running vision quality filter")
        log.info("=" * 60)
        from vision_validate_dataset import vision_filter_dataset
        leya_env = Path(args.leya_env).expanduser().resolve()
        db_path = SCRIPT_DIR / "processed" / "pdf_page_store.sqlite"
        pdf_root = Path(args.input_dir).expanduser().resolve()
        vision_checkpoint = SCRIPT_DIR / "processed" / "progress" / f"{args.output}__vision_filter_progress.jsonl"
        strict_df, relaxed_df, filter_stats = vision_filter_dataset(
            df,
            leya_env_path=leya_env,
            db_path=db_path,
            pdf_root=pdf_root,
            model=args.vision_model,
            concurrency=5,
            checkpoint_path=vision_checkpoint,
        )
        strict_df.to_csv(f"{output_base}_filtered.csv", index=False)
        strict_df.to_json(f"{output_base}_filtered.json", orient="records", indent=2)
        relaxed_df.to_csv(f"{output_base}_filtered_relaxed.csv", index=False)
        relaxed_df.to_json(f"{output_base}_filtered_relaxed.json", orient="records", indent=2)
        log.info("  Strict filtered:  %d/%d → %s_filtered.csv",
                 filter_stats["strict_kept"], filter_stats["total"], output_base)
        log.info("  Relaxed filtered: %d/%d → %s_filtered_relaxed.csv",
                 filter_stats["relaxed_kept"], filter_stats["total"], output_base)

    # --- Preview ---
    log.info("\nSample preview:")
    preview_cols = ["user_input", "source_file_with_page"]
    if "hard_negatives" in df.columns:
        preview_cols.append("hard_negatives")
    print(df[preview_cols].head(5).to_string(index=False))

    log.info("\n" + "=" * 60)
    log.info("COMPLETE")
    log.info("  Total queries: %d", len(df))
    log.info("  Output: %s.{csv,json}", output_base)
    log.info("  Elapsed: %s", _fmt_duration(time.monotonic() - t0))
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
