"""Single-hop query generation without a Knowledge Graph.

Provides reusable functions for generating single-hop Q&A pairs from
individual PDF pages + profiles stored in the SQLite page store.

Used by:
  - generate_synthetic_dataset.py (integrated pipeline, world mode)
  - generate_single_hop.py (standalone parallel script)
"""

import json
import logging
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .config import CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION
from .db import open_pdf_page_store
from .utils import extract_json_object

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default personas (legal domain)
# ---------------------------------------------------------------------------
DEFAULT_PERSONAS = [
    {
        "name": "Legal Researcher",
        "role_description": (
            "Researches case law, statutes, and legal precedents "
            "for litigation support."
        ),
    },
    {
        "name": "Contract Analyst",
        "role_description": (
            "Reviews and analyzes contracts, agreements, "
            "and commercial terms."
        ),
    },
    {
        "name": "Compliance Officer",
        "role_description": (
            "Ensures regulatory compliance and reviews policy documents."
        ),
    },
    {
        "name": "Paralegal",
        "role_description": (
            "Assists attorneys with document review, case preparation, "
            "and legal research."
        ),
    },
    {
        "name": "Corporate Counsel",
        "role_description": (
            "Advises on corporate transactions, governance, "
            "and risk management."
        ),
    },
    {
        "name": "Litigation Associate",
        "role_description": (
            "Prepares legal briefs, motions, and discovery documents "
            "for court proceedings."
        ),
    },
]

QUERY_STYLES = ["Perfect grammar", "Web search like", "Poor grammar"]
QUERY_LENGTHS = ["short", "medium", "long"]


# ---------------------------------------------------------------------------
# Page selection
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

    best = max(candidates, key=lambda c: c["chars"])
    return best


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
def load_profile(
    conn: sqlite3.Connection, rel_path: str,
) -> Optional[Dict[str, Any]]:
    """Load the PDF profile for a file from SQLite."""
    row = conn.execute(
        "SELECT pdf_profile_json FROM pdf_page_store "
        "WHERE rel_path = ? AND pdf_profile_json IS NOT NULL LIMIT 1",
        (rel_path,),
    ).fetchone()
    if not row or not row[0]:
        return None
    try:
        prof = json.loads(row[0])
        return prof if isinstance(prof, dict) else None
    except Exception:
        return None


def get_profile_entities(profile: Optional[Dict[str, Any]]) -> List[str]:
    """Extract key_entities from a profile."""
    if not profile:
        return []
    llm_prof = profile.get("llm_profile") or {}
    entities = llm_prof.get("key_entities") or []
    return [str(e) for e in entities if isinstance(e, str) and e.strip()]


def get_profile_topics(profile: Optional[Dict[str, Any]]) -> List[str]:
    """Extract topics from a profile."""
    if not profile:
        return []
    llm_prof = profile.get("llm_profile") or {}
    topics = llm_prof.get("topics") or []
    return [str(t) for t in topics if isinstance(t, str) and t.strip()]


def build_single_hop_llm_context(
    profile: Optional[Dict[str, Any]],
    *,
    corpus_size_hint: int = 7000,
) -> str:
    """Build LLM context string from a profile for single-hop query generation."""
    base = (
        "You are generating queries for a retrieval/search/RAG system.\n"
        f"The user is searching across a large corpus of "
        f"~{corpus_size_hint}+ PDF documents.\n\n"
        "CRITICAL: Each query must be a standalone, first-turn query.\n"
        "- Do NOT assume a document/case has already been selected.\n"
        '- Avoid deictic references like "this case", "this document", '
        '"the above", "herein", etc.\n'
        "- Include concrete identifiers from the provided context "
        "to disambiguate.\n"
        "- Do NOT mention filenames or page numbers.\n"
    )

    if not profile:
        return base

    llm_prof = profile.get("llm_profile") or {}
    doc_type = llm_prof.get("doc_type", "")
    title = llm_prof.get("title_guess", "")
    summary = llm_prof.get("summary", "")
    intents = llm_prof.get("likely_user_intents") or []

    parts: List[str] = []
    if doc_type:
        parts.append(f"Document type: {doc_type}")
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary[:300]}")
    if intents:
        parts.append(
            "Likely user intents: "
            + ", ".join(str(i) for i in intents[:4])
        )

    if parts:
        base += (
            "\nDocument context:\n"
            + "\n".join(f"  - {p}" for p in parts)
            + "\n"
        )

    return base


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_query_prompt(
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
        '{"query": "<the generated query>", '
        '"answer": "<the generated answer>"}'
    )

    return [
        SystemMessage(content=system_msg),
        HumanMessage(content=user_msg),
    ]


# ---------------------------------------------------------------------------
# Single query generation
# ---------------------------------------------------------------------------
def generate_one_query(
    llm,
    *,
    rel_path: str,
    page_content: str,
    page_number: int,
    filename: str,
    profile: Optional[Dict[str, Any]],
    rng: random.Random,
    personas: Optional[List[Dict[str, str]]] = None,
    corpus_size_hint: int = 7000,
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Generate a single Q&A pair for one file+page.

    Args:
        llm: A LangChain chat model (raw, NOT RAGAS-wrapped).

    Returns:
        A result dict with keys matching the combined output schema, or None.
    """
    entities = get_profile_entities(profile)
    topics = get_profile_topics(profile)
    term_pool = entities + topics
    if not term_pool:
        term_pool = ["this document's subject matter"]
    term = rng.choice(term_pool)

    persona_list = personas or DEFAULT_PERSONAS
    persona = rng.choice(persona_list)
    style = rng.choice(QUERY_STYLES)
    length = rng.choice(QUERY_LENGTHS)
    llm_context = build_single_hop_llm_context(
        profile, corpus_size_hint=corpus_size_hint,
    )

    messages = build_query_prompt(
        persona_name=persona["name"],
        persona_role=persona["role_description"],
        term=term,
        context=page_content,
        query_style=style,
        query_length=length,
        llm_context=llm_context,
    )

    for attempt in range(max_retries):
        try:
            resp = llm.invoke(messages)
            content = getattr(resp, "content", None)
            content = content if isinstance(content, str) else str(resp)
            data = extract_json_object(content)

            if (
                not isinstance(data, dict)
                or "query" not in data
                or "answer" not in data
            ):
                log.debug(
                    "Bad LLM response for %s (attempt %d)",
                    rel_path[:40], attempt + 1,
                )
                continue

            source_with_page = f"{filename} (page {page_number})"
            return {
                "user_input": str(data["query"]).strip(),
                "reference": str(data["answer"]).strip(),
                "reference_contexts": json.dumps([page_content]),
                "source_file": filename,
                "source_file_with_page": source_with_page,
                "source_files": json.dumps([filename]),
                "source_files_with_pages": json.dumps([source_with_page]),
                "page_numbers": json.dumps([page_number]),
                "source_files_readable": filename,
                "source_files_with_pages_readable": source_with_page,
                "rel_path": rel_path,
                "entity_term": term,
                "persona_name": persona["name"],
                "query_style": style,
                "query_length": length,
                "synthesizer_name": "single_hop_direct",
            }

        except Exception as exc:
            log.warning(
                "Single-hop gen failed for %s (attempt %d): %s",
                rel_path[:40], attempt + 1, str(exc)[:120],
            )
            continue

    return None


# ---------------------------------------------------------------------------
# Worker for parallel generation
# ---------------------------------------------------------------------------
def _worker_generate(
    *,
    worker_id: int,
    llm,
    tasks: List[Dict[str, Any]],
    db_path: Path,
    result_slots: List[Optional[Dict[str, Any]]],
    result_indices: List[int],
    stats: Dict[str, int],
    stats_lock: threading.Lock,
    seed: Optional[int],
    personas: Optional[List[Dict[str, str]]],
    corpus_size_hint: int,
) -> int:
    """Process assigned tasks in a worker thread."""
    conn = open_pdf_page_store(db_path)
    rng = random.Random(seed + worker_id if seed is not None else None)
    generated = 0

    for task_idx, task in zip(result_indices, tasks):
        rel_path = task["rel_path"]
        filename = task["filename"]

        profile = load_profile(conn, rel_path)
        entities = get_profile_entities(profile)
        page = select_best_page(conn, rel_path, profile_entities=entities)

        if not page:
            with stats_lock:
                stats["skipped"] += 1
                stats["processed"] += 1
            continue

        result = generate_one_query(
            llm,
            rel_path=rel_path,
            page_content=page["content"],
            page_number=page["page_number"],
            filename=filename,
            profile=profile,
            rng=rng,
            personas=personas,
            corpus_size_hint=corpus_size_hint,
        )

        result_slots[task_idx] = result
        with stats_lock:
            stats["processed"] += 1
            if result:
                stats["generated"] += 1
                generated += 1

    conn.close()
    return generated


# ---------------------------------------------------------------------------
# Batch generation (parallel when multiple LLM clients provided)
# ---------------------------------------------------------------------------
def generate_single_hop_queries(
    llm,
    conn: sqlite3.Connection,
    *,
    num_queries: int,
    seed: Optional[int] = None,
    personas: Optional[List[Dict[str, str]]] = None,
    corpus_size_hint: int = 7000,
    extra_llms: Optional[List[Any]] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Generate single-hop queries from the full corpus (no KG required).

    Selects random PDFs from the SQLite store, picks best pages, and
    generates Q&A pairs using the LLM.

    Args:
        llm: A LangChain chat model (raw, NOT RAGAS-wrapped).
        conn: SQLite connection to the PDF page store (used for file
            discovery; each worker opens its own connection).
        num_queries: Number of queries to generate.
        seed: Random seed for reproducibility.
        personas: Custom persona dicts [{name, role_description}, ...].
        corpus_size_hint: Approximate corpus size for prompt context.
        extra_llms: Additional LLM clients for parallel generation.
            Each client runs in its own thread.  If None or empty,
            generation is sequential using *llm* only.
        db_path: Path to the SQLite DB file.  Required when
            *extra_llms* is provided (workers open their own connections).

    Returns:
        List of result dicts (one per successfully generated query).
    """
    rng = random.Random(seed)

    # Find eligible files (with at least one page >= 200 chars)
    rows = conn.execute(
        """
        SELECT DISTINCT rel_path, filename
        FROM pdf_page_store
        WHERE file_type = 'pdf'
          AND content_chars >= 200
        ORDER BY rel_path
        """
    ).fetchall()

    eligible = [{"rel_path": r[0], "filename": r[1]} for r in rows]
    if not eligible:
        print("  Warning: No eligible files found for single-hop generation")
        return []

    print(f"  Eligible files for single-hop: {len(eligible)}")

    # Sample files
    if num_queries <= len(eligible):
        selected = rng.sample(eligible, num_queries)
    else:
        selected = rng.choices(eligible, k=num_queries)
        print(
            f"  Note: num_queries ({num_queries}) > eligible files "
            f"({len(eligible)}), some files will repeat"
        )

    # Build the full list of LLM clients
    all_llms = [llm]
    if extra_llms:
        all_llms.extend(extra_llms)
    num_workers = min(len(all_llms), len(selected))

    # --- Sequential path (single LLM) ---
    if num_workers <= 1 or db_path is None:
        if num_workers > 1 and db_path is None:
            print(
                "  Warning: parallel generation requires db_path; "
                "falling back to sequential"
            )
        print("  Workers: 1 (sequential)")
        results: List[Dict[str, Any]] = []
        for i, file_info in enumerate(selected):
            rel_path = file_info["rel_path"]
            filename = file_info["filename"]

            profile = load_profile(conn, rel_path)
            entities = get_profile_entities(profile)
            page = select_best_page(
                conn, rel_path, profile_entities=entities,
            )
            if not page:
                continue

            result = generate_one_query(
                llm,
                rel_path=rel_path,
                page_content=page["content"],
                page_number=page["page_number"],
                filename=filename,
                profile=profile,
                rng=rng,
                personas=personas,
                corpus_size_hint=corpus_size_hint,
            )
            if result:
                results.append(result)

            if (i + 1) % 25 == 0 or (i + 1) == len(selected):
                print(
                    f"  Single-hop progress: {len(results)}/{num_queries} "
                    f"generated ({i + 1}/{len(selected)} attempted)"
                )
        return results

    # --- Parallel path (multiple LLM clients) ---
    print(f"  Workers: {num_workers} (parallel)")

    result_slots: List[Optional[Dict[str, Any]]] = [None] * len(selected)
    worker_tasks: List[List[Dict[str, Any]]] = [
        [] for _ in range(num_workers)
    ]
    worker_indices: List[List[int]] = [[] for _ in range(num_workers)]
    for idx, task in enumerate(selected):
        w = idx % num_workers
        worker_tasks[w].append(task)
        worker_indices[w].append(idx)

    stats: Dict[str, int] = {
        "processed": 0, "generated": 0, "skipped": 0,
    }
    stats_lock = threading.Lock()
    t0 = time.monotonic()

    # Progress reporter
    stop_event = threading.Event()

    def _progress():
        while not stop_event.is_set():
            stop_event.wait(15)
            with stats_lock:
                processed = stats["processed"]
                generated = stats["generated"]
            elapsed = time.monotonic() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (
                (len(selected) - processed) / rate if rate > 0 else 0
            )
            pct = processed / len(selected) * 100 if selected else 0
            print(
                f"  Single-hop progress: {generated}/{num_queries} "
                f"generated ({processed}/{len(selected)} attempted, "
                f"{pct:.0f}%, {rate:.1f}/s, "
                f"ETA {remaining:.0f}s)"
            )
            if processed >= len(selected):
                break

    progress_thread = threading.Thread(target=_progress, daemon=True)
    progress_thread.start()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            future = executor.submit(
                _worker_generate,
                worker_id=i + 1,
                llm=all_llms[i],
                tasks=worker_tasks[i],
                db_path=db_path,
                result_slots=result_slots,
                result_indices=worker_indices[i],
                stats=stats,
                stats_lock=stats_lock,
                seed=seed,
                personas=personas,
                corpus_size_hint=corpus_size_hint,
            )
            futures[future] = i + 1

        total_generated = 0
        for future in as_completed(futures):
            wid = futures[future]
            try:
                count = future.result()
                total_generated += count
                log.info(
                    "Worker %d finished: %d queries", wid, count,
                )
            except Exception as exc:
                log.error("Worker %d failed: %s", wid, exc)

    stop_event.set()
    progress_thread.join(timeout=5)

    elapsed = time.monotonic() - t0
    valid = [r for r in result_slots if r is not None]
    print(
        f"  Single-hop complete: {len(valid)}/{num_queries} generated "
        f"in {elapsed:.0f}s ({len(valid)/elapsed:.1f}/s)"
    )
    return valid
