"""Hard negative mining (BM25, embedding, LLM judge)."""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from ragas.testset.graph import NodeType

from .utils import (
    extract_json_object,
    normalize_ynu,
    parse_reference_contexts,
    strip_hop_prefix,
    truncate_for_judge,
)


# ---------------------------------------------------------------------------
# LLM-based hard negative judge
# ---------------------------------------------------------------------------
def llm_is_hard_negative(
    judge_llm: Any,
    *,
    question: str,
    passage: str,
) -> Optional[bool]:
    """
    Ask an LLM whether a passage is a *hard negative* for a question.

    Returns True (relevant but not answerable), False (otherwise), or None (uncertain).
    """
    if judge_llm is None:
        return None

    system = (
        "You are a strict evaluator.\n"
        "Decide whether the PASSAGE is relevant to the QUESTION, and whether it "
        "contains enough information to answer the QUESTION.\n"
        "Use ONLY the passage. Do not use outside knowledge.\n"
        'If answerable="yes", you MUST include an exact verbatim quote from the '
        "passage that contains the key information.\n"
        "Return ONLY valid JSON."
    )

    passage_snippet = truncate_for_judge(passage, max_chars=8000)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"PASSAGE:\n{passage_snippet}\n\n"
        'Return JSON in this exact schema:\n'
        '{"relevant":"yes|no|uncertain",'
        '"answerable":"yes|no|uncertain",'
        '"evidence":"<verbatim quote if answerable=yes else empty>"}'
    )

    try:
        resp = judge_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception:
        return None

    content = getattr(resp, "content", None)
    content = content if isinstance(content, str) else str(resp)
    data = extract_json_object(content)
    if not isinstance(data, dict):
        return None

    relevant = normalize_ynu(data.get("relevant"))
    answerable = normalize_ynu(data.get("answerable"))
    evidence = str(data.get("evidence") or "")

    if answerable == "yes":
        if not evidence or evidence not in passage_snippet:
            return None

    if relevant == "yes" and answerable == "no":
        return True
    if answerable == "yes":
        return False
    if relevant == "no":
        return False
    return None


# ---------------------------------------------------------------------------
# Corpus page extraction from KG
# ---------------------------------------------------------------------------
def _extract_pages_from_kg(kg: Any) -> List[Dict[str, Any]]:
    """Build a page-level candidate set from KG DOCUMENT nodes."""
    pages_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for node in kg.nodes:
        if node.type != NodeType.DOCUMENT:
            continue
        md = node.get_property("document_metadata") or {}
        source = md.get("source")
        page0 = md.get("page")
        if not source or not isinstance(page0, int):
            continue

        filename = os.path.basename(str(source))
        page = page0 + 1
        key = (filename, page)

        page_content = node.get_property("page_content") or ""
        embedding = node.get_property("page_content_embedding")
        if embedding is None:
            embedding = node.get_property("summary_embedding")

        pages_by_key[key] = {
            "file": filename,
            "page": page,
            "source": str(source),
            "page_content": page_content,
            "embedding": embedding,
        }

    return list(pages_by_key.values())


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------
def build_bm25_index(pages: List[Dict[str, Any]]) -> Tuple[Any, List[List[str]]]:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank_bm25 is required for BM25 hard negative mining. "
            "Install with: pip install rank_bm25"
        )

    tokenized_corpus = [p["page_content"].lower().split() for p in pages]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def _top_indices_desc(scores: Any, top_n: int) -> List[int]:
    scores_np = np.asarray(scores, dtype=float)
    n = scores_np.shape[0]
    if n == 0:
        return []
    top_n = max(0, min(int(top_n), n))
    if top_n == 0:
        return []
    if top_n == n:
        return [int(i) for i in np.argsort(scores_np)[::-1]]
    idx = np.argpartition(scores_np, -top_n)[-top_n:]
    idx = idx[np.argsort(scores_np[idx])[::-1]]
    return [int(i) for i in idx]


# ---------------------------------------------------------------------------
# BM25 hard negative finder
# ---------------------------------------------------------------------------
def find_bm25_hard_negative_pages(
    query: str,
    pages: List[Dict[str, Any]],
    bm25: Any,
    *,
    exclude_files: set,
    exclude_pages: set,
    top_k: int = 50,
    pool_size: int = 200,
) -> List[Tuple[str, int]]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_idxs = _top_indices_desc(scores, top_n=min(pool_size, len(pages)))

    hard_negs: List[Tuple[str, int]] = []
    seen: set = set()
    for i in top_idxs:
        if float(scores[i]) <= 0:
            break
        p = pages[i]
        key = (p.get("file"), p.get("page"))
        if key in exclude_pages or p.get("file") in exclude_files or key in seen:
            continue
        if not p.get("file") or not p.get("page"):
            continue
        hard_negs.append((str(p["file"]), int(p["page"])))
        seen.add(key)
        if len(hard_negs) >= top_k:
            break

    return hard_negs


# ---------------------------------------------------------------------------
# Embedding hard negative finder
# ---------------------------------------------------------------------------
def find_embedding_hard_negative_pages(
    query_embedding: List[float],
    pages: List[Dict[str, Any]],
    *,
    candidate_indices: Optional[List[int]] = None,
    exclude_files: set,
    exclude_pages: set,
    top_k: int = 50,
    min_similarity: float = 0.25,
) -> List[Tuple[str, int]]:
    if candidate_indices is None:
        candidate_indices = list(range(len(pages)))

    cand = []
    for i in candidate_indices:
        if i < 0 or i >= len(pages):
            continue
        p = pages[i]
        key = (p.get("file"), p.get("page"))
        if key in exclude_pages or p.get("file") in exclude_files:
            continue
        emb = p.get("embedding")
        if emb is None:
            continue
        cand.append((p, emb))

    if not cand:
        return []

    q = np.array(query_embedding, dtype=float)
    qn = np.linalg.norm(q)
    if qn == 0:
        return []
    q = q / qn

    sims = []
    for _p, emb in cand:
        v = np.array(emb, dtype=float)
        vn = np.linalg.norm(v)
        sims.append(0.0 if vn == 0 else float(np.dot(q, v / vn)))

    order = np.argsort(np.asarray(sims))[::-1]
    out: List[Tuple[str, int]] = []
    seen: set = set()
    for j in order:
        sim = float(sims[int(j)])
        if sim < min_similarity:
            break
        p, _ = cand[int(j)]
        key = (p.get("file"), p.get("page"))
        if key in seen or not p.get("file") or not p.get("page"):
            continue
        out.append((str(p["file"]), int(p["page"])))
        seen.add(key)
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Source-file finder (for testset rows)
# ---------------------------------------------------------------------------
def find_source_files(
    reference_contexts: List[str], docs: List[Document]
) -> Dict[str, Any]:
    """Find source files and page numbers for given reference contexts."""
    all_sources: List[str] = []
    all_sources_with_pages: List[str] = []
    all_page_numbers: List[Any] = []
    all_source_page_pairs: List[Dict[str, Any]] = []
    seen_pairs: set = set()

    for context in reference_contexts:
        context_normalized = strip_hop_prefix(str(context)).lower().strip()

        for doc in docs:
            doc_content = doc.page_content.lower().strip()
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source)
            page_num = doc.metadata.get("page")

            match_found = False

            # Strategy 1: Exact containment
            if context_normalized in doc_content or doc_content in context_normalized:
                match_found = True

            # Strategy 2: Significant portion overlap
            if not match_found and len(context_normalized) > 100:
                chunk_size = 100
                for i in range(0, min(len(context_normalized), 500), chunk_size):
                    chunk = context_normalized[i : i + chunk_size]
                    if len(chunk) > 50 and chunk in doc_content:
                        match_found = True
                        break

            # Strategy 3: Key phrases (first/last parts)
            if not match_found and len(context_normalized) > 50:
                first_part = context_normalized[:50]
                last_part = context_normalized[-50:]
                if first_part in doc_content or last_part in doc_content:
                    match_found = True

            if match_found:
                if page_num is not None:
                    page_display = page_num + 1
                    source_with_page = f"{filename} (page {page_display})"
                else:
                    source_with_page = filename

                if source_with_page not in all_sources_with_pages:
                    all_sources_with_pages.append(source_with_page)
                    all_page_numbers.append(
                        page_num + 1 if page_num is not None else None
                    )
                    if page_num is not None:
                        pair = (filename, page_num + 1)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            all_source_page_pairs.append(
                                {"file": filename, "page": page_num + 1}
                            )

                if filename not in all_sources:
                    all_sources.append(filename)

    return {
        "sources": all_sources or ["unknown"],
        "sources_with_pages": all_sources_with_pages or ["unknown"],
        "page_numbers": all_page_numbers or [None],
        "source_page_pairs": all_source_page_pairs,
    }


# ---------------------------------------------------------------------------
# Main hard negative mining function
# ---------------------------------------------------------------------------
def mine_hard_negatives_for_testset(
    testset_df,
    kg: Any,
    docs: List[Document],
    embedding_model: Any,
    judge_llm: Any = None,
    num_bm25_negatives: int = 5,
    num_embedding_negatives: int = 5,
    max_judge_calls_per_query: int = 12,
) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """Mine hard negatives for all queries in the testset.

    Returns:
        (hard_negatives_list, source_mappings) - source_mappings is the
        per-row find_source_files result, reusable by save_testset.
    """
    print("  Mining hard negatives...")

    pages = _extract_pages_from_kg(kg)
    print(f"  Candidate pages from KG: {len(pages)}")
    pages_with_embeddings = sum(1 for p in pages if p.get("embedding") is not None)
    print(f"  Pages with embeddings: {pages_with_embeddings}/{len(pages)}")

    bm25 = None
    if num_bm25_negatives > 0:
        try:
            bm25, _ = build_bm25_index(pages)
            print("  Built BM25 index")
        except ImportError as e:
            print(f"  Warning: {e}")
            print("  Skipping BM25 hard negatives")

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
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            n = np.linalg.norm(v)
        if not np.isfinite(n) or n == 0:
            page_emb_norm_by_key[k] = None
            continue
        page_emb_norm_by_key[k] = v / n

    near_duplicate_cosine_threshold = 0.92
    hard_negatives_list: List[List[str]] = []
    source_mappings: List[Dict[str, Any]] = []

    # Pre-compute all query embeddings in a single batch (avoids N sequential API calls)
    query_embeddings: Dict[int, Optional[List[float]]] = {}
    if num_embedding_negatives > 0 and pages_with_embeddings > 0:
        all_queries = [(idx, row["user_input"]) for idx, row in testset_df.iterrows()]
        query_texts = [q for _, q in all_queries]
        query_indices = [idx for idx, _ in all_queries]
        try:
            batch_embs = embedding_model.embed_documents(query_texts)
            if isinstance(batch_embs, list) and len(batch_embs) == len(query_texts):
                for qi, emb in zip(query_indices, batch_embs):
                    query_embeddings[qi] = emb if isinstance(emb, list) else None
                print(f"  Pre-embedded {len(query_texts)} queries (batched)")
        except Exception as e:
            print(f"  Warning: Batch query embedding failed, falling back to per-query: {e}")

    for idx, row in testset_df.iterrows():
        query = row["user_input"]
        desired_count = max(int(num_bm25_negatives), int(num_embedding_negatives))
        if desired_count <= 0:
            hard_negatives_list.append([])
            source_mappings.append({
                "sources": ["unknown"], "sources_with_pages": ["unknown"],
                "page_numbers": [None], "source_page_pairs": [],
            })
            continue

        contexts = [
            strip_hop_prefix(c)
            for c in parse_reference_contexts(row.get("reference_contexts"))
        ]
        pos_info = find_source_files(contexts, docs)
        pos_files = {f for f in (pos_info.get("sources") or []) if f and f != "unknown"}
        pos_pages = {
            (d.get("file"), d.get("page"))
            for d in (pos_info.get("source_page_pairs") or [])
            if isinstance(d, dict) and d.get("file") and d.get("page")
        }

        source_mappings.append(pos_info)

        if not pos_files or not pos_pages:
            hard_negatives_list.append([])
            continue

        pos_embs = [
            page_emb_norm_by_key.get(k)
            for k in pos_pages
            if page_emb_norm_by_key.get(k) is not None
        ]
        pos_embs = [v for v in pos_embs if v is not None]

        bm25_negs: List[Tuple[str, int]] = []
        if bm25 is not None and num_bm25_negatives > 0:
            bm25_negs = find_bm25_hard_negative_pages(
                query, pages, bm25,
                exclude_files=pos_files, exclude_pages=pos_pages,
                top_k=max(50, num_bm25_negatives * 10),
            )

        emb_negs: List[Tuple[str, int]] = []
        if num_embedding_negatives > 0 and pages_with_embeddings > 0:
            try:
                # Use pre-computed embedding if available, else fallback
                q_emb = query_embeddings.get(idx)
                if q_emb is None:
                    q_emb = embedding_model.embed_query(query)
                cand_idxs: Optional[List[int]] = None
                if bm25 is not None:
                    scores = bm25.get_scores(query.lower().split())
                    cand_idxs = _top_indices_desc(scores, top_n=300)
                emb_negs = find_embedding_hard_negative_pages(
                    q_emb, pages,
                    candidate_indices=cand_idxs,
                    exclude_files=pos_files, exclude_pages=pos_pages,
                    top_k=max(50, num_embedding_negatives * 10),
                )
            except Exception as e:
                print(f"  Warning: Failed to embed query {idx}: {e}")

        # Merge strategies
        if num_bm25_negatives > 0 and num_embedding_negatives > 0:
            if bm25 is not None and bm25_negs and emb_negs:
                base_keys = sorted(set(bm25_negs).intersection(set(emb_negs)))
            else:
                base_keys = emb_negs or bm25_negs
        elif num_embedding_negatives > 0:
            base_keys = emb_negs
        else:
            base_keys = bm25_negs

        # Conservative filtering
        filtered_keys: List[Tuple[str, int]] = []
        seen_files: set = set()
        judged = 0
        for key in base_keys:
            f, p = key
            if f in pos_files or key in pos_pages:
                continue
            page_rec = page_by_key.get(key)
            if not page_rec:
                continue

            cand_text = str(page_rec.get("page_content") or "")

            cand_v = page_emb_norm_by_key.get(key)
            if cand_v is not None and pos_embs:
                max_sim = max(float(np.dot(cand_v, pv)) for pv in pos_embs)
                if max_sim >= near_duplicate_cosine_threshold:
                    continue

            if f in seen_files:
                continue

            verdict = llm_is_hard_negative(judge_llm, question=query, passage=cand_text)
            judged += 1
            if verdict is not True:
                if judged >= max_judge_calls_per_query:
                    break
                continue

            seen_files.add(f)
            filtered_keys.append(key)
            if len(filtered_keys) >= desired_count:
                break
            if judged >= max_judge_calls_per_query:
                break

        hard_negatives_list.append([f"{f} (page {p})" for (f, p) in filtered_keys])

    total_negs = sum(len(x) for x in hard_negatives_list)
    print(f"  Mined {total_negs} hard negative(s) for {len(testset_df)} queries")
    return hard_negatives_list, source_mappings
