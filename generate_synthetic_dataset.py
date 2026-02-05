#!/usr/bin/env python3
"""
RAGAS Synthetic Dataset Generator for Legal Documents

This script generates a synthetic Q&A dataset from legal documents using RAGAS.
It supports PDF files (non-PDF inputs are skipped).

Usage:
    python generate_synthetic_dataset.py [OPTIONS]

Environment Variables (OpenAI):
    OPENAI_API_KEY: Your OpenAI API key

Environment Variables (Azure OpenAI):
    AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    AZURE_OPENAI_DEPLOYMENT_NAME: Your chat model deployment name
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: Your embedding model deployment name
    AZURE_OPENAI_API_VERSION: API version (default: 2024-02-15-preview)
"""

import os
import ast
import argparse
import gzip
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dotenv import load_dotenv
import json
import numpy as np

# Document loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

# RAGAS imports
import ragas
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.testset_schema import Testset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.transforms.splitters import HeadlineSplitter
from ragas.testset.transforms import (
    default_transforms,
    HeadlinesExtractor,
    NodeFilter,
    apply_transforms,
)
from ragas.testset.transforms.extractors import EmbeddingExtractor
from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder
from ragas.testset.graph import Node, NodeType, Relationship, KnowledgeGraph, UUIDEncoder
from ragas.run_config import RunConfig
from ragas.testset.persona import Persona, generate_personas_from_kg
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()


class SafeHeadlineSplitter(HeadlineSplitter):
    """
    A wrapper around HeadlineSplitter that handles missing headlines gracefully.

    If a DOCUMENT node has no usable `headlines` property, this splitter will
    **skip splitting** (and not raise). Downstream filtering ensures such docs
    are excluded from question generation.
    """

    async def split(self, node: Node) -> Tuple[List[Node], List[Relationship]]:
        # Check if headlines property exists
        headlines = node.get_property("headlines")
        if not headlines:
            # No headlines found - safely skip splitting this node.
            # (We do NOT create fallback chunks; these docs are filtered out upstream.)
            return [node], []

        # Headlines exist - use parent class implementation
        return await super().split(node)


@dataclass
class HeadlinesRequiredFilter(NodeFilter):
    """
    Remove DOCUMENT nodes that don't have usable headlines.

    This prevents generating queries from documents/chunks where headline
    extraction failed or returned non-matching headings.
    """

    require_match_in_text: bool = True
    case_insensitive_match: bool = True

    async def custom_filter(self, node: Node, kg) -> bool:  # type: ignore[override]
        headlines = node.get_property("headlines")
        if not isinstance(headlines, list):
            return True

        cleaned = [
            h.strip() for h in headlines if isinstance(h, str) and h.strip()
        ]
        if not cleaned:
            return True

        if not self.require_match_in_text:
            return False

        text = node.get_property("page_content")
        if not isinstance(text, str) or not text.strip():
            return True

        text_cmp = text.lower() if self.case_insensitive_match else text
        for h in cleaned:
            h_cmp = h.lower() if self.case_insensitive_match else h
            if h_cmp in text_cmp:
                return False

        # None of the extracted headlines appear in the actual text → treat as unusable
        return True


def patch_transforms_with_safe_splitter(transforms, llm, embedding_model, *, add_content_embeddings: bool = True):
    """
    Patch RAGAS transforms to enforce headline-only generation and add page_content embeddings:

    - Extract headlines for all DOCUMENT nodes
    - Filter out DOCUMENT nodes without usable headlines
    - Replace HeadlineSplitter with SafeHeadlineSplitter (no fallback chunking)
    - Add page_content embedding for ALL nodes (for better KG connectivity + hard negatives)
    - Add content_similarity edges based on page_content embeddings
    """
    def doc_nodes_only(node: Node) -> bool:
        return node.type == NodeType.DOCUMENT

    def all_nodes_with_content(node: Node) -> bool:
        return node.get_property("page_content") is not None

    # Avoid duplicate headline extraction from default_transforms
    base_transforms = [
        t for t in transforms if not isinstance(t, HeadlinesExtractor)
    ]

    patched = [
        HeadlinesExtractor(llm=llm, filter_nodes=doc_nodes_only),
        HeadlinesRequiredFilter(filter_nodes=doc_nodes_only),
    ]

    for transform in base_transforms:
        if isinstance(transform, HeadlineSplitter):
            # Replace with safe version, preserving settings
            safe_splitter = SafeHeadlineSplitter(
                min_tokens=transform.min_tokens,
                max_tokens=transform.max_tokens,
                filter_nodes=transform.filter_nodes,
            )
            patched.append(safe_splitter)
        else:
            patched.append(transform)

    # Add page_content embedding for ALL nodes (not just those with summaries)
    if add_content_embeddings:
        # Embed page_content directly
        page_content_embedder = EmbeddingExtractor(
            embedding_model=embedding_model,
            property_name="page_content_embedding",
            embed_property_name="page_content",
            filter_nodes=all_nodes_with_content,
        )
        patched.append(page_content_embedder)

        # Build similarity edges from page_content embeddings
        content_similarity_builder = CosineSimilarityBuilder(
            property_name="page_content_embedding",
            new_property_name="content_similarity",
            threshold=0.5,  # Only create edges for reasonably similar content
        )
        patched.append(content_similarity_builder)

    return patched


CACHE_SCHEMA_VERSION = 2  # Bumped for page_content embeddings
PIPELINE_ID_BASE = "pdf_only__headlines_required"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj: Any) -> str:
    # Stable serialization for cache keys
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def compute_docs_cache_id(
    pdf_paths: List[Path],
    *,
    recursive: bool,
    max_files: Optional[int],
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "documents",
        "recursive": recursive,
        "max_files": max_files,
        "pdfs": [_file_fingerprint(p) for p in pdf_paths],
    }
    return _sha256_hex(_canonical_json(payload))[:16]


def compute_kg_cache_id(
    docs_cache_id: str,
    *,
    provider: str,
    llm_id: str,
    embedding_id: str,
    add_content_embeddings: bool = True,
) -> str:
    pipeline_id = PIPELINE_ID_BASE
    if add_content_embeddings:
        pipeline_id += "__content_embeddings"
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "knowledge_graph",
        "pipeline_id": pipeline_id,
        "ragas_version": getattr(ragas, "__version__", "unknown"),
        "docs_cache_id": docs_cache_id,
        "provider": provider,
        "llm_id": llm_id,
        "embedding_id": embedding_id,
        "add_content_embeddings": add_content_embeddings,
    }
    return _sha256_hex(_canonical_json(payload))[:16]


def save_documents_cache(docs: List[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for doc in docs:
            rec = {"page_content": doc.page_content, "metadata": doc.metadata}
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")


def load_documents_cache(path: Path) -> List[Document]:
    docs: List[Document] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(
                Document(
                    page_content=rec.get("page_content", ""),
                    metadata=rec.get("metadata") or {},
                )
            )
    return docs


def save_knowledge_graph_cache(kg: KnowledgeGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "nodes": [node.model_dump() for node in kg.nodes],
        "relationships": [rel.model_dump() for rel in kg.relationships],
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, cls=UUIDEncoder, ensure_ascii=False)


def load_knowledge_graph_cache(path: Path) -> KnowledgeGraph:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    nodes = [Node(**node_data) for node_data in data.get("nodes", [])]
    nodes_map = {str(node.id): node for node in nodes}

    relationships = []
    for rel_data in data.get("relationships", []):
        relationships.append(
            Relationship(
                id=rel_data["id"],
                type=rel_data["type"],
                source=nodes_map[rel_data["source"]],
                target=nodes_map[rel_data["target"]],
                bidirectional=rel_data.get("bidirectional", False),
                properties=rel_data.get("properties", {}),
            )
        )

    return KnowledgeGraph(nodes=nodes, relationships=relationships)


def save_personas_cache(personas: List[Persona], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in personas], f, ensure_ascii=False, indent=2)


def load_personas_cache(path: Path) -> List[Persona]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Persona(**p) for p in data]


_HOP_PREFIX_RE = re.compile(r"^\s*<\d+-hop>\s*", flags=re.IGNORECASE)


def strip_hop_prefix(text: str) -> str:
    """
    Remove RAGAS hop markers like "<1-hop>" that are sometimes prepended to reference contexts.
    """
    return _HOP_PREFIX_RE.sub("", text)


def parse_reference_contexts(value: Any) -> List[str]:
    """
    Robustly parse `reference_contexts` into a list of strings.

    RAGAS often stores this as a Python-list string (single quotes), which is not valid JSON.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            # Prefer Python literal parsing for "['a', 'b']" style strings.
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(x) for x in v if x is not None]
                return [str(v)]
            except Exception:
                pass
        # Fallback to JSON if it happens to be valid JSON.
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            return [str(v)]
        except Exception:
            return [s]

    return [str(value)]


def _truncate_for_judge(text: str, *, max_chars: int = 8000) -> str:
    """
    Keep judge prompts bounded in size (cost + latency).

    We keep both the start and end of the passage because legal docs often place
    the key clause near headings or near the end of a page.
    """
    if not text:
        return ""
    t = text.replace("\x00", " ").strip()
    if len(t) <= max_chars:
        return t
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return f"{t[:head]}\n...\n{t[-tail:]}"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first JSON object found in a model response.
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    # Strip code fences if present
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = t.replace("```", "").strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    blob = t[start : end + 1]
    try:
        return json.loads(blob)
    except Exception:
        return None


def _normalize_ynu(value: Any) -> str:
    """
    Normalize various yes/no/uncertain representations to 'yes'|'no'|'uncertain'.
    """
    s = str(value or "").strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    return "uncertain"


def llm_is_hard_negative(
    judge_llm: Any,
    *,
    question: str,
    passage: str,
) -> Optional[bool]:
    """
    Ask an LLM to determine whether a passage is a *hard negative* for a question:
    relevant to the question, but does NOT contain enough information to answer it.

    Returns:
      - True  => relevant=yes AND answerable=no
      - False => otherwise (answerable=yes OR irrelevant)
      - None  => parsing/format uncertainty (treat as unreliable)
    """
    if judge_llm is None:
        return None

    system = (
        "You are a strict evaluator.\n"
        "Decide whether the PASSAGE is relevant to the QUESTION, and whether it contains\n"
        "enough information to answer the QUESTION.\n"
        "Use ONLY the passage. Do not use outside knowledge.\n"
        "If answerable=\"yes\", you MUST include an exact verbatim quote from the passage\n"
        "that contains the key information.\n"
        "Return ONLY valid JSON."
    )

    passage_snippet = _truncate_for_judge(passage, max_chars=8000)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"PASSAGE:\n{passage_snippet}\n\n"
        'Return JSON in this exact schema:\n'
        '{"relevant":"yes|no|uncertain","answerable":"yes|no|uncertain","evidence":"<verbatim quote if answerable=yes else empty>"}'
    )

    try:
        resp = judge_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    except Exception:
        return None

    content = getattr(resp, "content", None)
    content = content if isinstance(content, str) else str(resp)
    data = _extract_json_object(content)
    if not isinstance(data, dict):
        return None

    relevant = _normalize_ynu(data.get("relevant"))
    answerable = _normalize_ynu(data.get("answerable"))
    evidence = str(data.get("evidence") or "")

    if answerable == "yes":
        # Require a verbatim quote to reduce hallucinated \"yes\" decisions
        if not evidence or evidence not in passage_snippet:
            return None

    if relevant == "yes" and answerable == "no":
        return True
    if answerable == "yes":
        return False
    if relevant == "no":
        return False
    return None


def get_all_chunks_from_kg(kg: "KnowledgeGraph") -> List[Dict[str, Any]]:
    """
    Extract all chunks (page_content + embedding) from the KnowledgeGraph.
    Returns a list of dicts with 'id', 'page_content', 'embedding', 'metadata'.
    """
    chunks = []
    for node in kg.nodes:
        page_content = node.get_property("page_content")
        if not page_content:
            continue

        # Prefer page_content_embedding, fall back to summary_embedding
        embedding = node.get_property("page_content_embedding")
        if embedding is None:
            embedding = node.get_property("summary_embedding")

        metadata = node.get_property("document_metadata") or {}

        chunks.append({
            "id": str(node.id),
            "page_content": page_content,
            "embedding": embedding,
            "metadata": metadata,
            "node_type": str(node.type),
        })
    return chunks


def build_bm25_index(chunks: List[Dict[str, Any]]) -> Tuple[Any, List[List[str]]]:
    """
    Build a BM25 index from chunk page_contents.
    Returns (BM25Okapi instance, tokenized_corpus).
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError(
            "rank_bm25 is required for BM25 hard negative mining. "
            "Install with: pip install rank_bm25"
        )

    # Simple whitespace tokenization (can be improved with better tokenizer)
    tokenized_corpus = [chunk["page_content"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus


def _extract_pages_from_kg(kg: "KnowledgeGraph") -> List[Dict[str, Any]]:
    """
    Build a page-level candidate set from KG DOCUMENT nodes.

    Returns dicts containing:
      - file (basename)
      - page (1-indexed int)
      - source (full path, if present)
      - page_content (text)
      - embedding (page_content_embedding preferred, else summary_embedding)
    """
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
        page = page0 + 1  # human-readable, 1-indexed
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


def _top_indices_desc(scores: Any, top_n: int) -> List[int]:
    """
    Return indices of the top_n scores in descending order, without sorting the entire array.
    """
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
    """
    Find hard negatives using BM25 scoring.
    Returns page refs (filename+page) that are lexically similar and satisfy exclusions.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_idxs = _top_indices_desc(scores, top_n=min(pool_size, len(pages)))

    hard_negs: List[Tuple[str, int]] = []
    seen: set = set()
    for i in top_idxs:
        score = float(scores[i])
        if score <= 0:
            break
        p = pages[i]
        key = (p.get("file"), p.get("page"))
        if key in exclude_pages:
            continue
        if p.get("file") in exclude_files:
            continue
        if key in seen:
            continue
        if not p.get("file") or not p.get("page"):
            continue
        hard_negs.append((str(p["file"]), int(p["page"])))
        seen.add(key)
        if len(hard_negs) >= top_k:
            break

    return hard_negs


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
    """
    Find hard negatives using embedding cosine similarity.
    Returns page refs (filename+page) that are semantically similar and satisfy exclusions.
    """
    if candidate_indices is None:
        candidate_indices = list(range(len(pages)))

    cand = []
    for i in candidate_indices:
        if i < 0 or i >= len(pages):
            continue
        p = pages[i]
        key = (p.get("file"), p.get("page"))
        if key in exclude_pages:
            continue
        if p.get("file") in exclude_files:
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
        p, _emb = cand[int(j)]
        key = (p.get("file"), p.get("page"))
        if key in seen:
            continue
        if not p.get("file") or not p.get("page"):
            continue
        out.append((str(p["file"]), int(p["page"])))
        seen.add(key)
        if len(out) >= top_k:
            break
    return out


def mine_hard_negatives_for_testset(
    testset_df,
    kg: "KnowledgeGraph",
    docs: List[Document],
    embedding_model,
    judge_llm: Any = None,
    num_bm25_negatives: int = 5,
    num_embedding_negatives: int = 5,
    max_judge_calls_per_query: int = 12,
) -> List[List[str]]:
    """
    Mine hard negatives for all queries in the testset.

    Hard negatives are stored as "<filename>.pdf (page N)" (1-indexed pages).
    Exclusions:
      - No hard negatives from the same source PDF as any positive context
      - Exclude positives by (file,page) rather than content matching

    Returns:
        hard_negatives_list - one list per row
    """
    print("\nMining hard negatives...")

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

    # Fast lookup for filtering
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
        if v.ndim != 1 or v.size == 0:
            page_emb_norm_by_key[k] = None
            continue
        if not np.all(np.isfinite(v)):
            page_emb_norm_by_key[k] = None
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            n = np.linalg.norm(v)
        if not np.isfinite(n) or n == 0:
            page_emb_norm_by_key[k] = None
            continue
        page_emb_norm_by_key[k] = v / n

    # Thresholds for reliability filters (general, non rule-based)
    near_duplicate_cosine_threshold = 0.92

    hard_negatives_list: List[List[str]] = []

    for idx, row in testset_df.iterrows():
        query = row["user_input"]
        desired_count = max(int(num_bm25_negatives), int(num_embedding_negatives))
        if desired_count <= 0:
            hard_negatives_list.append([])
            continue

        contexts = [strip_hop_prefix(c) for c in parse_reference_contexts(row.get("reference_contexts"))]
        pos_info = find_source_files(contexts, docs)
        pos_files = {f for f in (pos_info.get("sources") or []) if f and f != "unknown"}
        pos_pages = {
            (d.get("file"), d.get("page"))
            for d in (pos_info.get("source_page_pairs") or [])
            if isinstance(d, dict) and d.get("file") and d.get("page")
        }

        # Reliability > coverage: if we can't identify positives confidently, skip.
        if not pos_files or not pos_pages:
            hard_negatives_list.append([])
            continue

        # Positive embeddings (for near-duplicate filtering)
        pos_embs = [
            page_emb_norm_by_key.get(k)
            for k in pos_pages
            if page_emb_norm_by_key.get(k) is not None
        ]
        pos_embs = [v for v in pos_embs if v is not None]

        bm25_negs: List[Tuple[str, int]] = []
        if bm25 is not None and num_bm25_negatives > 0:
            bm25_negs = find_bm25_hard_negative_pages(
                query,
                pages,
                bm25,
                exclude_files=pos_files,
                exclude_pages=pos_pages,
                top_k=max(50, num_bm25_negatives * 10),
            )

        emb_negs: List[Tuple[str, int]] = []
        if num_embedding_negatives > 0 and pages_with_embeddings > 0:
            try:
                q_emb = embedding_model.embed_query(query)
                cand_idxs: Optional[List[int]] = None
                if bm25 is not None:
                    scores = bm25.get_scores(query.lower().split())
                    cand_idxs = _top_indices_desc(scores, top_n=300)
                emb_negs = find_embedding_hard_negative_pages(
                    q_emb,
                    pages,
                    candidate_indices=cand_idxs,
                    exclude_files=pos_files,
                    exclude_pages=pos_pages,
                    top_k=max(50, num_embedding_negatives * 10),
                )
            except Exception as e:
                print(f"  Warning: Failed to embed query {idx}: {e}")

        # Reliability-first merge:
        # - If both approaches are enabled and available, only keep intersection.
        # - If only one approach is enabled, use it.
        if num_bm25_negatives > 0 and num_embedding_negatives > 0:
            # Prefer intersection when both approaches are available; otherwise fall back
            if bm25 is not None and bm25_negs and emb_negs:
                base_keys = sorted(set(bm25_negs).intersection(set(emb_negs)))
            else:
                base_keys = emb_negs or bm25_negs
        elif num_embedding_negatives > 0:
            base_keys = emb_negs
        else:
            base_keys = bm25_negs

        # Apply conservative filters to avoid false negatives
        filtered_keys: List[Tuple[str, int]] = []
        seen_files: set = set()
        judged = 0
        for key in base_keys:
            f, p = key
            if f in pos_files:
                continue
            if key in pos_pages:
                continue
            page_rec = page_by_key.get(key)
            if not page_rec:
                continue

            cand_text = str(page_rec.get("page_content") or "")

            # Embedding near-duplicate filter (when available)
            cand_v = page_emb_norm_by_key.get(key)
            if cand_v is not None and pos_embs:
                max_sim = max(float(np.dot(cand_v, pv)) for pv in pos_embs)
                if max_sim >= near_duplicate_cosine_threshold:
                    continue

            # Optional diversity: don't emit multiple negatives from same file
            if f in seen_files:
                continue

            # LLM-based validation (general): keep only if relevant but not answerable.
            verdict = llm_is_hard_negative(
                judge_llm,
                question=query,
                passage=cand_text,
            )
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
    return hard_negatives_list


def sort_documents_deterministically(docs: List[Document]) -> List[Document]:
    def _key(d: Document):
        source = str(d.metadata.get("source", ""))
        page = d.metadata.get("page")
        page_sort = page if isinstance(page, int) else -1
        return (source, page_sort)

    return sorted(docs, key=_key)


def load_documents(
    base_path: str,
    file_types: List[str] = ["pdf"],
    max_files: Optional[int] = None,
    recursive: bool = True,
) -> List[Document]:
    """
    Load documents from the specified directory.

    Args:
        base_path: Root directory containing documents
        file_types: List of file extensions to load
        max_files: Maximum number of files to load (None for all)
        recursive: Whether to search subdirectories

    Returns:
        List of loaded Document objects
    """
    all_docs = []
    glob_pattern = "**/*" if recursive else "*"

    loaders_config = {
        "pdf": (PyPDFLoader, "*.pdf"),
    }

    # Enforce PDF-only loading (skip non-PDF types even if requested)
    normalized_types = [ft.lower().lstrip(".") for ft in (file_types or [])]
    non_pdf_types = sorted({ft for ft in normalized_types if ft and ft != "pdf"})
    if non_pdf_types:
        print(
            f"Skipping non-PDF file types: {', '.join(non_pdf_types)} "
            "(only PDF is supported)"
        )

    # Always load PDFs (this script is PDF-only); ignore any other entries.
    file_types = ["pdf"]

    for file_type in file_types:
        if file_type.lower() not in loaders_config:
            print(f"Warning: Unsupported file type '{file_type}', skipping...")
            continue

        loader_class, pattern = loaders_config[file_type.lower()]
        full_pattern = f"{glob_pattern.rstrip('*')}{pattern}" if recursive else pattern

        print(f"Loading {file_type.upper()} files with pattern: {full_pattern}")

        try:
            loader = DirectoryLoader(
                base_path,
                glob=full_pattern,
                loader_cls=loader_class,
                show_progress=True,
                use_multithreading=True,
                max_concurrency=4,
            )
            docs = loader.load()
            print(f"  Loaded {len(docs)} {file_type.upper()} documents")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  Error loading {file_type} files: {e}")

    if max_files and len(all_docs) > max_files:
        print(f"Limiting to {max_files} documents (from {len(all_docs)} total)")
        all_docs = all_docs[:max_files]

    print(f"\nTotal documents loaded: {len(all_docs)}")
    return all_docs


def setup_llm_and_embeddings(model: str = "gpt-4o-mini", provider: str = "auto"):
    """
    Set up the LLM and embedding models for RAGAS.

    Args:
        model: Model/deployment name to use
        provider: Provider to use ('openai', 'azure', or 'auto' to detect)

    Returns:
        Tuple of (generator_llm, generator_embeddings)
    """
    # Auto-detect provider based on environment variables
    if provider == "auto":
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError(
                "No API credentials found. Please set either:\n"
                "  - OPENAI_API_KEY for OpenAI, or\n"
                "  - AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT for Azure OpenAI"
            )

    if provider == "azure":
        return setup_azure_openai(model)
    else:
        return setup_openai(model)


def setup_openai(model: str):
    """Set up OpenAI LLM and embeddings."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it with: export OPENAI_API_KEY='your-key'"
        )

    print(f"Setting up OpenAI LLM with model: {model}")
    llm = ChatOpenAI(model=model, temperature=0.3)
    generator_llm = LangchainLLMWrapper(llm)

    print("Setting up OpenAI embeddings...")
    embeddings = OpenAIEmbeddings()
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return generator_llm, generator_embeddings


def setup_azure_openai(model: str):
    """Set up Azure OpenAI LLM and embeddings."""
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    # Deployment names - use provided model or fall back to env vars
    chat_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", model)
    embedding_deployment = os.environ.get(
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
        "text-embedding-ada-002"
    )

    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

    print("Setting up Azure OpenAI LLM")
    print(f"  Endpoint: {endpoint}")
    print(f"  Chat deployment: {chat_deployment}")
    print(f"  API version: {api_version}")

    llm = AzureChatOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=chat_deployment,
        temperature=1,  # Some Azure models (like o1/gpt-5) only support temperature=1
    )
    generator_llm = LangchainLLMWrapper(llm)

    print(f"Setting up Azure OpenAI embeddings (deployment: {embedding_deployment})...")
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        azure_deployment=embedding_deployment,
    )
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return generator_llm, generator_embeddings


def build_knowledge_graph(
    docs: List[Document],
    generator_llm,
    generator_embeddings,
    *,
    add_content_embeddings: bool = True,
) -> KnowledgeGraph:
    """
    Build a RAGAS KnowledgeGraph from loaded documents by applying transforms.

    This is the expensive "processing" step (LLM extraction + embeddings + relationship building).

    Args:
        docs: List of loaded documents
        generator_llm: Wrapped LLM for generation
        generator_embeddings: Wrapped embeddings model
        add_content_embeddings: If True, embed page_content for ALL nodes (improves
            KG connectivity and enables better hard negative mining)
    """
    print("\nBuilding knowledge graph (applying RAGAS transforms)...")

    # Get default transforms and patch for headline-only generation
    from langchain_core.documents import Document as LCDocument

    lc_docs = [
        LCDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in docs
    ]
    transforms = default_transforms(
        documents=lc_docs,
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )
    transforms = patch_transforms_with_safe_splitter(
        transforms,
        llm=generator_llm,
        embedding_model=generator_embeddings,
        add_content_embeddings=add_content_embeddings,
    )
    print("Headline-only generation enabled (skipping documents without usable headlines)")
    if add_content_embeddings:
        print("Page content embeddings enabled (100% node coverage for similarity edges)")

    nodes: List[Node] = []
    for doc in docs:
        nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    kg = KnowledgeGraph(nodes=nodes)
    apply_transforms(kg, transforms, run_config=RunConfig())
    print(f"Knowledge graph ready: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
    return kg


DEFAULT_CORPUS_SIZE_HINT = 7000

# Heuristic check for "doc-internal" / deictic queries that don't stand alone in a large corpus.
_REFERENTIAL_QUERY_RE = re.compile(
    r"("
    r"\bthis case\b|"
    r"\bthis matter\b|"
    r"\bthis document\b|"
    r"\bthis agreement\b|"
    r"\bthe above\b|"
    r"\bherein\b|"
    r"\bhereinafter\b|"
    r"\baforementioned\b|"
    r"\bin this (case|matter|document|agreement)\b|"
    r"\bthe parties\b"
    r")",
    flags=re.IGNORECASE,
)


def build_corpus_llm_context(*, corpus_size_hint: Optional[int] = None) -> str:
    """
    LLM guidance to generate *standalone* queries suitable for a large multi-document corpus.

    This string is passed to RAGAS as `llm_context` and is surfaced inside query-generation prompts.
    """
    size_phrase = (
        f"~{int(corpus_size_hint)}+ PDF documents"
        if isinstance(corpus_size_hint, int) and corpus_size_hint > 0
        else "thousands of PDF documents"
    )
    return (
        "You are generating queries for a retrieval/search/RAG system.\n"
        f"The user is searching across a large corpus of {size_phrase}.\n\n"
        "CRITICAL: Each query must be a standalone, first-turn query.\n"
        "- Do NOT assume a document/case has already been selected.\n"
        "- Avoid deictic references like \"this case\", \"this document\", \"the above\", \"herein\", etc.\n"
        "- Include concrete identifiers from the provided context to disambiguate (e.g., case caption with \"v.\",\n"
        "  party/organization names, court/jurisdiction, docket/case number, statute/section, property address,\n"
        "  agreement name, distinctive dates).\n"
        "- Do NOT mention filenames or page numbers.\n"
    )


CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION = (
    "Generate a single-hop query and answer based on the specified conditions (persona, term, style, length) "
    "and the provided context. Ensure the answer is entirely faithful to the context, using only the information "
    "directly from the provided context.\n\n"
    "### Search Query Constraints (CRITICAL)\n"
    "The query will be asked in a large corpus setting (thousands of PDFs). Therefore:\n"
    "- The query MUST be standalone as a first-turn query.\n"
    "- Do NOT write queries that assume a document/case is already selected (avoid: \"this case\", \"this document\", "
    "\"the above\", \"herein\", \"the parties\" without naming them).\n"
    "- The query MUST include at least one concrete identifier copied verbatim from the context (case caption with \"v.\", "
    "party/organization name, court/jurisdiction, docket/case number, statute/section, property address, agreement name, "
    "or a distinctive date).\n"
    "- Do NOT mention filenames, page numbers, or \"the provided context\".\n"
    "- Follow query_style/query_length. If query_style asks for misspellings/poor grammar, keep identifiers intact "
    "(do not misspell party names or case numbers).\n\n"
    "### Instructions:\n"
    "1. **Generate a Query**: Based on the context, persona, term, style, and length, create a question that aligns with "
    "the persona's perspective, incorporates the term, and satisfies the standalone constraints above.\n"
    "2. **Generate an Answer**: Using only the content from the provided context, construct a detailed answer to the query. "
    "Do not add any information not included in or inferable from the context.\n"
    "3. **Additional Context** (if provided): If llm_context is provided, use it as additional guidance for query framing. "
    "Still ensure the content comes only from the provided context.\n"
)


CORPUS_MULTI_HOP_PROMPT_INSTRUCTION = (
    "Generate a multi-hop query and answer based on the specified conditions (persona, themes, style, length) "
    "and the provided context. The themes represent a set of phrases either extracted or generated from the "
    "context, which highlight the suitability of the selected context for multi-hop query creation. Ensure the query "
    "explicitly incorporates these themes.\n\n"
    "### Search Query Constraints (CRITICAL)\n"
    "The query will be asked in a large corpus setting (thousands of PDFs). Therefore:\n"
    "- The query MUST be standalone as a first-turn query.\n"
    "- Do NOT write queries that assume a document/case is already selected (avoid: \"this case\", \"this document\", "
    "\"the above\", \"herein\", \"the parties\" without naming them).\n"
    "- The query MUST include at least one concrete identifier copied verbatim from the contexts (case caption with \"v.\", "
    "party/organization name, court/jurisdiction, docket/case number, statute/section, property address, agreement name, "
    "or a distinctive date).\n"
    "- Do NOT mention filenames, page numbers, or \"the provided context\".\n"
    "- Follow query_style/query_length. If query_style asks for misspellings/poor grammar, keep identifiers intact "
    "(do not misspell party names or case numbers).\n\n"
    "### Instructions:\n"
    "1. **Generate a Multi-Hop Query**: Use the provided context segments and themes to form a query that requires combining "
    "information from multiple segments (e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or more "
    "themes, includes concrete identifiers, and makes sense with *no prior conversation*.\n"
    "2. **Generate an Answer**: Use only the content from the provided context to create a detailed and faithful answer to "
    "the query. Avoid adding information that is not directly present or inferable from the given context.\n"
    "3. **Multi-Hop Context Tags**:\n"
    "   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.\n"
    "   - Ensure the query uses information from at least two segments and connects them meaningfully.\n"
    "4. **Additional Context** (if provided): If llm_context is provided, use it as additional guidance for query framing. "
    "Still ensure the content comes only from the provided context.\n"
)


def build_query_distribution_for_pipeline(
    llm,
    kg: KnowledgeGraph,
    *,
    standalone_queries: bool,
    llm_context: Optional[str],
):
    """
    Build a query_distribution that nudges RAGAS to produce *standalone, corpus-appropriate* queries.

    If standalone_queries is False, returns None (RAGAS defaults are used).
    """
    if not standalone_queries:
        return None

    # Use RAGAS's compatibility filtering, then override the prompts.
    from ragas.testset.synthesizers import default_query_distribution
    from ragas.testset.synthesizers.single_hop.prompts import (
        QueryAnswerGenerationPrompt as SingleHopQueryAnswerGenerationPrompt,
    )
    from ragas.testset.synthesizers.multi_hop.prompts import (
        QueryAnswerGenerationPrompt as MultiHopQueryAnswerGenerationPrompt,
    )

    qd = default_query_distribution(llm, kg, llm_context)
    patched = []
    for synthesizer, prob in qd:
        name = str(getattr(synthesizer, "name", "") or "").lower()
        if name.startswith("single_hop"):
            p = SingleHopQueryAnswerGenerationPrompt()
            p.instruction = CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION
            synthesizer.generate_query_reference_prompt = p
        else:
            p = MultiHopQueryAnswerGenerationPrompt()
            p.instruction = CORPUS_MULTI_HOP_PROMPT_INSTRUCTION
            synthesizer.generate_query_reference_prompt = p
        patched.append((synthesizer, prob))

    return patched


def generate_testset_from_knowledge_graph(
    kg: KnowledgeGraph,
    generator_llm,
    generator_embeddings,
    *,
    testset_size: int = 50,
    personas: Optional[List[Persona]] = None,
    llm_context: Optional[str] = None,
    query_distribution=None,
) -> Testset:
    """
    Generate a synthetic testset from a pre-built KnowledgeGraph.

    This is the "generation" step (LLM writes the final Q/A pairs).
    """
    print("\nInitializing TestsetGenerator...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
        llm_context=llm_context,
    )

    print(f"Generating testset with {testset_size} samples...")
    print("This may take a while depending on testset size...")
    return generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
    )


def _warn_on_referential_queries(testset: Testset, *, limit: int = 5) -> None:
    """
    Best-effort sanity check: warn if generated queries look "doc-internal".
    """
    try:
        df = testset.to_pandas()
    except Exception:
        return
    if "user_input" not in df.columns:
        return

    bad = []
    for q in df["user_input"].tolist():
        qs = str(q or "")
        if _REFERENTIAL_QUERY_RE.search(qs):
            bad.append(qs)

    if not bad:
        return

    print(
        f"\nWarning: {len(bad)}/{len(df)} generated query(ies) look referential "
        "(e.g., 'this case', 'the above'). Consider tightening corpus-query guidance."
    )
    for i, q in enumerate(bad[: max(1, int(limit))], start=1):
        print(f"  Referenced query {i}: {q}")


def build_content_to_source_map(docs: List[Document]) -> Dict[str, str]:
    """
    Build a mapping from document content to source file paths.

    Args:
        docs: List of loaded documents with metadata

    Returns:
        Dictionary mapping content snippets to source file paths
    """
    content_map = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        # Extract just the filename from the path
        filename = os.path.basename(source)
        content_map[doc.page_content] = filename
    return content_map


def find_source_files(reference_contexts: List[str], docs: List[Document]) -> Dict[str, any]:
    """
    Find the source files and page numbers for given reference contexts.

    Args:
        reference_contexts: List of context strings from the testset
        docs: Original loaded documents

    Returns:
        Dictionary with:
            - 'sources': List of source file names
            - 'sources_with_pages': List of source strings with page numbers (e.g., "file.pdf (page 5)")
            - 'page_numbers': List of page numbers (or None for non-paginated docs)
    """
    all_sources = []
    all_sources_with_pages = []
    all_page_numbers = []
    all_source_page_pairs: List[Dict[str, Any]] = []
    seen_pairs: set = set()

    for context in reference_contexts:
        # Normalize context for comparison
        context_normalized = strip_hop_prefix(str(context)).lower().strip()

        for doc in docs:
            doc_content = doc.page_content.lower().strip()
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source)

            # Get page number if available (PyPDFLoader uses 'page', 0-indexed)
            page_num = doc.metadata.get("page")

            # Check various matching strategies
            match_found = False

            # Strategy 1: Exact containment
            if context_normalized in doc_content or doc_content in context_normalized:
                match_found = True

            # Strategy 2: Check if significant portion of context is in document
            if not match_found and len(context_normalized) > 100:
                # Check multiple chunks of the context
                chunk_size = 100
                for i in range(0, min(len(context_normalized), 500), chunk_size):
                    chunk = context_normalized[i:i+chunk_size]
                    if len(chunk) > 50 and chunk in doc_content:
                        match_found = True
                        break

            # Strategy 3: Check for key phrases (first and last parts)
            if not match_found and len(context_normalized) > 50:
                first_part = context_normalized[:50]
                last_part = context_normalized[-50:]
                if first_part in doc_content or last_part in doc_content:
                    match_found = True

            if match_found:
                # Build source string with page number if available
                if page_num is not None:
                    # Convert to 1-indexed for human readability
                    page_display = page_num + 1
                    source_with_page = f"{filename} (page {page_display})"
                else:
                    source_with_page = filename

                # Avoid duplicates
                if source_with_page not in all_sources_with_pages:
                    all_sources_with_pages.append(source_with_page)
                    all_page_numbers.append(page_num + 1 if page_num is not None else None)
                    if page_num is not None:
                        pair = (filename, page_num + 1)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            all_source_page_pairs.append({"file": filename, "page": page_num + 1})

                if filename not in all_sources:
                    all_sources.append(filename)

    return {
        'sources': all_sources if all_sources else ["unknown"],
        'sources_with_pages': all_sources_with_pages if all_sources_with_pages else ["unknown"],
        'page_numbers': all_page_numbers if all_page_numbers else [None],
        # Structured page ids (filename+page) for reliable comparisons downstream
        'source_page_pairs': all_source_page_pairs,
    }


def save_testset(
    testset,
    output_path: str,
    formats: List[str] = ["csv", "json"],
    docs: Optional[List[Document]] = None,
    hard_negatives: Optional[List[List[str]]] = None,
):
    """
    Save the generated testset to files.

    Args:
        testset: Generated RAGAS testset
        output_path: Base path for output files (without extension)
        formats: List of output formats ('csv', 'json', 'parquet')
        docs: Original documents for source file mapping
        hard_negatives: Hard negatives per row as "<filename>.pdf (page N)"
    """
    df = testset.to_pandas()

    # Add source file names if documents are provided
    if docs:
        print("\nMapping reference contexts to source files...")
        source_files_list = []
        source_files_with_pages_list = []
        page_numbers_list = []

        for idx, row in df.iterrows():
            contexts = parse_reference_contexts(row.get("reference_contexts"))

            # Find sources for all contexts in this row
            result = find_source_files(contexts if isinstance(contexts, list) else [contexts], docs)
            source_files_list.append(result['sources'])
            source_files_with_pages_list.append(result['sources_with_pages'])
            page_numbers_list.append(result['page_numbers'])

        # Add source files as a new column (just filenames)
        df["source_files"] = [json.dumps(files) for files in source_files_list]

        # Add source files with page numbers
        df["source_files_with_pages"] = [json.dumps(files) for files in source_files_with_pages_list]

        # Add page numbers as a separate column
        df["page_numbers"] = [json.dumps(pages) for pages in page_numbers_list]

        # Also add simple comma-separated versions for easier reading
        df["source_files_readable"] = [", ".join(files) if files != ["unknown"] else "unknown" for files in source_files_list]
        df["source_files_with_pages_readable"] = [", ".join(files) if files != ["unknown"] else "unknown" for files in source_files_with_pages_list]

    # Add hard negatives if provided
    if hard_negatives is not None:
        df["hard_negatives"] = [json.dumps(negs, ensure_ascii=False) for negs in hard_negatives]
        neg_count = sum(len(negs) for negs in hard_negatives)
        print(f"  Added {neg_count} hard negatives ({neg_count / len(df):.1f} avg per query)")

    print(f"\nGenerated {len(df)} test samples")
    print("\nSample preview:")
    preview_cols = ["user_input", "source_files_with_pages_readable"] if "source_files_with_pages_readable" in df.columns else ["user_input"]
    print(df[preview_cols].head())

    for fmt in formats:
        file_path = f"{output_path}.{fmt}"
        if fmt == "csv":
            df.to_csv(file_path, index=False)
        elif fmt == "json":
            df.to_json(file_path, orient="records", indent=2)
        elif fmt == "parquet":
            df.to_parquet(file_path, index=False)
        else:
            print(f"Warning: Unknown format '{fmt}', skipping...")
            continue
        print(f"Saved testset to: {file_path}")

    return df


def main():
    script_dir = Path(__file__).resolve().parent
    default_dataset_dir = script_dir / "search-dataset"

    def resolve_input_dir(path_str: str) -> Path:
        """
        Resolve an input directory path robustly.

        - Absolute paths are used as-is.
        - Relative paths are first interpreted relative to the current working
          directory; if that doesn't exist, relative to this script's folder.
        """
        p = Path(path_str).expanduser()
        if p.is_absolute():
            return p

        cwd_candidate = (Path.cwd() / p)
        if cwd_candidate.exists():
            return cwd_candidate.resolve()

        return (script_dir / p).resolve()

    def resolve_input_file(path_str: str, base_dir: Path) -> Path:
        """
        Resolve an input file path robustly.

        - Absolute paths are used as-is.
        - Relative paths are tried in this order:
          1) relative to current working directory
          2) relative to the resolved input directory
          3) relative to this script's folder
        """
        p = Path(path_str).expanduser()
        if p.is_absolute():
            return p

        cwd_candidate = (Path.cwd() / p)
        if cwd_candidate.exists():
            return cwd_candidate.resolve()

        base_candidate = (base_dir / p)
        if base_candidate.exists():
            return base_candidate.resolve()

        return (script_dir / p).resolve()

    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A dataset from legal documents using RAGAS"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(default_dataset_dir) if default_dataset_dir.exists() else ".",
        help=(
            "Directory containing documents. "
            "Defaults to ./search-dataset (next to this script) if present; otherwise current directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_dataset",
        help="Output file base name without extension (default: synthetic_dataset)",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=50,
        help="Number of test samples to generate (default: 50)",
    )
    parser.add_argument(
        "--standalone-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate standalone, corpus-level search queries that make sense with many documents in context "
            "(default: enabled). Disable with --no-standalone-queries to use RAGAS defaults."
        ),
    )
    parser.add_argument(
        "--corpus-size-hint",
        type=int,
        default=DEFAULT_CORPUS_SIZE_HINT,
        help=(
            "Approximate number of PDFs in the corpus (used only to guide query generation when "
            "--standalone-queries is enabled)."
        ),
    )
    parser.add_argument(
        "--query-llm-context",
        type=str,
        default=None,
        help=(
            "Optional additional guidance passed to RAGAS as llm_context for query generation. "
            "When --standalone-queries is enabled, this will be appended to the default corpus guidance."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--file-types",
        type=str,
        nargs="+",
        default=["pdf"],
        help="File types to load (PDF only; non-PDF entries are ignored)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "Maximum number of extracted documents (PDF pages) to process (default: no limit). "
            "Note: PyPDFLoader yields one Document per PDF page."
        ),
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help=(
            "Maximum number of PDF files to include (default: no limit). "
            "Applied before loading pages; use this to limit by PDF count."
        ),
    )
    parser.add_argument(
        "--output-formats",
        type=str,
        nargs="+",
        default=["csv", "json"],
        help="Output formats (default: csv json)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories",
    )
    parser.add_argument(
        "--specific-folders",
        type=str,
        nargs="+",
        default=None,
        help="Only load from specific subdirectories",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Specific files to include (in addition to folders)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["auto", "openai", "azure"],
        default="auto",
        help="LLM provider to use (default: auto-detect from env vars)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(script_dir / "processed"),
        help=(
            "Directory to store cached intermediate artifacts (documents/KG/personas). "
            "If caches exist, they will be reused on subsequent runs."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving cached intermediate artifacts",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Recompute intermediate artifacts even if cache exists",
    )
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help="Enable hard negative mining for IR evaluation",
    )
    parser.add_argument(
        "--num-bm25-negatives",
        type=int,
        default=5,
        help="BM25 candidate mining budget per query (default: 5)",
    )
    parser.add_argument(
        "--num-embedding-negatives",
        type=int,
        default=5,
        help="Embedding candidate mining budget per query (default: 5)",
    )
    parser.add_argument(
        "--no-content-embeddings",
        action="store_true",
        help="Disable page_content embeddings (use summary_embedding only)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAGAS Synthetic Dataset Generator")
    print("=" * 60)

    # Resolve base input dir (handles script moved one level up)
    base_input_dir = resolve_input_dir(args.input_dir)

    # Determine input directories
    if args.specific_folders:
        input_dirs = [(base_input_dir / folder) for folder in args.specific_folders]
    else:
        input_dirs = [base_input_dir]

    # Cache config
    cache_enabled = not args.no_cache
    processed_dir = Path(args.processed_dir).expanduser()
    if not processed_dir.is_absolute():
        processed_dir = (script_dir / processed_dir).resolve()

    docs_cache_dir = processed_dir / "docs"
    kg_cache_dir = processed_dir / "kg"
    personas_cache_dir = processed_dir / "personas"
    meta_dir = processed_dir / "meta"

    if cache_enabled:
        docs_cache_dir.mkdir(parents=True, exist_ok=True)
        kg_cache_dir.mkdir(parents=True, exist_ok=True)
        personas_cache_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

    # Collect PDF paths for a stable cache key
    def collect_pdf_paths_from_dir(d: Path, *, recursive: bool) -> List[Path]:
        if not d.exists() or not d.is_dir():
            return []
        iterator = d.rglob("*.pdf") if recursive else d.glob("*.pdf")
        out: List[Path] = []
        for p in iterator:
            if p.is_file():
                out.append(p.resolve())
        return out

    pdf_paths: List[Path] = []
    for d in input_dirs:
        pdf_paths.extend(
            collect_pdf_paths_from_dir(Path(d), recursive=not args.no_recursive)
        )
    if args.files:
        for file_path in args.files:
            resolved_file_path = resolve_input_file(file_path, base_input_dir)
            if resolved_file_path.exists() and resolved_file_path.suffix.lower() == ".pdf":
                pdf_paths.append(resolved_file_path.resolve())

    pdf_paths = sorted(set(pdf_paths))
    if args.max_pdfs is not None:
        if args.max_pdfs <= 0:
            print("\nError: --max-pdfs must be a positive integer.")
            return 1
        if len(pdf_paths) > args.max_pdfs:
            print(f"\nLimiting to {args.max_pdfs} PDF file(s) (from {len(pdf_paths)} total)")
            pdf_paths = pdf_paths[: args.max_pdfs]

    docs_cache_id = compute_docs_cache_id(
        pdf_paths,
        recursive=not args.no_recursive,
        max_files=args.max_files,
    )
    docs_cache_path = docs_cache_dir / f"docs_{docs_cache_id}.jsonl.gz"
    docs_meta_path = meta_dir / f"docs_{docs_cache_id}.json"

    # Load documents (prefer cache)
    if cache_enabled and docs_cache_path.exists() and not args.reprocess:
        print(f"\nLoading cached extracted documents: {docs_cache_path}")
        all_docs = load_documents_cache(docs_cache_path)
        print(f"  Loaded {len(all_docs)} cached document(s)")
    else:
        all_docs = []

        # If --max-pdfs is set, load only the selected PDF paths (avoid reading the entire folder).
        if args.max_pdfs is not None:
            print(f"\nLoading {len(pdf_paths)} selected PDF file(s)...")
            for pdf_path in pdf_paths:
                try:
                    loader = PyPDFLoader(str(pdf_path))
                    file_docs = loader.load()
                    all_docs.extend(file_docs)
                except Exception as e:
                    print(f"  Error loading {pdf_path}: {e}")
        else:
            # Load documents from all specified directories
            for input_dir in input_dirs:
                input_dir = Path(input_dir)
                if not input_dir.exists():
                    print(f"Warning: Directory not found: {input_dir}")
                    continue
                print(f"\nLoading documents from: {input_dir}")
                docs = load_documents(
                    str(input_dir),
                    file_types=args.file_types,
                    max_files=None,  # We'll limit after combining
                    recursive=not args.no_recursive,
                )
                all_docs.extend(docs)

            # Load individual files if specified
            if args.files:
                print("\nLoading individual files...")
                for file_path in args.files:
                    resolved_file_path = resolve_input_file(file_path, base_input_dir)
                    if not resolved_file_path.exists():
                        print(f"  Warning: File not found: {resolved_file_path}")
                        continue
                    try:
                        if resolved_file_path.suffix.lower() != ".pdf":
                            print(f"  Skipping non-PDF file: {resolved_file_path}")
                            continue
                        loader = PyPDFLoader(str(resolved_file_path))
                        file_docs = loader.load()
                        print(
                            f"  Loaded {len(file_docs)} document(s) from: {resolved_file_path.name}"
                        )
                        all_docs.extend(file_docs)
                    except Exception as e:
                        print(f"  Error loading {resolved_file_path}: {e}")

        # Deterministic ordering before limiting
        all_docs = sort_documents_deterministically(all_docs)

        # Apply max_files limit if specified
        if args.max_files and len(all_docs) > args.max_files:
            print(
                f"\nLimiting to {args.max_files} documents (from {len(all_docs)} total)"
            )
            all_docs = all_docs[: args.max_files]

        if cache_enabled:
            print(f"\nSaving extracted documents cache: {docs_cache_path}")
            save_documents_cache(all_docs, docs_cache_path)
            with open(docs_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "created_at": _utc_now_iso(),
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "docs_cache_id": docs_cache_id,
                        "input_dirs": [str(Path(d)) for d in input_dirs],
                        "explicit_files": args.files or [],
                        "recursive": not args.no_recursive,
                        "max_files": args.max_files,
                        "max_pdfs": args.max_pdfs,
                        "num_pdfs": len(pdf_paths),
                        "pdf_files": [str(p) for p in pdf_paths],
                        "num_documents": len(all_docs),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    if not all_docs:
        print("\nError: No documents found. Please check your input directory.")
        return 1

    print(f"\nTotal documents to process: {len(all_docs)}")

    # Setup LLM and embeddings
    try:
        generator_llm, generator_embeddings = setup_llm_and_embeddings(args.model, args.provider)
    except ValueError as e:
        print(f"\nError: {e}")
        return 1

    # Determine provider used (for cache key stability)
    provider_used = args.provider
    if provider_used == "auto":
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider_used = "azure"
        else:
            provider_used = "openai"

    embedding_id = (
        os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
        if provider_used == "azure"
        else "openai-default"
    )

    # Determine if we should add content embeddings (default: True)
    add_content_embeddings = not args.no_content_embeddings

    kg_cache_id = compute_kg_cache_id(
        docs_cache_id,
        provider=provider_used,
        llm_id=args.model,
        embedding_id=embedding_id,
        add_content_embeddings=add_content_embeddings,
    )
    kg_cache_path = kg_cache_dir / f"kg_{kg_cache_id}.json.gz"
    kg_meta_path = meta_dir / f"kg_{kg_cache_id}.json"
    personas_cache_path = personas_cache_dir / f"personas_{kg_cache_id}.json"

    # Load or build knowledge graph
    if cache_enabled and kg_cache_path.exists() and not args.reprocess:
        print(f"\nLoading cached knowledge graph: {kg_cache_path}")
        kg = load_knowledge_graph_cache(kg_cache_path)
        print(f"  Loaded KG: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
    else:
        kg = build_knowledge_graph(
            all_docs,
            generator_llm,
            generator_embeddings,
            add_content_embeddings=add_content_embeddings,
        )
        if cache_enabled:
            print(f"\nSaving knowledge graph cache: {kg_cache_path}")
            save_knowledge_graph_cache(kg, kg_cache_path)
            pipeline_id = PIPELINE_ID_BASE
            if add_content_embeddings:
                pipeline_id += "__content_embeddings"
            with open(kg_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "created_at": _utc_now_iso(),
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "pipeline_id": pipeline_id,
                        "ragas_version": getattr(ragas, "__version__", "unknown"),
                        "docs_cache_id": docs_cache_id,
                        "kg_cache_id": kg_cache_id,
                        "provider": provider_used,
                        "llm_id": args.model,
                        "embedding_id": embedding_id,
                        "add_content_embeddings": add_content_embeddings,
                        "num_nodes": len(kg.nodes),
                        "num_relationships": len(kg.relationships),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    # Load or build personas
    personas: Optional[List[Persona]] = None
    if cache_enabled and personas_cache_path.exists() and not args.reprocess:
        print(f"\nLoading cached personas: {personas_cache_path}")
        personas = load_personas_cache(personas_cache_path)
        print(f"  Loaded {len(personas)} persona(s)")
    else:
        print("\nGenerating personas (will be cached)...")
        personas = generate_personas_from_kg(kg=kg, llm=generator_llm, num_personas=3)
        if cache_enabled:
            save_personas_cache(personas, personas_cache_path)
            print(f"  Saved personas cache: {personas_cache_path}")

    # Query generation guidance (standalone corpus queries vs document-internal queries)
    llm_context: Optional[str] = None
    if args.standalone_queries:
        llm_context = build_corpus_llm_context(corpus_size_hint=args.corpus_size_hint)
        extra = str(args.query_llm_context or "").strip()
        if extra:
            llm_context = llm_context.rstrip() + "\n" + extra + "\n"
        print("\nStandalone query generation: enabled (corpus-level queries)")
    else:
        extra = str(args.query_llm_context or "").strip()
        llm_context = extra if extra else None
        print("\nStandalone query generation: disabled (RAGAS defaults)")

    query_distribution = build_query_distribution_for_pipeline(
        generator_llm,
        kg,
        standalone_queries=args.standalone_queries,
        llm_context=llm_context,
    )

    # Generate testset from the (cached) knowledge graph
    testset = generate_testset_from_knowledge_graph(
        kg,
        generator_llm,
        generator_embeddings,
        testset_size=args.testset_size,
        personas=personas,
        llm_context=llm_context,
        query_distribution=query_distribution,
    )

    # Quick sanity check: highlight any still-referential queries.
    if args.standalone_queries:
        _warn_on_referential_queries(testset)

    # Mine hard negatives if requested
    hard_negatives = None
    if args.hard_negatives:
        # Use the embeddings wrapper directly for query embedding.
        # (RAGAS LangchainEmbeddingsWrapper stores the underlying LC embedder at `.embeddings`,
        # but we don't need to unwrap it here as the wrapper exposes `embed_query()`.)
        raw_embedding_model = generator_embeddings
        judge_llm = getattr(generator_llm, "langchain_llm", None)

        testset_df = testset.to_pandas()
        hard_negatives = mine_hard_negatives_for_testset(
            testset_df,
            kg,
            all_docs,
            raw_embedding_model,
            judge_llm=judge_llm,
            num_bm25_negatives=args.num_bm25_negatives,
            num_embedding_negatives=args.num_embedding_negatives,
        )

    # Save results (pass docs for source file mapping)
    df = save_testset(
        testset,
        args.output,
        formats=args.output_formats,
        docs=all_docs,
        hard_negatives=hard_negatives,
    )

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Total samples generated: {len(df)}")
    if args.hard_negatives:
        print("Hard negatives included (file + page)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
