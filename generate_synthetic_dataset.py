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
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Iterator
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

# RAGAS synthesizers/prompts (used for PDF-profile-aware query generation)
from ragas.dataset_schema import SingleTurnSample
from ragas.testset.synthesizers.single_hop.prompts import QueryCondition as SingleHopQueryCondition
from ragas.testset.synthesizers.multi_hop.prompts import QueryConditions as MultiHopQueryConditions
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer

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
        nodes, relationships = await super().split(node)

        # IMPORTANT: Propagate source metadata into CHUNK nodes.
        #
        # RAGAS's stock HeadlineSplitter creates CHUNK nodes with only `page_content`,
        # which loses the originating PDF identity (source path, page index, etc).
        #
        # We copy `document_metadata` from the parent DOCUMENT node so downstream
        # query generation can attach PDF-level context (profiles) per chunk.
        parent_md = node.get_property("document_metadata")
        if parent_md is not None:
            for n in nodes:
                if n.type == NodeType.CHUNK and n.get_property("document_metadata") is None:
                    # Shallow copy is sufficient (metadata is JSON-like dict from LC docs)
                    if isinstance(parent_md, dict):
                        n.add_property("document_metadata", dict(parent_md))
                    else:
                        n.add_property("document_metadata", parent_md)

        return nodes, relationships


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


# ---------------------------
# PDF Profile caching/config
# ---------------------------
PDF_PROFILE_SCHEMA_VERSION = 1
DEFAULT_PDF_PROFILE_MAX_PAGES = 3
DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE = 2500
DEFAULT_PDF_PROFILE_MAX_PROFILE_CHARS = 1800  # Cap injected profile text per PDF


def compute_pdf_profile_cache_id(
    pdf_path: Path,
    *,
    provider: str,
    llm_id: str,
    max_pages: int,
    max_chars_per_page: int,
) -> str:
    payload = {
        "schema_version": PDF_PROFILE_SCHEMA_VERSION,
        "kind": "pdf_profile",
        "provider": provider,
        "llm_id": llm_id,
        "max_pages": int(max_pages),
        "max_chars_per_page": int(max_chars_per_page),
        "pdf": _file_fingerprint(pdf_path),
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


# ---------------------------
# SQLite PDF page store (single-table, corpus-derived)
# ---------------------------
PDF_STORE_SCHEMA_VERSION = 1
DEFAULT_PDF_STORE_DB_NAME = "pdf_page_store.sqlite"

_EXTRACTIVE_SUMMARY_MODEL = "extractive_v1"


def _iter_batched(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _compute_rel_path_for_store(pdf_path: Path, base_input_dir: Path) -> str:
    """
    Return a stable corpus-relative path when possible; otherwise fallback to absolute.
    """
    resolved = pdf_path.expanduser().resolve()
    try:
        return str(resolved.relative_to(base_input_dir.expanduser().resolve()))
    except Exception:
        return str(resolved)


def _compute_source_path_from_rel_path(rel_path: str, base_input_dir: Path) -> str:
    """
    Convert a stored `rel_path` back into an absolute source path string.

    - If rel_path is already absolute, it is used as-is (resolved best-effort).
    - Otherwise, it is interpreted relative to `base_input_dir`.
    """
    try:
        p = Path(rel_path).expanduser()
        if p.is_absolute():
            return str(p.resolve())
    except Exception:
        pass
    try:
        return str((base_input_dir.expanduser().resolve() / rel_path).resolve())
    except Exception:
        return str(base_input_dir / rel_path)


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _extractive_summary(text: str, *, max_chars: int = 450) -> str:
    """
    Cheap, deterministic "summary" for a page: whitespace-normalized leading snippet.

    This is intentionally non-LLM so the PDF store can be built offline.
    """
    if not text:
        return ""
    t = str(text).replace("\x00", " ").strip()
    if not t:
        return ""
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= max_chars:
        return t
    cut = t[: max_chars].rstrip()
    # Avoid chopping mid-word when possible
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return cut


def open_pdf_page_store(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL is faster for large bulk inserts.
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except Exception:
        # Some environments disallow changing PRAGMA; safe to ignore.
        pass
    return conn


def init_pdf_page_store(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_page_store (
          id INTEGER PRIMARY KEY,
          pdf_sha256 TEXT NOT NULL,
          rel_path TEXT NOT NULL,
          filename TEXT NOT NULL,
          file_type TEXT NOT NULL DEFAULT 'pdf' CHECK(file_type = 'pdf'),
          size_bytes INTEGER NOT NULL,
          mtime_ns INTEGER,
          page_number INTEGER NOT NULL CHECK(page_number >= 1),
          doc_content TEXT NOT NULL,
          content_sha256 TEXT NOT NULL,
          content_chars INTEGER NOT NULL,
          summary TEXT,
          summary_model TEXT,
          embedding_f32 BLOB,
          embedding_model TEXT,
          embedding_dims INTEGER,
          pdf_profile_json TEXT,
          pdf_profile_model TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          updated_at TEXT,
          UNIQUE(pdf_sha256, page_number)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS pdf_page_store_path_page_idx ON pdf_page_store(rel_path, page_number)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS pdf_page_store_pdfsha_idx ON pdf_page_store(pdf_sha256)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS pdf_profile_one_per_pdf_idx "
        "ON pdf_page_store(pdf_sha256) WHERE pdf_profile_json IS NOT NULL"
    )
    try:
        conn.execute(f"PRAGMA user_version = {int(PDF_STORE_SCHEMA_VERSION)}")
    except Exception:
        pass
    conn.commit()


def pdf_store_needs_refresh(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    size_bytes: int,
    mtime_ns: int,
) -> bool:
    row = conn.execute(
        """
        SELECT
          MIN(size_bytes), MAX(size_bytes),
          MIN(mtime_ns),  MAX(mtime_ns),
          COUNT(*),
          MIN(page_number), MAX(page_number)
        FROM pdf_page_store
        WHERE rel_path = ?
        """,
        (rel_path,),
    ).fetchone()
    if not row:
        return True

    min_size, max_size, min_mtime, max_mtime, count_rows, min_page, max_page = row
    count_rows = int(count_rows or 0)
    if count_rows <= 0:
        return True

    if min_size is None or max_size is None or int(min_size) != int(max_size):
        return True
    if int(min_size) != int(size_bytes):
        return True

    if min_mtime is None or max_mtime is None or int(min_mtime) != int(max_mtime):
        return True
    if int(min_mtime) != int(mtime_ns):
        return True

    # Heuristic integrity check: pages should be contiguous 1..N
    if min_page is None or max_page is None:
        return True
    if int(min_page) != 1:
        return True
    if int(max_page) != count_rows:
        return True

    return False


def pdf_store_needs_embeddings(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    embedding_model_tag: Optional[str],
) -> bool:
    if not embedding_model_tag:
        # If we don't know the model, only check for missing blobs.
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM pdf_page_store
            WHERE rel_path = ?
              AND (embedding_f32 IS NULL OR embedding_dims IS NULL)
            """,
            (rel_path,),
        ).fetchone()
        return bool(row and int(row[0] or 0) > 0)

    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM pdf_page_store
        WHERE rel_path = ?
          AND (
            embedding_f32 IS NULL OR
            embedding_dims IS NULL OR
            embedding_model IS NULL OR
            embedding_model = '' OR
            embedding_model != ?
          )
        """,
        (rel_path, str(embedding_model_tag)),
    ).fetchone()
    return bool(row and int(row[0] or 0) > 0)


def pdf_store_needs_profile(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    profile_model_tag: Optional[str],
) -> bool:
    row = conn.execute(
        """
        SELECT pdf_profile_json, pdf_profile_model
        FROM pdf_page_store
        WHERE rel_path = ? AND pdf_profile_json IS NOT NULL
        LIMIT 1
        """,
        (rel_path,),
    ).fetchone()
    if not row:
        return True
    blob, model = row
    if not isinstance(blob, str) or not blob.strip():
        return True
    if profile_model_tag:
        if not isinstance(model, str) or model.strip() != str(profile_model_tag):
            return True
    return False


def pdf_store_fill_missing_summaries_for_pdf(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    summary_model: str = _EXTRACTIVE_SUMMARY_MODEL,
) -> int:
    rows = conn.execute(
        """
        SELECT id, doc_content
        FROM pdf_page_store
        WHERE rel_path = ?
          AND (summary IS NULL OR summary = '' OR summary_model IS NULL OR summary_model = '')
        ORDER BY page_number ASC
        """,
        (rel_path,),
    ).fetchall()
    if not rows:
        return 0

    now = _utc_now_iso()
    updated = 0
    with conn:
        for row_id, content in rows:
            s = _extractive_summary(str(content or ""))
            conn.execute(
                "UPDATE pdf_page_store SET summary = ?, summary_model = ?, updated_at = ? WHERE id = ?",
                (s, str(summary_model), now, int(row_id)),
            )
            updated += 1
    return updated


def pdf_store_compute_embeddings_for_pdf(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    embedding_model,
    embedding_model_tag: str,
) -> int:
    rows = conn.execute(
        """
        SELECT id, doc_content
        FROM pdf_page_store
        WHERE rel_path = ?
          AND (
            embedding_f32 IS NULL OR
            embedding_dims IS NULL OR
            embedding_model IS NULL OR
            embedding_model = '' OR
            embedding_model != ?
          )
        ORDER BY page_number ASC
        """,
        (rel_path, str(embedding_model_tag)),
    ).fetchall()
    if not rows:
        return 0

    ids = [int(r[0]) for r in rows]
    texts = [str(r[1] or "") for r in rows]

    vecs: List[Optional[List[float]]] = []
    for batch in _iter_batched(texts, batch_size=64):
        try:
            batch_vecs = embedding_model.embed_documents(batch)  # type: ignore[attr-defined]
            if not isinstance(batch_vecs, list) or len(batch_vecs) != len(batch):
                raise ValueError("Unexpected embed_documents result shape")
            vecs.extend(batch_vecs)
        except Exception:
            for t in batch:
                try:
                    v = embedding_model.embed_query(t)  # type: ignore[attr-defined]
                    vecs.append(v if isinstance(v, list) else None)
                except Exception:
                    vecs.append(None)

    if len(vecs) != len(ids):
        return 0

    now = _utc_now_iso()
    updated = 0
    with conn:
        for row_id, v in zip(ids, vecs):
            if not isinstance(v, list) or not v:
                continue
            arr = np.asarray(v, dtype=np.float32)
            conn.execute(
                """
                UPDATE pdf_page_store
                SET embedding_f32 = ?, embedding_model = ?, embedding_dims = ?, updated_at = ?
                WHERE id = ?
                """,
                (arr.tobytes(), str(embedding_model_tag), int(arr.shape[0]), now, int(row_id)),
            )
            updated += 1
    return updated


def upsert_pdf_into_store(
    conn: sqlite3.Connection,
    *,
    pdf_path: Path,
    base_input_dir: Path,
    embedding_model=None,
    embedding_model_id: Optional[str] = None,
    compute_embeddings: bool = False,
    reprocess: bool = False,
) -> int:
    """
    Extract all pages from `pdf_path` via PyPDFLoader and upsert them into `pdf_page_store`.

    Returns: number of pages stored.
    """
    resolved = pdf_path.expanduser().resolve()
    st = resolved.stat()
    rel_path = _compute_rel_path_for_store(resolved, base_input_dir)
    filename = resolved.name
    size_bytes = int(st.st_size)
    mtime_ns = int(st.st_mtime_ns)

    if reprocess:
        conn.execute("DELETE FROM pdf_page_store WHERE rel_path = ?", (rel_path,))

    pdf_sha256 = _sha256_file(resolved)

    loader = PyPDFLoader(str(resolved))
    docs = loader.load()
    # Deterministic ordering by page when available
    docs = sorted(
        docs,
        key=lambda d: d.metadata.get("page")
        if isinstance(d.metadata.get("page"), int)
        else 10**9,
    )

    embeddings: Optional[List[Optional[List[float]]]] = None
    if compute_embeddings and embedding_model is not None:
        texts = [str(d.page_content or "") for d in docs]
        embeddings = []
        # Conservative batching; can be tuned later.
        for batch in _iter_batched(texts, batch_size=64):
            try:
                batch_vecs = embedding_model.embed_documents(batch)  # type: ignore[attr-defined]
                if not isinstance(batch_vecs, list) or len(batch_vecs) != len(batch):
                    raise ValueError("Unexpected embed_documents result shape")
                embeddings.extend(batch_vecs)
            except Exception:
                # Fallback: per-text embed_query to avoid failing the whole build.
                for t in batch:
                    try:
                        v = embedding_model.embed_query(t)  # type: ignore[attr-defined]
                        embeddings.append(v if isinstance(v, list) else None)
                    except Exception:
                        embeddings.append(None)

        # Ensure alignment
        if len(embeddings) != len(docs):
            embeddings = None

    now = _utc_now_iso()
    stored = 0

    # Upsert rows; preserve optional derived columns when not recomputed.
    with conn:
        for idx, doc in enumerate(docs):
            page0 = doc.metadata.get("page")
            page_number = int(page0) + 1 if isinstance(page0, int) else (idx + 1)

            content = str(doc.page_content or "")
            content_sha256 = _sha256_hex(content)
            content_chars = int(len(content))
            summary = _extractive_summary(content)
            summary_model = _EXTRACTIVE_SUMMARY_MODEL

            md = doc.metadata or {}
            if not isinstance(md, dict):
                md = {}
            md = dict(md)
            # Make sure metadata matches the standard PyPDFLoader contract:
            md["source"] = str(resolved)
            md["page"] = int(page_number - 1)  # 0-indexed for compatibility
            metadata_json = json.dumps(md, ensure_ascii=False)

            emb_blob = None
            emb_dims = None
            emb_model = None
            if embeddings is not None:
                v = embeddings[idx]
                if isinstance(v, list) and v:
                    arr = np.asarray(v, dtype=np.float32)
                    emb_blob = arr.tobytes()
                    emb_dims = int(arr.shape[0])
                    emb_model = str(embedding_model_id or "")

            conn.execute(
                """
                INSERT INTO pdf_page_store (
                  pdf_sha256,
                  rel_path,
                  filename,
                  file_type,
                  size_bytes,
                  mtime_ns,
                  page_number,
                  doc_content,
                  content_sha256,
                  content_chars,
                  summary,
                  summary_model,
                  embedding_f32,
                  embedding_model,
                  embedding_dims,
                  metadata_json,
                  updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pdf_sha256, page_number) DO UPDATE SET
                  rel_path = excluded.rel_path,
                  filename = excluded.filename,
                  size_bytes = excluded.size_bytes,
                  mtime_ns = excluded.mtime_ns,
                  doc_content = excluded.doc_content,
                  content_sha256 = excluded.content_sha256,
                  content_chars = excluded.content_chars,
                  summary = excluded.summary,
                  summary_model = excluded.summary_model,
                  embedding_f32 = COALESCE(excluded.embedding_f32, pdf_page_store.embedding_f32),
                  embedding_model = COALESCE(excluded.embedding_model, pdf_page_store.embedding_model),
                  embedding_dims = COALESCE(excluded.embedding_dims, pdf_page_store.embedding_dims),
                  metadata_json = excluded.metadata_json,
                  updated_at = excluded.updated_at
                """,
                (
                    pdf_sha256,
                    rel_path,
                    filename,
                    "pdf",
                    size_bytes,
                    mtime_ns,
                    page_number,
                    content,
                    content_sha256,
                    content_chars,
                    summary,
                    summary_model,
                    emb_blob,
                    emb_model,
                    emb_dims,
                    metadata_json,
                    now,
                ),
            )
            stored += 1

    return stored


def load_documents_from_store(
    conn: sqlite3.Connection,
    *,
    base_input_dir: Path,
    pdf_paths: List[Path],
    max_files: Optional[int],
) -> List[Document]:
    rel_paths = [_compute_rel_path_for_store(p, base_input_dir) for p in pdf_paths]
    docs: List[Document] = []

    # SQLite has a parameter limit; chunk IN queries.
    for chunk in _iter_batched(rel_paths, batch_size=800):
        placeholders = ",".join(["?"] * len(chunk))
        rows = conn.execute(
            f"""
            SELECT rel_path, page_number, doc_content, metadata_json
            FROM pdf_page_store
            WHERE rel_path IN ({placeholders})
            ORDER BY rel_path ASC, page_number ASC
            """,
            tuple(chunk),
        ).fetchall()

        for rel_path, page_number, doc_content, metadata_json in rows:
            md: Dict[str, Any] = {}
            try:
                if isinstance(metadata_json, str) and metadata_json.strip():
                    parsed = json.loads(metadata_json)
                    if isinstance(parsed, dict):
                        md = parsed
            except Exception:
                md = {}

            source_path = _compute_source_path_from_rel_path(str(rel_path), base_input_dir)
            md["source"] = source_path
            if page_number is not None:
                try:
                    md["page"] = int(page_number) - 1
                except Exception:
                    pass

            docs.append(Document(page_content=str(doc_content or ""), metadata=md))

    # Deterministic ordering before limiting (matches file-loader path)
    docs = sort_documents_deterministically(docs)
    if max_files and len(docs) > max_files:
        docs = docs[: max_files]
    return docs


def _store_set_pdf_profile(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    profile: Dict[str, Any],
    pdf_profile_model: str,
) -> None:
    """
    Store a single profile payload on page 1 for this PDF (enforced by partial unique index).
    """
    now = _utc_now_iso()
    # Ensure we don't violate the "one profile per PDF" index if an older row had the profile.
    with conn:
        conn.execute(
            "UPDATE pdf_page_store SET pdf_profile_json = NULL, pdf_profile_model = NULL WHERE rel_path = ?",
            (rel_path,),
        )
        conn.execute(
            """
            UPDATE pdf_page_store
            SET pdf_profile_json = ?, pdf_profile_model = ?, updated_at = ?
            WHERE rel_path = ? AND page_number = 1
            """,
            (json.dumps(profile, ensure_ascii=False), str(pdf_profile_model), now, rel_path),
        )


def load_pdf_profiles_from_store(
    conn: sqlite3.Connection,
    *,
    pdf_paths: List[Path],
    base_input_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """
    Load profiles from SQLite and return a map: absolute source_path -> profile dict.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for pdf_path in pdf_paths:
        rel_path = _compute_rel_path_for_store(pdf_path, base_input_dir)
        row = conn.execute(
            """
            SELECT pdf_profile_json
            FROM pdf_page_store
            WHERE rel_path = ? AND pdf_profile_json IS NOT NULL
            LIMIT 1
            """,
            (rel_path,),
        ).fetchone()
        if not row:
            continue
        blob = row[0]
        if not isinstance(blob, str) or not blob.strip():
            continue
        try:
            prof = json.loads(blob)
        except Exception:
            continue
        if not isinstance(prof, dict):
            continue

        resolved = pdf_path.expanduser().resolve()
        source_path = str(resolved)
        # Refresh path-derived fields (corpus_path/folder_hints) in case input-dir moved
        prof["source_path"] = source_path
        prof["filename"] = resolved.name
        prof["filename_stem"] = resolved.stem
        prof["corpus_path"] = _compute_corpus_path(resolved, base_input_dir)
        folder_hints: List[str] = []
        if prof.get("corpus_path"):
            try:
                p = Path(str(prof["corpus_path"]))
                folder_hints = list(p.parts[:-1])
            except Exception:
                folder_hints = []
        if not folder_hints:
            parent_name = resolved.parent.name
            if parent_name:
                folder_hints = [parent_name]
        prof["folder_hints"] = folder_hints

        out[source_path] = prof

    return out


def generate_pdf_profile_from_store(
    conn: sqlite3.Connection,
    *,
    pdf_path: Path,
    base_input_dir: Path,
    llm: Any,
    provider: str,
    llm_id: str,
    max_pages: int,
    max_chars_per_page: int,
) -> Optional[Dict[str, Any]]:
    """
    Generate a PDF profile using the already-extracted pages in SQLite (no PDF re-read),
    then store it back into SQLite.
    """
    resolved = pdf_path.expanduser().resolve()
    rel_path = _compute_rel_path_for_store(resolved, base_input_dir)

    # Build excerpt from the first N stored pages
    rows = conn.execute(
        """
        SELECT page_number, doc_content
        FROM pdf_page_store
        WHERE rel_path = ?
        ORDER BY page_number ASC
        LIMIT ?
        """,
        (rel_path, int(max_pages)),
    ).fetchall()

    parts: List[str] = []
    for page_number, doc_content in rows:
        page_label = f"{int(page_number)}" if page_number is not None else "?"
        snippet = _truncate_for_profile(str(doc_content or ""), max_chars=int(max_chars_per_page))
        if snippet.strip():
            parts.append(f"--- PAGE {page_label} ---\n{snippet}")
    excerpt = "\n\n".join(parts).strip()

    prof = _generate_pdf_profile_with_llm(
        llm,
        pdf_path=resolved,
        base_input_dir=base_input_dir,
        excerpt=excerpt,
        provider=provider,
        llm_id=llm_id,
        max_pages=max_pages,
        max_chars_per_page=max_chars_per_page,
    )

    model_tag = f"{provider}:{llm_id}:p{int(max_pages)}:c{int(max_chars_per_page)}"
    _store_set_pdf_profile(conn, rel_path=rel_path, profile=prof, pdf_profile_model=model_tag)
    return prof


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


# ---------------------------
# PDF Profiles (LLM-based)
# ---------------------------
PDF_PROFILE_LLM_CONTEXT_RULES = (
    "You may receive one or more PDF_PROFILES below (PDF-level metadata).\n"
    "Use PDF_PROFILES ONLY to make the SEARCH QUERY more realistic in a large corpus and to pick a plausible user intent.\n"
    "CRITICAL:\n"
    "- Do NOT mention file paths, directories, '.pdf', or raw filenames in the final query.\n"
    "- Do NOT copy filename stems verbatim if they contain underscores/odd punctuation; use them only as hints.\n"
    "- Prefer identifiers present in the provided context excerpt(s). You MAY use folder/collection hints or a cleaned-up\n"
    "  title guess from the profile to disambiguate, but keep it natural and do not rely on it for the answer.\n"
    "- The ANSWER must be supported ONLY by the provided context excerpt(s), not by PDF_PROFILES.\n"
)


def _safe_resolve_path_str(path_like: Any) -> Optional[str]:
    """
    Best-effort canonicalization of a path-like value to an absolute, resolved string.
    """
    if path_like is None:
        return None
    s = str(path_like).strip()
    if not s:
        return None
    try:
        return str(Path(s).expanduser().resolve())
    except Exception:
        return s


def _truncate_for_profile(text: str, *, max_chars: int) -> str:
    """
    Keep PDF profile excerpts/snippets bounded in size.
    """
    if not text:
        return ""
    t = str(text).replace("\x00", " ").strip()
    if len(t) <= max_chars:
        return t
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return f"{t[:head]}\n...\n{t[-tail:]}"


def _group_docs_by_source(docs: List[Document]) -> Dict[str, List[Document]]:
    out: Dict[str, List[Document]] = {}
    for d in docs:
        src = _safe_resolve_path_str(d.metadata.get("source"))
        if not src:
            continue
        out.setdefault(src, []).append(d)
    # Deterministic page ordering per PDF (if page numbers exist)
    for src, items in out.items():
        def _k(doc: Document):
            p = doc.metadata.get("page")
            return p if isinstance(p, int) else 10**9
        out[src] = sorted(items, key=_k)
    return out


def _build_pdf_excerpt_from_docs(
    docs_for_pdf: List[Document],
    *,
    max_pages: int,
    max_chars_per_page: int,
) -> str:
    if not docs_for_pdf or max_pages <= 0 or max_chars_per_page <= 0:
        return ""
    chosen = docs_for_pdf[: max_pages]
    parts: List[str] = []
    for doc in chosen:
        page0 = doc.metadata.get("page")
        page_label = f"{int(page0) + 1}" if isinstance(page0, int) else "?"
        snippet = _truncate_for_profile(doc.page_content or "", max_chars=max_chars_per_page)
        if not snippet.strip():
            continue
        parts.append(f"--- PAGE {page_label} ---\n{snippet}")
    return "\n\n".join(parts).strip()


def _compute_corpus_path(pdf_path: Path, base_input_dir: Path) -> Optional[str]:
    """
    Compute a stable, non-personal "corpus path" for prompting (relative to input dir when possible).
    """
    try:
        rel = pdf_path.resolve().relative_to(base_input_dir.resolve())
        return str(rel)
    except Exception:
        return None


def _format_pdf_profile_for_prompt(
    profile: Dict[str, Any],
    *,
    max_chars: int = DEFAULT_PDF_PROFILE_MAX_PROFILE_CHARS,
) -> str:
    """
    Convert a cached profile dict into a compact text block for prompt injection.
    """
    corpus_path = str(profile.get("corpus_path") or "").strip()
    filename_stem = str(profile.get("filename_stem") or "").strip()
    folder_hints = profile.get("folder_hints") or []
    folder_hints = [str(x) for x in folder_hints if str(x).strip()]

    llm_profile = profile.get("llm_profile") or {}
    if not isinstance(llm_profile, dict):
        llm_profile = {}

    def _get_list(key: str, limit: int = 8) -> List[str]:
        v = llm_profile.get(key)
        if not isinstance(v, list):
            return []
        out = []
        for x in v:
            xs = str(x).strip()
            if xs:
                out.append(xs)
            if len(out) >= limit:
                break
        return out

    title_guess = str(llm_profile.get("title_guess") or "").strip()
    doc_type = str(llm_profile.get("doc_type") or "").strip()
    summary = str(llm_profile.get("summary") or "").strip()
    topics = _get_list("topics", limit=8)
    key_entities = _get_list("key_entities", limit=8)
    likely_intents = _get_list("likely_user_intents", limit=6)

    lines: List[str] = []
    if corpus_path:
        lines.append(f"- corpus_path: {corpus_path}")
    if folder_hints:
        lines.append(f"- folder_hints: {', '.join(folder_hints[:5])}")
    if filename_stem:
        lines.append(f"- filename_stem: {filename_stem}")
    if title_guess:
        lines.append(f"- title_guess: {title_guess}")
    if doc_type:
        lines.append(f"- doc_type: {doc_type}")
    if summary:
        lines.append(f"- summary: {summary}")
    if topics:
        lines.append(f"- topics: {', '.join(topics)}")
    if key_entities:
        lines.append(f"- key_entities: {', '.join(key_entities)}")
    if likely_intents:
        lines.append(f"- likely_user_intents: {', '.join(likely_intents)}")

    return _truncate_for_profile("\n".join(lines), max_chars=max_chars).strip()


def _generate_pdf_profile_with_llm(
    llm: Any,
    *,
    pdf_path: Path,
    base_input_dir: Path,
    excerpt: str,
    provider: str,
    llm_id: str,
    max_pages: int,
    max_chars_per_page: int,
) -> Dict[str, Any]:
    """
    Generate a compact PDF-level profile using an LLM.

    Returns a profile dict (always includes path + filename hints), with an optional
    `llm_profile` payload if generation succeeded.
    """
    resolved = pdf_path.resolve()
    source_path = str(resolved)
    corpus_path = _compute_corpus_path(resolved, base_input_dir)
    filename = resolved.name
    filename_stem = resolved.stem
    folder_hints: List[str] = []
    if corpus_path:
        try:
            p = Path(corpus_path)
            folder_hints = list(p.parts[:-1])
        except Exception:
            folder_hints = []
    # If we couldn't compute a relative path (or the file sits directly under input-dir),
    # still provide at least the immediate parent folder as a hint (e.g., "Claires").
    if not folder_hints:
        try:
            parent_name = resolved.parent.name
            if parent_name:
                folder_hints = [parent_name]
        except Exception:
            pass

    profile_id = compute_pdf_profile_cache_id(
        resolved,
        provider=provider,
        llm_id=llm_id,
        max_pages=max_pages,
        max_chars_per_page=max_chars_per_page,
    )

    base_profile: Dict[str, Any] = {
        "schema_version": PDF_PROFILE_SCHEMA_VERSION,
        "profile_id": profile_id,
        "created_at": _utc_now_iso(),
        "provider": provider,
        "llm_id": llm_id,
        # Always store absolute resolved path for stable lookup
        "source_path": source_path,
        # Prompt-safe path (relative to input dir) when available
        "corpus_path": corpus_path,
        "filename": filename,
        "filename_stem": filename_stem,
        "folder_hints": folder_hints,
    }

    if llm is None:
        base_profile["llm_profile"] = {
            "title_guess": "",
            "doc_type": "",
            "summary": "",
            "topics": [],
            "key_entities": [],
            "likely_user_intents": [],
        }
        base_profile["llm_profile_error"] = "llm_unavailable"
        return base_profile

    system = (
        "You are a careful document profiler.\n"
        "Given limited excerpts from a PDF plus filename/folder hints, produce a compact high-level profile.\n"
        "The profile will be used ONLY to help generate realistic user search queries across a large corpus.\n"
        "Do NOT include page numbers, file paths, or raw filenames in any output fields except where explicitly requested.\n"
        "Return ONLY valid JSON.\n"
    )

    # Include the filename/folder hints explicitly (as the user requested)
    user = (
        "Create a PDF profile in JSON.\n\n"
        f"PDF source_path (absolute): {source_path}\n"
        f"PDF corpus_path (relative, if available): {corpus_path or ''}\n"
        f"PDF filename: {filename}\n"
        f"PDF filename_stem: {filename_stem}\n"
        f"PDF folder_hints: {folder_hints}\n\n"
        "EXCERPT (may be partial):\n"
        f"{excerpt}\n\n"
        "Return JSON in this exact schema:\n"
        "{"
        "\"title_guess\":\"<string>\","
        "\"doc_type\":\"<string>\","
        "\"summary\":\"<1-2 sentences>\","
        "\"topics\":[\"<string>\",...],"
        "\"key_entities\":[\"<string>\",...],"
        "\"likely_user_intents\":[\"<string>\",...],"
        "\"confidence\":\"high|medium|low\""
        "}"
    )

    llm_profile: Dict[str, Any] = {}
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = getattr(resp, "content", None)
        content = content if isinstance(content, str) else str(resp)
        data = _extract_json_object(content)
        if isinstance(data, dict):
            llm_profile = data
        else:
            base_profile["llm_profile_error"] = "json_parse_failed"
    except Exception as e:
        base_profile["llm_profile_error"] = f"llm_invoke_failed: {e}"

    # Normalize to safe JSON-ish types
    def _as_list(v: Any, limit: int = 12) -> List[str]:
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for x in v:
            xs = str(x).strip()
            if xs:
                out.append(xs)
            if len(out) >= limit:
                break
        return out

    base_profile["llm_profile"] = {
        "title_guess": str(llm_profile.get("title_guess") or "").strip(),
        "doc_type": str(llm_profile.get("doc_type") or "").strip(),
        "summary": str(llm_profile.get("summary") or "").strip(),
        "topics": _as_list(llm_profile.get("topics")),
        "key_entities": _as_list(llm_profile.get("key_entities")),
        "likely_user_intents": _as_list(llm_profile.get("likely_user_intents"), limit=8),
        "confidence": str(llm_profile.get("confidence") or "").strip().lower(),
    }
    return base_profile


def _load_pdf_profile_cache(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_pdf_profile_cache(profile: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def build_pdf_profiles(
    *,
    pdf_paths: List[Path],
    docs: List[Document],
    base_input_dir: Path,
    llm: Any,
    provider: str,
    llm_id: str,
    profiles_dir: Path,
    cache_enabled: bool,
    reprocess: bool,
    max_pages: int = DEFAULT_PDF_PROFILE_MAX_PAGES,
    max_chars_per_page: int = DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
) -> Dict[str, Dict[str, Any]]:
    """
    Build/load per-PDF profiles and return a map: source_path -> profile dict.
    """
    docs_by_source = _group_docs_by_source(docs)

    profiles: Dict[str, Dict[str, Any]] = {}
    total = len(pdf_paths)
    if total == 0:
        return profiles

    print("\nBuilding PDF profiles (LLM-based; cached)...")
    print(f"  PDFs to profile: {total}")
    print(
        f"  Profile excerpt: first {max_pages} page(s), up to {max_chars_per_page} chars per page"
    )

    for i, pdf_path in enumerate(pdf_paths, start=1):
        resolved = pdf_path.resolve()
        source_path = str(resolved)
        profile_cache_id = compute_pdf_profile_cache_id(
            resolved,
            provider=provider,
            llm_id=llm_id,
            max_pages=max_pages,
            max_chars_per_page=max_chars_per_page,
        )
        cache_path = profiles_dir / f"profile_{profile_cache_id}.json"

        if cache_enabled and cache_path.exists() and not reprocess:
            cached = _load_pdf_profile_cache(cache_path)
            if isinstance(cached, dict):
                # Refresh path-derived fields (corpus_path/folder_hints) in case input-dir changes
                cached["source_path"] = source_path
                cached["filename"] = resolved.name
                cached["filename_stem"] = resolved.stem
                cached["corpus_path"] = _compute_corpus_path(resolved, base_input_dir)
                folder_hints: List[str] = []
                if cached.get("corpus_path"):
                    try:
                        p = Path(str(cached["corpus_path"]))
                        folder_hints = list(p.parts[:-1])
                    except Exception:
                        folder_hints = []
                if not folder_hints:
                    parent_name = resolved.parent.name
                    if parent_name:
                        folder_hints = [parent_name]
                cached["folder_hints"] = folder_hints
                profiles[source_path] = cached
                continue

        docs_for_pdf = docs_by_source.get(source_path, [])
        excerpt = _build_pdf_excerpt_from_docs(
            docs_for_pdf, max_pages=max_pages, max_chars_per_page=max_chars_per_page
        )

        prof = _generate_pdf_profile_with_llm(
            llm,
            pdf_path=resolved,
            base_input_dir=base_input_dir,
            excerpt=excerpt,
            provider=provider,
            llm_id=llm_id,
            max_pages=max_pages,
            max_chars_per_page=max_chars_per_page,
        )
        profiles[source_path] = prof

        if cache_enabled:
            _save_pdf_profile_cache(prof, cache_path)

        # Light progress for large corpora
        if i == 1 or i == total or (i % 25 == 0):
            print(f"  Profiled {i}/{total} PDFs")

    return profiles


def build_llm_context_with_pdf_profiles(
    *,
    base_llm_context: Optional[str],
    profiles: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Combine the global llm_context (corpus guidance) with one or more PDF profile blocks.
    """
    blocks: List[str] = []
    if base_llm_context and str(base_llm_context).strip():
        blocks.append(str(base_llm_context).strip())

    # Always include the rules when injecting profiles
    blocks.append(PDF_PROFILE_LLM_CONTEXT_RULES.strip())

    rendered = []
    for p in profiles:
        rendered_block = _format_pdf_profile_for_prompt(p, max_chars=DEFAULT_PDF_PROFILE_MAX_PROFILE_CHARS)
        if rendered_block:
            rendered.append(rendered_block)
    if rendered:
        blocks.append("PDF_PROFILES:\n" + "\n\n".join(rendered))

    out = "\n\n".join([b for b in blocks if b.strip()]).strip()
    return out or None


# --------------------------------
# PDF-profile-aware query synthesis
# --------------------------------
@dataclass
class PdfProfileSingleHopSpecificQuerySynthesizer(SingleHopSpecificQuerySynthesizer):
    pdf_profiles_by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _profiles_for_sources(self, sources: List[str], *, max_profiles: int = 1) -> List[Dict[str, Any]]:
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for s in sources:
            if not s or s in seen:
                continue
            seen.add(s)
            p = self.pdf_profiles_by_source.get(s)
            if isinstance(p, dict):
                out.append(p)
            if len(out) >= max_profiles:
                break
        return out

    async def _generate_sample(self, scenario, callbacks):  # type: ignore[override]
        reference_context = scenario.nodes[0].properties.get("page_content", "")

        md = scenario.nodes[0].get_property("document_metadata") or {}
        source = _safe_resolve_path_str(md.get("source")) if isinstance(md, dict) else None
        profiles = self._profiles_for_sources([source] if source else [], max_profiles=1)
        dynamic_llm_context = build_llm_context_with_pdf_profiles(
            base_llm_context=self.llm_context,
            profiles=profiles,
        ) if profiles else self.llm_context

        prompt_input = SingleHopQueryCondition(
            persona=scenario.persona,
            term=scenario.term,
            context=reference_context,
            query_length=scenario.length.value,
            query_style=scenario.style.value,
            llm_context=dynamic_llm_context,
        )
        response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=[reference_context],
            persona_name=getattr(scenario.persona, "name", None),
            query_style=getattr(scenario.style, "name", None),
            query_length=getattr(scenario.length, "name", None),
        )


@dataclass
class PdfProfileMultiHopAbstractQuerySynthesizer(MultiHopAbstractQuerySynthesizer):
    pdf_profiles_by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _profiles_for_nodes(self, nodes: List[Node], *, max_profiles: int = 2) -> List[Dict[str, Any]]:
        sources: List[str] = []
        for n in nodes:
            md = n.get_property("document_metadata") or {}
            src = _safe_resolve_path_str(md.get("source")) if isinstance(md, dict) else None
            if src:
                sources.append(src)
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for s in sources:
            if s in seen:
                continue
            seen.add(s)
            p = self.pdf_profiles_by_source.get(s)
            if isinstance(p, dict):
                out.append(p)
            if len(out) >= max_profiles:
                break
        return out

    async def _generate_sample(self, scenario, callbacks):  # type: ignore[override]
        reference_contexts = self.make_contexts(scenario)
        profiles = self._profiles_for_nodes(list(scenario.nodes), max_profiles=2)
        dynamic_llm_context = build_llm_context_with_pdf_profiles(
            base_llm_context=self.llm_context,
            profiles=profiles,
        ) if profiles else self.llm_context

        prompt_input = MultiHopQueryConditions(
            persona=scenario.persona,
            themes=scenario.combinations,
            context=reference_contexts,
            query_length=scenario.length.value,
            query_style=scenario.style.value,
            llm_context=dynamic_llm_context,
        )
        response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=reference_contexts,
        )


@dataclass
class PdfProfileMultiHopSpecificQuerySynthesizer(MultiHopSpecificQuerySynthesizer):
    pdf_profiles_by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _profiles_for_nodes(self, nodes: List[Node], *, max_profiles: int = 2) -> List[Dict[str, Any]]:
        sources: List[str] = []
        for n in nodes:
            md = n.get_property("document_metadata") or {}
            src = _safe_resolve_path_str(md.get("source")) if isinstance(md, dict) else None
            if src:
                sources.append(src)
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for s in sources:
            if s in seen:
                continue
            seen.add(s)
            p = self.pdf_profiles_by_source.get(s)
            if isinstance(p, dict):
                out.append(p)
            if len(out) >= max_profiles:
                break
        return out

    async def _generate_sample(self, scenario, callbacks):  # type: ignore[override]
        reference_contexts = self.make_contexts(scenario)
        profiles = self._profiles_for_nodes(list(scenario.nodes), max_profiles=2)
        dynamic_llm_context = build_llm_context_with_pdf_profiles(
            base_llm_context=self.llm_context,
            profiles=profiles,
        ) if profiles else self.llm_context

        prompt_input = MultiHopQueryConditions(
            persona=scenario.persona,
            themes=scenario.combinations,
            context=reference_contexts,
            query_length=scenario.length.value,
            query_style=scenario.style.value,
            llm_context=dynamic_llm_context,
        )
        response = await self.generate_query_reference_prompt.generate(
            data=prompt_input, llm=self.llm, callbacks=callbacks
        )
        return SingleTurnSample(
            user_input=response.query,
            reference=response.answer,
            reference_contexts=reference_contexts,
        )


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
    pdf_profiles_by_source: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Build a query_distribution that nudges RAGAS to produce *standalone, corpus-appropriate* queries.

    Returns:
      - None only when we are using fully stock RAGAS behavior (no standalone prompt patching,
        and no PDF-profile injection).
      - Otherwise, returns a query_distribution with compatible synthesizers.
    """
    use_pdf_profiles = bool(pdf_profiles_by_source)
    if not standalone_queries and not use_pdf_profiles:
        return None

    # Use RAGAS's compatibility filtering, then override the prompts.
    from ragas.testset.synthesizers import default_query_distribution
    from ragas.testset.synthesizers.single_hop.prompts import (
        QueryAnswerGenerationPrompt as SingleHopQueryAnswerGenerationPrompt,
    )
    from ragas.testset.synthesizers.multi_hop.prompts import (
        QueryAnswerGenerationPrompt as MultiHopQueryAnswerGenerationPrompt,
    )

    if use_pdf_profiles:
        # Build our PDF-profile-aware synthesizers and run the same compatibility filtering logic
        # as RAGAS's default_query_distribution.
        candidates = [
            PdfProfileSingleHopSpecificQuerySynthesizer(
                llm=llm,
                llm_context=llm_context,
                pdf_profiles_by_source=pdf_profiles_by_source or {},
            ),
            PdfProfileMultiHopAbstractQuerySynthesizer(
                llm=llm,
                llm_context=llm_context,
                pdf_profiles_by_source=pdf_profiles_by_source or {},
            ),
            PdfProfileMultiHopSpecificQuerySynthesizer(
                llm=llm,
                llm_context=llm_context,
                pdf_profiles_by_source=pdf_profiles_by_source or {},
            ),
        ]

        available = []
        for query in candidates:
            try:
                if query.get_node_clusters(kg):
                    available.append(query)
            except Exception as e:
                print(
                    f"Warning: Skipping {getattr(query, 'name', type(query).__name__)} due to unexpected error: {e}"
                )
                continue

        if not available:
            raise ValueError(
                "No compatible query synthesizers for the provided KnowledgeGraph."
            )

        qd = [(q, 1 / len(available)) for q in available]
    else:
        qd = default_query_distribution(llm, kg, llm_context)

    patched = []
    for synthesizer, prob in qd:
        if standalone_queries:
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
        "--pdf-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate per-PDF LLM profiles and inject them into query generation to improve realism "
            "(default: enabled). Disable with --no-pdf-profiles."
        ),
    )
    parser.add_argument(
        "--pdf-profile-max-pages",
        type=int,
        default=DEFAULT_PDF_PROFILE_MAX_PAGES,
        help=(
            "Max number of starting pages to include in each PDF profile excerpt (default: 3). "
            "Lower this to reduce cost."
        ),
    )
    parser.add_argument(
        "--pdf-profile-max-chars-per-page",
        type=int,
        default=DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
        help=(
            "Max characters per page included in PDF profile excerpt (default: 2500). "
            "Lower this to reduce cost."
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
        "--pdf-store-db",
        type=str,
        default=None,
        help=(
            "Optional SQLite DB path for a single-table PDF page store (pdf_page_store). "
            "Use this to pre-extract PDF pages once and load documents from SQLite on later runs."
        ),
    )
    parser.add_argument(
        "--pdf-store-use",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load extracted PDF pages from the SQLite PDF store instead of reading PDFs / processed/docs "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--pdf-store-build",
        action="store_true",
        help=(
            "Build/update the SQLite PDF store for the selected PDFs, then exit (unless --pdf-store-use is also set)."
        ),
    )
    parser.add_argument(
        "--pdf-store-auto-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --pdf-store-use is enabled, automatically extract and insert missing/stale PDFs into the store "
            "(default: enabled). Disable with --no-pdf-store-auto-build."
        ),
    )
    parser.add_argument(
        "--pdf-store-reprocess",
        action="store_true",
        help="Force re-extraction and overwrite of SQLite PDF store rows for selected PDFs.",
    )
    parser.add_argument(
        "--pdf-store-embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When building the SQLite PDF store, also compute and store page embeddings "
            "(default: disabled; requires embedding API access)."
        ),
    )
    parser.add_argument(
        "--pdf-store-profiles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When building the SQLite PDF store, also compute and store PDF profiles "
            "(default: disabled; requires LLM API access)."
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
    pdf_profiles_cache_dir = processed_dir / "pdf_profiles"
    meta_dir = processed_dir / "meta"

    # SQLite PDF store (single-table page store) — optional.
    pdf_store_conn: Optional[sqlite3.Connection] = None
    pdf_store_db_path: Optional[Path] = None
    if args.pdf_store_use or args.pdf_store_build:
        if args.pdf_store_db:
            pdf_store_db_path = Path(args.pdf_store_db).expanduser()
            if not pdf_store_db_path.is_absolute():
                pdf_store_db_path = (script_dir / pdf_store_db_path).resolve()
        else:
            pdf_store_db_path = (processed_dir / DEFAULT_PDF_STORE_DB_NAME).resolve()

        pdf_store_conn = open_pdf_page_store(pdf_store_db_path)
        init_pdf_page_store(pdf_store_conn)
        print(f"\nSQLite PDF store: enabled ({pdf_store_db_path})")

    if cache_enabled:
        docs_cache_dir.mkdir(parents=True, exist_ok=True)
        kg_cache_dir.mkdir(parents=True, exist_ok=True)
        personas_cache_dir.mkdir(parents=True, exist_ok=True)
        pdf_profiles_cache_dir.mkdir(parents=True, exist_ok=True)
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

    # Optional: build/sync the SQLite PDF store for the selected PDFs.
    #
    # - `--pdf-store-build` builds the store upfront (and exits unless --pdf-store-use is also set).
    # - When `--pdf-store-use` is enabled, we can auto-populate missing/stale PDFs to keep the
    #   pipeline fully functional even if the store is incomplete.
    if pdf_store_conn is not None and (args.pdf_store_build or (args.pdf_store_use and args.pdf_store_auto_build)):
        store_force = bool(args.pdf_store_reprocess or args.reprocess)

        # These flags apply both when explicitly building the store and when auto-building
        # missing/stale PDFs during a --pdf-store-use run.
        want_embeddings = bool(args.pdf_store_embeddings)
        want_profiles = bool(args.pdf_store_profiles)

        store_embedding_model = None
        store_embedding_model_id: Optional[str] = None
        store_profile_llm = None
        store_provider_used = args.provider
        if store_provider_used == "auto":
            if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
                store_provider_used = "azure"
            elif os.environ.get("OPENAI_API_KEY"):
                store_provider_used = "openai"
            else:
                store_provider_used = "auto"  # will error if models are needed

        if want_embeddings or want_profiles:
            try:
                tmp_llm, tmp_emb = setup_llm_and_embeddings(args.model, args.provider)
            except ValueError as e:
                print(f"\nError: {e}")
                return 1

            # Embedder/LLM handles (match later cache-id logic)
            store_embedding_model = tmp_emb if want_embeddings else None
            store_profile_llm = getattr(tmp_llm, "langchain_llm", None) if want_profiles else None
            store_embedding_model_id = (
                os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
                if store_provider_used == "azure"
                else "openai-default"
            )

        print("\nSyncing SQLite PDF store (pdf_page_store)...")
        refreshed_pdfs = 0
        upserted_pages = 0
        skipped = 0
        filled_summaries = 0
        filled_embeddings = 0
        filled_profiles = 0

        total = len(pdf_paths)
        for i, pdf_path in enumerate(pdf_paths, start=1):
            try:
                st = pdf_path.stat()
            except Exception:
                # If the file disappeared between scan and now, skip.
                continue

            rel_path = _compute_rel_path_for_store(pdf_path, base_input_dir)
            needs_extract = store_force or pdf_store_needs_refresh(
                pdf_store_conn,
                rel_path=rel_path,
                size_bytes=int(st.st_size),
                mtime_ns=int(st.st_mtime_ns),
            )

            try:
                if needs_extract:
                    pages = upsert_pdf_into_store(
                        pdf_store_conn,
                        pdf_path=pdf_path,
                        base_input_dir=base_input_dir,
                        embedding_model=store_embedding_model,
                        embedding_model_id=store_embedding_model_id,
                        compute_embeddings=bool(want_embeddings),
                        # IMPORTANT: if the PDF is missing/stale, replace rows for this rel_path
                        # to avoid accumulating multiple versions (same rel_path, different pdf_sha256).
                        reprocess=True,
                    )
                    upserted_pages += int(pages)
                    refreshed_pdfs += 1
                else:
                    skipped += 1

                # Always backfill missing extractive summaries (cheap + offline)
                filled_summaries += pdf_store_fill_missing_summaries_for_pdf(
                    pdf_store_conn, rel_path=rel_path
                )

                # Backfill embeddings (if requested) without re-extracting pages
                if want_embeddings and store_embedding_model is not None:
                    if pdf_store_needs_embeddings(
                        pdf_store_conn,
                        rel_path=rel_path,
                        embedding_model_tag=store_embedding_model_id,
                    ):
                        filled_embeddings += pdf_store_compute_embeddings_for_pdf(
                            pdf_store_conn,
                            rel_path=rel_path,
                            embedding_model=store_embedding_model,
                            embedding_model_tag=str(store_embedding_model_id or ""),
                        )

                # Backfill profiles (if requested) without re-extracting pages
                if want_profiles:
                    profile_tag = (
                        f"{store_provider_used}:{args.model}:p{int(args.pdf_profile_max_pages)}:"
                        f"c{int(args.pdf_profile_max_chars_per_page)}"
                    )
                    if pdf_store_needs_profile(
                        pdf_store_conn,
                        rel_path=rel_path,
                        profile_model_tag=profile_tag,
                    ):
                        prof = generate_pdf_profile_from_store(
                            pdf_store_conn,
                            pdf_path=pdf_path,
                            base_input_dir=base_input_dir,
                            llm=store_profile_llm,
                            provider=store_provider_used,
                            llm_id=args.model,
                            max_pages=int(args.pdf_profile_max_pages),
                            max_chars_per_page=int(args.pdf_profile_max_chars_per_page),
                        )
                        if prof is not None:
                            filled_profiles += 1

            except Exception as e:
                print(f"  Warning: Failed to store/backfill {pdf_path}: {e}")
                continue

            if i == 1 or i == total or (i % 25 == 0):
                print(
                    f"  Processed {i}/{total} PDFs "
                    f"(refreshed: {refreshed_pdfs}, skipped: {skipped}, "
                    f"summaries+={filled_summaries}, embeddings+={filled_embeddings}, profiles+={filled_profiles})"
                )

        print(
            f"SQLite PDF store sync complete: refreshed {refreshed_pdfs}/{total} PDFs, "
            f"upserted {upserted_pages} page(s), skipped {skipped} up-to-date. "
            f"Backfilled: summaries {filled_summaries}, embeddings {filled_embeddings}, profiles {filled_profiles}."
        )

        if args.pdf_store_build and not args.pdf_store_use:
            print("\nSQLite PDF store build complete (--pdf-store-build). Exiting.")
            return 0

    docs_cache_id = compute_docs_cache_id(
        pdf_paths,
        recursive=not args.no_recursive,
        max_files=args.max_files,
    )
    docs_cache_path = docs_cache_dir / f"docs_{docs_cache_id}.jsonl.gz"
    docs_meta_path = meta_dir / f"docs_{docs_cache_id}.json"

    # Load documents
    if args.pdf_store_use:
        if pdf_store_conn is None:
            print("\nError: --pdf-store-use was set but SQLite PDF store is not initialized.")
            return 1

        # If auto-build is disabled, verify that the store covers the selected PDFs.
        if not args.pdf_store_auto_build:
            missing_or_stale = 0
            for pdf_path in pdf_paths:
                try:
                    st = pdf_path.stat()
                except Exception:
                    continue
                rel_path = _compute_rel_path_for_store(pdf_path, base_input_dir)
                if pdf_store_needs_refresh(
                    pdf_store_conn,
                    rel_path=rel_path,
                    size_bytes=int(st.st_size),
                    mtime_ns=int(st.st_mtime_ns),
                ):
                    missing_or_stale += 1
            if missing_or_stale:
                print(
                    f"\nError: SQLite PDF store is missing/stale for {missing_or_stale} PDF(s). "
                    "Run with --pdf-store-build or enable --pdf-store-auto-build."
                )
                return 1

        print("\nLoading documents from SQLite PDF store...")
        all_docs = load_documents_from_store(
            pdf_store_conn,
            base_input_dir=base_input_dir,
            pdf_paths=pdf_paths,
            max_files=args.max_files,
        )
        print(f"  Loaded {len(all_docs)} document(s) from SQLite store")

    # Fallback: file-based extracted-doc cache
    elif cache_enabled and docs_cache_path.exists() and not args.reprocess:
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

    # PDF profiles (LLM-derived, per-PDF high-level metadata) — enabled by default.
    #
    # This is intentionally done early (right after loading documents and setting up the LLM),
    # so query generation can use PDF-level context even when the KG is loaded from cache.
    pdf_profiles_by_source: Dict[str, Dict[str, Any]] = {}
    if args.pdf_profiles:
        # If a SQLite store is enabled, prefer loading profiles from SQLite (when present),
        # and only generate missing ones via the existing file-based cache path.
        if pdf_store_conn is not None and not args.reprocess:
            pdf_profiles_by_source = load_pdf_profiles_from_store(
                pdf_store_conn,
                pdf_paths=pdf_paths,
                base_input_dir=base_input_dir,
            )

        missing_pdf_paths = [
            p for p in pdf_paths if str(p.expanduser().resolve()) not in pdf_profiles_by_source
        ]
        if missing_pdf_paths:
            raw_profile_llm = getattr(generator_llm, "langchain_llm", None)
            new_profiles = build_pdf_profiles(
                pdf_paths=missing_pdf_paths,
                docs=all_docs,
                base_input_dir=base_input_dir,
                llm=raw_profile_llm,
                provider=provider_used,
                llm_id=args.model,
                profiles_dir=pdf_profiles_cache_dir,
                cache_enabled=cache_enabled,
                reprocess=args.reprocess,
                max_pages=int(args.pdf_profile_max_pages),
                max_chars_per_page=int(args.pdf_profile_max_chars_per_page),
            )
            pdf_profiles_by_source.update(new_profiles)

            # Persist newly created profiles into SQLite for future runs.
            if pdf_store_conn is not None:
                model_tag = (
                    f"{provider_used}:{args.model}:p{int(args.pdf_profile_max_pages)}:"
                    f"c{int(args.pdf_profile_max_chars_per_page)}"
                )
                for source_path, prof in new_profiles.items():
                    try:
                        rel_path = _compute_rel_path_for_store(Path(source_path), base_input_dir)
                        _store_set_pdf_profile(
                            pdf_store_conn,
                            rel_path=rel_path,
                            profile=prof,
                            pdf_profile_model=model_tag,
                        )
                    except Exception:
                        continue

        print(f"PDF profiles: enabled ({len(pdf_profiles_by_source)} cached/generated)")
    else:
        print("PDF profiles: disabled (--no-pdf-profiles)")

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
        pdf_profiles_by_source=pdf_profiles_by_source if args.pdf_profiles else None,
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
