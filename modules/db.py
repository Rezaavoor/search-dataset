"""SQLite PDF page store operations (single-table, corpus-derived)."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .config import EXTRACTIVE_SUMMARY_MODEL, PDF_STORE_SCHEMA_VERSION
from .utils import (
    compute_rel_path_for_store,
    compute_source_path_from_rel_path,
    extractive_summary,
    iter_batched,
    sha256_file,
    sha256_hex,
    sort_documents_deterministically,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Connection / initialisation
# ---------------------------------------------------------------------------
def open_pdf_page_store(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    except Exception:
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
          ragas_headlines_json TEXT,
          ragas_headlines_model TEXT,
          ragas_summary TEXT,
          ragas_summary_model TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          updated_at TEXT,
          UNIQUE(pdf_sha256, page_number)
        )
        """
    )
    # Lightweight schema migration: add missing columns.
    try:
        existing_cols = {
            str(r[1])
            for r in conn.execute("PRAGMA table_info(pdf_page_store)").fetchall()
            if r and r[1]
        }
    except Exception:
        existing_cols = set()

    def _add_col(col: str, decl: str) -> None:
        if col in existing_cols:
            return
        try:
            conn.execute(f"ALTER TABLE pdf_page_store ADD COLUMN {col} {decl}")
            existing_cols.add(col)
        except Exception:
            pass

    _add_col("ragas_headlines_json", "TEXT")
    _add_col("ragas_headlines_model", "TEXT")
    _add_col("ragas_summary", "TEXT")
    _add_col("ragas_summary_model", "TEXT")
    _add_col("ragas_entities_json", "TEXT")
    _add_col("ragas_entities_model", "TEXT")
    _add_col("ragas_themes_json", "TEXT")
    _add_col("ragas_themes_model", "TEXT")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS pdf_page_store_path_page_idx "
        "ON pdf_page_store(rel_path, page_number)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS pdf_page_store_pdfsha_idx "
        "ON pdf_page_store(pdf_sha256)"
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


# ---------------------------------------------------------------------------
# Cached RAGAS extractions (headlines/summary) — read helpers
# ---------------------------------------------------------------------------
def pdf_store_load_cached_pairs(
    conn: sqlite3.Connection,
    *,
    rel_paths: List[str],
    value_col: str,
    model_col: str,
) -> Dict[Tuple[str, int], Tuple[Optional[str], Optional[str]]]:
    """Load cached (value, model_tag) pairs keyed by (rel_path, page_number)."""
    out: Dict[Tuple[str, int], Tuple[Optional[str], Optional[str]]] = {}
    if not rel_paths:
        return out
    for chunk in iter_batched(rel_paths, batch_size=800):
        placeholders = ",".join(["?"] * len(chunk))
        try:
            rows = conn.execute(
                f"""
                SELECT rel_path, page_number, {value_col}, {model_col}
                FROM pdf_page_store
                WHERE rel_path IN ({placeholders})
                  AND {value_col} IS NOT NULL
                """,
                tuple(chunk),
            ).fetchall()
        except Exception:
            continue
        for rel_path, page_number, blob, model in rows:
            try:
                pn = int(page_number)
            except Exception:
                continue
            out[(str(rel_path), pn)] = (
                str(blob) if blob is not None else None,
                str(model) if model is not None else None,
            )
    return out


# ---------------------------------------------------------------------------
# Cached RAGAS extractions — write-back after KG build
# ---------------------------------------------------------------------------
def pdf_store_persist_ragas_extractions(
    conn: sqlite3.Connection,
    *,
    kg: Any,
    base_input_dir: Path,
    headlines_model_tag: str,
    summary_model_tag: str,
    entities_model_tag: str = "",
    themes_model_tag: str = "",
) -> Dict[str, int]:
    """Persist per-page RAGAS LLM extractions from the KG back into SQLite."""
    from ragas.testset.graph import NodeType

    counts = {"headlines": 0, "summary": 0, "entities": 0, "themes": 0}
    now = utc_now_iso()

    def _persist_json_list(node, prop_name, col_name, model_col, model_tag):
        value = node.get_property(prop_name)
        if not isinstance(value, list) or not model_tag:
            return 0
        try:
            blob = json.dumps(value, ensure_ascii=False)
        except Exception:
            return 0
        try:
            cur = conn.execute(
                f"""
                UPDATE pdf_page_store
                SET {col_name} = ?, {model_col} = ?, updated_at = ?
                WHERE rel_path = ? AND page_number = ?
                  AND (
                    {col_name} IS NULL OR {col_name} = ''
                    OR {model_col} IS NULL OR {model_col} = ''
                    OR {model_col} != ?
                  )
                """,
                (blob, str(model_tag), now,
                 str(rel_path), int(page_number), str(model_tag)),
            )
            return int(cur.rowcount or 0)
        except Exception:
            return 0

    def _persist_text(node, prop_name, col_name, model_col, model_tag):
        value = node.get_property(prop_name)
        if not isinstance(value, str) or not model_tag:
            return 0
        try:
            cur = conn.execute(
                f"""
                UPDATE pdf_page_store
                SET {col_name} = ?, {model_col} = ?, updated_at = ?
                WHERE rel_path = ? AND page_number = ?
                  AND (
                    {col_name} IS NULL
                    OR {model_col} IS NULL OR {model_col} = ''
                    OR {model_col} != ?
                  )
                """,
                (value, str(model_tag), now,
                 str(rel_path), int(page_number), str(model_tag)),
            )
            return int(cur.rowcount or 0)
        except Exception:
            return 0

    with conn:
        for node in kg.nodes:
            # Headlines/summary live on DOCUMENT nodes;
            # entities/themes live on CHUNK nodes.
            # Both carry document_metadata (propagated by SafeHeadlineSplitter).
            if node.type not in (NodeType.DOCUMENT, NodeType.CHUNK):
                continue

            md = node.get_property("document_metadata")
            if not isinstance(md, dict):
                continue
            source = md.get("source")
            page0 = md.get("page")
            if not isinstance(source, str) or not source.strip():
                continue
            try:
                page_number = int(page0) + 1
            except Exception:
                continue
            if page_number < 1:
                continue
            try:
                rel_path = compute_rel_path_for_store(Path(source), base_input_dir)
            except Exception:
                continue

            counts["headlines"] += _persist_json_list(
                node, "headlines",
                "ragas_headlines_json", "ragas_headlines_model", headlines_model_tag,
            )
            counts["summary"] += _persist_text(
                node, "summary",
                "ragas_summary", "ragas_summary_model", summary_model_tag,
            )
            counts["entities"] += _persist_json_list(
                node, "entities",
                "ragas_entities_json", "ragas_entities_model", entities_model_tag,
            )
            counts["themes"] += _persist_json_list(
                node, "themes",
                "ragas_themes_json", "ragas_themes_model", themes_model_tag,
            )

    return counts


# ---------------------------------------------------------------------------
# Freshness checks
# ---------------------------------------------------------------------------
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
        FROM pdf_page_store WHERE rel_path = ?
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
    if min_page is None or max_page is None:
        return True
    if int(min_page) != 1 or int(max_page) != count_rows:
        return True
    return False


def pdf_store_needs_embeddings(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    embedding_model_tag: Optional[str],
) -> bool:
    if not embedding_model_tag:
        row = conn.execute(
            "SELECT COUNT(*) FROM pdf_page_store "
            "WHERE rel_path = ? AND (embedding_f32 IS NULL OR embedding_dims IS NULL)",
            (rel_path,),
        ).fetchone()
        return bool(row and int(row[0] or 0) > 0)

    row = conn.execute(
        """
        SELECT COUNT(*) FROM pdf_page_store
        WHERE rel_path = ?
          AND (
            embedding_f32 IS NULL OR embedding_dims IS NULL
            OR embedding_model IS NULL OR embedding_model = ''
            OR embedding_model != ?
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
        "SELECT pdf_profile_json, pdf_profile_model "
        "FROM pdf_page_store "
        "WHERE rel_path = ? AND pdf_profile_json IS NOT NULL LIMIT 1",
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


# ---------------------------------------------------------------------------
# Backfill helpers
# ---------------------------------------------------------------------------
def pdf_store_fill_missing_summaries(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    summary_model: str = EXTRACTIVE_SUMMARY_MODEL,
) -> int:
    rows = conn.execute(
        """
        SELECT id, doc_content FROM pdf_page_store
        WHERE rel_path = ?
          AND (summary IS NULL OR summary = ''
               OR summary_model IS NULL OR summary_model = '')
        ORDER BY page_number ASC
        """,
        (rel_path,),
    ).fetchall()
    if not rows:
        return 0

    now = utc_now_iso()
    updated = 0
    with conn:
        for row_id, content in rows:
            s = extractive_summary(str(content or ""))
            conn.execute(
                "UPDATE pdf_page_store "
                "SET summary = ?, summary_model = ?, updated_at = ? WHERE id = ?",
                (s, str(summary_model), now, int(row_id)),
            )
            updated += 1
    return updated


def pdf_store_compute_embeddings(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    embedding_model: Any,
    embedding_model_tag: str,
) -> int:
    rows = conn.execute(
        """
        SELECT id, doc_content FROM pdf_page_store
        WHERE rel_path = ?
          AND (
            embedding_f32 IS NULL OR embedding_dims IS NULL
            OR embedding_model IS NULL OR embedding_model = ''
            OR embedding_model != ?
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
    for batch in iter_batched(texts, batch_size=64):
        try:
            batch_vecs = embedding_model.embed_documents(batch)
            if not isinstance(batch_vecs, list) or len(batch_vecs) != len(batch):
                raise ValueError("Unexpected embed_documents result shape")
            vecs.extend(batch_vecs)
        except Exception:
            for t in batch:
                try:
                    v = embedding_model.embed_query(t)
                    vecs.append(v if isinstance(v, list) else None)
                except Exception:
                    vecs.append(None)

    if len(vecs) != len(ids):
        return 0

    now = utc_now_iso()
    updated = 0
    with conn:
        for row_id, v in zip(ids, vecs):
            if not isinstance(v, list) or not v:
                continue
            arr = np.asarray(v, dtype=np.float32)
            conn.execute(
                "UPDATE pdf_page_store "
                "SET embedding_f32 = ?, embedding_model = ?, embedding_dims = ?, "
                "    updated_at = ? WHERE id = ?",
                (arr.tobytes(), str(embedding_model_tag), int(arr.shape[0]),
                 now, int(row_id)),
            )
            updated += 1
    return updated


# ---------------------------------------------------------------------------
# Upsert (extract PDF pages → store)
# ---------------------------------------------------------------------------
def upsert_pdf_into_store(
    conn: sqlite3.Connection,
    *,
    pdf_path: Path,
    base_input_dir: Path,
    embedding_model: Any = None,
    embedding_model_id: Optional[str] = None,
    compute_embeddings: bool = False,
    reprocess: bool = False,
) -> int:
    """Extract all pages from *pdf_path* via PyPDFLoader and upsert into the store."""
    resolved = pdf_path.expanduser().resolve()
    st = resolved.stat()
    rel_path = compute_rel_path_for_store(resolved, base_input_dir)
    filename = resolved.name
    size_bytes = int(st.st_size)
    mtime_ns = int(st.st_mtime_ns)

    if reprocess:
        conn.execute("DELETE FROM pdf_page_store WHERE rel_path = ?", (rel_path,))

    pdf_sha256 = sha256_file(resolved)

    loader = PyPDFLoader(str(resolved))
    docs = loader.load()
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
        for batch in iter_batched(texts, batch_size=64):
            try:
                batch_vecs = embedding_model.embed_documents(batch)
                if not isinstance(batch_vecs, list) or len(batch_vecs) != len(batch):
                    raise ValueError("Unexpected embed_documents result shape")
                embeddings.extend(batch_vecs)
            except Exception:
                for t in batch:
                    try:
                        v = embedding_model.embed_query(t)
                        embeddings.append(v if isinstance(v, list) else None)
                    except Exception:
                        embeddings.append(None)
        if len(embeddings) != len(docs):
            embeddings = None

    now = utc_now_iso()
    stored = 0

    with conn:
        for idx, doc in enumerate(docs):
            page0 = doc.metadata.get("page")
            page_number = int(page0) + 1 if isinstance(page0, int) else (idx + 1)
            content = str(doc.page_content or "")
            content_sha256 = sha256_hex(content)
            content_chars = int(len(content))
            summary = extractive_summary(content)
            summary_model = EXTRACTIVE_SUMMARY_MODEL

            md = dict(doc.metadata or {}) if isinstance(doc.metadata, dict) else {}
            md["source"] = str(resolved)
            md["page"] = int(page_number - 1)
            metadata_json = json.dumps(md, ensure_ascii=False)

            emb_blob = emb_dims = emb_model = None
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
                  pdf_sha256, rel_path, filename, file_type,
                  size_bytes, mtime_ns, page_number,
                  doc_content, content_sha256, content_chars,
                  summary, summary_model,
                  embedding_f32, embedding_model, embedding_dims,
                  metadata_json, updated_at
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
                  embedding_f32 = COALESCE(excluded.embedding_f32,
                                           pdf_page_store.embedding_f32),
                  embedding_model = COALESCE(excluded.embedding_model,
                                             pdf_page_store.embedding_model),
                  embedding_dims = COALESCE(excluded.embedding_dims,
                                            pdf_page_store.embedding_dims),
                  metadata_json = excluded.metadata_json,
                  updated_at = excluded.updated_at
                """,
                (
                    pdf_sha256, rel_path, filename, "pdf",
                    size_bytes, mtime_ns, page_number,
                    content, content_sha256, content_chars,
                    summary, summary_model,
                    emb_blob, emb_model, emb_dims,
                    metadata_json, now,
                ),
            )
            stored += 1

    return stored


# ---------------------------------------------------------------------------
# Load documents from store
# ---------------------------------------------------------------------------
def load_documents_from_store(
    conn: sqlite3.Connection,
    *,
    base_input_dir: Path,
    pdf_paths: List[Path],
) -> List[Document]:
    rel_paths = [compute_rel_path_for_store(p, base_input_dir) for p in pdf_paths]
    docs: List[Document] = []

    for chunk in iter_batched(rel_paths, batch_size=800):
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

            source_path = compute_source_path_from_rel_path(str(rel_path), base_input_dir)
            md["source"] = source_path
            if page_number is not None:
                try:
                    md["page"] = int(page_number) - 1
                except Exception:
                    pass

            docs.append(Document(page_content=str(doc_content or ""), metadata=md))

    return sort_documents_deterministically(docs)


# ---------------------------------------------------------------------------
# PDF profile store helpers
# ---------------------------------------------------------------------------
def store_set_pdf_profile(
    conn: sqlite3.Connection,
    *,
    rel_path: str,
    profile: Dict[str, Any],
    pdf_profile_model: str,
) -> None:
    now = utc_now_iso()
    with conn:
        conn.execute(
            "UPDATE pdf_page_store "
            "SET pdf_profile_json = NULL, pdf_profile_model = NULL "
            "WHERE rel_path = ?",
            (rel_path,),
        )
        conn.execute(
            """
            UPDATE pdf_page_store
            SET pdf_profile_json = ?, pdf_profile_model = ?, updated_at = ?
            WHERE rel_path = ? AND page_number = 1
            """,
            (json.dumps(profile, ensure_ascii=False),
             str(pdf_profile_model), now, rel_path),
        )


def load_pdf_profiles_from_store(
    conn: sqlite3.Connection,
    *,
    pdf_paths: List[Path],
    base_input_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load profiles from SQLite. Returns map: absolute source_path → profile dict."""
    out: Dict[str, Dict[str, Any]] = {}
    for pdf_path in pdf_paths:
        rel_path = compute_rel_path_for_store(pdf_path, base_input_dir)
        row = conn.execute(
            "SELECT pdf_profile_json FROM pdf_page_store "
            "WHERE rel_path = ? AND pdf_profile_json IS NOT NULL LIMIT 1",
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
        prof["source_path"] = source_path
        prof["filename"] = resolved.name
        prof["filename_stem"] = resolved.stem
        prof["corpus_path"] = compute_corpus_path(resolved, base_input_dir)

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


def compute_corpus_path(pdf_path: Path, base_input_dir: Path) -> Optional[str]:
    """Compute a stable, non-personal corpus path (relative to input dir)."""
    try:
        rel = pdf_path.resolve().relative_to(base_input_dir.resolve())
        return str(rel)
    except Exception:
        return None
