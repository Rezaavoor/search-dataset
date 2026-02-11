"""Shared utility functions (hashing, JSON, path helpers, text parsing)."""

import ast
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import ragas
from langchain_core.documents import Document

from .config import CACHE_SCHEMA_VERSION, PIPELINE_ID_BASE

# ---------------------------------------------------------------------------
# Hop-prefix regex (RAGAS sometimes prepends "<1-hop>" to reference contexts)
# ---------------------------------------------------------------------------
_HOP_PREFIX_RE = re.compile(r"^\s*<\d+-hop>\s*", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Time / hashing helpers
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(obj: Any) -> str:
    """Stable serialisation for cache keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def iter_batched(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# ---------------------------------------------------------------------------
# JSON / LLM response parsing
# ---------------------------------------------------------------------------
def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the first JSON object found in a model response."""
    if not isinstance(text, str):
        return None
    t = text.strip()
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


def normalize_ynu(value: Any) -> str:
    """Normalise various yes/no/uncertain representations."""
    s = str(value or "").strip().lower()
    if s in {"yes", "y", "true", "1"}:
        return "yes"
    if s in {"no", "n", "false", "0"}:
        return "no"
    return "uncertain"


def normalize_topical_similarity(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in {"high", "h"}:
        return "high"
    if s in {"medium", "med", "m", "moderate"}:
        return "medium"
    if s in {"low", "l"}:
        return "low"
    return "none"


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------
def truncate_for_judge(text: str, *, max_chars: int = 8000) -> str:
    if not text:
        return ""
    t = text.replace("\x00", " ").strip()
    if len(t) <= max_chars:
        return t
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return f"{t[:head]}\n...\n{t[-tail:]}"


def truncate_for_profile(text: str, *, max_chars: int) -> str:
    if not text:
        return ""
    t = str(text).replace("\x00", " ").strip()
    if len(t) <= max_chars:
        return t
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return f"{t[:head]}\n...\n{t[-tail:]}"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def safe_resolve_path_str(path_like: Any) -> Optional[str]:
    if path_like is None:
        return None
    s = str(path_like).strip()
    if not s:
        return None
    try:
        return str(Path(s).expanduser().resolve())
    except Exception:
        return s


def compute_rel_path_for_store(pdf_path: Path, base_input_dir: Path) -> str:
    """Stable corpus-relative path when possible; otherwise absolute."""
    resolved = pdf_path.expanduser().resolve()
    try:
        return str(resolved.relative_to(base_input_dir.expanduser().resolve()))
    except Exception:
        return str(resolved)


def compute_source_path_from_rel_path(rel_path: str, base_input_dir: Path) -> str:
    """Convert a stored `rel_path` back into an absolute source path string."""
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


def compute_corpus_path(pdf_path: Path, base_input_dir: Path) -> Optional[str]:
    """Compute a stable, non-personal "corpus path" relative to input dir."""
    try:
        rel = pdf_path.resolve().relative_to(base_input_dir.resolve())
        return str(rel)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Extractive summary (cheap, deterministic)
# ---------------------------------------------------------------------------
def extractive_summary(text: str, *, max_chars: int = 450) -> str:
    if not text:
        return ""
    t = str(text).replace("\x00", " ").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return cut


# ---------------------------------------------------------------------------
# RAGAS text helpers
# ---------------------------------------------------------------------------
def strip_hop_prefix(text: str) -> str:
    """Remove RAGAS hop markers like '<1-hop>' from reference contexts."""
    return _HOP_PREFIX_RE.sub("", text)


def parse_reference_contexts(value: Any) -> List[str]:
    """Robustly parse `reference_contexts` into a list of strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(x) for x in v if x is not None]
                return [str(v)]
            except Exception:
                pass
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x) for x in v if x is not None]
            return [str(v)]
        except Exception:
            return [s]
    return [str(value)]


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------
def sort_documents_deterministically(docs: List[Document]) -> List[Document]:
    def _key(d: Document):
        source = str(d.metadata.get("source", ""))
        page = d.metadata.get("page")
        page_sort = page if isinstance(page, int) else -1
        return (source, page_sort)
    return sorted(docs, key=_key)


def group_docs_by_source(docs: List[Document]) -> Dict[str, List[Document]]:
    out: Dict[str, List[Document]] = {}
    for d in docs:
        src = safe_resolve_path_str(d.metadata.get("source"))
        if not src:
            continue
        out.setdefault(src, []).append(d)
    for src, items in out.items():
        def _k(doc: Document):
            p = doc.metadata.get("page")
            return p if isinstance(p, int) else 10**9
        out[src] = sorted(items, key=_k)
    return out


# ---------------------------------------------------------------------------
# Cache-key computation
# ---------------------------------------------------------------------------
def compute_docs_fingerprint(pdf_paths: List[Path]) -> str:
    """Compute a stable fingerprint for a set of PDF files.

    NOTE: The payload shape (including ``recursive`` and ``max_files``) is kept
    identical to the previous ``compute_docs_cache_id`` so that existing KG
    caches remain valid after the refactor.
    """
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "documents",
        "recursive": True,
        "max_files": None,
        "pdfs": [file_fingerprint(p) for p in pdf_paths],
    }
    return sha256_hex(canonical_json(payload))[:16]


def compute_kg_cache_id(
    docs_fingerprint: str,
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
        "docs_cache_id": docs_fingerprint,
        "provider": provider,
        "llm_id": llm_id,
        "embedding_id": embedding_id,
        "add_content_embeddings": add_content_embeddings,
    }
    return sha256_hex(canonical_json(payload))[:16]
