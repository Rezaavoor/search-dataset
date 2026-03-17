#!/usr/bin/env python3
"""Compute embeddings for all pages in the SQLite page store.

Supports multiple OpenAI API keys for parallel throughput, AWS Bedrock
Cohere models, and open-source SentenceTransformer/HuggingFace models.
Resume-safe: only processes pages that are missing embeddings for the
target model. Commits after every batch so progress is never lost.

Features:
  - Multi-key parallelism for OpenAI (up to N concurrent workers).
  - Exponential backoff with jitter on rate-limit / throttle errors.
  - Graceful shutdown on Ctrl+C (finishes current batches, commits, exits).
  - Progress tracking with ETA.
  - Configurable batch size, max retries, timeouts.

Environment Variables (Azure OpenAI — supports multiple deployments):
    AZURE_OPENAI_API_KEY                  - Primary key
    AZURE_OPENAI_ENDPOINT                 - Primary endpoint
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME - Deployment (default: text-embedding-3-large)
    AZURE_OPENAI_API_VERSION              - API version (default: 2024-02-15-preview)
    AZURE_OPENAI_API_KEY_2  + AZURE_OPENAI_ENDPOINT_2   - Second deployment (optional)
    AZURE_OPENAI_API_KEY_3  + AZURE_OPENAI_ENDPOINT_3   - Third deployment  (optional)

Environment Variables (Direct OpenAI):
    OPENAI_API_KEY      - Primary API key
    OPENAI_API_KEY_2    - Second API key (optional)
    OPENAI_API_KEY_3    - Third API key  (optional)

Environment Variables (Cohere via Bedrock):
    AWS_ACCESS_KEY_ID     - Required for Bedrock
    AWS_SECRET_ACCESS_KEY - Required for Bedrock
    AWS_BEDROCK_REGION    - Region (default: eu-central-1)

Usage:
    # Azure OpenAI (auto-detected, uses all 3 deployments in parallel)
    python embed_corpus.py --model text-embedding-3-large

    # Direct OpenAI
    python embed_corpus.py --provider openai --model text-embedding-3-large

    # Cohere embed-v4 via Bedrock
    python embed_corpus.py --provider bedrock --model cohere.embed-v4

    # Open-source model (single)
    python embed_corpus.py --provider hf --model bge-large-en-v1.5

    # All 6 open-source models sequentially (for multi-embedding hard-negative mining)
    python embed_corpus.py --provider hf --models all-hf

    # Specific subset of models
    python embed_corpus.py --provider hf --models bge-large-en-v1.5,LaBSE,all-mpnet-base-v2

    # Dry run — show how many pages need embedding
    python embed_corpus.py --dry-run

    # Custom batch size and explicit worker count
    python embed_corpus.py --batch-size 128 --workers 3
"""

import argparse
import logging
import os
import random
import signal
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

load_dotenv(SCRIPT_DIR / ".env")

from modules.config import DEFAULT_PDF_STORE_DB_NAME
from modules.db import (
    _emb_col,
    ensure_embedding_columns,
    init_pdf_page_store,
    open_pdf_page_store,
    store_page_embedding,
)
from modules.utils import utc_now_iso

# ---------------------------------------------------------------------------
# Open-source (HuggingFace / SentenceTransformer) model registry
# ---------------------------------------------------------------------------
# Each entry maps a short alias to its config.
#   hf_id              – full HuggingFace model ID
#   trust_remote_code  – needed by models with custom code (jina-v3, stella)
#   device             – force a specific device (None = auto, "cpu" = force CPU)
#                        Models with Flash Attention / xformers crash on MPS;
#                        forcing CPU is slower but reliable on Apple Silicon.
#   doc_prompt_name    – prompt_name kwarg for encode() on documents (None = no prompt)
#   query_prompt_name  – prompt_name kwarg for encode() on queries   (None = no prompt)
#   query_prefix       – plain-text prefix prepended to query text   (None = no prefix)
#
# prompt_name takes priority over query_prefix when both are set.

_HF_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "stella_en_400m_v5": {
        "hf_id": "dunzhang/stella_en_400M_v5",
        "trust_remote_code": True,
        "device": "cpu",   # custom attention requires xformers; crashes on MPS
        "model_kwargs": {"attn_implementation": "eager"},       # use eager attention class
        "config_kwargs": {"use_memory_efficient_attention": False, "unpad_inputs": False},  # disable xformers
        "doc_prompt_name": None,
        "query_prompt_name": "s2p_query",   # built-in retrieval prompt
        "query_prefix": None,
    },
    "jina-embeddings-v3": {
        "hf_id": "jinaai/jina-embeddings-v3",
        "trust_remote_code": True,
        "device": "cpu",   # Flash Attention in custom code segfaults on MPS
        "doc_prompt_name": "retrieval.passage",
        "query_prompt_name": "retrieval.query",
        "query_prefix": None,
    },
    "snowflake-arctic-embed-m-v2.0": {
        "hf_id": "Snowflake/snowflake-arctic-embed-m-v2.0",
        "trust_remote_code": True,
        "device": "cpu",   # GTE-based custom attention requires xformers; force CPU
        "model_kwargs": {"attn_implementation": "eager"},
        "config_kwargs": {"use_memory_efficient_attention": False, "unpad_inputs": False},
        "doc_prompt_name": None,
        "query_prompt_name": "query",  # built-in query prompt
        "query_prefix": None,
    },
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "trust_remote_code": False,
        "device": None,   # standard XLM-RoBERTa, works on MPS
        "doc_prompt_name": None,
        "query_prompt_name": None,
        "query_prefix": None,
    },
}

# Convenience: build a flat alias->hf_id map (includes lowercase variants)
_HF_MODEL_ALIASES: Dict[str, str] = {}
for _alias, _cfg in _HF_MODEL_REGISTRY.items():
    _HF_MODEL_ALIASES[_alias] = _cfg["hf_id"]
    _HF_MODEL_ALIASES[_alias.lower()] = _cfg["hf_id"]

# Ordered list of the four ensemble models (for --models all-hf)
HF_ALL_MODELS: List[str] = [
    "stella_en_400m_v5",
    "jina-embeddings-v3",
    "snowflake-arctic-embed-m-v2.0",
    "bge-m3",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("embed")

# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 8
BASE_DELAY = 3.0        # seconds
MAX_DELAY = 120.0        # cap on backoff
JITTER_FACTOR = 0.3      # ± 30% jitter

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
    log.warning("Ctrl+C received — finishing current batches and exiting cleanly...")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# API key / endpoint discovery
# ---------------------------------------------------------------------------
def _discover_openai_keys() -> List[str]:
    """Discover all available direct OpenAI API keys from environment."""
    keys: List[str] = []
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        keys.append(k)
    for i in range(2, 10):
        k = os.environ.get(f"OPENAI_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)
    return keys


def _discover_azure_credentials() -> List[Dict[str, str]]:
    """Discover all available Azure OpenAI key+endpoint pairs from environment.

    Looks for:
        AZURE_OPENAI_API_KEY   + AZURE_OPENAI_ENDPOINT     (primary)
        AZURE_OPENAI_API_KEY_2 + AZURE_OPENAI_ENDPOINT_2   (second)
        AZURE_OPENAI_API_KEY_3 + AZURE_OPENAI_ENDPOINT_3   (third)
        ...
    All share the same deployment name and API version from the primary config.
    """
    creds: List[Dict[str, str]] = []
    deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    # Primary
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if key and endpoint:
        creds.append({
            "api_key": key,
            "endpoint": endpoint,
            "deployment": deployment,
            "api_version": api_version,
        })
    # Numbered (2, 3, ...)
    for i in range(2, 10):
        key = os.environ.get(f"AZURE_OPENAI_API_KEY_{i}", "").strip()
        endpoint = os.environ.get(f"AZURE_OPENAI_ENDPOINT_{i}", "").strip()
        if key and endpoint:
            creds.append({
                "api_key": key,
                "endpoint": endpoint,
                "deployment": deployment,
                "api_version": api_version,
            })
    return creds


# ---------------------------------------------------------------------------
# Embedding client creation
# ---------------------------------------------------------------------------
def _create_openai_client(model: str, api_key: str):
    """Create a LangChain OpenAIEmbeddings client with a specific API key."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key,
        request_timeout=60,
        max_retries=0,  # We handle retries ourselves
    )


def _create_azure_client(cred: Dict[str, str]):
    """Create a LangChain AzureOpenAIEmbeddings client."""
    from langchain_openai import AzureOpenAIEmbeddings

    return AzureOpenAIEmbeddings(
        azure_endpoint=cred["endpoint"],
        api_key=cred["api_key"],
        api_version=cred["api_version"],
        azure_deployment=cred["deployment"],
        request_timeout=60,
        max_retries=0,  # We handle retries ourselves
    )


class _DirectBedrockCohereEmbedder:
    """Direct Bedrock API wrapper for Cohere Embed v4.

    Bypasses langchain_aws which has parameter compatibility issues with
    Cohere v4.  Uses the same proven approach as run_mleb.py.
    """

    def __init__(self, model_id: str, region: str = "eu-central-1"):
        import boto3
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts as documents (search_document input_type)."""
        import json as _json

        body = _json.dumps({
            "texts": texts,
            "input_type": "search_document",
            "embedding_types": ["float"],
        })
        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
            accept="*/*",
            contentType="application/json",
        )
        response_body = _json.loads(response["body"].read())

        embeddings = response_body.get("embeddings", {})
        if isinstance(embeddings, dict) and "float" in embeddings:
            return embeddings["float"]
        if isinstance(embeddings, list):
            return embeddings
        raise ValueError(f"Unexpected Bedrock Cohere response: {list(response_body.keys())}")

    def embed_query(self, text: str) -> List[float]:
        """Embed a single text (fallback, uses search_document type)."""
        results = self.embed_documents([text])
        return results[0] if results else []


def _create_bedrock_client(model: str):
    """Create a direct Bedrock Cohere embedder (bypasses langchain_aws)."""
    region = os.environ.get("AWS_BEDROCK_REGION", "eu-central-1")
    return _DirectBedrockCohereEmbedder(model_id=model, region=region)


VOYAGE_OUTPUT_DIM = 2048


class _VoyageEmbedder:
    """Voyage AI embedder for corpus documents.

    Uses ``input_type="document"`` and 2048 output dimensions.
    The VOYAGE_API_KEY environment variable must be set.
    """

    def __init__(self, model_id: str):
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai is required for --provider voyage. "
                "Install with: pip install voyageai"
            ) from exc

        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )
        self.model_id = model_id
        self.client = voyageai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.client.embed(
            texts,
            model=self.model_id,
            input_type="document",
            output_dimension=VOYAGE_OUTPUT_DIM,
        )
        return result.embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def _create_voyage_client(model: str) -> _VoyageEmbedder:
    return _VoyageEmbedder(model_id=model)


def _resolve_hf_model_name(model: str) -> str:
    """Resolve short aliases to full HuggingFace model IDs."""
    key = str(model or "").strip()
    return _HF_MODEL_ALIASES.get(key, _HF_MODEL_ALIASES.get(key.lower(), key))


def _get_model_config(model: str) -> Dict[str, Any]:
    """Return the registry config for *model*, falling back to safe defaults."""
    key = str(model or "").strip()
    cfg = _HF_MODEL_REGISTRY.get(key) or _HF_MODEL_REGISTRY.get(key.lower())
    if cfg:
        return cfg
    # Not in registry — return safe defaults (works for any SentenceTransformer)
    return {
        "hf_id": _resolve_hf_model_name(key),
        "trust_remote_code": False,
        "device": None,
        "doc_prompt_name": None,
        "query_prompt_name": None,
        "query_prefix": None,
    }


class _SentenceTransformerEmbedder:
    """Thin wrapper exposing embed_documents/embed_query for local HF models.

    Handles model-specific requirements:
      - trust_remote_code for jina-v3, stella, etc.
      - prompt_name for models with built-in task prompts
      - query_prefix for models that need a retrieval instruction
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for --provider hf. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.alias = str(model_name or "").strip()
        self.cfg = _get_model_config(self.alias)
        self.model_id = self.cfg["hf_id"]
        self.batch_size = max(1, int(batch_size))

        # Build kwargs for SentenceTransformer constructor
        st_kwargs: Dict[str, Any] = dict(
            trust_remote_code=self.cfg.get("trust_remote_code", False),
        )
        forced_device = self.cfg.get("device")
        if forced_device:
            st_kwargs["device"] = forced_device
        model_kwargs = self.cfg.get("model_kwargs")
        if model_kwargs:
            st_kwargs["model_kwargs"] = model_kwargs
        config_kwargs = self.cfg.get("config_kwargs")
        if config_kwargs:
            st_kwargs["config_kwargs"] = config_kwargs

        extras = []
        if forced_device:
            extras.append(f"device={forced_device}")
        if model_kwargs:
            extras.append(f"model_kwargs={model_kwargs}")
        if config_kwargs:
            extras.append(f"config_kwargs={config_kwargs}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        log.info("Loading SentenceTransformer model: %s%s ...", self.model_id, suffix)

        self.model = SentenceTransformer(self.model_id, **st_kwargs)
        log.info("Model loaded — embedding dim: %d", self.model.get_sentence_embedding_dimension())

    # ----- helpers -----

    @staticmethod
    def _normalize_input(texts: List[str]) -> List[str]:
        return [str(t or "").strip() for t in texts]

    # ----- document embedding (used by embed_corpus) -----

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        docs = self._normalize_input(texts)
        kwargs: Dict[str, Any] = dict(
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if self.cfg.get("doc_prompt_name"):
            kwargs["prompt_name"] = self.cfg["doc_prompt_name"]
        vecs = self.model.encode(docs, **kwargs)
        if getattr(vecs, "ndim", 1) == 1:
            vecs = np.expand_dims(vecs, axis=0)
        return vecs.astype(np.float32).tolist()

    # ----- query embedding (used by hard-negative miner at search time) -----

    def embed_query(self, text: str) -> List[float]:
        q = str(text or "").strip()
        kwargs: Dict[str, Any] = dict(
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if self.cfg.get("query_prompt_name"):
            kwargs["prompt_name"] = self.cfg["query_prompt_name"]
        elif self.cfg.get("query_prefix"):
            q = f"{self.cfg['query_prefix']}{q}"
        vec = self.model.encode(q, **kwargs)
        return np.asarray(vec, dtype=np.float32).tolist()


def _create_hf_client(model: str, *, batch_size: int = 64):
    """Create a local SentenceTransformer embedder."""
    return _SentenceTransformerEmbedder(model_name=model, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Is-retriable check
# ---------------------------------------------------------------------------
def _is_retriable(exc: Exception) -> bool:
    """Check if an exception is likely a transient / rate-limit error."""
    err = str(exc).lower()
    return any(s in err for s in RETRIABLE_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Backoff delay with jitter
# ---------------------------------------------------------------------------
def _backoff_delay(attempt: int) -> float:
    delay = min(BASE_DELAY * (2 ** attempt), MAX_DELAY)
    jitter = delay * JITTER_FACTOR * (2 * random.random() - 1)
    return max(0.5, delay + jitter)


# ---------------------------------------------------------------------------
# Single-batch embedding with retry
# ---------------------------------------------------------------------------
def embed_batch_with_retry(
    client,
    texts: List[str],
    *,
    worker_id: int = 0,
) -> List[Optional[List[float]]]:
    """Embed a batch of texts with exponential backoff.

    Returns a list of embedding vectors (or None for failures).
    """
    for attempt in range(MAX_RETRIES):
        if _shutdown_event.is_set():
            return [None] * len(texts)

        try:
            vecs = client.embed_documents(texts)
            if isinstance(vecs, list) and len(vecs) == len(texts):
                return vecs
            raise ValueError(f"Unexpected result shape: got {len(vecs)}, expected {len(texts)}")
        except Exception as exc:
            if _is_retriable(exc) and attempt < MAX_RETRIES - 1:
                delay = _backoff_delay(attempt)
                log.warning(
                    "Worker %d: retriable error (attempt %d/%d), retrying in %.1fs: %s",
                    worker_id, attempt + 1, MAX_RETRIES, delay,
                    str(exc)[:120],
                )
                # Sleep in small increments so we can check shutdown
                deadline = time.monotonic() + delay
                while time.monotonic() < deadline and not _shutdown_event.is_set():
                    time.sleep(min(1.0, deadline - time.monotonic()))
                continue

            # Non-retriable or last attempt: fall back to per-text embedding
            if attempt == MAX_RETRIES - 1:
                log.warning(
                    "Worker %d: max retries (%d) exhausted for batch, falling back to per-text",
                    worker_id, MAX_RETRIES,
                )
            else:
                log.warning(
                    "Worker %d: non-retriable error, falling back to per-text: %s",
                    worker_id, str(exc)[:120],
                )
            return _embed_texts_individually(client, texts, worker_id=worker_id)

    return [None] * len(texts)


def _embed_texts_individually(
    client,
    texts: List[str],
    *,
    worker_id: int = 0,
) -> List[Optional[List[float]]]:
    """Embed texts one at a time (fallback for batch failures)."""
    results: List[Optional[List[float]]] = []
    for text in texts:
        if _shutdown_event.is_set():
            results.append(None)
            continue

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                v = client.embed_query(text)
                results.append(v if isinstance(v, list) else None)
                success = True
                break
            except Exception as exc:
                if _is_retriable(exc) and attempt < MAX_RETRIES - 1:
                    delay = _backoff_delay(attempt)
                    deadline = time.monotonic() + delay
                    while time.monotonic() < deadline and not _shutdown_event.is_set():
                        time.sleep(min(1.0, deadline - time.monotonic()))
                    continue
                results.append(None)
                success = True  # We "handled" it — None result
                break
        if not success:
            results.append(None)
    return results


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _fmt_count(n: int) -> str:
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


# ---------------------------------------------------------------------------
# Worker function (runs in thread)
# ---------------------------------------------------------------------------
def _worker_process_batches(
    *,
    worker_id: int,
    client,
    batches: List[List[Tuple[int, str]]],  # list of [(row_id, text), ...]
    db_path: Path,
    model_name: str,
    stats: Dict[str, int],
    stats_lock: threading.Lock,
    db_lock: threading.Lock,
) -> int:
    """Process assigned batches: embed + write to DB.

    Each worker opens its OWN DB connection for thread safety.
    Returns count of successfully stored embeddings.
    """
    # Each thread gets its own connection
    conn = open_pdf_page_store(db_path)
    ensure_embedding_columns(conn, model_name)
    stored = 0

    for batch in batches:
        if _shutdown_event.is_set():
            break

        row_ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]

        vecs = embed_batch_with_retry(client, texts, worker_id=worker_id)

        # Write to DB
        with db_lock:
            with conn:
                for row_id, v in zip(row_ids, vecs):
                    if not isinstance(v, list) or not v:
                        continue
                    arr = np.asarray(v, dtype=np.float32)
                    store_page_embedding(
                        conn, model_name, row_id=row_id, embedding=arr,
                    )
                    stored += 1

        with stats_lock:
            stats["processed"] += len(batch)
            stats["stored"] += stored - stats.get(f"_worker_{worker_id}_prev", 0)
            stats[f"_worker_{worker_id}_prev"] = stored

    conn.close()
    return stored


# ---------------------------------------------------------------------------
# Main embedding pipeline
# ---------------------------------------------------------------------------
def embed_corpus(
    *,
    db_path: Path,
    model_name: str,
    provider: str,
    batch_size: int = 64,
    max_workers: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Compute embeddings for all pages missing the target model.

    Returns stats dict: {total, processed, stored, skipped, errors}.
    """
    # --- Open DB and check what needs embedding ---
    conn = open_pdf_page_store(db_path)
    init_pdf_page_store(conn)
    ensure_embedding_columns(conn, model_name)
    blob_col = _emb_col(model_name)

    log.info("Querying pages missing embeddings for model=%s ...", model_name)
    rows = conn.execute(
        f"""
        SELECT id, doc_content FROM pdf_page_store
        WHERE {blob_col} IS NULL
        ORDER BY id
        """
    ).fetchall()

    # Filter out pages with empty/whitespace-only content — these can't be embedded
    # and would cause API errors (especially with Cohere)
    empty_ids = [int(r[0]) for r in rows if not str(r[1] or "").strip()]
    rows = [r for r in rows if str(r[1] or "").strip()]
    if empty_ids:
        log.info("Skipping %d pages with empty content (can't embed empty text)", len(empty_ids))

    total_in_db = conn.execute("SELECT COUNT(*) FROM pdf_page_store").fetchone()[0]
    already_done = total_in_db - len(rows)
    conn.close()

    log.info("Total pages in store:      %s", _fmt_count(total_in_db))
    log.info("Already embedded:          %s", _fmt_count(already_done))
    log.info("Pages needing embedding:   %s", _fmt_count(len(rows)))

    if not rows:
        log.info("Nothing to do — all pages have embeddings for %s", model_name)
        return {"total": total_in_db, "processed": 0, "stored": 0}

    if dry_run:
        log.info("DRY RUN — no embeddings will be computed")
        return {"total": total_in_db, "processed": 0, "stored": 0, "to_embed": len(rows)}

    # --- Auto-detect provider ---
    if provider == "auto":
        resolved_model = _resolve_hf_model_name(model_name)
        if (
            model_name in _HF_MODEL_ALIASES
            or model_name.lower() in _HF_MODEL_ALIASES
            or resolved_model in _HF_MODEL_ALIASES.values()
            or model_name in _HF_MODEL_REGISTRY
        ):
            # Model is a known open-source alias — use local HF inference
            provider = "hf"
            log.info("Auto-detected provider: hf (local open-source model)")
        elif os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider = "azure"
            log.info("Auto-detected provider: azure")
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
            log.info("Auto-detected provider: openai")
        elif os.environ.get("VOYAGE_API_KEY"):
            provider = "voyage"
            log.info("Auto-detected provider: voyage")
        elif os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_BEDROCK_REGION"):
            provider = "bedrock"
            log.info("Auto-detected provider: bedrock")
        elif "/" in str(model_name):
            # Looks like a HuggingFace model ID (org/model)
            provider = "hf"
            log.info("Auto-detected provider: hf (HuggingFace model ID)")
        else:
            log.error("Cannot auto-detect provider. Set Azure/OpenAI/AWS env vars or use --provider.")
            sys.exit(1)

    # --- Create embedding clients ---
    clients = []
    if provider == "azure":
        creds = _discover_azure_credentials()
        if not creds:
            log.error(
                "No Azure OpenAI credentials found. Set AZURE_OPENAI_API_KEY + "
                "AZURE_OPENAI_ENDPOINT (and optionally _2, _3 suffixed pairs)."
            )
            sys.exit(1)
        for i, cred in enumerate(creds):
            c = _create_azure_client(cred)
            clients.append(c)
            # Extract region from endpoint for logging
            ep = cred["endpoint"]
            region_hint = ep.split("//")[-1].split(".")[0] if "//" in ep else ep[:30]
            log.info("  Azure client %d: endpoint=%s  deployment=%s",
                     i + 1, region_hint, cred["deployment"])
    elif provider == "openai":
        keys = _discover_openai_keys()
        if not keys:
            log.error("No OpenAI API keys found. Set OPENAI_API_KEY (and optionally OPENAI_API_KEY_2, OPENAI_API_KEY_3).")
            sys.exit(1)
        for i, key in enumerate(keys):
            c = _create_openai_client(model_name, key)
            clients.append(c)
            masked = key[:8] + "..." + key[-4:] if len(key) > 16 else "***"
            log.info("  OpenAI client %d: key=%s", i + 1, masked)
    elif provider == "bedrock":
        c = _create_bedrock_client(model_name)
        clients.append(c)
        log.info("  Bedrock client: model=%s", model_name)
    elif provider == "hf":
        resolved_model = _resolve_hf_model_name(model_name)
        c = _create_hf_client(model_name, batch_size=batch_size)
        clients.append(c)
        log.info("  HF client: model=%s", resolved_model)
    elif provider == "voyage":
        c = _create_voyage_client(model_name)
        clients.append(c)
        log.info("  Voyage client: model=%s  input_type=document  output_dimension=%d",
                 model_name, VOYAGE_OUTPUT_DIM)
    else:
        log.error("Unknown provider: %s", provider)
        sys.exit(1)

    num_workers = max_workers if max_workers else len(clients)
    num_workers = min(num_workers, len(clients))  # Can't have more workers than clients
    if provider in ("hf", "voyage") and num_workers > 1:
        log.warning("%s provider uses a single client instance; capping workers to 1.", provider)
        num_workers = 1
    log.info("Workers: %d  |  Batch size: %d  |  Total batches: ~%d",
             num_workers, batch_size, (len(rows) + batch_size - 1) // batch_size)

    # --- Split rows into batches ---
    all_batches: List[List[Tuple[int, str]]] = []
    for i in range(0, len(rows), batch_size):
        batch = [(int(r[0]), str(r[1] or "")) for r in rows[i : i + batch_size]]
        all_batches.append(batch)

    # --- Distribute batches across workers (round-robin) ---
    worker_batches: List[List[List[Tuple[int, str]]]] = [[] for _ in range(num_workers)]
    for idx, batch in enumerate(all_batches):
        worker_batches[idx % num_workers].append(batch)

    for i in range(num_workers):
        log.info("  Worker %d: %d batches (%d pages)",
                 i + 1, len(worker_batches[i]),
                 sum(len(b) for b in worker_batches[i]))

    # --- Shared state ---
    stats: Dict[str, int] = {"processed": 0, "stored": 0}
    stats_lock = threading.Lock()
    db_lock = threading.Lock()

    # --- Progress reporter thread ---
    # Use a local event so we can stop the progress thread without touching
    # the global _shutdown_event (which must remain clean for multi-model runs).
    _progress_stop = threading.Event()
    t0 = time.monotonic()

    def _progress_reporter():
        while not _progress_stop.is_set() and not _shutdown_event.is_set():
            _progress_stop.wait(15)  # Report every 15 seconds
            with stats_lock:
                processed = stats["processed"]
                stored = stats["stored"]
            elapsed = time.monotonic() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(rows) - processed) / rate if rate > 0 else 0
            pct = (processed / len(rows)) * 100 if rows else 0
            log.info(
                "Progress: %d/%d (%.1f%%)  stored=%d  rate=%.1f pages/s  elapsed=%s  ETA=%s",
                processed, len(rows), pct, stored,
                rate, _fmt_duration(elapsed), _fmt_duration(remaining),
            )
            if processed >= len(rows):
                break

    progress_thread = threading.Thread(target=_progress_reporter, daemon=True)
    progress_thread.start()

    # --- Launch workers ---
    log.info("Starting embedding (Ctrl+C to stop gracefully)...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            future = executor.submit(
                _worker_process_batches,
                worker_id=i + 1,
                client=clients[i],
                batches=worker_batches[i],
                db_path=db_path,
                model_name=model_name,
                stats=stats,
                stats_lock=stats_lock,
                db_lock=db_lock,
            )
            futures[future] = i + 1

        # Wait for all workers
        total_stored = 0
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                worker_stored = future.result()
                total_stored += worker_stored
                log.info("Worker %d finished: %d embeddings stored", worker_id, worker_stored)
            except Exception as exc:
                log.error("Worker %d failed: %s", worker_id, exc)

    # Signal progress reporter to stop (local event, not the global shutdown)
    _progress_stop.set()
    progress_thread.join(timeout=5)

    elapsed = time.monotonic() - t0

    # --- Final summary ---
    log.info("=" * 80)
    log.info("EMBEDDING COMPLETE in %s", _fmt_duration(elapsed))
    log.info("  Pages processed: %d / %d", stats["processed"], len(rows))
    log.info("  Embeddings stored: %d", total_stored)
    log.info("  Rate: %.1f pages/s", stats["processed"] / elapsed if elapsed > 0 else 0)

    # Verify from DB
    try:
        conn2 = open_pdf_page_store(db_path)
        done_now = conn2.execute(
            f"SELECT COUNT(*) FROM pdf_page_store WHERE {blob_col} IS NOT NULL"
        ).fetchone()[0]
        still_missing = conn2.execute(
            f"SELECT COUNT(*) FROM pdf_page_store WHERE {blob_col} IS NULL"
        ).fetchone()[0]
        conn2.close()
        log.info("  DB verify: %s embedded, %s still missing",
                 _fmt_count(done_now), _fmt_count(still_missing))
    except Exception:
        pass

    if _shutdown_event.is_set():
        log.info("  (Stopped early due to Ctrl+C — re-run to resume)")

    return {
        "total": len(rows),
        "processed": stats["processed"],
        "stored": total_stored,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute embeddings for pages in the SQLite page store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--provider",
        type=str,
        choices=["auto", "azure", "openai", "bedrock", "hf", "voyage"],
        default="auto",
        help="Embedding provider (default: auto-detect from env vars)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-large",
        help="Embedding model name (default: text-embedding-3-large)",
    )
    p.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite DB path (default: processed/pdf_page_store.sqlite)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help=(
            "Texts per API call (default: 64). "
            "OpenAI max is 2048. Voyage max is 128 (use --batch-size 128 for Voyage)."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: auto (one per API key for OpenAI, 1 for Bedrock).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Count pages needing embedding without computing anything.",
    )
    p.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Embed with multiple models sequentially (comma-separated aliases). "
            "Use 'all-hf' to run all 6 open-source models. "
            "Overrides --model when provided. "
            "Example: --models bge-large-en-v1.5,LaBSE,all-mpnet-base-v2"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.db_path:
        db_path = Path(args.db_path).expanduser().resolve()
    else:
        db_path = (SCRIPT_DIR / "processed" / DEFAULT_PDF_STORE_DB_NAME).resolve()

    # --- Resolve model list ---
    if args.models:
        raw = str(args.models).strip()
        if raw.lower() == "all-hf":
            model_list = list(HF_ALL_MODELS)
        else:
            model_list = [m.strip() for m in raw.split(",") if m.strip()]
        if not model_list:
            log.error("--models produced an empty list.")
            sys.exit(1)
    else:
        model_list = [args.model]

    log.info("Provider: %s", args.provider)
    log.info("DB:       %s", db_path)
    log.info("Models:   %s", ", ".join(model_list))

    for idx, model_name in enumerate(model_list, 1):
        # Voyage embeddings are always stored with a -2048 suffix to avoid
        # collisions with any 1024-dim embeddings that may exist.
        storage_name = f"{model_name}-2048" if args.provider == "voyage" else model_name

        if len(model_list) > 1:
            log.info("=" * 80)
            log.info("MODEL %d/%d: %s", idx, len(model_list), model_name)
            log.info("=" * 80)
            # Reset shutdown event between runs so Ctrl+C works per-model
            _shutdown_event.clear()

        if storage_name != model_name:
            log.info("Storage column: %s", storage_name)

        embed_corpus(
            db_path=db_path,
            model_name=storage_name,
            provider=args.provider,
            batch_size=args.batch_size,
            max_workers=args.workers,
            dry_run=args.dry_run,
        )

        if _shutdown_event.is_set():
            log.info("Stopping model loop due to Ctrl+C")
            break

    if len(model_list) > 1:
        log.info("=" * 80)
        log.info("ALL MODELS COMPLETE (%d models)", len(model_list))


if __name__ == "__main__":
    main()
