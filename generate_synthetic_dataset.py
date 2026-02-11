#!/usr/bin/env python3
"""
RAGAS Synthetic Dataset Generator for Legal Documents

Generates a synthetic Q&A dataset from legal PDF documents using RAGAS.
All intermediate data is stored in a SQLite PDF page store.

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

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import ragas
from dotenv import load_dotenv
from ragas.testset.persona import Persona, generate_personas_from_kg

from modules.config import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_CORPUS_SIZE_HINT,
    DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
    DEFAULT_PDF_PROFILE_MAX_PAGES,
    DEFAULT_PDF_STORE_DB_NAME,
    PIPELINE_ID_BASE,
    RAGAS_DOC_EXTRACT_CACHE_VERSION,
)
from modules.db import (
    init_pdf_page_store,
    load_documents_from_store,
    open_pdf_page_store,
    pdf_store_compute_embeddings,
    pdf_store_fill_missing_summaries,
    pdf_store_needs_embeddings,
    pdf_store_needs_refresh,
    upsert_pdf_into_store,
)
from modules.hard_negatives import (
    load_all_pages_from_store,
    mine_hard_negatives_for_df,
    mine_hard_negatives_for_testset,
)
from modules.llm_setup import setup_llm_and_embeddings
from modules.profiles import build_pdf_profiles_from_store
from modules.synthesizers import (
    build_corpus_llm_context,
    build_query_distribution_for_pipeline,
    list_query_synthesizers,
)
from modules.single_hop import generate_single_hop_queries
from modules.testset import (
    add_source_mapping_columns,
    build_knowledge_graph,
    generate_testset,
    load_knowledge_graph_cache,
    load_personas_cache,
    load_personas_from_file,
    save_combined_dataframe,
    save_knowledge_graph_cache,
    save_personas_cache,
    save_testset,
    warn_on_referential_queries,
)
from modules.utils import (
    compute_docs_fingerprint,
    compute_kg_cache_id,
    compute_rel_path_for_store,
    utc_now_iso,
)

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"


# ============================================================================
# Step indicator helper
# ============================================================================
class StepTracker:
    """Simple step tracker for progress logging."""

    def __init__(self, total_steps: int):
        self.total = total_steps
        self.current = 0

    def next(self, description: str) -> None:
        self.current += 1
        print(f"\n{'=' * 60}")
        print(f"[Step {self.current}/{self.total}] {description}")
        print(f"{'=' * 60}")


# ============================================================================
# Path resolution helpers
# ============================================================================
def _resolve_input_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (SCRIPT_DIR / p).resolve()


def _resolve_input_file(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    for candidate_base in [Path.cwd(), base_dir, SCRIPT_DIR]:
        candidate = candidate_base / p
        if candidate.exists():
            return candidate.resolve()
    return (SCRIPT_DIR / p).resolve()


def _collect_pdf_paths(d: Path, *, recursive: bool) -> List[Path]:
    if not d.exists() or not d.is_dir():
        return []
    iterator = d.rglob("*.pdf") if recursive else d.glob("*.pdf")
    return sorted([p.resolve() for p in iterator if p.is_file()])


# ============================================================================
# Argument parser
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    default_dataset_dir = SCRIPT_DIR / "search-dataset"

    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A dataset from legal PDF documents using RAGAS"
    )

    # --- Input / output ---
    parser.add_argument(
        "--input-dir", type=str,
        default=str(default_dataset_dir) if default_dataset_dir.exists() else ".",
        help="Directory containing PDF documents (default: ./search-dataset or cwd)",
    )
    parser.add_argument(
        "--output", type=str, default="synthetic_dataset",
        help="Output file base name (files saved to output/ folder)",
    )
    parser.add_argument(
        "--output-formats", type=str, nargs="+", default=["csv", "json"],
        help="Output formats: csv, json, parquet (default: csv json)",
    )

    # --- PDF selection ---
    parser.add_argument(
        "--specific-folders", type=str, nargs="+", default=None,
        help="Only load PDFs from these subdirectories",
    )
    parser.add_argument(
        "--files", type=str, nargs="+", default=None,
        help="Specific PDF files to include (in addition to folders)",
    )
    parser.add_argument(
        "--max-pdfs", type=int, default=None,
        help="Maximum number of PDF files to include (default: no limit)",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Don't search subdirectories for PDFs",
    )

    # --- LLM / provider ---
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--provider", type=str, choices=["auto", "openai", "azure"], default="auto",
        help="LLM provider (default: auto-detect from env vars)",
    )

    # --- Testset generation ---
    parser.add_argument(
        "--testset-size", type=int, default=50,
        help="Number of test samples to generate (default: 50)",
    )
    parser.add_argument(
        "--num-personas", type=int, default=3,
        help="Number of personas to generate from KG (default: 3)",
    )
    parser.add_argument(
        "--personas-path", type=str, default=None,
        help="Path to a JSON/JSONL file with pre-defined personas",
    )

    # --- World-based generation (single-hop + per-world multi-hop) ---
    parser.add_argument(
        "--multi-hop-worlds", type=str, nargs="+", default=None,
        help=(
            "Enable world-based generation: list of subfolder paths (relative "
            "to --input-dir) for which per-world KGs are built and multi-hop "
            "queries are generated.  Single-hop queries are generated from the "
            "full corpus without a KG.  Supports nested paths.  Example: "
            '--multi-hop-worlds Claires "Law worlds/415"'
        ),
    )
    parser.add_argument(
        "--single-hop-size", type=int, default=None,
        help=(
            "Number of single-hop queries to generate from the full corpus "
            "(world mode only; default: half of --testset-size)"
        ),
    )
    parser.add_argument(
        "--multi-hop-size", type=int, default=None,
        help=(
            "Total number of multi-hop queries across all worlds "
            "(world mode only; default: half of --testset-size)"
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for single-hop file sampling (world mode)",
    )

    # --- Query generation ---
    parser.add_argument(
        "--standalone-queries", action=argparse.BooleanOptionalAction, default=True,
        help="Generate standalone corpus-level queries (default: enabled)",
    )
    parser.add_argument(
        "--corpus-size-hint", type=int, default=DEFAULT_CORPUS_SIZE_HINT,
        help="Approximate corpus size to guide query generation",
    )
    parser.add_argument(
        "--query-llm-context", type=str, default=None,
        help="Additional LLM guidance for query generation",
    )
    parser.add_argument(
        "--query-mix", type=str, nargs="+", default=None,
        help="Custom query synthesizer mix (e.g., single_hop_entities=0.5)",
    )
    parser.add_argument(
        "--list-query-synthesizers", action="store_true",
        help="Print available query synthesizer names and exit",
    )

    # --- PDF profiles ---
    parser.add_argument(
        "--pdf-profiles", action=argparse.BooleanOptionalAction, default=True,
        help="Generate per-PDF LLM profiles for better queries (default: enabled)",
    )
    parser.add_argument(
        "--pdf-profile-max-pages", type=int, default=DEFAULT_PDF_PROFILE_MAX_PAGES,
        help="Max pages per PDF profile excerpt (default: 3)",
    )
    parser.add_argument(
        "--pdf-profile-max-chars-per-page", type=int,
        default=DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
        help="Max chars per page in profile excerpt (default: 2500)",
    )

    # --- Hard negatives ---
    parser.add_argument(
        "--hard-negatives", action="store_true",
        help="Enable hard negative mining for IR evaluation",
    )
    parser.add_argument(
        "--num-bm25-negatives", type=int, default=5,
        help="BM25 hard negatives per query (default: 5)",
    )
    parser.add_argument(
        "--num-embedding-negatives", type=int, default=5,
        help="Embedding hard negatives per query (default: 5)",
    )

    # --- SQLite store ---
    parser.add_argument(
        "--pdf-store-db", type=str, default=None,
        help="Custom SQLite DB path for the PDF page store",
    )
    parser.add_argument(
        "--pdf-store-embeddings", action=argparse.BooleanOptionalAction, default=False,
        help="Compute and store page embeddings in the SQLite store",
    )

    # --- Caching / reprocessing ---
    parser.add_argument(
        "--processed-dir", type=str, default=str(SCRIPT_DIR / "processed"),
        help="Directory for cached KG and persona artifacts",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable loading cached KG / personas",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Recompute all artifacts even if cache exists",
    )
    parser.add_argument(
        "--no-content-embeddings", action="store_true",
        help="Disable page_content embeddings in KG (use summary only)",
    )

    return parser


# ============================================================================
# Main pipeline
# ============================================================================
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # --- Handle --list-query-synthesizers ---
    if args.list_query_synthesizers:
        print("\nAvailable query synthesizers for --query-mix:\n")
        for name, desc in list_query_synthesizers():
            print(f"  - {name}: {desc}")
        print("\nAliases also accepted:")
        print("  - single_hop_specific_query_synthesizer")
        print("  - multi_hop_abstract_query_synthesizer")
        print("  - multi_hop_specific_query_synthesizer")
        return 0

    # --- Validate args ---
    if args.num_personas is not None and int(args.num_personas) <= 0:
        print("\nError: --num-personas must be a positive integer.")
        return 1
    if args.max_pdfs is not None and args.max_pdfs <= 0:
        print("\nError: --max-pdfs must be a positive integer.")
        return 1

    # --- Compute total steps ---
    world_mode = bool(args.multi_hop_worlds)
    if world_mode:
        # collect, setup LLM, sync store, load docs,
        # [profiles], single-hop, N×(world KG+multihop), save
        total_steps = 5  # collect, LLM, store, load docs, save
        if args.pdf_profiles:
            total_steps += 1
        single_hop_size = (
            args.single_hop_size
            if args.single_hop_size is not None
            else max(args.testset_size, 1) // 2
        )
        if single_hop_size > 0:
            total_steps += 1  # single-hop step
        total_steps += len(args.multi_hop_worlds)  # one step per world
        if args.hard_negatives:
            total_steps += 1
    else:
        # collect, setup LLM, sync store, load docs,
        # [profiles], KG, personas, testset, save
        total_steps = 8
        if args.pdf_profiles:
            total_steps += 1
        if args.hard_negatives:
            total_steps += 1
    steps = StepTracker(total_steps)

    print("\n" + "=" * 60)
    print("RAGAS Synthetic Dataset Generator")
    print("=" * 60)

    # --- Resolve paths ---
    base_input_dir = _resolve_input_dir(args.input_dir)
    recursive = not args.no_recursive
    cache_enabled = not args.no_cache
    add_content_embeddings = not args.no_content_embeddings

    processed_dir = Path(args.processed_dir).expanduser()
    if not processed_dir.is_absolute():
        processed_dir = (SCRIPT_DIR / processed_dir).resolve()
    kg_cache_dir = processed_dir / "kg"
    personas_cache_dir = processed_dir / "personas"
    meta_dir = processed_dir / "meta"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # STEP 1: Collect PDF paths
    # -----------------------------------------------------------------------
    steps.next("Collecting PDF paths")

    if args.specific_folders:
        input_dirs = [base_input_dir / folder for folder in args.specific_folders]
    else:
        input_dirs = [base_input_dir]

    pdf_paths: List[Path] = []
    for d in input_dirs:
        pdf_paths.extend(_collect_pdf_paths(d, recursive=recursive))
    if args.files:
        for file_path in args.files:
            resolved = _resolve_input_file(file_path, base_input_dir)
            if resolved.exists() and resolved.suffix.lower() == ".pdf":
                pdf_paths.append(resolved.resolve())

    pdf_paths = sorted(set(pdf_paths))

    if args.max_pdfs is not None and len(pdf_paths) > args.max_pdfs:
        print(f"  Limiting to {args.max_pdfs} PDFs (from {len(pdf_paths)} found)")
        pdf_paths = pdf_paths[: args.max_pdfs]

    if not pdf_paths:
        print("\n  Error: No PDF files found. Check your --input-dir.")
        return 1
    print(f"  Found {len(pdf_paths)} PDF file(s)")

    # -----------------------------------------------------------------------
    # STEP 2: Setup LLM & embeddings + sync SQLite PDF store
    # -----------------------------------------------------------------------
    steps.next("Setting up LLM & embeddings")

    try:
        generator_llm, generator_embeddings = setup_llm_and_embeddings(
            args.model, args.provider
        )
    except ValueError as e:
        print(f"\n  Error: {e}")
        return 1

    # Resolve provider for cache keys
    provider_used = args.provider
    if provider_used == "auto":
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider_used = "azure"
        elif os.environ.get("OPENAI_API_KEY"):
            provider_used = "openai"
        else:
            provider_used = "openai"

    embedding_id = (
        os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
                       "text-embedding-ada-002")
        if provider_used == "azure" else "openai-default"
    )

    # Attach the embedding model tag so SQLiteCachedEmbeddingExtractor can match
    # against stored embedding_f32 blobs in SQLite.
    generator_embeddings._embedding_model_tag = embedding_id

    ragas_doc_extraction_model_tag_base = (
        f"{provider_used}:{args.model}:ragas{getattr(ragas, '__version__', 'unknown')}:"
        f"doc_extract_v{int(RAGAS_DOC_EXTRACT_CACHE_VERSION)}"
    )
    reuse_cached_store_extractions = bool(not args.reprocess)

    # -----------------------------------------------------------------------
    # STEP 3: Initialize & sync SQLite PDF store
    # -----------------------------------------------------------------------
    steps.next("Syncing SQLite PDF store")

    if args.pdf_store_db:
        pdf_store_db_path = Path(args.pdf_store_db).expanduser()
        if not pdf_store_db_path.is_absolute():
            pdf_store_db_path = (SCRIPT_DIR / pdf_store_db_path).resolve()
    else:
        pdf_store_db_path = (processed_dir / DEFAULT_PDF_STORE_DB_NAME).resolve()

    pdf_store_conn = open_pdf_page_store(pdf_store_db_path)
    init_pdf_page_store(pdf_store_conn)
    print(f"  SQLite store: {pdf_store_db_path}")

    want_embeddings = bool(args.pdf_store_embeddings)
    store_force = bool(args.reprocess)

    # Reuse the already-initialized embedding model for store operations
    store_embedding_model = generator_embeddings if want_embeddings else None
    store_embedding_model_id: Optional[str] = embedding_id if want_embeddings else None

    refreshed_pdfs = 0
    upserted_pages = 0
    skipped = 0
    filled_summaries = 0
    filled_embeddings = 0
    total_pdf_count = len(pdf_paths)

    for i, pdf_path in enumerate(pdf_paths, start=1):
        try:
            st = pdf_path.stat()
        except Exception:
            continue

        rel_path = compute_rel_path_for_store(pdf_path, base_input_dir)
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
                    compute_embeddings=want_embeddings,
                    reprocess=True,
                )
                upserted_pages += int(pages)
                refreshed_pdfs += 1
            else:
                skipped += 1

            filled_summaries += pdf_store_fill_missing_summaries(
                pdf_store_conn, rel_path=rel_path
            )

            if want_embeddings and store_embedding_model is not None:
                if pdf_store_needs_embeddings(
                    pdf_store_conn,
                    rel_path=rel_path,
                    embedding_model_tag=store_embedding_model_id,
                ):
                    filled_embeddings += pdf_store_compute_embeddings(
                        pdf_store_conn,
                        rel_path=rel_path,
                        embedding_model=store_embedding_model,
                        embedding_model_tag=str(store_embedding_model_id or ""),
                    )
        except Exception as e:
            print(f"  Warning: Failed for {pdf_path.name}: {e}")
            continue

        if i == 1 or i == total_pdf_count or (i % 25 == 0):
            print(
                f"  Processed {i}/{total_pdf_count} PDFs "
                f"(new: {refreshed_pdfs}, cached: {skipped})"
            )

    print(
        f"  Store sync complete: {refreshed_pdfs} refreshed, {skipped} cached, "
        f"{upserted_pages} pages upserted"
    )

    # -----------------------------------------------------------------------
    # STEP 4: Load documents from SQLite store
    # -----------------------------------------------------------------------
    steps.next("Loading documents from SQLite store")

    all_docs = load_documents_from_store(
        pdf_store_conn, base_input_dir=base_input_dir, pdf_paths=pdf_paths,
    )
    print(f"  Loaded {len(all_docs)} document page(s)")

    if not all_docs:
        print("\n  Error: No documents found. Check your input directory.")
        return 1

    # -----------------------------------------------------------------------
    # STEP 5 (optional): Build PDF profiles
    # -----------------------------------------------------------------------
    pdf_profiles_by_source: Dict[str, Dict[str, Any]] = {}
    if args.pdf_profiles:
        steps.next("Building PDF profiles")
        raw_profile_llm = getattr(generator_llm, "langchain_llm", None)
        pdf_profiles_by_source = build_pdf_profiles_from_store(
            pdf_store_conn,
            pdf_paths=pdf_paths,
            base_input_dir=base_input_dir,
            llm=raw_profile_llm,
            provider=provider_used,
            llm_id=args.model,
            reprocess=args.reprocess,
            max_pages=int(args.pdf_profile_max_pages),
            max_chars_per_page=int(args.pdf_profile_max_chars_per_page),
        )
        print(f"  PDF profiles: {len(pdf_profiles_by_source)} ready")
    else:
        print("\n  PDF profiles: disabled (--no-pdf-profiles)")

    # ===================================================================
    # Build LLM context (shared by both world and legacy modes)
    # ===================================================================
    llm_context: Optional[str] = None
    if args.standalone_queries:
        llm_context = build_corpus_llm_context(corpus_size_hint=args.corpus_size_hint)
        extra = str(args.query_llm_context or "").strip()
        if extra:
            llm_context = llm_context.rstrip() + "\n" + extra + "\n"
    else:
        extra = str(args.query_llm_context or "").strip()
        llm_context = extra if extra else None

    # ===================================================================
    # Decide pipeline mode
    # ===================================================================
    world_mode = bool(args.multi_hop_worlds)

    if world_mode:
        # ==============================================================
        # WORLD MODE: single-hop from full corpus + multi-hop per world
        # ==============================================================
        return _run_world_pipeline(
            args=args,
            steps=steps,
            pdf_paths=pdf_paths,
            pdf_store_conn=pdf_store_conn,
            base_input_dir=base_input_dir,
            all_docs=all_docs,
            generator_llm=generator_llm,
            generator_embeddings=generator_embeddings,
            provider_used=provider_used,
            embedding_id=embedding_id,
            add_content_embeddings=add_content_embeddings,
            cache_enabled=cache_enabled,
            kg_cache_dir=kg_cache_dir,
            meta_dir=meta_dir,
            personas_cache_dir=personas_cache_dir,
            pdf_profiles_by_source=pdf_profiles_by_source,
            llm_context=llm_context,
            reuse_cached_store_extractions=reuse_cached_store_extractions,
            ragas_doc_extraction_model_tag_base=ragas_doc_extraction_model_tag_base,
        )
    else:
        # ==============================================================
        # LEGACY MODE: single KG for all docs
        # ==============================================================
        return _run_legacy_pipeline(
            args=args,
            steps=steps,
            pdf_paths=pdf_paths,
            pdf_store_conn=pdf_store_conn,
            base_input_dir=base_input_dir,
            all_docs=all_docs,
            generator_llm=generator_llm,
            generator_embeddings=generator_embeddings,
            provider_used=provider_used,
            embedding_id=embedding_id,
            add_content_embeddings=add_content_embeddings,
            cache_enabled=cache_enabled,
            kg_cache_dir=kg_cache_dir,
            meta_dir=meta_dir,
            personas_cache_dir=personas_cache_dir,
            pdf_profiles_by_source=pdf_profiles_by_source,
            llm_context=llm_context,
            reuse_cached_store_extractions=reuse_cached_store_extractions,
            ragas_doc_extraction_model_tag_base=ragas_doc_extraction_model_tag_base,
        )


# ============================================================================
# Helper: check if a path is under a directory
# ============================================================================
def _is_under_dir(path: Path, directory: Path) -> bool:
    """Return True if *path* is inside *directory* (inclusive)."""
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


# ============================================================================
# WORLD MODE pipeline
# ============================================================================
def _run_world_pipeline(
    *,
    args,
    steps: StepTracker,
    pdf_paths: List[Path],
    pdf_store_conn,
    base_input_dir: Path,
    all_docs,
    generator_llm,
    generator_embeddings,
    provider_used: str,
    embedding_id: str,
    add_content_embeddings: bool,
    cache_enabled: bool,
    kg_cache_dir: Path,
    meta_dir: Path,
    personas_cache_dir: Path,
    pdf_profiles_by_source: Dict[str, Dict[str, Any]],
    llm_context: Optional[str],
    reuse_cached_store_extractions: bool,
    ragas_doc_extraction_model_tag_base: Optional[str],
) -> int:
    """Single-hop from full corpus + multi-hop from per-world KGs."""
    worlds: List[str] = args.multi_hop_worlds

    # --- Determine sizes ---
    total = max(args.testset_size, 1)
    single_hop_size = (
        args.single_hop_size
        if args.single_hop_size is not None
        else total // 2
    )
    multi_hop_size = (
        args.multi_hop_size
        if args.multi_hop_size is not None
        else total - single_hop_size
    )

    print("\n  World mode enabled")
    print(f"  Worlds: {', '.join(worlds)}")
    print(f"  Single-hop from full corpus: {single_hop_size}")
    print(f"  Multi-hop across worlds: {multi_hop_size}")

    # -------------------------------------------------------------------
    # Phase A: Single-hop queries from full corpus (no KG)
    # -------------------------------------------------------------------
    single_hop_results: List[Dict[str, Any]] = []
    if single_hop_size > 0:
        steps.next("Generating single-hop queries (full corpus)")
        print("  Standalone queries: enabled" if args.standalone_queries
              else "  Standalone queries: disabled")

        # Use the raw LangChain LLM (not the RAGAS wrapper)
        raw_llm = getattr(generator_llm, "langchain_llm", generator_llm)

        single_hop_results = generate_single_hop_queries(
            raw_llm,
            pdf_store_conn,
            num_queries=single_hop_size,
            seed=args.seed,
            corpus_size_hint=args.corpus_size_hint,
        )
        print(f"  Generated {len(single_hop_results)} single-hop queries")

    # -------------------------------------------------------------------
    # Phase B: Per-world multi-hop queries
    # -------------------------------------------------------------------
    multi_hop_dfs: List[pd.DataFrame] = []
    if multi_hop_size > 0 and worlds:
        queries_per_world = max(1, multi_hop_size // len(worlds))
        remainder = multi_hop_size - queries_per_world * len(worlds)

        if cache_enabled:
            kg_cache_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)
            personas_cache_dir.mkdir(parents=True, exist_ok=True)

        for world_idx, world_name in enumerate(worlds):
            steps.next(
                f"Multi-hop for world: {world_name} "
                f"({world_idx + 1}/{len(worlds)})"
            )

            # --- Filter PDFs for this world ---
            world_dir = base_input_dir / world_name
            world_pdf_paths = sorted([
                p for p in pdf_paths if _is_under_dir(p, world_dir)
            ])
            if not world_pdf_paths:
                print(f"  No PDFs found in world '{world_name}', skipping")
                continue
            print(f"  World '{world_name}': {len(world_pdf_paths)} PDF(s)")

            # --- Load docs for this world ---
            world_docs = load_documents_from_store(
                pdf_store_conn,
                base_input_dir=base_input_dir,
                pdf_paths=world_pdf_paths,
            )
            print(f"  Loaded {len(world_docs)} pages for world '{world_name}'")
            if not world_docs:
                print(f"  Warning: No document pages for world '{world_name}'")
                continue

            # --- Build / load KG for this world ---
            world_docs_fp = compute_docs_fingerprint(world_pdf_paths)
            world_kg_id = compute_kg_cache_id(
                world_docs_fp,
                provider=provider_used,
                llm_id=args.model,
                embedding_id=embedding_id,
                add_content_embeddings=add_content_embeddings,
            )
            world_kg_cache_path = kg_cache_dir / f"kg_{world_kg_id}.json.gz"
            world_kg_meta_path = meta_dir / f"kg_{world_kg_id}.json"

            if (
                cache_enabled
                and world_kg_cache_path.exists()
                and not args.reprocess
            ):
                print(f"  Loading cached KG: {world_kg_cache_path.name}")
                world_kg = load_knowledge_graph_cache(world_kg_cache_path)
                print(
                    f"  KG: {len(world_kg.nodes)} nodes, "
                    f"{len(world_kg.relationships)} relationships"
                )
            else:
                print(
                    f"  Building KG for '{world_name}' "
                    f"({len(world_pdf_paths)} PDFs, "
                    f"{len(world_docs)} pages)..."
                )
                world_kg = build_knowledge_graph(
                    world_docs,
                    generator_llm,
                    generator_embeddings,
                    add_content_embeddings=add_content_embeddings,
                    pdf_store_conn=pdf_store_conn,
                    base_input_dir=base_input_dir,
                    reuse_cached_store_extractions=reuse_cached_store_extractions,
                    ragas_doc_extraction_model_tag_base=(
                        ragas_doc_extraction_model_tag_base
                    ),
                )
                if cache_enabled:
                    print(f"  Saving KG cache: {world_kg_cache_path.name}")
                    save_knowledge_graph_cache(world_kg, world_kg_cache_path)

                    pipeline_id = PIPELINE_ID_BASE
                    if add_content_embeddings:
                        pipeline_id += "__content_embeddings"
                    with open(world_kg_meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "created_at": utc_now_iso(),
                                "schema_version": CACHE_SCHEMA_VERSION,
                                "pipeline_id": pipeline_id,
                                "ragas_version": getattr(
                                    ragas, "__version__", "unknown"
                                ),
                                "world": world_name,
                                "docs_fingerprint": world_docs_fp,
                                "kg_cache_id": world_kg_id,
                                "provider": provider_used,
                                "llm_id": args.model,
                                "embedding_id": embedding_id,
                                "add_content_embeddings": add_content_embeddings,
                                "num_nodes": len(world_kg.nodes),
                                "num_relationships": len(world_kg.relationships),
                            },
                            f, ensure_ascii=False, indent=2,
                        )

            # --- Generate personas for this world ---
            world_personas_cache_path = (
                personas_cache_dir
                / f"personas_{world_kg_id}__n{int(args.num_personas)}.json"
            )
            world_personas: Optional[List[Persona]] = None
            if args.personas_path:
                personas_path = Path(args.personas_path).expanduser()
                if not personas_path.is_absolute():
                    personas_path = (Path.cwd() / personas_path).resolve()
                world_personas = load_personas_from_file(personas_path)
            elif (
                cache_enabled
                and world_personas_cache_path.exists()
                and not args.reprocess
            ):
                print(
                    f"  Loading cached personas: "
                    f"{world_personas_cache_path.name}"
                )
                world_personas = load_personas_cache(world_personas_cache_path)
            else:
                print("  Generating personas from world KG...")
                world_personas = generate_personas_from_kg(
                    kg=world_kg,
                    llm=generator_llm,
                    num_personas=int(args.num_personas),
                )
                if cache_enabled:
                    save_personas_cache(
                        world_personas, world_personas_cache_path,
                    )
            print(
                f"  Personas: "
                f"{len(world_personas) if world_personas else 0}"
            )

            # --- Build multi-hop-only query distribution ---
            multi_hop_synth_names = [
                "multi_hop_abstract_summary",
                "multi_hop_abstract_content",
                "multi_hop_specific_entities",
            ]
            try:
                world_qd = build_query_distribution_for_pipeline(
                    generator_llm,
                    world_kg,
                    standalone_queries=args.standalone_queries,
                    llm_context=llm_context,
                    pdf_profiles_by_source=(
                        pdf_profiles_by_source if args.pdf_profiles else None
                    ),
                    query_mix=multi_hop_synth_names,
                )
            except ValueError as e:
                print(
                    f"  Warning: No multi-hop synthesizers compatible "
                    f"with '{world_name}': {e}"
                )
                continue

            # --- Generate multi-hop testset ---
            # Give the first world any remainder queries
            world_size = queries_per_world
            if world_idx == 0 and remainder > 0:
                world_size += remainder

            print(f"  Generating {world_size} multi-hop queries...")
            world_testset = generate_testset(
                world_kg,
                generator_llm,
                generator_embeddings,
                testset_size=world_size,
                personas=world_personas,
                llm_context=llm_context,
                query_distribution=world_qd,
            )

            if args.standalone_queries:
                warn_on_referential_queries(world_testset)

            # --- Convert to DataFrame + source mapping ---
            world_df = world_testset.to_pandas()
            world_df["world"] = world_name
            world_df["synthesizer_name"] = "multi_hop_ragas"

            if world_docs:
                add_source_mapping_columns(
                    world_df, world_docs, strict=True, kg=world_kg,
                )

            multi_hop_dfs.append(world_df)
            print(
                f"  Generated {len(world_df)} multi-hop queries "
                f"for world '{world_name}'"
            )

    # -------------------------------------------------------------------
    # Combine all results
    # -------------------------------------------------------------------
    all_dfs: List[pd.DataFrame] = []
    if single_hop_results:
        sh_df = pd.DataFrame(single_hop_results)
        all_dfs.append(sh_df)
    all_dfs.extend(multi_hop_dfs)

    if not all_dfs:
        print("\n  Error: No queries generated in any phase.")
        return 1

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # -------------------------------------------------------------------
    # Hard negatives (optional) — mines from full corpus via SQLite
    # -------------------------------------------------------------------
    if args.hard_negatives:
        steps.next("Mining hard negatives")

        raw_embedding_model = generator_embeddings
        judge_llm = getattr(generator_llm, "langchain_llm", None)

        print("  Loading all pages from SQLite store...")
        hn_pages = load_all_pages_from_store(
            pdf_store_conn,
            embedding_model_name=embedding_id,
        )

        hard_negs = mine_hard_negatives_for_df(
            combined_df,
            hn_pages,
            raw_embedding_model,
            judge_llm=judge_llm,
            num_bm25_negatives=args.num_bm25_negatives,
            num_embedding_negatives=args.num_embedding_negatives,
        )

        import json as _json
        combined_df["hard_negatives"] = [
            _json.dumps(negs, ensure_ascii=False) for negs in hard_negs
        ]
        neg_count = sum(len(x) for x in hard_negs)
        print(
            f"  Added {neg_count} hard negatives "
            f"({neg_count / max(len(combined_df), 1):.1f} avg per query)"
        )

    # -------------------------------------------------------------------
    # FINAL STEP: Save combined results to output/
    # -------------------------------------------------------------------
    steps.next("Saving results")

    output_path = str(OUTPUT_DIR / args.output)
    save_combined_dataframe(
        combined_df, output_path, formats=args.output_formats,
    )

    # Summary
    sh_count = int(
        (combined_df.get("synthesizer_name", pd.Series())
         == "single_hop_direct").sum()
    )
    mh_count = len(combined_df) - sh_count
    world_counts = {}
    if "world" in combined_df.columns:
        for w in combined_df["world"].dropna().unique():
            world_counts[w] = int((combined_df["world"] == w).sum())

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Single-hop (full corpus): {sh_count}")
    print(f"  Multi-hop (per-world KGs): {mh_count}")
    for w, c in world_counts.items():
        print(f"    - {w}: {c}")
    if args.hard_negatives:
        print("  Hard negatives: included")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    return 0


# ============================================================================
# LEGACY MODE pipeline (original single-KG approach)
# ============================================================================
def _run_legacy_pipeline(
    *,
    args,
    steps: StepTracker,
    pdf_paths: List[Path],
    pdf_store_conn,
    base_input_dir: Path,
    all_docs,
    generator_llm,
    generator_embeddings,
    provider_used: str,
    embedding_id: str,
    add_content_embeddings: bool,
    cache_enabled: bool,
    kg_cache_dir: Path,
    meta_dir: Path,
    personas_cache_dir: Path,
    pdf_profiles_by_source: Dict[str, Dict[str, Any]],
    llm_context: Optional[str],
    reuse_cached_store_extractions: bool,
    ragas_doc_extraction_model_tag_base: Optional[str],
) -> int:
    """Original pipeline: build a single KG from all docs, generate mixed queries."""

    # -----------------------------------------------------------------------
    # Build or load knowledge graph (file-cached)
    # -----------------------------------------------------------------------
    steps.next("Building knowledge graph")

    docs_fp = compute_docs_fingerprint(pdf_paths)
    kg_id = compute_kg_cache_id(
        docs_fp,
        provider=provider_used,
        llm_id=args.model,
        embedding_id=embedding_id,
        add_content_embeddings=add_content_embeddings,
    )

    if cache_enabled:
        kg_cache_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

    kg_cache_path = kg_cache_dir / f"kg_{kg_id}.json.gz"
    kg_meta_path = meta_dir / f"kg_{kg_id}.json"

    if cache_enabled and kg_cache_path.exists() and not args.reprocess:
        print(f"  Loading cached KG: {kg_cache_path.name}")
        kg = load_knowledge_graph_cache(kg_cache_path)
        print(
            f"  Loaded KG: {len(kg.nodes)} nodes, "
            f"{len(kg.relationships)} relationships"
        )
    else:
        kg = build_knowledge_graph(
            all_docs,
            generator_llm,
            generator_embeddings,
            add_content_embeddings=add_content_embeddings,
            pdf_store_conn=pdf_store_conn,
            base_input_dir=base_input_dir,
            reuse_cached_store_extractions=reuse_cached_store_extractions,
            ragas_doc_extraction_model_tag_base=ragas_doc_extraction_model_tag_base,
        )
        if cache_enabled:
            print(f"  Saving KG cache: {kg_cache_path.name}")
            save_knowledge_graph_cache(kg, kg_cache_path)
            pipeline_id = PIPELINE_ID_BASE
            if add_content_embeddings:
                pipeline_id += "__content_embeddings"
            with open(kg_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "created_at": utc_now_iso(),
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "pipeline_id": pipeline_id,
                        "ragas_version": getattr(ragas, "__version__", "unknown"),
                        "docs_fingerprint": docs_fp,
                        "kg_cache_id": kg_id,
                        "provider": provider_used,
                        "llm_id": args.model,
                        "embedding_id": embedding_id,
                        "add_content_embeddings": add_content_embeddings,
                        "num_nodes": len(kg.nodes),
                        "num_relationships": len(kg.relationships),
                    },
                    f, ensure_ascii=False, indent=2,
                )

    # -----------------------------------------------------------------------
    # Generate or load personas (file-cached)
    # -----------------------------------------------------------------------
    steps.next("Generating personas")

    if cache_enabled:
        personas_cache_dir.mkdir(parents=True, exist_ok=True)

    personas_cache_path = (
        personas_cache_dir / f"personas_{kg_id}__n{int(args.num_personas)}.json"
    )

    personas: Optional[List[Persona]] = None
    if args.personas_path:
        personas_path = Path(args.personas_path).expanduser()
        if not personas_path.is_absolute():
            personas_path = (Path.cwd() / personas_path).resolve()
        print(f"  Loading from file: {personas_path}")
        personas = load_personas_from_file(personas_path)
        print(f"  Loaded {len(personas)} persona(s)")
    elif cache_enabled and personas_cache_path.exists() and not args.reprocess:
        print(f"  Loading cached personas: {personas_cache_path.name}")
        personas = load_personas_cache(personas_cache_path)
        print(f"  Loaded {len(personas)} persona(s)")
    else:
        print("  Generating personas from knowledge graph...")
        personas = generate_personas_from_kg(
            kg=kg, llm=generator_llm, num_personas=int(args.num_personas)
        )
        if cache_enabled:
            save_personas_cache(personas, personas_cache_path)
            print(f"  Saved personas cache: {personas_cache_path.name}")
    print(f"  Personas ready: {len(personas) if personas else 0}")

    # -----------------------------------------------------------------------
    # Generate testset
    # -----------------------------------------------------------------------
    steps.next("Generating testset")

    print(
        "  Standalone queries: "
        + ("enabled" if args.standalone_queries else "disabled (RAGAS defaults)")
    )

    query_distribution = build_query_distribution_for_pipeline(
        generator_llm,
        kg,
        standalone_queries=args.standalone_queries,
        llm_context=llm_context,
        pdf_profiles_by_source=(
            pdf_profiles_by_source if args.pdf_profiles else None
        ),
        query_mix=args.query_mix,
    )

    testset = generate_testset(
        kg,
        generator_llm,
        generator_embeddings,
        testset_size=args.testset_size,
        personas=personas,
        llm_context=llm_context,
        query_distribution=query_distribution,
    )

    if args.standalone_queries:
        warn_on_referential_queries(testset)

    # -----------------------------------------------------------------------
    # Mine hard negatives (optional)
    # -----------------------------------------------------------------------
    hard_negatives = None
    source_mappings = None
    if args.hard_negatives:
        steps.next("Mining hard negatives")

        raw_embedding_model = generator_embeddings
        judge_llm = getattr(generator_llm, "langchain_llm", None)

        testset_df = testset.to_pandas()
        hard_negatives, source_mappings = mine_hard_negatives_for_testset(
            testset_df,
            kg,
            all_docs,
            raw_embedding_model,
            judge_llm=judge_llm,
            num_bm25_negatives=args.num_bm25_negatives,
            num_embedding_negatives=args.num_embedding_negatives,
        )

    # -----------------------------------------------------------------------
    # FINAL STEP: Save results to output/
    # -----------------------------------------------------------------------
    steps.next("Saving results")

    output_path = str(OUTPUT_DIR / args.output)
    df = save_testset(
        testset,
        output_path,
        formats=args.output_formats,
        docs=all_docs,
        hard_negatives=hard_negatives,
        source_mappings=source_mappings,
    )

    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"  Total samples: {len(df)}")
    if args.hard_negatives:
        print("  Hard negatives: included")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
