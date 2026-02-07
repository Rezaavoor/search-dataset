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
from modules.hard_negatives import mine_hard_negatives_for_testset
from modules.llm_setup import setup_llm_and_embeddings
from modules.profiles import build_pdf_profiles_from_store
from modules.synthesizers import (
    build_corpus_llm_context,
    build_query_distribution_for_pipeline,
    list_query_synthesizers,
)
from modules.testset import (
    build_knowledge_graph,
    generate_testset,
    load_knowledge_graph_cache,
    load_personas_cache,
    load_personas_from_file,
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
    total_steps = 7  # collect, sync store, load docs, setup LLM, build KG, personas, save
    if args.pdf_profiles:
        total_steps += 1
    total_steps += 1  # generate testset
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
    # STEP 2: Initialize & sync SQLite PDF store
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

    # Detect provider early for store operations
    provider_used = args.provider
    if provider_used == "auto":
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider_used = "azure"
        elif os.environ.get("OPENAI_API_KEY"):
            provider_used = "openai"
        else:
            provider_used = "auto"

    store_embedding_model = None
    store_embedding_model_id: Optional[str] = None
    if want_embeddings:
        try:
            tmp_llm, tmp_emb = setup_llm_and_embeddings(args.model, args.provider)
            store_embedding_model = tmp_emb
            store_embedding_model_id = (
                os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
                               "text-embedding-ada-002")
                if provider_used == "azure" else "openai-default"
            )
        except ValueError as e:
            print(f"\n  Error setting up embeddings for store: {e}")
            return 1

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
    # STEP 3: Load documents from SQLite store
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
    # STEP 4: Setup LLM & embeddings
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
    if provider_used == "auto":
        if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
            provider_used = "azure"
        else:
            provider_used = "openai"

    embedding_id = (
        os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
                       "text-embedding-ada-002")
        if provider_used == "azure" else "openai-default"
    )

    ragas_doc_extraction_model_tag_base = (
        f"{provider_used}:{args.model}:ragas{getattr(ragas, '__version__', 'unknown')}:"
        f"doc_extract_v{int(RAGAS_DOC_EXTRACT_CACHE_VERSION)}"
    )
    reuse_cached_store_extractions = bool(not args.reprocess)

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

    # -----------------------------------------------------------------------
    # STEP 6: Build or load knowledge graph (file-cached)
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
        print(f"  Loaded KG: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
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
    # STEP 7: Generate or load personas (file-cached)
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
    # STEP 8: Generate testset
    # -----------------------------------------------------------------------
    steps.next("Generating testset")

    # Build query distribution
    llm_context: Optional[str] = None
    if args.standalone_queries:
        llm_context = build_corpus_llm_context(corpus_size_hint=args.corpus_size_hint)
        extra = str(args.query_llm_context or "").strip()
        if extra:
            llm_context = llm_context.rstrip() + "\n" + extra + "\n"
        print("  Standalone queries: enabled")
    else:
        extra = str(args.query_llm_context or "").strip()
        llm_context = extra if extra else None
        print("  Standalone queries: disabled (RAGAS defaults)")

    query_distribution = build_query_distribution_for_pipeline(
        generator_llm,
        kg,
        standalone_queries=args.standalone_queries,
        llm_context=llm_context,
        pdf_profiles_by_source=pdf_profiles_by_source if args.pdf_profiles else None,
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
    # STEP 9 (optional): Mine hard negatives
    # -----------------------------------------------------------------------
    hard_negatives = None
    if args.hard_negatives:
        steps.next("Mining hard negatives")

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
