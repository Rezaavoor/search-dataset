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
import argparse
import gzip
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
from dotenv import load_dotenv
import json

# Document loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
from langchain_core.documents import Document

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


def patch_transforms_with_safe_splitter(transforms, llm):
    """
    Patch RAGAS transforms to enforce headline-only generation:

    - Extract headlines for all DOCUMENT nodes
    - Filter out DOCUMENT nodes without usable headlines
    - Replace HeadlineSplitter with SafeHeadlineSplitter (no fallback chunking)
    """
    def doc_nodes_only(node: Node) -> bool:
        return node.type == NodeType.DOCUMENT

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
    return patched


CACHE_SCHEMA_VERSION = 1
PIPELINE_ID = "pdf_only__headlines_required"


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
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "knowledge_graph",
        "pipeline_id": PIPELINE_ID,
        "ragas_version": getattr(ragas, "__version__", "unknown"),
        "docs_cache_id": docs_cache_id,
        "provider": provider,
        "llm_id": llm_id,
        "embedding_id": embedding_id,
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
) -> KnowledgeGraph:
    """
    Build a RAGAS KnowledgeGraph from loaded documents by applying transforms.

    This is the expensive "processing" step (LLM extraction + embeddings + relationship building).
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
    transforms = patch_transforms_with_safe_splitter(transforms, llm=generator_llm)
    print("Headline-only generation enabled (skipping documents without usable headlines)")

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


def generate_testset_from_knowledge_graph(
    kg: KnowledgeGraph,
    generator_llm,
    generator_embeddings,
    *,
    testset_size: int = 50,
    personas: Optional[List[Persona]] = None,
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
    )

    print(f"Generating testset with {testset_size} samples...")
    print("This may take a while depending on testset size...")
    return generator.generate(testset_size=testset_size)


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

    for context in reference_contexts:
        # Normalize context for comparison
        context_normalized = context.lower().strip()

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

                if filename not in all_sources:
                    all_sources.append(filename)

    return {
        'sources': all_sources if all_sources else ["unknown"],
        'sources_with_pages': all_sources_with_pages if all_sources_with_pages else ["unknown"],
        'page_numbers': all_page_numbers if all_page_numbers else [None]
    }


def save_testset(
    testset,
    output_path: str,
    formats: List[str] = ["csv", "json"],
    docs: Optional[List[Document]] = None,
):
    """
    Save the generated testset to files.

    Args:
        testset: Generated RAGAS testset
        output_path: Base path for output files (without extension)
        formats: List of output formats ('csv', 'json', 'parquet')
        docs: Original documents for source file mapping
    """
    df = testset.to_pandas()

    # Add source file names if documents are provided
    if docs:
        print("\nMapping reference contexts to source files...")
        source_files_list = []
        source_files_with_pages_list = []
        page_numbers_list = []

        for idx, row in df.iterrows():
            # Parse reference_contexts - it might be a string representation of a list
            contexts = row["reference_contexts"]
            if isinstance(contexts, str):
                try:
                    contexts = json.loads(contexts.replace("'", '"'))
                except Exception:
                    contexts = [contexts]

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
        help="Maximum number of files to load (default: no limit)",
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
        # Load documents from all specified directories
        all_docs = []
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
    kg_cache_id = compute_kg_cache_id(
        docs_cache_id,
        provider=provider_used,
        llm_id=args.model,
        embedding_id=embedding_id,
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
        kg = build_knowledge_graph(all_docs, generator_llm, generator_embeddings)
        if cache_enabled:
            print(f"\nSaving knowledge graph cache: {kg_cache_path}")
            save_knowledge_graph_cache(kg, kg_cache_path)
            with open(kg_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "created_at": _utc_now_iso(),
                        "schema_version": CACHE_SCHEMA_VERSION,
                        "pipeline_id": PIPELINE_ID,
                        "ragas_version": getattr(ragas, "__version__", "unknown"),
                        "docs_cache_id": docs_cache_id,
                        "kg_cache_id": kg_cache_id,
                        "provider": provider_used,
                        "llm_id": args.model,
                        "embedding_id": embedding_id,
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

    # Generate testset from the (cached) knowledge graph
    testset = generate_testset_from_knowledge_graph(
        kg,
        generator_llm,
        generator_embeddings,
        testset_size=args.testset_size,
        personas=personas,
    )

    # Save results (pass docs for source file mapping)
    df = save_testset(testset, args.output, formats=args.output_formats, docs=all_docs)

    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Total samples generated: {len(df)}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
