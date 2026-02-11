"""Knowledge graph building, testset generation/saving, and persona management."""

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import ragas
from langchain_core.documents import Document
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship, UUIDEncoder
from ragas.testset.persona import Persona, generate_personas_from_kg
from ragas.testset.synthesizers.testset_schema import Testset
from ragas.testset.transforms import apply_transforms, default_transforms

import pandas as pd

from .config import REFERENTIAL_QUERY_RE, RAGAS_DOC_EXTRACT_CACHE_VERSION
from .db import pdf_store_persist_ragas_extractions
from .hard_negatives import find_source_files
from .transforms import patch_transforms_with_safe_splitter
from .utils import parse_reference_contexts, strip_hop_prefix, utc_now_iso


# ---------------------------------------------------------------------------
# Knowledge graph building
# ---------------------------------------------------------------------------
def build_knowledge_graph(
    docs: List[Document],
    generator_llm,
    generator_embeddings,
    *,
    add_content_embeddings: bool = True,
    pdf_store_conn=None,
    base_input_dir: Optional[Path] = None,
    reuse_cached_store_extractions: bool = True,
    ragas_doc_extraction_model_tag_base: Optional[str] = None,
) -> KnowledgeGraph:
    """Build a RAGAS KnowledgeGraph from loaded documents by applying transforms."""
    print("  Applying RAGAS transforms (LLM extraction + embeddings)...")

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
        pdf_store_conn=pdf_store_conn,
        base_input_dir=base_input_dir,
        reuse_cached_store_extractions=reuse_cached_store_extractions,
        ragas_doc_extraction_model_tag_base=ragas_doc_extraction_model_tag_base,
    )
    print("  Headline-only generation enabled")
    if add_content_embeddings:
        print("  Page content embeddings enabled (100% node coverage)")

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
    print(
        f"  Knowledge graph ready: {len(kg.nodes)} nodes, "
        f"{len(kg.relationships)} relationships"
    )

    # Persist per-page LLM extractions back into SQLite for reuse
    if (
        pdf_store_conn is not None
        and base_input_dir is not None
        and isinstance(ragas_doc_extraction_model_tag_base, str)
        and ragas_doc_extraction_model_tag_base.strip()
    ):
        try:
            counts = pdf_store_persist_ragas_extractions(
                pdf_store_conn,
                kg=kg,
                base_input_dir=base_input_dir,
                headlines_model_tag=f"{ragas_doc_extraction_model_tag_base}:headlines:max5",
                summary_model_tag=f"{ragas_doc_extraction_model_tag_base}:summary",
                entities_model_tag=f"{ragas_doc_extraction_model_tag_base}:entities",
                themes_model_tag=f"{ragas_doc_extraction_model_tag_base}:themes",
            )
            persisted = {k: v for k, v in counts.items() if v}
            if persisted:
                parts = ", ".join(f"{k}+={v}" for k, v in persisted.items())
                print(f"  SQLite store: cached RAGAS extractions ({parts})")
        except Exception:
            pass

    return kg


# ---------------------------------------------------------------------------
# KG cache (file-based — not in SQLite)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Persona cache (file-based — not in SQLite)
# ---------------------------------------------------------------------------
def save_personas_cache(personas: List[Persona], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in personas], f, ensure_ascii=False, indent=2)


def load_personas_cache(path: Path) -> List[Persona]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Persona(**p) for p in data]


def load_personas_from_file(path: Path) -> List[Persona]:
    """Load a persona list from a JSON or JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Personas file not found: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Personas file is empty: {path}")

    items: List[Any] = []
    if path.suffix.lower() == ".jsonl":
        for i, line in enumerate(raw.splitlines(), start=1):
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}") from e
    else:
        data = json.loads(raw)
        if isinstance(data, dict) and "personas" in data:
            data = data["personas"]
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list (or object with 'personas') in {path}, "
                f"got {type(data).__name__}"
            )
        items = data

    personas: List[Persona] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid persona entry at index {idx} in {path}: "
                f"expected object, got {type(item).__name__}"
            )
        try:
            p = Persona(**item)
        except Exception as e:
            raise ValueError(f"Invalid persona entry at index {idx} in {path}: {e}") from e
        if not str(p.name or "").strip():
            raise ValueError(f"Persona at index {idx} has empty 'name' in {path}")
        if not str(p.role_description or "").strip():
            raise ValueError(f"Persona '{p.name}' has empty 'role_description' in {path}")
        personas.append(p)

    # De-dupe by name
    deduped: Dict[str, Persona] = {}
    for p in personas:
        if p.name not in deduped:
            deduped[p.name] = p
    out = list(deduped.values())
    if not out:
        raise ValueError(f"No valid personas found in {path}")
    return out


# ---------------------------------------------------------------------------
# Testset generation
# ---------------------------------------------------------------------------
def generate_testset(
    kg: KnowledgeGraph,
    generator_llm,
    generator_embeddings,
    *,
    testset_size: int = 50,
    personas: Optional[List[Persona]] = None,
    llm_context: Optional[str] = None,
    query_distribution=None,
) -> Testset:
    """Generate a synthetic testset from a pre-built KnowledgeGraph."""
    print("  Initializing TestsetGenerator...")
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
        llm_context=llm_context,
    )

    print(f"  Generating {testset_size} samples (this may take a while)...")
    return generator.generate(
        testset_size=testset_size,
        query_distribution=query_distribution,
    )


def warn_on_referential_queries(testset: Testset, *, limit: int = 5) -> None:
    """Warn if generated queries look doc-internal (deictic references)."""
    try:
        df = testset.to_pandas()
    except Exception:
        return
    if "user_input" not in df.columns:
        return

    bad = []
    for q in df["user_input"].tolist():
        qs = str(q or "")
        if REFERENTIAL_QUERY_RE.search(qs):
            bad.append(qs)

    if not bad:
        return

    print(
        f"\n  Warning: {len(bad)}/{len(df)} query(ies) look referential "
        '(e.g., "this case", "the above"). Consider tightening corpus-query guidance.'
    )
    for i, q in enumerate(bad[: max(1, int(limit))], start=1):
        print(f"    Referenced query {i}: {q}")


# ---------------------------------------------------------------------------
# Source mapping helpers (shared between legacy and world-mode pipelines)
# ---------------------------------------------------------------------------
def _build_kg_page_content_index(
    kg: KnowledgeGraph,
) -> Dict[str, Dict[str, Any]]:
    """Build a lookup from normalised page_content → source metadata.

    Only DOCUMENT nodes are used (they carry page-level metadata).
    The key is the lowercased, stripped ``page_content``; the value is a
    dict with ``source`` (absolute path) and ``page`` (0-indexed).
    """
    index: Dict[str, Dict[str, Any]] = {}
    for node in kg.nodes:
        if node.type != NodeType.DOCUMENT:
            continue
        md = node.get_property("document_metadata")
        if not isinstance(md, dict):
            continue
        pc = str(node.get_property("page_content") or "").lower().strip()
        if not pc:
            continue
        source = md.get("source")
        page = md.get("page")
        if source is None:
            continue
        # First node wins (should be unique per page anyway)
        if pc not in index:
            index[pc] = {"source": str(source), "page": page}
    return index


def _find_sources_from_kg(
    reference_contexts: List[str],
    kg: KnowledgeGraph,
    *,
    _kg_index: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Map reference contexts back to their KG source nodes.

    RAGAS builds each multi-hop query from specific KG nodes.  Each
    DOCUMENT node carries ``document_metadata`` with ``source`` (path)
    and ``page`` (0-indexed).  Since the ``reference_contexts`` text is
    taken verbatim from those nodes, we can recover the exact source by
    matching the context text against the KG node ``page_content``.

    Matching strategy (per hop context, after stripping ``<N-hop>``):
      1. Exact match — context text == node page_content.
      2. Containment — context is a substring of a node, or vice-versa;
         pick the node with the largest overlap.
      3. Head-chunk fallback — first 150 chars of context found in a node.

    Returns at most **one source page per hop context**.
    """
    if _kg_index is None:
        _kg_index = _build_kg_page_content_index(kg)

    all_sources: List[str] = []
    all_sources_with_pages: List[str] = []
    all_page_numbers: List[Any] = []
    seen_pairs: set = set()

    for raw_context in reference_contexts:
        ctx = strip_hop_prefix(str(raw_context)).lower().strip()
        if not ctx:
            continue

        # --- Strategy 1: exact match ---
        info = _kg_index.get(ctx)
        if info is not None:
            _append_source(info, all_sources, all_sources_with_pages,
                           all_page_numbers, seen_pairs)
            continue

        # --- Strategy 2: containment (best overlap) ---
        best_info: Optional[Dict[str, Any]] = None
        best_overlap = 0
        for pc, node_info in _kg_index.items():
            overlap = 0
            if ctx in pc:
                overlap = len(ctx)
            elif pc in ctx:
                overlap = len(pc)
            if overlap > best_overlap:
                best_overlap = overlap
                best_info = node_info

        if best_info is not None:
            _append_source(best_info, all_sources, all_sources_with_pages,
                           all_page_numbers, seen_pairs)
            continue

        # --- Strategy 3: head-chunk fallback ---
        if len(ctx) > 150:
            head = ctx[:150]
            for pc, node_info in _kg_index.items():
                if head in pc:
                    _append_source(node_info, all_sources,
                                   all_sources_with_pages,
                                   all_page_numbers, seen_pairs)
                    break  # first match is good enough

    return {
        "sources": all_sources or ["unknown"],
        "sources_with_pages": all_sources_with_pages or ["unknown"],
        "page_numbers": all_page_numbers or [None],
    }


def _append_source(
    info: Dict[str, Any],
    all_sources: List[str],
    all_sources_with_pages: List[str],
    all_page_numbers: List[Any],
    seen_pairs: set,
) -> None:
    """Helper to append a single source entry (de-duplicated)."""
    source = info.get("source", "unknown")
    filename = os.path.basename(source)
    page = info.get("page")
    page_display = (page + 1) if isinstance(page, int) else None
    swp = f"{filename} (page {page_display})" if page_display else filename

    pair = (filename, page_display)
    if pair not in seen_pairs:
        seen_pairs.add(pair)
        if filename not in all_sources:
            all_sources.append(filename)
        all_sources_with_pages.append(swp)
        all_page_numbers.append(page_display)


def add_source_mapping_columns(
    df: pd.DataFrame,
    docs: List[Document],
    *,
    source_mappings: Optional[List[Dict[str, Any]]] = None,
    strict: bool = False,
    kg: Optional[KnowledgeGraph] = None,
) -> pd.DataFrame:
    """Add source_files, source_files_with_pages, page_numbers columns to *df*.

    If *source_mappings* is provided (pre-computed ``find_source_files`` results),
    it is reused; otherwise the mapping is computed from *docs*.

    When *strict* is True (recommended for multi-hop RAGAS output), source
    pages are resolved by matching reference contexts against KG DOCUMENT
    nodes (which carry exact ``source`` and ``page`` metadata).  This is
    both faster and more accurate than the fuzzy text-matching fallback.
    A *kg* must be provided when *strict* is True.
    """
    source_files_list = []
    source_files_with_pages_list = []
    page_numbers_list = []

    if source_mappings and len(source_mappings) == len(df):
        for result in source_mappings:
            source_files_list.append(result["sources"])
            source_files_with_pages_list.append(result["sources_with_pages"])
            page_numbers_list.append(result["page_numbers"])
    else:
        # Pre-build the KG index once if using strict mode
        kg_index = None
        if strict and kg is not None:
            kg_index = _build_kg_page_content_index(kg)

        for _, row in df.iterrows():
            contexts = parse_reference_contexts(row.get("reference_contexts"))
            if strict and kg is not None:
                result = _find_sources_from_kg(
                    contexts if isinstance(contexts, list) else [contexts],
                    kg,
                    _kg_index=kg_index,
                )
            else:
                result = find_source_files(
                    contexts if isinstance(contexts, list) else [contexts],
                    docs,
                )
            source_files_list.append(result["sources"])
            source_files_with_pages_list.append(result["sources_with_pages"])
            page_numbers_list.append(result["page_numbers"])

    df["source_files"] = [json.dumps(f) for f in source_files_list]
    df["source_files_with_pages"] = [
        json.dumps(f) for f in source_files_with_pages_list
    ]
    df["page_numbers"] = [json.dumps(p) for p in page_numbers_list]
    df["source_files_readable"] = [
        ", ".join(f) if f != ["unknown"] else "unknown"
        for f in source_files_list
    ]
    df["source_files_with_pages_readable"] = [
        ", ".join(f) if f != ["unknown"] else "unknown"
        for f in source_files_with_pages_list
    ]
    return df


# ---------------------------------------------------------------------------
# Save testset
# ---------------------------------------------------------------------------
def save_testset(
    testset,
    output_path: str,
    formats: List[str] = ["csv", "json"],
    docs: Optional[List[Document]] = None,
    hard_negatives: Optional[List[List[str]]] = None,
    source_mappings: Optional[List[Dict[str, Any]]] = None,
):
    """Save the generated testset to files in the output directory.

    Args:
        source_mappings: Pre-computed find_source_files results per row
            (from hard negative mining). If provided, skips recomputing.
    """
    df = testset.to_pandas()

    if docs:
        print("  Mapping reference contexts to source files...")
        source_files_list = []
        source_files_with_pages_list = []
        page_numbers_list = []

        if source_mappings and len(source_mappings) == len(df):
            # Reuse pre-computed source mappings from hard negative mining
            for result in source_mappings:
                source_files_list.append(result["sources"])
                source_files_with_pages_list.append(result["sources_with_pages"])
                page_numbers_list.append(result["page_numbers"])
        else:
            for _, row in df.iterrows():
                contexts = parse_reference_contexts(row.get("reference_contexts"))
                result = find_source_files(
                    contexts if isinstance(contexts, list) else [contexts], docs
                )
                source_files_list.append(result["sources"])
                source_files_with_pages_list.append(result["sources_with_pages"])
                page_numbers_list.append(result["page_numbers"])

        df["source_files"] = [json.dumps(f) for f in source_files_list]
        df["source_files_with_pages"] = [
            json.dumps(f) for f in source_files_with_pages_list
        ]
        df["page_numbers"] = [json.dumps(p) for p in page_numbers_list]
        df["source_files_readable"] = [
            ", ".join(f) if f != ["unknown"] else "unknown"
            for f in source_files_list
        ]
        df["source_files_with_pages_readable"] = [
            ", ".join(f) if f != ["unknown"] else "unknown"
            for f in source_files_with_pages_list
        ]

    if hard_negatives is not None:
        df["hard_negatives"] = [
            json.dumps(negs, ensure_ascii=False) for negs in hard_negatives
        ]
        neg_count = sum(len(negs) for negs in hard_negatives)
        print(
            f"  Added {neg_count} hard negatives "
            f"({neg_count / max(len(df), 1):.1f} avg per query)"
        )

    print(f"  Generated {len(df)} test samples")
    print("\n  Sample preview:")
    preview_cols = (
        ["user_input", "source_files_with_pages_readable"]
        if "source_files_with_pages_readable" in df.columns
        else ["user_input"]
    )
    print(df[preview_cols].head())

    _save_df_to_formats(df, output_path, formats)
    return df


def _save_df_to_formats(
    df: pd.DataFrame, output_path: str, formats: List[str],
) -> None:
    """Write a DataFrame to disk in each requested format."""
    for fmt in formats:
        file_path = f"{output_path}.{fmt}"
        if fmt == "csv":
            df.to_csv(file_path, index=False)
        elif fmt == "json":
            df.to_json(file_path, orient="records", indent=2)
        elif fmt == "parquet":
            df.to_parquet(file_path, index=False)
        else:
            print(f"  Warning: Unknown format '{fmt}', skipping...")
            continue
        print(f"  Saved: {file_path}")


def save_combined_dataframe(
    df: pd.DataFrame,
    output_path: str,
    formats: List[str] = ["csv", "json"],
) -> pd.DataFrame:
    """Save a pre-built combined DataFrame (world-mode pipeline).

    Unlike ``save_testset``, this assumes source mapping columns are already
    present (single-hop rows have them; multi-hop rows get them per-world).
    """
    print(f"  Total samples: {len(df)}")
    print("\n  Sample preview:")
    preview_cols = (
        ["user_input", "synthesizer_name", "source_files_with_pages_readable"]
        if "source_files_with_pages_readable" in df.columns
        else ["user_input", "synthesizer_name"]
        if "synthesizer_name" in df.columns
        else ["user_input"]
    )
    available = [c for c in preview_cols if c in df.columns]
    print(df[available].head())

    _save_df_to_formats(df, output_path, formats)
    return df
