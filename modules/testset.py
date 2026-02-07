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

from .config import REFERENTIAL_QUERY_RE, RAGAS_DOC_EXTRACT_CACHE_VERSION
from .db import pdf_store_persist_ragas_extractions
from .hard_negatives import find_source_files
from .transforms import patch_transforms_with_safe_splitter
from .utils import parse_reference_contexts, utc_now_iso


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
            )
            if counts.get("headlines") or counts.get("summary"):
                print(
                    f"  SQLite store: cached RAGAS extractions "
                    f"(headlines+={counts.get('headlines', 0)}, "
                    f"summary+={counts.get('summary', 0)})"
                )
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
# Save testset
# ---------------------------------------------------------------------------
def save_testset(
    testset,
    output_path: str,
    formats: List[str] = ["csv", "json"],
    docs: Optional[List[Document]] = None,
    hard_negatives: Optional[List[List[str]]] = None,
):
    """Save the generated testset to files in the output directory."""
    df = testset.to_pandas()

    if docs:
        print("  Mapping reference contexts to source files...")
        source_files_list = []
        source_files_with_pages_list = []
        page_numbers_list = []

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

    return df
