"""RAGAS transform patches for headline-only generation and safe splitting."""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms import HeadlinesExtractor, NodeFilter, SummaryExtractor
from ragas.testset.transforms.extractors import EmbeddingExtractor
from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder
from ragas.testset.transforms.splitters import HeadlineSplitter

from .db import pdf_store_load_cached_pairs
from .utils import compute_rel_path_for_store


# ---------------------------------------------------------------------------
# Safe headline splitter (handles missing headlines gracefully)
# ---------------------------------------------------------------------------
class SafeHeadlineSplitter(HeadlineSplitter):
    """HeadlineSplitter that skips splitting when headlines are missing."""

    async def split(self, node: Node) -> Tuple[List[Node], List[Relationship]]:
        headlines = node.get_property("headlines")
        if not headlines:
            return [node], []

        nodes, relationships = await super().split(node)

        # Propagate source metadata into CHUNK nodes
        parent_md = node.get_property("document_metadata")
        if parent_md is not None:
            for n in nodes:
                if n.type == NodeType.CHUNK and n.get_property("document_metadata") is None:
                    if isinstance(parent_md, dict):
                        n.add_property("document_metadata", dict(parent_md))
                    else:
                        n.add_property("document_metadata", parent_md)

        return nodes, relationships


# ---------------------------------------------------------------------------
# Filter: require usable headlines
# ---------------------------------------------------------------------------
@dataclass
class HeadlinesRequiredFilter(NodeFilter):
    """Remove DOCUMENT nodes without usable headlines."""

    require_match_in_text: bool = True
    case_insensitive_match: bool = True

    async def custom_filter(self, node: Node, kg) -> bool:  # type: ignore[override]
        headlines = node.get_property("headlines")
        if not isinstance(headlines, list):
            return True

        cleaned = [h.strip() for h in headlines if isinstance(h, str) and h.strip()]
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

        return True


# ---------------------------------------------------------------------------
# SQLite-cached extractors (avoids re-calling the LLM for the same pages)
# ---------------------------------------------------------------------------
@dataclass
class SQLiteCachedHeadlinesExtractor(HeadlinesExtractor):
    """HeadlinesExtractor that caches per-page extractions in SQLite."""

    conn: Optional[sqlite3.Connection] = field(default=None, repr=False)
    base_input_dir: Optional[Path] = None
    model_tag: Optional[str] = None

    _store_value_col: str = "ragas_headlines_json"
    _store_model_col: str = "ragas_headlines_model"

    def _node_store_key(self, node: Node) -> Optional[Tuple[str, int]]:
        md = node.get_property("document_metadata")
        if not isinstance(md, dict):
            return None
        source = md.get("source")
        if not isinstance(source, str) or not source.strip():
            return None
        page0 = md.get("page")
        try:
            page_number = int(page0) + 1
        except Exception:
            return None
        if page_number < 1 or self.base_input_dir is None:
            return None
        try:
            rel_path = compute_rel_path_for_store(Path(source), self.base_input_dir)
        except Exception:
            return None
        return (str(rel_path), int(page_number))

    def generate_execution_plan(self, kg: KnowledgeGraph) -> Any:  # type: ignore[override]
        filtered = self.filter(kg)

        nodes_with_key: List[Tuple[Node, Tuple[str, int]]] = []
        nodes_missing_key: List[Node] = []
        for node in filtered.nodes:
            if node.get_property(self.property_name) is not None:
                continue
            key = self._node_store_key(node)
            if key is None:
                nodes_missing_key.append(node)
            else:
                nodes_with_key.append((node, key))

        cache: Dict[Tuple[str, int], Tuple[Optional[str], Optional[str]]] = {}
        if (
            self.conn is not None
            and isinstance(self.model_tag, str) and self.model_tag.strip()
            and nodes_with_key
        ):
            rel_paths = sorted({k[0] for _, k in nodes_with_key})
            cache = pdf_store_load_cached_pairs(
                self.conn,
                rel_paths=rel_paths,
                value_col=self._store_value_col,
                model_col=self._store_model_col,
            )

        nodes_to_extract: List[Node] = list(nodes_missing_key)
        for node, key in nodes_with_key:
            cached = cache.get(key)
            if not cached:
                nodes_to_extract.append(node)
                continue
            blob, model = cached
            if not isinstance(model, str) or model.strip() != str(self.model_tag):
                nodes_to_extract.append(node)
                continue
            if not isinstance(blob, str) or not blob.strip():
                nodes_to_extract.append(node)
                continue
            try:
                parsed = json.loads(blob)
            except Exception:
                nodes_to_extract.append(node)
                continue
            if not isinstance(parsed, list):
                nodes_to_extract.append(node)
                continue
            node.add_property(self.property_name, parsed)

        async def apply_extract(node: Node):
            property_name, property_value = await self.extract(node)
            if node.get_property(property_name) is None:
                node.add_property(property_name, property_value)

        return [apply_extract(node) for node in nodes_to_extract]


@dataclass
class SQLiteCachedSummaryExtractor(SummaryExtractor):
    """SummaryExtractor that caches per-page summaries in SQLite."""

    conn: Optional[sqlite3.Connection] = field(default=None, repr=False)
    base_input_dir: Optional[Path] = None
    model_tag: Optional[str] = None

    _store_value_col: str = "ragas_summary"
    _store_model_col: str = "ragas_summary_model"

    def _node_store_key(self, node: Node) -> Optional[Tuple[str, int]]:
        md = node.get_property("document_metadata")
        if not isinstance(md, dict):
            return None
        source = md.get("source")
        if not isinstance(source, str) or not source.strip():
            return None
        page0 = md.get("page")
        try:
            page_number = int(page0) + 1
        except Exception:
            return None
        if page_number < 1 or self.base_input_dir is None:
            return None
        try:
            rel_path = compute_rel_path_for_store(Path(source), self.base_input_dir)
        except Exception:
            return None
        return (str(rel_path), int(page_number))

    def generate_execution_plan(self, kg: KnowledgeGraph) -> Any:  # type: ignore[override]
        filtered = self.filter(kg)

        nodes_with_key: List[Tuple[Node, Tuple[str, int]]] = []
        nodes_missing_key: List[Node] = []
        for node in filtered.nodes:
            if node.get_property(self.property_name) is not None:
                continue
            key = self._node_store_key(node)
            if key is None:
                nodes_missing_key.append(node)
            else:
                nodes_with_key.append((node, key))

        cache: Dict[Tuple[str, int], Tuple[Optional[str], Optional[str]]] = {}
        if (
            self.conn is not None
            and isinstance(self.model_tag, str) and self.model_tag.strip()
            and nodes_with_key
        ):
            rel_paths = sorted({k[0] for _, k in nodes_with_key})
            cache = pdf_store_load_cached_pairs(
                self.conn,
                rel_paths=rel_paths,
                value_col=self._store_value_col,
                model_col=self._store_model_col,
            )

        nodes_to_extract: List[Node] = list(nodes_missing_key)
        for node, key in nodes_with_key:
            cached = cache.get(key)
            if not cached:
                nodes_to_extract.append(node)
                continue
            blob, model = cached
            if not isinstance(model, str) or model.strip() != str(self.model_tag):
                nodes_to_extract.append(node)
                continue
            if not isinstance(blob, str):
                nodes_to_extract.append(node)
                continue
            node.add_property(self.property_name, blob)

        async def apply_extract(node: Node):
            property_name, property_value = await self.extract(node)
            if node.get_property(property_name) is None:
                node.add_property(property_name, property_value)

        return [apply_extract(node) for node in nodes_to_extract]


# ---------------------------------------------------------------------------
# Build patched transform list
# ---------------------------------------------------------------------------
def patch_transforms_with_safe_splitter(
    transforms,
    llm,
    embedding_model,
    *,
    add_content_embeddings: bool = True,
    pdf_store_conn: Optional[sqlite3.Connection] = None,
    base_input_dir: Optional[Path] = None,
    reuse_cached_store_extractions: bool = True,
    ragas_doc_extraction_model_tag_base: Optional[str] = None,
):
    """
    Patch RAGAS transforms:
    - Replace HeadlineSplitter with SafeHeadlineSplitter
    - Filter out DOCUMENT nodes without usable headlines
    - Add page_content embeddings for all nodes (optional)
    - Add content_similarity edges
    """
    def doc_nodes_only(node: Node) -> bool:
        return node.type == NodeType.DOCUMENT

    def all_nodes_with_content(node: Node) -> bool:
        return node.get_property("page_content") is not None

    base_transforms = [t for t in transforms if not isinstance(t, HeadlinesExtractor)]

    use_store_cache = bool(
        reuse_cached_store_extractions
        and pdf_store_conn is not None
        and base_input_dir is not None
        and isinstance(ragas_doc_extraction_model_tag_base, str)
        and ragas_doc_extraction_model_tag_base.strip()
    )

    if use_store_cache:
        headlines_tag = f"{ragas_doc_extraction_model_tag_base}:headlines:max5"
        headlines_extractor = SQLiteCachedHeadlinesExtractor(
            llm=llm,
            filter_nodes=doc_nodes_only,
            conn=pdf_store_conn,
            base_input_dir=base_input_dir,
            model_tag=headlines_tag,
        )
    else:
        headlines_extractor = HeadlinesExtractor(llm=llm, filter_nodes=doc_nodes_only)

    patched = [
        headlines_extractor,
        HeadlinesRequiredFilter(filter_nodes=doc_nodes_only),
    ]

    for transform in base_transforms:
        if isinstance(transform, HeadlineSplitter):
            safe_splitter = SafeHeadlineSplitter(
                min_tokens=transform.min_tokens,
                max_tokens=transform.max_tokens,
                filter_nodes=transform.filter_nodes,
            )
            patched.append(safe_splitter)
        elif use_store_cache and isinstance(transform, SummaryExtractor):
            summary_tag = f"{ragas_doc_extraction_model_tag_base}:summary"
            patched.append(
                SQLiteCachedSummaryExtractor(
                    name=transform.name,
                    filter_nodes=transform.filter_nodes,
                    llm=transform.llm,
                    merge_if_possible=transform.merge_if_possible,
                    max_token_limit=transform.max_token_limit,
                    tokenizer=transform.tokenizer,
                    property_name=transform.property_name,
                    prompt=transform.prompt,
                    conn=pdf_store_conn,
                    base_input_dir=base_input_dir,
                    model_tag=summary_tag,
                )
            )
        else:
            patched.append(transform)

    if add_content_embeddings:
        page_content_embedder = EmbeddingExtractor(
            embedding_model=embedding_model,
            property_name="page_content_embedding",
            embed_property_name="page_content",
            filter_nodes=all_nodes_with_content,
        )
        patched.append(page_content_embedder)

        content_similarity_builder = CosineSimilarityBuilder(
            property_name="page_content_embedding",
            new_property_name="content_similarity",
            threshold=0.5,
        )
        patched.append(content_similarity_builder)

    return patched
