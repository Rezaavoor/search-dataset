"""PDF-profile-aware query synthesizers and query distribution building."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ragas.dataset_schema import SingleTurnSample
from ragas.testset.graph import KnowledgeGraph, Node
from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.prompts import (
    QueryConditions as MultiHopQueryConditions,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.prompts import (
    QueryCondition as SingleHopQueryCondition,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

from .config import (
    CORPUS_MULTI_HOP_PROMPT_INSTRUCTION,
    CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION,
    DEFAULT_CORPUS_SIZE_HINT,
)
from .profiles import build_llm_context_with_pdf_profiles
from .utils import safe_resolve_path_str


# ---------------------------------------------------------------------------
# PDF-profile-aware synthesizers
# ---------------------------------------------------------------------------
@dataclass
class PdfProfileSingleHopSpecificQuerySynthesizer(SingleHopSpecificQuerySynthesizer):
    pdf_profiles_by_source: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _profiles_for_sources(
        self, sources: List[str], *, max_profiles: int = 1
    ) -> List[Dict[str, Any]]:
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
        source = safe_resolve_path_str(md.get("source")) if isinstance(md, dict) else None
        profiles = self._profiles_for_sources(
            [source] if source else [], max_profiles=1
        )
        dynamic_llm_context = (
            build_llm_context_with_pdf_profiles(
                base_llm_context=self.llm_context, profiles=profiles,
            )
            if profiles
            else self.llm_context
        )

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

    def _profiles_for_nodes(
        self, nodes: List[Node], *, max_profiles: int = 2
    ) -> List[Dict[str, Any]]:
        sources: List[str] = []
        for n in nodes:
            md = n.get_property("document_metadata") or {}
            src = safe_resolve_path_str(md.get("source")) if isinstance(md, dict) else None
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
        dynamic_llm_context = (
            build_llm_context_with_pdf_profiles(
                base_llm_context=self.llm_context, profiles=profiles,
            )
            if profiles
            else self.llm_context
        )

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

    def _profiles_for_nodes(
        self, nodes: List[Node], *, max_profiles: int = 2
    ) -> List[Dict[str, Any]]:
        sources: List[str] = []
        for n in nodes:
            md = n.get_property("document_metadata") or {}
            src = safe_resolve_path_str(md.get("source")) if isinstance(md, dict) else None
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
        dynamic_llm_context = (
            build_llm_context_with_pdf_profiles(
                base_llm_context=self.llm_context, profiles=profiles,
            )
            if profiles
            else self.llm_context
        )

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


# ---------------------------------------------------------------------------
# Corpus-level LLM context
# ---------------------------------------------------------------------------
def build_corpus_llm_context(*, corpus_size_hint: Optional[int] = None) -> str:
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
        '- Avoid deictic references like "this case", "this document", "the above", '
        '"herein", etc.\n'
        "- Include concrete identifiers from the provided context to disambiguate "
        '(e.g., case caption with "v.", party/organization names, court/jurisdiction, '
        "docket/case number, statute/section, property address, agreement name, "
        "distinctive dates).\n"
        "- Do NOT mention filenames or page numbers.\n"
    )


# ---------------------------------------------------------------------------
# Query distribution builder
# ---------------------------------------------------------------------------

# Registry of synthesizer specs
_QUERY_SYNTHESIZER_SPECS: Dict[str, Dict[str, Any]] = {
    "single_hop_entities": {
        "description": "Single-hop specific queries driven by `entities`.",
        "cls": SingleHopSpecificQuerySynthesizer,
        "pdf_cls": PdfProfileSingleHopSpecificQuerySynthesizer,
        "kwargs": {"property_name": "entities"},
    },
    "single_hop_themes": {
        "description": "Single-hop specific queries driven by `themes`.",
        "cls": SingleHopSpecificQuerySynthesizer,
        "pdf_cls": PdfProfileSingleHopSpecificQuerySynthesizer,
        "kwargs": {"property_name": "themes"},
    },
    "multi_hop_abstract_summary": {
        "description": "Multi-hop abstract queries using summary similarity edges.",
        "cls": MultiHopAbstractQuerySynthesizer,
        "pdf_cls": PdfProfileMultiHopAbstractQuerySynthesizer,
        "kwargs": {
            "relation_property": "summary_similarity",
            "abstract_property_name": "themes",
        },
    },
    "multi_hop_abstract_content": {
        "description": "Multi-hop abstract queries using content similarity edges.",
        "cls": MultiHopAbstractQuerySynthesizer,
        "pdf_cls": PdfProfileMultiHopAbstractQuerySynthesizer,
        "kwargs": {
            "relation_property": "content_similarity",
            "abstract_property_name": "themes",
        },
    },
    "multi_hop_specific_entities": {
        "description": "Multi-hop specific queries using entity-overlap edges.",
        "cls": MultiHopSpecificQuerySynthesizer,
        "pdf_cls": PdfProfileMultiHopSpecificQuerySynthesizer,
        "kwargs": {
            "property_name": "entities",
            "relation_type": "entities_overlap",
            "relation_overlap_property": "overlapped_items",
        },
    },
}

_QUERY_SYNTHESIZER_ALIASES: Dict[str, str] = {
    "single_hop_specific_query_synthesizer": "single_hop_entities",
    "multi_hop_abstract_query_synthesizer": "multi_hop_abstract_summary",
    "multi_hop_specific_query_synthesizer": "multi_hop_specific_entities",
}


def list_query_synthesizers() -> List[Tuple[str, str]]:
    """Return (name, description) pairs for --list-query-synthesizers."""
    return [(k, v["description"]) for k, v in _QUERY_SYNTHESIZER_SPECS.items()]


def build_query_distribution_for_pipeline(
    llm,
    kg: KnowledgeGraph,
    *,
    standalone_queries: bool,
    llm_context: Optional[str],
    pdf_profiles_by_source: Optional[Dict[str, Dict[str, Any]]] = None,
    query_mix: Optional[List[str]] = None,
):
    """Build a query_distribution for RAGAS TestsetGenerator."""

    use_pdf_profiles = bool(pdf_profiles_by_source)
    if query_mix is None and not standalone_queries and not use_pdf_profiles:
        return None

    from ragas.testset.synthesizers import default_query_distribution
    from ragas.testset.synthesizers.single_hop.prompts import (
        QueryAnswerGenerationPrompt as SingleHopQAPrompt,
    )
    from ragas.testset.synthesizers.multi_hop.prompts import (
        QueryAnswerGenerationPrompt as MultiHopQAPrompt,
    )

    def _normalize_key(key: str) -> str:
        k = str(key or "").strip().lower()
        return _QUERY_SYNTHESIZER_ALIASES.get(k, k)

    def _parse_query_mix(specs: List[str]) -> List[Tuple[str, float]]:
        ordered_keys: List[str] = []
        weights: Dict[str, float] = {}
        for raw in specs:
            spec = str(raw or "").strip()
            if not spec:
                continue
            if "=" in spec:
                name_part, weight_part = spec.split("=", 1)
                key = _normalize_key(name_part)
                try:
                    w = float(weight_part.strip())
                except Exception as e:
                    raise ValueError(f"Invalid weight for {name_part!r}: {weight_part!r}") from e
            else:
                key = _normalize_key(spec)
                w = 1.0
            if not key or key not in _QUERY_SYNTHESIZER_SPECS:
                raise ValueError(f"Unknown query synthesizer: {key!r}")
            if not (w > 0):
                raise ValueError(f"Weight must be > 0 for {key!r}, got {w}")
            if key not in weights:
                ordered_keys.append(key)
                weights[key] = 0.0
            weights[key] += w
        if not ordered_keys:
            raise ValueError("Empty --query-mix")
        return [(k, weights[k]) for k in ordered_keys]

    def _make_synth(key: str):
        spec = _QUERY_SYNTHESIZER_SPECS[key]
        cls = spec["pdf_cls"] if use_pdf_profiles else spec["cls"]
        kwargs = dict(spec.get("kwargs", {}) or {})
        kwargs.update({"llm": llm, "llm_context": llm_context, "name": key})
        if use_pdf_profiles:
            kwargs["pdf_profiles_by_source"] = pdf_profiles_by_source or {}
        return cls(**kwargs)

    # Build the distribution
    if query_mix:
        requested = _parse_query_mix(query_mix)
        candidates = [(_make_synth(key), weight) for key, weight in requested]
        available = []
        for query, weight in candidates:
            try:
                if query.get_node_clusters(kg):
                    available.append((query, weight))
                else:
                    print(f"  Warning: Skipping {getattr(query, 'name', type(query).__name__)} (incompatible)")
            except Exception as e:
                print(f"  Warning: Skipping {getattr(query, 'name', type(query).__name__)}: {e}")
        if not available:
            raise ValueError("No compatible query synthesizers for the KnowledgeGraph.")
        total = sum(w for _, w in available) or 1.0
        qd = [(q, w / total) for q, w in available]

    elif use_pdf_profiles:
        candidates_list = [
            PdfProfileSingleHopSpecificQuerySynthesizer(
                llm=llm, llm_context=llm_context,
                pdf_profiles_by_source=pdf_profiles_by_source or {},
            ),
            PdfProfileMultiHopAbstractQuerySynthesizer(
                llm=llm, llm_context=llm_context,
                pdf_profiles_by_source=pdf_profiles_by_source or {},
            ),
            PdfProfileMultiHopSpecificQuerySynthesizer(
                llm=llm, llm_context=llm_context,
                pdf_profiles_by_source=pdf_profiles_by_source or {},
            ),
        ]
        available_list = []
        for query in candidates_list:
            try:
                if query.get_node_clusters(kg):
                    available_list.append(query)
            except Exception as e:
                print(f"  Warning: Skipping {getattr(query, 'name', type(query).__name__)}: {e}")
        if not available_list:
            raise ValueError("No compatible query synthesizers for the KnowledgeGraph.")
        qd = [(q, 1 / len(available_list)) for q in available_list]

    else:
        qd = default_query_distribution(llm, kg, llm_context)

    # Patch prompts for standalone corpus-level queries
    patched = []
    for synthesizer, prob in qd:
        if standalone_queries:
            name = str(getattr(synthesizer, "name", "") or "").lower()
            if name.startswith("single_hop"):
                p = SingleHopQAPrompt()
                p.instruction = CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION
                synthesizer.generate_query_reference_prompt = p
            else:
                p = MultiHopQAPrompt()
                p.instruction = CORPUS_MULTI_HOP_PROMPT_INSTRUCTION
                synthesizer.generate_query_reference_prompt = p
        patched.append((synthesizer, prob))

    return patched
