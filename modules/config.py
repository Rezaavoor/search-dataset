"""Constants and defaults for the synthetic dataset generator."""

import re

# ---------------------------------------------------------------------------
# Cache / schema versions
# ---------------------------------------------------------------------------
CACHE_SCHEMA_VERSION = 2
PIPELINE_ID_BASE = "pdf_only__headlines_required"
RAGAS_DOC_EXTRACT_CACHE_VERSION = 1

# ---------------------------------------------------------------------------
# PDF profile defaults
# ---------------------------------------------------------------------------
PDF_PROFILE_SCHEMA_VERSION = 1
DEFAULT_PDF_PROFILE_MAX_PAGES = 3
DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE = 2500
DEFAULT_PDF_PROFILE_MAX_PROFILE_CHARS = 1800

# ---------------------------------------------------------------------------
# SQLite PDF store
# ---------------------------------------------------------------------------
PDF_STORE_SCHEMA_VERSION = 2
DEFAULT_PDF_STORE_DB_NAME = "pdf_page_store.sqlite"
EXTRACTIVE_SUMMARY_MODEL = "extractive_v1"

# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------
DEFAULT_CORPUS_SIZE_HINT = 7000

# ---------------------------------------------------------------------------
# PDF profile LLM context rules (injected into query-generation prompts)
# ---------------------------------------------------------------------------
PDF_PROFILE_LLM_CONTEXT_RULES = (
    "You may receive one or more PDF_PROFILES below (PDF-level metadata).\n"
    "Use PDF_PROFILES ONLY to make the SEARCH QUERY more realistic in a large corpus "
    "and to pick a plausible user intent.\n"
    "CRITICAL:\n"
    "- Do NOT mention file paths, directories, '.pdf', or raw filenames in the final query.\n"
    "- Do NOT copy filename stems verbatim if they contain underscores/odd punctuation; "
    "use them only as hints.\n"
    "- Prefer identifiers present in the provided context excerpt(s). You MAY use "
    "folder/collection hints or a cleaned-up\n"
    "  title guess from the profile to disambiguate, but keep it natural and do not "
    "rely on it for the answer.\n"
    "- The ANSWER must be supported ONLY by the provided context excerpt(s), not by "
    "PDF_PROFILES.\n"
)

# ---------------------------------------------------------------------------
# Corpus-level prompt instructions (standalone queries)
# ---------------------------------------------------------------------------
CORPUS_SINGLE_HOP_PROMPT_INSTRUCTION = (
    "Generate a single-hop query and answer based on the specified conditions "
    "(persona, term, style, length) and the provided context. Ensure the answer is "
    "entirely faithful to the context, using only the information directly from the "
    "provided context.\n\n"
    "### Search Query Constraints (CRITICAL)\n"
    "The query will be asked in a large corpus setting (thousands of PDFs). Therefore:\n"
    '- The query MUST be standalone as a first-turn query.\n'
    '- Do NOT write queries that assume a document/case is already selected '
    '(avoid: "this case", "this document", "the above", "herein", "the parties" '
    "without naming them).\n"
    "- The query MUST include at least one concrete identifier copied verbatim from "
    'the context (case caption with "v.", party/organization name, court/jurisdiction, '
    "docket/case number, statute/section, property address, agreement name, or a "
    "distinctive date).\n"
    '- Do NOT mention filenames, page numbers, or "the provided context".\n'
    "- Follow query_style/query_length. If query_style asks for misspellings/poor "
    "grammar, keep identifiers intact (do not misspell party names or case numbers).\n\n"
    "### Instructions:\n"
    "1. **Generate a Query**: Based on the context, persona, term, style, and length, "
    "create a question that aligns with the persona's perspective, incorporates the "
    "term, and satisfies the standalone constraints above.\n"
    "2. **Generate an Answer**: Using only the content from the provided context, "
    "construct a detailed answer to the query. Do not add any information not included "
    "in or inferable from the context.\n"
    "3. **Additional Context** (if provided): If llm_context is provided, use it as "
    "additional guidance for query framing. Still ensure the content comes only from "
    "the provided context.\n"
)

CORPUS_MULTI_HOP_PROMPT_INSTRUCTION = (
    "Generate a multi-hop query and answer based on the specified conditions "
    "(persona, themes, style, length) and the provided context. The themes represent "
    "a set of phrases either extracted or generated from the context, which highlight "
    "the suitability of the selected context for multi-hop query creation. Ensure the "
    "query explicitly incorporates these themes.\n\n"
    "### Search Query Constraints (CRITICAL)\n"
    "The query will be asked in a large corpus setting (thousands of PDFs). Therefore:\n"
    '- The query MUST be standalone as a first-turn query.\n'
    '- Do NOT write queries that assume a document/case is already selected '
    '(avoid: "this case", "this document", "the above", "herein", "the parties" '
    "without naming them).\n"
    "- The query MUST include at least one concrete identifier copied verbatim from "
    'the contexts (case caption with "v.", party/organization name, court/jurisdiction, '
    "docket/case number, statute/section, property address, agreement name, or a "
    "distinctive date).\n"
    '- Do NOT mention filenames, page numbers, or "the provided context".\n'
    "- Follow query_style/query_length. If query_style asks for misspellings/poor "
    "grammar, keep identifiers intact (do not misspell party names or case numbers).\n\n"
    "### Instructions:\n"
    "1. **Generate a Multi-Hop Query**: Use the provided context segments and themes "
    "to form a query that requires combining information from multiple segments "
    "(e.g., `<1-hop>` and `<2-hop>`). Ensure the query explicitly incorporates one or "
    "more themes, includes concrete identifiers, and makes sense with *no prior "
    "conversation*.\n"
    "2. **Generate an Answer**: Use only the content from the provided context to "
    "create a detailed and faithful answer to the query. Avoid adding information that "
    "is not directly present or inferable from the given context.\n"
    "3. **Multi-Hop Context Tags**:\n"
    "   - Each context segment is tagged as `<1-hop>`, `<2-hop>`, etc.\n"
    "   - Ensure the query uses information from at least two segments and connects "
    "them meaningfully.\n"
    "4. **Additional Context** (if provided): If llm_context is provided, use it as "
    "additional guidance for query framing. Still ensure the content comes only from "
    "the provided context.\n"
)

# ---------------------------------------------------------------------------
# Regex for detecting referential (deictic) queries
# ---------------------------------------------------------------------------
REFERENTIAL_QUERY_RE = re.compile(
    r"("
    r"\bthis case\b|"
    r"\bthis matter\b|"
    r"\bthis document\b|"
    r"\bthis agreement\b|"
    r"\bthe above\b|"
    r"\bherein\b|"
    r"\bhereinafter\b|"
    r"\baforementioned\b|"
    r"\bin this (case|matter|document|agreement)\b|"
    r"\bthe parties\b"
    r")",
    flags=re.IGNORECASE,
)
