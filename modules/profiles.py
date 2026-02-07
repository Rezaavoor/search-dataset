"""PDF profile generation, formatting, and LLM context building."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .config import (
    DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
    DEFAULT_PDF_PROFILE_MAX_PAGES,
    DEFAULT_PDF_PROFILE_MAX_PROFILE_CHARS,
    PDF_PROFILE_LLM_CONTEXT_RULES,
    PDF_PROFILE_SCHEMA_VERSION,
)
from .db import (
    load_pdf_profiles_from_store,
    pdf_store_needs_profile,
    store_set_pdf_profile,
)
from .utils import (
    compute_corpus_path,
    compute_rel_path_for_store,
    extract_json_object,
    truncate_for_profile,
    utc_now_iso,
)


# ---------------------------------------------------------------------------
# Format a profile for prompt injection
# ---------------------------------------------------------------------------
def format_pdf_profile_for_prompt(
    profile: Dict[str, Any],
    *,
    max_chars: int = DEFAULT_PDF_PROFILE_MAX_PROFILE_CHARS,
) -> str:
    corpus_path = str(profile.get("corpus_path") or "").strip()
    filename_stem = str(profile.get("filename_stem") or "").strip()
    folder_hints = profile.get("folder_hints") or []
    folder_hints = [str(x) for x in folder_hints if str(x).strip()]

    llm_profile = profile.get("llm_profile") or {}
    if not isinstance(llm_profile, dict):
        llm_profile = {}

    def _get_list(key: str, limit: int = 8) -> List[str]:
        v = llm_profile.get(key)
        if not isinstance(v, list):
            return []
        out = []
        for x in v:
            xs = str(x).strip()
            if xs:
                out.append(xs)
            if len(out) >= limit:
                break
        return out

    title_guess = str(llm_profile.get("title_guess") or "").strip()
    doc_type = str(llm_profile.get("doc_type") or "").strip()
    summary = str(llm_profile.get("summary") or "").strip()
    topics = _get_list("topics", limit=8)
    key_entities = _get_list("key_entities", limit=8)
    likely_intents = _get_list("likely_user_intents", limit=6)

    lines: List[str] = []
    if corpus_path:
        lines.append(f"- corpus_path: {corpus_path}")
    if folder_hints:
        lines.append(f"- folder_hints: {', '.join(folder_hints[:5])}")
    if filename_stem:
        lines.append(f"- filename_stem: {filename_stem}")
    if title_guess:
        lines.append(f"- title_guess: {title_guess}")
    if doc_type:
        lines.append(f"- doc_type: {doc_type}")
    if summary:
        lines.append(f"- summary: {summary}")
    if topics:
        lines.append(f"- topics: {', '.join(topics)}")
    if key_entities:
        lines.append(f"- key_entities: {', '.join(key_entities)}")
    if likely_intents:
        lines.append(f"- likely_user_intents: {', '.join(likely_intents)}")

    return truncate_for_profile("\n".join(lines), max_chars=max_chars).strip()


# ---------------------------------------------------------------------------
# Build LLM context with profile blocks
# ---------------------------------------------------------------------------
def build_llm_context_with_pdf_profiles(
    *,
    base_llm_context: Optional[str],
    profiles: List[Dict[str, Any]],
) -> Optional[str]:
    blocks: List[str] = []
    if base_llm_context and str(base_llm_context).strip():
        blocks.append(str(base_llm_context).strip())

    blocks.append(PDF_PROFILE_LLM_CONTEXT_RULES.strip())

    rendered = []
    for p in profiles:
        rendered_block = format_pdf_profile_for_prompt(
            p, max_chars=DEFAULT_PDF_PROFILE_MAX_PROFILE_CHARS
        )
        if rendered_block:
            rendered.append(rendered_block)
    if rendered:
        blocks.append("PDF_PROFILES:\n" + "\n\n".join(rendered))

    out = "\n\n".join([b for b in blocks if b.strip()]).strip()
    return out or None


# ---------------------------------------------------------------------------
# Generate a single PDF profile via LLM (using pages already in SQLite)
# ---------------------------------------------------------------------------
def generate_pdf_profile_from_store(
    conn,
    *,
    pdf_path: Path,
    base_input_dir: Path,
    llm: Any,
    provider: str,
    llm_id: str,
    max_pages: int = DEFAULT_PDF_PROFILE_MAX_PAGES,
    max_chars_per_page: int = DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
) -> Optional[Dict[str, Any]]:
    """Generate a PDF profile from SQLite-stored pages, then persist it back."""
    resolved = pdf_path.expanduser().resolve()
    rel_path = compute_rel_path_for_store(resolved, base_input_dir)

    # Build excerpt from the first N stored pages
    rows = conn.execute(
        """
        SELECT page_number, doc_content FROM pdf_page_store
        WHERE rel_path = ? ORDER BY page_number ASC LIMIT ?
        """,
        (rel_path, int(max_pages)),
    ).fetchall()

    parts: List[str] = []
    for page_number, doc_content in rows:
        page_label = f"{int(page_number)}" if page_number is not None else "?"
        snippet = truncate_for_profile(
            str(doc_content or ""), max_chars=int(max_chars_per_page)
        )
        if snippet.strip():
            parts.append(f"--- PAGE {page_label} ---\n{snippet}")
    excerpt = "\n\n".join(parts).strip()

    prof = _generate_pdf_profile_with_llm(
        llm,
        pdf_path=resolved,
        base_input_dir=base_input_dir,
        excerpt=excerpt,
        provider=provider,
        llm_id=llm_id,
        max_pages=max_pages,
        max_chars_per_page=max_chars_per_page,
    )

    model_tag = f"{provider}:{llm_id}:p{int(max_pages)}:c{int(max_chars_per_page)}"
    store_set_pdf_profile(
        conn, rel_path=rel_path, profile=prof, pdf_profile_model=model_tag
    )
    return prof


# ---------------------------------------------------------------------------
# Build profiles for a set of PDFs (load existing + generate missing)
# ---------------------------------------------------------------------------
def build_pdf_profiles_from_store(
    conn,
    *,
    pdf_paths: List[Path],
    base_input_dir: Path,
    llm: Any,
    provider: str,
    llm_id: str,
    reprocess: bool = False,
    max_pages: int = DEFAULT_PDF_PROFILE_MAX_PAGES,
    max_chars_per_page: int = DEFAULT_PDF_PROFILE_MAX_CHARS_PER_PAGE,
) -> Dict[str, Dict[str, Any]]:
    """Build/load per-PDF profiles using SQLite as the source and cache."""
    # Load existing profiles from SQLite
    if reprocess:
        profiles: Dict[str, Dict[str, Any]] = {}
    else:
        profiles = load_pdf_profiles_from_store(
            conn, pdf_paths=pdf_paths, base_input_dir=base_input_dir
        )

    missing = [
        p for p in pdf_paths
        if str(p.expanduser().resolve()) not in profiles
    ]

    total = len(pdf_paths)
    if not missing:
        print(f"  PDF profiles: {total}/{total} loaded from store")
        return profiles

    print(f"  PDF profiles to generate: {len(missing)}/{total}")
    print(
        f"  Profile excerpt: first {max_pages} page(s), "
        f"up to {max_chars_per_page} chars per page"
    )

    for i, pdf_path in enumerate(missing, start=1):
        prof = generate_pdf_profile_from_store(
            conn,
            pdf_path=pdf_path,
            base_input_dir=base_input_dir,
            llm=llm,
            provider=provider,
            llm_id=llm_id,
            max_pages=max_pages,
            max_chars_per_page=max_chars_per_page,
        )
        if prof is not None:
            profiles[str(pdf_path.expanduser().resolve())] = prof

        if i == 1 or i == len(missing) or (i % 25 == 0):
            print(f"  Profiled {i}/{len(missing)} PDFs")

    return profiles


# ---------------------------------------------------------------------------
# Internal: LLM-based profile generation
# ---------------------------------------------------------------------------
def _generate_pdf_profile_with_llm(
    llm: Any,
    *,
    pdf_path: Path,
    base_input_dir: Path,
    excerpt: str,
    provider: str,
    llm_id: str,
    max_pages: int,
    max_chars_per_page: int,
) -> Dict[str, Any]:
    resolved = pdf_path.resolve()
    source_path = str(resolved)
    corpus_path_str = compute_corpus_path(resolved, base_input_dir)
    filename = resolved.name
    filename_stem = resolved.stem

    folder_hints: List[str] = []
    if corpus_path_str:
        try:
            p = Path(corpus_path_str)
            folder_hints = list(p.parts[:-1])
        except Exception:
            folder_hints = []
    if not folder_hints:
        try:
            parent_name = resolved.parent.name
            if parent_name:
                folder_hints = [parent_name]
        except Exception:
            pass

    base_profile: Dict[str, Any] = {
        "schema_version": PDF_PROFILE_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "provider": provider,
        "llm_id": llm_id,
        "source_path": source_path,
        "corpus_path": corpus_path_str,
        "filename": filename,
        "filename_stem": filename_stem,
        "folder_hints": folder_hints,
    }

    if llm is None:
        base_profile["llm_profile"] = {
            "title_guess": "", "doc_type": "", "summary": "",
            "topics": [], "key_entities": [], "likely_user_intents": [],
        }
        base_profile["llm_profile_error"] = "llm_unavailable"
        return base_profile

    system = (
        "You are a careful document profiler.\n"
        "Given limited excerpts from a PDF plus filename/folder hints, produce a "
        "compact high-level profile.\n"
        "The profile will be used ONLY to help generate realistic user search queries "
        "across a large corpus.\n"
        "Do NOT include page numbers, file paths, or raw filenames in any output "
        "fields except where explicitly requested.\n"
        "Return ONLY valid JSON.\n"
    )

    user = (
        "Create a PDF profile in JSON.\n\n"
        f"PDF source_path (absolute): {source_path}\n"
        f"PDF corpus_path (relative, if available): {corpus_path_str or ''}\n"
        f"PDF filename: {filename}\n"
        f"PDF filename_stem: {filename_stem}\n"
        f"PDF folder_hints: {folder_hints}\n\n"
        "EXCERPT (may be partial):\n"
        f"{excerpt}\n\n"
        "Return JSON in this exact schema:\n"
        '{"title_guess":"<string>",'
        '"doc_type":"<string>",'
        '"summary":"<1-2 sentences>",'
        '"topics":["<string>",...],'
        '"key_entities":["<string>",...],'
        '"likely_user_intents":["<string>",...],'
        '"confidence":"high|medium|low"}'
    )

    llm_profile: Dict[str, Any] = {}
    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = getattr(resp, "content", None)
        content = content if isinstance(content, str) else str(resp)
        data = extract_json_object(content)
        if isinstance(data, dict):
            llm_profile = data
        else:
            base_profile["llm_profile_error"] = "json_parse_failed"
    except Exception as e:
        base_profile["llm_profile_error"] = f"llm_invoke_failed: {e}"

    def _as_list(v: Any, limit: int = 12) -> List[str]:
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for x in v:
            xs = str(x).strip()
            if xs:
                out.append(xs)
            if len(out) >= limit:
                break
        return out

    base_profile["llm_profile"] = {
        "title_guess": str(llm_profile.get("title_guess") or "").strip(),
        "doc_type": str(llm_profile.get("doc_type") or "").strip(),
        "summary": str(llm_profile.get("summary") or "").strip(),
        "topics": _as_list(llm_profile.get("topics")),
        "key_entities": _as_list(llm_profile.get("key_entities")),
        "likely_user_intents": _as_list(
            llm_profile.get("likely_user_intents"), limit=8
        ),
        "confidence": str(llm_profile.get("confidence") or "").strip().lower(),
    }
    return base_profile
