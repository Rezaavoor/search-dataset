"""Azure Document Intelligence client for OCR-based page extraction.

Sends PDF files to Azure Document Intelligence (prebuilt-layout model) and
returns per-page markdown text — the same approach used by Leya's file
processing pipeline.

Features:
  - Batching: PDFs > PAGES_PER_BATCH pages are split into sub-PDFs using pypdf,
    processed in parallel, then merged.
  - Retry: Exponential backoff on HTTP 429 / transient errors.
  - Per-page span mapping: The full markdown is sliced back to per-page text
    using the page-span offsets returned by the API.

Configuration (env vars):
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT  – e.g. https://westeurope.api.cognitive.microsoft.com
    AZURE_DOCUMENT_INTELLIGENCE_KEY       – API key

Requires:
    pip install azure-ai-documentintelligence
"""

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("azure_doc_intel")

# Suppress extremely verbose Azure SDK HTTP logging (every poll request)
# Only show warnings and above from these loggers
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.core").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAGES_PER_BATCH = 30          # Leya uses 30 pages per batch
MAX_PARALLEL_BATCHES = 6      # concurrent Azure requests
MAX_RETRIES = 5
BASE_RETRY_DELAY = 2.0        # seconds


# ---------------------------------------------------------------------------
# Lazy client singleton
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Create or return the cached DocumentIntelligenceClient."""
    global _client
    if _client is not None:
        return _client

    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        raise ImportError(
            "azure-ai-documentintelligence is required for OCR. "
            "Install it with: pip install azure-ai-documentintelligence"
        )

    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY", "").strip()

    if not endpoint or not key:
        raise EnvironmentError(
            "Azure Document Intelligence credentials not found. "
            "Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
            "AZURE_DOCUMENT_INTELLIGENCE_KEY in your .env file."
        )

    _client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )
    return _client


# ---------------------------------------------------------------------------
# PDF splitting (using pypdf, already a project dependency)
# ---------------------------------------------------------------------------
def _split_pdf_bytes(pdf_bytes: bytes, pages_per_batch: int) -> List[bytes]:
    """Split a PDF into sub-PDFs of at most *pages_per_batch* pages.

    Returns a list of PDF byte-strings.  If the source has fewer pages
    than the batch size, returns a single-element list with the original bytes.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    if total_pages <= pages_per_batch:
        return [pdf_bytes]

    batches: List[bytes] = []
    for start in range(0, total_pages, pages_per_batch):
        writer = PdfWriter()
        for page_idx in range(start, min(start + pages_per_batch, total_pages)):
            writer.add_page(reader.pages[page_idx])
        buf = io.BytesIO()
        writer.write(buf)
        batches.append(buf.getvalue())

    return batches


def _get_total_pdf_pages(pdf_bytes: bytes) -> int:
    """Return the total number of pages in a PDF without fully parsing it."""
    from pypdf import PdfReader
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)


# ---------------------------------------------------------------------------
# Analyze a single PDF blob via Azure Document Intelligence
# ---------------------------------------------------------------------------
def _analyze_blob(pdf_bytes: bytes) -> Dict[str, Any]:
    """Send *pdf_bytes* to Azure Doc Intel and return the analyze result.

    Uses the ``prebuilt-layout`` model with ``markdown`` output format —
    identical to Leya's approach.

    Retries on transient / 429 errors with exponential backoff.
    """
    client = _get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                body=pdf_bytes,
                output_content_format="markdown",
                content_type="application/pdf",
            )
            result = poller.result()
            return result
        except Exception as exc:
            err_str = str(exc).lower()
            is_transient = any(
                kw in err_str
                for kw in ("429", "throttl", "rate", "too many", "timeout", "503", "retry")
            )
            if is_transient and attempt < MAX_RETRIES:
                delay = BASE_RETRY_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "Azure Doc Intel transient error (attempt %d/%d), "
                    "retrying in %.0fs: %s",
                    attempt, MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
                continue
            raise


# ---------------------------------------------------------------------------
# Extract per-page markdown from the analyze result
# ---------------------------------------------------------------------------
def _extract_pages_from_result(
    result: Any,
    page_offset: int = 0,
) -> List[Tuple[int, str, Dict[str, Any]]]:
    """Map the Azure result back to per-page (page_number, markdown, metadata) tuples.

    *page_offset* is added to page indices so that batched sub-PDFs produce
    correct absolute page numbers.

    Azure Doc Intel returns ``result.pages`` with ``spans`` that index into
    ``result.content``.  We use these spans to slice the full markdown into
    per-page fragments.
    """
    content: str = result.content or ""
    pages_info = result.pages or []

    if not pages_info:
        # Fallback: single page with all content
        return [(
            page_offset + 1,
            content,
            {"page": page_offset, "extraction": "azure_doc_intel"},
        )]

    page_tuples: List[Tuple[int, str, Dict[str, Any]]] = []

    for page in pages_info:
        page_number_0 = page.page_number - 1  # Azure uses 1-indexed
        abs_page_number = page_offset + page.page_number  # 1-indexed absolute

        # Collect all spans for this page and extract corresponding text
        spans = getattr(page, "spans", None) or []
        if spans:
            page_text_parts: List[str] = []
            for span in spans:
                offset = span.offset
                length = span.length
                page_text_parts.append(content[offset:offset + length])
            page_text = "".join(page_text_parts)
        else:
            # If no spans, fallback: try to split evenly
            page_text = ""

        md: Dict[str, Any] = {
            "page": page_number_0 + page_offset,
            "extraction": "azure_doc_intel",
        }

        # Capture page dimensions if available (useful for debugging)
        if hasattr(page, "width") and page.width is not None:
            md["page_width"] = page.width
        if hasattr(page, "height") and page.height is not None:
            md["page_height"] = page.height

        page_tuples.append((abs_page_number, page_text, md))

    return page_tuples


# ---------------------------------------------------------------------------
# Public API: extract pages from a PDF file via OCR
# ---------------------------------------------------------------------------
def extract_pages_ocr(
    path: Path,
    *,
    pages_per_batch: int = PAGES_PER_BATCH,
    max_parallel: int = MAX_PARALLEL_BATCHES,
) -> List[Tuple[int, str, Dict[str, Any]]]:
    """Extract per-page markdown from a PDF using Azure Document Intelligence.

    Returns a list of ``(page_number, markdown_text, metadata)`` tuples —
    same shape as ``_load_pdf()`` in ``loaders.py``.

    For PDFs larger than *pages_per_batch*, the file is split into sub-PDFs
    and processed in parallel.
    """
    pdf_bytes = path.read_bytes()
    total_pages = _get_total_pdf_pages(pdf_bytes)

    log.info(
        "OCR: %s (%d pages, %d batches)",
        path.name, total_pages,
        max(1, (total_pages + pages_per_batch - 1) // pages_per_batch),
    )

    batches = _split_pdf_bytes(pdf_bytes, pages_per_batch)

    if len(batches) == 1:
        # Small file — single request
        result = _analyze_blob(batches[0])
        return _extract_pages_from_result(result, page_offset=0)

    # Large file — parallel batch processing
    all_pages: List[Tuple[int, str, Dict[str, Any]]] = []
    batch_results: Dict[int, Any] = {}

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {}
        for batch_idx, batch_bytes in enumerate(batches):
            fut = pool.submit(_analyze_blob, batch_bytes)
            futures[fut] = batch_idx

        for fut in as_completed(futures):
            batch_idx = futures[fut]
            try:
                batch_results[batch_idx] = fut.result()
            except Exception as exc:
                log.error(
                    "OCR batch %d/%d failed for %s: %s",
                    batch_idx + 1, len(batches), path.name, exc,
                )
                raise

    # Merge in order
    for batch_idx in range(len(batches)):
        result = batch_results[batch_idx]
        page_offset = batch_idx * pages_per_batch
        pages = _extract_pages_from_result(result, page_offset=page_offset)
        all_pages.extend(pages)

    # Sort by page number to be safe
    all_pages.sort(key=lambda t: t[0])

    log.info("OCR: %s — extracted %d pages", path.name, len(all_pages))
    return all_pages
