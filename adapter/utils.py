"""Shared utilities for the adapter training pipeline.

Functions used by both preprocess.py and train.py live here to avoid duplication.
"""

from typing import Optional, Tuple


def parse_page_ref(ref: str) -> Optional[Tuple[str, int]]:
    """Parse 'filename (page N)' -> (filename, page_number), or None on failure."""
    ref = ref.strip()
    if " (page " in ref and ref.endswith(")"):
        parts = ref.rsplit(" (page ", 1)
        try:
            return (parts[0].strip(), int(parts[1].rstrip(")")))
        except ValueError:
            pass
    return None


def fmt_page_ref(fname: str, page: int) -> str:
    """Format (filename, page_number) -> 'filename (page N)'."""
    return f"{fname} (page {page})"
