"""Render pages of the source PDFs, so a citation can be opened and read.

A citation that cannot be checked is only half a citation. This module turns
"Gunderson, ch.38, PDF p.872" into the actual page image, rendered on demand
straight from the source file.

Rendering rather than linking is deliberate. The books are 52 MB and 296 MB;
serving either one to a browser so it can jump to `#page=872` means shipping
the whole file, and a `file://` link is blocked from a web page anyway.
PyMuPDF renders a single page in a few milliseconds without reading the rest
of the document, so the expensive-looking option is in fact the cheap one.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import fitz  # PyMuPDF
import yaml

from rag.config import ROOT


@lru_cache(maxsize=1)
def _book_paths() -> dict[str, Path]:
    """Map both the book id and its short name to the PDF on disk."""
    cfg = yaml.safe_load((ROOT / "config" / "books.yaml").read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    for book in cfg["books"]:
        path = ROOT / book["path"]
        paths[book["id"].lower()] = path
        paths[book["short"].lower()] = path
    return paths


def resolve_book(name: str) -> Path | None:
    """Find the source PDF for a book id, short name or full title."""
    if not name:
        return None
    paths = _book_paths()
    key = name.strip().lower()
    if key in paths:
        return paths[key]
    # Full titles ("DeVita, Hellman & Rosenberg's Cancer: ...") -- match on the
    # short name appearing anywhere in the string.
    for short, path in paths.items():
        if short in key:
            return path
    return None


def page_count(book: str) -> int:
    path = resolve_book(book)
    if path is None or not path.exists():
        return 0
    doc = fitz.open(path)
    try:
        return doc.page_count
    finally:
        doc.close()


@lru_cache(maxsize=64)
def render_page(book: str, pdf_page: int, dpi: int = 130) -> bytes | None:
    """Render one 1-based PDF page to PNG bytes.

    Cached because the viewer re-renders on every Streamlit rerun -- paging
    back and forth through a section would otherwise re-rasterise each time.
    """
    path = resolve_book(book)
    if path is None or not path.exists():
        return None
    doc = fitz.open(path)
    try:
        if not 1 <= pdf_page <= doc.page_count:
            return None
        pix = doc[pdf_page - 1].get_pixmap(dpi=dpi)
        buf = io.BytesIO(pix.tobytes("png"))
        return buf.getvalue()
    finally:
        doc.close()


def extract_pages(book: str, start: int, end: int, max_pages: int = 12) -> bytes | None:
    """Pull a page range out into a small standalone PDF, for downloading.

    Capped at `max_pages` so a section with a runaway page span cannot be used
    to rebuild most of a 296 MB book through the browser.
    """
    path = resolve_book(book)
    if path is None or not path.exists():
        return None
    src = fitz.open(path)
    try:
        start = max(1, start)
        end = min(src.page_count, max(start, end), start + max_pages - 1)
        out = fitz.open()
        out.insert_pdf(src, from_page=start - 1, to_page=end - 1)
        data = out.tobytes()
        out.close()
        return data
    finally:
        src.close()
