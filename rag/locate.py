"""Topic locator: where in the books is this subject discussed?

This answers a different question from `rag.pipeline`. The pipeline is asked
"what is the answer?" and spends an LLM call to write one. The locator is
asked "where is this?" and must not paraphrase anything -- it returns
positions in the source books, so the user can go and read the passage
themselves.

Consequences of that difference:

  * No LLM, so it returns in well under a second and can never hallucinate.
  * Results are grouped by SECTION, not by chunk. Chunks overlap by design
    (80 tokens), so an ungrouped list shows the same passage three times with
    three slightly different page spans, which is useless as a reference.
  * It casts a wider net than question-answering, and applies a much lower
    relevance floor -- but not *no* floor. Nearest-neighbour search always
    returns something, so without a floor "what is the capital of France?"
    comes back with five confident-looking page references to an oncology
    textbook. Measured on this corpus the two populations separate cleanly:
    real topics score +2.1 to +8.3 on the cross-encoder, questions the books
    have nothing to say about score -9.7 to -2.6. The floor sits between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.config import load_config


@dataclass
class Location:
    """One place in one book where a topic is discussed."""

    book: str
    book_short: str
    chapter: str | None
    section_path: list[str]
    breadcrumb: str
    citation: str
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: str | None
    tier: str
    score: float
    n_chunks: int = 1
    snippet: str = ""
    section_id: str = ""
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def page_label(self) -> str:
        """Human-facing page reference, preferring the book's printed number."""
        if self.printed_page_start:
            base = f"p. {self.printed_page_start}"
        else:
            base = f"PDF p. {self.pdf_page_start}"
        if self.pdf_page_end > self.pdf_page_start:
            base += f" (PDF {self.pdf_page_start}-{self.pdf_page_end})"
        elif self.printed_page_start:
            base += f" (PDF {self.pdf_page_start})"
        return base


def _snippet(text: str, topic: str, width: int = 260) -> str:
    """A window of `text` around the first mention of the topic.

    Falls back to the opening of the passage when no term matches, which is
    the common case for a purely semantic hit.
    """
    flat = " ".join(text.split())
    terms = [t for t in topic.lower().split() if len(t) > 3]
    low = flat.lower()
    pos = min((low.find(t) for t in terms if low.find(t) >= 0), default=-1)
    if pos < 0:
        return flat[:width] + ("..." if len(flat) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    return ("..." if start else "") + flat[start:end] + ("..." if end < len(flat) else "")


def locate(
    topic: str,
    limit: int = 15,
    candidates: int = 40,
    min_score: float | None = None,
    scope: str | None = None,
) -> list[Location]:
    """Find where `topic` is discussed, best first, one entry per section.

    `candidates` is how many chunks are retrieved before grouping. It is much
    larger than the question-answering path uses, because several chunks
    usually collapse into one section and a small candidate pool would return
    only a handful of distinct places.

    `scope` restricts the search to one pathology or organ system, so "where
    is brachytherapy discussed" can be asked of the prostate chapters alone.

    `min_score` is the relevance floor; it defaults to
    `retrieval.min_locate_score` in config/rag.yaml. Anything below it is
    dropped, and a topic with nothing above it returns an empty list rather
    than the nearest five irrelevant passages.
    """
    from rag.retriever import get_retriever

    topic = (topic or "").strip()
    if len(topic) < 2:
        return []

    if min_score is None:
        min_score = load_config()["retrieval"].get("min_locate_score", 0.0)

    retriever = get_retriever()
    hits = retriever.search_wide(topic, candidates=candidates, scope=scope)
    hits = [h for h in hits if h.score >= min_score]
    if not hits:
        return []

    # Group by section: overlapping chunks of one section are one place in
    # the book, and should be reported once, at its best score.
    grouped: dict[str, Location] = {}
    for hit in hits:
        chunk = hit.chunk
        key = chunk.get("section_id") or chunk.get("chunk_id")
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = Location(
                book=chunk.get("book", ""),
                book_short=chunk.get("book_short", ""),
                chapter=chunk.get("chapter"),
                section_path=list(chunk.get("section_path") or []),
                breadcrumb=chunk.get("breadcrumb", ""),
                citation=chunk.get("citation", ""),
                pdf_page_start=chunk.get("pdf_page_start", 0),
                pdf_page_end=chunk.get("pdf_page_end", chunk.get("pdf_page_start", 0)),
                printed_page_start=chunk.get("printed_page_start"),
                tier=chunk.get("tier", ""),
                score=round(hit.score, 4),
                snippet=_snippet(chunk.get("text", ""), topic),
                section_id=key,
                chunk_ids=[chunk.get("chunk_id", "")],
            )
        else:
            existing.n_chunks += 1
            existing.chunk_ids.append(chunk.get("chunk_id", ""))
            # Widen the span to cover every chunk of the section that matched,
            # so the page range shown is the range worth reading.
            existing.pdf_page_start = min(
                existing.pdf_page_start, chunk.get("pdf_page_start", existing.pdf_page_start)
            )
            existing.pdf_page_end = max(
                existing.pdf_page_end, chunk.get("pdf_page_end", existing.pdf_page_end)
            )

    out = sorted(grouped.values(), key=lambda loc: -loc.score)
    return out[:limit]


def corpus_summary() -> dict:
    """Counts for the UI: what is actually indexed, per book."""
    from rag.retriever import get_retriever

    meta = get_retriever().meta
    return {
        "chunks": meta.get("chunk_count"),
        "books": meta.get("books", []),
        "mode": load_config().get("selection", {}).get("mode", "tongue"),
        "embedding_model": meta.get("embedding_model"),
        "built_at": meta.get("built_at"),
        "partial": meta.get("partial", False),
    }
