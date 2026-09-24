"""End-to-end query pipeline: gate -> retrieve -> gate -> generate -> verify.

This is the only entry point the app (or any future API) needs. Every return
path produces an Answer, so callers never have to special-case a refusal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rag import guard


@dataclass
class Source:
    number: int
    breadcrumb: str
    citation: str
    book: str
    # Short name ("Gunderson"), which is what resolves back to the PDF on disk
    # for the page viewer. The full title does too, but only by substring.
    book_short: str
    chapter: str | None
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: str | None
    text: str
    score: float
    tier: str
    # Human-readable organ-system section, so a source card says where in the
    # books it came from without the reader decoding a pathology id.
    pathology_label: str = ""


@dataclass
class Answer:
    question: str
    text: str
    status: str  # answered | not_available | out_of_scope | error
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    # True when the answer was trimmed back to the word budget. Recorded for
    # the trace only: the trim is clean, so there is nothing for the user to
    # act on and no banner is shown.
    truncated: bool = False

    @property
    def is_answer(self) -> bool:
        return self.status == "answered"


def _maybe_repair_citations(question, supporting, raw, verdict, client, trace):
    """One bounded retry when the only fault is missing citation markers.

    A correct, fully-grounded answer with no `[n]` markers is discarded by
    layer 3, which on short definitional questions is the commonest way a good
    answer is lost -- "what is brachytherapy" retrieves at 8.3 and was thrown
    away for having no brackets. Asking once for the markers recovers those.

    This does not weaken the guarantee: the rewritten answer is verified by
    the same check, and a failure still abstains. It is skipped entirely when
    the model emitted the sentinel (it has already said it cannot answer) or
    cited a source that does not exist (a different, worse failure).
    """
    from rag.config import load_config

    attempts = load_config()["guard"]["citation"].get("repair_attempts", 0)
    if verdict.allowed or attempts < 1:
        return raw, verdict
    if "cited no source" not in verdict.reason:
        return raw, verdict

    from rag.generate import build_prompt, build_repair_prompt
    from rag.llm import LLMClient, LLMError

    client = client or LLMClient.from_config()
    system, _ = build_prompt(question, supporting)
    try:
        repaired = client.chat(
            system, build_repair_prompt(question, supporting, raw)
        ).strip()
    except LLMError as exc:
        trace["citation_repair"] = {"attempted": True, "error": str(exc)}
        return raw, verdict

    new_verdict = guard.verify_answer(repaired, len(supporting))
    trace["citation_repair"] = {
        "attempted": True,
        "recovered": new_verdict.allowed,
        "reason": new_verdict.reason,
    }
    return (repaired, new_verdict) if new_verdict.allowed else (raw, verdict)


@lru_cache(maxsize=1)
def _pathology_labels() -> dict[str, str]:
    """pathology id -> label, from the taxonomy the tagger wrote."""
    import json

    from rag.config import path_of

    path = path_of("index_dir") / "taxonomy.json"
    if not path.exists():
        return {}
    tax = json.loads(path.read_text(encoding="utf-8"))
    return {
        p["id"]: f"{sys['label']} / {p['label']}"
        for sys in tax["systems"]
        for p in sys["pathologies"]
    }


def _pathology_label(pid: str | None) -> str:
    return _pathology_labels().get(pid or "", "")


def _fit_answer(raw: str, trace: dict) -> str:
    """Bring the answer inside the configured word budget, cleanly.

    Runs before verification so the text that is checked is the text that is
    shown. Trimming happens at a sentence or bullet boundary, so the result is
    always whole sentences with their citation markers intact -- the point
    being that the user never sees a stopped-mid-clause answer, and never sees
    a banner apologising for one either.
    """
    from rag.generate import fit_to_length

    from rag.config import load_config

    budget = load_config()["llm"].get("max_answer_words", 0)
    before = len(raw.split())
    fitted, trimmed = fit_to_length(raw, budget)
    trace["length"] = {
        "words": len(fitted.split()),
        "words_generated": before,
        "budget": budget,
        "trimmed": trimmed,
    }
    return fitted


def _scoped_message(message: str, scope: str | None) -> str:
    """Say which section was searched when the answer was not found in it.

    Without this, restricting to "Prostate" and asking a tongue question
    returns a bare "not available in the provided sources", which reads as a
    statement about the whole corpus rather than about the filter the user
    themselves set two clicks earlier.
    """
    if not scope or scope == "all":
        return message
    return (
        f"{message.rstrip()} This search was limited to the selected section, "
        "so the answer may exist elsewhere in the books -- switch the section "
        "selector to All to look across both."
    )


def _to_sources(hits: list) -> list[Source]:
    out = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        out.append(
            Source(
                number=i,
                breadcrumb=c.get("breadcrumb", ""),
                citation=c.get("citation", ""),
                book=c.get("book", ""),
                book_short=c.get("book_short", ""),
                chapter=c.get("chapter"),
                pdf_page_start=c.get("pdf_page_start", 0),
                pdf_page_end=c.get("pdf_page_end", 0),
                printed_page_start=c.get("printed_page_start"),
                text=c["text"],
                score=round(hit.score, 4),
                tier=c.get("tier", ""),
                pathology_label=_pathology_label(c.get("pathology")),
            )
        )
    return out


def answer_question_stream(
    question: str, client=None, on_delta=None, scope: str | None = None
) -> Answer:
    """Same contract as `answer_question`, but streams the draft as it arrives.

    `on_delta(text_so_far, sources)` is called as the model writes. It is
    invoked only once generation actually starts, so a refusal from layer 1 or
    layer 2 never produces a single character of model output -- those paths
    are identical to the non-streaming ones.

    The returned Answer is still fully verified: layer 3 runs on the completed
    text, and if it rejects, the Answer carries the refusal, NOT the streamed
    draft. Callers must therefore render the final `Answer.text`, replacing
    whatever they showed during streaming -- that is why on_delta hands back
    text rather than the caller accumulating it privately.
    """
    started = time.time()
    trace: dict[str, Any] = {}

    def finish(text: str, status: str, sources=None, cited=None) -> Answer:
        return Answer(
            question=question,
            text=text,
            status=status,
            sources=sources or [],
            cited=cited or [],
            trace=trace,
            elapsed_ms=int((time.time() - started) * 1000),
            truncated=bool(trace.get("length", {}).get("trimmed")),
        )

    from rag.retriever import get_retriever

    retriever = get_retriever()
    query_vec = retriever.embed_query(question)

    topic = guard.topic_gate(question, query_vec)
    trace["topic_gate"] = {"allowed": topic.allowed, "reason": topic.reason, **(topic.detail or {})}
    if not topic.allowed:
        return finish(topic.message, "out_of_scope")

    hits = retriever.search(question, query_vec, scope=scope)
    trace["scope"] = scope or "all"
    trace["retrieved"] = [
        {"citation": h.citation, "score": round(h.score, 4), "tier": h.chunk.get("tier")}
        for h in hits
    ]

    grounding = guard.grounding_gate(hits)
    trace["grounding_gate"] = {
        "allowed": grounding.allowed,
        "reason": grounding.reason,
        **(grounding.detail or {}),
    }
    if not grounding.allowed:
        return finish(_scoped_message(grounding.message, scope), "not_available",
                      _to_sources(hits))

    supporting = guard.filter_supporting(hits)
    sources = _to_sources(supporting)

    from rag.generate import build_prompt
    from rag.llm import LLMClient, LLMError

    client = client or LLMClient.from_config()
    system, user = build_prompt(question, supporting)

    parts: list[str] = []
    try:
        for piece in client.chat_stream(system, user):
            parts.append(piece)
            if on_delta is not None:
                on_delta("".join(parts), sources)
    except LLMError as exc:
        trace["llm_error"] = str(exc)
        return finish(f"The language model backend is unavailable: {exc}", "error", sources)

    raw = "".join(parts).strip()
    trace["finish_reason"] = getattr(client, "last_finish_reason", None)
    raw = _fit_answer(raw, trace)

    verdict = guard.verify_answer(raw, len(supporting))
    trace["citation_check"] = {
        "allowed": verdict.allowed,
        "reason": verdict.reason,
        **(verdict.detail or {}),
    }
    raw, verdict = _maybe_repair_citations(
        question, supporting, raw, verdict, client, trace
    )
    if not verdict.allowed:
        return finish(verdict.message, "not_available", sources)

    cited = (verdict.detail or {}).get("cited", [])
    used = [s for s in sources if s.number in cited] or sources
    return finish(raw, "answered", used, cited)


def answer_question(question: str, client=None, scope: str | None = None) -> Answer:
    started = time.time()
    trace: dict[str, Any] = {}

    def finish(text: str, status: str, sources=None, cited=None) -> Answer:
        return Answer(
            question=question,
            text=text,
            status=status,
            sources=sources or [],
            cited=cited or [],
            trace=trace,
            elapsed_ms=int((time.time() - started) * 1000),
            truncated=bool(trace.get("length", {}).get("trimmed")),
        )

    from rag.retriever import get_retriever

    retriever = get_retriever()
    query_vec = retriever.embed_query(question)

    # --- Layer 1: is this even a tongue-cancer question? -------------------
    topic = guard.topic_gate(question, query_vec)
    trace["topic_gate"] = {"allowed": topic.allowed, "reason": topic.reason, **(topic.detail or {})}
    if not topic.allowed:
        return finish(topic.message, "out_of_scope")

    # --- Retrieval ---------------------------------------------------------
    hits = retriever.search(question, query_vec, scope=scope)
    trace["scope"] = scope or "all"
    trace["retrieved"] = [
        {"citation": h.citation, "score": round(h.score, 4), "tier": h.chunk.get("tier")}
        for h in hits
    ]

    # --- Layer 2: did we actually find supporting text? --------------------
    grounding = guard.grounding_gate(hits)
    trace["grounding_gate"] = {
        "allowed": grounding.allowed,
        "reason": grounding.reason,
        **(grounding.detail or {}),
    }
    if not grounding.allowed:
        # Nothing cleared the threshold, so the answer is still "not available"
        # -- but attach what was retrieved anyway. The user asked an in-scope
        # question and deserves to see the closest passages the books do have,
        # clearly marked as rejected rather than presented as an answer.
        return finish(_scoped_message(grounding.message, scope), "not_available",
                      _to_sources(hits))

    supporting = guard.filter_supporting(hits)
    sources = _to_sources(supporting)

    # --- Generation --------------------------------------------------------
    from rag.generate import generate
    from rag.llm import LLMError

    try:
        raw = generate(question, supporting, client=client)
        trace["finish_reason"] = getattr(client, "last_finish_reason", None)
    except LLMError as exc:
        trace["llm_error"] = str(exc)
        return finish(f"The language model backend is unavailable: {exc}", "error", sources)

    raw = _fit_answer(raw, trace)

    # --- Layer 3: is the answer really tied to those sources? --------------
    verdict = guard.verify_answer(raw, len(supporting))
    trace["citation_check"] = {
        "allowed": verdict.allowed,
        "reason": verdict.reason,
        **(verdict.detail or {}),
    }

    raw, verdict = _maybe_repair_citations(
        question, supporting, raw, verdict, client, trace
    )

    if not verdict.allowed:
        # The model either abstained or produced something unverifiable.
        # Either way the user gets the honest answer, not the model's text.
        return finish(verdict.message, "not_available", sources)

    cited = (verdict.detail or {}).get("cited", [])
    # Show only the sources the answer actually used.
    used = [s for s in sources if s.number in cited] or sources
    return finish(raw, "answered", used, cited)
