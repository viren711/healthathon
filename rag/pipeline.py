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
    # answered | excerpts | not_available | out_of_scope | error | safety
    # `excerpts` (Patient Portal only): the evidence is strong but the model
    # could not produce a checked answer, so the guides' own passages are shown.
    status: str
    sources: list[Source] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    # True when the answer was trimmed back to the word budget. Recorded for
    # the trace only: the trim is clean, so there is nothing for the user to
    # act on and no banner is shown.
    truncated: bool = False
    # Patient Portal only: safety notices to show ABOVE the answer, each
    # {"kind": crisis|emergency|personal_decision, "message", "guide"}.
    alerts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_answer(self) -> bool:
        return self.status == "answered"


def _maybe_repair_citations(
    question, supporting, raw, verdict, client, trace, corpus: str = "books"
):
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

    audience = "patient" if corpus == "patient" else "clinician"
    client = client or LLMClient.from_config()
    system, _ = build_prompt(question, supporting, audience)
    try:
        repaired = client.chat(
            system, build_repair_prompt(question, supporting, raw, audience)
        ).strip()
    except LLMError as exc:
        trace["citation_repair"] = {"attempted": True, "error": str(exc)}
        return raw, verdict

    new_verdict = guard.verify_answer(repaired, len(supporting), corpus)
    trace["citation_repair"] = {
        "attempted": True,
        "recovered": new_verdict.allowed,
        "reason": new_verdict.reason,
    }
    return (repaired, new_verdict) if new_verdict.allowed else (raw, verdict)


def _search_query(question: str, corpus: str) -> str:
    """The text to search with. Patient Portal: everyday words get the guides'
    own terms appended (config/patient.yaml `query_expansions`), so "tired"
    also looks for "fatigue". Only retrieval sees this; the model and the
    user see the question as asked."""
    if corpus != "patient":
        return question
    import re

    from rag.config import load_patient_config

    extra = [
        e["add"]
        for e in load_patient_config().get("query_expansions", [])
        if re.search(e["match"], question, flags=re.I)
    ]
    return f"{question} ({'; '.join(extra)})" if extra else question


def _excerpt_fallback(corpus, supporting, trace, why):
    """Patient Portal only: the guides' own passages when no answer can be written.

    A 3B model sometimes abstains, drops its citations or times out although
    retrieval found the page that answers the question. A patient is better
    served by that page, verbatim and cited, than by "not available". Only
    when the best passage clears a bar well above the grounding gate, so a
    question the guides barely touch still ends in "not available".
    """
    if corpus != "patient" or not supporting:
        return None
    from rag.config import load_patient_config

    cfg = load_patient_config()
    gen = cfg["generation"]
    floor = gen.get("excerpt_min_score")
    if floor is None or max(h.score for h in supporting) < floor:
        return None
    trace["excerpt_fallback"] = why
    return cfg["messages"]["excerpts"], _to_sources(supporting[: gen.get("excerpt_count", 3)])


def _maybe_retry_sentinel(question, supporting, raw, verdict, client, trace, corpus):
    """Patient Portal only: one retry when the model abstains on strong evidence.

    A 3B model reads "Will I...?" as a question about the reader that the
    guides cannot answer, and emits the not-available token with the answer
    sitting in source [1]. When retrieval's best passage is far above the
    grounding bar, ask once more, naming that passage. The result is held to
    exactly the same citation check; if it still abstains, so does the answer.
    """
    if corpus != "patient" or verdict.allowed or "sentinel" not in verdict.reason:
        return raw, verdict
    from rag.config import load_patient_config

    floor = load_patient_config()["generation"].get("sentinel_retry_min_score", 0)
    top = max((h.score for h in supporting), default=float("-inf"))
    if not floor or top < floor:
        return raw, verdict

    from rag.generate import build_prompt
    from rag.llm import LLMClient, LLMError

    client = client or LLMClient.from_config()
    system, user = build_prompt(question, supporting, "patient")
    nudge = (
        "\n\nNote: these SOURCES do cover this topic -- source [1] is about it "
        "directly. Answer the question from them in plain words, describing what "
        "the guides say for people in general, with a citation on every sentence."
    )
    try:
        retried = client.chat(system, user + nudge).strip()
    except LLMError as exc:
        trace["sentinel_retry"] = {"attempted": True, "error": str(exc)}
        return raw, verdict
    retried = _fit_answer(retried, trace, corpus)
    new_verdict = guard.verify_answer(retried, len(supporting), corpus)
    trace["sentinel_retry"] = {
        "attempted": True,
        "top_score": round(top, 3),
        "recovered": new_verdict.allowed,
        "reason": new_verdict.reason,
    }
    return (retried, new_verdict) if new_verdict.allowed else (raw, verdict)


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


@lru_cache(maxsize=1)
def _guide_labels() -> dict[str, str]:
    """guide id -> "Supportive care / Distress During Cancer Care"."""
    from rag.config import load_patient_config

    pcfg = load_patient_config()
    return {
        g["id"]: f"{pcfg['categories'][g['category']]} / {g['title']}" for g in pcfg["guides"]
    }


def _source_label(chunk: dict) -> str:
    if chunk.get("tier") == "patient":
        return _guide_labels().get(chunk.get("guide_id", ""), "")
    return _pathology_label(chunk.get("pathology"))


def _fit_answer(raw: str, trace: dict, corpus: str = "books") -> str:
    """Bring the answer inside the configured word budget, cleanly.

    Runs before verification so the text that is checked is the text that is
    shown. Trimming happens at a sentence or bullet boundary, so the result is
    always whole sentences with their citation markers intact -- the point
    being that the user never sees a stopped-mid-clause answer, and never sees
    a banner apologising for one either.
    """
    from rag.generate import fit_to_length

    from rag.config import load_config

    if corpus == "patient":
        from rag.config import load_patient_config

        budget = load_patient_config()["generation"]["max_answer_words"]
    else:
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


def _scoped_message(message: str, scope: str | None, corpus: str = "books") -> str:
    """Say which section was searched when the answer was not found in it.

    Without this, restricting to "Prostate" and asking a tongue question
    returns a bare "not available in the provided sources", which reads as a
    statement about the whole corpus rather than about the filter the user
    themselves set two clicks earlier.
    """
    if not scope or scope == "all":
        return message
    if corpus == "patient":
        return (
            f"{message.rstrip()} This search was limited to the guide or topic you "
            "selected -- choose All guides to look across every NCCN patient guide."
        )
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
                pathology_label=_source_label(c),
            )
        )
    return out


def _safety_alerts(question: str, corpus: str, trace: dict):
    """Run the patient safety screen. Returns (alerts, blocking_message)."""
    if corpus != "patient":
        return [], None
    from rag.patient_safety import screen

    result = screen(question)
    trace["safety"] = {"kind": result.kind}
    if result.kind is None:
        return [], None
    alert = {"kind": result.kind, "message": result.message, "guide": result.guide}
    return [alert], (result.message if result.blocks_answer else None)


def answer_question_stream(
    question: str,
    client=None,
    on_delta=None,
    scope: str | None = None,
    corpus: str = "books",
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
    alerts, blocking = _safety_alerts(question, corpus, trace)

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
            alerts=alerts,
        )

    if blocking:
        return finish(blocking, "safety")

    from rag.retriever import get_retriever

    retriever = get_retriever(corpus)
    search_q = _search_query(question, corpus)
    query_vec = retriever.embed_query(search_q)

    topic = guard.topic_gate(question, query_vec, corpus)
    trace["topic_gate"] = {"allowed": topic.allowed, "reason": topic.reason, **(topic.detail or {})}
    if not topic.allowed:
        return finish(topic.message, "out_of_scope")

    hits = retriever.search(search_q, query_vec, scope=scope)
    trace["scope"] = scope or "all"
    trace["retrieved"] = [
        {"citation": h.citation, "score": round(h.score, 4), "tier": h.chunk.get("tier")}
        for h in hits
    ]

    grounding = guard.grounding_gate(hits, corpus)
    trace["grounding_gate"] = {
        "allowed": grounding.allowed,
        "reason": grounding.reason,
        **(grounding.detail or {}),
    }
    if not grounding.allowed:
        return finish(_scoped_message(grounding.message, scope, corpus), "not_available",
                      _to_sources(hits))

    supporting = guard.filter_supporting(hits, corpus)
    sources = _to_sources(supporting)

    from rag.generate import build_prompt
    from rag.llm import LLMClient, LLMError

    client = client or LLMClient.from_config()
    system, user = build_prompt(
        question, supporting, "patient" if corpus == "patient" else "clinician"
    )

    parts: list[str] = []
    try:
        for piece in client.chat_stream(system, user):
            parts.append(piece)
            if on_delta is not None:
                on_delta("".join(parts), sources)
    except LLMError as exc:
        trace["llm_error"] = str(exc)
        fb = _excerpt_fallback(corpus, supporting, trace, "llm_error")
        if fb:
            return finish(fb[0], "excerpts", fb[1])
        return finish(f"The language model backend is unavailable: {exc}", "error", sources)

    raw = "".join(parts).strip()
    trace["finish_reason"] = getattr(client, "last_finish_reason", None)
    raw = _fit_answer(raw, trace, corpus)

    verdict = guard.verify_answer(raw, len(supporting), corpus)
    trace["citation_check"] = {
        "allowed": verdict.allowed,
        "reason": verdict.reason,
        **(verdict.detail or {}),
    }
    raw, verdict = _maybe_repair_citations(
        question, supporting, raw, verdict, client, trace, corpus
    )
    raw, verdict = _maybe_retry_sentinel(
        question, supporting, raw, verdict, client, trace, corpus
    )
    if not verdict.allowed:
        fb = _excerpt_fallback(corpus, supporting, trace, verdict.reason)
        if fb:
            return finish(fb[0], "excerpts", fb[1])
        return finish(verdict.message, "not_available", sources)

    cited = (verdict.detail or {}).get("cited", [])
    used = [s for s in sources if s.number in cited] or sources
    return finish(raw, "answered", used, cited)


def answer_question(
    question: str, client=None, scope: str | None = None, corpus: str = "books"
) -> Answer:
    started = time.time()
    trace: dict[str, Any] = {}
    alerts, blocking = _safety_alerts(question, corpus, trace)

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
            alerts=alerts,
        )

    if blocking:
        return finish(blocking, "safety")

    from rag.retriever import get_retriever

    retriever = get_retriever(corpus)
    search_q = _search_query(question, corpus)
    query_vec = retriever.embed_query(search_q)

    # --- Layer 1: is this even an in-scope question? -----------------------
    topic = guard.topic_gate(question, query_vec, corpus)
    trace["topic_gate"] = {"allowed": topic.allowed, "reason": topic.reason, **(topic.detail or {})}
    if not topic.allowed:
        return finish(topic.message, "out_of_scope")

    # --- Retrieval ---------------------------------------------------------
    hits = retriever.search(search_q, query_vec, scope=scope)
    trace["scope"] = scope or "all"
    trace["retrieved"] = [
        {"citation": h.citation, "score": round(h.score, 4), "tier": h.chunk.get("tier")}
        for h in hits
    ]

    # --- Layer 2: did we actually find supporting text? --------------------
    grounding = guard.grounding_gate(hits, corpus)
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
        return finish(_scoped_message(grounding.message, scope, corpus), "not_available",
                      _to_sources(hits))

    supporting = guard.filter_supporting(hits, corpus)
    sources = _to_sources(supporting)

    # --- Generation --------------------------------------------------------
    from rag.generate import generate
    from rag.llm import LLMError

    try:
        raw = generate(
            question, supporting, client=client,
            audience="patient" if corpus == "patient" else "clinician",
        )
        trace["finish_reason"] = getattr(client, "last_finish_reason", None)
    except LLMError as exc:
        trace["llm_error"] = str(exc)
        fb = _excerpt_fallback(corpus, supporting, trace, "llm_error")
        if fb:
            return finish(fb[0], "excerpts", fb[1])
        return finish(f"The language model backend is unavailable: {exc}", "error", sources)

    raw = _fit_answer(raw, trace, corpus)

    # --- Layer 3: is the answer really tied to those sources? --------------
    verdict = guard.verify_answer(raw, len(supporting), corpus)
    trace["citation_check"] = {
        "allowed": verdict.allowed,
        "reason": verdict.reason,
        **(verdict.detail or {}),
    }

    raw, verdict = _maybe_repair_citations(
        question, supporting, raw, verdict, client, trace, corpus
    )
    raw, verdict = _maybe_retry_sentinel(
        question, supporting, raw, verdict, client, trace, corpus
    )

    if not verdict.allowed:
        # The model either abstained or produced something unverifiable.
        # Either way the user gets the honest answer, not the model's text --
        # for a patient, the guides' own passages when the evidence is strong.
        fb = _excerpt_fallback(corpus, supporting, trace, verdict.reason)
        if fb:
            return finish(fb[0], "excerpts", fb[1])
        return finish(verdict.message, "not_available", sources)

    cited = (verdict.detail or {}).get("cited", [])
    # Show only the sources the answer actually used.
    used = [s for s in sources if s.number in cited] or sources
    return finish(raw, "answered", used, cited)
