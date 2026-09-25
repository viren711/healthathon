"""Content Studio: turn a physician's brief into a draft article.

Three steps, mirrored by the page in app/views/content_studio.py:

  1. the brief      -- topic and a short description of the piece
  2. the questions  -- a fixed set every brief gets (type, audience, tone,
                       length, must-include, avoid), plus a few follow-ups the
                       model generates for this particular topic
  3. the draft      -- generated from all of the above, streamed as it writes

Unlike the Oncology Reference, this is not retrieval-augmented: it writes from
the model's own knowledge, so it cannot cite and can be wrong. That is why the
prompt forbids invented statistics and references, and why the page presents
the result as a DRAFT for the physician to check, never as finished copy.

This module has no Streamlit dependency, so it can be tested and scripted
without the UI.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_writer_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "writer.yaml").read_text(encoding="utf-8"))


def available_models() -> list[dict]:
    """Writer models whose weights are actually on disk, in preference order.

    Offering a model the router cannot load would fail only after the
    physician had filled in the whole brief, so missing files are filtered out
    up front.
    """
    return [m for m in load_writer_config()["models"] if (ROOT / m["file"]).exists()]


def make_client(model_id: str, max_tokens: int):
    """An LLM client aimed at the shared router, asking for the writer model."""
    from rag.config import load_config
    from rag.llm import LLMClient

    cfg = load_writer_config()
    spec = next((m for m in cfg["models"] if m["id"] == model_id), None)
    llm = load_config()["llm"]
    return LLMClient(
        base_url=llm["base_url"].rstrip("/"),
        model=model_id,
        api_key=llm.get("api_key", "not-needed"),
        temperature=cfg.get("temperature", 0.7),
        max_tokens=max_tokens,
        timeout=max(llm.get("timeout_seconds", 300), 600),
        extra_body=(spec or {}).get("extra_body") or None,
    )


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------


@dataclass
class Brief:
    topic: str
    description: str
    article_type: str
    audience: str
    tone: str
    length_words: int
    must_include: str = ""
    avoid: str = ""
    call_to_action: bool = True
    add_disclaimer: bool = True
    # Generated follow-up question -> the physician's answer (may be blank).
    followups: dict[str, str] = field(default_factory=dict)

    def problems(self) -> list[str]:
        """What must be fixed before a draft is worth generating."""
        out = []
        if len(self.topic.strip()) < 3:
            out.append("Add a topic.")
        if len(self.description.strip()) < 10:
            out.append("Add a sentence or two describing the article you want.")
        return out


# ---------------------------------------------------------------------------
# Step 2: generated follow-up questions
# ---------------------------------------------------------------------------

_QUESTIONS_SYSTEM = """You help a physician plan an article before it is written.

Given the topic and brief, ask the questions whose answers would most improve
THIS particular article: what angle to take, which specifics to cover, which
patient situations matter, what the physician wants readers to do afterwards.

Do NOT ask about length, tone, audience or article type -- those are already
settled. Each question must be one sentence, at most 20 words, and specific to
the topic rather than generic.

Reply with a JSON array of strings and nothing else."""


def _parse_questions(raw: str, want: int) -> list[str]:
    """Pull a list of questions out of whatever the model returned.

    Small models do not always return clean JSON -- a code fence, a leading
    sentence, a numbered list instead of an array -- so this tries the array
    first and falls back to lines that end in a question mark.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    found: list[str] = []
    match = re.search(r"\[.*\]", raw, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            found = [str(q).strip() for q in data if isinstance(q, (str, int, float))]
        except (json.JSONDecodeError, TypeError):
            found = []
    if not found:
        for line in raw.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d{1,2}[.)])\s*", "", line).strip().strip('"')
            if line.endswith("?"):
                found.append(line)
    clean = []
    for q in found:
        q = re.sub(r"\s+", " ", q).strip().strip('"').strip()
        if 8 <= len(q) <= 200 and q not in clean:
            clean.append(q if q.endswith("?") else q + "?")
    return clean[:want]


def generate_followups(
    topic: str, description: str, article_type: str, model_id: str
) -> tuple[list[str], bool]:
    """Topic-specific questions to ask before writing.

    Returns (questions, generated). `generated` is False when the fixed
    fallback set was used -- the model unreachable, or its reply unparseable --
    so the page can say so instead of passing fallbacks off as tailored.
    """
    from rag.llm import LLMError

    cfg = load_writer_config()["followups"]
    want = int(cfg.get("count", 3))
    client = make_client(model_id, max_tokens=260)
    client.temperature = 0.3
    user = (
        f"Topic: {topic.strip()}\n"
        f"Article type: {article_type}\n"
        f"Physician's brief: {description.strip()}\n\n"
        f"Give exactly {want} questions."
    )
    try:
        questions = _parse_questions(client.chat(_QUESTIONS_SYSTEM, user), want)
    except LLMError:
        questions = []
    if len(questions) >= max(1, want - 1):
        return questions, True
    return list(cfg["fallback"])[:want], False


# ---------------------------------------------------------------------------
# Step 3: the article
# ---------------------------------------------------------------------------

ARTICLE_SYSTEM = """You are a medical writer helping a physician draft content that they will review, edit and publish under their own name.

Accuracy rules -- these matter more than style:
- Write only well-established, mainstream medical information. If you are not sure a fact is correct, leave it out rather than guess.
- Do not invent statistics, percentages, study results, dates, drug doses or references. Do not write "recent research shows", "studies suggest" or similar unless the physician supplied the source in the brief.
- Do not give individual medical advice. Encourage readers to speak with their own doctor where appropriate.

Format rules:
- Markdown. Exactly one H1 title on the first line, then H2 section headings.
- Use bullet lists where they genuinely help; otherwise write in short paragraphs.
- No emoji.
- Do not add a disclaimer or a sign-off; those are handled separately.

Brief rules:
- Match the requested article type, audience and tone exactly.
- Cover everything listed under "must include", and nothing listed under "avoid".
- Stay within 10% of the requested word count, and finish your final sentence."""


def build_article_prompt(brief: Brief) -> tuple[str, str]:
    cfg = load_writer_config()
    style = cfg["article_types"].get(brief.article_type, "")
    lines = [
        "Write the article described below.",
        "",
        f"Topic: {brief.topic.strip()}",
        f"Physician's brief: {brief.description.strip()}",
        f"Article type: {brief.article_type}",
        f"How this type should read: {style}" if style else "",
        f"Audience: {brief.audience}",
        f"Tone: {brief.tone}",
        f"Length: about {brief.length_words} words",
    ]
    if brief.must_include.strip():
        lines.append(f"Must include: {brief.must_include.strip()}")
    if brief.avoid.strip():
        lines.append(f"Avoid: {brief.avoid.strip()}")
    answered = {q: a.strip() for q, a in brief.followups.items() if a and a.strip()}
    if answered:
        lines += ["", "The physician's answers to planning questions:"]
        lines += [f"- Q: {q}\n  A: {a}" for q, a in answered.items()]
    lines += [
        "",
        "End with a short, practical call to action for the reader."
        if brief.call_to_action
        else "Do not end with a call to action.",
    ]
    return ARTICLE_SYSTEM, "\n".join(line for line in lines if line is not None)


@dataclass
class Article:
    text: str
    words: int
    target_words: int
    model: str
    elapsed_s: float
    finish_reason: str | None
    # True when the draft hit the token cap and was trimmed back to its last
    # complete paragraph, rather than left stopping mid-sentence.
    trimmed: bool = False


_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF☀-➿️‍]"
)
_THINK = re.compile(r"<think>.*?(?:</think>|$)", flags=re.S)


def visible_text(raw: str) -> str:
    """What the reader should see of a (possibly partial) generation.

    Drops any reasoning block -- including one still open mid-stream, which
    is what the `$` alternative in _THINK is for.
    """
    return _THINK.sub("", raw or "").lstrip()


def tidy_article(raw: str, finish_reason: str | None) -> tuple[str, bool]:
    """Normalise a finished draft. Returns (text, trimmed).

    - strips any reasoning block and emoji (the prompt forbids them, and a
      physician's article should not carry them if the model slips);
    - promotes a bold first line to an H1 -- Qwen3-4B in particular likes
      "**Title**" over "# Title";
    - if generation stopped on the token cap, cuts back to the last complete
      paragraph (or, failing that, sentence) so the draft never ends
      mid-clause.
    """
    text = visible_text(raw)
    had_emoji = bool(_EMOJI.search(text))
    text = _EMOJI.sub("", text)
    if had_emoji:
        # Removing "doctor 👉 today" leaves a double space; removing a trailing
        # emoji leaves a space before the full stop. Tidy both, and only when
        # an emoji was actually removed, so deliberate spacing is untouched.
        text = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)
        text = re.sub(r" +([.,!?;:])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text).strip()

    lines = text.split("\n")
    if lines:
        first = lines[0].strip()
        bold = re.fullmatch(r"\*\*(.+?)\*\*", first)
        if bold and not first.startswith("#"):
            lines[0] = f"# {bold.group(1).strip()}"
        text = "\n".join(lines)

    trimmed = False
    if finish_reason == "length":
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        # A last paragraph with no terminal punctuation is the cut-off one.
        while paragraphs and not re.search(r"[.!?:)\]\*]\s*$", paragraphs[-1].strip()):
            paragraphs.pop()
            trimmed = True
        if paragraphs:
            text = "\n\n".join(paragraphs)
        else:
            # One long unfinished paragraph: fall back to the last sentence end.
            cut = max(text.rfind(". "), text.rfind(".\n"))
            if cut > 0:
                text = text[: cut + 1]
                trimmed = True
    return text.strip(), trimmed


def token_budget(target_words: int) -> int:
    cfg = load_writer_config()
    return int(target_words * cfg.get("tokens_per_word", 1.6)) + int(cfg.get("token_headroom", 200))


def write_article(brief: Brief, model_id: str, on_delta=None) -> Article:
    """Generate the draft, streaming it through `on_delta(text_so_far)`."""
    cfg = load_writer_config()
    client = make_client(model_id, max_tokens=token_budget(brief.length_words))
    system, user = build_article_prompt(brief)

    started = time.time()
    parts: list[str] = []
    for piece in client.chat_stream(system, user):
        parts.append(piece)
        if on_delta is not None:
            on_delta(visible_text("".join(parts)))

    text, trimmed = tidy_article("".join(parts), client.last_finish_reason)
    if brief.add_disclaimer and cfg.get("disclaimer"):
        text = f"{text}\n\n---\n\n*{cfg['disclaimer'].strip()}*"

    body_words = len(visible_text(text).split())
    return Article(
        text=text,
        words=body_words,
        target_words=brief.length_words,
        model=model_id,
        elapsed_s=time.time() - started,
        finish_reason=client.last_finish_reason,
        trimmed=trimmed,
    )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def available_memory_gb() -> float:
    """MemAvailable from /proc/meminfo, in GB; a large number if unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 1e9


def release_reference_models() -> bool:
    """Drop the Oncology Reference retrieval stack from this process.

    Returns True if anything was released. The LLM itself needs no help --
    the router swaps it out on its own -- this is only the embedder, reranker
    and indexes that live in the portal process.
    """
    import gc

    try:
        from rag import guard
        from rag.retriever import get_retriever
    except ImportError:
        return False
    had = get_retriever.cache_info().currsize > 0
    get_retriever.cache_clear()
    guard._seed_vectors_for.cache_clear()
    gc.collect()
    return had
