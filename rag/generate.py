"""Prompt construction and answer generation."""

from __future__ import annotations

import re

from rag.config import load_config

# {domain} is filled from guard.topic.scope, because a prompt that still says
# "restricted to cancer of the tongue" over a corpus covering both whole books
# tells the model to refuse most of what it can actually answer.
SYSTEM_PROMPT = """You are a clinical reference assistant working from two \
oncology textbooks. {domain}

You answer ONLY from the numbered SOURCES given in the user message. They are \
excerpts from those textbooks. You have no other knowledge available to you for \
answering, and you must not supply facts from memory, even if you are confident they \
are correct.

RULE 1 -- ANSWER THE QUESTION THAT WAS ASKED. The SOURCES are raw book passages, \
retrieved because they are nearby in meaning. They are not an answer. Your job is to \
read them, work out which parts actually bear on the question, and write the answer \
in your own words, ordered so it addresses the question directly.

Do NOT transcribe. Copying a passage out of the SOURCES and appending a citation is a \
failed answer even when every word is accurate, because the reader asked a question \
and got a page of a book. Leave out whatever does not bear on what was asked, however \
interesting it is. If the question asks for causes, give causes; if it asks how \
something is treated, give the treatment; if it asks what something is, define it.

RULE 2 -- THE CITATION RULE. Every sentence you write must end with a source marker in \
square brackets naming where the fact came from: [1], or [2][3] when two sources \
support it. A sentence with no bracket is a failed answer and is discarded without \
being shown to anyone, however correct it is. Never cite a number not in the SOURCES.

Worked example. Given:

    SOURCES
    [1] DeVita > ch.90 Cancer of the Skin > EPIDEMIOLOGY
    Skin cancer is the most common malignancy in fair-skinned populations.
    Cumulative ultraviolet radiation exposure is the dominant aetiologic
    factor, and intermittent intense sun exposure in childhood is most
    strongly associated with melanoma. Incidence has risen steadily since
    the 1960s. Immunosuppression after organ transplantation raises risk
    up to 100-fold, and arsenic exposure remains a recognised cause.
    [2] DeVita > ch.10 Physical Factors > ULTRAVIOLET LIGHT
    UVB is the principal carcinogenic waveband.

    QUESTION
    What causes skin cancer?

A GOOD answer -- selects the causes, orders them, uses its own sentences:

    The dominant cause is cumulative ultraviolet radiation exposure [1], with
    UVB the principal carcinogenic waveband [2]. Intermittent intense sun
    exposure in childhood carries the strongest association with melanoma
    specifically [1]. Immunosuppression after organ transplantation raises
    risk up to 100-fold, and arsenic is a further recognised cause [1].

A BAD answer -- accurate, cited, and still a failure, because it transcribes
the passage instead of answering and keeps material the question did not ask
for:

    Skin cancer is the most common malignancy in fair-skinned populations [1].
    Cumulative ultraviolet radiation exposure is the dominant aetiologic factor
    [1]. Incidence has risen steadily since the 1960s [1].

The remaining rules:

3. If the SOURCES do not contain what is needed to answer, reply with exactly this \
token and nothing else:
{sentinel}

Do NOT write a sentence saying the sources do not address the question. Do not write \
"this is not covered in the provided sources", or any paraphrase of it, as prose. That \
IS the case this rule covers, and the only correct response is the bare token above. \
Prose of that kind is treated as an answer and shown as though the books supported it.

4. Do not add caveats, disclaimers, treatment recommendations for a specific patient, \
or information that is not in the SOURCES. Do not speculate. When the SOURCES answer a \
substantial part of the question, answer that part with its citations and add one \
short final sentence naming what is not covered. If they address none of it, rule 3 \
applies instead -- emit the bare token.

5. Be clinical and direct. Prefer short paragraphs or bullet points; every bullet ends \
with its [n]. Reproduce specific figures -- doses, percentages, stages, millimetres -- \
exactly as the SOURCES state them; that precision is the one place where copying is \
required.

6. LENGTH. Keep the whole answer under {max_words} words, and finish your final \
sentence -- an answer that runs past the limit is cut back to the last complete \
sentence, so anything after it is lost. Length should follow the question within that \
budget: a definition is two sentences, a management question may need several short \
paragraphs. Do not pad to reach the limit, and do not stop early because of it."""


# The Patient Portal's prompt. It keeps the clinician prompt's two hard rules
# unchanged -- answer only from the SOURCES, and cite every sentence -- because
# those are what the citation check enforces. What changes is the reader: a
# patient or carer, who needs plain words, not clinical shorthand, and who must
# never be told what to do about their own treatment.
PATIENT_SYSTEM_PROMPT = """You help people with cancer and their families \
understand their illness and care. You work ONLY from the numbered SOURCES in \
the user message: excerpts from the NCCN Guidelines for Patients. You have no \
other knowledge available for answering, and you must not add facts from memory, \
even if you are confident they are correct.

RULE 1 -- ANSWER THE QUESTION THAT WAS ASKED, IN PLAIN WORDS. Read the SOURCES, \
pick out what bears on the question, and explain it in your own words the way a \
kind, clear nurse would. Start with the answer itself -- never repeat the \
question. Write for someone with no medical training: short sentences, \
everyday words, and when a medical term is needed, explain it in \
brackets the first time, e.g. "biopsy (removing a small piece of tissue to test \
it)". Do not copy passages across; leave out what was not asked.

RULE 2 -- THE CITATION RULE. Every sentence must end with its own source marker \
in square brackets naming where that fact came from: [1], or [2][3]. Put the \
marker on EACH sentence, not one pile of numbers at the end of a paragraph. A \
sentence with no bracket is a failed answer and is discarded without being shown, \
however correct it is. Never cite a number not in the SOURCES.

Worked example. Given:

    SOURCES
    [1] NCCN · Fatigue and Cancer > What is cancer-related fatigue?
    Cancer-related fatigue is an overwhelming sense of exhaustion related to
    cancer or its treatment. It does not improve with normal amounts of rest
    or sleep, and it disrupts daily life.
    [2] NCCN · Fatigue and Cancer > During and after cancer treatment
    Physical activity, such as walking, is one of the best ways to reduce
    fatigue. Talk to your care team before starting an exercise program.

    QUESTION
    Why am I so tired, and what helps?

A GOOD answer -- plain words, one marker per sentence, ends with the care team:

    Many people with cancer feel a deep tiredness called cancer-related
    fatigue, and it is caused by the cancer or its treatment [1]. Unlike normal
    tiredness, it often does not get better with rest or sleep [1]. Gentle
    physical activity, such as walking, is one of the best ways to ease it [2].
    Ask your care team which kind of activity is safe for you before you start [2].

A BAD answer -- repeats the question, and piles the markers at the end:

    Why am I so tired? Cancer-related fatigue is exhaustion from cancer. It
    does not improve with rest. Walking helps. [1][2]

3. Patients often ask about themselves ("Will I...?", "How can I...?"). Those \
are answerable: say what the SOURCES say about people in general. Only if the \
SOURCES say nothing at all about the topic of the question, reply with exactly \
this token and nothing else:
{sentinel}
Do not write a sentence saying the sources do not cover it -- emit the bare token.

4. Describe options, what usually happens and what helps, as the guides do. \
But never make the decision for the reader: do not tell them to start, stop, \
skip or change their own treatment or medicine, or what dose to take. Make the \
final sentence suggest what they could ask or tell their care team, with its \
citation.

5. Be warm and calm, never alarming. Do not guess anyone's chances of cure or \
survival. Use short paragraphs or a few bullet points; every bullet ends with \
its [n].

6. LENGTH. Keep the whole answer under {max_words} words and finish your final \
sentence. A simple question deserves a short answer."""


DOMAIN_CORPUS = (
    "Between them the books cover the whole of clinical oncology -- every disease "
    "site, radiation physics, radiobiology and systemic therapy -- so any question "
    "the SOURCES support is in scope."
)

DOMAIN_TONGUE = (
    "You are restricted to ONE topic: cancer of the tongue (oral tongue and base of "
    "tongue) and the head & neck oncology directly needed to discuss it. If the "
    "question is about anything else, emit the not-available token."
)


USER_TEMPLATE = """SOURCES
{sources}

QUESTION
{question}

Answer the QUESTION using only the SOURCES above. If the SOURCES do not answer it, \
reply with exactly {sentinel} and nothing else.

Otherwise: work out which parts of the SOURCES bear on the question and write the \
answer in your own words -- do not copy passages across. EVERY sentence must end with \
its source number in square brackets, like [1] or [2][3]. Before you finish, check \
two things: that no sentence is missing its bracket, and that what you wrote answers \
what was actually asked rather than summarising the passages."""


# The clinician template's "if the SOURCES do not answer it" reads, to a 3B
# model, as covering every "Will I...?" question -- the sources never mention
# the reader. Measured: "Will I lose my hair during chemotherapy?" came back as
# the bare token with the answer sitting in source [1]. So patients get a
# template that reserves the token for a topic the sources do not touch.
PATIENT_USER_TEMPLATE = """SOURCES
{sources}

QUESTION
{question}

Answer the QUESTION in plain, kind words using only the SOURCES above. If it \
is asked about the reader ("Will I...?", "How can I...?"), answer with what the \
SOURCES say about people in general. Reply with exactly {sentinel} and nothing \
else ONLY if the SOURCES say nothing at all about this topic.

EVERY sentence must end with its own source number in square brackets, like [1] \
or [2][3]. Do not repeat the question. Before you finish, check that no \
sentence is missing its bracket."""


def format_sources(hits: list) -> str:
    """Render retrieved chunks as the numbered SOURCES block the prompt expects."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        header = f"[{i}] {chunk.get('breadcrumb', '')} ({chunk.get('citation', '')})"
        blocks.append(f"{header}\n{chunk['text'].strip()}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list, audience: str = "clinician") -> tuple[str, str]:
    cfg = load_config()
    sentinel = cfg["guard"]["citation"]["sentinel"]
    template = PATIENT_USER_TEMPLATE if audience == "patient" else USER_TEMPLATE
    user = template.format(
        sources=format_sources(hits), question=question.strip(), sentinel=sentinel
    )
    if audience == "patient":
        from rag.config import load_patient_config

        words = load_patient_config()["generation"]["max_answer_words"]
        return PATIENT_SYSTEM_PROMPT.format(sentinel=sentinel, max_words=words), user
    scope = cfg["guard"]["topic"].get("scope", "tongue")
    domain = DOMAIN_CORPUS if scope == "corpus" else DOMAIN_TONGUE
    system = SYSTEM_PROMPT.format(
        sentinel=sentinel,
        domain=domain,
        max_words=cfg["llm"].get("max_answer_words", 500),
    )
    return system, user


def generate(question: str, hits: list, client=None, audience: str = "clinician") -> str:
    from rag.llm import LLMClient

    client = client or LLMClient.from_config()
    system, user = build_prompt(question, hits, audience)
    return client.chat(system, user)


REPAIR_TEMPLATE = """{original}

--- 

Your previous answer was rejected because at least one \
sentence had no source marker:

---
{draft}
---

Rewrite it so that EVERY sentence ends with its source number in square \
brackets, like [1] or [2][3], drawn from the SOURCES you were given. Change \
nothing else -- do not add facts, do not remove facts, do not soften \
anything. If the SOURCES do not actually support the content, reply with \
exactly {sentinel} instead."""


def build_repair_prompt(
    question: str, hits: list, draft: str, audience: str = "clinician"
) -> str:
    """Ask the model to add the citation markers it omitted.

    Layer 3 discards an uncited answer even when it is entirely correct and
    drawn from the sources, which on short definitional questions ("what is
    brachytherapy") is the single most common way a good answer is lost. One
    bounded retry converts those into answers without weakening anything: the
    rewritten text goes through exactly the same verification, and if it still
    cannot be verified the platform still abstains.
    """
    sentinel = load_config()["guard"]["citation"]["sentinel"]
    # The SOURCES block must be repeated. Sending only the draft -- which is
    # the obvious way to write this -- asks the model to add source numbers it
    # can no longer see, and it cannot comply.
    _, original = build_prompt(question, hits, audience)
    return REPAIR_TEMPLATE.format(
        original=original, draft=draft.strip(), sentinel=sentinel
    )


# A sentence ends at .!? -- but the citation marker may sit either side of the
# stop ("... lateral border [1]." or "... lateral border. [1]"), so the
# terminator has to swallow any markers that follow it before the split.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*(?:\s*\[\d{1,2}\])*(?=\s|$)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*\u2022\u2013]|\d{1,2}[.)])\s+")


def _split_units(text: str) -> list[tuple[str, str]]:
    """Break an answer into the smallest pieces it is safe to cut at.

    Returns (separator_before, unit) pairs so the original line structure can
    be rebuilt exactly.

    Sentence splitting alone is not enough, because answers mix prose with
    bullet lists and a bullet routinely has no terminal full stop -- "- Surgery
    is preferred [1]" would be glued onto whatever followed it. So lines are
    the outer unit, and only non-list lines are split further into sentences.
    """
    units: list[tuple[str, str]] = []
    for i, line in enumerate(text.split("\n")):
        sep = "" if i == 0 else "\n"
        if not line.strip():
            units.append((sep, ""))
            continue
        if _LIST_ITEM.match(line):
            units.append((sep, line))
            continue
        pos, first = 0, True
        for m in _SENTENCE_END.finditer(line):
            piece = line[pos : m.end()]
            if piece.strip():
                units.append((sep if first else " ", piece.strip()))
                first = False
            pos = m.end()
        tail = line[pos:].strip()
        if tail:
            units.append((sep if first else " ", tail))
    return units


def fit_to_length(text: str, max_words: int) -> tuple[str, bool]:
    """Trim an answer to `max_words`, cutting only at a sentence or bullet end.

    Returns (text, was_trimmed).

    A hard token cap cuts wherever the budget runs out, which lands mid-word
    or mid-clause and leaves a dangling fragment with no citation -- the answer
    then either reads as unfinished or fails verification outright. This drops
    whole units from the end instead, so whatever remains is complete
    sentences, each still carrying its marker.

    The first unit is always kept even if it alone exceeds the budget: a
    slightly-long answer beats an empty one.
    """
    if max_words <= 0 or not text.strip():
        return text, False
    if len(text.split()) <= max_words:
        return text, False

    kept: list[tuple[str, str]] = []
    words = 0
    for sep, unit in _split_units(text):
        n = len(unit.split())
        if kept and words + n > max_words:
            break
        kept.append((sep, unit))
        words += n

    # Do not end on a lead-in whose list was just cut away: "Management
    # depends on stage [1]:" followed by nothing reads as a bug.
    while len(kept) > 1 and kept[-1][1].rstrip().endswith(":"):
        kept.pop()

    if not kept:
        return text, False

    out = "".join(sep + unit for sep, unit in kept).strip()
    # Never hand back something the citation check would reject when the
    # untrimmed answer would have passed.
    if not _CITE_IN_ANSWER.search(out):
        return text, False
    return out, True


_CITE_IN_ANSWER = re.compile(r"\[\d{1,2}\]")
