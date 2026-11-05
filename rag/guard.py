"""The three independent layers that keep the platform honest.

The requirement is that the system answers ONLY tongue-cancer questions, and
ONLY from the two source books -- otherwise it must say so. A prompt alone
cannot guarantee that, so the behaviour is enforced at three separate points
and any one of them can refuse:

  1. topic gate      -- before retrieval, is this even a tongue-cancer question?
  2. grounding gate  -- after retrieval, did we actually find supporting text?
  3. citation check  -- after generation, is the answer really tied to sources?

Layers 1 and 2 refuse without ever calling the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from rag.config import load_config

# Questions that are unambiguously in scope. The topic gate compares an
# incoming query against these as well as against the corpus centroid, which
# catches in-scope phrasings that share no literal keyword with the corpus.
SEED_QUESTIONS = [
    "What is the treatment for early stage oral tongue cancer?",
    "How is base of tongue carcinoma staged?",
    "What is the lymphatic drainage of the oral tongue?",
    "When is elective neck dissection indicated for tongue cancer?",
    "What is the role of brachytherapy in carcinoma of the oral tongue?",
    "What are the radiotherapy doses for squamous cell carcinoma of the tongue?",
    "What is the prognosis of advanced tongue cancer?",
    "What is the depth of invasion threshold for neck treatment in oral tongue cancer?",
    "How does HPV status affect base of tongue cancer outcomes?",
    "What are the complications of glossectomy?",
    "How is speech and swallowing rehabilitated after tongue surgery?",
    "What chemotherapy is used for recurrent head and neck squamous cell carcinoma?",
    "What is the risk of occult nodal metastasis in tongue cancer?",
    "What margins are required when resecting an oral tongue tumour?",
]

# Used when guard.topic.scope is "corpus": the books cover all of oncology,
# so the in-scope region has to be described by more than tongue questions.
# These span both volumes -- disease sites, radiation physics and biology,
# systemic therapy, staging, supportive care -- so that a legitimate question
# about, say, prostate brachytherapy is not measured against a tongue seed.
CORPUS_SEED_QUESTIONS = [
    "What is the standard treatment for locally advanced prostate cancer?",
    "How is breast cancer staged and what determines adjuvant therapy?",
    "What chemotherapy regimen is used for non-small cell lung cancer?",
    "How is glioblastoma treated with radiation and temozolomide?",
    "What is the role of immunotherapy in metastatic melanoma?",
    "How does fractionation affect normal tissue complication probability?",
    "What is the linear quadratic model in radiobiology?",
    "How is intensity-modulated radiation therapy planned and delivered?",
    "What are the dose constraints for the spinal cord and parotid glands?",
    "What is the mechanism of action of cisplatin and its toxicities?",
    "How are lymphomas classified and treated?",
    "What is the management of febrile neutropenia?",
    "How is pancreatic adenocarcinoma resected and adjuvantly treated?",
    "What is the role of proton therapy in paediatric tumours?",
    "How is a clinical target volume defined for a pelvic malignancy?",
    "What are the principles of palliative radiotherapy for bone metastases?",
    "How do tumour suppressor genes and oncogenes drive carcinogenesis?",
    "What is the staging system for colorectal cancer?",
]


def _settings(corpus: str = "books") -> dict:
    """Gate thresholds and refusal messages for one corpus.

    The patient corpus overrides only what config/patient.yaml sets; the
    citation rules (sentinel, range checks, repair) are shared.
    """
    cfg = load_config()
    if corpus != "patient":
        return {
            "topic": cfg["guard"]["topic"],
            "grounding": cfg["guard"]["grounding"],
            "messages": cfg["messages"],
        }
    from rag.config import load_patient_config

    pcfg = load_patient_config()
    return {
        "topic": {**cfg["guard"]["topic"], **pcfg["guard"]["topic"], "scope": "corpus"},
        "grounding": {**cfg["guard"]["grounding"], **pcfg["guard"]["grounding"]},
        "messages": {**cfg["messages"], **pcfg["messages"]},
    }


@dataclass
class GateResult:
    """Outcome of a guard layer. `allowed=False` means stop and use `message`."""

    allowed: bool
    reason: str = ""
    message: str = ""
    detail: dict | None = None


# ---------------------------------------------------------------------------
# Layer 1 -- topic gate (pre-retrieval)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _lexicon_patterns() -> tuple[re.Pattern, re.Pattern, re.Pattern]:
    lex = load_config()["lexicon"]

    def build(terms: list[str]) -> re.Pattern:
        esc = sorted((re.escape(t) for t in terms), key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(esc) + r")\b", re.IGNORECASE)

    # off_topic_sites are prefixes ("pancrea", "esophag"), so no trailing \b.
    esc = sorted((re.escape(t) for t in lex["off_topic_sites"]), key=len, reverse=True)
    off = re.compile(r"\b(?:" + "|".join(esc) + r")", re.IGNORECASE)
    return build(lex["tongue"]), build(lex["support"]), off


def _scope() -> str:
    """"tongue" (answer tongue-cancer questions only) or "corpus" (anything
    the two books cover). Set by guard.topic.scope in config/rag.yaml."""
    return load_config()["guard"]["topic"].get("scope", "tongue")


@lru_cache(maxsize=2)
def _seed_vectors_for(scope: str):
    from rag.retriever import get_retriever

    seeds = SEED_QUESTIONS + (CORPUS_SEED_QUESTIONS if scope == "corpus" else [])
    return get_retriever().embed_query_batch(seeds)


def _seed_vectors():
    """Embed the seed questions once, with the index's own embedding model."""
    return _seed_vectors_for(_scope())


@lru_cache(maxsize=1)
def _patient_seed_vectors():
    from rag.config import load_patient_config
    from rag.retriever import get_retriever

    return get_retriever("patient").embed_query_batch(load_patient_config()["seed_questions"])


def topic_gate(
    query: str, query_vec: np.ndarray | None = None, corpus: str = "books"
) -> GateResult:
    """Cheap pre-retrieval scope check. Never calls the LLM.

    What counts as "in scope" depends on guard.topic.scope:

      tongue -- the original behaviour: only tongue cancer and the head & neck
                oncology needed to discuss it. Naming another disease site is
                an outright refusal.
      corpus -- the whole of both textbooks. Every disease site, radiation
                physics, radiobiology and systemic therapy is fair game, so
                the site lexicon becomes evidence FOR relevance rather than
                against it, and only questions unrelated to the books at all
                ("write me a poem") are refused here.

    In corpus scope this layer is deliberately permissive: deciding whether
    the books actually answer a medical question is the grounding gate's job,
    and it does that on evidence rather than on vocabulary.
    """
    settings = _settings(corpus)
    gcfg = settings["topic"]
    msg = settings["messages"]["out_of_scope"].strip()
    tongue_re, support_re, site_re = _lexicon_patterns()
    scope = "corpus" if corpus == "patient" else _scope()

    if not query or len(query.strip()) < 3:
        return GateResult(False, "empty query", msg)

    has_tongue = bool(tongue_re.search(query))
    sites = sorted(set(m.group(0).lower() for m in site_re.finditer(query)))
    has_support = bool(support_re.search(query))

    # An explicit tongue term is decisive and costs nothing to check.
    if has_tongue and gcfg.get("lexicon_override", True):
        return GateResult(True, "tongue term present", detail={"sites": sites})

    if scope == "corpus":
        # These books have a chapter for every one of these sites, so naming
        # one is the strongest cheap signal that the question IS in scope.
        if sites:
            return GateResult(
                True,
                f"names a disease site the books cover: {', '.join(sites)}",
                detail={"sites": sites},
            )
        if has_support:
            return GateResult(True, "uses oncology terminology", detail={"sites": sites})
    elif sites and gcfg.get("reject_on_off_topic_site", True):
        # Tongue scope only: naming a different disease site is a refusal,
        # regardless of how similar the wording happens to be to our corpus.
        return GateResult(
            False,
            f"question is about a different disease site: {', '.join(sites)}",
            msg,
            detail={"sites": sites},
        )

    from rag.retriever import get_retriever

    retriever = get_retriever(corpus)
    if query_vec is None:
        query_vec = retriever.embed_query(query)

    seeds = _patient_seed_vectors() if corpus == "patient" else _seed_vectors()
    centroid_sim = retriever.centroid_similarity(query_vec)
    seed_sim = float(np.max(seeds @ query_vec[0]))

    detail = {
        "scope": scope,
        "centroid_similarity": round(centroid_sim, 4),
        "seed_similarity": round(seed_sim, 4),
        "has_support_term": has_support,
    }

    if scope == "corpus":
        # Layer 1 is a JUNK FILTER here, not a scope decision.
        #
        # It was built for a tongue-only corpus, where a cheap vocabulary and
        # similarity check was a reasonable proxy for "can we answer this?".
        # Over the whole of both textbooks that proxy is wrong, and it fails
        # in the worst way: on PHRASING rather than topic. Measured, "What
        # causes skin cancer?" scores 0.619 and was admitted while "state
        # major causes of skin cancer" scores 0.510 and was refused -- the
        # same question, and the books have three chapters on it.
        #
        # So the only thing refused here is a query that looks nothing like a
        # medical question on EITHER signal. Anything else goes to retrieval,
        # and the grounding gate decides on evidence. That costs one retrieval
        # (~0.9s) for a junk query and still never reaches the LLM.
        if seed_sim < gcfg["junk_max_seed"] and centroid_sim < gcfg["junk_max_centroid"]:
            return GateResult(
                False, "not related to the indexed textbooks", msg, detail=detail
            )
        return GateResult(True, "plausibly a question about the corpus", detail=detail)

    # Tongue scope keeps the original, deliberately strict behaviour: the
    # corpus genuinely does not contain other disease sites, so admitting a
    # question it cannot answer helps nobody.
    if seed_sim >= gcfg["min_seed_similarity"]:
        return GateResult(True, "close to a canonical in-scope question", detail=detail)

    if centroid_sim >= gcfg["min_centroid_similarity"] and has_support:
        return GateResult(True, "near the corpus and uses in-domain terminology", detail=detail)

    return GateResult(
        False, "not semantically related to tongue cancer", msg, detail=detail
    )


# ---------------------------------------------------------------------------
# Layer 2 -- grounding gate (post-retrieval)
# ---------------------------------------------------------------------------


def grounding_gate(hits: list, corpus: str = "books") -> GateResult:
    """Refuse before spending an LLM call if nothing retrieved is good enough."""
    settings = _settings(corpus)
    gcfg = settings["grounding"]
    msg = settings["messages"]["not_available"].strip()

    if not hits:
        return GateResult(False, "no chunks retrieved", msg, detail={"supporting": 0})

    use_rerank = hits[0].rerank_score is not None
    threshold = gcfg["min_rerank_score"] if use_rerank else gcfg["min_dense_similarity"]
    supporting = [h for h in hits if h.score >= threshold]

    detail = {
        "scored_by": "reranker" if use_rerank else "dense",
        "threshold": threshold,
        "top_score": round(hits[0].score, 4),
        "supporting": len(supporting),
    }

    if len(supporting) < gcfg["min_supporting_chunks"]:
        return GateResult(
            False,
            f"top score {hits[0].score:.3f} below threshold {threshold}",
            msg,
            detail=detail,
        )
    return GateResult(True, "sufficient supporting evidence", detail=detail)


def filter_supporting(hits: list, corpus: str = "books") -> list:
    """Choose which passages the model is shown.

    This is a different question from the one `grounding_gate` answers, and
    conflating the two costs real answers. The gate asks "is there evidence
    here at all?", and the calibrated `min_rerank_score` is fitted to exactly
    that decision -- it is swept against the *top* score of each question.
    Reusing it to select passages then discards every chunk below the bar,
    which on near-ties leaves the model a single source and nothing to
    corroborate it with: retrieval scores like 2.97 / 2.96 / 2.91 / 2.80 are
    five equally relevant passages, not one relevant and four irrelevant.

    So once the gate has passed, keep everything within `support_margin` of
    the best hit as well. The gate still decides whether to answer; this only
    decides how much context that answer gets.
    """
    gcfg = _settings(corpus)["grounding"]
    if not hits:
        return []
    use_rerank = hits[0].rerank_score is not None
    threshold = gcfg["min_rerank_score"] if use_rerank else gcfg["min_dense_similarity"]
    margin = gcfg.get("support_margin", 1.5 if use_rerank else 0.08)
    floor = min(threshold, hits[0].score - margin)
    return [h for h in hits if h.score >= floor] or hits[:1]


# ---------------------------------------------------------------------------
# Layer 3 -- citation verification (post-generation)
# ---------------------------------------------------------------------------

_CITE = re.compile(r"\[(\d{1,2})\]")


def verify_answer(answer: str, n_sources: int, corpus: str = "books") -> GateResult:
    """Reject an answer that is not demonstrably tied to the supplied sources.

    KNOWN LIMIT, stated plainly: this verifies that a citation marker EXISTS
    and points at a source that was supplied. It does not verify the marker is
    the RIGHT one, nor that the sentence it follows is actually supported by
    that source. A model that appends "[1]" to an invented sentence passes.

    Checking real support was attempted and abandoned on evidence. Scoring
    each sentence against each source with the cross-encoder does not separate
    the two populations: measured on this index, true sentences drawn from the
    sources scored -2.67 to 5.14 while fabricated ones reached -1.90, so a
    plausible invention outscored a genuine quotation. The cross-encoder
    answers "is this passage relevant to this text", which is not the same
    question as "does this passage support this claim", and no amount of
    threshold-picking turns one into the other. Doing it anyway would staple
    citations onto claims the books do not make -- strictly worse than
    refusing. An entailment model would be the right tool; there is not one in
    this stack.

    What carries the weight instead is layer 2: the sources in the prompt are
    the ones retrieval actually scored as relevant, and the user can open the
    cited page in one click and check.
    """
    cfg = load_config()
    ccfg = cfg["guard"]["citation"]
    msg = _settings(corpus)["messages"]["not_available"].strip()

    if not answer or not answer.strip():
        return GateResult(False, "model returned nothing", msg)

    if ccfg["sentinel"] in answer:
        return GateResult(False, "model emitted the not-available sentinel", msg)

    cited = {int(m) for m in _CITE.findall(answer)}

    if ccfg["require_at_least_one_citation"] and not cited:
        return GateResult(
            False,
            "answer cited no source, so it cannot be verified against the books",
            msg,
            detail={"cited": []},
        )

    out_of_range = sorted(c for c in cited if not 1 <= c <= n_sources)
    if out_of_range and ccfg["reject_out_of_range_citations"]:
        return GateResult(
            False,
            f"answer cited sources that were never supplied: {out_of_range}",
            msg,
            detail={"cited": sorted(cited), "out_of_range": out_of_range},
        )

    return GateResult(True, "answer is cited and in range", detail={"cited": sorted(cited)})
