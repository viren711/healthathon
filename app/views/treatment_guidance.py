"""Treatment Guidance: pick a cancer, answer two or three questions, see the
ASCO recommendations ranked by how well they fit.

One page of the multi-page portal (app/streamlit_app.py is the entrypoint).
Written for clinicians. Built on the 19 ASCO guidelines parsed by
scripts/09_build_asco.py into recommendation records.

  1. Describe the case in words (or tap an example), or just choose the cancer.
  2. A few optional questions, picked from what that cancer's own
     recommendations actually depend on: the situation (first line, after
     surgery, recurrence ...) always; then the two facets that split its
     recommendations most (biomarkers, PD-L1, histology, prior drugs).
     Skip any of them.
  3. The best three recommendations are shown at once and re-ranked after
     every answer, each with its strength, evidence, why it fits, and the
     guideline's own figures and page behind a "Details" fold.

Typing a case answers the questions for you, and they stay editable.

No language model writes anything here: every statement and figure on the
page is a verbatim span of a guideline PDF.

    streamlit run app/streamlit_app.py               # via the home page
    streamlit run app/views/treatment_guidance.py    # this page alone
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag import guidance  # noqa: E402

TOP = 3  # recommendations shown before "Show more"
MAX_EXTRA_QUESTIONS = 2

# What the guideline calls the situation, in the words a clinician would use.
SETTING_LABEL = {
    "first-line": "Starting treatment (first line)",
    "second-line": "First treatment stopped working (second line)",
    "later-line": "Several treatments already tried (later line)",
    "metastatic": "Cancer has spread (metastatic)",
    "locally advanced": "Locally advanced",
    "early stage": "Early stage",
    "neoadjuvant": "Before surgery (neoadjuvant)",
    "adjuvant": "After surgery (adjuvant)",
    "maintenance": "Maintenance after a response",
    "recurrence": "Cancer has come back",
    "surveillance": "Follow-up after treatment",
    "symptom": "Managing symptoms",
}
# PD-L1 in bands: a doctor knows "about 60%", not an exact TPS.
PDL1_BAND = {"Under 1%": 0, "1 to 49%": 25, "50% or more": 60}
STRENGTH_COLOR = {"strong": "green", "moderate": "blue", "conditional": "orange", "weak": "orange"}
KEYS = ("tg_setting", "tg_tags", "tg_pdl1", "tg_hist", "tg_prior", "tg_more", "tg_base",
        "tg_unmatched")

# Sample cases, each checked against the guideline's own answer: the matching
# recommendation comes out first (eval/guidance_cases.yaml, scripts/10_eval_guidance.py).
EXAMPLES = {
    "Lung · EGFR, progressed on osimertinib":
        "Stage IV NSCLC with EGFR L858R, progressed on osimertinib after 18 months. What next?",
    "Lung · squamous, PD-L1 70%":
        "Metastatic squamous NSCLC, PD-L1 TPS 70%, EGFR and ALK negative, ECOG 0. First-line options?",
    "Melanoma · BRAF V600E":
        "Unresectable metastatic cutaneous melanoma, BRAF V600E, no prior systemic therapy. First-line?",
    "Ovarian · BRCA1 maintenance":
        "Newly diagnosed advanced epithelial ovarian cancer, germline BRCA1 mutation, complete "
        "response to first-line platinum chemotherapy. Maintenance?",
    "Thyroid · RET fusion":
        "Radioiodine-refractory thyroid cancer with a RET fusion, first-line systemic therapy?",
    "Biliary · IDH1 after chemo":
        "Advanced cholangiocarcinoma with an IDH1 mutation, progressed after "
        "gemcitabine-cisplatin-durvalumab. Next?",
}


def md(text: str) -> str:
    return text.replace("$", r"\$")


# ---------------------------------------------------------------------------
# What to ask: derived from the chosen cancer's own recommendations
# ---------------------------------------------------------------------------


def _pool(cancer: str) -> list[dict]:
    recs, _, guides = guidance.load()
    by_gid = {g["id"]: g["cancer"] for g in guides}
    return [r for r in recs if by_gid[r["guideline_id"]] == cancer]


def _tags(rec: dict) -> list[str]:
    return [b for b in rec.get("biomarkers", []) if b != "PD-L1"]


def _prior(rec: dict) -> list[str]:
    p = rec.get("prior") or {}
    return list(p.get("has", [])) if isinstance(p, dict) else []


def _hist(rec: dict) -> bool:
    return bool(re.search(r"\b(?:non-)?squamous\b", f"{rec.get('heading', '')} {rec['statement']}", re.I))


def _count(of, pool: list[dict]) -> list[str]:
    """Values of `of(rec)` across the pool, most common first."""
    n: dict[str, int] = {}
    for r in pool:
        for v in set(of(r)):
            n[v] = n.get(v, 0) + 1
    return [k for k, _ in sorted(n.items(), key=lambda kv: -kv[1])]


def settings_for(pool: list[dict]) -> list[str]:
    seen = {s for r in pool for s in r.get("settings", [])}
    return [s for s in SETTING_LABEL if s in seen]


def questions_for(cancer: str) -> list[str]:
    """The facets worth asking about this cancer, in order: the situation
    first, then the (at most two) that the most recommendations depend on."""
    pool = _pool(cancer)
    out = ["setting"] if len(settings_for(pool)) > 1 else []
    weight = {
        "pdl1": sum("PD-L1" in r.get("biomarkers", []) for r in pool),
        "tags": sum(bool(_tags(r)) for r in pool),
        "prior": sum(bool(_prior(r)) for r in pool),
        "hist": sum(_hist(r) for r in pool),
    }
    ranked = [k for k, w in sorted(weight.items(), key=lambda kv: -kv[1]) if w >= 3]
    return out + ranked[:MAX_EXTRA_QUESTIONS]


def tag_options(pool: list[dict]) -> list[str]:
    return _count(_tags, pool)[:8]


def prior_options(pool: list[dict]) -> list[str]:
    return _count(_prior, pool)[:6]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _reset_answers() -> None:
    """A new cancer has different questions: forget the old answers."""
    for k in KEYS:
        st.session_state.pop(k, None)
    st.session_state.pop("tg_text", None)


def _use_example() -> None:
    picked = st.session_state.get("tg_example")
    if picked:
        st.session_state["tg_paste"] = EXAMPLES[picked]
        _fill_from_text(EXAMPLES[picked])
    st.session_state["tg_example"] = None


def _start_over() -> None:
    _reset_answers()
    st.session_state["tg_cancer"] = None


def _fill_from_text(text: str) -> None:
    """Answer the questions from a pasted case."""
    case = guidance.read_case(text)
    ss = st.session_state
    _reset_answers()
    ss["tg_cancer"] = case.cancer or None
    ss["tg_text"] = text
    ss["tg_base"] = case  # what the reader found that the questions do not show
    if not case.cancer:
        ss["tg_unmatched"] = True
        return
    ss.pop("tg_unmatched", None)
    pool = _pool(case.cancer)
    ss["tg_setting"] = [s for s in case.settings if s in settings_for(pool)]
    ss["tg_tags"] = list(case.biomarkers)
    if case.pdl1 is not None:
        ss["tg_pdl1"] = next(k for k, v in reversed(list(PDL1_BAND.items())) if case.pdl1 >= v)
    if case.histology:
        ss["tg_hist"] = case.histology
    ss["tg_prior"] = list(case.prior)


def _paste() -> None:
    _fill_from_text(st.session_state.get("tg_paste", ""))


def build_case(cancer: str) -> guidance.Case:
    ss = st.session_state
    base: guidance.Case | None = ss.get("tg_base")
    band = ss.get("tg_pdl1") or ""
    pdl1 = PDL1_BAND.get(band)
    # The exact TPS from a pasted case stands while its band is the one chosen.
    if base is not None and base.pdl1 is not None and band and \
            next(k for k, v in reversed(list(PDL1_BAND.items())) if base.pdl1 >= v) == band:
        pdl1 = base.pdl1
    return guidance.Case(
        cancer=cancer,
        settings=list(ss.get("tg_setting") or []),
        biomarkers=list(ss.get("tg_tags") or []),
        not_biomarkers=list(base.not_biomarkers) if base else [],
        pdl1=pdl1,
        histology=ss.get("tg_hist") or "",
        prior=[p.lower() for p in ss.get("tg_prior") or []],
        ps=base.ps if base else None,
        pregnant=base.pregnant if base else False,
        question=base.question if base else "",
        text=ss.get("tg_text", ""),
    )


# ---------------------------------------------------------------------------
# Step 1 and 2: the cancer, then a few questions
# ---------------------------------------------------------------------------


def start() -> str | None:
    """Step 1: describe the case in words, or just pick the cancer."""
    st.markdown("#### 1 · Describe the case")
    st.pills("Examples", list(EXAMPLES), key="tg_example", on_change=_use_example,
             label_visibility="collapsed")
    st.text_area(
        "Case", key="tg_paste", height=100, label_visibility="collapsed",
        placeholder="Type or paste the case and your question — e.g. 'Stage IV NSCLC, EGFR exon 19 "
                    "deletion, progressed on osimertinib. What next?'",
    )
    st.button("Find recommendations", on_click=_paste, type="primary", icon=":material/search:",
              help="Reads the cancer, setting, biomarkers and prior drugs from your text "
                   "and fills in the questions below, which you can change.")
    if st.session_state.get("tg_unmatched"):
        st.caption("No cancer covered by these guidelines was found in that text. The textbooks "
                   "are checked below, or you can pick a cancer.")
    st.caption("**Or pick the cancer and answer a few questions**")
    st.pills("Cancer", guidance.cancers(), key="tg_cancer", on_change=_reset_answers,
             label_visibility="collapsed")
    return st.session_state.get("tg_cancer")


def _with(options: list[str], key: str) -> list[str]:
    """Options plus anything already selected, so a pasted case's answers
    never fall off the list."""
    return list(dict.fromkeys(options + list(st.session_state.get(key) or [])))


def ask_questions(cancer: str) -> None:
    pool = _pool(cancer)
    facets = questions_for(cancer)
    # A pasted case may have answered a facet that is not among the usual
    # questions; show it rather than let an answer act unseen.
    for facet, key in (("tags", "tg_tags"), ("pdl1", "tg_pdl1"), ("hist", "tg_hist"),
                       ("prior", "tg_prior")):
        if st.session_state.get(key) and facet not in facets:
            facets.append(facet)
    st.markdown("#### 2 · A few questions")
    if not facets:
        st.caption("Nothing more to narrow for this cancer — the recommendations are below.")
        return
    st.caption("Each answer re-ranks the results below. Skip any you do not know.")
    for facet in facets:
        if facet == "setting":
            st.caption("**Where is the patient in treatment?**")
            st.pills("Situation", settings_for(pool), key="tg_setting", selection_mode="multi",
                     format_func=SETTING_LABEL.get, label_visibility="collapsed")
        elif facet == "tags":
            st.caption("**Does any of this apply?**")
            st.pills("Details", _with(tag_options(pool), "tg_tags"), key="tg_tags", selection_mode="multi",
                     label_visibility="collapsed")
        elif facet == "pdl1":
            st.caption("**PD-L1 level (TPS)?**")
            st.pills("PD-L1", list(PDL1_BAND), key="tg_pdl1", label_visibility="collapsed")
        elif facet == "hist":
            st.caption("**Histology?**")
            st.pills("Histology", ["non-squamous", "squamous"], key="tg_hist",
                     label_visibility="collapsed")
        elif facet == "prior":
            st.caption("**Already had any of these?**")
            st.pills("Prior therapy", _with(prior_options(pool), "tg_prior"), key="tg_prior",
                     selection_mode="multi", label_visibility="collapsed")


# ---------------------------------------------------------------------------
# Step 3: ranked results
# ---------------------------------------------------------------------------


def page_view(gid: str, page: int, key: str) -> None:
    from rag import pdfview

    open_key = f"tg_open::{key}"
    is_open = st.session_state.get(open_key, False)
    if st.button("Hide page" if is_open else f"See page {page} in the guideline",
                 key=f"tg_btn::{key}",
                 icon=":material/close:" if is_open else ":material/description:"):
        st.session_state[open_key] = not is_open
        st.rerun()
    if is_open:
        png = pdfview.render_page(f"asco:{gid}", page)
        if png:
            st.image(png, width="stretch")
        else:
            st.warning("The guideline's PDF is not in data/asco/, so the page cannot be shown.")


def render_match(rank_no: int, m: guidance.Match, fit: float, key: str) -> None:
    r = m.rec
    g = guidance.guideline(r["guideline_id"])
    with st.container(border=True):
        head = f"**{rank_no}. {md(g['title'])}**"
        if r.get("number") and not r["number"].endswith("+"):
            head += f" — Recommendation {r['number']}"
        st.markdown(head)
        st.markdown(f"> {md(r['statement'])}")
        with st.container(horizontal=True, gap="small"):
            if "recommends AGAINST" in m.reasons:
                st.badge("Recommends against", icon=":material/block:", color="red")
            if r.get("strength"):
                color = STRENGTH_COLOR.get(r["strength"].split()[0].lower(), "gray")
                st.badge(f"{r['strength']} recommendation", color=color)
            if r.get("evidence"):
                st.badge(f"{r['evidence']} evidence", color="gray")
        st.progress(fit, text="Fit with your answers")
        reasons = [x for x in m.reasons if x != "recommends AGAINST"]
        if reasons:
            st.caption("Why it fits: " + " · ".join(reasons))
        with st.expander("Details", icon=":material/query_stats:"):
            if r.get("heading"):
                st.caption(md(r["heading"]))
            if r.get("version"):
                st.caption(r["version"])
            if m.stats:
                for s in m.stats:
                    st.markdown(f"- {guidance.highlight(s['text'])} *(p.{s['pdf_page']})*")
                st.caption("Figures quoted word for word from the guideline, in bold.")
            page_view(r["guideline_id"], r["pdf_page"], key)


def fit_of(m: guidance.Match, top: guidance.Match) -> float:
    """0..1 against the best match, so the bars show how far apart they are."""
    if top.score <= 0:
        return 0.5
    return min(1.0, max(0.1, m.score / top.score))


def results(cancer: str) -> None:
    case = build_case(cancer)
    ranked, cross = guidance.rank(case, k=8)
    st.markdown("#### 3 · Best matches")
    answered = sum(bool(v) for v in (case.settings, case.biomarkers, case.pdl1 is not None,
                                     case.histology, case.prior))
    st.caption(f"{len(_pool(cancer))} recommendations for {cancer.lower()}"
               + (f", ranked for: {case.summary()}" if answered else ", in the guideline's own order"
                  " until you answer a question"))
    if not ranked:
        st.warning("Nothing in the ASCO guidelines matches those answers. Try removing one.",
                   icon=":material/search_off:")
        textbook_fallback(_book_question(cancer),
                          "Nothing in the ASCO guidelines here matches this case.")
        return
    for i, m in enumerate(ranked[:TOP], 1):
        render_match(i, m, fit_of(m, ranked[0]), f"r{i}")
    if len(ranked) > TOP:
        with st.expander(f"Show {len(ranked) - TOP} more"):
            for i, m in enumerate(ranked[TOP:], TOP + 1):
                render_match(i, m, fit_of(m, ranked[0]), f"r{i}")
    if cross:
        with st.expander("Also relevant to this case"):
            for j, m in enumerate(cross, 1):
                render_match(j, m, 0.5, f"x{j}")


def _book_question(cancer: str | None) -> str:
    """The question to put to the textbooks: the text the doctor wrote, or one
    built from the cancer and the answers picked."""
    ss = st.session_state
    text = (ss.get("tg_text") or ss.get("tg_paste") or "").strip()
    if text:
        return text
    case = build_case(cancer or "")
    detail = f" ({case.summary()})" if case.summary() else ""
    return f"What are the treatment options for {(cancer or 'this cancer').lower()}{detail}?"


def textbook_fallback(question: str, why: str) -> None:
    """The ASCO guidelines here had nothing, so ask the textbooks, as the
    Knowledge Assistant would, and say plainly that the answer is from the books."""
    from rag.llm import LLMClient
    from rag.pipeline import answer_question

    st.markdown("#### From the textbooks")
    st.warning(why + " Checking the oncology textbooks instead.", icon=":material/menu_book:")
    cache = st.session_state.setdefault("tg_book_cache", {})
    key = question.strip().lower()
    answer = cache.get(key)
    if answer is None:
        try:
            with st.spinner("Searching the textbooks (this takes about half a minute)…"):
                answer = answer_question(question, client=LLMClient.from_config())
        except Exception as exc:  # noqa: BLE001 - the model server may be down
            st.error(f"Could not search the textbooks: {exc}")
            return
        if answer.status in ("answered", "not_available", "out_of_scope"):
            cache[key] = answer
    if answer.status != "answered":
        st.info("The textbooks do not cover this question either. "
                + ("" if answer.status == "not_available" else answer.text),
                icon=":material/search_off:")
        return
    st.success("**This answer is not from the ASCO guidelines.** It comes from the textbooks "
               "(DeVita, Gunderson & Tepper, MD Anderson Surgical Oncology Handbook, Oncologic "
               "Imaging), and every sentence is cited to a book, chapter and page. "
               "Check the source before relying on it.", icon=":material/menu_book:")
    st.markdown(md(answer.text))
    for src in answer.sources:
        pages = f"p. {src.printed_page_start}" if src.printed_page_start else f"PDF p. {src.pdf_page_start}"
        if src.pdf_page_end != src.pdf_page_start:
            pages += f"-{src.pdf_page_end}"
        with st.expander(f"**[{src.number}]** {src.breadcrumb} — {pages}"):
            st.caption(f"{src.book} · {src.chapter or 'n/a'} · {src.citation}")
            st.markdown(md(src.text))
    st.caption(f"{answer.elapsed_ms / 1000:.0f} s · from the textbooks, not the ASCO guidelines")
    if st.button("Open in the Knowledge Assistant", icon=":material/menu_book:", key="tg_to_ka",
                 help="Ask the same question there, with the page viewer and organ-system filters."):
        st.session_state["pending_q"] = question
        st.session_state["mode"] = "Ask a question"
        try:
            st.switch_page("views/oncology_reference.py")
        except Exception:  # noqa: BLE001 - only resolvable under the entrypoint
            st.info("Open the Knowledge Assistant from the navigation bar.")


def main() -> None:
    st.title(":material/clinical_notes: Treatment Guidance")
    st.caption("Describe the case, or pick the cancer and answer a few questions, and see the "
               "ASCO guideline recommendations that fit best.")
    st.info(
        "Decision support from published ASCO guidelines, not a treatment plan. Every "
        "recommendation and figure is quoted from the guideline; nothing is generated. "
        "If the guidelines here have nothing for a case, the oncology textbooks are searched "
        "and the answer is labelled as coming from the books. "
        "Check the guideline version and apply clinical judgement.",
        icon=":material/verified:",
    )
    if guidance.available():
        covered = guidance.cancers()
        st.caption(
            ":material/library_books: **Covered by the ASCO guidelines here:** "
            + " · ".join(covered)
            + f". {len(guidance.load()[2])} guidelines in all, including cross-cutting guidance "
            "on ctDNA testing, growth factors, cancer in pregnancy and communication. "
            "Any other cancer is not covered."
        )
    if not guidance.available():
        st.warning("The guideline records are not built yet. Run "
                   "`python scripts/09_build_asco.py` (the PDFs go in `data/asco/`).")
        return
    if st.session_state.get("guidance_pending"):
        pending = st.session_state.pop("guidance_pending")
        st.session_state["tg_paste"] = pending
        _fill_from_text(pending)
    with st.sidebar:
        st.subheader("Treatment Guidance")
        recs, stats, guides = guidance.load()
        st.caption(f"{len(guides)} ASCO guidelines · {len(recs)} recommendations")
        st.button("Start over", width="stretch", key="tg_clear", on_click=_start_over)
        with st.expander("Guidelines covered"):
            for g in sorted(guides, key=lambda g: g["cancer"]):
                st.markdown(f"- **{g['cancer']}** — {md(g['title'])}")

    cancer = start()
    if cancer:
        ask_questions(cancer)
        results(cancer)
    elif st.session_state.get("tg_unmatched") and st.session_state.get("tg_text", "").strip():
        textbook_fallback(st.session_state["tg_text"],
                          "No cancer covered by the ASCO guidelines here was found in that text.")


main()
