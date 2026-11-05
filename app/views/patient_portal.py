"""Patient Portal: answers and reading material from the NCCN Guidelines for Patients.

One page of the multi-page portal (app/streamlit_app.py is the entrypoint).
Written for patients and carers, not clinicians, and built only on the 76
NCCN Guidelines for Patients indexed by scripts/07_build_patient_corpus.py.

Two modes:

  Ask    a question in plain words; the answer comes from the guides, in
         plain language, with every sentence cited and the page one click
         away. A safety screen runs first (rag/patient_safety.py).
  Learn  the guides' own reading material, unchanged: common questions,
         key points, questions to ask your doctor, and words to know. No
         language model is involved, so it is instant and exactly what
         NCCN wrote.

Both are narrowed by the two selectors above them: a topic area (detection
and prevention, treatment, supportive care, specific populations) and a guide.

Voice and languages (app/voice_ui.py, rag/sarvam.py): a question can be
spoken or typed in any of 11 Indian languages, picked or detected. It is
translated to English, answered by the unchanged pipeline, and the checked
answer is shown in the patient's language and in English, with a Listen
button for each. Without SARVAM_API_KEY the page is typed English, as before.

    streamlit run app/streamlit_app.py           # via the home page
    streamlit run app/views/patient_portal.py    # this page alone
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.voice_ui import (  # noqa: E402
    AUTO,
    Query,
    bilingual,
    english_query,
    language_banner,
    language_picker,
    prefetch_ui,
    set_ui_lang,
    ui_text,
    listen,
    read_input,
    recording_indicator,
    show_question,
    translate_toggle,
    translated,
    voice_available,
)
from rag.config import load_config, load_patient_config, patient_path  # noqa: E402
from rag.pipeline import answer_question, answer_question_stream  # noqa: E402
from rag.sarvam import language_name, load_voice_config  # noqa: E402

DEFAULT_QUESTIONS = [
    "What are the side effects of radiation therapy?",
    "How can I cope with distress during cancer treatment?",
    "What is a biopsy and what happens during one?",
    "How can I manage tiredness during treatment?",
    "Who should get screened for lung cancer?",
    "What is palliative care?",
]

MODES = ["Ask a question", "Learn"]

# Everything fixed that the page says, translated in one parallel batch the
# first time a language is chosen (saved to disk after that). In English it is
# all unused. Names, guide titles and the NCCN text itself stay English.
UI_STRINGS = [
    "Patient Portal",
    "Answers and reading material for patients and families, from the NCCN "
    "Guidelines for Patients — detection, prevention, treatment, side "
    "effects, and coping with cancer.",
    "Topic area", "Guide", "All topic areas", "Ask a question", "Learn",
    "Try one of these, or ask your own question below — type it, or press the "
    "microphone and speak in your language.",
    "Ask a question about cancer…",
    "Where this comes from", "Closest passages in the guides",
    "See this page in the guide", "Hide page", "← Previous", "Next →",
    "Open the guide on distress and coping", "Open this guide",
    "Clear conversation", "Search all the guides", "Look up a word",
    "Common questions", "Key points", "Questions to ask your doctor",
    "Words to know", "Read by chapter",
    "Download as a checklist to take to your appointment",
    "Or choose a guide to read its common questions, key points and glossary.",
    "Hide translation",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_library() -> dict | None:
    """The Learn tab's content, written by scripts/07_build_patient_corpus.py."""
    path = patient_path("library_file")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading the patient guides…")
def boot():
    from rag.llm import LLMClient
    from rag.retriever import get_retriever

    retriever = get_retriever("patient")
    return retriever.meta, LLMClient.from_config()


def md(text: str) -> str:
    """NCCN text -> safe Markdown. '$' would otherwise start a LaTeX span."""
    return text.replace("$", r"\$")


def guides_by_id(library: dict) -> dict[str, dict]:
    return {g["id"]: g for g in library["guides"]}


# ---------------------------------------------------------------------------
# Selectors -- in the main panel, above the content they narrow
# ---------------------------------------------------------------------------


def selectors(library: dict) -> tuple[str | None, dict | None, str]:
    """Topic area -> guide. Returns (scope, guide or None, label)."""
    cats = library["categories"]
    area_ids = ["all"] + list(cats)
    left, right = st.columns(2)
    with left:
        area = st.selectbox(
            ui_text("Topic area", inline=True),
            area_ids,
            format_func=lambda a: ui_text("All topic areas", inline=True) if a == "all" else cats[a],
            key="patient_area",
        )
    pool = [g for g in library["guides"] if area == "all" or g["category"] == area]
    pool.sort(key=lambda g: g["title"])
    guide_ids = ["all"] + [g["id"] for g in pool]
    # A guide picked earlier may not belong to a newly chosen area.
    if st.session_state.get("patient_guide") not in guide_ids:
        st.session_state["patient_guide"] = "all"
    by_id = guides_by_id(library)
    with right:
        gid = st.selectbox(
            ui_text("Guide", inline=True),
            guide_ids,
            format_func=lambda g: (
                f"All guides ({len(pool)})" if g == "all" else by_id[g]["title"]
            ),
            key="patient_guide",
        )
    if gid != "all":
        return gid, by_id[gid], by_id[gid]["title"]
    if area != "all":
        return area, None, cats[area]
    return None, None, "all NCCN patient guides"


def _open_guide(gid: str) -> None:
    """on_click: jump to one guide's reading material."""
    st.session_state["patient_area"] = "all"
    st.session_state["patient_guide"] = gid
    st.session_state["patient_mode"] = "Learn"


# ---------------------------------------------------------------------------
# Page viewer
# ---------------------------------------------------------------------------


def page_view(label: str, start: int, end: int, key: str) -> None:
    """A 'See this page' toggle that shows the real guide page in place."""
    from rag import pdfview

    open_key = f"pp_open::{key}"
    is_open = st.session_state.get(open_key, False)
    if st.button(
        ui_text("Hide page" if is_open else "See this page in the guide"),
        key=f"pp_btn::{key}",
        icon=":material/close:" if is_open else ":material/description:",
    ):
        st.session_state[open_key] = not is_open
        st.rerun()
    if not is_open:
        return
    total = pdfview.page_count(label)
    if not total:
        st.warning("The guide's PDF is not in data/nccn/, so the page cannot be shown.")
        return
    end = max(start, min(end, start + 10))
    cur_key = f"pp_cur::{key}"
    current = max(start, min(st.session_state.get(cur_key, start), end))
    if end > start:
        prev, mid, nxt = st.columns([1, 3, 1])
        with prev:
            if st.button(ui_text("← Previous"), key=f"pp_prev::{key}", disabled=current <= start,
                         width="stretch"):
                st.session_state[cur_key] = current - 1
                st.rerun()
        with mid:
            st.caption(f"PDF page {current} (pages {start}–{end})")
        with nxt:
            if st.button(ui_text("Next →"), key=f"pp_next::{key}", disabled=current >= end,
                         width="stretch"):
                st.session_state[cur_key] = current + 1
                st.rerun()
    png = pdfview.render_page(label, current)
    if png:
        st.image(png, width="stretch")


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


def _answer_lang() -> str:
    """The language to show things in: the chosen one, or on Auto the last
    one a question was detected in (English until then)."""
    lang = st.session_state.get("patient_lang", "en-IN")
    if lang == AUTO:
        return st.session_state.get("patient_lang::detected", "en-IN")
    return lang


def render_alerts(alerts: list[dict], key: str, lang: str = "en-IN") -> None:
    for i, alert in enumerate(alerts):
        # A crisis or emergency notice must be read in the patient's own
        # language first; the English stays beneath it for a carer or nurse.
        local = translated(alert["message"], lang, "patient")
        message = f"{local}\n\n*{alert['message']}*" if local else alert["message"]
        if alert["kind"] in ("crisis", "emergency"):
            st.error(message, icon=":material/emergency:")
        else:
            st.info(message, icon=":material/info:")
        listen(local or alert["message"], lang if local else "en-IN", f"alert::{key}::{i}")
        if alert.get("guide"):
            st.button(
                ui_text("Open the guide on distress and coping"),
                key=f"alert_guide::{key}::{i}",
                on_click=_open_guide,
                args=(alert["guide"],),
                icon=":material/menu_book:",
            )


def _md_render(text: str) -> None:
    st.markdown(md(text))


def render_sources(sources, key_prefix: str, lang: str = "en-IN") -> None:
    st.markdown("**" + ui_text("Where this comes from") + "**")
    for s in sources:
        page = s.printed_page_start or s.pdf_page_start
        key = f"{key_prefix}-src{s.number}"
        with st.expander(f"[{s.number}] {s.breadcrumb} — page {page}"):
            if s.pathology_label:
                st.caption(s.pathology_label)
            # NCCN's own words stay as written; a translation is one click away.
            st.markdown(md(s.text))
            translate_toggle(s.text, lang, key, render=_md_render)
            page_view(s.book_short, s.pdf_page_start, s.pdf_page_end, key)


def _notice(kind, text: str, lang: str, key: str, icon: str) -> None:
    """A fixed message (not available, out of scope) in both languages."""
    local = translated(text, lang, "patient")
    kind(f"{local}\n\n*{text}*" if local else text, icon=icon)
    listen(local or text, lang if local else "en-IN", key)


def render_answer(answer, key_prefix: str, lang: str = "en-IN") -> None:
    render_alerts(answer.alerts, key_prefix, lang)
    if answer.status == "answered":
        # Only the checked English answer is translated, so the citation
        # check has already run on every sentence the patient sees.
        bilingual(answer.text, translated(answer.text, lang, "patient"), lang,
                  key_prefix, render=_md_render)
        if answer.sources:
            render_sources(answer.sources, key_prefix, lang)
    elif answer.status == "excerpts":
        # The evidence is strong but no checked answer could be written, so
        # show what the guides say, in their own words.
        _notice(st.info, answer.text, lang, f"{key_prefix}-exn", ":material/menu_book:")
        for s in answer.sources:
            page = s.printed_page_start or s.pdf_page_start
            key = f"{key_prefix}-ex{s.number}"
            with st.container(border=True):
                st.caption(f"{s.breadcrumb} — page {page}")
                st.markdown(md(s.text))
                translate_toggle(s.text, lang, key, render=_md_render)
                page_view(s.book_short, s.pdf_page_start, s.pdf_page_end, key)
    elif answer.status == "not_available":
        _notice(st.warning, answer.text, lang, f"{key_prefix}-na", ":material/search_off:")
        if answer.sources:
            with st.expander(ui_text("Closest passages in the guides", inline=True)):
                render_sources(answer.sources, key_prefix, lang)
    elif answer.status == "out_of_scope":
        _notice(st.info, answer.text, lang, f"{key_prefix}-oos", ":material/info:")
    elif answer.status == "error":
        st.error(answer.text)
    if answer.status != "safety":
        st.caption(f"{answer.elapsed_ms / 1000:.0f} s · from the NCCN Guidelines for Patients")


def _stream(question: str, client, scope):
    status = st.empty()
    draft = st.empty()
    status.caption("Looking through the guides…")

    def on_delta(text_so_far, sources):
        status.caption(
            f"Writing from {len(sources)} passage{'s' if len(sources) != 1 else ''} "
            "— sources are checked when it finishes"
        )
        draft.markdown(
            "<div style='opacity:.65;border-left:3px solid #0d9488;padding-left:12px'>"
            f"{md(text_so_far)}</div>",
            unsafe_allow_html=True,
        )

    answer = answer_question_stream(
        question, client=client, on_delta=on_delta, scope=scope, corpus="patient"
    )
    status.empty()
    draft.empty()
    return answer


def _use_suggestion() -> None:
    picked = st.session_state.get("patient_suggestion")
    if picked:
        st.session_state["patient_pending_q"] = picked
    st.session_state["patient_suggestion"] = None


def ask_view(client, scope: str | None, guide: dict | None, label: str) -> None:
    history = st.session_state.setdefault("patient_history", [])

    if not history:
        suggestions = DEFAULT_QUESTIONS
        if guide:
            faqs = [
                s["title"]
                for ch in guide["chapters"]
                for s in ch["sections"]
                if s["kind"] == "faq"
            ]
            suggestions = faqs[:6] or suggestions
        lang = _answer_lang()
        st.caption(
            ui_text("Try one of these, or ask your own question below — type it, or press the "
                    "microphone and speak in your language.") if voice_available()
            else "Try one of these, or type your own question below."
        )
        # Shown in the chosen language (translated once, then cached for
        # everyone); a click still sends the English original.
        st.pills(
            "Suggested questions", suggestions, key="patient_suggestion",
            on_change=_use_suggestion, label_visibility="collapsed",
            format_func=lambda s: (f"{s} · {local}" if (local := translated(s, lang, "patient"))
                                   else s),
        )

    for i, turn in enumerate(history):
        q: Query = turn["query"]
        with st.chat_message("user"):
            show_question(q)
        with st.chat_message("assistant", avatar=":material/health_and_safety:"):
            render_answer(turn["answer"], f"ph{i}", q.lang)

    placeholder = (f"Ask about {label}…" if scope else "Ask a question about cancer…")
    # Two lines in another language; the generic line stands for both.
    if (local := translated("Ask a question about cancer…", _answer_lang(), "ui")):  # prefetched
        placeholder = f"{placeholder}\n{local}"
    langs = load_voice_config()["patient_languages"]
    choice = st.session_state.get("patient_lang", "en-IN")
    recording_indicator()
    typed = st.chat_input(placeholder, key="patient_chat", accept_audio=voice_available())
    picked = st.session_state.pop("patient_pending_q", None)
    if typed:
        q = read_input(typed, choice, langs, "patient_lang")
    elif picked:
        # A suggestion is already English: no detection, no translation in.
        q = Query(picked, picked, "en-IN", _answer_lang())
    else:
        q = read_input(st.session_state.pop("patient_pending", None), choice, langs,
                       "patient_lang")
    if q is None:
        return
    key_prefix = f"ph{len(history)}"
    with st.chat_message("user"):
        show_question(q)
    with st.chat_message("assistant", avatar=":material/health_and_safety:"):
        # Retrieval, the safety screen and the citation check all run on the
        # English question, exactly as they do for a typed English one.
        if load_config()["llm"].get("stream", True):
            answer = _stream(q.english, client, scope)
        else:
            with st.spinner("Looking through the guides…"):
                answer = answer_question(q.english, client=client, scope=scope, corpus="patient")
        render_answer(answer, key_prefix, q.lang)
    history.append({"query": q, "answer": answer})


# ---------------------------------------------------------------------------
# Learn -- NCCN's own reading material, no language model
# ---------------------------------------------------------------------------


def _sections(guide: dict, kind: str):
    for ch in guide["chapters"]:
        for s in ch["sections"]:
            if s["kind"] == kind:
                yield ch["title"], s


def _page_caption(guide: dict, s: dict) -> str:
    page = s.get("printed_page_start") or s["pdf_page_start"]
    return f"{guide['label']} ({guide['year']}), page {page}"


def guide_view(guide: dict) -> None:
    cats = load_library()["categories"]
    st.subheader(guide["title"])
    st.caption(
        f"{cats[guide['category']]} · NCCN Guidelines for Patients, {guide['year']} edition"
    )
    counts = guide.get("counts", {})
    with st.container(horizontal=True, gap="small", wrap=True):
        st.badge(f"{counts.get('faq', 0)} common questions", icon=":material/help:", color="blue")
        st.badge(f"{counts.get('key_points', 0)} key-point summaries", icon=":material/checklist:",
                 color="green")
        st.badge(f"{len(guide['glossary'])} words explained", icon=":material/dictionary:",
                 color="gray")

    faq, keys, ask, words, read = st.tabs(
        [ui_text(t, inline=True) for t in
         ("Common questions", "Key points", "Questions to ask your doctor", "Words to know",
          "Read by chapter")]
    )

    with faq:
        items = list(_sections(guide, "faq"))
        if not items:
            st.info("This guide is not organised around questions. See *Read by chapter*.")
        lang = _answer_lang()
        for i, (chapter, s) in enumerate(items):
            with st.expander(s["title"]):
                st.markdown(md(s["text"]))
                st.caption(f"{chapter} · {_page_caption(guide, s)}")
                # Both on demand: nothing is translated or spoken until asked.
                listen(f"{s['title']}\n{s['text']}", "en-IN", f"faq-{guide['id']}-{i}::en")
                translate_toggle(f"**{s['title']}**\n\n{s['text']}", lang,
                                 f"faq-{guide['id']}-{i}", render=_md_render)
                page_view(guide["label"], s["pdf_page_start"], s["pdf_page_end"],
                          f"faq-{guide['id']}-{i}")

    with keys:
        items = list(_sections(guide, "key_points"))
        if not items:
            st.info("This guide has no key-point summaries.")
        for chapter, s in items:
            st.markdown(f"**{chapter}**")
            bullets = s["bullets"] or [s["text"]]
            st.markdown("\n".join(f"- {md(b)}" for b in bullets))
            st.caption(_page_caption(guide, s))

    with ask:
        items = list(_sections(guide, "questions"))
        if not items:
            st.info("This guide has no question lists.")
        lines: list[str] = []
        for chapter, s in items:
            qs = s["bullets"] or [ln.lstrip("- ").strip() for ln in s["text"].splitlines()
                                  if ln.strip()]
            qs = [q for q in qs if q and not q.startswith("**")]
            if not qs:
                continue
            title = s["title"] if s["title"] != "Questions to ask" else chapter
            st.markdown(f"**{title}**")
            st.markdown("\n".join(f"- {md(q)}" for q in qs))
            lines += [f"\n{title}"] + [f"[ ] {q}" for q in qs]
        if lines:
            st.download_button(
                ui_text("Download as a checklist to take to your appointment"),
                data=(f"Questions to ask -- {guide['title']}\n"
                      f"From the NCCN Guidelines for Patients ({guide['year']})\n"
                      + "\n".join(lines) + "\n"),
                file_name=f"questions-{guide['id']}.txt",
                mime="text/plain",
                icon=":material/download:",
                key=f"dlq-{guide['id']}",
            )

    with words:
        terms = guide["glossary"]
        if not terms:
            st.info("This guide has no glossary.")
        else:
            q = st.text_input(ui_text("Look up a word", inline=True), key=f"gloss-{guide['id']}",
                              placeholder="e.g. biopsy, adjuvant, margin")
            qe = english_query(q) if q else ""
            if qe and qe != q.strip():
                st.caption(f"Searching for: {qe}")
            rows = [t for t in terms if not qe or qe.lower() in t["term"].lower()
                    or qe.lower() in t["definition"].lower()]
            st.dataframe(
                [{"Word": t["term"], "What it means": t["definition"]} for t in rows],
                hide_index=True, width="stretch",
                column_config={"Word": st.column_config.TextColumn(width="small")},
            )

    with read:
        titles = [ch["title"] for ch in guide["chapters"]]
        pick = st.selectbox("Chapter", range(len(titles)), format_func=lambda i: titles[i],
                            key=f"chap-{guide['id']}")
        for j, s in enumerate(guide["chapters"][pick]["sections"]):
            st.markdown(f"#### {s['title']}")
            st.markdown(md(s["text"]))
            st.caption(_page_caption(guide, s))


def search_library(library: dict, query: str, area: str | None) -> list[dict]:
    """Plain word match across questions, key points and glossary terms."""
    words = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2]
    if not words:
        return []
    hits = []
    for g in library["guides"]:
        if area and area != "all" and g["category"] != area:
            continue
        for ch in g["chapters"]:
            for s in ch["sections"]:
                if s["kind"] not in ("faq", "key_points", "body"):
                    continue
                hay = (s["title"] + " " + s["text"]).lower()
                score = sum(w in s["title"].lower() for w in words) * 3 + sum(w in hay for w in words)
                if all(w in hay for w in words):
                    hits.append({"score": score, "guide": g, "chapter": ch["title"], "section": s})
        for t in g["glossary"]:
            if all(w in (t["term"] + " " + t["definition"]).lower() for w in words):
                hits.append({"score": 4 + sum(w in t["term"].lower() for w in words) * 3,
                             "guide": g, "term": t})
    hits.sort(key=lambda h: -h["score"])
    return hits[:25]


def browse_view(library: dict, area: str) -> None:
    cats = library["categories"]
    query = st.text_input(
        ui_text("Search all the guides", inline=True),
        placeholder="e.g. hair loss, mouth sores, genetic testing",
        key="patient_lib_search",
    )
    if query.strip():
        qe = english_query(query)
        if qe != query.strip():
            st.caption(f"Searching for: {qe}")
        hits = search_library(library, qe, area)
        if not hits:
            st.info("Nothing in the guides matched all of those words. Try fewer or simpler words.")
        for i, h in enumerate(hits):
            g = h["guide"]
            if "term" in h:
                st.markdown(f"**{h['term']['term']}** — {md(h['term']['definition'])}")
                st.caption(f"Words to know · {g['title']}")
                continue
            s = h["section"]
            with st.expander(f"{s['title']} — {g['title']}"):
                st.markdown(md(s["text"]))
                st.caption(f"{h['chapter']} · {_page_caption(g, s)}")
                st.button(ui_text("Open this guide"), key=f"open-hit-{i}", on_click=_open_guide,
                          args=(g["id"],), icon=":material/menu_book:")
        return

    st.caption(ui_text("Or choose a guide to read its common questions, key points and glossary."))
    for cat, label in cats.items():
        if area != "all" and cat != area:
            continue
        guides = sorted((g for g in library["guides"] if g["category"] == cat),
                        key=lambda g: g["title"])
        st.markdown(f"##### {label}")
        with st.container(horizontal=True, gap="small", wrap=True):
            for g in guides:
                st.button(g["title"], key=f"open-{g['id']}", on_click=_open_guide,
                          args=(g["id"],))


# ---------------------------------------------------------------------------


def language_row() -> None:
    """Top of the page: "Choose your language", cycling through every
    language, beside the picker. English until the patient picks another."""
    if not voice_available():
        st.session_state["patient_lang"] = "en-IN"
        return
    langs = load_voice_config()["patient_languages"]
    line, pick = st.columns([3, 2], vertical_alignment="center")
    with line:
        language_banner(langs)
    with pick:
        language_picker(langs, "patient_lang", allow_auto=True, label_visibility="collapsed")


PORTAL_CSS = """<style>
/* Patient Portal: lighter and warmer than the clinician pages. Same glass and
   aurora (app/ui_theme.py), with the colours turned to mint, sky and a touch
   of peach, and a friendlier heading gradient. */
:root {
  --hv-base: #f4fbf9;
  --hv-blob1: rgba(52,211,153,.50);  --hv-blob2: rgba(125,211,252,.55);  --hv-blob3: rgba(253,186,116,.40);
  --hv-grad: linear-gradient(120deg, #059669 0%, #0d9488 40%, #0284c7 100%);
}
[data-testid="stPills"] button { border-radius: 999px; }
</style>"""


def main() -> None:
    st.html(PORTAL_CSS)
    pcfg = load_patient_config()
    disclaimer = " ".join(pcfg["messages"]["disclaimer"].split())
    # Everything follows the language picked below the title: English first,
    # that language beneath. English alone (the default) calls nothing. The
    # picker's value is already in session state when the script starts, so
    # the title can use it although the picker is drawn after the title.
    lang = _answer_lang() if voice_available() else "en-IN"
    set_ui_lang(lang)
    name = language_name(lang, native=False)
    prefetch_ui(UI_STRINGS + [disclaimer, f"Listen ({name})", f"Show in {name}",
                              "Listen (English)"])
    # A page with a chat box opens scrolled to its bottom, which hides the
    # title and the language picker under the navigation bar. Once per page
    # load, before any conversation, put the view back at the top.
    st.html("""<script>
      (function () {
        if (window.__hvTop) return; window.__hvTop = true;
        const go = () => { const s = document.querySelector('[data-testid="stAppScrollToBottomContainer"]');
                           if (s) s.scrollTo(0, 0); };
        const iv = setInterval(go, 250), stop = () => clearInterval(iv);
        setTimeout(stop, 45000);   // the guides can take 30+ s to load on a cold start
        ['wheel', 'touchstart', 'keydown', 'mousedown'].forEach(
          e => window.addEventListener(e, stop, { once: true, passive: true }));
      })();
    </script>""", unsafe_allow_javascript=True)
    st.title(":material/health_and_safety: " + ui_text("Patient Portal", inline=True))
    language_row()
    st.caption(ui_text(
        "Answers and reading material for patients and families, from the "
        "NCCN Guidelines for Patients — detection, prevention, treatment, side "
        "effects, and coping with cancer."
    ))
    st.info(ui_text(disclaimer), icon=":material/info:")

    library = load_library()
    if library is None:
        st.error(
            "The patient guides have not been built yet. Run "
            "`python scripts/07_build_patient_corpus.py` and then "
            "`python scripts/03_build_index.py --chunks corpus/patient/chunks.jsonl "
            "--out-dir artifacts/index_patient`."
        )
        return

    # A question handed over from the home page opens in Ask mode.
    if st.session_state.get("patient_pending"):
        st.session_state["patient_mode"] = MODES[0]

    scope, guide, label = selectors(library)
    # Seeded through session state rather than `default=`, because other
    # widgets (the guide buttons, the home launcher) set this key too.
    st.session_state.setdefault("patient_mode", MODES[0])
    mode = st.segmented_control(
        "Mode", MODES, key="patient_mode", label_visibility="collapsed",
        format_func=lambda m: ui_text(m, inline=True),
    ) or MODES[0]

    with st.sidebar:
        st.subheader("Patient Portal")
        st.caption(f"{len(library['guides'])} NCCN patient guides")
        if st.button(ui_text("Clear conversation"), width="stretch", key="patient_clear"):
            st.session_state["patient_history"] = []
            st.session_state.pop("patient_lang::detected", None)
            st.rerun()

    if mode == "Learn":
        if guide:
            guide_view(guide)
        else:
            browse_view(library, st.session_state.get("patient_area", "all"))
        return

    if not patient_path("index_dir").joinpath("index_meta.json").exists():
        st.warning(
            "Questions need the patient search index, which is not built yet. "
            "The *Learn* tab works without it."
        )
        return
    try:
        _, client = boot()
    except Exception as exc:  # noqa: BLE001 - shown to the user, not swallowed
        st.error(f"Could not load the patient search index: {exc}")
        return
    with st.sidebar:
        st.subheader("Language model")
        ok, detail = client.health()
        (st.success if ok else st.error)(f"{client.model}\n\n{detail}")
        if not ok:
            st.caption(
                "Start it with `./start`. Without it, questions show the guides' "
                "own passages instead of a written answer; *Learn* works as usual."
            )
    ask_view(client, scope, guide, label)


main()
