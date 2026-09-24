"""Oncology reference portal over two indexed textbooks.

A view layer only -- all retrieval, guarding, locating and generation lives in
rag/, so the same pipeline can sit behind an HTTP API later without changes.

Two modes, because they answer genuinely different questions:

  Ask   -- "what is the answer?"  Spends an LLM call, cites every fact, and
           refuses when the books do not support an answer.
  Find  -- "where is this in the books?"  No LLM, sub-second, returns
           positions rather than prose. Every result opens the real page.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import load_config  # noqa: E402
from rag.locate import locate  # noqa: E402
from rag.pipeline import answer_question, answer_question_stream  # noqa: E402

st.set_page_config(page_title="Oncology Reference", page_icon="🔬", layout="wide")

SAMPLE_QUESTIONS = [
    "What is the lymphatic drainage of the oral tongue?",
    "When is elective neck dissection indicated in oral tongue cancer?",
    "What are the dose constraints for the parotid glands?",
    "How is glioblastoma treated with radiation and temozolomide?",
    "What is the mechanism of action of cisplatin?",
]

SAMPLE_TOPICS = [
    "brachytherapy",
    "depth of invasion",
    "osteoradionecrosis",
    "linear quadratic model",
    "sentinel lymph node biopsy",
]


@st.cache_data(show_spinner=False)
def load_taxonomy():
    """The organ-system / pathology tree written by scripts/05_tag_pathologies.py."""
    import json

    from rag.config import path_of

    path = path_of("index_dir") / "taxonomy.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def scope_selector():
    """Left-hand section picker. Returns (scope_id, human label).

    Two levels: pick an organ system, then optionally narrow to one pathology
    inside it. A flat list of 77 pathologies would be unusable, and a
    system-only list could not express "tongue cancer", which is the thing
    this was asked for.
    """
    taxonomy = load_taxonomy()
    if not taxonomy:
        st.caption(
            "No taxonomy found. Run `python scripts/05_tag_pathologies.py` "
            "to enable section filtering."
        )
        return None, "All sections"

    systems = [s for s in taxonomy["systems"] if s["pathologies"]]
    labels = ["All sections (both books)"] + [
        f"{s['label']}  ({sum(p['chunks'] for p in s['pathologies']):,})" for s in systems
    ]
    # Streamlit refuses to let a widget's session_state key be written once the
    # widget exists, so "Search all" cannot just assign the selectbox back to
    # its first option. Bumping a nonce in the key builds a fresh widget
    # instead, which comes up on its default -- the standard idiom for a
    # programmatic reset.
    nonce = st.session_state.get("scope_nonce", 0)
    choice = st.selectbox("Organ system", labels, key=f"sys_choice_{nonce}")

    if choice == labels[0]:
        return None, "All sections"

    system = systems[labels.index(choice) - 1]
    paths = sorted(system["pathologies"], key=lambda p: -p["chunks"])
    sub_labels = [f"All of {system['label']}"] + [
        f"{p['label']}  ({p['chunks']:,})" for p in paths
    ]
    sub = st.selectbox("Pathology", sub_labels, key=f"path_choice_{nonce}")
    if sub == sub_labels[0]:
        return system["id"], system["label"]
    path = paths[sub_labels.index(sub) - 1]
    return path["id"], path["label"]


@st.cache_resource(show_spinner="Loading index and models (first run downloads them)...")
def boot():
    from rag.llm import LLMClient
    from rag.retriever import get_retriever

    retriever = get_retriever()
    return retriever.meta, LLMClient.from_config()


# ---------------------------------------------------------------------------
# The page viewer -- what makes a citation checkable
# ---------------------------------------------------------------------------


def page_viewer(book: str, start: int, end: int, key: str) -> None:
    """Show the actual PDF page, with paging across the section's range."""
    from rag import pdfview

    total = pdfview.page_count(book)
    if not total:
        st.warning(f"Source PDF for {book} not found in data/ -- cannot open the page.")
        return

    end = max(start, min(end, start + 20))  # a runaway span must not page forever
    state_key = f"page::{key}"
    current = st.session_state.get(state_key, start)
    current = max(start, min(current, end))

    nav_prev, nav_label, nav_next, nav_dl = st.columns([1, 3, 1, 2])
    with nav_prev:
        if st.button("← Prev", key=f"prev::{key}", disabled=current <= start,
                     use_container_width=True):
            st.session_state[state_key] = current - 1
            st.rerun()
    with nav_label:
        span = f"{start}-{end}" if end > start else f"{start}"
        st.markdown(
            f"<div style='text-align:center;padding-top:6px'>"
            f"<b>{book}</b> &nbsp; PDF page {current} &nbsp;"
            f"<span style='opacity:.6'>(section spans {span} of {total})</span></div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Next →", key=f"next::{key}", disabled=current >= end,
                     use_container_width=True):
            st.session_state[state_key] = current + 1
            st.rerun()
    with nav_dl:
        excerpt = pdfview.extract_pages(book, start, end)
        if excerpt:
            st.download_button(
                "Download these pages (PDF)",
                data=excerpt,
                file_name=f"{book}_p{start}-{end}.pdf",
                mime="application/pdf",
                key=f"dl::{key}",
                use_container_width=True,
            )

    png = pdfview.render_page(book, current)
    if png:
        st.image(png, use_container_width=True)
    else:
        st.error(f"Could not render {book} page {current}.")


def go_to_section_block(book: str, start: int, end: int, key: str) -> None:
    """A 'Go to section' toggle that opens the page viewer in place."""
    open_key = f"open::{key}"
    is_open = st.session_state.get(open_key, False)
    label = "✕ Close book view" if is_open else "📖 Go to section in the book"
    if st.button(label, key=f"btn::{key}", use_container_width=False):
        st.session_state[open_key] = not is_open
        st.session_state.pop(f"page::{key}", None)
        # Keep any enclosing expander open from here on. Closing the book view
        # should return the user to the passage, not collapse the whole card.
        st.session_state[f"stick::{key}"] = True
        st.rerun()
    if is_open:
        page_viewer(book, start, end, key)


# ---------------------------------------------------------------------------
# Ask mode
# ---------------------------------------------------------------------------


def render_sources(sources, key_prefix: str) -> None:
    st.markdown("#### Sources")
    for s in sources:
        pages = (
            f"p. {s.printed_page_start}"
            if s.printed_page_start
            else f"PDF p. {s.pdf_page_start}"
        )
        if s.pdf_page_end != s.pdf_page_start:
            pages += f"-{s.pdf_page_end}"
        key = f"{key_prefix}-src{s.number}"
        # `expanded` is not just cosmetic here. The "Go to section" button and
        # the page it opens both live INSIDE this expander, and clicking a
        # button triggers a rerun -- which rebuilds the expander from this
        # argument. Hard-coding False meant the page rendered correctly and
        # was then hidden by its own container snapping shut, which looks
        # exactly like the button doing nothing.
        stick = st.session_state.get(f"stick::{key}", False)
        with st.expander(
            f"**[{s.number}]** {s.breadcrumb}  —  {pages}", expanded=stick
        ):
            bits = [s.book, s.chapter or "n/a", s.citation, f"score {s.score}"]
            if getattr(s, "pathology_label", None):
                bits.insert(2, f"section: {s.pathology_label}")
            st.caption(" · ".join(bits))
            st.markdown(
                f"<div style='border-left:3px solid #888;padding-left:12px;"
                f"white-space:pre-wrap'>{s.text}</div>",
                unsafe_allow_html=True,
            )
            go_to_section_block(
                s.book_short or s.book,
                s.pdf_page_start,
                s.pdf_page_end,
                key,
            )


def render_answer(answer, show_trace: bool, key_prefix: str) -> None:
    if answer.status == "answered":
        st.markdown(answer.text)
        # No length banner: an over-long answer is trimmed at a sentence
        # boundary before it gets here, so there is nothing broken to warn
        # about and nothing for the reader to do.
        if answer.sources:
            render_sources(answer.sources, key_prefix)
    elif answer.status == "not_available":
        st.warning(f"**{answer.text}**")
        st.caption(
            "The question is in scope, but the indexed textbook content does not "
            "contain the answer."
        )
        if answer.sources:
            with st.expander("Closest passages that were considered"):
                render_sources(answer.sources, key_prefix)
    elif answer.status == "out_of_scope":
        st.info(f"**{answer.text}**")
    else:
        st.error(answer.text)

    st.caption(f"{answer.elapsed_ms} ms · {answer.status}")

    if show_trace:
        with st.expander("Retrieval trace"):
            st.code(json.dumps(answer.trace, indent=2), language="json")


def ask_mode(client, show_trace: bool, scope=None, scope_label="All sections") -> None:
    if "history" not in st.session_state:
        st.session_state["history"] = []

    for i, turn in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            render_answer(turn["answer"], show_trace, f"h{i}")

    placeholder = (
        f"Ask about {scope_label}..." if scope else "Ask a question..."
    )
    question = st.chat_input(placeholder) or st.session_state.pop("pending_q", None)

    if question:
        key_prefix = f"h{len(st.session_state['history'])}"
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            # Asking the same thing twice should not cost another 22 seconds.
            # Kept in session state rather than st.cache_data so the streaming
            # path can be skipped entirely on a hit -- streaming a reply we
            # already have would reintroduce the wait it exists to hide.
            cache = st.session_state.setdefault("answer_cache", {})
            ckey = (question.strip().lower(), scope)
            answer = cache.get(ckey)
            if answer is not None:
                render_answer(answer, show_trace, key_prefix)
                st.caption("↑ repeated question — served from this session's cache")
            else:
                if load_config()["llm"].get("stream", True):
                    answer = _stream_answer(question, client, scope)
                else:
                    with st.spinner("Searching the textbooks..."):
                        answer = answer_question(question, client=client, scope=scope)
                render_answer(answer, show_trace, key_prefix)
                # Only cache a real answer. Caching a refusal would make a
                # transient backend failure look permanent for the session.
                if answer.status in ("answered", "not_available", "out_of_scope"):
                    cache[ckey] = answer
                    if len(cache) > 64:
                        cache.pop(next(iter(cache)))
        st.session_state["history"].append({"question": question, "answer": answer})


def _stream_answer(question, client, scope=None):
    """Show the draft as the model writes it, then replace it with the verdict.

    Generation is ~10 tokens/s on CPU, so a 250-token answer takes 25 seconds.
    Streaming does not shorten that; it stops it being 25 seconds of blank
    spinner.

    The draft is styled as provisional on purpose. Layer 3 has not run yet, so
    this text may still be discarded -- and if it is, the placeholder is
    overwritten with the refusal and the user never keeps it. Rendering it as
    a finished answer would quietly undo the guarantee the platform makes.
    """
    status = st.empty()
    draft = st.empty()
    status.caption("Searching the textbooks...")

    def on_delta(text_so_far, sources):
        status.caption(
            f"Drafting from {len(sources)} source"
            f"{'s' if len(sources) != 1 else ''} — citations not yet verified"
        )
        draft.markdown(
            "<div style='opacity:.65;border-left:3px solid #f0ad4e;padding-left:12px'>"
            f"{text_so_far}</div>",
            unsafe_allow_html=True,
        )

    answer = answer_question_stream(
        question, client=client, on_delta=on_delta, scope=scope
    )
    status.empty()
    draft.empty()
    return answer


# ---------------------------------------------------------------------------
# Find mode -- the topic locator
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=128)
def cached_locate(topic: str, limit: int, scope: str | None):
    """Memoise locator results per (topic, limit, scope).

    Streamlit reruns the whole script on every interaction -- changing the
    result-count slider, opening a page, paging through it -- and without this
    each of those repeats a ~2 s retrieval that cannot have changed. Locations
    are plain dataclasses, so they pickle into the cache cleanly.
    """
    return locate(topic, limit=limit, scope=scope)


def find_mode(scope=None, scope_label="All sections") -> None:
    st.caption(
        "Type a topic, not a question. This finds **where** the subject is discussed "
        "in the books and takes you to the page. No model writes anything here, so "
        "nothing can be invented and it returns in about a second."
        + (f" Restricted to **{scope_label}**." if scope else "")
    )

    topic = st.text_input(
        "Topic",
        value=st.session_state.pop("pending_topic", ""),
        placeholder="e.g. osteoradionecrosis, IMRT planning, depth of invasion, cisplatin toxicity",
        label_visibility="collapsed",
    )
    limit = st.slider("How many places to show", 5, 40, 15, key="loc_limit")

    if not topic.strip():
        st.info("Enter a topic above, or pick one from the sidebar.")
        return

    with st.spinner(f"Locating '{topic}' in the books..."):
        results = cached_locate(topic, limit, scope)

    if not results:
        if scope:
            st.warning(
                f"No mention of **{topic}** in **{scope_label}**."
            )
            # A scoped miss is usually a filter problem, not an absence. Say so
            # concretely by checking the rest of the corpus, which costs one
            # more retrieval and saves the user guessing which section to try.
            elsewhere = cached_locate(topic, 3, None)
            if elsewhere:
                where = ", ".join(
                    dict.fromkeys(f"{loc.book_short} {loc.page_label}" for loc in elsewhere)
                )
                st.info(f"It does appear elsewhere in the books: {where}. "
                        "Switch the section selector to **All** to see those.")
        else:
            st.warning(f"No mention of **{topic}** was found in the indexed content.")
        return

    st.success(
        f"Found **{len(results)}** places discussing **{topic}**"
        + (f" within {scope_label}." if scope else ".")
    )

    for i, loc in enumerate(results, 1):
        header = f"**{i}. {loc.book_short}** — {loc.page_label}"
        if loc.chapter:
            header += f"  ·  ch.{loc.chapter}"
        with st.container(border=True):
            st.markdown(header)
            trail = " › ".join(loc.section_path) if loc.section_path else "(chapter body)"
            st.caption(
                f"{trail}  ·  relevance {loc.score:.2f}  ·  "
                f"{loc.n_chunks} matching passage{'s' if loc.n_chunks > 1 else ''}"
                + (f"  ·  tier {loc.tier}" if loc.tier else "")
            )
            st.markdown(
                f"<div style='border-left:3px solid #888;padding-left:12px;opacity:.9'>"
                f"{loc.snippet}</div>",
                unsafe_allow_html=True,
            )
            st.write("")
            go_to_section_block(
                loc.book_short, loc.pdf_page_start, loc.pdf_page_end, f"loc{i}"
            )


# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Oncology Reference")
    st.caption(
        "Two indexed textbooks. **Ask** a question and every fact comes back cited, "
        "or say so if the books do not cover it. **Find** a topic and jump straight "
        "to the page it is on."
    )

    try:
        meta, client = boot()
    except Exception as exc:  # noqa: BLE001 - surface any startup failure in the UI
        st.error(f"Could not load the index: {exc}")
        st.info("Build it first:  `bash scripts/run_all.sh`")
        return

    with st.sidebar:
        st.subheader("Section")
        scope, scope_label = scope_selector()
        if scope:
            st.success(f"Answering only from: **{scope_label}**")
        else:
            st.caption("Searching everything in both books.")

        st.subheader("Index")
        st.write(f"**Chunks:** {meta['chunk_count']:,}")
        st.write(f"**Books:** {', '.join(meta.get('books', []))}")
        st.write(f"**Embedding:** `{meta['embedding_model']}`")
        st.write(f"**Built:** {meta.get('built_at', 'unknown')}")
        if meta.get("partial"):
            st.warning(
                "This index was built with `--limit` and covers only part of the "
                "corpus. Answers will be incomplete. Rebuild with "
                "`bash scripts/run_all.sh`."
            )

        st.subheader("Language model")
        ok, detail = client.health()
        (st.success if ok else st.error)(f"{client.model}\n\n{detail}")
        if not ok:
            st.caption("Start it with `bash scripts/serve_llm.sh`, or set LLM_BASE_URL in .env")
            st.caption("**Find** mode still works without it.")

        show_trace = st.toggle("Show retrieval trace", value=False)

        st.subheader("Sample questions")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True, key=f"sq::{q}"):
                st.session_state["pending_q"] = q
                st.session_state["mode"] = "Ask a question"

        st.subheader("Sample topics")
        for t in SAMPLE_TOPICS:
            if st.button(t, use_container_width=True, key=f"st::{t}"):
                st.session_state["pending_topic"] = t
                st.session_state["mode"] = "Find a topic"

    mode = st.radio(
        "Mode",
        ["Ask a question", "Find a topic"],
        horizontal=True,
        key="mode",
        label_visibility="collapsed",
    )

    # An active filter has to be visible where the user is looking, not only
    # in the sidebar. A section left selected from an earlier question makes
    # the platform look broken: it answers "not available" to something the
    # books plainly cover, because it is only searching one chapter.
    if scope:
        banner, clear = st.columns([5, 1])
        with banner:
            st.warning(
                f"Searching **{scope_label}** only — other parts of the books "
                "are excluded from answers.",
                icon="🔎",
            )
        with clear:
            if st.button("Search all", use_container_width=True, key="clear_scope"):
                st.session_state["scope_nonce"] = (
                    st.session_state.get("scope_nonce", 0) + 1
                )
                st.rerun()

    if mode == "Ask a question":
        ask_mode(client, show_trace, scope, scope_label)
    else:
        find_mode(scope, scope_label)


if __name__ == "__main__":
    main()
