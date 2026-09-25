"""Content Studio: draft a blog or article from a physician's brief.

Three steps, held in session state so a rerun never loses the doctor's input:

  brief      topic, a short description, and the kind of piece
  questions  fixed questions (audience, tone, length, must-include, avoid)
             plus a few the model generates for this particular topic
  draft      the article, streamed as it is written, then offered for download

All logic lives in writer/studio.py; this file only renders it.

The writer shares the model router with the Knowledge Assistant, which holds
one model at a time -- so opening this page and generating swaps the writer
into memory and the reference model out. If free RAM is short, the reference
page's retrieval models are released too (see release_reference_below_gb in
config/writer.yaml).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from writer import studio  # noqa: E402

K = "studio_"  # every session key this page owns, so nothing collides with other pages


def ss(name, default=None):
    return st.session_state.get(K + name, default)


def set_ss(name, value) -> None:
    st.session_state[K + name] = value


def goto(step: str) -> None:
    set_ss("step", step)
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar: which writer, and what state the model server is in
# ---------------------------------------------------------------------------


def sidebar() -> str | None:
    with st.sidebar:
        st.subheader("Writer model")
        models = studio.available_models()
        if not models:
            st.error("No writer model found in models/gguf.")
            st.caption("Fetch one with `bash scripts/00_fetch_models.sh`.")
            return None
        labels = [m["label"] for m in models]
        choice = st.radio(
            "Writer model", labels, key=K + "model_label", label_visibility="collapsed"
        )
        model_id = models[labels.index(choice)]["id"]

        from rag.llm import LLMClient

        reachable, loaded = LLMClient.from_config().resident_models()
        if not reachable:
            st.error("Model server is not running. Start the platform with `./start`.")
        elif model_id in loaded:
            st.success("Writer is loaded and ready.")
        else:
            st.caption(
                "The writer loads on first use (about 10 s), and the "
                "Knowledge Assistant's model is unloaded to make room."
            )
        st.caption(f"Free memory: {studio.available_memory_gb():.1f} GB")
    return model_id


# ---------------------------------------------------------------------------
# Step 1: the brief
# ---------------------------------------------------------------------------


def step_brief(model_id: str) -> None:
    cfg = studio.load_writer_config()
    st.subheader("1 · What would you like to write?")
    topic = st.text_input(
        "Topic",
        value=ss("topic", ""),
        placeholder="e.g. Early warning signs of oral cancer",
    )
    description = st.text_area(
        "Describe the piece",
        value=ss("description", ""),
        placeholder=(
            "e.g. A friendly blog post for my clinic's website encouraging "
            "adults over 40 to get their mouth checked, especially smokers."
        ),
        height=110,
    )
    types = list(cfg["article_types"])
    current = ss("article_type", types[0])
    article_type = st.selectbox(
        "Kind of piece", types, index=types.index(current) if current in types else 0
    )
    st.caption(cfg["article_types"][article_type])

    if st.button("Next: a few questions →", type="primary"):
        set_ss("topic", topic)
        set_ss("description", description)
        set_ss("article_type", article_type)
        problems = studio.Brief(topic, description, article_type, "", "", 0).problems()
        if problems:
            for p in problems:
                st.warning(p)
            return
        with st.spinner("Preparing a few questions about your topic…"):
            questions, generated = studio.generate_followups(
                topic, description, article_type, model_id
            )
        set_ss("followups", questions)
        set_ss("followups_generated", generated)
        set_ss("answers", {})
        goto("questions")


# ---------------------------------------------------------------------------
# Step 2: questions
# ---------------------------------------------------------------------------


def brief_summary() -> None:
    with st.container(border=True):
        st.markdown(f"**{ss('topic')}** · {ss('article_type')}")
        st.caption(ss("description"))


def step_questions() -> None:
    cfg = studio.load_writer_config()
    st.subheader("2 · A few questions")
    brief_summary()
    if st.button("← Edit the brief"):
        goto("brief")

    patient_facing = "clinician" not in ss("article_type", "").lower()

    left, right = st.columns(2)
    with left:
        audiences = cfg["audiences"]
        default_aud = 0 if patient_facing else audiences.index("Fellow clinicians")
        audience = st.selectbox(
            "Who is it for?", audiences,
            index=audiences.index(ss("audience")) if ss("audience") in audiences else default_aud,
        )
        tones = cfg["tones"]
        default_tone = 0 if patient_facing else tones.index("Neutral and informative")
        tone = st.selectbox(
            "Tone", tones,
            index=tones.index(ss("tone")) if ss("tone") in tones else default_tone,
        )
        lengths = list(cfg["lengths"])
        length_label = st.radio(
            "Length", lengths, horizontal=True,
            index=lengths.index(ss("length_label")) if ss("length_label") in lengths else 1,
        )
    with right:
        must_include = st.text_area(
            "Anything it must cover?", value=ss("must_include", ""), height=90,
            placeholder="e.g. self-check steps, when to see a dentist, the role of HPV",
        )
        avoid = st.text_area(
            "Anything to avoid?", value=ss("avoid", ""), height=70,
            placeholder="e.g. treatment specifics, alarming language",
        )
        cta = st.checkbox("End with a call to action", value=ss("cta", True))
        disclaimer = st.checkbox(
            "Add a medical-information disclaimer", value=ss("disclaimer", patient_facing),
            help=cfg["disclaimer"].strip(),
        )

    st.markdown("##### About your topic")
    if ss("followups_generated"):
        st.caption("Suggested for this topic by the writer model. Answer any that help — all optional.")
    else:
        st.caption(
            "General planning questions (the model server was unavailable, so "
            "topic-specific ones could not be generated). All optional."
        )
    answers = dict(ss("answers", {}))
    for i, q in enumerate(ss("followups", [])):
        answers[q] = st.text_input(q, value=answers.get(q, ""), key=f"{K}fq_{i}")

    target = cfg["lengths"][length_label]
    # ~1.35 tokens per written word at the measured ~5 tok/s, plus the prompt
    # read and a possible model swap. The token *ceiling* would overstate it.
    minutes = max(1, round((target * 1.35 / 5.0 + 25) / 60))
    st.caption(f"A {target}-word draft takes about {minutes} minute{'s' if minutes != 1 else ''} on this computer.")

    if st.button("Write the article ✍️", type="primary"):
        set_ss("audience", audience)
        set_ss("tone", tone)
        set_ss("length_label", length_label)
        set_ss("must_include", must_include)
        set_ss("avoid", avoid)
        set_ss("cta", cta)
        set_ss("disclaimer", disclaimer)
        set_ss("answers", answers)
        set_ss("article", None)
        set_ss("pending", True)
        goto("draft")


# ---------------------------------------------------------------------------
# Step 3: the draft
# ---------------------------------------------------------------------------


def current_brief() -> studio.Brief:
    cfg = studio.load_writer_config()
    return studio.Brief(
        topic=ss("topic", ""),
        description=ss("description", ""),
        article_type=ss("article_type", ""),
        audience=ss("audience", cfg["audiences"][0]),
        tone=ss("tone", cfg["tones"][0]),
        length_words=cfg["lengths"].get(ss("length_label"), 600),
        must_include=ss("must_include", ""),
        avoid=ss("avoid", ""),
        call_to_action=ss("cta", True),
        add_disclaimer=ss("disclaimer", True),
        followups=dict(ss("answers", {})),
    )


def make_room() -> None:
    """Release the reference page's retrieval models if RAM is short.

    The LLM itself is swapped by the router; this handles the embedder,
    reranker and indexes that live in this process. Measured: releasing them
    frees ~0.56 GB and costs a 15-35 s reload on the next reference question,
    so it happens only when free memory is below the configured threshold.
    """
    threshold = studio.load_writer_config().get("release_reference_below_gb", 0)
    if threshold and studio.available_memory_gb() < threshold:
        st.cache_resource.clear()
        if studio.release_reference_models():
            st.caption("Freed the Knowledge Assistant's search models to make room.")


def generate(model_id: str) -> None:
    from rag.llm import LLMError

    brief = current_brief()
    make_room()
    status = st.empty()
    draft = st.empty()
    status.info("Loading the writer model and starting the draft…", icon="⏳")

    def on_delta(text: str) -> None:
        words = len(text.split())
        status.info(
            f"Writing… {words} of about {brief.length_words} words", icon="✍️"
        )
        draft.markdown(text)

    try:
        article = studio.write_article(brief, model_id, on_delta=on_delta)
    except LLMError as exc:
        status.empty()
        draft.empty()
        st.error(f"The writer model could not be reached: {exc}")
        st.caption("Your answers are kept — start the model server with `./start` and try again.")
        set_ss("pending", False)
        return
    status.empty()
    draft.empty()
    set_ss("article", article)
    set_ss("pending", False)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "article"


def step_draft(model_id: str) -> None:
    st.subheader("3 · Your draft")
    brief_summary()

    if ss("pending"):
        generate(model_id)

    article = ss("article")
    if article is None:
        if st.button("← Back to the questions"):
            goto("questions")
        return

    st.warning(
        "**Draft for your review.** It is written from the model's general "
        "knowledge — not from the reference textbooks — so it cannot cite "
        "sources and can be wrong. Check every clinical statement before "
        "publishing it under your name.",
        icon="🩺",
    )

    with st.container(border=True):
        st.markdown(article.text)

    st.caption(
        f"{article.words} words (asked for ~{article.target_words}) · "
        f"{article.elapsed_s:.0f} s · {article.model}"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Download (.md)",
            data=article.text,
            file_name=f"{slug(ss('topic', 'article'))}.md",
            mime="text/markdown",
            width="stretch",
        )
    with c2:
        if st.button("Write another version", width="stretch"):
            set_ss("article", None)
            set_ss("pending", True)
            st.rerun()
    with c3:
        if st.button("Change my answers", width="stretch"):
            goto("questions")
    with c4:
        if st.button("Start a new article", width="stretch"):
            for key in [k for k in st.session_state if k.startswith(K) and k != K + "model_label"]:
                del st.session_state[key]
            goto("brief")

    with st.expander("Markdown source (to copy into your website or editor)"):
        st.code(article.text, language="markdown")


# ---------------------------------------------------------------------------

st.title("✍️ Content Studio")
st.caption(
    "Describe the piece you want to write. You will get a few questions, then a "
    "draft to review, edit and publish."
)

model = sidebar()
if model is not None:
    step = ss("step", "brief")
    if step == "questions" and not ss("followups"):
        step = "brief"
    if step == "draft" and not ss("topic"):
        step = "brief"
    {"brief": step_brief, "questions": lambda _m: step_questions(), "draft": step_draft}[step](model)
