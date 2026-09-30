"""Home: one place to start, whichever tool the doctor needs.

Three parts, top to bottom:

  launcher   one box: pick what you want to do, type it, go. The text is
             handed to the right tool through session state, so the doctor
             lands mid-task instead of on an empty page.
  tools      a card per tool with what it does, where it runs (on this
             computer, or online), and a full-width button into it.
  status     what is live right now: the model server, which model is in
             memory, the textbook index, the Scite connection.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KNOWLEDGE = "views/oncology_reference.py"
STUDIO = "views/content_studio.py"
RESEARCH = "views/research_platform.py"

# What the launcher can do. Each entry says where the text goes and how that
# page is primed to pick it up (see go()).
ACTIONS = {
    "Ask the textbooks": {
        "page": KNOWLEDGE,
        "placeholder": "e.g. When is elective neck dissection indicated in oral tongue cancer?",
        "button": "Ask",
        "examples": [
            "What are the dose constraints for the parotid glands?",
            "How is glioblastoma treated with radiation and temozolomide?",
            "What is the lymphatic drainage of the oral tongue?",
        ],
    },
    "Draft an article": {
        "page": STUDIO,
        "placeholder": "Topic, e.g. Early warning signs of oral cancer",
        "button": "Start draft",
        "examples": [
            "Early warning signs of oral cancer",
            "Managing fatigue during radiotherapy",
            "What to expect at your first chemotherapy session",
        ],
    },
    "Search the literature": {
        "page": RESEARCH,
        "placeholder": "e.g. Does a close margin alone warrant adjuvant radiotherapy in oral SCC?",
        "button": "Search",
        "examples": [
            "SBRT versus surgery for operable stage I NSCLC",
            "De-escalation of radiotherapy in HPV-positive oropharyngeal cancer",
            "Hypofractionated whole-breast irradiation outcomes",
        ],
    },
}

TOOLS = [
    {
        "page": KNOWLEDGE,
        "icon": ":material/menu_book:",
        "title": "Knowledge Assistant",
        "tagline": "Cited answers from two oncology textbooks",
        "where": ("On this computer", "green", ":material/lock:"),
        "body": (
            "Answers drawn only from *DeVita's Cancer* and *Gunderson & Tepper's "
            "Clinical Radiation Oncology*, every fact cited to book, chapter and "
            "page. If the books do not cover it, it says so."
        ),
        "points": [
            "**Ask** a question for a cited answer",
            "**Find** a topic and jump to its pages",
            "Narrow to an organ system or pathology",
        ],
        "cta": "Open Knowledge Assistant",
    },
    {
        "page": STUDIO,
        "icon": ":material/edit_note:",
        "title": "Content Studio",
        "tagline": "Blogs, handouts and articles from a short brief",
        "where": ("On this computer", "green", ":material/lock:"),
        "body": (
            "Describe the piece, answer a few questions about audience, tone "
            "and length, and get a draft to review, edit and publish."
        ),
        "points": [
            "Questions tailored to your topic",
            "Short, medium or long drafts",
            "Download as Markdown",
        ],
        "cta": "Open Content Studio",
    },
    {
        "page": RESEARCH,
        "icon": ":material/biotech:",
        "title": "Research Platform",
        "tagline": "Evidence from the published literature",
        "where": ("Online · scite.ai", "orange", ":material/public:"),
        "body": (
            "Search published papers through Scite and get an evidence "
            "summary citing each one, with abstracts and excerpts one click "
            "away."
        ),
        "points": [
            "Summary citing every paper as [n]",
            "Authors, journal, year and DOI",
            "Follow-ups keep the context",
        ],
        "cta": "Open Research Platform",
    },
]


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def go(page: str, text: str = "") -> None:
    """Open a tool, handing it the launcher's text if there is any.

    Each tool already reads these keys: the Knowledge Assistant asks
    `pending_q` on arrival, the studio prefills its topic from `studio_topic`,
    and the research page sends `scite_pending` straight away.
    """
    text = text.strip()
    if text:
        if page == KNOWLEDGE:
            st.session_state["pending_q"] = text
            st.session_state["mode"] = "Ask a question"
        elif page == STUDIO:
            st.session_state["studio_topic"] = text
            st.session_state["studio_step"] = "brief"
        elif page == RESEARCH:
            st.session_state["scite_pending"] = text
    try:
        st.switch_page(page)
    except Exception:  # noqa: BLE001 - only resolvable under the entrypoint
        st.info("Open this page through `streamlit run app/streamlit_app.py` to switch tools.")


def _use_example() -> None:
    """Copy the picked example into the launcher box (runs before the rerun)."""
    picked = st.session_state.get("home_example")
    if picked:
        st.session_state["home_query"] = picked
    st.session_state["home_example"] = None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@st.cache_data(ttl=5, show_spinner=False)
def _model_status() -> tuple[bool, list[str]]:
    from rag.llm import LLMClient

    try:
        return LLMClient.from_config().resident_models()
    except Exception:  # noqa: BLE001 - status is informational only
        return False, []


@st.cache_data(ttl=60, show_spinner=False)
def _index_status() -> dict | None:
    from rag.config import path_of

    try:
        return json.loads((path_of("index_dir") / "index_meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _scite_status() -> str | None:
    from rag.config import _load_dotenv

    _load_dotenv()
    if os.environ.get("SCITE_API_KEY"):
        return "API"
    if os.environ.get("SCITE_MCP_API_KEY"):
        return "search"
    return None


def status_row() -> None:
    reachable, loaded = _model_status()
    meta = _index_status()
    scite = _scite_status()
    with st.container(horizontal=True, gap="small", wrap=True):
        if reachable:
            st.badge("Model server ready", icon=":material/check_circle:", color="green")
            st.badge(
                f"In memory: {', '.join(loaded)}" if loaded else "No model loaded yet",
                icon=":material/memory:", color="gray",
                help="One model is held at a time and swapped as you move between tools.",
            )
        else:
            st.badge("Model server off — run ./start", icon=":material/error:", color="red")
        if meta:
            st.badge(
                f"{len(meta.get('books', []))} textbooks · {meta['chunk_count']:,} passages",
                icon=":material/library_books:", color="blue",
            )
        else:
            st.badge("Textbook index not built", icon=":material/warning:", color="orange")
        if scite:
            st.badge("Scite connected", icon=":material/cloud_done:", color="blue")
        else:
            st.badge("Scite not configured", icon=":material/cloud_off:", color="gray",
                     help="Add SCITE_API_KEY or SCITE_MCP_API_KEY to .env.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def hero() -> None:
    st.title(":material/stethoscope: Doctor Knowledge & Productivity Assistant")
    st.markdown(
        "Look something up in the textbooks, draft something for your patients, "
        "or check the literature — from one place."
    )
    status_row()


def launcher() -> None:
    with st.container(border=True):
        action = st.segmented_control(
            "What would you like to do?",
            list(ACTIONS),
            default=list(ACTIONS)[0],
            key="home_action",
        ) or list(ACTIONS)[0]
        spec = ACTIONS[action]
        with st.form("home_launcher", border=False, enter_to_submit=True):
            box, button = st.columns([5, 1], vertical_alignment="bottom")
            with box:
                text = st.text_input(
                    action, key="home_query", placeholder=spec["placeholder"],
                    label_visibility="collapsed",
                )
            with button:
                submitted = st.form_submit_button(
                    spec["button"], type="primary", width="stretch",
                    icon=":material/arrow_forward:", icon_position="right",
                )
        st.pills(
            "Try", spec["examples"], key="home_example", on_change=_use_example,
            label_visibility="collapsed",
        )
        if submitted:
            if text.strip():
                go(spec["page"], text)
            else:
                st.warning("Type something first, or pick one of the examples.")


def tool_cards() -> None:
    st.markdown("#### Tools")
    for column, tool in zip(st.columns(len(TOOLS), gap="medium"), TOOLS):
        with column, st.container(border=True, height="stretch", gap="small"):
            st.markdown(f"### {tool['icon']} {tool['title']}")
            label, color, icon = tool["where"]
            st.badge(label, icon=icon, color=color)
            st.markdown(f"**{tool['tagline']}**")
            st.markdown(tool["body"])
            st.markdown("\n".join(f"- {p}" for p in tool["points"]))
            # A spacer keeps the buttons level across cards whose text runs
            # to different lengths.
            st.space("stretch")
            if st.button(
                tool["cta"], key=f"open::{tool['title']}", type="primary",
                width="stretch", icon=":material/arrow_forward:", icon_position="right",
            ):
                go(tool["page"])


def footer() -> None:
    st.write("")
    notes = [
        (":material/fact_check:", "Cited, or it says so",
         "Textbook answers cite every fact; when the books do not cover a "
         "question, you are told rather than given a guess."),
        (":material/lock:", "Private by default",
         "The Knowledge Assistant and Content Studio run entirely on this "
         "computer. Only the Research Platform goes online."),
        (":material/rate_review:", "Drafts are for your review",
         "Generated text is a starting point. Check every clinical statement "
         "before you rely on or publish it."),
    ]
    for column, (icon, title, text) in zip(st.columns(len(notes), gap="medium"), notes):
        with column:
            st.markdown(f"**{icon} {title}**")
            st.caption(text)


hero()
launcher()
tool_cards()
footer()
