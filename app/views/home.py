"""Home: what the three tools are, and a way into each."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS = [
    {
        "page": "views/oncology_reference.py",
        "icon": "🔬",
        "title": "Knowledge Assistant",
        "tagline": "Cited answers from two oncology textbooks",
        "body": (
            "Ask a clinical question in plain English and get an answer drawn only "
            "from *DeVita's Cancer* and *Gunderson & Tepper's Clinical Radiation "
            "Oncology* — every fact cited to its book, chapter and page, and the "
            "page itself one click away. If the books do not cover it, it says so "
            "instead of guessing."
        ),
        "points": [
            "**Ask** a question for a cited answer",
            "**Find** a topic to jump straight to the pages that discuss it",
            "Narrow to one organ system or pathology",
        ],
        "cta": "Open the Knowledge Assistant",
        "status": None,
    },
    {
        "page": "views/content_studio.py",
        "icon": "✍️",
        "title": "Content Studio",
        "tagline": "Draft blogs and articles from a short brief",
        "body": (
            "Describe the piece you want to write — a patient blog, a handout, an "
            "article for colleagues. The studio asks a few questions about "
            "audience, tone and length, plus some specific to your topic, then "
            "drafts the article for you to review and edit."
        ),
        "points": [
            "Topic-specific planning questions",
            "Short, medium or long drafts, streamed as they are written",
            "Download as Markdown",
        ],
        "cta": "Open the Content Studio",
        "status": None,
    },
    {
        "page": "views/research_platform.py",
        "icon": "🧪",
        "title": "Research Platform",
        "tagline": "Coming soon",
        "body": "This space is reserved for the research platform, which is still being planned.",
        "points": [],
        "cta": "View placeholder",
        "status": "Coming soon",
    },
]


def system_status() -> None:
    """Which model the shared router currently holds.

    Worth showing because it is the whole memory story: the two tools share
    one model slot, so this changes as you move between them.
    """
    from rag.llm import LLMClient

    try:
        reachable, loaded = LLMClient.from_config().resident_models()
    except Exception:  # noqa: BLE001 - status is informational only
        reachable, loaded = False, []
    if not reachable:
        st.caption("Model server: not running — start it with `./start`.")
    elif loaded:
        st.caption(f"Model server: running · in memory now: `{', '.join(loaded)}`")
    else:
        st.caption("Model server: running · no model loaded (one loads on first use)")


st.title("Doctor Knowledge & Productivity Assistant")
st.markdown(
    "Tools for clinical reference and professional writing, running entirely on "
    "this machine — nothing you type leaves it."
)
st.write("")

for column, tool in zip(st.columns(len(TOOLS)), TOOLS):
    with column:
        with st.container(border=True, height="stretch"):
            st.markdown(f"### {tool['icon']} {tool['title']}")
            st.caption(tool["tagline"])
            if tool["status"]:
                st.info(tool["status"], icon="🕒")
            st.markdown(tool["body"])
            for point in tool["points"]:
                st.markdown(f"- {point}")
            try:
                st.page_link(tool["page"], label=tool["cta"], icon="➡️")
            except Exception:  # noqa: BLE001 - only resolvable under the entrypoint
                # Run on its own (not via app/streamlit_app.py) there is no
                # navigation to link into; say where the tool lives instead.
                st.caption(f"Open via `streamlit run app/streamlit_app.py` → {tool['title']}")

st.write("")
system_status()
