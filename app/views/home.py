"""Home: one place to start, whichever tool the doctor needs.

Three parts, top to bottom:

  launcher   one box: pick what you want to do, type it, go. The text is
             handed to the right tool through session state, so the doctor
             lands mid-task instead of on an empty page.
  tools      one short card per tool (what it does, where it runs). The
             whole card is the link into the tool; the longer description
             is its tooltip.
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
PATIENTS = "views/patient_portal.py"
GUIDANCE = "views/treatment_guidance.py"

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
    "Find treatment guidance": {
        "page": GUIDANCE,
        "placeholder": "Paste a case, e.g. Stage IV NSCLC, EGFR exon 19 deletion, progressed on osimertinib. What next?",
        "button": "Find",
        "examples": [
            "Stage IV NSCLC, EGFR exon 19 deletion, treatment naive. First-line?",
            "Metastatic squamous NSCLC, PD-L1 TPS 70%, no driver alterations. First-line?",
            "Platinum-sensitive recurrent ovarian cancer, BRCA1 mutation. Maintenance?",
        ],
    },
    "Answer a patient question": {
        "page": PATIENTS,
        "placeholder": "e.g. How can I cope with tiredness during chemotherapy?",
        "button": "Answer",
        "examples": [
            "What are the signs of mouth cancer?",
            "How can I manage nausea during chemotherapy?",
            "Who should get screened for lung cancer?",
        ],
    },
}

TOOLS = [
    {
        "page": KNOWLEDGE,
        "icon": ":material/menu_book:",
        "title": "Knowledge Assistant",
        "tagline": "Cited answers from four oncology textbooks",
        "blurb": "Every fact cited to book, chapter and page.",
        "where": ("On this computer", "green", ":material/lock:"),
        "body": (
            "Answers drawn only from *DeVita's Cancer*, *Gunderson & Tepper's "
            "Clinical Radiation Oncology*, the *MD Anderson Surgical Oncology "
            "Handbook* and *Oncologic Imaging*, every fact cited to book, chapter "
            "and page. If the books do not cover it, it says so."
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
        "tagline": "Blogs and handouts from a short brief",
        "blurb": "Answer a few questions, get a draft to edit.",
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
        "blurb": "Papers via Scite, summarised with [n] citations.",
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
    {
        "page": GUIDANCE,
        "icon": ":material/clinical_notes:",
        "title": "Treatment Guidance",
        "tagline": "A case in, ASCO recommendations out",
        "blurb": "Ranked by strength and evidence, 19 guidelines.",
        "where": ("On this computer", "green", ":material/lock:"),
        "body": (
            "Paste a case summary; the matching recommendations from 19 ASCO "
            "guidelines come back with their strength, evidence quality and "
            "the trial figures the guideline prints."
        ),
        "points": [
            "Case read into editable fields first",
            "Ranked by biomarkers, line and prior therapy",
            "Statistics quoted, never generated",
        ],
        "cta": "Open Treatment Guidance",
    },
    {
        "page": PATIENTS,
        "icon": ":material/health_and_safety:",
        "title": "Patient Portal",
        "tagline": "Plain-language answers for patients",
        "blurb": "From NCCN patient guides, in 11 languages.",
        "where": ("On this computer", "green", ":material/lock:"),
        "body": (
            "Questions answered from the *NCCN Guidelines for Patients* in "
            "everyday words, plus each guide's common questions, key points "
            "and glossary to read."
        ),
        "points": [
            "**Ask**, with every sentence cited",
            "**Learn** from 76 NCCN patient guides",
            "Safety notices for urgent symptoms",
        ],
        "cta": "Open Patient Portal",
    },
]


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def go(page: str, text: str = "") -> None:
    """Open a tool, handing it the launcher's text if there is any.

    Each tool already reads these keys: the Knowledge Assistant asks
    `pending_q` on arrival, the studio prefills its topic from `studio_topic`,
    the research page sends `scite_pending` straight away, Treatment
    Guidance reads `guidance_pending` into its case form, and the
    Patient Portal answers `patient_pending` on arrival.
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
        elif page == GUIDANCE:
            st.session_state["guidance_pending"] = text
        elif page == PATIENTS:
            st.session_state["patient_pending"] = text
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


@st.cache_data(ttl=60, show_spinner=False)
def _patient_index_status() -> dict | None:
    from rag.config import patient_path

    try:
        return json.loads((patient_path("index_dir") / "index_meta.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _guidance_status() -> tuple[int, int] | None:
    from rag import guidance

    if not guidance.available():
        return None
    recs, _, guides = guidance.load()
    return len(guides), len(recs)


def _scite_status() -> str | None:
    from rag.config import _load_dotenv

    _load_dotenv()
    if os.environ.get("SCITE_API_KEY"):
        return "API"
    if os.environ.get("SCITE_MCP_API_KEY"):
        return "search"
    return None


def _voice_status() -> bool:
    from rag.config import _load_dotenv

    _load_dotenv()
    return bool(os.environ.get("SARVAM_API_KEY") or os.environ.get("SARVAM_MOCK"))


def status_row() -> None:
    reachable, loaded = _model_status()
    meta = _index_status()
    patient = _patient_index_status()
    scite = _scite_status()
    with st.container(horizontal=True, gap="xsmall", wrap=True):
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
        if patient:
            st.badge(
                f"{len(patient.get('books', []))} patient guides · {patient['chunk_count']:,} passages",
                icon=":material/health_and_safety:", color="blue",
            )
        else:
            st.badge("Patient index not built", icon=":material/warning:", color="orange",
                     help="Run scripts/07_build_patient_corpus.py, then 03_build_index.py "
                          "--chunks corpus/patient/chunks.jsonl --out-dir artifacts/index_patient")
        asco = _guidance_status()
        if asco:
            st.badge(f"{asco[0]} ASCO guidelines · {asco[1]} recommendations",
                     icon=":material/clinical_notes:", color="blue")
        else:
            st.badge("ASCO records not built", icon=":material/warning:", color="orange",
                     help="Run scripts/09_build_asco.py (PDFs in data/asco/).")
        if scite:
            st.badge("Scite connected", icon=":material/cloud_done:", color="blue")
        else:
            st.badge("Scite not configured", icon=":material/cloud_off:", color="gray",
                     help="Add SCITE_API_KEY or SCITE_MCP_API_KEY to .env.")
        if _voice_status():
            st.badge("Voice & 11 languages (Sarvam)", icon=":material/mic:", color="green",
                     help="Speak or type in Indian languages in the Patient Portal, and in "
                          "Hindi in the Knowledge Assistant.")
        else:
            st.badge("Voice off", icon=":material/mic_off:", color="gray",
                     help="Add SARVAM_API_KEY to .env for voice input and answers in "
                          "Indian languages.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def hero() -> None:
    st.html(
        """<div class="hv-hero">
        <span class="hv-eyebrow">Oncology · clinical assistant</span>
        <h1>Doctor Knowledge &amp; Productivity Assistant</h1>
        <p>Look something up, draft for patients, check the literature, or answer a
        patient's questions, all from one place.</p></div>"""
    )
    status_row()


def launcher() -> None:
    with st.container(border=True, key="launcher"):
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
    """One short card per tool. The title is a real link (keyboard and screen
    reader friendly); CSS stretches it over the whole card, so a click anywhere
    on the card opens the tool."""
    for i, (column, tool) in enumerate(zip(st.columns(len(TOOLS), gap="small"), TOOLS)):
        with column, st.container(border=True, key=f"toolcard-{i}", height="stretch", gap="xsmall"):
            try:
                st.page_link(tool["page"], label=tool["title"], icon=tool["icon"],
                             help=f"{tool['tagline']}. {tool['blurb']} {tool['body']}")
            except Exception:  # noqa: BLE001 - only resolvable under the entrypoint
                st.markdown(f"**{tool['icon']} {tool['title']}**")
            st.caption(f"**{tool['tagline']}**")
            label, color, icon = tool["where"]
            st.badge(label, icon=icon, color=color)


def footer() -> None:
    st.caption(
        ":material/fact_check: **Cited, or it says so**  ·  "
        ":material/lock: **Private by default**, only the Research Platform goes online  ·  "
        ":material/rate_review: **Drafts are for your review**"
    )


HOME_CSS = """<style>
/* One screen: tight vertical rhythm on the home page only. */
.st-key-home { gap: .75rem; }
[data-testid="stMainBlockContainer"] { padding-bottom: 1rem; }
.st-key-home [data-testid="stBadge"] { font-size: .72rem; }
.st-key-home [data-testid="stHorizontalBlock"] { row-gap: .4rem; }

/* Hero */
.hv-hero { padding: .2rem 0 .1rem; }
.hv-eyebrow { display: inline-block; font-size: .74rem; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; padding: .22rem .7rem; border-radius: 999px; margin-bottom: .5rem;
  color: var(--hv-teal); background: var(--hv-glass-strong); border: 1px solid var(--hv-line); }
.hv-hero h1 { margin: 0; font-size: 2.25rem; line-height: 1.08; font-weight: 800; letter-spacing: -.03em;
  background: var(--hv-grad); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent; }
.hv-hero p { margin: .3rem 0 0; font-size: .98rem; color: var(--hv-muted); }
[data-testid="stMainBlockContainer"] { padding-top: 3.2rem !important; }

/* Launcher: the command bar */
.st-key-launcher { border-radius: 20px !important; box-shadow: var(--hv-shadow), 0 0 0 1px var(--hv-line) !important; }

/* Tool cards: the title link covers the whole card; each tool has its own colour. */
.st-key-toolcard-0 { --acc: #0d9488; }  .st-key-toolcard-1 { --acc: #ea580c; }
.st-key-toolcard-2 { --acc: #4f46e5; }  .st-key-toolcard-3 { --acc: #0891b2; }
.st-key-toolcard-4 { --acc: #db2777; }
[class*="st-key-toolcard-"] {
  position: relative; cursor: pointer; overflow: hidden;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
[class*="st-key-toolcard-"]::before {      /* the colour bar along the top */
  content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px;
  background: linear-gradient(90deg, var(--acc), color-mix(in srgb, var(--acc) 40%, #67e8f9));
}
[class*="st-key-toolcard-"]::after {      /* the "this opens" cue */
  content: "Open →"; position: absolute; right: 14px; bottom: 12px; z-index: 1;
  font-size: .78rem; font-weight: 700; color: var(--acc); opacity: .75; transition: transform .2s ease, opacity .2s ease;
}
[class*="st-key-toolcard-"]:hover::after { opacity: 1; transform: translateX(3px); }
/* Streamlit wraps each element in a position:relative box; the stretched link
   must be positioned against the card, so those wrappers are made static. */
[class*="st-key-toolcard-"] [data-testid="stElementContainer"] { position: static; }
[class*="st-key-toolcard-"] a { text-decoration: none; }
[class*="st-key-toolcard-"] a::after { content: ""; position: absolute; inset: 0; z-index: 5; }
[class*="st-key-toolcard-"] a p { font-size: .95rem; line-height: 1.25; font-weight: 800; white-space: normal;
  overflow: visible; text-overflow: clip; color: var(--hv-ink); }
[class*="st-key-toolcard-"] a span[data-testid="stIconMaterial"], [class*="st-key-toolcard-"] a svg {
  color: var(--acc); font-size: 1.3rem; }
[class*="st-key-toolcard-"] [data-testid="stCaptionContainer"] { color: var(--hv-ink); }
[class*="st-key-toolcard-"]:hover {
  transform: translateY(-4px); border-color: var(--acc) !important;
  box-shadow: 0 16px 38px color-mix(in srgb, var(--acc) 30%, transparent) !important;
}
[class*="st-key-toolcard-"]:focus-within { outline: 2px solid var(--acc); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  [class*="st-key-toolcard-"]:hover { transform: none; }
}
</style>"""


with st.container(key="home"):
    st.html(HOME_CSS)
    hero()
    launcher()
    tool_cards()
    footer()
