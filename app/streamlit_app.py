"""Doctor Knowledge & Productivity Assistant -- the portal entrypoint.

A home page plus five tools, wired together with st.navigation:

  Home                 app/views/home.py
  Knowledge Assistant  app/views/oncology_reference.py  cited answers from the textbooks
  Content Studio       app/views/content_studio.py      draft articles from a brief
  Research Platform    app/views/research_platform.py   evidence from the literature (Scite)
  Treatment Guidance   app/views/treatment_guidance.py  a case -> matching ASCO recommendations
  Patient Portal       app/views/patient_portal.py      NCCN patient guides: ask and learn

Each view is a plain script, so a single tool can also be run on its own
(`streamlit run app/views/content_studio.py`) -- which is how the UI tests
drive them.

The two tools that use a language model share one llama.cpp router that
holds a single model at a time (config/models.ini). Moving between them swaps
the model rather than running both, so neither page needs to know the other
exists.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui_theme import apply as apply_theme  # noqa: E402

st.set_page_config(
    page_title="Doctor Knowledge & Productivity Assistant",
    page_icon="🩺",
    layout="wide",
)

# Paths are relative to this file. `url_path` fixes each tool's address
# (/knowledge, /studio, /research, /guidance, /patients) independently of the file names.
HOME = st.Page("views/home.py", title="Home", icon=":material/home:", default=True)
PAGES = [
    HOME,
    st.Page("views/oncology_reference.py", title="Knowledge Assistant",
            icon=":material/menu_book:", url_path="knowledge"),
    st.Page("views/content_studio.py", title="Content Studio",
            icon=":material/edit_note:", url_path="studio"),
    st.Page("views/research_platform.py", title="Research Platform",
            icon=":material/biotech:", url_path="research"),
    st.Page("views/treatment_guidance.py", title="Treatment Guidance",
            icon=":material/clinical_notes:", url_path="guidance"),
    st.Page("views/patient_portal.py", title="Patient Portal",
            icon=":material/health_and_safety:", url_path="patients"),
]

# A bar across the top rather than a sidebar list: the five destinations are
# always one click away, and each tool keeps the sidebar for its own settings.
page = st.navigation(PAGES, position="top")

# Streamlit leaves ~6rem above the first element, which with a navigation bar
# already at the top reads as a blank band. Trim it, and keep page titles from
# wrapping to four lines on a phone.
st.html(
    """<style>
    [data-testid="stMainBlockContainer"] { padding-top: 2.5rem; }
    @media (max-width: 640px) {
      [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
      h1 { font-size: 1.9rem !important; }
    }
    </style>"""
)
# The portal's look (aurora backdrop, frosted glass, gradient accents): app/ui_theme.py.
apply_theme()
# Light/dark switch, top right. Streamlit can only change theme through its own
# settings menu (hidden here) or a reload, and a reload throws the conversation
# away. So the button flips the page client-side: it inverts the colours of the
# whole document (hue kept, so the teal stays teal) and puts images back, and
# the choice is remembered in the browser. It is relative to the system theme:
# on a dark system it gives the light look. The icon shows the look you would
# switch TO.
st.html(
    """<style>
    html.hv-flip { filter: invert(1) hue-rotate(180deg); }
    html.hv-flip img, html.hv-flip video, html.hv-flip canvas,
    html.hv-flip [data-testid="stImage"] { filter: invert(1) hue-rotate(180deg); }
    #hv-theme { position: fixed; top: .65rem; right: 1rem; z-index: 1000002;
      width: 2.1rem; height: 2.1rem; border-radius: 50%; padding: 0; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      border: 1px solid rgba(128,128,128,.45); }
    #hv-theme:hover { border-color: #0d9488; }
    #hv-theme { font-size: 1.15rem; line-height: 1; }
    </style>"""
)
st.html(
    """<script>
    (function () {
      const KEY = "hv-flip", root = document.documentElement;
      // Plain glyphs, not inline SVG: Streamlit's sanitiser drops a script
      // whose text contains "<" followed by a tag-like character.
      const SUN = "☀", MOON = "☾";
      let flipped = false;
      try { flipped = localStorage.getItem(KEY) === "1"; } catch (e) {}
      function paint() {
        const a = document.querySelector(".stApp"), b = document.getElementById("hv-theme");
        if (!a || !b) return;
        const cs = getComputedStyle(a);
        const c = (cs.backgroundColor.match(/[0-9.]+/g) || [255, 255, 255]).map(Number);
        const systemDark = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) < 128;
        const dark = systemDark !== flipped;
        b.style.background = cs.backgroundColor;
        b.style.color = cs.color;
        b.textContent = dark ? SUN : MOON;
        b.title = dark ? "Switch to light theme" : "Switch to dark theme";
        b.setAttribute("aria-label", b.title);
      }
      root.classList.toggle("hv-flip", flipped);
      if (window.__hvTheme) { paint(); return; }
      window.__hvTheme = true;
      const b = document.createElement("button");
      b.id = "hv-theme"; b.type = "button";
      b.onclick = () => {
        flipped = !flipped;
        root.classList.toggle("hv-flip", flipped);
        try { localStorage.setItem(KEY, flipped ? "1" : "0"); } catch (e) {}
        paint();
      };
      document.body.appendChild(b);
      paint();
    })();
    </script>""",
    unsafe_allow_javascript=True,
)
page.run()
