"""Doctor Knowledge & Productivity Assistant -- the portal entrypoint.

A home page plus three tools, wired together with st.navigation:

  Home                 app/views/home.py
  Knowledge Assistant  app/views/oncology_reference.py  cited answers from the textbooks
  Content Studio       app/views/content_studio.py      draft articles from a brief
  Research Platform    app/views/research_platform.py   placeholder

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

st.set_page_config(
    page_title="Doctor Knowledge & Productivity Assistant",
    page_icon="🩺",
    layout="wide",
)

# Paths are relative to this file. `url_path` fixes each tool's address
# (/knowledge, /studio, /research) independently of the file names.
HOME = st.Page("views/home.py", title="Home", icon=":material/home:", default=True)
PAGES = [
    HOME,
    st.Page("views/oncology_reference.py", title="Knowledge Assistant",
            icon=":material/menu_book:", url_path="knowledge"),
    st.Page("views/content_studio.py", title="Content Studio",
            icon=":material/edit_note:", url_path="studio"),
    st.Page("views/research_platform.py", title="Research Platform",
            icon=":material/biotech:", url_path="research"),
]

# A bar across the top rather than a sidebar list: the four destinations are
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
page.run()
