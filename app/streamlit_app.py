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

# Paths are relative to this file. The file name becomes the URL path
# (/oncology_reference, /content_studio, ...), so the names are kept stable.
PAGES = {
    "": [
        st.Page("views/home.py", title="Home", icon="🏠", default=True),
    ],
    "Tools": [
        st.Page("views/oncology_reference.py", title="Knowledge Assistant", icon="🔬"),
        st.Page("views/content_studio.py", title="Content Studio", icon="✍️"),
        st.Page("views/research_platform.py", title="Research Platform", icon="🧪"),
    ],
}

st.navigation(PAGES).run()
