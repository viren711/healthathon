"""The portal's look: an aurora backdrop, frosted-glass cards, gradient accents.

Pure CSS on top of Streamlit's own theme (.streamlit/config.toml), applied once
from the entrypoint. Everything is a variable on :root, so a page can restyle
itself by overriding a few of them (the Patient Portal does). Light is the default
whatever the browser prefers; the toggle at top right inverts the page to
dark, which keeps the hues, so both stay coherent.

Rules followed (ui-ux-pro-max: AI-native + glassmorphism, accessibility first):
  - text stays dark-on-light / light-on-dark with at least 4.5:1 contrast; glass
    only ever sits behind text at 55%+ opacity;
  - a visible focus ring on everything that can take focus;
  - motion is slow, decorative, and switched off for prefers-reduced-motion;
  - no external fonts or images: nothing leaves this computer.
"""

from __future__ import annotations

import streamlit as st

CSS = """<style>
:root {
  --hv-teal: #0d9488;  --hv-cyan: #0891b2;  --hv-indigo: #4f46e5;  --hv-coral: #f97316;
  --hv-ink: #0f2a2c;   --hv-muted: #4b6366;
  --hv-glass: rgba(255,255,255,.58);   --hv-glass-strong: rgba(255,255,255,.80);
  --hv-edge: rgba(255,255,255,.75);    --hv-line: rgba(13,148,136,.22);
  --hv-shadow: 0 10px 34px rgba(30,64,100,.12);
  --hv-glow: 0 10px 30px rgba(13,148,136,.30);
  --hv-base: #f3fbfa;
  --hv-blob1: rgba(45,212,191,.55);  --hv-blob2: rgba(99,102,241,.38);  --hv-blob3: rgba(56,189,248,.45);
  --hv-grad: linear-gradient(120deg, #0d9488 0%, #0891b2 45%, #4f46e5 100%);
  --hv-btn: linear-gradient(120deg, #0f766e 0%, #0e7490 50%, #4338ca 100%);   /* white text on this: 5:1+ */
}

/* ---- Backdrop: slow aurora behind everything ------------------------- */
.stApp { background: var(--hv-base); }
.stApp::before {
  content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
  background:
    radial-gradient(38% 34% at 18% 22%, var(--hv-blob1), transparent 70%),
    radial-gradient(34% 32% at 82% 14%, var(--hv-blob2), transparent 70%),
    radial-gradient(40% 36% at 70% 88%, var(--hv-blob3), transparent 70%);
  filter: blur(36px) saturate(115%);
  animation: hv-drift 38s ease-in-out infinite alternate;
}
@keyframes hv-drift {
  from { transform: translate3d(-2%, -1%, 0) scale(1); }
  to   { transform: translate3d(3%, 2%, 0) scale(1.08); }
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp > header { background: transparent !important; }
[data-testid="stMain"] { position: relative; z-index: 1; }

/* ---- Top navigation: a frosted bar ------------------------------------ */
[data-testid="stHeader"] {
  background: var(--hv-glass) !important; border-bottom: 1px solid var(--hv-edge);
  backdrop-filter: blur(16px) saturate(140%); -webkit-backdrop-filter: blur(16px) saturate(140%);
  z-index: 100;
}
[data-testid="stSidebar"] {
  background: var(--hv-glass) !important; backdrop-filter: blur(16px);
  border-right: 1px solid var(--hv-edge);
}

[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent !important; }

/* ---- Type ------------------------------------------------------------------ */
h1, h2, h3 { letter-spacing: -.02em; }
h1 { font-weight: 800 !important; }
.hv-grad-text, [data-testid="stHeading"] h1 {
  background: var(--hv-grad); -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
[data-testid="stCaptionContainer"] { color: var(--hv-muted); opacity: 1; }

/* ---- Glass cards ------------------------------------------------------------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--hv-glass); border: 1px solid var(--hv-edge) !important; border-radius: 18px;
  box-shadow: var(--hv-shadow);
  backdrop-filter: blur(14px) saturate(130%); -webkit-backdrop-filter: blur(14px) saturate(130%);
}
[data-testid="stExpander"] {
  background: var(--hv-glass); border: 1px solid var(--hv-edge); border-radius: 14px;
  backdrop-filter: blur(10px);
}
[data-testid="stAlert"] {
  border-radius: 14px; backdrop-filter: blur(8px); border: 1px solid var(--hv-edge);
}

/* ---- Buttons: a glowing gradient for the main action ----------------------- */
.stButton > button, .stDownloadButton > button, [data-testid^="stBaseButton"] {
  border-radius: 12px; transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--hv-btn) !important; color: #fff !important; border: 0 !important;
  font-weight: 700; box-shadow: var(--hv-glow);
}
[data-testid="stBaseButton-primary"] *, [data-testid="stBaseButton-primaryFormSubmit"] * { color: #fff !important; }
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  transform: translateY(-1px); filter: brightness(1.07) saturate(1.1);
}
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--hv-teal); box-shadow: var(--hv-glow); }
.stButton > button:active { transform: translateY(1px); }

/* ---- Inputs, pills, tabs ------------------------------------------------------ */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, textarea {
  background: var(--hv-glass-strong) !important; border-radius: 12px !important;
}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within, textarea:focus {
  box-shadow: 0 0 0 3px rgba(13,148,136,.28) !important;
}
[data-testid="stChatInput"] {
  border-radius: 28px; background: var(--hv-glass-strong); border: 1px solid var(--hv-edge);
  box-shadow: var(--hv-shadow);
}
[data-testid="stPills"] button, [data-testid="stSegmentedControl"] button { transition: all .15s ease; }
[data-testid="stPills"] button:hover { border-color: var(--hv-teal); transform: translateY(-1px); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--hv-grad); height: 3px; border-radius: 3px; }

/* ---- Focus and motion ---------------------------------------------------------- */
:focus-visible { outline: 2px solid var(--hv-teal) !important; outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .stApp::before { animation: none; }
  *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
}
@media (max-width: 640px) { .stApp::before { filter: blur(24px); } }
</style>"""


def apply() -> None:
    st.html(CSS)
