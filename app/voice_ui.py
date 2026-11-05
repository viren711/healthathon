"""Voice and language widgets shared by the Patient Portal and the Knowledge Assistant.

The pages keep their retrieval in English. This module sits on either side
of it:

  in   read_input()   the chat box's text or recording -> a Query carrying
                      the original words, their English for retrieval, and
                      the language to answer in
  out  translated()   a checked English answer -> the user's language
       listen()       a "Listen" button that reads text aloud on click only

Both pages start in English, and typed English never calls Sarvam: only a
recording, or a language other than English chosen on the page, does.
Everything degrades to the old behaviour when SARVAM_API_KEY is absent:
typed English in, English out, no microphone.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import streamlit as st

from rag.sarvam import (
    SarvamError,
    dominant_script,
    get_voice,
    language_name,
    load_voice_config,
)

AUTO = "auto"


@dataclass
class Query:
    original: str          # as typed, or as heard
    english: str           # what retrieval and the safety screen see
    source_lang: str       # language the question was in
    lang: str              # language to answer in
    via_voice: bool = False
    note: str | None = None  # shown under the question, e.g. a fallback


def voice_available() -> bool:
    return get_voice() is not None


def language_label(code: str) -> str:
    if code == AUTO:
        return "Auto-detect"
    native, english = language_name(code), language_name(code, native=False)
    return native if native == english else f"{native} · {english}"


def language_picker(codes: list[str], key: str, allow_auto: bool,
                    label: str = "Language", label_visibility: str = "visible") -> str:
    """A selectbox of languages for the main panel, starting on English.
    Returns a code or AUTO."""
    options = codes + ([AUTO] if allow_auto else [])
    st.session_state.setdefault(key, "en-IN")
    detected = st.session_state.get(f"{key}::detected")

    def fmt(code: str) -> str:
        if code == AUTO and detected:
            return f"Auto-detect (last: {language_name(detected)})"
        return language_label(code)

    disabled = not voice_available()
    if disabled:
        st.session_state[key] = "en-IN"
    return st.selectbox(
        label, options, key=key, format_func=fmt, disabled=disabled,
        label_visibility=label_visibility,
        help=(
            "Speak or type in this language; answers come back in it and in English."
            if not disabled else "Voice and languages need SARVAM_API_KEY in .env."
        ),
    )


def read_input(value, choice: str, allowed: list[str], picker_key: str,
               stt_mode: str = "transcribe") -> Query | None:
    """The chat box's value -> a Query, or None if there is nothing to answer.

    `value` is what st.chat_input returns (a string, or a ChatInputValue with
    .text and .audio when accept_audio=True), or a plain string handed over
    from a button or the home page.
    """
    # A question re-sent by the "Ask in <language>" button below.
    value = value or st.session_state.pop(f"{picker_key}::retry", None)
    if not value:
        return None
    text = value if isinstance(value, str) else (value.text or "")
    audio = None if isinstance(value, str) else getattr(value, "audio", None)
    v = get_voice()
    fixed = None if choice == AUTO else choice

    if audio is not None:
        if v is None:
            st.warning("Voice questions need SARVAM_API_KEY in .env. Please type instead.")
            return None
        try:
            with st.spinner("Listening…"):
                text, heard = v.transcribe(audio.getvalue(), fixed, stt_mode)
        except SarvamError as exc:
            st.error(str(exc), icon=":material/mic_off:")
            return None
        source = fixed or heard or "en-IN"
        via_voice = True
    else:
        text = text.strip()
        if not text:
            return None
        via_voice = False
        if v is None:
            return Query(text, text, "en-IN", "en-IN")
        if fixed == "en-IN":
            # English is chosen: nothing is sent to Sarvam. Text plainly in
            # another script gets a one-click offer to ask in that language.
            script = dominant_script(text)
            if script not in (None, "latn"):
                _offer_language(text, "hi-IN" if script == "deva" else script,
                                allowed, picker_key)
                return None
            source = "en-IN"
        elif fixed:
            # A language was chosen, so the script alone says whether this is
            # it or English: no language-ID call.
            script = dominant_script(text)
            if script == "latn":
                source = "en-IN"
            elif script == "deva":
                source = fixed if fixed in ("hi-IN", "mr-IN") else "hi-IN"
            else:
                source = script if script in allowed else fixed
        else:
            try:
                source = v.detect(text, allowed)
            except SarvamError:
                source = "en-IN"

    note = None
    if source not in allowed:
        note = (f"Heard {language_name(source, native=False)}, which this page does not "
                "support yet — answering in English.")
        source_for_tr = source if source in load_voice_config()["languages"] else "en-IN"
        answer_lang = "en-IN"
    else:
        source_for_tr = source
        answer_lang = fixed or source
    if choice == AUTO and answer_lang != "en-IN":
        st.session_state[f"{picker_key}::detected"] = answer_lang

    english = text
    if source_for_tr != "en-IN" and v is not None:
        try:
            with st.spinner("Translating your question…"):
                english = v.translate(text, source_for_tr, "en-IN", "to_english")
        except SarvamError as exc:
            note = f"{exc} Searching with the question as written."
    return Query(text, english, source_for_tr, answer_lang, via_voice, note)


def _offer_language(text: str, lang: str, allowed: list[str], picker_key: str) -> None:
    def switch() -> None:
        st.session_state[picker_key] = lang
        st.session_state[f"{picker_key}::retry"] = text

    if lang not in allowed or get_voice() is None:
        st.info("English is selected, and this page cannot answer in that language. "
                "Please ask in English.", icon=":material/translate:")
        return
    st.info(f"This looks like {language_name(lang)} ({language_name(lang, native=False)}), "
            "but English is selected, so nothing was sent. Choose your language above, "
            "or:", icon=":material/translate:")
    st.button(f"Ask in {language_name(lang)}", key=f"{picker_key}::offer",
              on_click=switch, icon=":material/translate:")


def show_question(q: Query) -> None:
    """The user's turn: their own words, and the English searched for."""
    st.write(q.original)
    bits = []
    if q.via_voice:
        bits.append(f"🎤 heard in {language_name(q.source_lang)}")
    if q.english.strip() != q.original.strip():
        bits.append(f"searched in English as: *{q.english}*")
    if bits:
        st.caption(" · ".join(bits))
    if q.note:
        st.caption(q.note)


def translated(text: str, lang: str, purpose: str) -> str | None:
    """English text -> `lang`, or None if it could not be translated."""
    v = get_voice()
    if v is None or lang == "en-IN" or not text.strip():
        return None
    memo = st.session_state.setdefault("voice_memo", {})
    key = (purpose, lang, hashlib.sha1(text.encode("utf-8")).hexdigest())
    if key not in memo:
        try:
            memo[key] = v.translate(text, "en-IN", lang, purpose)
        except SarvamError as exc:
            st.caption(f"Could not translate: {exc}")
            return None
    return memo[key]


def english_query(text: str) -> str:
    """A search box's words in English. Latin text is returned untouched with no
    call; text in another script is translated (once, then cached) so the
    English-only guides can be searched. Falls back to the text as typed."""
    text = text.strip()
    script = dominant_script(text)
    v = get_voice()
    if v is None or script in (None, "latn"):
        return text
    src = "hi-IN" if script == "deva" else script
    memo = st.session_state.setdefault("voice_memo", {})
    key = ("to_english", src, hashlib.sha1(text.encode("utf-8")).hexdigest())
    if key not in memo:
        try:
            memo[key] = v.translate(text, src, "en-IN", "to_english").strip()
        except SarvamError as exc:
            st.caption(f"Could not translate your search: {exc}")
            return text
    return memo[key]


def set_ui_lang(lang: str) -> None:
    """The language the page's own buttons and labels carry beneath the English.
    The Patient Portal sets it every run; other pages leave it English."""
    st.session_state["ui_lang"] = lang


def ui_lang() -> str:
    return st.session_state.get("ui_lang", "en-IN")


def ui_text(text: str, inline: bool = False) -> str:
    """A button, label or caption: English alone, or with the chosen language
    beneath it (" · " between them for single-line widgets, which cannot break)."""
    # Memo only: prefetch_ui() fills it in one parallel batch, so drawing a
    # button never waits on, or pays for, a translation of its own.
    memo = st.session_state.get("voice_memo", {})
    local = memo.get(("ui", ui_lang(), hashlib.sha1(text.encode("utf-8")).hexdigest()))
    if not local or local.strip() == text.strip():
        return text
    return f"{text} · {local}" if inline else f"{text}\n\n{local}"


def prefetch_ui(strings: list[str]) -> None:
    """Translate all of a page's fixed strings at once (4 in parallel) the first
    time a language is shown, so the page does not wait on them one by one.
    Saved to disk afterwards, so this costs nothing the next time."""
    lang, v = ui_lang(), get_voice()
    if v is None or lang == "en-IN":
        return
    memo = st.session_state.setdefault("voice_memo", {})

    def key(s: str):
        return ("ui", lang, hashlib.sha1(s.encode("utf-8")).hexdigest())

    todo = [s for s in dict.fromkeys(strings) if s.strip() and key(s) not in memo]
    if not todo:
        return

    def one(s: str):
        try:
            return s, v.translate(s, "en-IN", lang, "ui")
        except SarvamError:
            return s, None  # shown in English only

    with st.spinner("Preparing your language…"):
        with ThreadPoolExecutor(max_workers=load_voice_config()["limits"]["parallel_requests"]) as pool:
            for s, out in pool.map(one, todo):
                if out:
                    memo[key(s)] = out


def listen(text: str, lang: str, key: str, voice: str = "patient") -> None:
    """A Listen button. Audio is made only when it is pressed, then cached.
    `voice` is "patient" or "clinician" (voices set in config/voice.yaml)."""
    v = get_voice()
    if v is None or not text.strip():
        return
    store = st.session_state.setdefault("voice_audio", {})
    clicked = st.button(
        ui_text(f"Listen ({language_name(lang, native=False)})"), key=f"listen::{key}",
        icon=":material/volume_up:", type="tertiary",
    )
    if clicked and key not in store:
        try:
            with st.spinner("Preparing audio…"):
                store[key] = v.speak(text, lang, voice)
        except SarvamError as exc:
            st.error(str(exc), icon=":material/volume_off:")
            return
    if key in store:
        data, mime = store[key]
        st.audio(data, format=mime, autoplay=clicked)


def bilingual(english: str, local: str | None, lang: str, key: str, render=None,
              voice: str = "patient") -> None:
    """Show text in the user's language, followed by the English.

    `render` draws one version (default st.markdown); pages pass their own to
    escape '$' and the like.
    """
    render = render or st.markdown
    if not local:
        render(english)
        listen(english, "en-IN", f"{key}::en", voice)
        return
    st.caption(f":material/translate: {language_name(lang)}")
    render(local)
    listen(local, lang, f"{key}::{lang}", voice)
    with st.container(border=True):
        st.caption("English")
        render(english)
        listen(english, "en-IN", f"{key}::en", voice)


def translate_toggle(text: str, lang: str, key: str, purpose: str = "patient",
                     render=None) -> None:
    """'Show in <language>' for reference text: translated only when pressed."""
    if get_voice() is None or lang in ("en-IN", AUTO) or not text.strip():
        return
    render = render or st.markdown
    open_key = f"tr_open::{key}"
    is_open = st.session_state.get(open_key, False)
    if st.button(
        ui_text("Hide translation" if is_open
                else f"Show in {language_name(lang, native=False)}"),
        key=f"tr_btn::{key}", icon=":material/translate:", type="tertiary",
    ):
        is_open = not is_open
        st.session_state[open_key] = is_open
    if is_open:
        local = translated(text, lang, purpose)
        if local:
            with st.container(border=True):
                render(local)
                listen(local, lang, f"{key}::tr", purpose)


_REC_CSS = """<style>
/* While Streamlit's microphone box is recording (it shows the send and cancel
   buttons), the input glows red, and a bar above it counts down. */
[data-testid="stChatInput"]:has([data-testid="stChatInputApproveButton"]) {
  border-color: #ef4444 !important; animation: hv-recglow 1.3s ease-in-out infinite;
}
@keyframes hv-recglow {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,.45); }
  50%      { box-shadow: 0 0 0 7px rgba(239,68,68,0); }
}
#hv-rec { position: fixed; z-index: 1000001; display: none; align-items: center; gap: .7rem;
  box-sizing: border-box; padding: .5rem .9rem; border-radius: 999px; font-size: .9rem; font-weight: 600;
  color: #7f1d1d; background: rgba(254,226,226,.95); border: 1px solid #fca5a5;
  box-shadow: 0 6px 20px rgba(127,29,29,.18); backdrop-filter: blur(8px); overflow: hidden; }
#hv-rec.done { color: #14532d; background: rgba(220,252,231,.97); border-color: #86efac; }
#hv-rec .dot { width: .8rem; height: .8rem; border-radius: 50%; background: #ef4444; flex: none;
  animation: hv-dot 1s ease-in-out infinite; }
#hv-rec.done .dot { background: #22c55e; animation: none; }
@keyframes hv-dot { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.5); opacity: .55; } }
#hv-rec .eq { display: inline-flex; align-items: center; gap: 2px; height: 1.1rem; flex: none; }
#hv-rec .eq i { width: 3px; height: 100%; border-radius: 2px; background: currentColor; transform-origin: center;
  animation: hv-eq .9s ease-in-out infinite; }
#hv-rec .eq i:nth-child(2) { animation-delay: .15s; } #hv-rec .eq i:nth-child(3) { animation-delay: .3s; }
#hv-rec .eq i:nth-child(4) { animation-delay: .45s; } #hv-rec .eq i:nth-child(5) { animation-delay: .6s; }
@keyframes hv-eq { 0%, 100% { transform: scaleY(.25); } 50% { transform: scaleY(1); } }
#hv-rec.done .eq { display: none; }
@media (prefers-reduced-motion: reduce) {
  [data-testid="stChatInput"]:has([data-testid="stChatInputApproveButton"]) { animation: none; }
  #hv-rec .dot, #hv-rec .eq i { animation: none; }
}
</style>"""

# No "<" followed by a letter anywhere in the script: Streamlit's sanitiser
# drops such a script. Elements are built with createElement for that reason.
_REC_JS = """<script>
(function () {
  window.__hvRecMax = %(max)d;
  if (window.__hvRec) return;
  window.__hvRec = true;
  const el = document.createElement("div");
  el.id = "hv-rec"; el.setAttribute("role", "status"); el.setAttribute("aria-live", "polite");
  const mk = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };
  const dot = mk("span", "dot"), eq = mk("span", "eq"), txt = mk("span", "txt");
  for (let i = 0; i !== 5; i++) eq.appendChild(mk("i"));
  [dot, eq, txt].forEach(n => el.appendChild(n));
  document.body.appendChild(el);

  let t0 = null, how = null, hideAt = 0, auto = false;
  document.addEventListener("click", e => {
    if (auto) return;   // the page's own press at the limit
    const b = e.target.closest && e.target.closest("button");
    const id = b && b.getAttribute("data-testid");
    if (id === "stChatInputApproveButton") how = "sent";
    else if (id === "stChatInputCancelButton") how = "cancel";
  }, true);

  function place() {
    const ci = document.querySelector('[data-testid="stChatInput"]');
    if (!ci) { el.style.display = "none"; return false; }
    const r = ci.getBoundingClientRect();
    el.style.left = r.left + "px"; el.style.width = r.width + "px";
    el.style.bottom = (window.innerHeight - r.top + 10) + "px";
    el.style.display = "flex";
    return true;
  }

  setInterval(() => {
    const MAX = window.__hvRecMax, rec = document.querySelector('[data-testid="stChatInputApproveButton"]');
    if (rec) {
      if (t0 === null) { t0 = Date.now(); how = null; }
      const s = (Date.now() - t0) / 1000;
      el.className = "on";
      txt.textContent = "Recording… tick to send, cross to cancel";
      place();
      if (s >= MAX) { how = "limit"; auto = true; rec.click(); auto = false; }
    } else if (t0 !== null) {
      t0 = null;
      el.className = "done";
      txt.textContent = how === "cancel" ? "Recording cancelled"
        : how === "limit" ? "Recording sent (30 s maximum)"
        : "Recording sent";
      hideAt = Date.now() + 2200;
      place();
    } else if (hideAt && Date.now() > hideAt) {
      hideAt = 0; el.style.display = "none";
    }
  }, 200);
})();
</script>"""


def recording_indicator() -> None:
    """Shows when the microphone is recording, and when it has ended.

    Streamlit's own recorder shows only a dotted line with a cross and a tick.
    This adds a red glow on the input, a pulsing dot and moving bars, and a
    closing note ("sent" or "cancelled"). The time limit in config/voice.yaml
    (max_record_seconds) is kept out of sight: nothing counts down on screen,
    the page just presses the tick at the limit so the recording is sent
    instead of lost, and rag.sarvam refuses anything longer on the server.
    """
    if not voice_available():
        return
    limit = int(load_voice_config()["limits"].get("max_record_seconds") or 30)
    st.html(_REC_CSS)
    st.html(_REC_JS % {"max": limit}, unsafe_allow_javascript=True)


def language_banner(codes: list[str]) -> None:
    """A line that cycles through "Choose your language" in every language,
    beside the language picker. Pure CSS, so it costs no reruns."""
    lines = load_voice_config().get("choose_language", {})
    items = [lines[c] for c in codes if c in lines]
    step, n = 3, len(items)
    shown = 100 / n
    spans = "".join(
        f'<span style="animation-delay:{i * step}s">{t}</span>' for i, t in enumerate(items)
    )
    st.html(f"""<style>
    .hv-langline {{ position: relative; display: block; height: 2.4rem;
      overflow: hidden; font-size: 1.15rem; font-weight: 600; color: #0d9488; }}
    .hv-langline span {{ position: absolute; left: 0; top: .45rem; white-space: nowrap;
      opacity: 0; animation: hv-lang {step * n}s linear infinite; }}
    @keyframes hv-lang {{
      0% {{ opacity: 0; transform: translateY(.6rem); }}
      {shown * .12:.2f}% {{ opacity: 1; transform: none; }}
      {shown * .88:.2f}% {{ opacity: 1; transform: none; }}
      {shown:.2f}%, 100% {{ opacity: 0; transform: translateY(-.6rem); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .hv-langline span {{ animation: none; }}
      .hv-langline span:first-child {{ opacity: 1; }}
    }}
    </style>
    <div class="hv-langline" role="note" aria-label="{items[0] if items else ''}">{spans}</div>""")
