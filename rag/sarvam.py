"""Voice and language services for the two chat pages, on Sarvam AI.

    speech -> text      transcribe()   saaras:v4; "unknown" detects the language
    text -> language    detect()       Unicode script first (free), the
                                       language-ID API only when the script
                                       cannot decide (Devanagari, Latin)
    text -> text        translate()    sarvam-translate:v1 / mayura:v1, with
                                       citation markers [n] kept on the
                                       sentence they belong to
    text -> speech      speak()        bulbul:v4-flash, MP3

The retrieval pipeline itself stays English: a question is translated to
English, answered and checked exactly as before, and only the checked answer
is translated back. Nothing here can weaken the citation guarantee.

Credits: every result is cached on disk (artifacts/voice_cache), so a repeated
question, answer or replay costs nothing; each billable call is logged with
its estimated cost to logs/sarvam_usage.jsonl; and a daily budget in
config/voice.yaml refuses further calls once reached. SARVAM_MOCK=1 swaps in
a deterministic fake that never touches the network, for the tests.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sqlite3
import threading
import time
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from rag import keyring
from rag.config import ROOT, _load_dotenv


class SarvamError(RuntimeError):
    """A Sarvam call failed. The message is written to be shown to the user."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=1)
def load_voice_config() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "config" / "voice.yaml").read_text(encoding="utf-8"))


def language_name(code: str | None, native: bool = True) -> str:
    names = load_voice_config()["languages"].get(code or "")
    if not names:
        return code or "unknown"
    return names[1] if native else names[0]


# ---------------------------------------------------------------------------
# Script detection -- free, and enough for 9 of the 11 languages
# ---------------------------------------------------------------------------

# Unicode block -> language. Devanagari is shared by Hindi and Marathi, and
# Latin by English and romanised Indian languages, so those two return a
# script tag and need the language-ID API to decide.
_BLOCKS = [
    (0x0900, 0x097F, "deva"),
    (0x0980, 0x09FF, "bn-IN"),
    (0x0A00, 0x0A7F, "pa-IN"),
    (0x0A80, 0x0AFF, "gu-IN"),
    (0x0B00, 0x0B7F, "od-IN"),
    (0x0B80, 0x0BFF, "ta-IN"),
    (0x0C00, 0x0C7F, "te-IN"),
    (0x0C80, 0x0CFF, "kn-IN"),
    (0x0D00, 0x0D7F, "ml-IN"),
]


def dominant_script(text: str) -> str | None:
    """Language code, "deva", "latn", or None when the text has no letters."""
    counts: Counter[str] = Counter()
    for ch in text:
        cp = ord(ch)
        for lo, hi, tag in _BLOCKS:
            if lo <= cp <= hi:
                counts[tag] += 1
                break
        else:
            if ch.isascii() and ch.isalpha():
                counts["latn"] += 1
    if not counts:
        return None
    indic = {k: v for k, v in counts.items() if k != "latn"}
    # Any real run of Indic script wins: a Hindi sentence with "CT scan" or
    # "chemo" in it is still Hindi.
    if indic and max(indic.values()) >= 3:
        return max(indic, key=indic.get)
    return "latn" if counts["latn"] else max(counts, key=counts.get)


def daily_budget(cfg: dict, day: date | None = None) -> float:
    """The daily cap in force on `day`: daily_budget_inr, replaced by the
    latest daily_budget_from entry whose date has been reached."""
    limits = cfg["limits"]
    budget = float(limits.get("daily_budget_inr") or 0)
    today = (day or date.today()).isoformat()
    for start, value in sorted((limits.get("daily_budget_from") or {}).items(),
                               key=lambda kv: str(kv[0])):
        if str(start) <= today:
            budget = float(value or 0)
    return budget


_API_ROWS = {
    "stt": ("Speech to text", "saaras:v4", "sec"),
    "translate": ("Translation", "sarvam-translate:v1 / mayura:v1", "chr"),
    "tts": ("Text to speech", "bulbul", "chr"),
    "lid": ("Language identification", "text-lid", "chr"),
}


def write_usage_report(usage_log: Path, report: Path, cfg: dict | None = None) -> str:
    """Rewrite the human-readable usage report from the usage log: total spend,
    spend per API, then every call. Rewritten after each billable call."""
    cfg = cfg or load_voice_config()
    rows = []
    if usage_log.exists():
        for line in usage_log.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    today = date.today().isoformat()
    total = sum(r.get("inr", 0) for r in rows)
    spent_today = sum(r.get("inr", 0) for r in rows if r.get("date") == today)
    by: dict[str, list[float]] = {}
    for r in rows:
        b = by.setdefault(r["kind"], [0, 0.0, 0.0, 0, 0.0])  # calls, units, rupees, hits, hit units
        units = r.get("chars", r.get("seconds", 0))
        if r.get("cached"):
            b[3] += 1
            b[4] += units
            continue
        b[0] += 1
        b[1] += units
        b[2] += r.get("inr", 0)
    n_calls = sum(int(b[0]) for b in by.values())
    n_hits = sum(int(b[3]) for b in by.values())
    out = ["SARVAM API USAGE", "=" * 16,
           f"Updated {today} {time.strftime('%H:%M:%S')} -- rewritten after every Sarvam call",
           "Costs are estimates from the published prices, in rupees (Rs).",
           "Sarvam has no balance endpoint, so 'credits' = rupees spent;",
           "see the Sarvam dashboard for what is left.", "",
           f"TOTAL CREDITS USED:   Rs {total:.2f}   ({n_calls} paid calls; {n_hits} repeats served free from the cache)",
           f"Used today:           Rs {spent_today:.2f} of the Rs {daily_budget(cfg):.0f} daily cap", "",
           "USED PER API", "-" * 12,
           f"{'API':<26}{'model':<34}{'calls':>6}{'amount':>12}{'Rs':>10}{'  free hits':>12}"]
    for kind, (name, model, unit) in _API_ROWS.items():
        if kind in by:
            c, u, inr, h, _hu = by[kind]
            out.append(f"{name:<26}{model:<34}{int(c):>6}{u:>9.0f} {unit:<3}{inr:>10.2f}{int(h):>12}")
    out += ["", "ALL TRANSACTIONS (newest first)", "-" * 31,
            f"{'date':<11}{'time':<10}{'api':<11}{'Rs':>8}  detail  (cached = free, no call made)"]
    for r in reversed(rows):
        detail = ", ".join(f"{k}={v!r}" if k == "text" else f"{k}={v}" for k, v in r.items()
                           if k not in ("date", "time", "kind", "inr") and v is not None)
        out.append(f"{r['date']:<11}{r['time']:<10}{r['kind']:<11}{r['inr']:>8.3f}  {detail}")
    text = "\n".join(out) + "\n"
    report.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _peek(text: str, n: int = 50) -> str:
    """The start of a text on one line, for the usage log."""
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


MARKER = re.compile(r"\[\d+\]")
_SENT_END = re.compile(r"(?<=[.!?।॥])\s+")
_LINE_PREFIX = re.compile(r"^(\s*(?:#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)?)(.*)$", re.S)


def _markers(text: str) -> Counter[str]:
    return Counter(MARKER.findall(text))


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT_END.split(text.strip()) if s.strip()]


def _with_marks(sentence: str, marks: str) -> str:
    """A sentence with `marks` placed before its closing punctuation."""
    sentence = re.sub(r"\s+([.,;:!?।॥])", r"\1", sentence.strip())
    if not marks:
        return sentence
    body, punct = re.match(r"^(.*?)([.!?।॥]*)$", sentence, re.S).groups()
    return f"{body.rstrip()} {marks}{punct}"


def attach_markers(text: str) -> str:
    """'...machinery. [1] [2] The DNA...' -> '...machinery [1][2]. The DNA...'

    The 3B model sometimes writes its markers after the full stop. In
    English that still cites the sentence before; split into sentences, it
    would open the next one. Normalised before anything is translated.
    """
    return re.sub(
        r"([.!?])((?:\s*\[\d+\])+)",
        lambda m: " " + "".join(MARKER.findall(m.group(2))) + m.group(1),
        text,
    )


def align_markers(source: str, translation: str) -> str | None:
    """Give each translated sentence the citation markers of its English original.

    Translations do not keep markers where they belong. mayura:v1 writes
    "S1। [4] S2। [4]" for "S1 [4]. S2 [4]." -- markers after the full stop,
    reading as the NEXT sentence's -- and for another answer put a marker that
    really did belong to the next sentence in exactly that position. Position
    cannot tell the two apart; sentence order can. So the markers are
    stripped, and when both sides have the same number of sentences, sentence
    i gets English sentence i's markers. None when the counts differ.
    """
    src = _sentences(attach_markers(source))
    out = [s for s in _sentences(MARKER.sub("", translation)) if s.strip(" .।॥")]
    if len(src) != len(out):
        return None
    return " ".join(_with_marks(o, "".join(MARKER.findall(s))) for s, o in zip(src, out))


def strip_for_speech(text: str) -> str:
    """Markdown answer -> what should be read aloud: no [n], no ** or #."""
    lines = []
    for line in text.split("\n"):
        line = MARKER.sub("", line)
        line = re.sub(r"[*_`#>]+", "", line)
        line = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", line)
        line = re.sub(r"\s+([.,;:!?।])", r"\1", line).strip()
        if line:
            # A list item has no full stop; without one the voice runs the
            # items together into one breathless sentence.
            lines.append(line if line[-1] in ".!?।॥:;" else line + ".")
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def chunk_text(text: str, size: int) -> list[str]:
    """Split at sentence ends into pieces of at most `size` characters."""
    parts: list[str] = []
    cur = ""
    for sentence in re.split(r"(?<=[.!?।॥])\s+", text):
        while len(sentence) > size:  # one enormous "sentence": hard split
            parts += [cur] if cur else []
            cur = ""
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            parts.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if cur and len(cur) + len(sentence) + 1 > size:
            parts.append(cur)
            cur = ""
        cur = f"{cur} {sentence}".strip()
    if cur:
        parts.append(cur)
    return parts


def wav_seconds(audio: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        return len(audio) / 32000  # 16 kHz, 16-bit mono


def audio_seconds(audio: bytes) -> float:
    """Length of a recording: from the WAV header, else estimated from the size
    (16 kHz, 16-bit mono, which is what the browser microphone sends)."""
    try:
        with wave.open(io.BytesIO(audio)) as w:
            return w.getnframes() / float(w.getframerate())
    except (wave.Error, EOFError):
        return len(audio) / 32000.0


def _key(*parts: Any) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cache and usage log
# ---------------------------------------------------------------------------


class _Store:
    """Translations and transcripts in SQLite, audio as files, usage as JSONL."""

    def __init__(self, cache_dir: Path, usage_log: Path, report: Path | None = None) -> None:
        self.report = report
        self.cache_dir = cache_dir
        self.tts_dir = cache_dir / "tts"
        self.tts_dir.mkdir(parents=True, exist_ok=True)
        self.usage_log = usage_log
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(cache_dir / "cache.db", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()
        self._today: tuple[str, float] | None = None

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, value))
            self._db.commit()

    def audio(self, key: str) -> bytes | None:
        path = self.tts_dir / f"{key}.mp3"
        return path.read_bytes() if path.exists() else None

    def put_audio(self, key: str, data: bytes) -> None:
        (self.tts_dir / f"{key}.mp3").write_bytes(data)

    def spent_today(self) -> float:
        today = date.today().isoformat()
        with self._lock:
            if self._today and self._today[0] == today:
                return self._today[1]
            total = 0.0
            if self.usage_log.exists():
                for line in self.usage_log.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("date") == today:
                        total += rec.get("inr", 0.0)
            self._today = (today, total)
            return total

    def record(self, kind: str, inr: float, **extra: Any) -> None:
        today = date.today().isoformat()
        rec = {"date": today, "time": time.strftime("%H:%M:%S"), "kind": kind,
               "inr": round(inr, 4), **extra}
        spent = self.spent_today()
        with self._lock:
            with self.usage_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._today = (today, spent + inr)
            if self.report:
                try:
                    write_usage_report(self.usage_log, self.report)
                except OSError:
                    pass  # the report is a convenience; never fail a call over it


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


def _error_message(status: int | None) -> str:
    if status == 402:
        return ("Every Sarvam key is out of credit. Add credits on the Sarvam dashboard "
                "(or delete artifacts/key_state.json to try the keys again).")
    if status == 429:
        return "Too many voice or translation requests just now. Please wait a moment and try again."
    if status in (401, 403):
        return "The Sarvam API key was rejected. Check SARVAM_API_KEY in .env."
    return "The voice and translation service could not be reached. Please try again."


class _MockAPI:
    """Stands in for the SDK under SARVAM_MOCK=1: deterministic, offline."""

    TRANSCRIPT = "कीमोथेरेपी के दुष्प्रभाव क्या हैं?"
    # Questions the tests type, with the English a real translation gives,
    # so the English-only safety screen and topic gate see real English.
    PHRASEBOOK = {
        TRANSCRIPT: "What are the side effects of chemotherapy?",
        "मैं मरना चाहता हूँ, अब सहा नहीं जाता": "I want to die, I can't bear it anymore",
        "1998 கால்பந்து உலகக் கோப்பையை வென்றது யார்?": "Who won the football World Cup in 1998?",
        "क्रिकेट विश्व कप किसने जीता?": "Who won the cricket World Cup?",
    }

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def transcribe(self, audio: bytes, lang: str, mode: str,
                   mime: str = "audio/wav") -> tuple[str, str]:
        self.calls["stt"] += 1
        return self.TRANSCRIPT, (lang if lang != "unknown" else "hi-IN")

    def identify(self, text: str) -> str:
        self.calls["lid"] += 1
        return "hi-IN" if dominant_script(text) == "deva" else "en-IN"

    def translate(self, text: str, src: str, tgt: str, model: str, mode: str | None) -> str:
        self.calls["translate"] += 1
        if tgt == "en-IN" and text.strip() in self.PHRASEBOOK:
            return self.PHRASEBOOK[text.strip()]
        # One tag per line, so line structure and markers survive exactly as
        # they would in a faithful translation.
        return "\n".join(f"[{tgt[:2]}] {line}" if line.strip() else line for line in text.split("\n"))

    def speak(self, text: str, lang: str, model: str, speaker: str | None) -> bytes:
        self.calls["tts"] += 1
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(b"\x00\x00" * 800)
        return buf.getvalue()


class _NoKeys(Exception):
    status_code = 402


class _LiveAPI:
    """The SDK, with several keys: each call uses the first key that still has
    credit. A key refused as out of credit (402, 401/403, or a 429 that names
    credit or quota) is remembered and skipped from then on."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self._tl = threading.local()

    @property
    def key_label(self) -> str | None:
        return getattr(self._tl, "label", None)

    def _run(self, fn: Callable[[Any], Any]) -> Any:
        from sarvamai import SarvamAI

        last: Exception | None = None
        for name, key in keyring.usable("sarvam"):
            client = self._clients.get(key)
            if client is None:
                client = self._clients[key] = SarvamAI(api_subscription_key=key)
            try:
                result = fn(client)
                self._tl.label = name
                return result
            except Exception as exc:  # noqa: BLE001 - the SDK raises several types
                status = getattr(exc, "status_code", None)
                if keyring.exhausted("sarvam", status, getattr(exc, "body", exc)):
                    keyring.mark_exhausted("sarvam", name, key, status)
                    last = exc
                    continue
                raise
        if last is not None:
            raise last
        raise _NoKeys("every Sarvam key is out of credit")

    def transcribe(self, audio: bytes, lang: str, mode: str,
                   mime: str = "audio/wav") -> tuple[str, str | None]:
        cfg = load_voice_config()
        name = "voice.mp3" if mime == "audio/mpeg" else "voice.wav"
        r = self._run(lambda c: c.speech_to_text.transcribe(
            file=(name, audio, mime), model=cfg["models"]["stt"],
            mode=mode, language_code=lang,
        ))
        return (r.transcript or "").strip(), r.language_code

    def identify(self, text: str) -> str | None:
        return self._run(lambda c: c.text.identify_language(input=text)).language_code

    def translate(self, text: str, src: str, tgt: str, model: str, mode: str | None) -> str:
        kw: dict[str, Any] = dict(input=text, source_language_code=src,
                                  target_language_code=tgt, model=model)
        if mode:
            kw["mode"] = mode
        return self._run(lambda c: c.text.translate(**kw)).translated_text

    def speak(self, text: str, lang: str, model: str, speaker: str | None) -> bytes:
        kw: dict[str, Any] = dict(text=text, language_code=lang, model=model,
                                  output_audio_codec="mp3")
        if speaker:
            kw["speaker"] = speaker
        r = self._run(lambda c: c.text_to_speech.convert(**kw))
        return b"".join(base64.b64decode(a) for a in r.audios)


class SarvamVoice:
    def __init__(self, api: Any, store: _Store, mock: bool = False) -> None:
        self.api = api
        self.store = store
        self.mock = mock
        self.cfg = load_voice_config()
        self._flights: dict[str, threading.Lock] = {}
        self._flights_lock = threading.Lock()

    def _flight(self, key: str) -> threading.Lock:
        """One lock per cache key: two identical requests arriving together
        (a double rerun, two tabs) wait for each other, and the second reads
        the first's cached result instead of paying again."""
        with self._flights_lock:
            return self._flights.setdefault(key, threading.Lock())

    def _hit(self, kind: str, **log: Any) -> None:
        """A cache hit: free, but logged so the report shows what was saved."""
        self.store.record(kind, 0.0, cached=True, **log)

    @classmethod
    def from_env(cls) -> "SarvamVoice | None":
        _load_dotenv()
        cfg = load_voice_config()
        cache_dir = Path(os.environ.get("VOICE_CACHE_DIR") or ROOT / cfg["paths"]["cache_dir"])
        usage_log = Path(os.environ.get("VOICE_USAGE_LOG") or ROOT / cfg["paths"]["usage_log"])
        # Beside the log when the log is redirected (tests), so a test never
        # touches the real report.
        report = (usage_log.with_name("sarvam_api_usage.txt") if os.environ.get("VOICE_USAGE_LOG")
                  else ROOT / cfg["paths"]["usage_report"])
        if os.environ.get("SARVAM_MOCK", "").lower() in ("1", "true", "yes"):
            return cls(_MockAPI(), _Store(cache_dir, usage_log, report), mock=True)
        if not keyring.configured("sarvam"):
            return None
        return cls(_LiveAPI(), _Store(cache_dir, usage_log, report))

    # -- plumbing -----------------------------------------------------------

    def spent_today(self) -> float:
        return self.store.spent_today()

    def _call(self, kind: str, inr: float, fn: Callable[[], Any],
              after: Callable[[Any], dict] | None = None, **log: Any) -> Any:
        """One billable request: budget check, 429 backoff, readable errors, log."""
        budget = daily_budget(self.cfg)
        if budget and self.store.spent_today() + inr > budget:
            raise SarvamError(
                f"Today's voice and translation budget (₹{budget:.0f}, set in "
                "config/voice.yaml) is used up. Cached answers and English still work."
            )
        for attempt in range(4):
            try:
                result = fn()
                break
            except SarvamError:
                raise
            except Exception as exc:  # noqa: BLE001 - the SDK raises several types
                status = getattr(exc, "status_code", None)
                if status == 429 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise SarvamError(_error_message(status), status) from exc
        if after:
            log.update(after(result))  # details only known from the reply
        label = getattr(self.api, "key_label", None)
        if label:
            log["key"] = label
        self.store.record(kind, inr, **log)
        return result

    # -- speech to text -----------------------------------------------------

    def transcribe(self, audio: bytes, lang: str | None = None,
                   mode: str = "transcribe", mime: str = "audio/wav") -> tuple[str, str | None]:
        """Recording -> (text, language code). lang=None lets Sarvam detect it.

        The browser's microphone gives 16 kHz WAV; MP3 is accepted for tests.
        """
        if len(audio) < self.cfg["limits"]["min_audio_bytes"]:
            raise SarvamError("The recording was too short. Please hold the microphone a little longer.")
        limit = self.cfg["limits"].get("max_record_seconds")
        if limit and mime == "audio/wav" and audio_seconds(audio) > limit + 1.5:
            # Refused before any request is made, so a long recording costs nothing.
            raise SarvamError(f"That recording is about {audio_seconds(audio):.0f} seconds; "
                              f"the limit is {limit}. Please record a shorter question.")
        code = lang or "unknown"
        key = _key("stt", self.cfg["models"]["stt"], code, mode, hashlib.sha1(audio).hexdigest())
        with self._flight(key):
            return self._transcribe(audio, code, mode, mime, key)

    def _transcribe(self, audio: bytes, code: str, mode: str, mime: str,
                    key: str) -> tuple[str, str | None]:
        cached = self.store.get(key)
        if cached:
            text, heard = json.loads(cached)
            self._hit("stt", detail="same recording heard before", requested=code,
                      detected=heard, heard=_peek(text))
            return text, heard
        # bulbul's MP3 is ~64 kbit/s; only the cost estimate depends on this.
        secs = wav_seconds(audio) if mime == "audio/wav" else len(audio) / 8000
        inr = secs / 60 * self.cfg["prices_inr"]["stt_per_minute"]
        text, heard = self._call("stt", inr, lambda: self.api.transcribe(audio, code, mode, mime),
                                 after=lambda r: {"detected": r[1], "heard": _peek(r[0])},
                                 seconds=round(secs, 1), bytes=len(audio),
                                 model=self.cfg["models"]["stt"], mode=mode, requested=code)
        if not text:
            raise SarvamError("I could not hear anything in that recording. Please try again.")
        self.store.put(key, json.dumps([text, heard], ensure_ascii=False))
        return text, heard

    # -- language -------------------------------------------------------------

    def detect(self, text: str, allowed: list[str]) -> str:
        """Language of typed text, restricted to `allowed`; English if unsure."""
        script = dominant_script(text)
        if script not in ("deva", "latn"):
            return script if script in allowed else "en-IN"
        candidates = {"deva": {"hi-IN", "mr-IN"}, "latn": {"en-IN", "hi-IN"}}[script]
        open_ = [c for c in allowed if c in candidates]
        if len(open_) <= 1:  # nothing to decide: no call
            return open_[0] if open_ else "en-IN"
        sample = text[:300]
        key = _key("lid", sample)
        lang = self.store.get(key)
        if lang is None:
            inr = len(sample) / 1000 * self.cfg["prices_inr"]["lid_per_1k_chars"]
            lang = self._call("lid", inr, lambda: self.api.identify(sample),
                              after=lambda r: {"detected": r}, chars=len(sample),
                              text=_peek(sample)) or ""
            self.store.put(key, lang)
        return lang if lang in allowed else ("hi-IN" if script == "deva" and "hi-IN" in allowed else "en-IN")

    # -- translation ----------------------------------------------------------

    def _translate_raw(self, text: str, src: str, tgt: str, model: str, mode: str | None,
                       purpose: str = "", step: str = "whole line") -> str:
        key = _key("tr", model, mode, src, tgt, text)
        with self._flight(key):
            cached = self.store.get(key)
            if cached is not None:
                self._hit("translate", chars=len(text), purpose=purpose, source=src,
                          target=tgt, text=_peek(text))
                return cached
            inr = len(text) / 1000 * self.cfg["prices_inr"]["translate_per_1k_chars"]
            out = self._call("translate", inr,
                             lambda: self.api.translate(text, src, tgt, model, mode),
                             chars=len(text), model=model, mode=mode, purpose=purpose,
                             step=step, source=src, target=tgt, text=_peek(text))
            out = (out or "").strip()
            self.store.put(key, out)
            return out

    def _translate_line(self, line: str, src: str, tgt: str, model: str, mode: str | None,
                        purpose: str = "") -> str:
        """One line, prefix ("- ", "1. ", "## ") kept, markers on the right sentence."""
        prefix, body = _LINE_PREFIX.match(line).groups()
        if not body.strip():
            return line
        bold = body.startswith("**") and body.endswith("**") and len(body) > 4
        if bold:
            body = body[2:-2]
        if MARKER.search(body):
            body = attach_markers(body)
        raw = self._translate_raw(body, src, tgt, model, mode, purpose)
        out = raw if not MARKER.search(body) else align_markers(body, raw)
        if out is None and _markers(raw) == _markers(body) and not any(
                MARKER.match(s) for s in _sentences(raw)):
            # Sentence counts differ (a merged or split sentence) but every
            # marker survived and none opens a sentence: keep it as written.
            out = raw
        if out is None:
            # Translate sentence by sentence with the markers held back, then
            # put each sentence's markers after it. The line was already
            # billed once above, so this bills its characters a SECOND time --
            # only reached when the cheap paths fail, and logged as
            # "sentence retry" so the report shows when it happens.
            pieces = []
            for sent in _SENT_END.split(body):
                marks = "".join(MARKER.findall(sent))
                bare = MARKER.sub("", sent).strip()
                if not bare:
                    pieces.append(marks)
                    continue
                t = MARKER.sub("", self._translate_raw(
                    bare, src, tgt, model, mode, purpose, "sentence retry")).strip()
                if marks:
                    m = re.match(r"^(.*?)([.!?।॥]*)$", t, re.S)
                    t = f"{m.group(1).rstrip()} {marks}{m.group(2)}"
                pieces.append(t)
            out = " ".join(p for p in pieces if p)
        if bold:
            out = f"**{out}**"
        return prefix + out

    def translate(self, text: str, src: str, tgt: str, purpose: str = "patient") -> str:
        """Translate Markdown text line by line, keeping its layout and citations.

        purpose: "patient" (plain native script), "clinician" (code-mixed,
        medical terms kept in English) or "to_english" (questions).
        """
        if not text.strip() or src == tgt:
            return text
        tcfg = self.cfg["translate"][purpose]
        model, mode = tcfg["model"], tcfg.get("mode")
        limit = self.cfg["limits"]["translate_chunk_chars"]
        lines = text.split("\n")
        # Long lines are split at sentence ends so each request fits the API.
        work: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            if len(line) <= limit:
                work.append((i, line))
            else:
                prefix, body = _LINE_PREFIX.match(line).groups()
                for j, piece in enumerate(chunk_text(body, limit)):
                    work.append((i, (prefix if j == 0 else "") + piece))

        def run(item: tuple[int, str]) -> str:
            return self._translate_line(item[1], src, tgt, model, mode, purpose)

        with ThreadPoolExecutor(max_workers=self.cfg["limits"]["parallel_requests"]) as pool:
            done = list(pool.map(run, work))
        merged: dict[int, list[str]] = {}
        for (i, _), out in zip(work, done):
            merged.setdefault(i, []).append(out)
        return "\n".join(" ".join(merged[i]) if i in merged else lines[i] for i in range(len(lines)))

    # -- text to speech -------------------------------------------------------

    def speak(self, text: str, lang: str, voice: str = "patient") -> tuple[bytes, str]:
        """Text -> (audio bytes, mime type), read in `lang` in the `voice`
        ("patient" or "clinician") configured in tts_speakers. Cached on disk."""
        clean = strip_for_speech(text)[: self.cfg["limits"]["tts_max_chars"]]
        if not clean:
            raise SarvamError("There is nothing to read aloud.")
        mime = "audio/wav" if self.mock else "audio/mpeg"
        model, speaker = self.cfg["models"]["tts"], self.cfg["models"]["tts_speakers"][voice]
        key = _key("tts", model, speaker, lang, clean)
        with self._flight(key):
            return self._speak(clean, lang, mime, model, speaker, key, voice)

    def _speak(self, clean: str, lang: str, mime: str, model: str, speaker: str | None,
               key: str, voice: str) -> tuple[bytes, str]:
        cached = self.store.audio(key)
        if cached:
            self._hit("tts", chars=len(clean), model=model, speaker=speaker or "default",
                      lang=lang, text=_peek(clean))
            return cached, mime
        pieces = chunk_text(clean, self.cfg["limits"]["tts_chunk_chars"])
        price = self.cfg["prices_inr"]["tts_per_1k_chars"]

        def say(piece: str, model: str, speaker: str | None) -> bytes:
            inr = len(piece) / 1000 * price.get(model, 3.0)
            part = f" part {pieces.index(piece) + 1}/{len(pieces)}" if len(pieces) > 1 else ""
            return self._call("tts", inr, lambda: self.api.speak(piece, lang, model, speaker),
                              chars=len(piece), model=model, speaker=speaker or "default",
                              lang=lang, voice=voice, step="Listen button" + part, text=_peek(piece))

        try:
            audio = [say(pieces[0], model, speaker)]
        except SarvamError as exc:
            if exc.status != 400:
                raise
            model = self.cfg["models"]["tts_fallback_model"]
            speaker = self.cfg["models"]["tts_fallback_speaker"]
            audio = [say(pieces[0], model, speaker)]
        audio += [say(p, model, speaker) for p in pieces[1:]]
        data = b"".join(audio)  # MP3 frames concatenate cleanly
        self.store.put_audio(key, data)
        return data, mime


@lru_cache(maxsize=1)
def get_voice() -> SarvamVoice | None:
    """The process-wide client, or None when no key is configured."""
    return SarvamVoice.from_env()
