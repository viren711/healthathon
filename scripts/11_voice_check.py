#!/usr/bin/env python3
"""The one live check of voice and languages -- spends real Sarvam credits, a few rupees.

Everything else (scripts/smoke_test.py, scripts/ui_test.py) runs against the
offline mock. This runs the real round trip once, with inputs kept short:

  1. read a short Hindi question aloud           text to speech, ~35 characters
  2. transcribe that audio, language "unknown"   speech to text, ~3 seconds
     -> must come back as Hindi, close to the original
  3. translate it to English, answer it with the live patient pipeline
     (needs the model server), and translate the first sentences of the
     answer back to Hindi -> every citation marker must survive
  4. a Tamil question: detected by its script, no language-ID call
  5. a Hinglish clinician question -> English

Every step is cached in artifacts/voice_cache, so running this again costs
nothing. The rupees spent are printed at the end, from the usage log.

    python scripts/11_voice_check.py
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import sarvam  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'  ok  ' if ok else ' FAIL '}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    v = sarvam.SarvamVoice.from_env()
    if v is None or v.mock:
        print("Needs SARVAM_API_KEY in .env (and SARVAM_MOCK unset).")
        return 1
    before = v.spent_today()

    question_hi = "कीमोथेरेपी के दुष्प्रभाव क्या हैं?"
    audio, mime = v.speak(question_hi, "hi-IN")
    check("Hindi text is read aloud", len(audio) > 2000, f"{len(audio):,} bytes {mime}")

    heard, lang = v.transcribe(audio, None, mime=mime)
    similarity = difflib.SequenceMatcher(None, heard, question_hi).ratio()
    check("the recording is heard as Hindi", lang == "hi-IN", f"{lang}: {heard!r}")
    check("the transcript matches what was said", similarity > 0.7, f"{similarity:.2f}")

    english = v.translate(heard, lang or "hi-IN", "en-IN", "to_english")
    check("the question is translated to English",
          sarvam.dominant_script(english) == "latn", repr(english))

    from rag.llm import LLMClient
    from rag.pipeline import answer_question

    client = LLMClient.from_config()
    ok, detail = client.health()
    if ok:
        ans = answer_question(english, client=client, corpus="patient")
        check("the English question is answered from the guides", ans.status == "answered",
              f"{ans.status}: {ans.text[:80]!r}")
        if ans.status == "answered":
            first = " ".join(sarvam._SENT_END.split(ans.text.split("\n")[0])[:3])
            local = v.translate(first, "en-IN", "hi-IN", "patient")
            same = sorted(sarvam.MARKER.findall(local)) == sorted(sarvam.MARKER.findall(first))
            check("the Hindi answer keeps every citation", same, local)
            print(f"\n  EN: {first}\n  HI: {local}\n")
    else:
        print(f"(skipping the pipeline step: model server unreachable -- {detail})")

    tamil = "கீமோதெரபியின் பக்க விளைவுகள் என்ன?"
    lid_spend = v.spent_today()
    check("a Tamil question is detected by its script",
          v.detect(tamil, ["en-IN", "ta-IN", "hi-IN"]) == "ta-IN"
          and v.spent_today() == lid_spend, "no API call")
    check("the Tamil question is translated to English",
          "side effect" in v.translate(tamil, "ta-IN", "en-IN", "to_english").lower())

    hinglish = "cisplatin ka renal dose adjustment kya hai?"
    lang = v.detect(hinglish, ["en-IN", "hi-IN"])
    out = v.translate(hinglish, lang, "en-IN", "to_english") if lang != "en-IN" else hinglish
    check("a Hinglish clinician question reaches English", "cisplatin" in out.lower(),
          f"{lang}: {out!r}")

    spent = v.spent_today() - before
    print(f"\nSarvam spend for this run: about ₹{spent:.2f} "
          f"(today so far ₹{v.spent_today():.2f}; a rerun costs ₹0 from the cache)")
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("all live voice checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
