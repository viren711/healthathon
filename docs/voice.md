# Voice and languages

Both chat tools take a question by **voice or typing, in Indian languages**,
and answer in that language **and** in English:

| | languages | translation style |
|---|---|---|
| **Patient Portal** | English (the default), one of 10 more — Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu — or Auto-detect | plain native script (`sarvam-translate:v1`) |
| **Knowledge Assistant** | English or Hindi | code-mixed Hindi — drug names, doses and staging stay in English (`mayura:v1`, `code-mixed`) |

The 11 are the languages Sarvam can both transcribe **and** speak in their
own voice, so every answer can also be read aloud.

**Both pages start in English, and typed English never calls Sarvam.** A
call is made only when the user picks another language (or Auto-detect),
records a question, or presses Listen or "Show in …". Text typed in another
script while English is selected is not sent anywhere: the page says which
language it looks like and offers an **Ask in हिन्दी** (or தமிழ், …) button
that switches the page and asks it.

**The Patient Portal's own text follows the language.** With English selected,
every button, label and caption is English only. With another language
selected, each one is English with that language beneath it (or after a " · "
in single-line widgets such as selectors and tabs): the title, intro,
disclaimer, Topic area and Guide, Ask / Learn, the suggested questions, the
chat box prompt, Listen, "Show in …", "Where this comes from", the Learn tabs
and the sidebar button. The page's ~40 fixed strings are translated in one
parallel batch the first time a language is chosen (about 5 s, about ₹2) and
saved, so the next time costs nothing. Guide titles and the NCCN text stay
English, with "Show in …" for a translation. The Knowledge Assistant's own
text stays English.

At the top of the Patient Portal, "Choose your language" cycles through all
11 languages, one every 3 s, beside the picker. The lines are fixed text in
`config/voice.yaml` (`choose_language`), animated in CSS, so they cost
nothing and trigger no reruns; with reduced motion set, only the English
line shows.

All speech and translation run on [Sarvam AI](https://docs.sarvam.ai). The
key is `SARVAM_API_KEY` in `.env`. Without it, both pages are exactly what
they were: typed English in, English out, no microphone.

---

## How a question flows

```
  microphone (16 kHz WAV, st.chat_input accept_audio)      typed text
          │                                                    │
          ▼                                                    ▼
  saaras:v4 speech-to-text                    script detection (free)
  language "unknown" → detected                ├ Tamil, Telugu, Kannada, Malayalam,
  (Knowledge Assistant: codemix,               │ Gujarati, Gurmukhi, Odia, Bengali → decided
   forced to en / hi)                          └ Devanagari, Latin → language-ID API
          │                                                    │
          └───────────────► question in language L ◄───────────┘
                                     │
                      translate L → English (sarvam-translate:v1)
                                     │
            ┌────────────────────────▼─────────────────────────┐
            │  the unchanged English pipeline: safety screen,  │
            │  retrieval, three refusal layers, citation check │
            └────────────────────────┬─────────────────────────┘
                                     │  checked English answer
                      translate English → L, citations kept
                                     │
             the answer in L, then the English beneath it, each with Listen
                                     │  (only when pressed)
                              bulbul:v4-flash → MP3
```

Two properties follow from translating at the edges only:

- **The citation guarantee is untouched.** Only an answer that has already
  passed the citation check is translated; nothing a patient reads in Tamil
  skipped a check the English went through.
- **The safety screen works in every language.** It reads the English
  translation, so "मैं मरना चाहता हूँ" is caught exactly like "I want to die",
  and the crisis notice is shown in the patient's language with the English
  beneath it (for a carer or nurse).

The user's own words stay on screen, with a caption showing the English that
was searched ("🎤 heard in हिन्दी · searched in English as: *When does hair
loss stop after chemo?*"), so a mis-hearing or mistranslation is visible.

---

## What else is translated

| | when | cost control |
|---|---|---|
| the answer | after the citation check | cached |
| safety notices, "not available", "out of scope" | when shown | fixed strings: translated once per language, for every user |
| suggested questions (Patient Portal) | when a language is chosen | 6 strings, translated once per language |
| Learn-mode search boxes ("Search all the guides", "Look up a word") | words in another script are translated to English once (then cached), so the English guides can be searched; the page shows "Searching for: …". Latin text costs nothing | ~Rs0.01–0.05 per new phrase |
| source passages, Learn-mode questions | **only on "Show in …"** | nothing is translated until asked |
| read aloud | **only on "Listen"** | cached audio; replays are free |

NCCN's own text in the source cards stays in English as written, with a
translation one click away. The interface itself (labels, buttons) stays in
English.

---

## Keeping citations attached through translation

The hardest part was not translation but keeping each `[n]` on the sentence
it cites. Three behaviours were measured on real answers (2026-11-03):

1. `mayura:v1` put a marker **after** the full stop, where it reads as the
   next sentence's: `S1। [4] S2। [4]` for `S1 [4]. S2 [4].`
2. In another answer it put a marker that really **did** belong to the next
   sentence in exactly that position: `S1 [1]। [2] S2।` for
   `S1 [1]. S2 [2].` Position alone cannot tell (1) from (2).
3. The 3B model itself sometimes writes the English that way:
   `…machinery. [1] The DNA adducts…`, where `[1]` cites the sentence before.

What `rag/sarvam.py` does:

- **Before translating**, markers written after an English full stop are
  moved inside their sentence (`attach_markers`): `machinery [1].`
- **After translating**, markers are stripped from the output and, when both
  sides have the same number of sentences, sentence *i* gets English sentence
  *i*'s markers (`align_markers`). Sentence order is unambiguous where position
  was not.
- If the counts differ, the output is kept as written only if every marker
  survived and none opens a sentence; otherwise that line is translated
  sentence by sentence with the markers held back and re-attached. Same
  characters billed, more requests, so it is the last resort.

Lists, headings and bold lines keep their Markdown prefix: each line is
translated on its own, with `- `, `1. ` and `## ` held back.

## Choosing the models

Measured, not assumed (2026-11-03):

| | result |
|---|---|
| `mayura:v1`, `modern-colloquial`, patient answer | mostly English words in Devanagari glue ("Chemotherapy से fatigue, nausea और hair loss…") — unreadable for a patient who does not read English. Rejected for patients. |
| `sarvam-translate:v1`, patient answer | plain Hindi ("कीमोथेरेपी के कारण थकान, मतली और बाल झड़ सकते हैं [1]।"), markers in place; Tamil the same. **Used for patients.** |
| `mayura:v1`, `code-mixed`, clinician answer | Hindi grammar around English medical terms ("Cisplatin का mechanism of action DNA adducts और cross-links के formation से involve होता है") — how clinicians in India actually speak. **Used for doctors.** |
| `sarvam-translate:v1` with `code-mixed` | 400: the mode exists only on mayura |
| `bulbul:v4-flash` | works with its own named voices (`roopa_hi_companion` is used, and reads all 11 languages); rejects the v3 names (`shubh`, plain `ishita` → 400). ₹1 / 1,000 characters against ₹3 for `bulbul:v3`. **Used**, with v3 as automatic fallback. |
| Hinglish typed ("cisplatin ka renal dose adjustment kya hai?") | language-ID says Hindi; translates to "What is the renal dose adjustment for cisplatin?" |

---

## Credits

Prices from docs.sarvam.ai (read 2026-11-01):

| step | price | typical use |
|---|---|---|
| speech to text, `saaras:v4` | ₹0.49 / min | a 6-second question ≈ ₹0.05 |
| translate | ₹2 / 1,000 chars | a question ≈ ₹0.1; a 300-word patient answer ≈ ₹3–4; a 500-word code-mixed doctor answer ≈ ₹5–7 |
| text to speech, `bulbul:v4-flash` | ₹1 / 1,000 chars | a whole patient answer ≈ ₹1.5–2, only when Listen is pressed |
| language ID | ₹0.35 / 1,000 chars | only for Devanagari (Hindi vs Marathi) and Latin text; ≈ ₹0.02 |

How spending is kept down:

- **Everything is cached on disk** (`artifacts/voice_cache/`: SQLite for
  text, MP3 files for audio), shared by all users. A repeated question, the
  same answer reopened, a replayed Listen — all ₹0.
- **No speech unless asked.** Answers are never read aloud automatically.
- **Detection is free where it can be.** 9 of the 11 languages are decided by
  their script alone.
- **On-demand translation** for reference text (source passages, Learn).
- **A daily budget**: `limits.daily_budget_inr` in `config/voice.yaml`
  (₹500 a day). `limits.daily_budget_from` can schedule a
  different cap from a given date. Once reached, calls are refused with a
  readable message until midnight; cached results and English still work.
- **A readable report**: `sarvam_api_usage.txt` in the project root is
  rewritten after every billable call: total spent, spend per API, then every
  transaction (newest first). Rebuild by hand with
  `python scripts/12_sarvam_usage.py`.
- **A usage log**: every billable call is appended to
  `logs/sarvam_usage.jsonl` with its estimated cost. Spend is not shown in the
  web app; read it in `sarvam_api_usage.txt`.
- **Tests spend nothing.** `smoke_test.py` and `ui_test.py` run against an
  offline mock (`SARVAM_MOCK=1`, set automatically by `ui_test.py`), with a
  throwaway cache. `scripts/11_voice_check.py` is the one live check.

Building and verifying this feature, including every probe used to choose
the models above, cost about ₹16 in total, almost all of it translation:
each browser re-run of the doctor check produced a new ~500-word answer,
and so a new translation (₹3–5 each).

---

## Results

`python scripts/11_voice_check.py` — the live round trip (₹0.62 the first
time, ₹0 on a rerun):

| check | result |
|---|---|
| Hindi question read aloud (`bulbul:v4-flash`) | 34 KB MP3 |
| that audio transcribed with language "unknown" | heard as **hi-IN**, transcript identical to what was said (similarity 1.00) |
| translated to English | "What are the side effects of chemotherapy?" |
| answered by the live patient pipeline | answered, cited |
| answer translated back to Hindi | every citation marker kept |
| Tamil question | detected from its script, no API call |
| Hinglish clinician question | "What is the renal dose adjustment for cisplatin?" |

In the browser, on the running app:

- Patient Portal, Hindi: answer in ~35 s (the model's time; translation adds
  ~2 s), the Hindi answer followed by the English, citations on every
  line, Listen plays the Hindi answer.
- Knowledge Assistant, Hindi: answer in ~60 s, code-mixed Hindi, citations
  identical to the English beneath it.
- Suggested questions appear in the chosen language.
- On a phone-width screen the rotating line stacks above the language
  selector, at the top of the page.

Tests: `smoke_test.py` adds 19 voice checks (script detection for all 11
scripts, the three citation-placement cases, caching, the usage log, the
budget and its 8 November step-up, and a Hindi crisis message caught after
translation); `ui_test.py` adds 15 offline checks (English by default with
no Sarvam call for typed text, the rotating line, the "Ask in हिन्दी"
button, a Hindi crisis typed in the portal, a Tamil off-topic refusal on
Auto-detect, a Hindi question to the textbooks) and a live check that a
Hindi patient answer is shown in Hindi followed by the English.

---

## Known limits

- **The certificate is self-signed.** Browsers allow the microphone only on
  a secure page, so `./start` serves HTTPS with a certificate it makes in
  `certs/` for `localhost` and the machine's LAN addresses. Each device
  warns once (*Advanced → Proceed*). A public deployment needs a real
  certificate; `./start --http` turns HTTPS off, leaving the microphone
  working only on `localhost`.
- **The interface stays in English.** Answers, notices and suggestions are
  translated; labels and buttons are not.
- **Code-mixed doctor Hindi is English-heavy by design.** If plain Hindi is
  wanted instead, set `translate.clinician` in `config/voice.yaml` to
  `{model: "sarvam-translate:v1", mode: null}`.
- **Hindi or Marathi** in Devanagari, and **English or romanised Hindi** in
  Latin script, are decided by Sarvam's language ID; a very short question
  can be misjudged. Picking the language in the dropdown removes the guess.
- **Translation can soften medical precision.** The English is always
  one click away, and is what was checked.
- **Read-aloud length** is capped at 4,000 characters (`limits.tts_max_chars`);
  longer doctor answers are read up to that point.

## Files

| file | what it does |
|---|---|
| `rag/sarvam.py` | the Sarvam client: transcribe, detect, translate (with citation alignment), speak; disk cache, usage log, budget, offline mock |
| `app/voice_ui.py` | the Streamlit side: language picker, reading the chat box (text or audio), the rotating language line, the answer in both languages, Listen and "Show in …" buttons |
| `config/voice.yaml` | models, languages, translation styles, limits, budget, prices |
| `scripts/11_voice_check.py` | the one live, credit-capped check |

## Several keys

`.env` can hold spare keys. They are tried in this order and the first one
with credit is used: `SARVAM_API_KEY`, `SARVAM_API_KEY_1`,
`SARVAM_API_KEY_2`, then `SARVAM_API_KEY_coiffeur`. When Sarvam refuses a key as
out of credit (402, 401/403, or a 429 that names credit or quota) the key is
remembered in `artifacts/key_state.json` (a short fingerprint, never the key)
and skipped from then on, across restarts. A plain "too many requests" 429 does
not retire a key. Each transaction in `sarvam_api_usage.txt` shows which key
served it (`key=`). Delete `artifacts/key_state.json` after topping up to try
the keys again. The daily rupee cap still applies to all keys together.

Scite works the same way (`rag/keyring.py`): `SCITE_API_KEY` then
`extra_scite_api` for the Assistant API, and `SCITE_MCP_API_KEY` then
`extra_scite_api` for the literature-search fallback; a key over its monthly
limit is skipped until the 1st of next month.

## Recording limit and the recording indicator

Speech to text is billed per second, so a recording is capped at
`limits.max_record_seconds` (30) in `config/voice.yaml`.

- **On the page** (`recording_indicator()` in `app/voice_ui.py`): while the
  microphone is recording, the input glows red and a bar above it shows a pulsing
  dot, moving bars and "Recording… tick to send, cross to cancel". No countdown
  is shown; the limit works quietly in the background: at the limit the page
  presses the tick itself, so the recording is sent rather than lost. When it
  ends, the bar says "Recording sent" or "Recording cancelled" for two seconds.
  Streamlit's own recorder offers only a dotted line with a cross and a tick, and
  has no length limit.
- **On the server** (`rag/sarvam.py`): a recording over the limit (plus 1.5 s of
  slack) is refused before any request is made, so it costs nothing, even if the
  page's own stop fails.

The fixed "not found" messages (not available, excerpts, out of scope, and the
same with the "this search was limited to the guide you selected" tail) are
translated and voiced once per language. After that every use is read from the
disk cache, free (`scripts/smoke_test.py` checks this).
