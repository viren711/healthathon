# Patient Portal

The platform's fourth tool, and the only one written for **patients and their
families** rather than clinicians. It answers questions about detection,
prevention, treatment, side effects and living with cancer, and offers
topic-wise reading material — all of it from the **NCCN Guidelines for
Patients**, and nothing else.

Two modes, on one page (`app/views/patient_portal.py`, at `/patients`):

| | what it does | language model? |
|---|---|---|
| **Ask** | a question in everyday words → a plain-language answer from the guides, every sentence cited, the guide page one click away | yes (the same 3B reference model) |
| **Learn** | each guide's own common questions, key points, questions to ask your doctor, and glossary — unchanged NCCN text, browsable and searchable | **no** |

Both are narrowed by two selectors **above** the content (not in the
sidebar — the lesson of an earlier regression): a **topic area** and a
**guide**.

---

## The source material

76 guides, 5,006 pages, from
[nccn.org](https://www.nccn.org/patientresources/patient-resources/guidelines-for-patients),
in four topic areas mirroring NCCN's own folders:

| topic area | guides | e.g. |
|---|---|---|
| Detection, prevention & risk reduction | 5 | breast, colorectal, lung and prostate screening; genetic testing |
| Treatment by cancer type | 56 | mouth, throat, lung, breast, colon, prostate, lymphomas, leukaemias… |
| Supportive care | 12 | distress, fatigue, nausea, blood clots, low blood counts, immunotherapy side effects, palliative care, survivorship, quitting smoking |
| Specific populations | 3 | adolescents & young adults, older adults, caregivers |

65 are 2026 editions; the rest are 2023–2025. The edition year is part of
every citation. The registry is `config/patient_guides.yaml`: each guide's
id, title, year, topic area, and the Knowledge Assistant's pathology ids it
corresponds to (`mouth` → `hn_oral_cavity`), so the two tools can be linked.

The guides are copyrighted. Like the textbooks they live in `data/` and
everything derived from them in `corpus/` and `artifacts/`, all gitignored.
Running the portal locally is fine; publishing it would need NCCN's
permission.

### Why these guides suit the job

They are already written as patient education, and structured that way:

- most sections are headed by the **question a patient would ask** — "What is
  mouth cancer?", "How is distress managed?" (465 such sections);
- every chapter ends with **Key points** (419) and **Questions to ask** (465);
- every guide ends with **Words to know**, a glossary (3,412 terms in all).

So the Learn mode needs no generation at all: it is a matter of finding those
pieces reliably.

---

## Build: `scripts/07_build_patient_corpus.py`

```bash
python scripts/07_build_patient_corpus.py      # ~20 s on 12 workers
python scripts/03_build_index.py --chunks corpus/patient/chunks.jsonl \
    --out-dir artifacts/index_patient          # ~30 min (embedding)
python scripts/08_eval_patient.py --apply      # ~1 min
```

| output | what it is |
|---|---|
| `corpus/patient/sections.jsonl` | 4,082 sections, each with its guide, chapter, kind and pages |
| `corpus/patient/library.json` | the Learn mode's content: per guide, chapters → sections, plus glossary |
| `corpus/patient/chunks.jsonl` | 6,267 retrieval chunks (median 244 tokens), same record shape as the textbook chunks |
| `artifacts/index_patient/` | FAISS + BM25 + centroid, built by the unchanged stage 3 |

### Parallelism

| step | how |
|---|---|
| parse 76 PDFs | one process per guide (`ProcessPoolExecutor`, default all 12 threads), largest first so no core idles at the end; reading, sectioning and glossary extraction happen in that one pass. **1.8 s** wall-clock. |
| chunk 4,082 sections | a second pool; the tokenizer is loaded once per worker through an `initializer`, as in stage 2. **~15 s**. |
| embed | stage 3 pins torch to the 6 physical cores and fits BM25 in a concurrent process. Rebuilds reuse the vectors of unchanged chunks. |
| evaluate | all questions embedded in one batch; retrievals in a thread pool. |

### Sectioning

68 guides carry a PDF outline: level 1 is the chapter, level 2 the section.
The other 8 have none, but every page carries a **breadcrumb running header**
("Initial treatment » Options for limited-stage cancer"), so an outline is
rebuilt from where the breadcrumb changes, and both kinds go through the same
code. Things that had to be handled, each found by checking output, not
assumed:

- **Outlines out of page order** (Throat, Bladder): walked as-is, most of the
  guide collapsed into one section. Entries are sorted by page.
- **Outlines that stop early**: several end at the last "Questions to ask", so
  that section swallowed the glossary, contributors list and index. Back
  matter is detected from page headings and given its own (excluded) entry.
- **"Chapter 1", "Chapter 2" outlines**: named from the breadcrumb, or from
  the chapter's first heading in the three guides with no breadcrumb.
- **Chapter opener pages** (a mini contents list): in guides with
  breadcrumbs, every content page has one, and every page without one is an
  opener, blank or back matter — measured, none over 120 words — so their text
  is dropped.
- **Long bullet lists** (drug lists) have no sentence ends for the chunker to
  split on; they are regrouped item by item so no chunk exceeds the budget.

Page text is read with PyMuPDF's `dict` mode, so line size and weight are
known: headings are ≥ 12.5 pt, bullets are the dingbat "h" glyph, wrapped
lines are joined by vertical gap rather than punctuation, and glossary terms
are the **bold** lines (some guides set the whole glossary in heading-sized
type, so weight, not size, is what marks a term).

---

## Ask: the same pipeline, a second corpus

`rag/` takes a `corpus` argument — `"books"` (default) or `"patient"`:

| | Knowledge Assistant | Patient Portal |
|---|---|---|
| index | `artifacts/index` | `artifacts/index_patient` |
| embedder, reranker | shared — one loaded copy serves both | |
| scope filter | pathology / organ system | guide / topic area (stored in the same two chunk fields, so search code is unchanged) |
| layer 1 | junk filter, textbook seeds | junk filter, patient-phrased seeds (`config/patient.yaml`) |
| layer 2 | rerank threshold from `rag.yaml` | its own threshold, calibrated by stage 8 |
| layer 3 | citation check | identical, plus one retry when the model abstains on very strong evidence |
| prompt | clinical, ≤ 500 words | plain language, ≤ 300 words, never personal treatment advice, ends by pointing to the care team; its own worked example and user template |
| before retrieval | — | safety screen |

### When the model cannot write an answer

The 3B model sometimes abstains, drops its citations or times out although
retrieval found the page that answers the question. Rather than end in "not
available", the pipeline then returns status `excerpts`: the top 2 passages,
verbatim and cited, with the page viewer, under "I couldn't put this into a
short answer, but these passages from the NCCN Guidelines for Patients cover
it." It applies only when the best passage scores at least
`excerpt_min_score` (2.0; answerable questions score 6.0+, uncovered ones −9.7
to −2.2), so a question the guides barely touch still ends in "not available".
The same fallback covers a model server that is down or too slow, so the portal
keeps answering in *Learn*-style when the model cannot.

### Everyday words

Patients say "tired", the guides say "cancer-related fatigue". A short list in
`config/patient.yaml` (`query_expansions`) appends the guides' terms to the
text used for **searching only**; the model and the page show the question as
asked. Each entry was measured: "Why am I so tired all the time?" went from a
best score of −2.66 (refused at the gate) to 6.39 on the Fatigue guide.

### Safety screen (`rag/patient_safety.py`)

Pattern matching, no model — instant, deterministic, and impossible to argue
out of its job. Patterns and wording live in `config/patient.yaml`.

| category | e.g. | response |
|---|---|---|
| crisis | "I want to die", "thinking about killing myself" | helpline message (India: Tele-MANAS 14416; 112) **instead of** an answer, plus a link to the Distress guide |
| emergency | fever during chemo, can't breathe, coughing up blood | "contact your care team / emergency services now", shown **above** the answer |
| personal decision | "should I stop my chemo?", "can I skip radiation?" | general information, with a reminder that the decision belongs to the care team |

"Should I quit smoking?" is deliberately **not** a personal-decision flag —
that is advice the guides give.

---

## Learn: NCCN's own words

No generation: everything shown is NCCN text extracted by stage 7.

- **Common questions** — the question-headed sections, as expanders, each with
  its page and a "See this page in the guide" viewer.
- **Key points** — each chapter's summary bullets.
- **Questions to ask your doctor** — per chapter, downloadable as a `.txt`
  checklist to take to an appointment.
- **Words to know** — the glossary as a searchable table.
- **Read by chapter** — every section of a chosen chapter, in order.
- **Search all the guides** (when no guide is selected) — a word match across
  question headings, key points, section text and glossary terms.

---

## Results

From `scripts/08_eval_patient.py` against `eval/patient_questions.yaml`
(full report regenerated in `eval/patient_report.md`):

| check | result |
|---|---|
| Retrieval Recall@5 — the guide page that answers the question is in the top 5 | **100%** (35/35) |
| Off-topic questions refused before retrieval | **8/8** |
| Answerable questions wrongly refused | **0/35** |
| Not-covered questions refused at the evidence threshold | **6/6** |
| Safety cases routed to the right category (incl. 4 that must *not* fire) | **12/12** |

Evidence scores separate cleanly: answerable questions' best passage scores
6.0–10.6, not-covered questions' −9.7 to −2.2, so the threshold inherited
from the textbook side (0.3) sits inside the gap with room on both sides.

Parse and index cost: 1.8 s to parse all 76 guides, ~15 s to chunk, ~30 min
to embed 6,267 chunks on the 6 physical cores (once — rebuilds reuse vectors).

### Answer quality (live, Qwen2.5-3B)

14 everyday patient questions, run end to end through the model:

| | |
|---|---|
| answered from the guides | **10/14** |
| sentences carrying their own citation | **85%** (47/55) |
| answers that repeat the question back | **0** |
| time per answer | 17–63 s (median ~25 s) |

The prompt took four measured iterations. The first draft answered 5 of 8
and piled every citation at the end of a paragraph; a worked example
(fatigue: one marker per sentence, ending with the care team) fixed the
citations. A first-person question ("Will I…?") read to the 3B model as one
the guides could not answer, because they never mention the reader — so the
patient prompt and a patient-specific user template both say such questions
are answered with what the guides say about people in general. Dropping the
worked example was tested and rejected: 7/14 answered, 46% of sentences
cited. Every miss abstains ("not available"); none invents an answer.

### Measured, then fixed

- **The topic gate let 3 of 8 off-topic questions through** at the textbook
  junk-filter settings ("write a poem", "a for loop in Python", "best
  stocks"). The patient corpus is everyday prose, so its centroid is closer to
  general text than the textbooks' is: off-topic questions reached centroid
  0.498. Seed similarity separates instead — off-topic ≤ 0.408, plain patient
  questions with no medical word ("Will I lose my hair?") ≥ 0.453 — so the
  patient bars are seed 0.43 / centroid 0.55. Now 8/8, with no answerable
  question refused.
- **The safety screen missed "thinking about killing myself"** (the pattern
  allowed only "kill") and **flagged "How can I quit smoking?"** as a personal
  treatment decision. Both fixed; smoking, tobacco, vaping and alcohol are
  excluded from the decision patterns.

### Wider live check (24 new everyday questions)

| | before the fallback | now |
|---|---|---|
| answered by the model, cited | 18/24 | 18/24 |
| shown the guides' own passages instead | 0 | 4 more, 5 with the lower bar |
| "not available" | 6 | 1–2 ("What can I eat if I have mouth sores?": the guides touch it only faintly) |

### Known limits

- The safety screen is a pattern match. It catches common phrasings, not every
  possible one, and it can over-fire on an informational question that uses
  emergency words ("Can cancer cause shortness of breath?" shows the urgent-care
  notice before answering). The second is the safer error.
- **A few well-covered questions still abstain.** "Will I lose my hair during
  chemotherapy?" returns "not available" although its best passage scores 8.3
  (the bar is 0.3) and answers it directly; ablation showed the 3B model's
  decision flips on small prompt changes. A bounded retry naming the top
  passage recovers some of these (distress, palliative care) but not that
  one. The failure is safe — no answer rather than a wrong one — and the
  Learn tab and "Search all the guides" still find the passage. A larger
  model is the real fix.
- "Should I stop my treatment?" shows the care-team notice and, in practice,
  usually no answer: the model treats it as outside what the guides can say.
- First-person questions full of specifics retrieve less well than the
  general form: "I have a fever of 39 during chemotherapy, what should I do?"
  shows the emergency notice but then finds no passage above the bar, while
  "What should I do if I get a fever during chemotherapy?" is answered from
  the Low Blood Cell Counts guide.
- Learn mode shows NCCN text as extracted. Figures and tables are not
  reproduced; "See this page in the guide" shows the real page.
- Answers are educational. The page says so, and the prompt forbids personal
  treatment advice, but the 3B model is not checked sentence-by-sentence for
  it beyond the citation check.
