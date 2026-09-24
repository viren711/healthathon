# Design: how the platform stays honest

The whole product rests on one promise -- it answers only from the two books,
cites what it used, and says so when it cannot. This is how that is enforced,
and what had to change as the corpus widened from one disease to two whole
textbooks.

### The refusal guarantee

"Say when you don't know" cannot rest on a prompt, so it is enforced at three
independent points. Any one of them can refuse, and the first two never spend
an LLM call:

| # | Layer | Where | Refuses when |
|---|---|---|---|
| 1 | Topic gate | before retrieval | the question is outside `guard.topic.scope` — judged by lexicon and by embedding distance from the corpus centroid and a set of canonical in-scope questions |
| 2 | Grounding gate | after retrieval | nothing retrieved clears the reranker threshold |
| 3 | Citation check | after generation | the model emitted the `ANSWER_NOT_AVAILABLE` sentinel, cited nothing, or cited a source number that was never supplied |

Layer 3 is what makes the guarantee independent of the model's goodwill: an
uncited or unverifiable answer is replaced with *"The answer is not available
in the provided sources."* — the user never sees it.

---

#### Streaming and the guarantee

Layer 3 runs on the *finished* answer, so streaming means putting unverified
text on screen. That is a real tension with the refusal guarantee, and it is
resolved rather than ignored: streamed text is rendered as a greyed,
amber-bordered draft captioned "citations not yet verified", and when
generation ends the draft is **discarded** and replaced by whatever the
verified `Answer` says. If layer 3 rejects it, the user is left with "the
answer is not available", not with the model's prose.

Set `llm.stream: false` in `config/rag.yaml` if you would rather never display
unverified text at all; the platform then waits and shows only the final,
checked answer.

---

#### Scope: what layer 1 is allowed to refuse

`guard.topic.scope` has two settings, and the lexicon of disease sites means
the *opposite* thing in each:

- `tongue` — naming another disease site (`prostate`, `larynx`, `thyroid`) is
  an outright refusal. Correct when only tongue content is indexed.
- `corpus` (default) — those same terms are the strongest cheap signal that a
  question is **in** scope, because the books have a chapter on every one of
  them. Only questions unrelated to the books at all are refused here.

Under `corpus` this layer is deliberately permissive. Deciding whether the
books actually answer a medical question is the grounding gate's job, and it
does that on retrieved evidence rather than on vocabulary. Measured on
`eval/questions.yaml`: **0/12** in-scope non-tongue questions wrongly refused,
**0/10** off-topic questions leaked.

That split is why the evaluator has an `in_scope_other` bucket. It exists to
catch the failure mode where the index holds the answer and the topic gate
refuses to look — which is exactly what the platform did before full-book
indexing, and which no other bucket would have detected.

**Deciding whether to answer and deciding what to show are separate.**
`min_rerank_score` is calibrated against the *top* score of each question, so
it answers "is there evidence here?". It is deliberately **not** reused to pick
which passages go into the prompt: cross-encoder scores cluster, and a question
whose top five chunks score 2.97 / 2.96 / 2.91 / 2.80 / 2.36 has five relevant
passages, not one. Filtering on the gate threshold there left the model a
single uncorroborated source, and it correctly declined to answer from it —
turning good retrieval into a false "not available" on 4 of 25 questions.
`guard.grounding.support_margin` fixes this: once the gate passes, every hit
within that margin of the best one is shown too.

---

#### Why layer 1 stopped deciding scope

Layer 1 began as a similarity check against canonical seed questions. Over a
tongue-only corpus that was a fair proxy for "can we answer this?". Over the
whole of both textbooks it is not, and it failed in the worst possible way —
on **phrasing rather than topic**:

| query | seed similarity | verdict |
|---|---|---|
| "What causes skin cancer?" | 0.619 | admitted |
| "state major causes of skin cancer" | 0.510 | **refused** |

The same question, and the books have three chapters on it. So under `corpus`
scope layer 1 is now only a **junk filter**: it refuses only a query that
looks unlike a medical question on *both* signals (`junk_max_seed` AND
`junk_max_centroid`). Everything else goes to retrieval, and the grounding
gate decides on evidence — which does not depend on how the user phrased it.
A junk query still costs one retrieval (~0.9 s) and still never reaches the
LLM.

---

#### Why the grounding threshold had to come down too

The same query was then refused by layer 2, at a threshold of 2.71. Measuring
the two populations on the full index showed why no threshold works:

| population | top rerank score |
|---|---|
| precise answerable questions | 0.30 – 8.77 |
| **broad, plainly-phrased questions** | **0.61 – 8.32** |
| unanswerable (must abstain) | −4.85 – **4.75** |

They **overlap completely**. Nothing admits "causes of skin cancer" (0.71)
while rejecting "five-year survival with CAR-T" (4.75). The old threshold was
calibrated on 25 *precise* questions only, so it sat where precise questions
live and refused ordinary ones.

Two fixes. `eval/questions.yaml` gained an `answerable_broad` bucket of
plainly-phrased questions, so calibration sees the distribution users actually
produce. And the sweep objective changed: instead of a symmetric harmonic mean
— which, given overlapping populations, just splits the difference and trades
real answers for abstentions layer 3 would have made anyway — it now takes the
most abstaining threshold that still answers `min_answer_rate` (95%) of the
answerable questions. That moved `min_rerank_score` from 2.71 to **0.30**.

This is a deliberate reallocation of work between layers, not a weakening.
Layer 2 is a cheap pre-filter that avoids LLM calls; **layer 3 is the
guarantee**, and it is the layer that actually verifies an answer against its
sources. Measured after the change: specific answerable **20→23/25**, broad
questions **8/12** (previously ~0), unanswerable still abstained.

---

#### A stale line in the prompt that quietly narrowed everything

Worth recording because it survived several rounds of scope work unnoticed.
The system prompt still opened with *"restricted to ONE topic: cancer of the
tongue"*, and carried a rule telling the model to emit the not-available token
for anything that was not tongue cancer. That was correct when it was written
and wrong from the moment both whole books were indexed: the gates had been
widened to the full corpus while the prompt kept telling the model to refuse
most of what the platform could now answer.

The domain sentence is now filled from `guard.topic.scope` rather than
hard-coded, so the prompt cannot drift away from the gate again. Scope lives
in one place and the prompt reads it.

---

### Answer length

The budget the reader sees is **`llm.max_answer_words: 500`**. It is enforced
in two places, and the order matters:

1. **The prompt states it.** Rule 6 tells the model to stay under the budget
   and to finish its final sentence. Measured across five long questions, all
   five came back at 142-452 words and none needed trimming — so the prompt is
   the mechanism and the trimmer is the backstop.
2. **`fit_to_length` enforces it**, by dropping whole units from the end until
   the answer fits. Never by cutting at whatever word the token budget ran out
   on: a mid-clause stop leaves a fragment with no citation, which reads as
   broken and can fail verification outright.

`llm.max_tokens` (800) sits deliberately **above** the word budget — roughly
1.5 tokens per word once citation markers are counted — so the model reaches a
natural ending on its own. Setting it down at the word budget would guarantee
the mid-sentence cut the trimmer exists to avoid.

#### What "a unit" means

Sentence splitting alone is not enough, because answers mix prose with bullet
lists and a bullet routinely has no terminal full stop — `- Surgery is
preferred [1]` would be glued onto whatever followed. So lines are the outer
unit, and only non-list lines are split further into sentences. The sentence
terminator also swallows any citation markers that follow it, since the marker
sits either side of the stop depending on the model's mood (`...border [1].`
or `...border. [1]`).

Two details that only show up when the trim actually bites: a trailing
lead-in whose list was cut away (`Management depends on stage [1]:` followed
by nothing) is dropped as well, and a trim that would leave the answer with no
citation at all is abandoned rather than handing verification something it
must reject.

Forced to a 60-word budget on real answers, 221 words trimmed to 57 ending
`...customization. [5]`, and 195 to 49 ending `...improved control with
surgery [1].` — whole sentences, citations intact.

**No length banner.** An over-long answer is trimmed cleanly before it reaches
the UI, so there is nothing broken to warn about and nothing for the reader to
do. `Answer.trace["length"]` records what happened for debugging.

---

### Find mode — the topic locator

"Ask" and "Find" answer different questions and are built differently.

`rag/locate.py` takes a topic rather than a question and returns **positions
in the books**, not prose. It never calls the LLM, so it returns in well under
a second and has no way to invent a reference. Three things it does that
question-answering does not:

- **Groups by section, not chunk.** Chunks overlap by 80 tokens on purpose, so
  an ungrouped list shows the same passage three times under three slightly
  different page spans — useless as a reference. Results collapse to one entry
  per section, at that section's best score, with the page range widened to
  cover every matching chunk.
- **Casts a wider net.** `search_wide()` retrieves ~80 candidates rather than
  5, because grouping collapses them; a normal `final_top_k` would leave two
  or three distinct places.
- **Applies no grounding threshold.** "Show me the ten weakest places this is
  mentioned" is a reasonable request for a reference tool, where answering
  from weak evidence is not. Every result carries its score so the user can
  judge it.

#### Go to section

Every result — in both modes — has a **Go to section** button that renders the
real PDF page inline, with prev/next paging across the section's range and a
download of just those pages.

It renders rather than links, for a reason: the books are 52 MB and 296 MB, so
serving one to the browser to jump to `#page=872` means shipping the whole
file, and a `file://` link is blocked from a web page anyway. PyMuPDF
rasterises a single page in a few milliseconds without touching the rest of
the document, so the expensive-looking option is the cheap one. Renders are
cached, since Streamlit re-runs the script on every interaction.

This is what makes a citation checkable rather than decorative: the page it
names is one click away.

---
