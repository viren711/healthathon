# Content Studio

The platform's second tool: a physician describes a blog post or article, answers
a few questions, and gets a draft to review, edit and publish. Like the
Knowledge Assistant, it runs entirely on this machine.

It is deliberately **not** grounded in the textbooks. The Knowledge Assistant
answers only from the indexed textbooks (*DeVita*, *Gunderson & Tepper*, *MD Anderson Surgical Oncology Handbook*, *Oncologic Imaging*) and cites every fact; the
studio writes from the model's general knowledge and cannot cite anything. The
page says so above every draft, and the prompt forbids the things that would
disguise it (see [Keeping drafts honest](#keeping-drafts-honest)).

---

## The flow

Three steps, held in Streamlit session state so a rerun never loses the doctor's
input. Every key the page owns is prefixed `studio_`, so it cannot collide with
the Knowledge Assistant's state.

| step | the doctor provides | what happens |
|---|---|---|
| **1 · Brief** | topic, a sentence or two describing the piece, kind of piece | validated locally (`Brief.problems()`) before any model call |
| **2 · Questions** | audience, tone, length, must-cover, avoid, call to action, disclaimer — plus answers to **3 questions generated for this topic** | the writer model proposes the topic questions; if the model server is unreachable, three generic fallbacks from `config/writer.yaml` are used instead and the page says so |
| **3 · Draft** | — | the article streams onto the page with a live word count, then is offered for download as Markdown, regeneration, or editing the answers |

Kinds of piece, audiences, tones and lengths are all lists in
`config/writer.yaml`, not code. Each kind of piece carries a style description
that goes into the prompt (a patient handout is short sections and plain
words; a professional article assumes clinical vocabulary).

The fixed questions cover what every piece needs. The generated ones exist
because what a good article on *oral cancer screening* needs to know (is this
for a dental audience? should it mention HPV?) is different from what one on
*chemotherapy fatigue* does. They are asked at temperature 0.3 as a JSON array;
the parser also accepts a fenced block or a numbered list, because small
models do not always follow format instructions.

---

## One model in memory at a time

The machine has 14 GB of RAM, of which ~7.4 GB is typically free. The
reference model needs 3.4 GB resident, a 4B writer 4.4 GB. Both at once would
not fit comfortably, and nothing needs both at once — a doctor is either
looking something up or writing.

So there is **one llama-server, in router mode**:

```
llama-server --models-preset config/models.ini --models-max 1
```

`config/models.ini` lists every model the platform can ask for. Each request
names its model (`"model": "qwen3.5-4b"`); the router loads it on demand and,
because at most one may be resident, unloads the previous one first.

| transition | measured cost |
|---|---|
| reference → writer (first draft) | ~10 s |
| writer → reference (next question) | ~4 s |
| same model again | none |

The home page and the studio's sidebar both show which model is resident, read
from the router's `/v1/models` status (`LLMClient.resident_models()`).

The router would also pick up any GGUF in the Hugging Face cache and offer it
as a model. `serve_llm.sh` points `HF_HUB_CACHE` and `LLAMA_CACHE` at an empty
directory (`models/.router-cache`) so that only the presets are served.

### The retrieval models

The LLM is not the only memory the Knowledge Assistant holds. Its embedder,
reranker, FAISS and BM25 indexes live in the Streamlit process — about 1.2 GB.
Releasing them before a draft was measured and is a poor default:

- only ~0.56 GB actually returns to the OS (the allocator keeps the rest), and
- the next reference question then pays a 15–35 s reload.

So the studio releases them **only when free memory is below
`release_reference_below_gb`** (5.0 GB) at the moment a draft starts. On a
machine with room to spare, switching back to the Knowledge Assistant stays
fast.

---

## Choosing the writer model

Constraints: CPU only, fits in ~4.5 GB resident next to everything else, and
writes clinically accurate, readable prose. Two 4B models at Q4_K_M were
benchmarked on the same briefs:

| | Qwen3-4B-Instruct-2507 | **Qwen3.5-4B** (default) |
|---|---|---|
| file size | 2.4 GB | 2.6 GB |
| speed | ~6.0 tok/s | ~5.0 tok/s |
| 600-word patient blog | 116 s, 555 words | 166 s, 604 words |
| 400-word clinical piece | 95 s, 390 words | 110 s, 381 words |
| clinical accuracy | **two errors**: claimed HPV-positive oropharyngeal cancer is *less* likely to involve nodes (the opposite is true); recommended the discontinued quadrivalent/bivalent vaccines | no errors found |

A faster writer that gets the clinical facts wrong is no help to a physician,
so Qwen3.5-4B is the default and Qwen3-4B is offered as "Faster" in the
sidebar. DeepSeek-R1 distills of similar size were not benchmarked: they are
reasoning models that spend much of their token budget thinking before
writing, which at ~5 tok/s would mean minutes of waiting for nothing visible.

**Qwen3.5 thinks by default too.** Left alone it produced 705 words of hidden
reasoning and a 171-word article cut off mid-sentence. `config/writer.yaml`
sends `chat_template_kwargs: {enable_thinking: false}` with every request (via
the `extra_body` field added to `LLMClient`), and `visible_text()` strips any
`<think>` block that slips through anyway, including an unclosed one.

Larger models (7–8B) were not tried. Generation here is memory-bandwidth
bound, so doubling the parameters should roughly halve the speed (~2.5 tok/s,
an estimate, not a measurement): a 600-word draft would take five to six
minutes, and the model would need ~6 GB resident.

---

## Keeping drafts honest

A draft that reads well and is wrong is worse than no draft. The system prompt
(`ARTICLE_SYSTEM` in `writer/studio.py`) tells the model:

- no statistics, percentages or survival figures unless the doctor supplied them;
- no references, study names, journals or "recent research shows";
- plain Markdown: one `#` title, `##` sections, no emoji;
- no disclaimer of its own (the page adds a fixed one if the doctor asks);
- stay within 10% of the requested length.

Then, after generation, `tidy_article()`:

- strips `<think>` blocks and emoji;
- promotes a bold first line to a proper `#` title;
- if the model hit its token limit, cuts back to the last complete paragraph
  (or sentence), so a draft never ends mid-sentence.

The token ceiling is `words × 1.6 + 200`, generous enough that a draft at the
requested length is never truncated; the length instruction does the steering.

The page shows the draft under a **"Draft for your review"** warning that
states it is written from general knowledge, not the textbooks, and that every
clinical statement must be checked before publication.

---

## Code

| file | what it does |
|---|---|
| `writer/studio.py` | all logic: config, the `Brief`, question generation, prompt, streaming generation, tidy-up, memory release. No Streamlit import, so it is testable on its own. |
| `app/views/content_studio.py` | the three steps, rendering only |
| `config/writer.yaml` | writer models, lengths, kinds of piece, audiences, tones, disclaimer, fallback questions, memory threshold |
| `config/models.ini` | the router presets shared with the Knowledge Assistant |

---

## Tuning

- **Drafts run long.** Measured drafts land 0–20% over the requested length;
  short pieces overshoot most. Lower the `lengths` values in
  `config/writer.yaml` if that matters.
- **Faster drafts:** choose "Faster — Qwen3-4B" in the sidebar, at the
  accuracy cost above.
- **More memory headroom:** raise `release_reference_below_gb` to always free
  the retrieval models before a draft, or lower `ctx-size` for the writers in
  `config/models.ini` (6144 covers the 1000-word draft with room to spare).
- **Adding a writer:** add a section to `config/models.ini`, an entry with the
  same `id` to `config/writer.yaml`, and the download to
  `scripts/00_fetch_models.sh`. Models whose GGUF is missing are hidden from
  the sidebar automatically.
