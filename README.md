# Doctor Knowledge & Productivity Assistant

A set of tools for physicians that runs entirely on one CPU-only machine —
nothing typed into it leaves the computer. The portal opens on a home page
that describes each tool and links to it:

| tool | what it does |
|---|---|
| 🔬 **Knowledge Assistant** | cited answers from two oncology textbooks, with refusals when the books do not cover the question (the rest of this README) |
| ✍️ **Content Studio** | drafts a blog post, patient handout or article from a short brief and a few questions — see **[docs/content-studio.md](docs/content-studio.md)** |
| 🧪 **Research Platform** | placeholder, not yet built |

The Knowledge Assistant and the Content Studio use different language models
but share **one model server that holds one model at a time**, swapping on
demand (~10 s into the writer, ~4 s back). Neither tool's model sits in memory
while the other is in use.

---

## Knowledge Assistant

A retrieval-augmented question-answering platform over two clinical oncology
textbooks, built to run entirely on a CPU. Ask a question in plain English and
every fact in the answer comes back with the book, chapter and page it came
from — and the page itself is one click away. When the books do not answer the
question, it says so instead of guessing.

Two modes over the same index:

- **Ask** — a question gets a cited answer, or an honest refusal.
- **Find** — a topic gets a list of *where it is discussed*, book by book and
  page by page, with no model involved. Instant, and it cannot invent a
  reference.

Filter either mode to a single organ system or pathology ("Head & Neck → Oral
cavity & tongue") from the left-hand selector.

---

## What is in this repository, and what is not

**Not included, and gitignored:** the two source PDFs, the extracted corpus,
the built index, and the model weights. The books are copyrighted and are not
redistributed here — neither are the ~20 million characters of text extracted
from them. You supply your own copies.

**Included:** all the code, configuration, the organ-system taxonomy, the
evaluation sets and the documentation. Everything needed to rebuild the index
from your own PDFs.

| you supply | the pipeline builds |
|---|---|
| `data/*.pdf` (2 books, ~350 MB) | `corpus/` — cleaned, sectioned text |
| | `artifacts/index/` — FAISS + BM25 + taxonomy |
| | `models/` — llama.cpp binary + GGUF weights (downloaded, ~7.7 GB) |

---

## Requirements

- Linux x86-64, Python 3.12
- ~8 GB RAM, ~12 GB free disk (7.7 GB models, 2 GB venv, ~0.9 GB corpus + index).
  Only the 3B reference model and one writer are required; the two optional
  models (1.5B reference, Qwen3-4B writer) can be dropped from
  `scripts/00_fetch_models.sh` to save ~3.3 GB.
- **No GPU and no C toolchain required.** The generator is a prebuilt
  llama.cpp binary, because this machine has neither.

The two books it was built against:

- *DeVita, Hellman & Rosenberg's Cancer: Principles & Practice of Oncology*, 11th ed. (4,360 pp)
- *Gunderson & Tepper's Clinical Radiation Oncology* (2,300 pp)

Place them in `data/` and set their filenames and page counts in
`config/books.yaml`. Other books will work, but the chapter/page-window
settings in that file are specific to these editions.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# put your two PDFs in data/, then:
bash scripts/run_all.sh     # models + PDFs -> corpus -> chunks -> index -> calibration

./start                     # http://localhost:8501
```

`run_all.sh` downloads the models in one lane while the data pipeline runs in
another. Budget about **90 minutes** end to end on a 6-core CPU, almost all of
it embedding 15,844 chunks; everything else is minutes.

```bash
./start                  # model server + portal (home page, all tools)
./start --model 1.5b     # smaller, ~2x faster generator (it transcribes; see docs)
./start --no-llm         # retrieval and topic lookup only, no model needed
./stop                   # stop everything from any terminal
./stop --status          # what is running, and which ports are held
```

The portal opens on the home page; the Knowledge Assistant, Content Studio
and Research Platform are in the left-hand navigation and linked from the
home page.

`./start` runs both services as children of that terminal — close it, or press
Ctrl+C, and both stop. `./stop` is for when that did not happen: a run killed
with `-9`, or backgrounded and forgotten, which shows up as `./start` refusing
to boot because port 8000 is held.

---

## How it works

```
your question
     |
     v
  [1] topic gate .......... is this a question these books could answer?
     |                      cheap, no model call
     v
  hybrid retrieval ........ dense (FAISS) + BM25, fused with RRF,
     |                      then cross-encoder reranked
     v
  [2] grounding gate ...... did anything retrieved actually clear the bar?
     |                      refuses before spending an LLM call
     v
  generation .............. Qwen2.5-3B, 4-bit, on CPU via llama.cpp
     |                      answer must cite every sentence
     v
  [3] citation check ...... is the answer really tied to those sources?
     |
     v
  cited answer, or "the answer is not available in the provided sources"
```

**The refusal guarantee.** "Say when you don't know" cannot rest on a prompt,
so it is enforced at three independent points. Any one of them can refuse, and
the first two never spend an LLM call. Layer 3 is what makes it independent of
the model's goodwill: an uncited or unverifiable answer is replaced with the
refusal, and the user never sees it.

The full technical picture — data, preprocessing, indexing, retrieval and the
query path end to end — is in **[docs/architecture.md](docs/architecture.md)**.
Why the guarantee is built this way, and what changed when the corpus widened
from tongue cancer to both whole books, is in
**[docs/design.md](docs/design.md)**.

---

## Results

Measured on an AMD Ryzen 5 5600G (6 cores, no GPU). Full numbers and method in
**[docs/measurements.md](docs/measurements.md)**.

| | |
|---|---|
| Retrieval Recall@5 | **96%** (24/25 labelled questions) |
| Retrieval latency | **~0.9 s** per question |
| Correct organ system in top-5 | **38/38** across all 14 systems |
| Answer latency | ~34 s, first words on screen at ~10 s |

End to end with all three refusal layers live, over 67 labelled questions:

| bucket | required behaviour | result |
|---|---|---|
| 25 answerable (precise) | answer, with citations | **23** |
| 12 answerable (broad phrasing) | answer, with citations | **9** |
| 12 in-scope, non-tongue | must not be refused as off-scope | **12/12** |
| 8 unanswerable | abstain | **8/8** |
| 10 off-topic | refuse | **10/10** |

Every shortfall is a *safe* failure: the platform said "not available" where it
could have answered, never the reverse.

---

## Repository layout

```
config/       books.yaml (page windows, lexicons), rag.yaml (models,
              thresholds, messages), organ_systems.yaml (chapter -> pathology),
              models.ini (model server presets), writer.yaml (Content Studio)
rag/          the library: retriever, guard, generate, pipeline, locate,
              pdfview, llm, config
writer/       Content Studio logic: brief, questions, prompt, tidy-up
app/          Streamlit portal (view layer only): streamlit_app.py is the
              entrypoint, one file per page in app/views/
scripts/      00-06 pipeline stages, plus smoke_test / ui_test / ask_batch
eval/         labelled question sets and generated reports
docs/         design, pipeline, measurements, operating, cancer classes
queries.txt   100 example queries, grouped by organ system
start, stop   run and stop the platform
```

| module | what it does |
|---|---|
| `rag/retriever.py` | hybrid search + rerank, and scoped search per pathology |
| `rag/guard.py` | the three refusal layers |
| `rag/generate.py` | prompt construction and answer-length fitting |
| `rag/pipeline.py` | `answer_question()` — the only entry point the UI needs |
| `rag/locate.py` | the topic locator (Find mode) |
| `rag/pdfview.py` | renders source PDF pages for "Go to section" |
| `writer/studio.py` | the Content Studio: follow-up questions, article prompt, streaming, tidy-up |

---

## Documentation

| | |
|---|---|
| **[docs/architecture.md](docs/architecture.md)** | **start here** — data, preprocessing, indexing, retrieval, and the full query path |
| **[docs/content-studio.md](docs/content-studio.md)** | the article writer: flow, one-model-at-a-time memory design, writer model benchmarks |
| **[docs/design.md](docs/design.md)** | the three refusal layers, scope decisions, streaming, answer length |
| **[docs/pipeline.md](docs/pipeline.md)** | the six build stages, parallelism, corpus modes, organ-system tagging |
| **[docs/measurements.md](docs/measurements.md)** | every measured number — including the optimisations that were tried and rejected |
| **[docs/operating.md](docs/operating.md)** | testing without a GPU, tuning notes |
| **[docs/cancer-classes.md](docs/cancer-classes.md)** | every malignancy the books cover, by tissue of origin and by organ system |

---

## Testing

```bash
python scripts/smoke_test.py        # 65 checks: guards, citations, scoping, PDF rendering
python scripts/ui_test.py --no-llm  # 33 checks: clicks the real UI via Streamlit AppTest
python scripts/ui_test.py           # 46 checks: adds a real answer and a real draft (~2.5 min)
python scripts/04_evaluate.py       # Recall@k, refusal accuracy, threshold calibration
python scripts/06_eval_coverage.py  # retrieval coverage across all 14 organ systems
python scripts/ask_batch.py --locate  # all 100 example queries, no model needed
```

`smoke_test.py` runs the whole stack against a stubbed model server, so the
plumbing — including all three refusal layers — can be verified without a GPU
or even a generator. `ui_test.py` exists because pipeline tests cannot catch
interaction bugs: it was written after a "Go to section" button rendered its
page correctly and then hid it inside a container that snapped shut. It
covers the home page, both Knowledge Assistant modes and the Content Studio.

---

## Reproducing this from scratch

Everything except the books is in this repository. The pipeline is
deterministic up to the embedding step, so with the same two PDFs you should
land on the same corpus, the same chunk count, and the same retrieval numbers.

### 1. Environment

```bash
git clone <this repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate

# torch must come from the CPU index -- the default wheel pulls ~2.5 GB of
# unusable CUDA libraries on a machine with no GPU.
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Linux x86-64, Python 3.12, ~8 GB RAM, ~7 GB free disk. No GPU and no C
compiler: the generator is a prebuilt llama.cpp binary, chosen precisely
because the target machine had neither.

### 2. Supply the books

Put your own copies in `data/`. They are not distributed here.

| book | edition used | file size | pages |
|---|---|---|---|
| DeVita, Hellman & Rosenberg's *Cancer: Principles & Practice of Oncology* | 11th | 54.3 MB | 4,360 |
| Gunderson & Tepper's *Clinical Radiation Oncology* | — | 309.9 MB | 2,300 |

Then set the filenames and page counts in `config/books.yaml`.

**Page counts must match your PDF exactly.** They are how the pipeline
converts a chapter into a page range. If your copy has a different front
matter or a different scan, the counts differ and so will everything
downstream.

### 3. Build

```bash
bash scripts/run_all.sh
```

That runs all six stages, downloading the models in a parallel lane. Roughly
90 minutes on a 6-core CPU, of which ~75 is embedding. To run them
individually:

| stage | command | output | time |
|---|---|---|---|
| 0 | `bash scripts/00_fetch_models.sh` | `models/` — llama.cpp binary, GGUF, encoders | ~5 min (2.8 GB download) |
| 1 | `python scripts/01_extract_corpus.py --workers 12` | `corpus/books/sections/*.md` | ~50 s |
| 2 | `python scripts/02_build_chunks.py --workers 12` | `corpus/books/chunks.jsonl` | ~30 s |
| 3 | `python scripts/03_build_index.py` | `artifacts/index/` — FAISS, BM25, centroid | **~75 min** |
| 5 | `python scripts/05_tag_pathologies.py --write-corpora` | organ-system tags, `taxonomy.json` | ~3 s |
| 4 | `python scripts/04_evaluate.py --apply` | `eval/report.md`, calibrated thresholds | ~1 min |
| 6 | `python scripts/06_eval_coverage.py --scoped` | `eval/coverage_report.md` | ~1 min |

Stage 5 runs before stage 4 on purpose: tagging is metadata only and must
never re-embed, and stage 4 needs the finished index to calibrate against.

### 4. Check you got the same thing

If your PDFs match the editions above, these should match too:

| checkpoint | expected | where to look |
|---|---|---|
| Pages read | 4,360 + 2,300 | `corpus/books/corpus_report.md` |
| Pages kept | 2,890 + 1,719 | same |
| Sections | **9,482** | same, and `ls corpus/books/sections \| wc -l` |
| Corpus size | ~19.7 M characters | same |
| Chunks | **15,844** | stage 2 output, `artifacts/index/index_meta.json` |
| Corpus fingerprint | `cb81b62f3ad6d73b` | `index_meta.json` |
| Taxonomy | 14 systems, 78 pathologies | `artifacts/index/taxonomy.json` |
| Recall@5 | **96%** (24/25) | `eval/report.md` |
| Organ-system coverage | **38/38** in top-5 | `eval/coverage_report.md` |
| Calibrated `min_rerank_score` | 0.30 | written into `config/rag.yaml` |

`corpus_fingerprint` is a hash of the chunk ids. **If it matches, stages 1-2
reproduced exactly** — different PDFs, or an edited `books.yaml`, will change
it.

### 5. Verify and run

```bash
python scripts/smoke_test.py         # 65 checks, stubbed model — no GPU needed
python scripts/ui_test.py --no-llm   # 20 UI checks
./start                              # http://localhost:8501
```

`smoke_test.py` exercises all three refusal layers against a stub, so the
guarantee can be checked without a generator at all.

### What is and is not deterministic

- **Stages 1-2 are exactly reproducible.** The parallel rewrite of stage 1 was
  verified to reproduce the previous single-threaded corpus byte for byte, and
  stage 2 re-sorts by section id so worker count cannot change the output.
- **Stage 3 is reproducible on the same hardware and library versions.**
  Embeddings are a deterministic function of the model and the text, but
  floating-point reduction order can differ across BLAS builds, so a different
  machine may produce vectors that differ in the last decimal. Ranking is
  unaffected in practice.
- **Generation is not deterministic.** It is sampled at temperature 0.1, so
  answers vary slightly between runs and the end-to-end bucket scores move by
  a question or two. Retrieval metrics (Recall@k, coverage) do not — those
  involve no model sampling and should reproduce exactly.

### Using different books

The pipeline is not hard-wired to these two titles, but three files carry
edition-specific knowledge and will need revisiting:

- **`config/books.yaml`** — page counts, exclude windows (back-of-book index,
  front matter), the DeVita watermark pattern, and Gunderson's running-header
  regexes. Chapter boundaries themselves are read from each PDF's embedded
  outline, so that part adapts on its own; the hand-written `chapter_map` is
  only a fallback for a PDF with no outline.
- **`config/organ_systems.yaml`** — maps chapter *numbers* to pathologies.
  Different books, different numbering.
- **`eval/questions.yaml`** — the `expect_pages` ground truth is PDF page
  numbers in these exact editions. Recall figures are meaningless against
  other books until it is rewritten; `eval/coverage.yaml`, which asserts only
  the organ system, survives a change of edition better.

Everything in `rag/` is book-agnostic.

## Scope and limitations

- This is a reference-retrieval tool over two textbooks, **not a clinical
  decision support system**, and it carries whatever those editions say.
- Figures and images are not indexed — captions are, prose is. Answers that
  would depend on reading a figure will abstain.
- Complex multi-column tables extract imperfectly from PDF. Staging tables are
  kept whole and cited, but verify against the book before clinical use.
- **Layer 3 verifies that a citation exists and points at a supplied source.
  It does not verify the citation is the right one.** A model appending `[1]`
  to an invented sentence passes. What carries the weight is layer 2 plus the
  fact that you can open the cited page in one click and check. Attribution
  that would close this properly was built, measured, and removed — the
  reasoning is in `rag/guard.verify_answer` and in the measurements doc.
- Retrieval quality was measured precisely only on tongue cancer (page-level
  ground truth); coverage across the other 13 organ systems is measured at the
  coarser organ-system level.
- `Find` mode can only point at what the pipeline kept. Back-of-book index and
  bibliography pages are excluded by design, so a lookup takes you to where a
  term is *discussed*, never to its index entry.
