# Technical architecture

How the system is put together: the data it starts from, how that data is
turned into a searchable index, and what happens to a query from keystroke to
cited answer.

This document covers the **Knowledge Assistant** — the textbook-grounded part
of the platform. The portal also hosts the **Content Studio** (article
drafting, [content-studio.md](content-studio.md)) and the **Research
Platform** (literature answers via Scite, below); all three share the process
layout and the model server described in §1.

Everything here runs on one machine with no GPU. That constraint shaped most
of the design decisions, and where it did, the reasoning is stated.

---

## 1. System overview

Three processes, started and torn down together by `./start`:

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │ ./start  (supervisor; owns the process groups, dies with the shell)  │
 └──────────────────────────────────────────────────────────────────────┘
        │                                             │
        │ spawns                                      │ spawns
        ▼                                             ▼
 ┌──────────────────────────┐              ┌─────────────────────────────┐
 │ llama-server   :8000     │              │ streamlit      :8501        │
 │ router, --models-max 1   │◀── HTTP ─────│ app/streamlit_app.py        │
 │ presets config/models.ini│  /v1/chat/   │  home · Knowledge Assistant │
 │ 6 gen threads, 12 prefill│  completions │  Content Studio · Research  │
 └──────────┬───────────────┘              │  imports rag/, writer/      │
            │ loads on demand,             └─────────────────────────────┘
            │ one at a time                             │
            ▼                               loads once, cached in RAM
 ┌──────────────────────────┐                           ▼
 │ model child process      │              ┌─────────────────────────────┐
 │ Qwen2.5-3B  (Ask mode)   │              │ artifacts/index/            │
 │   — or —                 │              │  FAISS · BM25 · embeddings  │
 │ Qwen3.5-4B  (Studio)     │              │  chunks · centroids · tax.  │
 └──────────────────────────┘              └─────────────────────────────┘
```

The model server runs in llama.cpp's **router mode**: every request names a
model, and the router starts that model in a child process, stopping the
previous one first because at most one may be resident. Asking a question
loads the 3B reference model; drafting an article swaps in the 4B writer.
Swaps cost ~10 s into the writer and ~4 s back, and nothing else needs to
coordinate them. The memory reasoning is in
[content-studio.md](content-studio.md#one-model-in-memory-at-a-time).

The **Research Platform** is the one tool that leaves the machine. Its page
sends the question to scite.ai (the Assistant API, falling back to the MCP
literature search) and then asks the local reference model, through the same
router, to summarise the returned abstracts and excerpts with `[n]` citations.
The API keys stay server-side in `.env`.

The retrieval stack is a **library, not a service**. `rag/` is imported
directly by the Streamlit process; there is no HTTP hop between the UI and
retrieval. Only the generator is a separate process, and only because
llama.cpp is a C++ binary.

That is deliberate. A single user on one machine gains nothing from a
microservice boundary and pays for it in latency and operational surface.
`rag/pipeline.answer_question()` is the only entry point the UI touches, so
putting an HTTP API in front later is a wrapper, not a refactor.

### Component responsibilities

| module | responsibility | depends on |
|---|---|---|
| `rag/config.py` | loads YAML + `.env`, applies env overrides, `lru_cache`d | — |
| `rag/retriever.py` | hybrid search, reranking, scoped search | FAISS, BM25, sentence-transformers |
| `rag/guard.py` | the three refusal layers | retriever (lazy, to avoid a cycle) |
| `rag/generate.py` | prompt construction, answer-length fitting | config |
| `rag/llm.py` | OpenAI-compatible HTTP client, streaming | stdlib only |
| `rag/pipeline.py` | orchestrates gate → retrieve → gate → generate → verify | all of the above |
| `rag/locate.py` | topic locator (Find mode) | retriever |
| `rag/pdfview.py` | renders source PDF pages | PyMuPDF |
| `writer/studio.py` | Content Studio: questions, article prompt, tidy-up | `rag/llm.py` |
| `rag/scite.py` | Research Platform: Scite Assistant API and MCP search clients | stdlib only |
| `app/streamlit_app.py` | entrypoint: top navigation bar and shared layout CSS | — |
| `app/views/*.py` | one file per page, view layer only — no retrieval or prompt logic | `rag/`, `writer/` |

`rag/llm.py` has **no third-party dependencies**: it speaks
`/v1/chat/completions` over `urllib`. That is what lets the backend be swapped
between llama.cpp, vLLM, Ollama or a hosted endpoint by changing one
environment variable.

---

## 2. Data layer

### Source data

| | DeVita | Gunderson |
|---|---|---|
| Pages | 4,360 | 2,300 |
| File size | 54.3 MB | 309.9 MB |
| Layout | single column | **two column** |
| Printed page numbers in text layer | no | yes |
| Quirks | `Dr. AVR @ TMH` watermark on 2,129 pages | running header split across two lines |

Both are copyrighted and not redistributed. The layout difference matters:
Gunderson must be extracted in **reading order**, because a layout-preserving
extraction interleaves the two columns and injects figure labels mid-sentence.

### Derived artefacts

| path | what | size |
|---|---|---|
| `corpus/books/sections/*.md` | 9,482 cleaned sections, YAML front matter + prose | 164 MB |
| `corpus/books/corpus_manifest.jsonl` | one record per section, no text | — |
| `corpus/books/chunks.jsonl` | 15,844 chunk records | 55 MB |
| `corpus/books/corpus_report.md` | page accounting audit trail | — |
| `artifacts/index/faiss.index` | `IndexFlatIP`, 15,844 × 768 float32 | 48.7 MB |
| `artifacts/index/embeddings.npy` | the same vectors, memory-mapped for scoped search | 48.7 MB |
| `artifacts/index/bm25.pkl` | pickled `BM25Okapi` | 23.7 MB |
| `artifacts/index/chunks.jsonl` | copy of the chunks, so index and text cannot drift | 55 MB |
| `artifacts/index/centroid.npy` | corpus centroid, for the topic gate | 3 KB |
| `artifacts/index/pathology_centroids.json` | 78 per-pathology centroids | 1.3 MB |
| `artifacts/index/taxonomy.json` | organ system → pathology tree with counts | 11 KB |
| `artifacts/index/index_meta.json` | provenance and integrity | 545 B |

The chunks are **copied next to the index** rather than referenced. Index
position *i* is FAISS id *i* is BM25 document *i* is `chunks[i]`; that
ordering contract is the whole addressing scheme, and keeping the text beside
the vectors is what stops it drifting when the corpus is rebuilt.

### The chunk record

```jsonc
{
  "chunk_id": "c4b4095b7959d7c7",        // sha1(section_id|i|text[:120])[:16]
  "section_id": "devita_p0127_applications-of-molecular-diagnostics...",
  "text":       "Molecular diagnostics is playing an increasing role ...",
  "embed_text": "DeVita > ch.2 - Molecular Methods in Cancer > APPLICATIONS ...\n\n<text>",
  "breadcrumb": "DeVita > ch.2 - Molecular Methods in Cancer > APPLICATIONS ...",
  "book": "...", "book_short": "DeVita",
  "chapter": "2 - Molecular Methods in Cancer",
  "section_path": ["APPLICATIONS OF MOLECULAR DIAGNOSTICS IN ONCOLOGY"],
  "pdf_page_start": 127, "pdf_page_end": 127,
  "printed_page_start": null, "printed_page_end": null,
  "tier": "general",                     // core | context | general
  "citation": "DeVita, ch.2 - Molecular Methods in Cancer, PDF p.127",
  "is_table": false,
  "n_tokens": 414,
  "pathology": "general",                // added by stage 5
  "organ_system": "general"
}
```

`text` and `embed_text` are **different on purpose**. `embed_text` carries the
breadcrumb prefix so a short passage keeps its context in the vector — "the
first echelon nodes are level IB and II" is nearly meaningless embedded alone,
and highly specific embedded under `Gunderson > ch.38 Oral Cavity`. The LLM
prompt is given `text`, with the breadcrumb shown separately in the source
header, so the model never sees the same string twice.

---

## 3. Preprocessing pipeline

### Stage 1 — extraction, cleaning, sectioning

`scripts/01_extract_corpus.py`. Two passes.

**Pass 1: page text, parallel.** `ProcessPoolExecutor` over contiguous page
blocks; each worker opens its own `fitz.Document` because PyMuPDF documents
cannot cross a process boundary. Blocks are sized at `total // (workers * 4)`
so that a dense chapter at the end of one worker's range does not leave the
other eleven idle. Results are reassembled in page order. **6,660 pages in
~50 s** on 12 workers, against ~8 minutes single-threaded.

**Pass 2: classify and section, sequential.** It carries cross-page state
(running-line detection, chapter tracking, open section buffers), and is cheap
once the text is in memory.

Per page, in order:

1. **Chapter and printed page** from the running header. Gunderson's header
   arrives as *two separate lines* in reading order (`CHAPTER 38 Oral Cavity`
   then `611`), so the label line is matched first and the bare number beside
   it is taken as the printed page.
2. **Chapter boundaries** come from the **PDF's own embedded outline** — 145
   chapters for DeVita, 91 for Gunderson. A hand-written `chapter_map` in
   `books.yaml` is only a fallback for a PDF with no outline. (This replaced a
   partial hand-written map that covered chapters 44–46 and silently
   mislabelled 75% of DeVita's citations once the corpus went full-book.)
3. **Page classification** → `body | index | references | figure_only | blank`,
   by line-shape ratios. Back-of-book index pages match any clinical lexicon
   constantly and would wreck retrieval; bibliography pages are noise.
   Dropped: 296 index + 934 reference pages in DeVita alone.
4. **Cleaning** — strip watermark, rejoin hyphenated line breaks, remove
   superscript citation runs (`...nodes.12 Lymphatic`) while preserving
   acronyms with digits (`p16`, `HPV16`, `T2N0`), collapse whitespace.
5. **Reflow** — undo the PDF's hard wrapping into real paragraphs, keeping
   table and figure captions on their own lines.
6. **Selection** — in `full` mode every prose page is kept; in `tongue` mode a
   page must fall in a chapter window or clear a lexicon sweep.

Sections are then cut at heading boundaries, at chapter changes, **and at any
gap in page numbers** — without that last rule an open section swallows a
distant page pulled in by the sweep and reports a span like `p.1110-3671`,
making every citation in it wrong.

Output: 9,482 sections, ~19.7 M characters, each with a full citation trail.

### Stage 2 — chunking

`scripts/02_build_chunks.py`. `ProcessPoolExecutor` over section files, with
an initializer that loads the HF tokenizer **once per worker** (loading it per
task would dominate). Results are re-sorted by section id, so worker count
cannot change the output — the file is byte-identical at any `--workers`.

Packing rules, all from `config/rag.yaml`:

- Target **450 tokens**, measured with the embedding model's own tokenizer,
  not a word estimate.
- **80 tokens of overlap**, carried as whole trailing sentences, so a fact
  spanning a seam is retrievable from either side.
- **Chunks never cross a section boundary.** A chunk is therefore about one
  thing and its citation is exact.
- **Tables are emitted whole** with their caption, up to 1,200 tokens. A
  staging table split mid-row retrieves as nonsense.
- Chunks under 60 tokens are dropped (20 for tables — a caption plus two rows
  is still a usable source).

Output: **15,844 chunks**, median 347 tokens. ~30 s on 12 workers.

---

## 4. Indexing

### Stage 3 — embed and build

`scripts/03_build_index.py`.

```
chunks.jsonl ──┬─▶ SentenceTransformer(bge-base-en-v1.5)  ──▶ (15844, 768) float32
               │        normalise=True, max_seq=512               │
               │        6 BLAS threads                            ├─▶ faiss.IndexFlatIP
               │                                                  ├─▶ embeddings.npy
               │                                                  └─▶ centroid.npy
               └─▶ BM25Okapi(tokenize)  [concurrent worker]   ──▶ bm25.pkl
```

Three decisions worth stating:

- **`IndexFlatIP`, not an ANN index.** At 15,844 vectors an exact inner-product
  scan takes well under a millisecond. An approximate index would add tuning
  burden and recall risk to save nothing measurable.
- **Vectors are L2-normalised**, so inner product *is* cosine.
- **BM25 is fitted in a concurrent worker process** while embedding runs. They
  share no data, so the sparse index is free.

Threads are set to **physical cores (6)**, not logical (12): transformer
inference on CPU is BLAS-bound and oversubscribing SMT siblings measurably
slows it. This is the long pole — **~75 minutes**, ~3.7 chunks/s.

Why hybrid rather than dense alone: BM25 earns its place on exact tokens that
embeddings blur — `T2N0`, `p16`, `cisplatin`, dose figures. Dense retrieval
handles the paraphrase cases BM25 cannot.

### Stage 5 — organ-system tagging

`scripts/05_tag_pathologies.py`. Assigns each chunk one `pathology` and one
`organ_system` from `config/organ_systems.yaml`, which maps **chapter numbers**
to 78 pathologies across 14 systems. Chapter ranges again come from the PDF
outlines, so the partition is the publisher's own and the script fails loudly
if two pathologies claim the same chapter.

**It does not re-embed, and that is the architectural point.** Tagging changes
metadata, not `embed_text`, so every vector stays valid. Building 78 separate
indexes would have produced vectors identical to the ones already on disk, at
75 minutes each. The stage runs in **3 seconds**, and per-pathology centroids
are recomputed from the existing `embeddings.npy`.

### Integrity and versioning

`index_meta.json` records the embedding model, dimension, normalisation, query
prefix, reranker, chunk count, `partial` flag, build time, and a
`corpus_fingerprint` (sha256 over chunk ids).

`Retriever.load()` **refuses to start** if the configured embedding model
differs from the one in the metadata. Querying an index across embedders
returns plausible, silently wrong results — the one failure mode with no
visible symptom, so it is made fatal.

---

## 5. Retrieval architecture

### Unscoped path

```
query
  │
  ├─▶ embed_query()  bge-base + query prefix, normalised     ~50 ms
  │
  ├─▶ FAISS IndexFlatIP.search(q, dense_top_k=50)            <1 ms
  ├─▶ BM25Okapi.get_scores(tokens) → top 50                  ~44 ms
  │
  ├─▶ Reciprocal Rank Fusion over the two rankings
  │       score(d) = Σ  1 / (rrf_k + rank_i(d) + 1),  rrf_k = 60
  │   rank-based, so cosine and unbounded BM25 scores never have
  │   to be made commensurate
  │
  ├─▶ take top rerank_candidates = 15
  ├─▶ CrossEncoder(ms-marco-MiniLM-L-6-v2).predict(query, embed_text)  ~800 ms
  │
  └─▶ final_top_k = 5 hits, ordered by cross-encoder score
```

`rerank_candidates = 15` is measured, not assumed: 30 candidates gives
Recall@5 92% at 1,799 ms; 15 gives **96% at 917 ms**. Past roughly 15, a 22 M
cross-encoder trained on web passages starts promoting spurious matches over
RRF's existing order — so this is not a speed/quality trade, it is a bad
default corrected.

The reranker scores `embed_text` (breadcrumb + body), so it sees the chapter
context; the LLM is given `text` with the breadcrumb in a separate header.

### Scoped path

When a pathology or organ system is selected, FAISS is bypassed for the dense
leg:

```
rows = index positions for that pathology (precomputed at load)
scores = embeddings[rows] @ query_vec        # exact, on the memory-mapped array
```

FAISS cannot restrict a search to a subset of ids without an `IDSelector`. At
this corpus size an exact dot product over the selected rows is simpler and
faster than building one, and it is *exact*: a scoped search returns the
identical best chunk and score as an unscoped one for a question belonging to
that scope (asserted in `smoke_test.py`).

**Scoping is applied before the top-k cut, never as a post-filter.** Filtering
afterwards would return whatever survived out of a corpus-wide top 50, which
for a 165-chunk pathology is usually nothing.

`embeddings.npy` is memory-mapped: 48 MB on disk, and only the selected rows
are ever touched.

---

## 6. The RAG pipeline — what a query does

`rag/pipeline.answer_question(question, client, scope)`. Every return path
produces an `Answer`, so callers never special-case a refusal.

```
question
   │
   ├─ embed_query() ─────────────────────────────────── once, reused below
   │
   ├─ LAYER 1  guard.topic_gate(question, vec)                    ~1 ms
   │     scope=corpus → junk filter only: refuse iff
   │       seed_sim < 0.45 AND centroid_sim < 0.50
   │     lexicon shortcuts allow first (tongue term, disease site,
   │     oncology vocabulary)
   │     ✗ → status=out_of_scope        ← no retrieval, no LLM
   │
   ├─ retriever.search(question, vec, scope)                    ~890 ms
   │
   ├─ LAYER 2  guard.grounding_gate(hits)                          ~0 ms
   │     top rerank score ≥ min_rerank_score (0.30)?
   │     ✗ → status=not_available, with the rejected hits attached
   │                                    ← still no LLM call
   │
   ├─ guard.filter_supporting(hits)
   │     keep everything within support_margin (1.5) of the best hit
   │     — a SEPARATE decision from the gate; see below
   │
   ├─ generate.build_prompt() → system + user with numbered SOURCES
   ├─ llm.chat_stream()                                    ~10-34 s
   │     Qwen2.5-3B Q4_K_M, temp 0.1, max_tokens 800
   │
   ├─ generate.fit_to_length(raw, max_answer_words=500)
   │     trims at sentence/bullet boundaries only
   │
   ├─ LAYER 3  guard.verify_answer(text, n_sources)
   │     sentinel emitted?  no citation?  citation out of range?
   │     ✗ → status=not_available, model's text DISCARDED
   │
   └─ status=answered, with only the sources actually cited
```

### Why the gate and the filter are separate

`min_rerank_score` is calibrated against the **top** score of each question —
it answers "is there evidence here at all?". Reusing it to *select* passages
discards near-ties: a question whose top five score 2.97 / 2.96 / 2.91 / 2.80
has five relevant passages, not one. Filtering on the gate threshold left the
model a single uncorroborated source, which it then declined to answer from,
turning good retrieval into a false "not available" on 4 of 25 questions.
`support_margin` fixes it — once the gate passes, everything within 1.5 of the
best hit is shown.

### Where the time goes

Profiled on a real question (2,203-token prompt, 252-token answer):

| stage | time | share |
|---|---|---|
| embed query | 50 ms | 0.1% |
| topic gate | 1 ms | ~0 |
| hybrid search + rerank | ~890 ms | ~3% |
| **LLM prefill** | ~25 s | ~48% |
| **LLM generation** | ~25 s | ~48% |

**96% of the wait is the model.** That ruled out a whole class of tempting
optimisations — flash attention, single-slot mode and batch sizes from 512 to
2048 were all measured and none moved prefill off 84–96 tok/s, because the CPU
backend is compute-bound. What worked was a smaller model, streaming, and
prefix-caching the system prompt (`./start` fires one warm-up request at boot).

### Failure and degradation

| failure | behaviour |
|---|---|
| Model server down | `status=error` with the reason; **Find mode still works** |
| Index missing | UI shows a build instruction instead of a traceback |
| Embedding model ≠ index | `IndexMismatchError` at load — fatal, never silent |
| Model emits sentinel / no citation | layer 3 discards the text, user sees the refusal |
| Answer exceeds 500 words | trimmed at a sentence boundary before verification |
| Retrieval finds nothing in scope | "not available *in this section*", offers to widen |

---

## 7. Query modes

Both run over the same index; they answer different questions.

| | **Ask** | **Find** |
|---|---|---|
| Question | "what is the answer?" | "where is this discussed?" |
| LLM | yes | **never** |
| Latency | ~34 s | ~2 s |
| Returns | prose with `[n]` citations | positions in the books |
| Candidates | `final_top_k` = 5 chunks | ~40, grouped into sections |
| Threshold | `min_rerank_score` = 0.30 | `min_locate_score` = 0.0 |
| Can hallucinate | mitigated by 3 layers | structurally impossible |

Find groups by **section, not chunk**: chunks overlap by 80 tokens, so an
ungrouped list shows the same passage three times under three slightly
different page spans. It applies a much lower floor than Ask — but not *no*
floor, because nearest-neighbour search always returns something, and without
one "what is the capital of France?" comes back with five confident page
references. Measured, real topics score +2.1 to +8.3 and unanswerable ones
−9.7 to −2.6, so the floor sits between the two populations.

**Go to section** (both modes) renders the actual PDF page with PyMuPDF at
request time. Rendering rather than linking is deliberate: the books are 54 MB
and 310 MB, so serving one to the browser to jump to `#page=872` means
shipping the whole file, and a `file://` link is blocked from a web page.
Rasterising one page takes a few milliseconds and is `lru_cache`d, since
Streamlit re-runs the script on every interaction.

---

## 8. Configuration and calibration

All behaviour lives in three YAML files; no thresholds are hard-coded.

| file | holds |
|---|---|
| `config/books.yaml` | per-book page counts, exclude windows, watermark patterns, header regexes, lexicons |
| `config/rag.yaml` | models, chunking, retrieval, all guard thresholds, messages, LLM settings |
| `config/organ_systems.yaml` | chapter → pathology → organ system |

Env overrides (`.env` or the environment): `EMBEDDING_MODEL`, `INDEX_DIR`,
`RERANKER_MODEL`, `RERANKER_ENABLED`, `LLM_BASE_URL`, `LLM_MODEL`,
`LLM_API_KEY`, `LLM_TEMPERATURE`.

Thresholds are **fitted, not chosen**. `scripts/04_evaluate.py --apply` sweeps
`min_rerank_score` against the labelled sets and writes the result back into
`config/rag.yaml`, editing the line in place so the comments explaining each
threshold survive.

The calibration objective matters as much as the sweep. The answerable and
unanswerable score populations *overlap* (0.30–8.77 against −4.85–4.75), so no
threshold separates them and a symmetric objective just splits the difference
— trading real answers for abstentions layer 3 would have caught anyway. The
objective is therefore: **the most abstaining threshold that still answers 95%
of answerable questions** (`min_answer_rate`). Layer 2 is a cheap pre-filter;
layer 3 is the guarantee.

---

## 9. Testing architecture

| suite | checks | needs a model? |
|---|---|---|
| `scripts/smoke_test.py` | 65 — guards, citations, scoping, length fitting, PDF rendering, stale-index detection | no (stubbed) |
| `scripts/ui_test.py` | 42 offline, plus live Ask, drafting and Scite checks — home page, Find, Ask, section filter, Content Studio, Research Platform, via Streamlit `AppTest` | Ask, drafting and the live Scite query only |
| `scripts/04_evaluate.py` | Recall@k, MRR, refusal accuracy, calibration | `--with-llm` only |
| `scripts/06_eval_coverage.py` | organ-system coverage, 38 questions | no |

`smoke_test.py` injects a `StubLLM`, so all three refusal layers — including
"an off-topic question never reaches the LLM" (asserted as `calls == 0`) — are
verifiable with no generator at all.

`ui_test.py` exists because pipeline tests structurally cannot catch
interaction bugs. It was written after "Go to section" rendered its page
correctly and then hid it: the button lives inside an expander, clicking a
button re-runs the Streamlit script, and the expander was rebuilt with a
hard-coded `expanded=False`. The page was in the element tree the whole time,
inside a container that had snapped shut.

---

## 10. Known architectural limits

- **Layer 3 verifies that a citation exists and is in range — not that it is
  correct.** A model appending `[1]` to an invented sentence passes. Real
  attribution was built and removed: scoring sentences against sources with the
  cross-encoder does not separate the populations (true sentences −2.67 to
  5.14, fabricated ones reaching −1.90), because a cross-encoder answers "is
  this passage relevant" and not "does this passage support this claim". An
  entailment model is the right tool and is not in this stack. What carries the
  weight instead is layer 2 plus one-click page verification.
- **Single-process, single-user.** No request queue, no concurrency control.
  The model server has one slot and one resident model: two people using the
  Knowledge Assistant and the Content Studio at once would force a model swap
  on every request.
- **The index is immutable at runtime.** Adding a book is a full rebuild; there
  is no incremental path.
- **Recall is measured precisely only on tongue cancer** (page-level ground
  truth); the rest of the corpus is verified at the coarser organ-system level.
- **`BM25Okapi` is unpickled at load** — fine locally, a code-execution vector
  if the index directory were ever untrusted.
