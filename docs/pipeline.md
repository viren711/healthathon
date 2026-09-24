# The build pipeline

> For the system as a whole — runtime topology, data structures,
> retrieval internals and the query path — see
> [architecture.md](architecture.md).

How the two PDFs become a searchable, citable index. Six stages, all
CPU-only, all re-runnable. `scripts/run_all.sh` runs the lot.

```
data/*.pdf
   |  01_extract_corpus.py    isolate content, clean, section it
   v                          [parallel: PDF pages across processes]
corpus/books/sections/*.md  + corpus_report.md  (audit trail)
   |  02_build_chunks.py      token-aware chunks that never cross a section
   v                          [parallel: sections across processes]
corpus/books/chunks.jsonl
   |  03_build_index.py       embed -> FAISS + BM25 + corpus centroid
   v                          [parallel: BLAS threads; BM25 fits concurrently]
artifacts/index/
   |  05_tag_pathologies.py   organ system + pathology per chunk (metadata only)
   v
artifacts/index/taxonomy.json  + corpus/by_system/
   |  04_evaluate.py --apply  calibrate the refusal thresholds, write them back
   |  06_eval_coverage.py     retrieval coverage across all 14 organ systems
   v
config/rag.yaml (calibrated)
```

There is no model training anywhere in this pipeline, and none is needed: the
encoders and the generator are used as published. What Stage 4 "trains" is the
handful of numbers the refusal behaviour depends on, fitted to the labelled
questions in `eval/questions.yaml`.

#### Stage 1 — what gets indexed

`selection.mode` in `config/rag.yaml` decides how much of the books enters the
corpus, and it is the single most consequential setting in the project:

| mode | pages kept | chunks | what it is for |
|---|---|---|---|
| `full` (default) | **4,609 of 6,660** | 15,866 | the whole of both books. Needed for the topic locator to find anything, and for the platform to answer beyond tongue cancer. |
| `tongue` | 354 of 6,660 | 988 | the original focused corpus: tongue cancer and the head & neck oncology needed to discuss it. |

In `full` mode nothing is dropped for being off-topic. What *is* still dropped
is material that is noise in either mode and would wreck retrieval: the
back-of-book index (296 pages in DeVita, which match any lexicon constantly),
bibliography pages (934 in DeVita, 543 in Gunderson), and blank pages. That
accounts for the 2,051-page gap between 6,660 and 4,609 — those pages are not
missing, they are deliberately excluded, and `corpus_report.md` shows the
accounting per book.

**`selection.mode` and `guard.topic.scope` must agree.** Scope `corpus` over a
`tongue` index is the dangerous combination: the gate cheerfully admits a
prostate question that the index has nothing to answer with.

#### How the tongue-focused mode selects pages

Selection is deliberately auditable rather than clever:

- **Chapter windows.** DeVita ch. 44–46 (PDF pp. 985–1112) and Gunderson
  ch. 37–46 (PDF pp. 848–1125) — the head & neck sections, located by reading
  the PDFs, recorded in `config/books.yaml`.
- **A whole-book keyword sweep** catches relevant material outside those
  windows, such as DeVita's *Sequelae of Oral Complications* (p. 3671).
- **Page-type classification runs first.** Back-of-book index pages and
  bibliography pages match the tongue lexicon constantly and would wreck
  retrieval, so they are detected and dropped (296 index + 934 reference pages
  in DeVita alone).
- **Cleaning** strips DeVita's `Dr. AVR @ TMH` watermark (on 2129 pages),
  removes running headers, rejoins hyphenated line breaks, drops superscript
  citation markers, and reflows the PDF's hard line wrapping into paragraphs.
- **Tiering** marks each section `core` (about the tongue, by lexicon density
  or heading) or `context` (needed to answer tongue questions: nodal
  management, staging, chemoRT, toxicity, rehab). Both are indexed.

Every run writes **`corpus/books/corpus_report.md`** — page accounting,
what was dropped and why, chapters covered, the sweep's extra pages. Read it
before trusting an index.

Two extraction details that are easy to get wrong and are handled explicitly:

- Gunderson is **two-column**. It must be read in reading order; a
  layout-preserving extraction interleaves the columns and injects figure
  labels mid-sentence.
- Its running header (`CHAPTER 38 Oral Cavity` / `611`) arrives as **two
  separate lines**, which is where the chapter number and the printed page
  number come from. DeVita has no printed page numbers in its text layer, so
  it is cited by PDF page.

---

#### Stages 2–3 — chunks and index

Chunks are ~450 tokens with 80 of overlap, measured with the embedding
model's own tokenizer, and **never cross a section boundary** — so a chunk is
about one thing and its citation is exact. Tables are kept whole with their
caption. Each chunk is embedded with a breadcrumb prefix
(`Gunderson > ch.38 Oral Cavity > ORAL TONGUE CANCERS > Anatomy`) so short
passages stay retrievable.

Retrieval is hybrid — dense FAISS plus BM25, fused with Reciprocal Rank Fusion,
then cross-encoder reranked. BM25 earns its place on exact tokens like `T2N0`,
`p16` and `cisplatin`, which embeddings blur.

`index_meta.json` records the embedding model and a corpus fingerprint, and the
app **refuses to start against a mismatched index** — querying an index built
with a different embedder fails silently otherwise.

---

---

#### Running on CPU

There is no GPU here and no C toolchain, which rules out both vLLM and
building `llama-cpp-python` from source. `scripts/00_fetch_models.sh` therefore
downloads llama.cpp's **prebuilt** Linux binaries and runs `llama-server`,
which speaks the same OpenAI `/v1/chat/completions` protocol — so `rag/llm.py`
is unchanged and the backend stays swappable via `LLM_BASE_URL`.

Model sizing is driven by measurement, not preference. On a 6-core Ryzen 5
5600G (AVX2, no AVX-512), for a ~2.6k-token prompt:

| | prompt prefill | generation | default |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct Q4_K_M | 185 tok/s | 16.4 tok/s | **yes** |
| Qwen2.5-3B-Instruct Q4_K_M | 84 tok/s | 8.7 tok/s | no |

**The 1.5B is the default, and that decision was reversed once.** Measured
first with the original prompt, it answered 10/25 against the 3B's 20/25 --
it simply would not emit citation markers, and layer 3 discarded correct
prose. After the worked-example prompt fix (below), the same model answered
13/14 with **8/8 abstentions and 8/8 refusals**: identical safety behaviour to
the 3B, at roughly a third of the latency.

The lesson is that the earlier comparison was not measuring model capability
at all. It was measuring how well each model guessed an output format that had
only been described to it. Fixing the prompt did not narrow the gap, it
removed it -- so re-run any model comparison after a prompt change rather than
trusting the old numbers.

---

### Section filtering — one pathology at a time

The left-hand selector narrows everything to a single disease area: pick an
organ system, then optionally one pathology inside it ("Head & Neck" → "Oral
cavity & tongue"). Both Ask and Find respect it.

#### How the corpus is partitioned

`scripts/05_tag_pathologies.py` tags every chunk with an `organ_system` and a
`pathology`, using the chapter page ranges in the PDFs' **own embedded
outlines** and the map in `config/organ_systems.yaml`. Chapters are the
publisher's own boundaries, so the partition is auditable and stable — unlike
a keyword heuristic, it does not drift when a lexicon changes. 77 pathologies
across 14 organ systems; the script refuses to run if two of them claim the
same chapter.

**It does not re-embed anything, and that is the point.** Tagging changes
metadata, not `embed_text`, so all 15,866 vectors stay exactly as valid as
they were. A naive "build a corpus per organ system and index each" would have
meant 75 minutes of CPU to produce vectors identical to the ones already on
disk. The whole stage runs in **3 seconds**, and per-pathology centroids are
recomputed from the existing `embeddings.npy`.

`--write-corpora` additionally materialises `corpus/by_system/<system>/
<pathology>.jsonl` for inspection — real per-pathology corpora on disk, sharing
one vector index.

Two places where the tagging follows pathology rather than the book's own
contents page, both deliberate and commented in the config:

- Gunderson files **thyroid cancer** under head and neck, because that is
  where it is irradiated. It is tagged endocrine here, because that is what it
  is. Same for its skin chapters (47/48).
- DeVita covers the whole of head and neck in one chapter, so there is no
  chapter boundary to split the tongue out with. Stage 1 had already scored
  every section against the tongue lexicon and marked the tongue-specific ones
  tier `core`; the tagger reuses that judgement rather than inventing a second.

#### How filtering works at query time

Scoping is applied **before** the top-k cut, not as a filter afterwards.
Filtering afterwards would return whatever survived out of a corpus-wide top
50, which for a 165-chunk pathology is usually nothing.

FAISS cannot restrict a search to a subset of ids without an `IDSelector`, so
the scoped path instead takes an exact dot product over the selected rows of
`embeddings.npy` — memory-mapped, and only the selected rows are touched. At
16k chunks that is faster than building a selector, and it is exact: measured,
a scoped search returns the identical best chunk and score as an unscoped one
for a question that belongs to the scope.

Asking a foreign question inside a scope is handled by the grounding gate
rather than a new rule: "how is prostate cancer treated" scored against the
oral-cavity chunks tops out at **-2.81**, far below the 2.43 bar, so the
platform says the answer is not in *this section* and offers to widen the
search. In Find mode a scoped miss additionally reports where the topic does
appear.

---
