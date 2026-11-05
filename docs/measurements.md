# Measurements

Every number here was measured on the target machine -- an AMD Ryzen 5 5600G
(6 cores / 12 threads, AVX2, no AVX-512), 14 GB RAM, no GPU, Ubuntu 24.04.

It includes the things that were measured and then **rejected**, because a
tempting optimisation that turns out to be wrong is worth more to the next
person than a clean list of the ones that worked.

### Measured on this machine

AMD Ryzen 5 5600G (6 cores / 12 threads, AVX2), 14 GB RAM, no GPU, Ubuntu
24.04. Build stages, wall-clock:

| stage | tongue corpus | full-book corpus |
|---|---|---|
| 1. extract 6,660 PDF pages | **18 s** (12 processes; was ~8 min single-threaded) | **~50 s** → 9,541 sections |
| 2. chunk sections | **9 s** (81 s of CPU across 12 processes) | **~30 s** → 15,866 chunks |
| 3. embed + BM25 | **4.5 min** (6 BLAS threads) | **~75 min** |
| 4. evaluate the question set | **1.8 s/question** incl. rerank | **0.9 s/question** |
| 5. tag organ system & pathology | — | **3 s** (metadata only, no re-embedding) |

Stage 1's parallel rewrite was verified to reproduce the previous
single-threaded corpus **byte for byte** — 568 sections, 988 chunks, identical
manifest and report — before the corpus was widened to full-book mode.

Embedding is the only stage where full-book mode really costs: 16x the chunks
at roughly 3.7 chunks/s on six cores. It is also the only stage you rarely
re-run, and `scripts/run_all.sh` overlaps it with the model downloads.

---

#### Full-book index (15,866 chunks, the current default)

| check | result |
|---|---|
| Recall@5 on the 25 answerable questions | **92%** (23/25) |
| MRR | 0.798 |
| Retrieval latency | **1.7 s/question** (embed + FAISS + BM25 + rerank) |
| Off-topic refused | **10/10** |
| False refusals on answerable | **0/25** |
| In-scope non-tongue wrongly refused | **0/12** |

Recall drops from 96% to 92% moving from 988 to 15,866 chunks, which is the
price of searching 16x more text for the same question. The scope numbers are
the ones that changed the product: before full-book indexing, every one of
those 12 non-tongue questions was refused outright.

**Find mode over all 50 queries in `eval/queries.txt`:** 41 return locations, 9
return nothing, at **3.6 s/query** with no LLM. All five genuinely off-topic
queries return zero places. Two of the "should abstain" queries still return
locations, correctly: asking where *proton therapy* is discussed is a
reasonable lookup even when the books cannot tell you what it costs in India.
The locator reports where a subject appears; it never claims to answer.

---

#### Tongue-only index (988 chunks), for comparison

Retrieval quality on `eval/questions.yaml`:

| reranker | Recall@5 | MRR | latency/question |
|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` (22M) | **96%** (24/25) | 0.806 | ~1.8 s |
| `bge-reranker-base` (278M) | 92% (23/25) | 0.880 | ~13.9 s |

---

### Coverage across the whole corpus

`eval/questions.yaml` measures Recall@k precisely, but only on tongue cancer,
because precise measurement needs PDF pages read off the source by hand and
that does not scale to 77 pathologies. Retrieval over the other 13 organ
systems was therefore unmeasured — a poor position for a platform claiming to
answer across clinical oncology.

`scripts/06_eval_coverage.py` closes that with weaker but automatable ground
truth: a prostate question must retrieve chunks tagged `gu_prostate`. The tags
come from the publishers' own chapter boundaries, so nothing is a judgement
call. 38 questions spanning every organ system plus radiobiology and physics:

| check | result |
|---|---|
| correct section is the **top** hit | 31/38 (82%) |
| correct section **within top-5** | **38/38 (100%)** |
| scoped search stayed inside its section | 38/38 |

The seven top-1 misses are mostly benign overlaps — an osteosarcoma question
landing in `paed_sarcoma`, a spinal-cord dose question landing in a pancreas
chapter that tabulates the same constraint. All seven still return the right
section inside the top 5.

---

#### The 1.5B transcribes; that is why the 3B is the default

Asked "how are early T1 and T2 oral tongue cancers treated", the 1.5B returned
a correct, fully cited answer that was 82% **verbatim source text**. It reads
as a page of the book rather than an answer to the question.

Measured as the share of an answer's 8-word windows appearing verbatim in its
own sources:

| question | 1.5B | 3B |
|---|---|---|
| early T1/T2 oral tongue treatment | 82% | **36%** |
| localised prostate radiation | *not answered* | **8%** |
| causes of skin cancer | 68% | **31%** |
| cervical chemoradiation | *not answered* | **42%** |

Prompt engineering did not move it. The rewritten prompt leads with "answer
the question that was asked", forbids transcription in as many words, and
carries a worked example contrasting a good answer against a bad one that is
accurate, cited and still a failure because it copies. The 1.5B kept copying;
the 3B synthesises.

That is not surprising in hindsight. A small model told to cite every sentence
and invent nothing satisfies both constraints most cheaply by copying — it is
the locally optimal response to the rules it was given, and only a model with
enough capacity to restate material safely does anything else. If answers read
like the textbook rather than like answers, the fix is a bigger model, not a
firmer instruction.

---

#### Choosing between them

Re-measured after the grounding threshold came down, on the 25 precise plus 12
broad questions in `eval/questions.yaml`:

| | answered | precise | broad | verbatim copying | per question |
|---|---|---|---|---|---|
| **3B** (default) | **32/37** | 23/25 | **9/12** | **8-42%** | ~34 s |
| **1.5B** | 29/37 | 22/25 | 7/12 | 68-82% | ~22 s |

The 3B's whole advantage is on **broadly-phrased** questions — exactly where
the smaller model writes a correct, grounded answer and omits the citation
markers, so layer 3 discards it. "what is brachytherapy" is the clean example:
retrieval scores 8.3 either way, and only the 3B gets it past layer 3.

Switching is one flag, no config edit — both are presets in
`config/models.ini`, and `LLM_MODEL` is set to name the one to load:

```bash
./start --model 1.5b     # smaller and ~2x faster, but it transcribes
```

Speed is the only thing the 1.5B wins, and it was the default only while
latency was the complaint. Answer quality outranks 12 seconds, especially with
streaming putting text on screen at ~10 s either way.

Prefill dominates total latency, which is why `final_top_k` and
`support_margin` are latency knobs as much as quality ones.

---

#### Where the time actually goes

Profiled on a real question (2,203-token prompt, 252-token answer):

| stage | time | share |
|---|---|---|
| embed the query | 50 ms | 0.1% |
| topic gate | 1 ms | ~0 |
| hybrid search + rerank | 1,864 ms | 3.5% |
| **LLM prefill** | 25,436 ms | 48% |
| **LLM generation** | 25,469 ms | 48% |

Retrieval is not the problem and never was; 96% of the wait is the model.
That ruled out a whole class of tempting optimisations. Things that were
measured and did **not** help: flash attention (`-fa`), a single server slot
(`--parallel 1`), and batch sizes from 512 to 2048 -- prefill stays at 84-96
tok/s across all of them, because the CPU backend is compute-bound.

What did help:

- **A smaller model** — the only change that moves both halves at once.
- **Streaming** (`llm.stream`) — does not make generation faster, but the user
  sees text after ~10 s instead of a blank spinner for ~25 s.
- **Prefix caching** — llama.cpp reuses the shared system-prompt prefix across
  requests (measured: 7.9 s → 4.3 s on the second call), so `./start` fires one
  warm-up request at boot to pay for those ~700 tokens before anyone asks.

End-to-end now: **~22-24 s**, with the first words on screen at ~10 s. It was
~50 s with the 3B and no streaming. Batch-size and thread tuning were measured and made no difference
(prefill stayed at 84–88 tok/s across `-b`/`-ub` from 512 to 2048).

Offloading to the 5600G's integrated Vega via llama.cpp's Vulkan build was
tried and is **slower than the CPU** — 32s against 12s on the same prompt — so
it is not used.

---

#### Reranking fewer candidates is faster *and* better

Retrieval was 1.8 s per question, of which the cross-encoder was **94%** (1657
of 1758 ms). Sweeping it on the full index:

| candidates | max_len | Recall@5 | ms/query |
|---|---|---|---|
| 30 (was the default) | 512 | 92% | 1799 |
| **15 (now)** | 512 | **96%** | **917** |
| 10 | 512 | 96% | 584 |
| 30 | 256 | 88% | 921 |
| 20 | 256 | 88% | 605 |

Cutting `rerank_candidates` from 30 to 15 halved the time **and** gained four
points of recall. A 22M MiniLM trained on web passages starts promoting
spurious matches once asked to rank a long tail — RRF had already ordered
those better than the reranker can. So this is not a speed/quality trade;
past about 15 candidates the reranker was actively making retrieval worse.

Truncating the reranker's input to 256 tokens is the opposite: it buys the
same speed and costs 4 points of recall. `reranker.max_length` is pinned at
512 with a comment saying so, because it is a tempting knob.

MiniLM is both more accurate at k=5 and 7x faster here, so it is the default.
bge-reranker-base ranks the single best chunk higher (better MRR) but loses
two answers out of the top 5 — and at k=5 it is recall that feeds the prompt.

Topic gate: **10/10** off-topic questions refused, **0/25** false refusals on
answerable ones.

End to end on the full-book index, with Qwen2.5-1.5B-Instruct Q4_K_M
generating and all three refusal layers live
(`python scripts/04_evaluate.py --with-llm`):

| bucket | required behaviour | result |
|---|---|---|
| 25 answerable (precise) | answer, with citations | **22 answered**, 3 abstained |
| 12 answerable (broad phrasing) | answer, with citations | **7 answered**, 5 abstained |
| 12 in-scope, non-tongue | must not be refused as off-scope | **12/12** (9 answered, 3 abstained) |
| 8 unanswerable | abstain | **8/8 abstained** |
| 10 off-topic | refuse | **9/10 refused**, 1 abstained instead |
| | | **58/67 overall** |

Generation is sampled at temperature 0.1, so these move by a question or two
between runs (the precise bucket has scored 20, 22 and 23 of 25 on identical
config). Treat single-question differences as noise; the bucket-level shape is
what is stable.

Every remaining shortfall is a *safe* failure — the platform said "not
available" where it could have answered, never the reverse. The broad-bucket
misses are layer 3 discarding correct, grounded prose that carried no `[n]`
markers: a 1.5B limitation, not a retrieval one ("what is brachytherapy"
retrieves at 8.3 and is then thrown away for having no brackets).

Asking the model to add the missing markers was tried — re-sending the SOURCES
with the draft and asking for a rewrite — and **measured not to work**: 0 of 3
known cases recovered, at 30-60 extra seconds each. The code survives behind
`guard.citation.repair_attempts: 0` for use with a larger model. A model that
will not cite the first time does not cite when asked again.

**~22-24 s** per question, first words on screen at ~10 s.

The six shortfalls are all *safe* failures — the platform said "not available"
where it could have answered, never the reverse. One is a genuine retrieval
miss ("Which nerve supplies motor function to the tongue?" scores 0.30 against
a 2.97 bar, so the grounding gate refuses before the LLM is called); the rest
are the 3B model declining to commit. That asymmetry is the design working:
every layer can only ever move an answer towards refusal.

---

#### The citation bug that full-book mode exposed

Worth recording because nothing in the pipeline complained. `books.yaml` gave
DeVita a hand-written `chapter_map` covering chapters 44-46 — correct and
sufficient while only the head & neck pages were indexed. In full-book mode
every page outside that range resolved to *no* chapter, and
`split_into_sections` only updated the current chapter when the new value was
not `None`. So once a chapter was set it was never unset: **6,658 of 8,918
DeVita chunks (75%) were cited as "ch.46 Rehabilitation After Treatment of
Head and Neck Cancer"**, including pages about prostate cancer and melanoma.

Retrieval was unaffected and every gate still passed, so no existing test
noticed. A citation that names the wrong chapter is worse than no citation:
it looks checkable and sends the reader to the wrong part of the book.

Both causes are fixed — Stage 1 now derives chapter boundaries from each PDF's
**own embedded outline** (145 chapters for DeVita, 91 for Gunderson) instead
of a hand-maintained list, and the sticky-chapter comparison no longer skips
`None`. `smoke_test.py` now cross-checks sampled citations against the outline,
which fails loudly on the old index and passes on the new one.

The general lesson: a lookup table that covers only the subset you currently
use becomes silently wrong the moment the subset widens. Prefer deriving
structure from the source when the source carries it.

**Two further fixes worth recording**, because both looked like threshold
problems and neither was:

1. *Gate threshold reused as a passage filter.* Raising `min_rerank_score` to
   its calibrated 2.97 silently cut the prompt to one source on near-ties,
   and the model declined to answer from a single uncorroborated passage.
   Separating the two decisions (`support_margin`) fixed it.
2. *The 3B model would not cite.* It wrote correct, grounded prose with no
   `[n]` markers at all, and layer 3 discarded every word of it — working
   exactly as designed, and invisible as anything but "not available". A
   worked example in the system prompt fixed it; 4 of 5 known failures
   recovered immediately. Small models need the format shown, not described.

Both were only findable by looking at the raw model output. If coverage drops
after a model or threshold change, read what the model actually returned
before touching any number.

---

---

### Things that were measured and deliberately NOT changed

Kept here because each is a tempting knob, and turning it is worse:

- **`final_top_k` stays at 5.** Sweeping it: k=3 gives 84% recall at ~11 s
  prefill, k=4 92% at ~14 s, **k=5 96% at ~17 s**, k=6 96% at ~20 s. Six costs
  three seconds for nothing; four saves three seconds and loses four points.
- **`reranker.max_length` stays at 512.** Halving it halves reranking time and
  costs 4 points of Recall@5.
- **Citation repair stays off** (`repair_attempts: 0`). Re-sending the sources
  and asking the model to add the markers it omitted recovered 0 of 3 cases at
  30-60 s each on the 1.5B.
- **Sentence-level attribution was built and then removed.** The idea was to
  assign `[n]` by checking which source actually supports each sentence,
  which would both recover uncited answers and be a stronger check than the
  current one. It does not work with the models here: scoring sentences
  against sources with the cross-encoder, true sentences drawn from the
  corpus scored −2.67 to 5.14 while fabricated ones reached −1.90, so a
  plausible invention outscored a genuine quotation. A cross-encoder answers
  "is this passage relevant to this text", not "does this passage support this
  claim". Shipping it would have stapled citations onto claims the books do
  not make. `rag/guard.verify_answer` carries the reasoning and the numbers.

**Known limit this exposed:** layer 3 verifies that a citation marker exists
and points at a supplied source. It does not verify the marker is the *right*
one. A model appending `[1]` to an invented sentence passes. What actually
carries the weight is layer 2 — the sources in the prompt are the ones
retrieval scored as relevant — and the fact that the reader can open the cited
page in one click and check. An entailment model would close this properly;
there is not one in this stack.
