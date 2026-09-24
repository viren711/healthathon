# Running and testing it

#### Checking it without a GPU or an LLM

`scripts/smoke_test.py` exercises the whole stack against a small index and a
stubbed model server, so the plumbing can be verified on a laptop:

```bash
# a deliberately tiny index -- for testing only
python scripts/03_build_index.py \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --device cpu --limit 120 --out-dir /tmp/smoke_index

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
INDEX_DIR=/tmp/smoke_index RERANKER_ENABLED=false \
    python scripts/smoke_test.py
```

It checks that hits carry citations and page numbers, that a stale index is
rejected, that each of the three layers refuses what it should, and that an
off-topic question never reaches the LLM. `EMBEDDING_MODEL`, `INDEX_DIR` and
`RERANKER_ENABLED` are env overrides for `config/rag.yaml`; an index built with
`--limit` is stamped `partial` and the portal warns when one is loaded.

Thresholds live in `config/rag.yaml` and are **calibrated, not guessed**:

```bash
python scripts/04_evaluate.py --apply     # writes eval/report.md AND the thresholds
```

`eval/questions.yaml` holds three buckets — answerable (with the PDF pages the
answer should come from), tongue-cancer-but-not-in-the-books (must abstain),
and off-topic (must refuse). The evaluator reports Recall@k and MRR, topic-gate
refusal accuracy and false refusals, then sweeps `min_rerank_score` and writes
the best value back into the config. Add `--with-llm` to exercise generation
and layer 3 too.

`--apply` deliberately refuses to write the *topic-gate* cosine thresholds when
the in-scope and off-topic score distributions overlap. On this corpus they do
(in-scope questions bottom out at 0.594, off-topic ones reach 0.647), so no
single cutoff separates them and a "midpoint" would only trade false refusals
for leaks. The lexicon rules carry that layer instead — and measurably do:
10/10 off-topic refused, 0/25 false refusals. Only `min_rerank_score`, which
*does* separate cleanly, is written.

---

---

### Tuning notes

**Swap the embedding model** in `config/rag.yaml` and rebuild. A biomedical
encoder (`NeuML/pubmedbert-base-embeddings`) may beat the general-purpose
default here — run `04_evaluate.py` on both and let Recall@k decide rather
than assuming.

**Swapping the reranker needs no reindex**, because cross-encoder scores are
computed at query time and never touch the stored vectors. Compare candidates
directly against the index you already have, in parallel:

```bash
python scripts/04_evaluate.py --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 --out eval/a.md &
python scripts/04_evaluate.py --reranker BAAI/bge-reranker-base              --out eval/b.md &
wait
```

Note that `min_rerank_score` is on a *model-specific logit scale* — 2.97 for
MiniLM is not the same bar as 0.92 for bge-reranker-base. Re-run `--apply`
after any reranker change, or the guard silently uses the wrong threshold.
`index_meta.json` records which reranker the index was calibrated against.

**Model choice for generation matters** more than usual, because the whole
refusal design depends on the model honouring "cite every fact, or emit the
sentinel". A weaker model fails layer 3 more often, which is safe (you get
"not available") but reduces coverage — so measure it with
`04_evaluate.py --with-llm` rather than assuming.

**If the app abstains too often**, lower `guard.grounding.min_rerank_score`;
too freely, raise it. Re-run the evaluator after either change.
