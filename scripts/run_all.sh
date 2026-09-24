#!/usr/bin/env bash
# Build the whole platform from the source PDFs, using every core available.
#
#   bash scripts/run_all.sh                 # full build
#   SKIP_EXTRACT=1 bash scripts/run_all.sh  # reuse the existing corpus
#
# Two lanes run concurrently:
#   A. fetch the generator (llama.cpp binary + GGUF) and the encoders
#   B. PDFs -> corpus -> chunks -> index
# They join for calibration, the smoke test, and the reranker bake-off.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=python
[[ -x .venv/bin/python ]] && PYTHON=.venv/bin/python

WORKERS="${WORKERS:-$(nproc)}"
mkdir -p logs

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# --- Lane A: models, in the background -------------------------------------
step "Lane A: fetching models (background)"
bash scripts/00_fetch_models.sh > logs/fetch_all.log 2>&1 &
LANE_A=$!
echo "  started (pid $LANE_A) -> logs/fetch_all.log"

# --- Lane B: the data pipeline ---------------------------------------------
if [[ -z "${SKIP_EXTRACT:-}" ]]; then
    step "Stage 1: extracting and sectioning the PDFs ($WORKERS workers)"
    "$PYTHON" scripts/01_extract_corpus.py --workers "$WORKERS"
else
    step "Stage 1: skipped (SKIP_EXTRACT set)"
fi

step "Stage 2: building chunks ($WORKERS workers)"
"$PYTHON" scripts/02_build_chunks.py --workers "$WORKERS"

step "Stage 3: embedding and indexing"
MODE=$("$PYTHON" -c "import yaml;print(yaml.safe_load(open('config/rag.yaml')).get('selection',{}).get('mode','tongue'))" 2>/dev/null || echo tongue)
if [[ "$MODE" == "full" ]]; then
    echo "  selection.mode=full: ~16k chunks, expect roughly 75 minutes on CPU."
    echo "  (set selection.mode=tongue in config/rag.yaml for the 5-minute build)"
fi
"$PYTHON" scripts/03_build_index.py

step "Stage 5: tagging chunks by organ system and pathology"
# Metadata only -- this must never re-embed, which is why it is its own stage
# after the index rather than part of Stage 2.
"$PYTHON" scripts/05_tag_pathologies.py --write-corpora

step "Lane A: waiting for models"
if wait "$LANE_A"; then
    echo "  models ready"
else
    echo "  !! model fetch failed -- see logs/fetch_all.log" >&2
    echo "  retrieval will still work; generation will not." >&2
fi

# --- Reranker bake-off: same index, two cross-encoders, in parallel --------
# Swapping the reranker does not invalidate the embeddings, so both candidates
# can be measured against the index that was just built.
step "Stage 4a: comparing rerankers in parallel"
"$PYTHON" scripts/04_evaluate.py --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
    --out eval/report_minilm.md --workers 3 > logs/eval_minilm.log 2>&1 &
P1=$!
"$PYTHON" scripts/04_evaluate.py --reranker BAAI/bge-reranker-base \
    --out eval/report_bge.md --workers 3 > logs/eval_bge.log 2>&1 &
P2=$!
wait "$P1" || true
wait "$P2" || true
for f in eval/report_minilm.md eval/report_bge.md; do
    [[ -f $f ]] && { echo "  $f:"; grep -E "Recall@|MRR:|latency" "$f" | sed 's/^/    /'; }
done

# --- Calibrate the configured reranker and write thresholds back ----------
step "Stage 4b: calibrating thresholds (writes config/rag.yaml)"
"$PYTHON" scripts/04_evaluate.py --apply --workers 4

step "Smoke test: guards, citations, stale-index detection"
"$PYTHON" scripts/smoke_test.py

step "Stage 6: cross-corpus retrieval coverage"
# 04_evaluate measures recall precisely but only on tongue cancer. This checks
# the other 13 organ systems with weaker, automatable ground truth.
"$PYTHON" scripts/06_eval_coverage.py --scoped 2>&1 | tail -6

step "UI test: the portal's buttons actually do something"
"$PYTHON" scripts/ui_test.py --no-llm

step "Done"
cat <<'MSG'
Start the platform:

    ./start

That runs the model server and the portal as children of your terminal, so
closing it stops both. Open http://localhost:8501

Other checks:

    python scripts/ask_batch.py --locate      # all 50 queries, no LLM, seconds
    python scripts/ask_batch.py               # all 50 through generation (slow)
    python scripts/04_evaluate.py --with-llm  # the full labelled evaluation
MSG
