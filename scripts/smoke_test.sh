#!/usr/bin/env bash
# Verify that Stages 3-4 and the guard layers actually run, WITHOUT building
# the real index. Uses tiny models and a slice of the corpus, and writes to a
# throwaway index directory so artifacts/index/ is left alone.
#
#   bash scripts/smoke_test.sh
#
# This proves the code paths work. It does NOT produce a usable index --
# run scripts/run_pipeline.sh on the GPU server for that.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; fi

# ~130 MB and ~90 MB respectively, vs ~1.3 GB + ~2.2 GB for the real pair.
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-small-en-v1.5}"
export RERANKER_MODEL="${RERANKER_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
export INDEX_DIR="${INDEX_DIR:-artifacts/index_smoke}"
LIMIT="${LIMIT:-200}"

echo "=== smoke: building a $LIMIT-chunk index in $INDEX_DIR ==="
"$PYTHON" scripts/03_build_index.py --device cpu --limit "$LIMIT" --batch-size 16

echo
echo "=== smoke: retrieval, guards and the stubbed-LLM pipeline ==="
"$PYTHON" scripts/smoke_test.py

echo
echo "=== smoke: evaluator and threshold sweep ==="
"$PYTHON" scripts/04_evaluate.py --out eval/report_smoke.md || true

echo
echo "Smoke test finished. Recall numbers here are meaningless -- the index"
echo "holds only $LIMIT of the corpus's chunks and uses a smaller embedder."
echo "What it verifies is that every stage runs and the guards fire."
