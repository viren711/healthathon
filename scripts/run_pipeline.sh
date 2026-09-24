#!/usr/bin/env bash
# Build the data pipeline from the source PDFs: corpus -> chunks -> index.
#
#   bash scripts/run_pipeline.sh
#   WORKERS=4 bash scripts/run_pipeline.sh
#
# All three stages parallelise across cores. Stage 3 uses the GPU if one is
# visible and falls back to CPU (a few minutes at this corpus size) if not.
#
# For the full build -- models, reranker bake-off, threshold calibration and
# the smoke test -- use scripts/run_all.sh instead.

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; fi

WORKERS="${WORKERS:-$(nproc)}"

echo "=== Stage 1/3: extracting and sectioning the PDFs ==="
"$PYTHON" scripts/01_extract_corpus.py --workers "$WORKERS" "$@"

echo
echo "=== Stage 2/3: chunking ==="
"$PYTHON" scripts/02_build_chunks.py --workers "$WORKERS"

echo
echo "=== Stage 3/3: embedding + index ==="
"$PYTHON" scripts/03_build_index.py

echo
echo "Done. Review corpus/books/corpus_report.md, then:"
echo "  $PYTHON scripts/04_evaluate.py --apply  # calibrate + write the thresholds"
echo "  streamlit run app/streamlit_app.py # start the portal"
