#!/usr/bin/env bash
# Start the local generator: llama.cpp's OpenAI-compatible server.
#
# This box has no GPU and no C toolchain, so we run the prebuilt llama-server
# binary fetched by scripts/00_fetch_models.sh. It exposes
# /v1/chat/completions and /v1/models, which is exactly the protocol
# rag/llm.py speaks -- nothing in the Python code changes.
#
#   bash scripts/serve_llm.sh          # foreground
#   bash scripts/serve_llm.sh &        # background
#
# The 3B is the default because the 1.5B TRANSCRIBES. Measured as the share of
# an answer's 8-word windows appearing verbatim in its sources: 1.5B 68-82%,
# 3B 6-44%. A 1.5B told to cite every sentence and invent nothing satisfies
# that the cheapest way it can -- by copying the passage -- which reads as a
# page of the book rather than an answer. Prompt engineering did not move it.
# Drop to the smaller, ~2x faster model with:
#   GGUF_FILE=Qwen2.5-1.5B-Instruct-Q4_K_M.gguf ALIAS=qwen2.5-1.5b-instruct \
#       bash scripts/serve_llm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="${MODELS_DIR:-models}"
GGUF_FILE="${GGUF_FILE:-Qwen2.5-3B-Instruct-Q4_K_M.gguf}"
MODEL="${MODEL:-$MODELS_DIR/gguf/$GGUF_FILE}"
ALIAS="${ALIAS:-qwen2.5-3b-instruct}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-8192}"

# 6 physical cores. Generation is memory-bandwidth bound, so the 6 extra SMT
# threads slow it down; prompt prefill is compute bound and does benefit, hence
# the split between -t (generation) and -tb (batch/prefill).
THREADS="${THREADS:-6}"
THREADS_BATCH="${THREADS_BATCH:-12}"

BIN="$MODELS_DIR/llama.cpp/llama-server"
if [[ ! -x "$BIN" ]]; then
    echo "llama-server not found at $BIN -- run: bash scripts/00_fetch_models.sh" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "model not found at $MODEL -- run: bash scripts/00_fetch_models.sh" >&2
    exit 1
fi

echo "serving $MODEL as '$ALIAS' on http://$HOST:$PORT/v1"
exec "$BIN" \
    --model "$MODEL" \
    --alias "$ALIAS" \
    --host "$HOST" \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    --batch-size 512 \
    --cache-reuse 256 \
    --no-webui
