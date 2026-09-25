#!/usr/bin/env bash
# Start the local model server: llama.cpp's OpenAI-compatible server, in
# ROUTER mode, holding at most one model at a time.
#
# This box has no GPU and no C toolchain, so we run the prebuilt llama-server
# binary fetched by scripts/00_fetch_models.sh. It exposes
# /v1/chat/completions and /v1/models, which is exactly the protocol
# rag/llm.py speaks.
#
#   bash scripts/serve_llm.sh          # foreground
#   bash scripts/serve_llm.sh &        # background
#
# Every model the platform uses is declared in config/models.ini. A request
# names the model it wants ("model": "qwen3-4b-instruct"); the router loads it
# on demand and, with --models-max 1, unloads whatever was resident first. That
# is what keeps the Oncology Reference model and the Content Studio writer
# from ever sitting in RAM together, without either page knowing about the
# other.
#
# Which reference model the portal asks for is llm.model in config/rag.yaml
# (or LLM_MODEL). The 3B is the default because the 1.5B TRANSCRIBES: 68-82%
# of its answers' 8-word windows appeared verbatim in the sources, against
# 6-44% for the 3B. ./start --model 1.5b switches anyway.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_DIR="${MODELS_DIR:-models}"
PRESET="${PRESET:-config/models.ini}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
# How many models may be resident at once. 1 is the point: raise it only on a
# machine with the RAM to hold the reference model and a writer together.
MODELS_MAX="${MODELS_MAX:-1}"

BIN="$MODELS_DIR/llama.cpp/llama-server"
if [[ ! -x "$BIN" ]]; then
    echo "llama-server not found at $BIN -- run: bash scripts/00_fetch_models.sh" >&2
    exit 1
fi
if [[ ! -f "$PRESET" ]]; then
    echo "model presets not found at $PRESET" >&2
    exit 1
fi

# Report which declared models are actually on disk. A missing file is not
# fatal -- the router simply cannot load that one -- but it is worth saying.
echo "model presets ($PRESET):"
current=""
while IFS= read -r line; do
    if [[ "$line" =~ ^\[([^]]+)\] ]]; then current="${BASH_REMATCH[1]}"; fi
    if [[ -n "$current" && "$line" =~ ^model[[:space:]]*=[[:space:]]*(.+)$ ]]; then
        path="${BASH_REMATCH[1]}"
        if [[ -f "$path" ]]; then echo "  ok       $current"; else echo "  MISSING  $current  ($path)"; fi
    fi
done < "$PRESET"

# The router also lists any GGUF it finds in the Hugging Face cache, which
# would put stray models on offer. Point its cache lookups at an empty
# directory so only the presets above are served.
EMPTY_CACHE="$MODELS_DIR/.router-cache"
mkdir -p "$EMPTY_CACHE"

echo "router on http://$HOST:$PORT/v1  (max $MODELS_MAX model resident)"
exec env HF_HUB_CACHE="$EMPTY_CACHE" LLAMA_CACHE="$EMPTY_CACHE" \
    "$BIN" \
    --models-preset "$PRESET" \
    --models-max "$MODELS_MAX" \
    --host "$HOST" \
    --port "$PORT"
