#!/usr/bin/env bash
# Stage 0 -- fetch everything the pipeline needs to run offline afterwards.
#
# Three independent downloads, started together and waited on at the end:
#   1. llama.cpp's prebuilt Linux CPU binaries (llama-server)
#   2. the GGUF weights the server loads
#   3. the sentence-transformers encoders, warmed into the HF cache
#
# There is no C compiler or cmake on this machine, so the llama.cpp binary MUST
# come prebuilt -- building llama-cpp-python from source is not an option here.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=python
[[ -x .venv/bin/python ]] && PYTHON=.venv/bin/python

MODELS_DIR="${MODELS_DIR:-models}"
LLAMA_BUILD="${LLAMA_BUILD:-b10964}"
GGUF_REPO="${GGUF_REPO:-bartowski/Qwen2.5-3B-Instruct-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen2.5-3B-Instruct-Q4_K_M.gguf}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
RERANKERS="${RERANKERS:-cross-encoder/ms-marco-MiniLM-L-6-v2 BAAI/bge-reranker-base}"

mkdir -p "$MODELS_DIR/llama.cpp" "$MODELS_DIR/gguf" logs

# --- 1. llama.cpp prebuilt CPU binaries ------------------------------------
fetch_llama() {
    if [[ -x "$MODELS_DIR/llama.cpp/llama-server" ]]; then
        echo "[llama.cpp] already present, skipping"
        return
    fi
    local url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz"
    echo "[llama.cpp] downloading ${LLAMA_BUILD}"
    # --retry-all-errors matters: GitHub's release CDN intermittently fails the
    # TLS handshake (curl exit 35), which plain --retry does not retry on.
    curl -fL --retry 5 --retry-all-errors --retry-delay 2 -o "$MODELS_DIR/llama.tar.gz" "$url"
    tar -xzf "$MODELS_DIR/llama.tar.gz" -C "$MODELS_DIR/llama.cpp" --strip-components=1
    rm -f "$MODELS_DIR/llama.tar.gz"
    # The tarball keeps binaries in build/bin and shared objects alongside them.
    if [[ ! -x "$MODELS_DIR/llama.cpp/llama-server" ]]; then
        local found
        found=$(find "$MODELS_DIR/llama.cpp" -name llama-server -type f | head -1)
        [[ -n "$found" ]] && ln -sf "$(realpath --relative-to="$MODELS_DIR/llama.cpp" "$found")" \
            "$MODELS_DIR/llama.cpp/llama-server"
    fi
    echo "[llama.cpp] done"
}

# --- 2. GGUF weights --------------------------------------------------------
fetch_gguf() {
    if [[ -f "$MODELS_DIR/gguf/$GGUF_FILE" ]]; then
        echo "[gguf] already present, skipping"
        return
    fi
    echo "[gguf] downloading $GGUF_REPO/$GGUF_FILE"
    "$PYTHON" - "$GGUF_REPO" "$GGUF_FILE" "$MODELS_DIR/gguf" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo, filename, dest = sys.argv[1], sys.argv[2], Path(sys.argv[3])
src = hf_hub_download(repo_id=repo, filename=filename)
dest.mkdir(parents=True, exist_ok=True)
# Copy out of the blob cache so the server does not depend on cache layout.
shutil.copyfile(src, dest / filename)
print("saved", dest / filename)
PY
    echo "[gguf] done"
}

# --- 3. Encoders ------------------------------------------------------------
fetch_encoders() {
    echo "[encoders] warming HF cache: $EMBED_MODEL $RERANKERS"
    "$PYTHON" - "$EMBED_MODEL" $RERANKERS <<'PY'
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder

embed, rerankers = sys.argv[1], sys.argv[2:]
SentenceTransformer(embed, device="cpu").encode(["warmup"])
print("cached", embed)
for name in rerankers:
    CrossEncoder(name, device="cpu").predict([("a", "b")])
    print("cached", name)
PY
    echo "[encoders] done"
}

fetch_llama    > logs/fetch_llama.log 2>&1 &
PID_LLAMA=$!
fetch_gguf     > logs/fetch_gguf.log 2>&1 &
PID_GGUF=$!
fetch_encoders > logs/fetch_encoders.log 2>&1 &
PID_ENC=$!

status=0
for spec in "llama.cpp:$PID_LLAMA" "gguf:$PID_GGUF" "encoders:$PID_ENC"; do
    name=${spec%%:*}; pid=${spec##*:}
    if wait "$pid"; then
        echo "  ok    $name"
    else
        echo "  FAIL  $name -- see logs/fetch_${name/llama.cpp/llama}.log"
        status=1
    fi
done

echo
echo "models/ contents:"
du -sh "$MODELS_DIR"/* 2>/dev/null || true
exit $status
