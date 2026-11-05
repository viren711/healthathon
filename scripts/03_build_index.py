#!/usr/bin/env python3
"""Stage 3 -- build the retrieval index. This is the pipeline you run on the server.

Embeds every chunk, builds an exact FAISS inner-product index and a BM25
index over the same chunks, and records a manifest so the app can refuse to
run against an index built with a different embedding model.

GPU is used automatically if available; this corpus is small enough that CPU
takes only a few minutes. On CPU the embedding pass is parallelised by the
BLAS threadpool, and the BM25 index -- which needs no embeddings -- is built
concurrently in a worker process rather than after it.

    python scripts/03_build_index.py
    python scripts/03_build_index.py --device cpu --model BAAI/bge-small-en-v1.5
    python scripts/03_build_index.py --threads 6
    python scripts/03_build_index.py --no-reuse      # re-embed everything
    python scripts/03_build_index.py --chunks corpus/patient/chunks.jsonl \
        --out-dir artifacts/index_patient            # the Patient Portal index

By default, vectors already in the existing index are reused for every chunk
whose text is unchanged (same chunk_id and the same embed_text, same model),
so adding a book only embeds the new book's chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def physical_cores() -> int:
    """Physical core count, falling back to half the logical count on SMT.

    torch scales with physical cores for this workload; counting SMT siblings
    oversubscribes the BLAS threadpool and measurably slows embedding down.
    """
    try:
        ids = set()
        current: dict[str, str] = {}
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if not line.strip():
                if "physical id" in current and "core id" in current:
                    ids.add((current["physical id"], current["core id"]))
                current = {}
                continue
            key, _, value = line.partition(":")
            current[key.strip()] = value.strip()
        if ids:
            return len(ids)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _build_bm25(texts: list[str]) -> bytes:
    """Tokenise and fit BM25, returning the pickled index.

    Runs in a worker process alongside the embedding pass. Returning bytes
    rather than the object keeps the pickling explicit and lets the parent
    write it straight to disk.
    """
    from rank_bm25 import BM25Okapi

    return pickle.dumps(BM25Okapi([tokenize_for_bm25(t) for t in texts]))


def tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, keeping staging codes like 't2n0' and 'p16'."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_reusable_vectors(out_dir: Path, model_name: str, normalize: bool, prefix: str):
    """Vectors from the index already in `out_dir`, keyed by chunk_id.

    Returns {chunk_id: (text_hash, vector)}. Empty when there is no previous
    index or it was built with a different model, prefix or normalisation --
    those vectors would not be comparable with new ones.
    """
    import numpy as np

    meta_path = out_dir / "index_meta.json"
    if not (meta_path.exists() and (out_dir / "embeddings.npy").exists()):
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("partial")
        or meta.get("embedding_model") != model_name
        or bool(meta.get("normalize")) != bool(normalize)
        or (meta.get("doc_prefix") or "") != (prefix or "")
    ):
        return {}
    old_rows = [
        json.loads(line)
        for line in (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    vectors = np.load(out_dir / "embeddings.npy")
    if len(old_rows) != len(vectors):
        return {}
    return {
        r["chunk_id"]: (hashlib.sha1(r["embed_text"].encode()).hexdigest(), vectors[i])
        for i, r in enumerate(old_rows)
    }


def corpus_fingerprint(rows: list[dict]) -> str:
    """Stable hash of the chunk set, so a stale index is detectable."""
    h = hashlib.sha256()
    for r in rows:
        h.update(r["chunk_id"].encode())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="override the embedding model")
    ap.add_argument("--device", help="cuda | cpu (default: auto)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument(
        "--limit",
        type=int,
        help="embed only the first N chunks -- for verifying the pipeline runs, "
        "not for building a usable index",
    )
    ap.add_argument("--out-dir", help="write the index here instead of paths.index_dir")
    ap.add_argument("--chunks", help="index this chunks file instead of paths.chunks_file")
    ap.add_argument(
        "--no-reuse",
        action="store_true",
        help="embed every chunk, ignoring vectors in the existing index",
    )
    ap.add_argument(
        "--threads",
        type=int,
        help="torch intra-op threads for CPU embedding (default: physical cores)",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from rag.config import load_config

    cfg = load_config()
    emb_cfg = cfg["embedding"]
    model_name = args.model or emb_cfg["model"]
    batch_size = args.batch_size or emb_cfg["batch_size"]

    chunks_path = Path(args.chunks) if args.chunks else ROOT / cfg["paths"]["chunks_file"]
    if not chunks_path.is_absolute():
        chunks_path = ROOT / chunks_path
    if not chunks_path.exists():
        print(f"missing {chunks_path} -- run 02_build_chunks.py first", file=sys.stderr)
        return 1

    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("chunks file is empty", file=sys.stderr)
        return 1
    if args.limit:
        rows = rows[: args.limit]
        print(f"!! --limit {args.limit}: indexing a SUBSET, not the full corpus")
    print(f"chunks: {len(rows)}")

    # Transformer inference on CPU is BLAS-bound: it scales with *physical*
    # cores, and oversubscribing with SMT siblings makes it slower. Set before
    # torch is imported -- OpenMP reads these at load time.
    threads = args.threads or physical_cores()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    # Imports are local so that Stages 1-2 stay runnable without the ML stack.
    import numpy as np
    import torch
    import faiss
    from sentence_transformers import SentenceTransformer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(threads)
    print(f"embedding model: {model_name}  device: {device}  threads: {threads}")

    # BM25 depends only on the chunk text, so fit it while the GPU/CPU is busy
    # embedding instead of afterwards.
    bm25_pool = ProcessPoolExecutor(max_workers=1)
    bm25_future = bm25_pool.submit(_build_bm25, [r["embed_text"] for r in rows])

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = emb_cfg.get("max_seq_length", 512)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / cfg["paths"]["index_dir"]
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    # Reuse what the previous index already computed; embed only the rest.
    reusable = {}
    if not args.no_reuse and not args.limit:
        reusable = load_reusable_vectors(
            out_dir, model_name, emb_cfg.get("normalize", True), emb_cfg.get("doc_prefix", "")
        )
    slots: list = [None] * len(rows)
    todo: list[int] = []
    for i, r in enumerate(rows):
        hit = reusable.get(r["chunk_id"])
        if hit and hit[0] == hashlib.sha1(r["embed_text"].encode()).hexdigest():
            slots[i] = hit[1]
        else:
            todo.append(i)
    print(f"reusing {len(rows) - len(todo)} vectors, embedding {len(todo)} new chunks")

    t0 = time.time()
    if todo:
        fresh = model.encode(
            [rows[i]["embed_text"] for i in todo],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=emb_cfg.get("normalize", True),
            show_progress_bar=True,
        ).astype("float32")
        for i, vec in zip(todo, fresh):
            slots[i] = vec
    vectors = np.stack(slots).astype("float32")
    print(f"embedded in {time.time() - t0:.1f}s -> {vectors.shape}")

    dim = int(vectors.shape[1])
    if dim != emb_cfg.get("dim"):
        print(f"  note: actual dim {dim} differs from config dim {emb_cfg.get('dim')}; using {dim}")

    # Exact search. At a few thousand chunks an approximate index would add
    # tuning burden and recall risk while saving microseconds.
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "embeddings.npy", vectors)

    # Collect the BM25 index that has been fitting in the background.
    (out_dir / "bm25.pkl").write_bytes(bm25_future.result())
    bm25_pool.shutdown()

    # The chunks are copied next to the index so the two can never drift apart.
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Centroid of the corpus, used by the topic gate to decide whether a
    # question is even in this domain before any retrieval happens.
    centroid = vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    np.save(out_dir / "centroid.npy", centroid.astype("float32"))

    meta = {
        "embedding_model": model_name,
        "embedding_dim": dim,
        "normalize": emb_cfg.get("normalize", True),
        "query_prefix": emb_cfg.get("query_prefix", ""),
        "doc_prefix": emb_cfg.get("doc_prefix", ""),
        "max_seq_length": model.max_seq_length,
        # Recorded for provenance only -- the retriever does not enforce it.
        # Cross-encoder scores are logits on a model-specific scale, so a
        # guard threshold calibrated for one reranker is meaningless for
        # another; this is what tells you which one it was calibrated against.
        "reranker_model": (
            cfg["reranker"]["model"] if cfg["reranker"].get("enabled") else None
        ),
        "chunk_count": len(rows),
        # Marked so a truncated smoke index can never be mistaken for a real one.
        "partial": bool(args.limit),
        "corpus_fingerprint": corpus_fingerprint(rows),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": device,
        "books": sorted({r.get("book_short") for r in rows if r.get("book_short")}),
        "tiers": {
            t: sum(1 for r in rows if r.get("tier") == t)
            for t in sorted({r.get("tier") for r in rows})
        },
    }
    (out_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    print(f"wrote index -> {out_dir}")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
