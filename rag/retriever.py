"""Hybrid retrieval: dense FAISS + BM25, fused with RRF, then cross-encoder reranked.

Loading is lazy and cached, so the Streamlit app pays the model load cost once.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from rag.config import load_config, path_of

# The textbook index and the patient index are embedded with the same model
# and reranked with the same cross-encoder, so both retrievers share one
# loaded copy of each -- the second index costs its vectors, not another
# ~500 MB of model weights.
_MODELS: dict[tuple, Any] = {}


def _shared_model(kind: str, name: str, device: str):
    key = (kind, name, device)
    if key not in _MODELS:
        if kind == "embedder":
            from sentence_transformers import SentenceTransformer

            _MODELS[key] = SentenceTransformer(name, device=device)
        else:
            from sentence_transformers import CrossEncoder

            _MODELS[key] = CrossEncoder(name, device=device)
    return _MODELS[key]


def tokenize_for_bm25(text: str) -> list[str]:
    """Must match scripts/03_build_index.py exactly or BM25 scores are garbage."""
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Hit:
    """One retrieved chunk with its scores and everything needed to cite it."""

    chunk: dict[str, Any]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """The score the grounding gate judges: reranker if present, else dense."""
        return self.rerank_score if self.rerank_score is not None else self.dense_score

    @property
    def text(self) -> str:
        return self.chunk["text"]

    @property
    def citation(self) -> str:
        return self.chunk.get("citation", "")


class IndexMismatchError(RuntimeError):
    """The index on disk was built with a different embedding model."""


@dataclass
class Retriever:
    chunks: list[dict]
    meta: dict
    _faiss: Any = field(repr=False, default=None)
    _bm25: Any = field(repr=False, default=None)
    _embedder: Any = field(repr=False, default=None)
    _reranker: Any = field(repr=False, default=None)
    _centroid: Any = field(repr=False, default=None)
    # Row vectors, memory-mapped. FAISS cannot restrict a search to a subset
    # of ids without an IDSelector, and at 16k chunks an exact dot product
    # over the selected rows is both simpler and faster than building one.
    _vectors: Any = field(repr=False, default=None)
    _rows_by_pathology: Any = field(repr=False, default=None)
    _rows_by_system: Any = field(repr=False, default=None)

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, device: str | None = None, index_dir=None) -> "Retriever":
        import faiss

        cfg = load_config()
        index_dir = index_dir or path_of("index_dir")
        meta_path = index_dir / "index_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no index at {index_dir}. Run scripts/run_pipeline.sh first."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        configured = cfg["embedding"]["model"]
        if meta["embedding_model"] != configured:
            raise IndexMismatchError(
                f"index was built with '{meta['embedding_model']}' but config/rag.yaml "
                f"now says '{configured}'. Rebuild the index or restore the model name -- "
                "querying across models returns silently wrong results."
            )

        chunks = [
            json.loads(line)
            for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        obj = cls(chunks=chunks, meta=meta)
        obj._faiss = faiss.read_index(str(index_dir / "faiss.index"))
        obj._bm25 = pickle.loads((index_dir / "bm25.pkl").read_bytes())
        centroid_path = index_dir / "centroid.npy"
        obj._centroid = np.load(centroid_path) if centroid_path.exists() else None

        # mmap: 16k x 768 float32 is ~48 MB, and only the selected rows are
        # ever touched, so it need not be resident.
        emb_path = index_dir / "embeddings.npy"
        obj._vectors = np.load(emb_path, mmap_mode="r") if emb_path.exists() else None

        by_path: dict[str, list[int]] = {}
        by_sys: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            if chunk.get("pathology"):
                by_path.setdefault(chunk["pathology"], []).append(i)
            if chunk.get("organ_system"):
                by_sys.setdefault(chunk["organ_system"], []).append(i)
        obj._rows_by_pathology = {k: np.array(v, dtype="int64") for k, v in by_path.items()}
        obj._rows_by_system = {k: np.array(v, dtype="int64") for k, v in by_sys.items()}

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        obj._embedder = _shared_model("embedder", meta["embedding_model"], device)
        obj._embedder.max_seq_length = meta.get("max_seq_length", 512)

        if cfg["reranker"]["enabled"]:
            obj._reranker = _shared_model("reranker", cfg["reranker"]["model"], device)
            # Explicit, because shortening it is a tempting "optimisation"
            # that measurably costs recall -- see the note in rag.yaml.
            obj._reranker.max_length = cfg["reranker"].get("max_length", 512)

        return obj

    # -- embedding ----------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        prefix = self.meta.get("query_prefix", "")
        vec = self._embedder.encode(
            [prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=self.meta.get("normalize", True),
        ).astype("float32")
        return vec

    def embed_query_batch(self, queries: list[str]) -> np.ndarray:
        prefix = self.meta.get("query_prefix", "")
        return self._embedder.encode(
            [prefix + q for q in queries],
            convert_to_numpy=True,
            normalize_embeddings=self.meta.get("normalize", True),
        ).astype("float32")

    def centroid_similarity(self, query_vec: np.ndarray) -> float:
        """Cosine between the query and the mean corpus vector (topic gate)."""
        if self._centroid is None:
            return 1.0
        return float(np.dot(query_vec[0], self._centroid))

    # -- search -------------------------------------------------------------

    # -- scope selection ----------------------------------------------------

    def scope_rows(self, scope: str | None) -> Any:
        """Row indices for a pathology or organ-system id, or None for all.

        Accepts either level of the taxonomy so the UI can offer "Head & Neck"
        as well as "Oral cavity & tongue" without two code paths.
        """
        if not scope or scope == "all":
            return None
        rows = (self._rows_by_pathology or {}).get(scope)
        if rows is None:
            rows = (self._rows_by_system or {}).get(scope)
        return rows

    def available_scopes(self) -> dict[str, int]:
        """Every selectable scope id with its chunk count."""
        out = {k: int(len(v)) for k, v in (self._rows_by_pathology or {}).items()}
        out.update({k: int(len(v)) for k, v in (self._rows_by_system or {}).items()})
        return out

    def _dense_within(self, query_vec: np.ndarray, rows: Any, top_k: int):
        """Exact dense search restricted to `rows`.

        Vectors are L2-normalised at build time, so an inner product is the
        cosine and this returns exactly what FAISS would have returned had it
        been given only these rows.
        """
        if self._vectors is None:
            raise RuntimeError(
                "embeddings.npy is missing from the index, so scoped search is "
                "unavailable. Rebuild with scripts/03_build_index.py."
            )
        subset = np.asarray(self._vectors[rows])
        scores = subset @ query_vec[0]
        k = min(top_k, len(rows))
        if k <= 0:
            return np.array([]), np.array([])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return rows[top], scores[top]

    def _fuse_and_rerank(
        self,
        query: str,
        dense_ids,
        dense_vals,
        bm25_ids,
        bm25_vals,
        rerank_candidates: int,
        final_k: int | None,
    ) -> list[Hit]:
        """RRF over the two rankings, then optional cross-encoder rerank."""
        cfg = load_config()["retrieval"]
        dense_rank = {int(i): r for r, i in enumerate(dense_ids)}
        dense_score = {int(i): float(v) for i, v in zip(dense_ids, dense_vals)}
        bm25_rank = {int(i): r for r, i in enumerate(bm25_ids)}
        bm25_score = {int(i): float(v) for i, v in zip(bm25_ids, bm25_vals)}

        k = cfg["rrf_k"]
        fused: dict[int, float] = {}
        for ids in (dense_rank, bm25_rank):
            for idx, rank in ids.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:rerank_candidates]
        hits = [
            Hit(
                chunk=self.chunks[idx],
                dense_score=dense_score.get(idx, 0.0),
                bm25_score=bm25_score.get(idx, 0.0),
                fused_score=score,
            )
            for idx, score in ordered
        ]

        if self._reranker is not None and hits:
            pairs = [(query, h.chunk["embed_text"]) for h in hits]
            scores = self._reranker.predict(
                pairs, batch_size=load_config()["reranker"]["batch_size"]
            )
            for hit, sc in zip(hits, scores):
                hit.rerank_score = float(sc)
            hits.sort(key=lambda h: -h.rerank_score)

        return hits[:final_k] if final_k else hits

    def _bm25_within(self, query: str, rows: Any, top_k: int):
        """BM25 scores, optionally masked to a subset of rows."""
        all_scores = self._bm25.get_scores(tokenize_for_bm25(query))
        if rows is None:
            ids = np.argsort(all_scores)[::-1][:top_k]
            return ids, all_scores[ids]
        sub = all_scores[rows]
        k = min(top_k, len(rows))
        if k <= 0:
            return np.array([]), np.array([])
        top = np.argpartition(-sub, k - 1)[:k]
        top = top[np.argsort(-sub[top])]
        return rows[top], sub[top]

    def search_wide(
        self, query: str, candidates: int = 80, scope: str | None = None
    ) -> list[Hit]:
        """Retrieve a wide candidate set for the topic locator.

        Same hybrid machinery as `search`, but with the pool sizes widened and
        `final_top_k` ignored: the locator groups chunks into sections
        afterwards, so cutting to 5 chunks first would leave only two or three
        distinct places. No grounding threshold is applied here either -- the
        locator applies its own, much lower, floor.
        """
        cfg = load_config()["retrieval"]
        rows = self.scope_rows(scope)
        query_vec = self.embed_query(query)
        pool = max(candidates, cfg["dense_top_k"])

        if rows is None:
            d_scores, d_ids = self._faiss.search(query_vec, pool)
            dense_ids = [int(i) for i in d_ids[0] if i >= 0]
            dense_vals = [float(v) for i, v in zip(d_ids[0], d_scores[0]) if i >= 0]
        else:
            dense_ids, dense_vals = self._dense_within(query_vec, rows, pool)

        bm25_ids, bm25_vals = self._bm25_within(query, rows, pool)
        return self._fuse_and_rerank(
            query, dense_ids, dense_vals, bm25_ids, bm25_vals, candidates, None
        )

    def search(
        self,
        query: str,
        query_vec: np.ndarray | None = None,
        scope: str | None = None,
    ) -> list[Hit]:
        """Hybrid retrieval, optionally restricted to one pathology or system.

        `scope` accepts a pathology id ("hn_oral_cavity"), an organ-system id
        ("head_neck"), or None/"all". Scoping is applied BEFORE the top-k cut,
        not as a filter afterwards -- filtering afterwards would return
        whatever survived out of a corpus-wide top 50, which for a small
        pathology is usually nothing.
        """
        cfg = load_config()["retrieval"]
        rows = self.scope_rows(scope)
        if query_vec is None:
            query_vec = self.embed_query(query)

        if rows is None:
            d_scores, d_ids = self._faiss.search(query_vec, cfg["dense_top_k"])
            dense_ids = [int(i) for i in d_ids[0] if i >= 0]
            dense_vals = [float(v) for i, v in zip(d_ids[0], d_scores[0]) if i >= 0]
        else:
            dense_ids, dense_vals = self._dense_within(query_vec, rows, cfg["dense_top_k"])

        bm25_ids, bm25_vals = self._bm25_within(query, rows, cfg["bm25_top_k"])

        # Reciprocal Rank Fusion: rank-based, so the two score scales (cosine
        # vs BM25's unbounded values) never have to be made commensurate.
        return self._fuse_and_rerank(
            query,
            dense_ids,
            dense_vals,
            bm25_ids,
            bm25_vals,
            cfg["rerank_candidates"],
            cfg["final_top_k"],
        )


@lru_cache(maxsize=2)
def get_retriever(corpus: str = "books") -> Retriever:
    """The retriever for one corpus: "books" (the textbooks, for the Knowledge
    Assistant) or "patient" (the NCCN Guidelines for Patients)."""
    if corpus == "patient":
        from rag.config import patient_path

        return Retriever.load(index_dir=patient_path("index_dir"))
    return Retriever.load()
