"""Configuration loading shared by the retriever, the guards and the app."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env reader so the app has no python-dotenv dependency."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    _load_dotenv()
    cfg = yaml.safe_load((ROOT / "config" / "rag.yaml").read_text(encoding="utf-8"))
    books = yaml.safe_load((ROOT / "config" / "books.yaml").read_text(encoding="utf-8"))
    cfg["lexicon"] = books["lexicon"]

    # Environment overrides, so the same checkout can point at a different
    # index or embedding model without editing YAML. Used by the smoke test
    # and by anyone A/B-ing two embedders on the same corpus.
    if os.environ.get("EMBEDDING_MODEL"):
        cfg["embedding"]["model"] = os.environ["EMBEDDING_MODEL"]
    if os.environ.get("INDEX_DIR"):
        cfg["paths"]["index_dir"] = os.environ["INDEX_DIR"]
    if os.environ.get("RERANKER_MODEL"):
        cfg["reranker"]["model"] = os.environ["RERANKER_MODEL"]
    if os.environ.get("RERANKER_ENABLED"):
        cfg["reranker"]["enabled"] = os.environ["RERANKER_ENABLED"].lower() in ("1", "true", "yes")

    llm = cfg["llm"]
    # `or` not `get(..., default)`: an env var set to the empty string should
    # fall back to the config, not blank the setting. Exported-but-empty is a
    # normal state for a shell wrapper that only sometimes sets a value.
    llm["base_url"] = os.environ.get("LLM_BASE_URL") or llm["base_url"]
    llm["model"] = os.environ.get("LLM_MODEL") or llm["model"]
    llm["api_key"] = os.environ.get("LLM_API_KEY", "not-needed")
    if os.environ.get("LLM_TEMPERATURE"):
        llm["temperature"] = float(os.environ["LLM_TEMPERATURE"])

    return cfg


def path_of(key: str) -> Path:
    return ROOT / load_config()["paths"][key]


@lru_cache(maxsize=1)
def load_patient_config() -> dict[str, Any]:
    """config/patient.yaml, with the guide registry attached as `guides`.

    Only what differs for the Patient Portal lives there; the embedder,
    reranker, retrieval pool sizes and LLM endpoint are shared with
    load_config(), because both indexes are searched with the same models.
    """
    cfg = yaml.safe_load((ROOT / "config" / "patient.yaml").read_text(encoding="utf-8"))
    registry = yaml.safe_load((ROOT / cfg["paths"]["guides"]).read_text(encoding="utf-8"))
    cfg["guides"] = registry["guides"]
    cfg["categories"] = registry["categories"]
    if os.environ.get("PATIENT_INDEX_DIR"):
        cfg["paths"]["index_dir"] = os.environ["PATIENT_INDEX_DIR"]
    return cfg


def patient_path(key: str) -> Path:
    return ROOT / load_patient_config()["paths"][key]
