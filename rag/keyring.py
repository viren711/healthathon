"""Several API keys per service, tried in order, with exhausted ones skipped.

A key that the service refuses because its credit or quota is used up is
remembered in artifacts/key_state.json (only a short fingerprint of the key,
never the key itself) so later requests, and later launches, go straight to
the next key instead of paying for another failed call.

  sarvam  stays skipped until you delete its entry (credit does not renew).
  scite   stays skipped until the end of the calendar month (monthly quota).

To try a key again, delete artifacts/key_state.json (or just that entry).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import date
from pathlib import Path

from rag.config import ROOT, _load_dotenv

# Order matters: the first name is tried first. A name that is not set is skipped.
NAMES = {
    "sarvam": ["SARVAM_API_KEY", "SARVAM_API_KEY_1", "SARVAM_API_KEY_2", "SARVAM_API_KEY_coiffeur"],
    "scite": ["SCITE_API_KEY", "extra_scite_api"],
    # The literature-search fallback: its own key first, then the extra one.
    "scite_mcp": ["SCITE_MCP_API_KEY", "extra_scite_api"],
}
# Words in an error body that mean "this key's credit/quota is used up" when the
# status is 429 (a plain rate limit says "rate", not these).
QUOTA_WORDS = {
    "sarvam": ("credit", "quota", "insufficient", "balance"),
    "scite": ("monthly", "plan", "api limit"),
    "scite_mcp": ("monthly", "plan", "api limit"),
}
MONTHLY = {"scite", "scite_mcp"}

_lock = threading.Lock()


def _path() -> Path:
    return Path(os.environ.get("KEY_STATE_FILE") or ROOT / "artifacts" / "key_state.json")


def _fp(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _read() -> dict:
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def configured(service: str) -> list[tuple[str, str]]:
    """(name, key) for every key set in .env, in order, without duplicates."""
    _load_dotenv()
    out, seen = [], set()
    for name in NAMES[service]:
        value = os.environ.get(name, "").strip()
        if value and value not in seen:
            seen.add(value)
            out.append((name, value))
    return out


def usable(service: str) -> list[tuple[str, str]]:
    """The configured keys that are not marked exhausted, in order."""
    dead = _read().get(service, {})
    today = date.today().isoformat()
    out = []
    for name, key in configured(service):
        entry = dead.get(_fp(key))
        if entry and not (entry.get("until") and entry["until"] <= today):
            continue
        out.append((name, key))
    return out


def exhausted(service: str, status: int | None, body: object = "") -> bool:
    """Does this failure mean the key itself is spent or refused?"""
    if status in (401, 402, 403):
        return True
    if status == 429:
        text = str(body).lower()
        return any(w in text for w in QUOTA_WORDS[service])
    return False


def mark_exhausted(service: str, name: str, key: str, status: int | None) -> None:
    today = date.today()
    until = None
    if service in MONTHLY:
        until = date(today.year + (today.month == 12), today.month % 12 + 1, 1).isoformat()
    with _lock:
        state = _read()
        state.setdefault(service, {})[_fp(key)] = {
            "name": name, "status": status, "since": today.isoformat(), "until": until,
        }
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
