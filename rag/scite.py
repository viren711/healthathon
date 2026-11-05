"""Small server-side client for the Scite Assistant API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from rag import keyring
from rag.config import _load_dotenv


class SciteError(RuntimeError):
    """Raised when the Scite API cannot answer a request."""

    def __init__(self, message: str, status: int | None = None, body: object = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class SciteAnswer:
    content: str
    references: list[dict[str, Any]]
    details: Any = None


class SciteClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.scite.ai",
        timeout: float = 120,
        poll_interval: float = 1,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval

    @classmethod
    def from_env(cls) -> "SciteClient | None":
        # SCITE_API_KEY, then extra_scite_api when the first is out of quota
        # (see rag/keyring.py). `api_key` is the one in use right now.
        keys = keyring.usable("scite") or keyring.configured("scite")
        if not keys:
            return None
        return cls(
            api_key=keys[0][1],
            base_url=os.environ.get("SCITE_API_BASE_URL", "https://api.scite.ai"),
            timeout=float(os.environ.get("SCITE_TIMEOUT_SECONDS", "120")),
            poll_interval=float(os.environ.get("SCITE_POLL_INTERVAL_SECONDS", "1")),
        )

    def ask(self, question: str, history: list[dict[str, Any]] | None = None, **kw: Any) -> SciteAnswer:
        """Ask with the first key that still has quota; a key refused for quota
        or access is remembered and skipped (rag/keyring.py), then the next."""
        last: SciteError | None = None
        for name, key in keyring.usable("scite"):
            self.api_key = key
            try:
                return self._ask(question, history, **kw)
            except SciteError as exc:
                if keyring.exhausted("scite", exc.status, exc.body):
                    keyring.mark_exhausted("scite", name, key, exc.status)
                    last = exc
                    continue
                raise
        if last is not None:
            raise last
        raise SciteError("Every Scite Assistant key has reached its limit for now "
                         "(delete artifacts/key_state.json to retry).", 429)

    def _ask(
        self,
        question: str,
        history: list[dict[str, Any]] | None = None,
        *,
        answer_length: str = "medium",
        citation_style: str = "apa",
        use_references: bool = True,
    ) -> SciteAnswer:
        turns = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in (history or [])
            if turn.get("role") in {"user", "assistant"} and turn.get("content")
        ]
        turns.append({"role": "user", "content": question})
        payload = {
            "turns": turns,
            "user_input": question,
            "session_id": None,
            "alwaysUseReferences": use_references,
            "neverUseReferences": not use_references,
            "abstractsOnly": False,
            "fullTextsOnly": False,
            "numReferences": 25,
            "rankBy": "all",
            "answerLength": answer_length,
            "yearFrom": "",
            "yearTo": "",
            "topics": [],
            "journals": [],
            "publicationTypes": [],
            "citationStyle": citation_style,
            "dashboards": [],
            "referenceChecks": [],
            "dois": [],
        }
        started = self._request("POST", "/api_partner/assistant/poll", payload)
        task_id = started.get("id") or started.get("task_id")
        if not task_id:
            raise SciteError("Scite did not return an assistant task id")

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = self._request("GET", f"/api_partner/assistant/tasks/{task_id}")
            status = str(result.get("status", "")).upper()
            if status == "SUCCESS":
                return self._answer_from(result)
            if status in {"FAILURE", "FAILED", "ERROR", "CANCELED", "CANCELLED"}:
                detail = result.get("error") or result.get("info") or status
                raise SciteError(f"Scite assistant task failed: {detail}")
            time.sleep(self.poll_interval)
        raise SciteError("Scite assistant request timed out")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400]
            raise SciteError(f"Scite API returned HTTP {exc.code}: {detail!r}", exc.code, detail) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SciteError(f"Cannot reach Scite at {self.base_url}: {exc}") from exc
        if not isinstance(body, dict):
            raise SciteError("Scite returned an invalid response")
        return body

    @staticmethod
    def _answer_from(result: dict[str, Any]) -> SciteAnswer:
        answer = result.get("answer") or result.get("content") or result.get("response") or ""
        if isinstance(answer, dict):
            answer = answer.get("content") or answer.get("text") or ""
        references = result.get("references") or result.get("sources") or []
        if not isinstance(references, list):
            references = []
        return SciteAnswer(str(answer).strip(), [ref for ref in references if isinstance(ref, dict)])


class SciteMCPClient:
    """Client for Scite's JSON-RPC literature-search tool."""

    def __init__(self, api_key: str, base_url: str = "https://api.scite.ai", timeout: float = 120) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "SciteMCPClient | None":
        keys = keyring.usable("scite_mcp") or keyring.configured("scite_mcp")
        if not keys:
            return None
        return cls(
            api_key=keys[0][1],
            base_url=os.environ.get("SCITE_MCP_BASE_URL", "https://api.scite.ai"),
            timeout=float(os.environ.get("SCITE_TIMEOUT_SECONDS", "120")),
        )

    def search(self, question: str, limit: int = 10) -> SciteAnswer:
        """First key that still has quota (see rag/keyring.py), then the next."""
        last: SciteError | None = None
        for name, key in keyring.usable("scite_mcp"):
            self.api_key = key
            try:
                return self._search(question, limit)
            except SciteError as exc:
                if keyring.exhausted("scite_mcp", exc.status, exc.body):
                    keyring.mark_exhausted("scite_mcp", name, key, exc.status)
                    last = exc
                    continue
                raise
        if last is not None:
            raise last
        raise SciteError("Every Scite search key has reached its limit for now "
                         "(delete artifacts/key_state.json to retry).", 429)

    def _search(self, question: str, limit: int = 10) -> SciteAnswer:
        body = self._request(
            {
                "jsonrpc": "2.0",
                "id": "scite-research-chat",
                "method": "tools/call",
                "params": {
                    "name": "search_literature",
                    "arguments": {
                        "term": question,
                        "limit": limit,
                        "sort": "relevance",
                        "sort_order": "desc",
                    },
                },
            }
        )
        if body.get("error"):
            raise SciteError(f"Scite MCP error: {body['error']}")
        result = body.get("result") or {}
        if result.get("isError"):
            raise SciteError(f"Scite MCP error: {result.get('content') or result}")
        references = self._references_from(result)
        content = f"Scite MCP found {len(references)} relevant source(s) for this question."
        return SciteAnswer(content, references, details=result)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/mcp",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400]
            raise SciteError(f"Scite MCP returned HTTP {exc.code}: {detail!r}", exc.code, detail) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SciteError(f"Cannot reach Scite MCP at {self.base_url}: {exc}") from exc
        if not isinstance(body, dict):
            raise SciteError("Scite MCP returned an invalid response")
        return body

    @staticmethod
    def _references_from(result: dict[str, Any]) -> list[dict[str, Any]]:
        values: list[Any] = [result]
        references: list[dict[str, Any]] = []
        while values:
            value = values.pop()
            if isinstance(value, list):
                values.extend(value)
            elif isinstance(value, dict):
                if any(key in value for key in ("title", "doi", "paper")):
                    references.append(value)
                values.extend(value.values())
            elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                try:
                    values.append(json.loads(value))
                except json.JSONDecodeError:
                    pass
        unique: dict[str, dict[str, Any]] = {}
        for reference in references:
            identity = str(reference.get("doi") or reference.get("title") or len(unique))
            unique.setdefault(identity, reference)
        return list(unique.values())