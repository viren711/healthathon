"""OpenAI-compatible chat client.

Deliberately thin and dependency-light: it speaks the /v1/chat/completions
protocol over plain HTTP, so it works unchanged against vLLM, TGI, llama.cpp's
server, Ollama's compatibility endpoint, or any hosted equivalent. Point it
somewhere with LLM_BASE_URL / LLM_MODEL in .env.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from rag.config import load_config


class LLMError(RuntimeError):
    """Raised when the backend is unreachable or returns an unusable reply."""


@dataclass
class LLMClient:
    base_url: str
    model: str
    api_key: str = "not-needed"
    temperature: float = 0.1
    max_tokens: int = 900
    timeout: int = 120
    # "stop" (the model finished) or "length" (it hit max_tokens). Set by the
    # most recent chat/chat_stream call.
    last_finish_reason: str | None = None
    # Extra request fields merged into every payload. Model-specific switches
    # live here -- e.g. Qwen3.5 thinks silently by default and must be sent
    # {"chat_template_kwargs": {"enable_thinking": False}} on each request,
    # or it spends the whole token budget reasoning before writing a word.
    extra_body: dict | None = None

    @classmethod
    def from_config(cls) -> "LLMClient":
        llm = load_config()["llm"]
        return cls(
            base_url=llm["base_url"].rstrip("/"),
            model=llm["model"],
            api_key=llm.get("api_key", "not-needed"),
            temperature=llm.get("temperature", 0.1),
            max_tokens=llm.get("max_tokens", 900),
            timeout=llm.get("timeout_seconds", 120),
        )

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **(self.extra_body or {}),
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMError(f"LLM backend returned HTTP {exc.code}: {exc.read()[:400]!r}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"cannot reach the LLM at {self.base_url} ({exc.reason}). "
                "Start it with scripts/serve_llm.sh or set LLM_BASE_URL in .env."
            ) from exc

        try:
            choice = body["choices"][0]
            # Recorded so callers can tell a finished answer from one the token
            # cap cut off mid-sentence. Presenting the second as the first is
            # how a clinical answer silently loses its last paragraph.
            self.last_finish_reason = choice.get("finish_reason")
            return choice["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {body}") from exc

    def chat_stream(self, system: str, user: str):
        """Yield the reply in pieces as the server produces them.

        Same request as `chat`, with `stream: true`, parsed as
        Server-Sent Events. On this CPU the model emits ~10 tokens/s, so a
        250-token answer takes 25 seconds to finish -- streaming does not make
        that any faster, it just stops the user staring at a spinner for all
        of it.

        IMPORTANT: what this yields is UNVERIFIED. The citation check in
        rag.guard runs on the finished text, so a caller that displays these
        deltas is showing something the platform may still reject. The UI
        marks streamed text as provisional for exactly that reason, and
        replaces it if verification fails.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **(self.extra_body or {}),
        }
        self.last_finish_reason = None
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or [{}]
                    if choices[0].get("finish_reason"):
                        self.last_finish_reason = choices[0]["finish_reason"]
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        yield piece
        except urllib.error.HTTPError as exc:
            raise LLMError(f"LLM backend returned HTTP {exc.code}: {exc.read()[:400]!r}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(
                f"cannot reach the LLM at {self.base_url} ({exc.reason}). "
                "Start it with scripts/serve_llm.sh or set LLM_BASE_URL in .env."
            ) from exc

    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe for the app's status panel."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            names = [m.get("id") for m in data.get("data", [])]
            return True, f"reachable; models: {', '.join(n for n in names if n) or 'unknown'}"
        except Exception as exc:  # noqa: BLE001 - any failure is "not reachable"
            return False, str(exc)

    def resident_models(self) -> tuple[bool, list[str]]:
        """Which models the router currently holds in memory.

        In router mode /v1/models lists every declared model with a status of
        "loaded", "loading" or "unloaded". Returns (reachable, loaded_ids).
        A plain single-model server reports no status; treat everything it
        lists as loaded, since that is literally true of it.
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - any failure is "not reachable"
            return False, []
        loaded = []
        for m in data.get("data", []):
            status = (m.get("status") or {}).get("value")
            if status in (None, "loaded", "loading"):
                loaded.append(m.get("id"))
        return True, [n for n in loaded if n]
