from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable


class OllamaUnavailable(RuntimeError):
    pass


@runtime_checkable
class ChatClient(Protocol):
    """The provider surface AnimeAgent depends on.

    Callers may supply any object with these members, including the
    deterministic no-LLM stand-in used when no provider is configured.
    `chat`/`chat_json` are reached only when `is_available()` returns True.
    """

    @property
    def model(self) -> str: ...

    @property
    def base_url(self) -> str: ...

    def is_available(self) -> bool: ...

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 45.0):
        resolved_base_url: str = base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
        resolved_model: str = model or os.getenv("OLLAMA_MODEL") or "gemma3:12b"
        self.base_url: str = resolved_base_url.rstrip("/")
        self.model: str = resolved_model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(request, timeout=1.5):
                return True
        except (OSError, urllib.error.URLError):
            return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._chat(messages)

    def chat_json(self, messages: list[dict[str, str]]) -> str:
        return self._chat(messages, response_format="json")

    def _chat(self, messages: list[dict[str, str]], response_format: str | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
            },
        }
        if response_format:
            payload["format"] = response_format
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise OllamaUnavailable(str(exc)) from exc

        message = result.get("message", {})
        content = message.get("content", "")
        return str(content).strip()
