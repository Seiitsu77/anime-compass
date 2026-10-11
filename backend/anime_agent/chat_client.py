from __future__ import annotations

from typing import Protocol, runtime_checkable


class ChatUnavailable(RuntimeError):
    """An injected chat client cannot generate a response."""


@runtime_checkable
class ChatClient(Protocol):
    """Optional text generation interface for the catalog agent."""

    @property
    def model(self) -> str: ...

    @property
    def base_url(self) -> str: ...

    def is_available(self) -> bool: ...

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class CatalogOnlyClient:
    """Use catalog tools without probing a local LLM or opening a connection."""

    model = "deterministic"
    base_url = "local"

    @staticmethod
    def is_available() -> bool:
        return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise ChatUnavailable("The catalog-only client cannot generate text.")
