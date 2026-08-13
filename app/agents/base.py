from __future__ import annotations

from typing import Any, Protocol

from .schemas import AgentIntent, ProviderHealth, ProviderParseContext


class AgentProvider(Protocol):
    name: str
    model: str

    async def parse_intent(self, message: str, context: ProviderParseContext) -> AgentIntent: ...

    async def generate_tool_response(
        self,
        user_message: str,
        verified_tool_data: dict[str, Any],
    ) -> str: ...

    async def generate_explanation(
        self,
        user_message: str,
        verified_recommendation: dict[str, Any],
    ) -> str: ...

    async def health_check(self) -> ProviderHealth: ...

    async def close(self) -> None: ...
