from __future__ import annotations

from typing import Any


class AppError(Exception):
    def __init__(self, message: str, *, code: str = "application_error", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ProviderUnavailable(AppError):
    def __init__(
        self,
        provider: str,
        message: str = "LLM provider unavailable",
        *,
        diagnostics: dict[str, Any] | None = None,
    ):
        super().__init__(message, code=f"{provider}_unavailable", status_code=503)
        self.provider = provider
        # Structured, already-sanitized detail for logs -- an upstream status
        # code, an API error code, that sort of thing. The message alone cannot
        # carry it: "Gemini returned HTTP 400" says nothing about whether the
        # key is wrong, the quota is spent or the model name is stale.
        #
        # Never a credential and never a request header. Producers sanitize
        # before putting anything here, and the orchestrator logs only an
        # allowlist of these keys.
        self.diagnostics: dict[str, Any] = dict(diagnostics or {})
