from __future__ import annotations


class AppError(Exception):
    def __init__(self, message: str, *, code: str = "application_error", status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ProviderUnavailable(AppError):
    def __init__(self, provider: str, message: str = "LLM provider unavailable"):
        super().__init__(message, code=f"{provider}_unavailable", status_code=503)
        self.provider = provider
