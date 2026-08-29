from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: Annotated[list[str], NoDecode] = ["http://127.0.0.1:8000", "http://localhost:8000"]
    database_url: str = "sqlite:///data/anime_compass.db"
    session_retention_days: int = Field(default=30, ge=1, le=365)
    max_request_body_bytes: int = Field(default=65_536, ge=1024, le=1_048_576)
    rate_limit_requests: int = Field(default=60, ge=1, le=10_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    rate_limit_max_clients: int = Field(default=10_000, ge=100, le=1_000_000)
    session_cleanup_interval_seconds: int = Field(default=3600, ge=60, le=86_400)
    max_conversation_history: int = Field(default=12, ge=0, le=40)
    # Bounded deterministic constraint relaxation when a catalog request is
    # over-constrained. Zero disables replanning entirely.
    max_replan_steps: int = Field(default=2, ge=0, le=5)

    llm_provider: Literal["gemini", "ollama"] = "ollama"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:12b"
    llm_timeout_seconds: float = Field(default=35.0, ge=1.0, le=120.0)
    llm_max_output_tokens: int = Field(default=1200, ge=128, le=8192)
    provider_failure_threshold: int = Field(default=3, ge=1, le=20)
    provider_cooldown_seconds: int = Field(default=60, ge=1, le=3600)

    hf_dataset_repo: str = ""
    hf_dataset_revision: str = "main"

    embedding_provider: Literal["none", "sentence_transformers"] = "none"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_dimensions: int = Field(default=384, ge=64, le=4096)
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    embedding_local_files_only: bool = True
    semantic_artifact_path: Path = PROJECT_ROOT / "data" / "processed" / "semantic_embeddings.npz"
    collaborative_enabled: bool = True
    collaborative_artifact_path: Path = PROJECT_ROOT / "data" / "processed" / "collaborative_embeddings.npz"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator(
        "semantic_artifact_path",
        "collaborative_artifact_path",
        mode="before",
    )
    @classmethod
    def resolve_artifact_path(cls, value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def sqlite_path(self) -> Path | None:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        path = Path(self.database_url[len(prefix) :])
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
