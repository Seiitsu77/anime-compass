from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agents.gemini_provider import GeminiAgentProvider
from app.agents.ollama_provider import OllamaAgentProvider
from app.agents.orchestrator import AgentOrchestrator
from app.api.routes import router
from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.embeddings.index import SemanticEmbeddingIndex
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from app.repositories.session_repository import SQLiteSessionRepository
from app.services.recommendation_service import RecommendationService
from backend.anime_agent.als_serving import (
    ALSArtifactError,
    ALSCatalogMismatchError,
    ALSCollaborativeIndex,
)
from backend.anime_agent.collaborative import CollaborativeIndex
from backend.anime_agent.data_pipeline import load_or_create_catalog
from backend.anime_agent.entities import EntityResolver
from backend.anime_agent.fast_path import FastPathConfig
from backend.anime_agent.recommender import AnimeRecommender
from backend.anime_agent.reranker_serving import try_load_reranker
from backend.anime_agent.retrieval import RetrievalConfig
from backend.anime_agent.routing import RoutingPolicy
from scripts.download_artifacts import ensure_artifacts

logger = logging.getLogger("anime_compass.api")


@dataclass
class AppContainer:
    settings: Settings
    recommender: AnimeRecommender
    recommendations: RecommendationService
    sessions: SQLiteSessionRepository
    entity_resolver: EntityResolver
    agent: AgentOrchestrator


class SlidingWindowRateLimiter:
    def __init__(self, request_limit: int, window_seconds: int, max_clients: int):
        self.request_limit = request_limit
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self.requests: OrderedDict[str, deque[float]] = OrderedDict()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        history = self.requests.pop(key, deque())
        while history and now - history[0] >= self.window_seconds:
            history.popleft()
        if len(history) >= self.request_limit:
            self.requests[key] = history
            return False
        history.append(now)
        self.requests[key] = history
        while len(self.requests) > self.max_clients:
            self.requests.popitem(last=False)
        return True


def create_app(
    *,
    settings: Settings | None = None,
    catalog: list[dict[str, Any]] | None = None,
    providers: dict[str, Any] | None = None,
    session_repository: SQLiteSessionRepository | None = None,
    semantic_index: SemanticEmbeddingIndex | None = None,
    collaborative_index: CollaborativeIndex | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if catalog is None and settings.hf_dataset_repo:
            await asyncio.to_thread(
                ensure_artifacts,
                settings.hf_dataset_repo,
                revision=settings.hf_dataset_revision,
            )
        loaded_catalog = catalog if catalog is not None else load_or_create_catalog(PROJECT_ROOT)
        loaded_semantic = semantic_index or _load_semantic_index(settings, loaded_catalog)
        loaded_collaborative = collaborative_index or _load_collaborative_index(
            settings,
            loaded_catalog,
        )
        # ALS is the primary collaborative source for users with history.
        # CountSketch stays loaded regardless: it is the sparse-user fallback,
        # the only cheap source with tail exposure, and the degradation path if
        # the ALS artifact is missing or fails validation.
        loaded_als = _load_als_index(settings, loaded_catalog, quality_source=loaded_collaborative)
        loaded_reranker = _load_reranker(settings, loaded_catalog, loaded_als)
        recommender = AnimeRecommender(
            loaded_catalog,
            semantic_index=loaded_semantic,
            collaborative_index=loaded_collaborative,
        )
        sessions = session_repository or SQLiteSessionRepository(
            settings.database_url,
            retention_days=settings.session_retention_days,
        )
        active_providers = providers or _build_providers(settings)
        agent = AgentOrchestrator(recommender, sessions, active_providers, settings)
        app.state.container = AppContainer(
            settings=settings,
            recommender=recommender,
            recommendations=RecommendationService(
                recommender,
                sessions,
                als_index=loaded_als,
                fallback_index=loaded_collaborative,
                fast_path_config=FastPathConfig(
                    reranker=loaded_reranker,
                    retrieval=RetrievalConfig(
                        als_top_n=settings.retrieval_als_top_n,
                        item_item_top_m=settings.retrieval_item_item_top_m,
                    ),
                    routing=RoutingPolicy(
                        medium_threshold=settings.routing_medium_threshold,
                        segment_aware=settings.routing_segment_aware,
                    ),
                    diversity_strength=settings.fast_path_diversity_strength,
                    diversity_window=settings.fast_path_diversity_window,
                ),
            ),
            sessions=sessions,
            entity_resolver=EntityResolver(loaded_catalog),
            agent=agent,
        )
        sessions.cleanup_expired()
        cleanup_stop = asyncio.Event()

        async def cleanup_sessions() -> None:
            while not cleanup_stop.is_set():
                try:
                    await asyncio.wait_for(
                        cleanup_stop.wait(),
                        timeout=settings.session_cleanup_interval_seconds,
                    )
                except TimeoutError:
                    removed = await asyncio.to_thread(sessions.cleanup_expired)
                    if removed:
                        logger.info("expired_sessions_removed", extra={"context": {"count": removed}})

        cleanup_task = asyncio.create_task(cleanup_sessions(), name="anime-compass-session-cleanup")
        logger.info(
            "application_started",
            extra={"context": {"catalog_count": len(loaded_catalog), "llm_provider": settings.llm_provider}},
        )
        try:
            yield
        finally:
            cleanup_stop.set()
            await cleanup_task
            await agent.close()
            sessions.engine.dispose()

    app = FastAPI(
        title="Anime Compass API",
        version="1.0.0",
        description="Explainable multi-channel anime recommendation API with agent orchestration.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    limiter = SlidingWindowRateLimiter(
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
        settings.rate_limit_max_clients,
    )

    @app.middleware("http")
    async def request_controls(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:128]
        request.state.request_id = request_id
        started = time.perf_counter()
        client_key = request.client.host if request.client else "unknown"

        content_length = request.headers.get("content-length")
        if content_length:
            try:
                too_large = int(content_length) > settings.max_request_body_bytes
            except ValueError:
                too_large = True
            if too_large:
                return _error_response(413, "request_too_large", "Request body is too large", request_id)

        if request.url.path.startswith("/api/") and not limiter.allow(client_key):
            return _error_response(429, "rate_limit_exceeded", "Too many requests", request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        logger.info(
            "request_completed",
            extra={
                "context": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            },
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return _error_response(exc.status_code, exc.code, exc.message, getattr(request.state, "request_id", None))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        safe_details = [
            {
                "type": error.get("type"),
                "location": list(error.get("loc", ())),
                "message": error.get("msg"),
            }
            for error in exc.errors()
        ]
        return _error_response(
            422,
            "validation_error",
            "Request validation failed",
            getattr(request.state, "request_id", None),
            safe_details,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        logger.exception(
            "unhandled_request_error",
            extra={
                "context": {"request_id": getattr(request.state, "request_id", None), "error_type": type(exc).__name__}
            },
        )
        return _error_response(
            500,
            "internal_error",
            "An unexpected server error occurred",
            getattr(request.state, "request_id", None),
        )

    app.include_router(router)
    frontend_root = PROJECT_ROOT / "frontend"
    app.mount("/", StaticFiles(directory=frontend_root, html=True), name="frontend")
    return app


def _build_providers(settings: Settings) -> dict[str, Any]:
    secret = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    return {
        "gemini": GeminiAgentProvider(
            api_key=secret,
            model=settings.gemini_model,
            base_url=settings.gemini_base_url,
            timeout=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
        ),
        "ollama": OllamaAgentProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.llm_timeout_seconds,
            max_output_tokens=settings.llm_max_output_tokens,
        ),
    }


def _load_semantic_index(
    settings: Settings,
    catalog: list[dict[str, Any]],
) -> SemanticEmbeddingIndex | None:
    if settings.embedding_provider != "sentence_transformers":
        return None
    if not settings.semantic_artifact_path.exists():
        logger.warning(
            "semantic_artifact_missing",
            extra={"context": {"path": str(settings.semantic_artifact_path)}},
        )
        return None
    try:
        provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            model_revision=settings.embedding_model_revision,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            local_files_only=settings.embedding_local_files_only,
        )
        return SemanticEmbeddingIndex.load(
            settings.semantic_artifact_path,
            provider,
            catalog,
            expected_dimension=settings.embedding_dimensions,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "semantic_index_unavailable",
            extra={"context": {"error_type": type(exc).__name__}},
        )
        return None


def _load_reranker(
    settings: Settings,
    catalog: list[dict[str, Any]],
    als_index: ALSCollaborativeIndex | None,
) -> Any | None:
    """Load the promoted LambdaMART reranker, or None to serve the ALS order.

    Never raises. The reranker improves an ordering the fast path already
    produces correctly, so a missing or invalid artifact costs the improvement
    and nothing else. `/health` reports which of the two is actually serving,
    so a degraded deployment is visible rather than silently claimed.
    """
    if not settings.reranker_enabled or als_index is None:
        return None
    reranker = try_load_reranker(
        settings.reranker_feature_path,
        settings.reranker_model_path,
        catalog,
        als_index.anime_ids,
        expected_feature_sha256=settings.reranker_feature_sha256 or None,
        expected_model_sha256=settings.reranker_model_sha256 or None,
    )
    if reranker is None:
        logger.warning(
            "reranker_unavailable",
            extra={"context": {"action": "serving_als_order", "enabled": settings.reranker_enabled}},
        )
    else:
        logger.info("reranker_loaded", extra={"context": reranker.model_info()})
    return reranker


def _load_als_index(
    settings: Settings,
    catalog: list[dict[str, Any]],
    *,
    quality_source: Any | None = None,
) -> ALSCollaborativeIndex | None:
    """Load the frozen ALS artifact, or return None to fall back.

    A *missing* artifact is a normal degraded state and logs a warning. A
    *present but invalid* artifact is not: a checksum or catalog mismatch means
    the model does not describe this catalog, so it is refused loudly rather
    than served. Nothing here trains or rebuilds.
    """
    if not settings.als_enabled:
        return None
    if not settings.als_artifact_path.exists():
        logger.warning(
            "als_artifact_missing",
            extra={"context": {"path": str(settings.als_artifact_path)}},
        )
        return None
    try:
        index = ALSCollaborativeIndex.load(
            settings.als_artifact_path,
            catalog,
            quality_source=quality_source,
            expected_artifact_sha256=settings.als_expected_sha256 or None,
            expected_role=settings.als_expected_role or None,
            expected_catalog_ids_sha256=settings.als_expected_catalog_ids_sha256 or None,
        )
    except ALSArtifactError as exc:
        # A catalog mismatch means the served catalog moved out from under the
        # model. Falling back quietly would keep serving recommendations from a
        # model that no longer describes the catalog, so this is escalated as an
        # explicit degradation event rather than logged as an ordinary warning.
        severity = "critical" if isinstance(exc, ALSCatalogMismatchError) else "high"
        logger.error(
            "als_artifact_degradation",
            extra={
                "context": {
                    "event": "als_artifact_invalid",
                    "severity": severity,
                    "error_type": type(exc).__name__,
                    "path": str(settings.als_artifact_path),
                    "detail": str(exc)[:300],
                    "expected_role": settings.als_expected_role,
                    "checksum_pinned": bool(settings.als_expected_sha256),
                    "catalog_pinned": bool(settings.als_expected_catalog_ids_sha256),
                    "strict": settings.als_require_valid_artifact,
                    "action": "refusing_startup"
                    if (
                        settings.als_require_valid_artifact
                        or settings.als_expected_sha256
                        or settings.als_expected_catalog_ids_sha256
                        or isinstance(exc, ALSCatalogMismatchError)
                    )
                    else "degraded_to_countsketch",
                }
            },
        )
        # Refuse to start when an operator pinned something specific and did not
        # get it, when strict mode is on, or when the catalog itself no longer
        # matches. Only a non-pinned, non-catalog validation failure degrades.
        if (
            settings.als_require_valid_artifact
            or settings.als_expected_sha256
            or settings.als_expected_catalog_ids_sha256
            or isinstance(exc, ALSCatalogMismatchError)
        ):
            raise
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            "als_index_unavailable",
            extra={"context": {"error_type": type(exc).__name__}},
        )
        return None
    logger.info(
        "als_index_loaded",
        extra={"context": {k: v for k, v in index.model_info().items() if k != "method"}},
    )
    return index


def _load_collaborative_index(
    settings: Settings,
    catalog: list[dict[str, Any]],
) -> CollaborativeIndex | None:
    if not settings.collaborative_enabled:
        return None
    if not settings.collaborative_artifact_path.exists():
        logger.warning(
            "collaborative_artifact_missing",
            extra={"context": {"path": str(settings.collaborative_artifact_path)}},
        )
        return None
    try:
        return CollaborativeIndex.load(
            settings.collaborative_artifact_path,
            catalog,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "collaborative_index_unavailable",
            extra={"context": {"error_type": type(exc).__name__}},
        )
        return None


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
    details: Any | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "ok": False,
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
                "details": details,
            },
        },
    )


app = create_app()
