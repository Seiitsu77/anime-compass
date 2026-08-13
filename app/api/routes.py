from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import container
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    EntitySearchRequest,
    EntitySearchResponse,
    HealthResponse,
    LegacySessionPreferenceUpdate,
    RankRequest,
    RecommendRequest,
    RecommendResponse,
    SearchRequest,
    SessionPreferenceUpdate,
)
from app.core.errors import AppError

router = APIRouter(prefix="/api")
Container = Annotated[Any, Depends(container)]
SessionId = Annotated[
    str,
    Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
@router.post("/agent", response_model=ChatResponse, response_model_exclude_none=True, include_in_schema=False)
async def chat(request: ChatRequest, state: Container) -> dict[str, Any]:
    state.sessions.log_event(request.session_id, "chat", "user_message", {"length": len(request.message)})
    return await state.agent.respond(
        request.message,
        history=[message.model_dump() for message in request.history],
        session_id=request.session_id,
        debug=request.debug,
    )


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest, state: Container) -> dict[str, Any]:
    return await state.recommendations.recommend(request)


@router.post("/search")
async def search(request: SearchRequest, state: Container) -> dict[str, Any]:
    results, total = await state.recommendations.search_page(
        query=request.query,
        limit=request.top_k,
        offset=request.offset,
        include_genres=request.include_genres,
        exclude_genres=request.exclude_genres,
        formats=request.formats,
        required_studios=request.required_studios,
        min_score=request.min_score,
        min_year=request.min_year,
        max_year=request.max_year,
        max_episodes=request.max_episodes,
        sort_by=request.sort_by,
        sort_order=request.sort_order,
        semantic=request.semantic,
    )
    return {
        "results": results,
        "total": total,
        "offset": request.offset,
        "limit": request.top_k,
        "has_more": request.offset + len(results) < total,
    }


@router.post("/rank")
async def rank(request: RankRequest, state: Container) -> dict[str, Any]:
    results, diagnostics = await asyncio.to_thread(
        state.recommender.rank_catalog,
        **request.model_dump(exclude={"top_k"}),
        limit=request.top_k,
    )
    return {"results": results, "diagnostics": diagnostics}


@router.get("/anime/search", include_in_schema=False)
async def legacy_search(
    state: Container,
    q: str = Query(default="", max_length=300),
    limit: str = Query(default="10", max_length=12),
    genres: Annotated[list[str] | None, Query()] = None,
    media_type: str | None = Query(default=None, max_length=40),
    min_score: float | None = Query(default=None, ge=0, le=10),
    max_episodes: int | None = Query(default=None, gt=0),
) -> dict[str, Any]:
    if limit.casefold() == "all":
        parsed_limit = min(200, len(state.recommender.catalog))
    else:
        try:
            parsed_limit = max(1, min(int(limit), len(state.recommender.catalog)))
        except ValueError as exc:
            raise AppError("limit must be an integer or 'all'", code="invalid_limit", status_code=422) from exc
    return {
        "results": await state.recommendations.search(
            q,
            limit=parsed_limit,
            genres=genres or [],
            media_type=media_type,
            min_score=min_score,
            max_episodes=max_episodes,
        )
    }


@router.post("/entities/search", response_model=EntitySearchResponse)
async def search_entities(request: EntitySearchRequest, state: Container) -> dict[str, Any]:
    results = await asyncio.to_thread(
        state.entity_resolver.search,
        request.query,
        request.entity_types,
        request.top_k,
    )
    return {"results": results}


@router.get("/anime/{anime_id}")
async def anime_details(anime_id: int, state: Container) -> dict[str, Any]:
    result = await asyncio.to_thread(state.recommender.details, anime_id)
    if result is None:
        raise AppError("Anime not found", code="anime_not_found", status_code=404)
    return {"result": result}


@router.get("/session/{session_id}")
async def get_session(session_id: SessionId, state: Container) -> dict[str, Any]:
    return {"session_id": session_id, "profile": await asyncio.to_thread(state.sessions.get, session_id)}


@router.post("/session/{session_id}/preferences")
async def update_session(
    session_id: SessionId,
    request: SessionPreferenceUpdate,
    state: Container,
) -> dict[str, Any]:
    profile = await asyncio.to_thread(state.sessions.update, session_id, request.model_dump())
    await asyncio.to_thread(state.sessions.log_event, session_id, "feedback", "preferences_updated", {})
    return {"session_id": session_id, "profile": profile}


@router.delete("/session/{session_id}")
async def delete_session(session_id: SessionId, state: Container) -> dict[str, Any]:
    deleted = await asyncio.to_thread(state.sessions.delete, session_id)
    return {"session_id": session_id, "deleted": deleted}


@router.get("/session/preferences", include_in_schema=False)
async def legacy_get_session(state: Container, session_id: str = Query(default="", max_length=128)) -> dict[str, Any]:
    return {"session_id": session_id, "profile": await asyncio.to_thread(state.sessions.get, session_id)}


@router.post("/session/preferences", include_in_schema=False)
async def legacy_update_session(request: LegacySessionPreferenceUpdate, state: Container) -> dict[str, Any]:
    payload = request.model_dump(exclude={"session_id"})
    profile = await asyncio.to_thread(state.sessions.update, request.session_id, payload)
    await asyncio.to_thread(state.sessions.log_event, request.session_id, "feedback", "preferences_updated", {})
    return {"session_id": request.session_id, "profile": profile}


@router.get("/meta")
async def metadata(state: Container) -> dict[str, Any]:
    return state.recommender.meta()


@router.get("/model-info")
async def model_info(state: Container) -> dict[str, Any]:
    return state.recommender.model_info()


@router.get("/health", response_model=HealthResponse)
async def health(state: Container) -> dict[str, Any]:
    provider_health = await state.agent.health()
    provider_order = state.agent._provider_order()
    selected = next(
        (provider_health[name] for name in provider_order if provider_health.get(name, {}).get("available")),
        None,
    )
    database_ok = await asyncio.to_thread(state.sessions.health)
    semantic_available = state.recommender.semantic_index is not None
    semantic_info = state.recommender.model_info()["semantic_embedding"]
    collaborative_available = state.recommender.collaborative_index is not None
    collaborative_info = state.recommender.model_info()["collaborative"]
    catalog = state.recommender.meta()
    essential_ok = bool(catalog["count"]) and database_ok
    status = "healthy" if essential_ok and selected else "degraded" if essential_ok else "unhealthy"
    return {
        "ok": essential_ok,
        "status": status,
        "components": {
            "api": {"status": "healthy", "detail": "FastAPI request handling is available"},
            "catalog": {
                "status": "healthy" if catalog["count"] else "unavailable",
                "detail": f"{catalog['count']} titles",
            },
            "database": {"status": "healthy" if database_ok else "unavailable", "detail": "SQLite session store"},
            **{
                name: {
                    "status": "healthy" if check.get("available") else "unavailable",
                    "detail": check.get("detail") or str(check.get("model") or "configured"),
                }
                for name, check in provider_health.items()
            },
            "semantic_embeddings": {
                "status": "healthy" if semantic_available else "degraded",
                "detail": (
                    f"{semantic_info.get('model_name')} ({semantic_info.get('vector_dimension')} dimensions, "
                    f"{semantic_info.get('document_count')} documents)"
                )
                if semantic_available
                else "Optional artifact not loaded",
            },
            "collaborative_model": {
                "status": "healthy" if collaborative_available else "degraded",
                "detail": (f"{collaborative_info.get('items')} items / {collaborative_info.get('ratings'):,} ratings")
                if collaborative_available
                else "Optional artifact not loaded",
            },
        },
        "catalog": catalog,
        "agent": {
            "provider": selected["provider"] if selected else "deterministic",
            "model": selected["model"] if selected else None,
            "available": bool(selected),
            "providers": provider_health,
        },
    }


@router.get("/ready", include_in_schema=False)
async def readiness(state: Container) -> dict[str, Any]:
    database_ok = await asyncio.to_thread(state.sessions.health)
    catalog_ok = bool(state.recommender.catalog)
    if not (database_ok and catalog_ok):
        raise AppError("Application is not ready", code="not_ready", status_code=503)
    return {"status": "ready", "catalog_count": len(state.recommender.catalog), "database": "healthy"}
