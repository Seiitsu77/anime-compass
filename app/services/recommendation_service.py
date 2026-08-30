from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from app.api.schemas import RecommendRequest
from app.core.errors import AppError
from app.repositories.session_repository import SQLiteSessionRepository, merge_profiles
from backend.anime_agent.als_serving import ALSCollaborativeIndex
from backend.anime_agent.fast_path import FastPathConfig, recommend_fast
from backend.anime_agent.path_policy import RecommendationPath, choose_recommendation_path
from backend.anime_agent.recommender import AnimeRecommender

logger = logging.getLogger("anime_compass.recommendation")


class RecommendationService:
    def __init__(
        self,
        recommender: AnimeRecommender,
        sessions: SQLiteSessionRepository,
        *,
        als_index: ALSCollaborativeIndex | None = None,
        fallback_index: Any | None = None,
        tail_index: Any | None = None,
        fast_path_config: FastPathConfig | None = None,
    ):
        self.recommender = recommender
        self.sessions = sessions
        # The fast path is the default for unconstrained personalized requests.
        # Without a collaborative source every request takes the
        # constraint-rich hybrid, which is the pre-existing behaviour.
        self.als_index = als_index
        self.fallback_index = fallback_index
        self.tail_index = tail_index
        self.fast_path_config = fast_path_config or FastPathConfig()
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._cache_ttl_seconds = 45.0
        self._cache_size = 128

    async def recommend(self, request: RecommendRequest) -> dict[str, Any]:
        self._validate_catalog_values(request)
        started = time.perf_counter()
        session_profile = await asyncio.to_thread(self.sessions.get, request.session_id)
        session_profile = merge_profiles(session_profile, request.session_profile)
        cache_key = self._cache_key(request, session_profile)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return {**cached, "cache_hit": True}

        path_decision = choose_recommendation_path(request.model_dump())
        if path_decision.path is RecommendationPath.FAST and self.als_index is not None:
            fast = await asyncio.to_thread(self._recommend_fast, request, session_profile, path_decision)
            if fast is not None:
                fast["timing_ms"] = round((time.perf_counter() - started) * 1000, 2)
                if cache_key:
                    self._put_cached(cache_key, fast)
                logger.info(
                    "recommendation_completed",
                    extra={
                        "context": {
                            "path": "fast",
                            "collaborative_route": fast["diagnostics"].get("collaborative_route"),
                            "candidate_pool_size": fast["diagnostics"].get("candidate_pool_size"),
                            "tail_source_used": fast["diagnostics"].get("tail_source_used"),
                            "result_count": len(fast["results"]),
                            "duration_ms": fast["timing_ms"],
                        }
                    },
                )
                return fast

        payload = request.model_dump(exclude={"session_id", "session_profile", "limit"})
        payload["session_profile"] = session_profile
        payload["limit"] = request.top_k
        diagnostics: dict[str, Any] = {}
        payload["diagnostics"] = diagnostics
        results = await asyncio.to_thread(self.recommender.recommend, **payload)
        resolved_titles = await asyncio.to_thread(
            self.recommender.resolve_title_details,
            [*request.reference_titles, *request.liked_titles],
        )
        public_query = request.model_dump(
            exclude={"liked_ids", "excluded_ids", "weights", "session_profile", "limit"},
        )
        response: dict[str, Any] = {
            "query": public_query,
            "resolved_titles": resolved_titles,
            "recommendations": results,
            "results": results,
            "model_info": self.recommender.model_info(),
            "diagnostics": diagnostics,
            "timing_ms": round((time.perf_counter() - started) * 1000, 2),
            "cache_hit": False,
        }
        if not results:
            response["message"] = "No catalog titles matched all hard constraints."

        await asyncio.to_thread(
            self.sessions.log_event,
            request.session_id,
            "recommendation",
            "recommend",
            {
                "result_ids": [item["id"] for item in results],
                "required_voice_actors": request.required_voice_actors,
                "required_studios": request.required_studios,
                "required_staff": request.required_staff,
                "required_characters": request.required_characters,
            },
        )
        if cache_key:
            self._put_cached(cache_key, response)
        logger.info(
            "recommendation_completed",
            extra={
                "context": {
                    "recommendation_mode": results[0].get("recommendation_mode") if results else "empty",
                    "candidate_count_before_filter": diagnostics.get("candidate_count_before_filter", 0),
                    "candidate_count_after_filter": diagnostics.get("candidate_count_after_entity_filter", 0),
                    "result_count": len(results),
                    "cache_hit": False,
                    "duration_ms": response["timing_ms"],
                }
            },
        )
        return response

    async def search(
        self,
        query: str,
        *,
        limit: int,
        genres: list[str] | None = None,
        media_type: str | None = None,
        min_score: float | None = None,
        max_episodes: int | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.recommender.search,
            query,
            limit,
            genres,
            media_type,
            min_score,
            max_episodes,
        )

    async def search_page(self, **query: Any) -> tuple[list[dict[str, Any]], int]:
        return await asyncio.to_thread(self.recommender.search_page, **query)

    def _recommend_fast(
        self,
        request: RecommendRequest,
        session_profile: dict[str, Any],
        path_decision: Any,
    ) -> dict[str, Any] | None:
        """Serve an unconstrained personalized request from the fast path.

        Returns None when the profile yields no candidates, so the caller falls
        through to the hybrid rather than returning an empty list.
        """
        assert self.als_index is not None
        positives = [int(value) for value in request.liked_ids]
        positives.extend(int(value) for value in session_profile.get("liked_ids", []) or [])
        excluded = [int(value) for value in request.excluded_ids]
        excluded.extend(int(value) for value in session_profile.get("disliked_ids", []) or [])
        excluded.extend(int(value) for value in session_profile.get("watched_ids", []) or [])

        result = recommend_fast(
            positives,
            catalog_by_id=self.recommender.by_id,
            als_source=self.als_index,
            fallback_source=self.fallback_index,
            tail_source=self.tail_index,
            quality_lookup=self.als_index,
            excluded_ids=excluded,
            limit=request.top_k,
            config=self.fast_path_config,
        )
        if not result.anime_ids:
            return None

        items = [
            self.recommender.public_item(self.recommender.by_id[anime_id])
            for anime_id in result.anime_ids
            if anime_id in self.recommender.by_id
        ]
        provenance = {candidate.anime_id: candidate.as_dict() for candidate in result.candidates}
        for item in items:
            entry = provenance.get(int(item["id"]))
            if entry is not None:
                item["candidate_sources"] = entry["sources"]

        diagnostics = {**result.diagnostics, **path_decision.as_dict()}
        diagnostics["artifact_versions"] = self._artifact_versions()
        return {
            "query": request.model_dump(exclude={"liked_ids", "excluded_ids", "weights", "session_profile", "limit"}),
            "resolved_titles": [],
            "recommendations": items,
            "results": items,
            "model_info": self.recommender.model_info(),
            "diagnostics": diagnostics,
            "cache_hit": False,
        }

    def _artifact_versions(self) -> dict[str, Any]:
        """Version metadata so a served result is traceable to its artifacts."""
        info = self.als_index.model_info() if self.als_index is not None else {}
        return {
            "als_artifact_sha256": info.get("artifact_sha256"),
            "als_split_sha256": info.get("split_sha256"),
            "routing_config_version": self.fast_path_config.routing.version,
            "retrieval_config_version": self.fast_path_config.retrieval.version,
            "ranking_config_version": self.fast_path_config.version,
        }

    def _validate_catalog_values(self, request: RecommendRequest) -> None:
        metadata = self.recommender.meta()
        known_formats = {value.casefold() for value in metadata["types"]}
        unknown_formats = [value for value in request.formats if value.casefold() not in known_formats]
        if unknown_formats:
            raise AppError(
                "Unknown anime format: " + ", ".join(unknown_formats),
                code="invalid_format",
                status_code=422,
            )

    @staticmethod
    def _cache_key(request: RecommendRequest, session_profile: dict[str, Any]) -> str:
        if request.session_id or any(session_profile.get(key) for key in session_profile):
            return ""
        return json.dumps(request.model_dump(exclude={"session_id", "session_profile"}), sort_keys=True)

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        if not key:
            return None
        cached = self._cache.get(key)
        if cached is None:
            return None
        created_at, value = cached
        if time.monotonic() - created_at > self._cache_ttl_seconds:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return value

    def _put_cached(self, key: str, value: dict[str, Any]) -> None:
        self._cache[key] = (time.monotonic(), value)
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
