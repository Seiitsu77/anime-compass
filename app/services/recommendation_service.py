from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.api.schemas import RecommendPageRequest, RecommendRequest
from app.core.errors import AppError
from app.repositories.session_repository import SQLiteSessionRepository, merge_profiles
from app.services.recommendation_sequence import (
    SEQUENCE_VERSION,
    RankedSequence,
    SequenceCache,
    decode_cursor,
    encode_cursor,
    sequence_fingerprint,
)
from backend.anime_agent.als_serving import ALSCollaborativeIndex
from backend.anime_agent.fast_path import FastPathConfig, recommend_fast
from backend.anime_agent.path_policy import RecommendationPath, choose_recommendation_path
from backend.anime_agent.recommender import AnimeRecommender, series_key

logger = logging.getLogger("anime_compass.recommendation")

# How many leading IDs the developer-only debug block reports per stage.
DEBUG_TOP_IDS = 20
# A starter sequence is ranked one batch at a time. The hybrid's diversity pass
# is quadratic in its limit, so ranking all 54 titles at once kept a new user
# waiting several seconds for the first screen; one batch is what that screen
# shows.
STARTER_CHUNK = 18


def _title_key(value: Any) -> str:
    return " ".join(str(value).casefold().split())


@dataclass(frozen=True)
class _ProfileInputs:
    """What one request tells the fast path about a user."""

    positives: list[int]
    # Everything that must never be returned, the positives included.
    excluded: list[int]
    excluded_titles: list[str]
    title_ids: list[int]
    unresolved_titles: int


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
        debug_diagnostics: bool = False,
        sequence_windows: int = 4,
        starter_depth: int = 54,
        sequence_cache_size: int = 256,
        sequence_ttl_seconds: float = 1800.0,
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
        # Developer-only: see Settings.recommendation_debug_diagnostics.
        self.debug_diagnostics = debug_diagnostics
        self._title_ids: dict[str, tuple[int, ...]] | None = None
        # Paged sequences: see app/services/recommendation_sequence.py.
        self.sequence_windows = max(1, int(sequence_windows))
        self.starter_depth = max(1, int(starter_depth))
        self._sequences = SequenceCache(sequence_cache_size, sequence_ttl_seconds)
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
                            "known_positive_count": fast["diagnostics"].get("known_positive_count"),
                            "excluded_count": fast["diagnostics"].get("excluded_count"),
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

    def _collapse_series(self, ranked: list[int], liked_ids: list[int]) -> list[int]:
        """Keep the best-ranked entry per franchise, and drop the profile's own.

        A display filter, not a ranking change: it reads the order the reranker
        produced and removes near-duplicates from the same series.
        """
        seen: set[str] = set()
        for liked in liked_ids:
            item = self.recommender.by_id.get(int(liked))
            if item is not None:
                seen.add(series_key(str(item.get("title") or "")))
        kept: list[int] = []
        for anime_id in ranked:
            item = self.recommender.by_id.get(int(anime_id))
            if item is None:
                continue
            key = series_key(str(item.get("title") or ""))
            if key in seen:
                continue
            seen.add(key)
            kept.append(int(anime_id))
        return kept

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
        inputs = self._profile_inputs(
            request.liked_ids,
            request.excluded_ids,
            [*request.excluded_titles, *request.seen_titles],
            session_profile,
        )
        positives, excluded = inputs.positives, inputs.excluded

        # `one_per_series` is part of the request contract, and the fast path was
        # silently ignoring it: liking Steins;Gate returned three Steins;Gate
        # entries. Collapsing happens after ranking, on extra candidates, so the
        # learned order is read rather than changed.
        fetch = request.top_k * 4 if request.one_per_series else request.top_k
        result = recommend_fast(
            positives,
            catalog_by_id=self.recommender.by_id,
            als_source=self.als_index,
            fallback_source=self.fallback_index,
            tail_source=self.tail_index,
            quality_lookup=self.als_index,
            profile_rows=positives,
            excluded_ids=excluded,
            limit=fetch,
            config=self.fast_path_config,
        )
        ranked_ids = result.anime_ids
        if request.one_per_series:
            ranked_ids = self._collapse_series(ranked_ids, positives)
        ranked_ids = ranked_ids[: request.top_k]
        if not ranked_ids:
            return None

        items = [
            self.recommender.public_item(self.recommender.by_id[anime_id])
            for anime_id in ranked_ids
            if anime_id in self.recommender.by_id
        ]
        provenance = {candidate.anime_id: candidate.as_dict() for candidate in result.candidates}
        for item in items:
            entry = provenance.get(int(item["id"]))
            if entry is not None:
                item["candidate_sources"] = entry["sources"]

        diagnostics = {**result.diagnostics, **path_decision.as_dict()}
        diagnostics["artifact_versions"] = self._artifact_versions()
        diagnostics["profile"] = self._profile_counts(
            inputs, len(request.liked_ids), len(request.excluded_ids), request.one_per_series
        )
        if self.debug_diagnostics:
            diagnostics["debug"] = self._debug_block(positives, excluded, result, ranked_ids)
        return {
            "query": request.model_dump(exclude={"liked_ids", "excluded_ids", "weights", "session_profile", "limit"}),
            "resolved_titles": [],
            "recommendations": items,
            "results": items,
            "model_info": self.recommender.model_info(),
            "diagnostics": diagnostics,
            "cache_hit": False,
        }

    def _profile_inputs(
        self,
        liked_ids: Sequence[int],
        excluded_ids: Sequence[int],
        titles: Sequence[str],
        session_profile: dict[str, Any],
    ) -> _ProfileInputs:
        """Positives and exclusions for one request, resolved once."""
        positives = [int(value) for value in liked_ids]
        positives.extend(int(value) for value in session_profile.get("liked_ids", []) or [])
        positives = list(dict.fromkeys(positives))
        excluded = [int(value) for value in excluded_ids]
        excluded.extend(int(value) for value in session_profile.get("disliked_ids", []) or [])
        excluded.extend(int(value) for value in session_profile.get("watched_ids", []) or [])

        # Negative feedback is stored by title: the browser's legacy session
        # schema and the agent both record disliked, hidden, and seen titles as
        # strings. The fast path used to read IDs only, so on the default
        # personalized route a "not interested" title was dropped by the
        # constraint-rich Hybrid and by the browser, but not here.
        excluded_titles = [
            *titles,
            *(session_profile.get("excluded_titles", []) or []),
            *(session_profile.get("seen_titles", []) or []),
            *(session_profile.get("disliked_titles", []) or []),
        ]
        title_ids, unresolved_titles = self._exact_title_ids(excluded_titles)
        excluded.extend(title_ids)
        # A liked title is a profile input and never a recommendation. ALS masks
        # its own positives, but that is one source's behaviour; stating it here
        # makes the guarantee hold for the fallback and tail sources too.
        excluded.extend(positives)
        excluded = list(dict.fromkeys(excluded))
        return _ProfileInputs(positives, excluded, excluded_titles, title_ids, unresolved_titles)

    def _profile_counts(
        self,
        inputs: _ProfileInputs,
        liked_received: int,
        excluded_received: int,
        one_per_series: bool,
    ) -> dict[str, Any]:
        """Counts only, so they are safe to return in any environment.

        They answer "did my feedback reach the ranker?" without revealing what
        the feedback was.
        """
        known_to_model = self.als_index.index_by_id if self.als_index is not None else {}
        return {
            "liked_ids_received": liked_received,
            "positives_used": len(inputs.positives),
            "positives_known_to_model": sum(1 for value in inputs.positives if value in known_to_model),
            "excluded_ids_received": excluded_received,
            "excluded_titles_received": len(
                {_title_key(value) for value in inputs.excluded_titles if str(value).strip()}
            ),
            "excluded_titles_resolved": len(inputs.title_ids),
            "excluded_titles_unresolved": inputs.unresolved_titles,
            "series_collapsed": bool(one_per_series),
        }

    # ------------------------------------------------------------ paging

    def sequence_limits(self) -> dict[str, int]:
        """How deep a sequence can go, for /api/health and the client."""
        return {
            "window_size": int(self.fast_path_config.retrieval.als_top_n),
            "max_windows": self.sequence_windows,
            "starter_depth": self.starter_depth,
            "starter_chunk": STARTER_CHUNK,
        }

    async def page(self, request: RecommendPageRequest) -> dict[str, Any]:
        """One page of the personalized sequence the request describes."""
        started = time.perf_counter()
        response = await asyncio.to_thread(self._page, request)
        response["timing_ms"] = round((time.perf_counter() - started) * 1000, 2)
        sequence = response["sequence"]
        logger.info(
            "recommendation_page_completed",
            extra={
                "context": {
                    "source": sequence["source"],
                    "offset": sequence["offset"],
                    "returned": sequence["returned"],
                    "windows_computed": sequence["windows_computed"],
                    "cache": sequence["cache"],
                    "has_more": sequence["has_more"],
                    "duration_ms": response["timing_ms"],
                }
            },
        )
        return response

    def _page(self, request: RecommendPageRequest) -> dict[str, Any]:
        # The request alone defines the sequence. The stored session profile is
        # deliberately not merged in: it can change between two pages, and it
        # does not survive an instance restart, so a sequence that depended on
        # it could not be rebuilt identically.
        inputs = self._profile_inputs(
            request.liked_ids,
            request.excluded_ids,
            [*request.excluded_titles, *request.seen_titles],
            request.session_profile,
        )
        fingerprint = sequence_fingerprint(
            {
                "request": request.model_dump(exclude={"cursor", "limit", "session_id"}),
                "versions": self._sequence_versions(),
            }
        )
        offset = 0
        if request.cursor:
            cursor_fingerprint, offset = decode_cursor(request.cursor)
            if cursor_fingerprint != fingerprint:
                # A different profile, or a deployment that changed what ranks
                # it. Either way the old position means nothing here, and
                # serving from it could repeat or skip titles.
                raise AppError(
                    "This recommendation sequence has changed; start a new one",
                    code="sequence_expired",
                    status_code=409,
                )

        sequence, cached = self._sequences.get_or_create(fingerprint)
        with sequence.lock:
            was_ranked = len(sequence.items)
            self._rank_until(sequence, inputs, request, offset, offset + request.limit)
            ids = sequence.items[offset : offset + request.limit]
            ranked = len(sequence.items)
            exhausted = sequence.exhausted
            source = sequence.source or "fast"
            windows = sequence.windows
            window_sizes = list(sequence.window_sizes)
            diagnostics = dict(sequence.diagnostics)
            payloads = {anime_id: sequence.payloads[anime_id] for anime_id in ids if anime_id in sequence.payloads}

        items = [
            payloads[anime_id]
            if anime_id in payloads
            else self.recommender.public_item(self.recommender.by_id[anime_id])
            for anime_id in ids
            if anime_id in payloads or anime_id in self.recommender.by_id
        ]
        end = offset + len(ids)
        has_more = end < ranked or not exhausted
        diagnostics["profile"] = self._profile_counts(
            inputs, len(request.liked_ids), len(request.excluded_ids), request.one_per_series
        )
        return {
            "results": items,
            "sequence": {
                "id": fingerprint[:16],
                "offset": offset,
                "returned": len(items),
                "next_cursor": encode_cursor(fingerprint, end) if has_more else None,
                "has_more": has_more,
                "exhausted": not has_more,
                "source": source,
                "ranked_so_far": ranked,
                "windows_computed": windows,
                "window_sizes": window_sizes,
                # "hit" means no ranking ran for this page.
                "cache": "hit" if cached and ranked == was_ranked else "miss",
                **self.sequence_limits(),
            },
            "diagnostics": diagnostics,
        }

    def _sequence_versions(self) -> dict[str, Any]:
        """Everything outside the request that decides the order of a sequence."""
        reranker = self.fast_path_config.reranker
        reranker_info = reranker.model_info() if reranker is not None else {}
        return {
            "sequence": SEQUENCE_VERSION,
            **self._artifact_versions(),
            "reranker_model_sha256": reranker_info.get("model_sha256"),
            "reranker_artifact_sha256": reranker_info.get("artifact_sha256"),
            "catalog_items": len(self.recommender.catalog),
            "als_available": self.als_index is not None,
            **self.sequence_limits(),
        }

    def _rank_until(
        self,
        sequence: RankedSequence,
        inputs: _ProfileInputs,
        request: RecommendPageRequest,
        offset: int,
        needed: int,
    ) -> None:
        """Rank more of `sequence` until it holds `needed` titles or runs out.

        A starter sequence is the exception: once there is something to return
        at `offset`, at most one further chunk is ranked per request, and the
        page may come back shorter than asked. The hybrid is slow, and a short
        first page now is better than a full one seconds later.
        """
        starter_chunks = 0
        while len(sequence.items) < needed and not sequence.exhausted:
            if sequence.source != "hybrid" and self.als_index is not None and inputs.positives:
                self._rank_next_window(sequence, inputs, request.one_per_series)
                if sequence.source == "fast":
                    continue
            # No Likes the collaborative model knows: a starter list instead.
            if starter_chunks and len(sequence.items) > offset:
                break
            self._rank_starter(sequence, inputs, request)
            starter_chunks += 1

    def _rank_next_window(self, sequence: RankedSequence, inputs: _ProfileInputs, one_per_series: bool) -> None:
        """Append the next ALS window, ordered exactly as production orders one.

        Candidates of earlier windows are excluded at retrieval, so ALS returns
        the next `window_size` titles by its own score. They then go through the
        unchanged fast path -- filters, quality blend, LambdaMART -- with a limit
        equal to the window, so the whole window comes back in learned order.
        """
        assert self.als_index is not None
        window_size = int(self.fast_path_config.retrieval.als_top_n)
        excluded = [*inputs.excluded, *sorted(sequence.consumed)]
        result = recommend_fast(
            inputs.positives,
            catalog_by_id=self.recommender.by_id,
            als_source=self.als_index,
            fallback_source=self.fallback_index,
            tail_source=self.tail_index,
            quality_lookup=self.als_index,
            profile_rows=inputs.positives,
            excluded_ids=excluded,
            limit=window_size,
            config=self.fast_path_config,
        )
        window = list(result.anime_ids)
        retrieved = set(window)
        if result.diagnostics.get("candidate_pool_size", 0) > len(window):
            # Some retrieved titles are not in the served catalog. They must
            # still count as consumed, or every later window would retrieve them
            # again and shrink.
            retrieved.update(self.als_index.top_candidates(inputs.positives, window_size, excluded_ids=excluded))
        if not retrieved:
            if sequence.source is None:
                return  # nothing the model can rank: the caller falls back to a starter list
            sequence.exhausted = True
            return

        if sequence.source is None:
            sequence.source = "fast"
            sequence.diagnostics = {**result.diagnostics, "recommendation_path": "fast"}
            sequence.diagnostics["artifact_versions"] = self._artifact_versions()
            if self.debug_diagnostics:
                sequence.diagnostics["debug"] = self._debug_block(inputs.positives, excluded, result, window[:50])
            if one_per_series:
                sequence.seen_series = self._series_of(inputs.positives)
        sequence.consumed.update(retrieved)
        accepted = self._accept(window, sequence.seen_series)
        sequence.items.extend(accepted)
        sequence.window_sizes.append(len(accepted))
        sequence.windows += 1
        if sequence.windows >= self.sequence_windows:
            sequence.exhausted = True

    def _rank_starter(self, sequence: RankedSequence, inputs: _ProfileInputs, request: RecommendPageRequest) -> None:
        """Append the next chunk of the starter list, ranked by the hybrid.

        Each chunk excludes the chunks before it, which is how the hybrid is
        paged everywhere else. The list stays deterministic, so any instance
        rebuilds the same chunks from the same request.
        """
        remaining = self.starter_depth - len(sequence.items)
        diagnostics: dict[str, Any] = {}
        results = self.recommender.recommend(
            liked_ids=list(inputs.positives),
            excluded_ids=[*inputs.excluded, *sequence.items],
            excluded_titles=list(request.excluded_titles),
            seen_titles=list(request.seen_titles),
            session_profile=dict(request.session_profile),
            one_per_series=request.one_per_series,
            # The hybrid keeps its production series policy here, which is
            # broader than the fast path's: it hides the whole franchise of
            # every excluded title, not only of Liked ones. Whether that is too
            # aggressive is an open policy question (see
            # experiments/v21_quality/SERIES_POLICY.md); it is not changed here.
            limit=min(STARTER_CHUNK, remaining),
            diagnostics=diagnostics,
        )
        blocked = set(inputs.excluded)
        added = 0
        for item in results:
            anime_id = int(item["id"])
            if anime_id in blocked or anime_id in sequence.payloads:
                continue
            sequence.items.append(anime_id)
            sequence.payloads[anime_id] = item
            added += 1
        if sequence.source is None:
            sequence.source = "hybrid"
            sequence.diagnostics = {
                **diagnostics,
                "recommendation_path": "constraint_rich",
                "path_reason": "no_positives_known_to_the_collaborative_model",
            }
        sequence.window_sizes.append(added)
        sequence.windows += 1
        if added == 0 or len(sequence.items) >= self.starter_depth:
            sequence.exhausted = True

    def _series_of(self, anime_ids: Iterable[int]) -> set[str]:
        keys: set[str] = set()
        for anime_id in anime_ids:
            item = self.recommender.by_id.get(int(anime_id))
            if item is not None:
                keys.add(series_key(str(item.get("title") or "")))
        return keys

    def _accept(self, ranked: Sequence[int], seen_series: set[str] | None) -> list[int]:
        """Keep the first entry of each franchise not seen earlier in the sequence.

        The same rule as `_collapse_series`, with the seen set carried across
        windows so a franchise shown on page 2 does not return on page 14.
        """
        kept: list[int] = []
        for anime_id in ranked:
            item = self.recommender.by_id.get(int(anime_id))
            if item is None:
                continue
            if seen_series is not None:
                key = series_key(str(item.get("title") or ""))
                if key in seen_series:
                    continue
                seen_series.add(key)
            kept.append(int(anime_id))
        return kept

    def _exact_title_ids(self, titles: Iterable[Any]) -> tuple[list[int], int]:
        """Catalog IDs whose title equals a requested title, and the miss count.

        Exact, case-insensitive matching on purpose. `resolve_titles` takes the
        top fuzzy search hit, which is right for "more like X" and wrong for an
        exclusion: a near miss would hide a title the user never rejected. It is
        also a catalog scan per title, where this is one dictionary lookup.
        """
        if self._title_ids is None:
            index: dict[str, list[int]] = {}
            for item in self.recommender.catalog:
                key = _title_key(item.get("title") or "")
                if key:
                    index.setdefault(key, []).append(int(item["id"]))
            self._title_ids = {key: tuple(values) for key, values in index.items()}
        resolved: list[int] = []
        unresolved = 0
        for key in dict.fromkeys(_title_key(value) for value in titles):
            if not key:
                continue
            matches = self._title_ids.get(key)
            if matches is None:
                unresolved += 1
            else:
                resolved.extend(matches)
        return list(dict.fromkeys(resolved)), unresolved

    def _debug_block(
        self,
        positives: list[int],
        excluded: list[int],
        result: Any,
        returned_ids: list[int],
    ) -> dict[str, Any]:
        """Evidence that a profile change moved the model, for developers.

        The digest is of the folded-in user vector, so two requests with the
        same likes agree and any Like or Unlike the model knows about changes
        it. The ID lists show the same thing one stage later: what ALS
        retrieved, and what the reranker did with it.
        """
        assert self.als_index is not None
        block: dict[str, Any] = {"als_user_vector_sha256_16": None, "als_user_vector_norm": None}
        vector = self.als_index.user_vector(positives)
        if vector is not None:
            rounded = np.round(np.asarray(vector, dtype=np.float32), 5).astype("<f4")
            block["als_user_vector_sha256_16"] = hashlib.sha256(rounded.tobytes()).hexdigest()[:16]
            block["als_user_vector_norm"] = round(float(np.linalg.norm(vector)), 6)
        block["als_retrieved_top_ids"] = self.als_index.top_candidates(positives, DEBUG_TOP_IDS, excluded_ids=excluded)
        block["reranked_top_ids"] = [candidate.anime_id for candidate in result.candidates[:DEBUG_TOP_IDS]]
        block["returned_ids"] = list(returned_ids)
        return block

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
