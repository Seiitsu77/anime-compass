"""Headless service backing the portfolio demo.

The demo runs in the same process as the model, so it calls this directly
rather than going Streamlit -> HTTP -> FastAPI -> service. The FastAPI app is
unchanged and still works; this is a second, thinner entry point over the same
`recommend_fast` core.

Deliberately stateless: no session database, no user accounts, no writes. A
visitor's likes and dislikes live in the browser session only.
"""

from __future__ import annotations

import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .als_serving import (
    ARTIFACT_ROLE_PRODUCTION,
    ALSArtifactError,
    ALSCollaborativeIndex,
)
from .fast_path import FastPathConfig, recommend_fast
from .path_policy import RecommendationPath, choose_recommendation_path
from .recommender import series_key

# Curated starting points so a visitor who does not know anime can still use the
# demo. These only pre-populate the "liked" list; nothing about the resulting
# recommendations is hard-coded.
EXAMPLE_PROFILES: dict[str, tuple[str, tuple[int, ...]]] = {
    "Sci-Fi / Psychological": (
        "Steins;Gate, Death Note, Psycho-Pass",
        (9253, 1535, 13601),
    ),
    "Shonen / Action": (
        "Fullmetal Alchemist: Brotherhood, Hunter x Hunter, Attack on Titan",
        (5114, 11061, 16498),
    ),
    "Romance / Drama": (
        "Your Lie in April, Clannad: After Story, Toradora!",
        (23273, 4181, 4224),
    ),
    "Fantasy / Adventure": (
        "Made in Abyss, Mushishi, Spirited Away",
        (34599, 457, 199),
    ),
}


@dataclass
class ModelHealth:
    """What the demo may honestly claim about the model it is serving."""

    als_available: bool
    artifact_role: str | None
    artifact_sha256: str | None
    catalog_items: int
    als_covered_items: int
    cold_start_items: int
    load_seconds: float
    error: str | None = None
    reranker_available: bool = False
    reranker_detail: str = "not configured"

    @property
    def serving_production_als(self) -> bool:
        return self.als_available and self.artifact_role == ARTIFACT_ROLE_PRODUCTION

    @property
    def headline(self) -> str:
        """What is actually serving, never what was hoped for."""
        if not self.als_available:
            return "Unavailable"
        stage = "ALS" if self.serving_production_als else f"ALS ({self.artifact_role or 'unknown role'})"
        if self.serving_production_als:
            stage = "Production ALS"
        return f"{stage} + LambdaMART" if self.reranker_available else stage

    @property
    def reranker_headline(self) -> str:
        return "LambdaMART active" if self.reranker_available else "degraded to ALS"


@dataclass
class Recommendation:
    """One result, shaped for display rather than for scoring."""

    anime_id: int
    title: str
    year: int | None
    media_type: str | None
    episodes: int | None
    score: float | None
    members: int | None
    genres: tuple[str, ...]
    image_url: str | None
    synopsis: str
    explanation: str
    because_of: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class RecommendationResult:
    items: list[Recommendation]
    path: str
    latency_ms: float
    candidate_pool_size: int
    diagnostics: dict[str, Any]


def _fold(text: str) -> str:
    """Case- and accent-insensitive key for title search."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


# The demo shows a short synopsis, so the serving catalog ships them already
# trimmed. This function is the single definition of that trim and is
# idempotent, which is what makes a pre-trimmed catalog provably identical to
# the full one at the display layer.
SYNOPSIS_DISPLAY_CHARS = 200


def truncate_synopsis(text: Any, limit: int = SYNOPSIS_DISPLAY_CHARS) -> str:
    """Trim a synopsis to display length. Applying it twice changes nothing."""
    synopsis = str(text or "")
    if len(synopsis) <= limit:
        return synopsis
    return synopsis[: limit - 3].rstrip() + "..."


class ShowcaseService:
    """Search the catalog and produce explained recommendations."""

    def __init__(
        self,
        catalog: Sequence[Mapping[str, Any]],
        als_index: ALSCollaborativeIndex | None,
        *,
        health: ModelHealth,
        config: FastPathConfig | None = None,
    ):
        self.catalog = list(catalog)
        self.als_index = als_index
        self.health = health
        # Which catalog file the deployment actually loaded. The compact serving
        # catalog and the full one produce identical recommendations, so the
        # health panel names the file rather than leaving it ambiguous.
        self.catalog_source: str = "unknown"
        self.config = config or FastPathConfig()
        self.by_id: dict[int, Mapping[str, Any]] = {int(item["id"]): item for item in self.catalog}
        self._search_keys = [(int(item["id"]), _fold(item.get("title", ""))) for item in self.catalog]
        # Popularity orders search results so "steins" surfaces Steins;Gate
        # rather than an obscure side story.
        self._members = {int(item["id"]): int(item.get("members") or 0) for item in self.catalog}

    # ---------------------------------------------------------------- search

    def search(self, query: str, limit: int = 20) -> list[Mapping[str, Any]]:
        """Prefix- and substring-match titles, most popular first."""
        needle = _fold(query).strip()
        if not needle:
            return []
        starts: list[tuple[int, int]] = []
        contains: list[tuple[int, int]] = []
        for anime_id, key in self._search_keys:
            position = key.find(needle)
            if position == 0:
                starts.append((anime_id, self._members[anime_id]))
            elif position > 0:
                contains.append((anime_id, self._members[anime_id]))
        starts.sort(key=lambda pair: -pair[1])
        contains.sort(key=lambda pair: -pair[1])
        ordered = [anime_id for anime_id, _ in starts] + [anime_id for anime_id, _ in contains]
        return [self.by_id[anime_id] for anime_id in ordered[:limit]]

    def popular(self, limit: int = 40) -> list[Mapping[str, Any]]:
        ranked = sorted(self.catalog, key=lambda item: -(int(item.get("members") or 0)))
        return ranked[:limit]

    def title_of(self, anime_id: int) -> str:
        item = self.by_id.get(int(anime_id))
        return str(item["title"]) if item else f"#{anime_id}"

    def is_cold_start(self, anime_id: int) -> bool:
        """True when the catalog knows the title but ALS has no factor for it."""
        if self.als_index is None:
            return True
        return int(anime_id) not in self.als_index.index_by_id

    # ------------------------------------------------------- recommendation

    def recommend(
        self,
        liked_ids: Sequence[int],
        *,
        disliked_ids: Sequence[int] = (),
        limit: int = 12,
        free_text: str = "",
        one_per_series: bool = True,
    ) -> RecommendationResult:
        """Recommend from a profile, using the fast ALS path.

        `free_text` is recorded and used only to report which path a real
        request would take. This service never invokes the LLM, so the demo
        works with no API key.

        `one_per_series` is a **display** filter, not a model change. ALS
        correctly ranks Steins;Gate side stories highly for someone who liked
        Steins;Gate, but a page of one franchise reads as a broken recommender.
        Extra candidates are requested and collapsed by series afterwards, so
        the ranking itself is untouched.
        """
        if self.als_index is None:
            raise ALSArtifactError("The production ALS model is not available in this deployment.")

        liked = [int(value) for value in liked_ids if int(value) in self.by_id]
        if not liked:
            return RecommendationResult([], "fast", 0.0, 0, {"reason": "no_liked_titles"})

        excluded = [int(value) for value in disliked_ids]
        # Ask for headroom so series collapsing still fills the page.
        fetch = limit * 4 if one_per_series else limit
        started = time.perf_counter()
        result = recommend_fast(
            liked,
            catalog_by_id=self.by_id,
            als_source=self.als_index,
            fallback_source=None,
            quality_lookup=self.als_index,
            profile_rows=liked,
            excluded_ids=excluded,
            limit=fetch,
            config=self.config,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        ranked = result.anime_ids
        if one_per_series:
            ranked = self._collapse_series(ranked, liked)
        ranked = ranked[:limit]

        decision = choose_recommendation_path({"intent": "recommend", "free_text_preferences": free_text})
        items = [self._present(anime_id, liked) for anime_id in ranked]
        return RecommendationResult(
            items=[item for item in items if item is not None],
            path=RecommendationPath.FAST.value,
            latency_ms=latency_ms,
            candidate_pool_size=int(result.diagnostics.get("candidate_pool_size", 0)),
            diagnostics={**result.diagnostics, "would_route_to": decision.path.value},
        )

    def _collapse_series(self, ranked: Sequence[int], liked_ids: Sequence[int]) -> list[int]:
        """Keep the best-ranked entry per franchise, and drop the profile's own.

        Recommending three Steins;Gate side stories to someone who listed
        Steins;Gate is defensible ranking and a poor answer.
        """
        seen: set[str] = set()
        for liked_id in liked_ids:
            item = self.by_id.get(int(liked_id))
            if item is not None:
                seen.add(series_key(str(item.get("title") or "")))

        kept: list[int] = []
        for anime_id in ranked:
            item = self.by_id.get(int(anime_id))
            if item is None:
                continue
            key = series_key(str(item.get("title") or ""))
            if key in seen:
                continue
            seen.add(key)
            kept.append(int(anime_id))
        return kept

    # -------------------------------------------------------- explanations

    def _closest_liked(self, anime_id: int, liked_ids: Sequence[int], top: int = 2) -> list[str]:
        """Which of the user's likes this result actually resembles.

        Grounded in the model: cosine similarity between learned item factors.
        Nothing here is generated text, so an explanation cannot claim a
        connection the model did not make.
        """
        index = self.als_index
        if index is None:
            return []
        row = index.index_by_id.get(int(anime_id))
        if row is None:
            return []
        target = index.item_factors[row]
        target_norm = float(np.linalg.norm(target))
        if target_norm <= 0:
            return []

        scored: list[tuple[float, int]] = []
        for liked_id in liked_ids:
            liked_row = index.index_by_id.get(int(liked_id))
            if liked_row is None:
                continue
            vector = index.item_factors[liked_row]
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            scored.append((float(target @ vector) / (target_norm * norm), int(liked_id)))
        scored.sort(reverse=True)
        return [self.title_of(anime_id) for _score, anime_id in scored[:top]]

    def _present(self, anime_id: int, liked_ids: Sequence[int]) -> Recommendation | None:
        item = self.by_id.get(int(anime_id))
        if item is None:
            return None

        because = self._closest_liked(anime_id, liked_ids)
        genres = tuple(str(value) for value in (item.get("genres") or [])[:4])
        shared = self._shared_genres(anime_id, liked_ids)

        if because:
            names = " and ".join(because)
            explanation = f"Your profile overlaps most strongly with {names}."
            if shared:
                explanation += f" Shares {', '.join(shared[:2])}."
        elif shared:
            explanation = f"Matches your preference for {', '.join(shared[:2])}."
        else:
            explanation = "Ranked highly for your overall profile."

        synopsis = truncate_synopsis(item.get("synopsis"))

        return Recommendation(
            anime_id=int(anime_id),
            title=str(item.get("title") or f"#{anime_id}"),
            year=item.get("start_year"),
            media_type=item.get("type"),
            episodes=item.get("episodes"),
            score=item.get("score"),
            members=item.get("members"),
            genres=genres,
            image_url=item.get("image_url") or None,
            synopsis=synopsis,
            explanation=explanation,
            because_of=tuple(because),
        )

    def _shared_genres(self, anime_id: int, liked_ids: Sequence[int]) -> list[str]:
        """Genres this result shares with the profile, most common first."""
        item = self.by_id.get(int(anime_id))
        if item is None:
            return []
        target = {str(value) for value in (item.get("genres") or [])}
        if not target:
            return []
        counts: dict[str, int] = {}
        for liked_id in liked_ids:
            liked = self.by_id.get(int(liked_id))
            if liked is None:
                continue
            for genre in {str(value) for value in (liked.get("genres") or [])} & target:
                counts[genre] = counts.get(genre, 0) + 1
        return [genre for genre, _count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]

    # -------------------------------------------------------------- profiles

    def example_profiles(self) -> dict[str, tuple[str, tuple[int, ...]]]:
        """Example profiles, filtered to titles this catalog actually has."""
        available: dict[str, tuple[str, tuple[int, ...]]] = {}
        for name, (description, ids) in EXAMPLE_PROFILES.items():
            present = tuple(anime_id for anime_id in ids if anime_id in self.by_id)
            if present:
                available[name] = (description, present)
        return available


def load_showcase_service(
    catalog: Sequence[Mapping[str, Any]],
    artifact_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_catalog_ids_sha256: str | None = None,
    require_production: bool = True,
    config: FastPathConfig | None = None,
    reranker_feature_path: Path | None = None,
    reranker_model_path: Path | None = None,
) -> ShowcaseService:
    """Load the model and build the service, recording what actually happened.

    A failure here never raises: the demo must be able to render an honest
    "model unavailable" state rather than a stack trace. What it must *not* do
    is claim to be serving production ALS when it is not, so the health record
    carries the artifact's real role.
    """
    started = time.perf_counter()
    index: ALSCollaborativeIndex | None = None
    error: str | None = None
    try:
        index = ALSCollaborativeIndex.load(
            artifact_path,
            catalog,
            expected_artifact_sha256=expected_sha256,
            expected_role=ARTIFACT_ROLE_PRODUCTION if require_production else None,
            expected_catalog_ids_sha256=expected_catalog_ids_sha256,
        )
    except ALSArtifactError as exc:
        error = str(exc)
    except (OSError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    load_seconds = time.perf_counter() - started

    # The promoted second stage. It is loaded only if ALS itself loaded, and a
    # failure here costs the improvement rather than the recommendations.
    reranker = None
    reranker_detail = "not configured"
    if index is not None and reranker_feature_path and reranker_model_path:
        from .reranker_serving import try_load_reranker

        reranker = try_load_reranker(
            reranker_feature_path,
            reranker_model_path,
            catalog,
            index.anime_ids,
        )
        reranker_detail = (
            f"LambdaMART, {reranker.model_info()['trees']} trees"
            if reranker is not None
            else "artifact missing or invalid; serving ALS order"
        )
    if reranker is not None:
        config = config or FastPathConfig()
        config = replace(config, reranker=reranker)

    catalog_ids = {int(item["id"]) for item in catalog}
    covered = len(catalog_ids & set(index.anime_ids.tolist())) if index is not None else 0
    info = index.model_info() if index is not None else {}
    health = ModelHealth(
        als_available=index is not None,
        artifact_role=info.get("artifact_role"),
        artifact_sha256=info.get("artifact_sha256"),
        catalog_items=len(catalog_ids),
        als_covered_items=covered,
        cold_start_items=len(catalog_ids) - covered,
        load_seconds=load_seconds,
        error=error,
        reranker_available=reranker is not None,
        reranker_detail=reranker_detail,
    )
    return ShowcaseService(catalog, index, health=health, config=config)


def cold_start_ids(catalog: Iterable[Mapping[str, Any]], index: ALSCollaborativeIndex | None) -> set[int]:
    """Catalog items with no trained ALS factor.

    These are searchable and displayable but must never receive a fabricated
    personalized score.
    """
    ids = {int(item["id"]) for item in catalog}
    if index is None:
        return ids
    return ids - set(index.anime_ids.tolist())
