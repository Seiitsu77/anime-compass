"""The default personalized recommendation path.

This replaces the ten-channel hybrid as the *default* ranking path. The evidence
for doing so is direct: on the same 800 users, substituting ALS into the hybrid
gave NDCG@10 0.2629 against standalone ALS at 0.2624 -- a difference whose 95%
interval includes zero -- for 947 ms instead of 7.6 ms. The hybrid's other nine
channels were not adding measurable ranking value once the collaborative channel
was strong.

The hybrid is not deleted. It remains the path for constraint-rich requests,
which this benchmark never scored (hard filters, entity joins, explanations).

Pipeline:

    profile -> route -> retrieve -> filter -> rank -> [rerank] -> results

Every stage is timed and every decision is recorded, so the trade-offs stay
observable in production rather than only in an offline report.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .retrieval import (
    Candidate,
    CandidateSource,
    RetrievalConfig,
    retrieve_candidates,
)
from .routing import CollaborativeRoute, RoutingDecision, RoutingPolicy, choose_collaborative_route

# Diversity is disabled by default. ALS costs 2.97% intra-list diversity against
# a 73% relevance gain (4.5 NDCG points per ILD point), which does not justify
# paying for a rerank on every request. The window is small when enabled so the
# correction stays cheap.
DEFAULT_DIVERSITY_STRENGTH = 0.0
DEFAULT_DIVERSITY_WINDOW = 30


@dataclass(frozen=True)
class FastPathConfig:
    """Ranking behaviour for the default path."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    routing: RoutingPolicy = field(default_factory=RoutingPolicy)
    diversity_strength: float = DEFAULT_DIVERSITY_STRENGTH
    diversity_window: int = DEFAULT_DIVERSITY_WINDOW
    # A cheap Bayesian quality prior, blended after retrieval. Kept small: it is
    # already computed for every catalog row, so it costs nothing to consult.
    quality_weight: float = 0.10
    version: str = "fastpath-v1"

    def __post_init__(self) -> None:
        if not 0.0 <= self.diversity_strength <= 1.0:
            raise ValueError("diversity_strength must be between 0 and 1")
        if self.diversity_window < 1:
            raise ValueError("diversity_window must be positive")
        if not 0.0 <= self.quality_weight <= 1.0:
            raise ValueError("quality_weight must be between 0 and 1")


@dataclass
class FastPathResult:
    """Ranked anime IDs plus the full decision trace."""

    anime_ids: list[int]
    candidates: list[Candidate]
    routing: RoutingDecision
    diagnostics: dict[str, Any]

    def trace(self) -> dict[str, Any]:
        return dict(self.diagnostics)


def _genre_key(item: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(str(value).casefold() for value in item.get("genres", []) or [])


def _diversity_penalty(item: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]) -> float:
    """Mean genre overlap against what has already been selected."""
    if not selected:
        return 0.0
    genres = _genre_key(item)
    if not genres:
        return 0.0
    overlaps = []
    for chosen in selected:
        other = _genre_key(chosen)
        if not other:
            continue
        union = genres | other
        overlaps.append(len(genres & other) / len(union) if union else 0.0)
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def recommend_fast(
    positive_ids: Sequence[int],
    *,
    catalog_by_id: Mapping[int, Mapping[str, Any]],
    als_source: CandidateSource | None,
    fallback_source: CandidateSource | None,
    tail_source: CandidateSource | None = None,
    quality_lookup: Any | None = None,
    excluded_ids: Sequence[int] = (),
    allowed_ids: set[int] | None = None,
    limit: int = 10,
    config: FastPathConfig | None = None,
) -> FastPathResult:
    """Run the default personalized path.

    `allowed_ids`, when supplied, is a hard filter computed by the caller (entity
    joins, metadata constraints). Candidates outside it are dropped before
    ranking, never merely down-weighted, so the fast path cannot satisfy a
    required constraint approximately.

    `excluded_ids` covers watched titles, explicit exclusions, dislikes, and
    blocks. They are removed at retrieval time *and* re-checked after ranking, so
    a source that ignores the exclusion list cannot leak an item through.
    """
    config = config or FastPathConfig()
    stage_ms: dict[str, float] = {}
    started = time.perf_counter()

    routing = choose_collaborative_route(
        positive_ids,
        policy=config.routing,
        als_available=als_source is not None,
    )
    stage_ms["routing"] = (time.perf_counter() - started) * 1000.0

    sources: dict[str, CandidateSource] = {}
    if routing.route is CollaborativeRoute.ALS and als_source is not None:
        sources["als"] = als_source
    elif fallback_source is not None:
        sources["countsketch"] = fallback_source
    if config.retrieval.include_tail_source and tail_source is not None:
        sources["item_item"] = tail_source

    blocked = {int(value) for value in excluded_ids}
    retrieval_started = time.perf_counter()
    retrieval = (
        retrieve_candidates(
            positive_ids,
            sources,
            config=config.retrieval,
            excluded_ids=sorted(blocked),
        )
        if sources
        else None
    )
    stage_ms["retrieval"] = (time.perf_counter() - retrieval_started) * 1000.0

    filter_started = time.perf_counter()
    candidates: list[Candidate] = list(retrieval.candidates) if retrieval else []
    filtered: list[Candidate] = []
    for candidate in candidates:
        if candidate.anime_id in blocked:
            continue
        if allowed_ids is not None and candidate.anime_id not in allowed_ids:
            continue
        if candidate.anime_id not in catalog_by_id:
            continue
        filtered.append(candidate)
    stage_ms["filtering"] = (time.perf_counter() - filter_started) * 1000.0

    rank_started = time.perf_counter()
    scored: list[tuple[float, Candidate]] = []
    for candidate in filtered:
        score = candidate.fused_score
        if config.quality_weight > 0.0 and quality_lookup is not None:
            quality = quality_lookup.quality_score(candidate.anime_id)
            if quality is not None:
                score = (1.0 - config.quality_weight) * score + config.quality_weight * float(quality)
        scored.append((score, candidate))
    scored.sort(key=lambda pair: (-pair[0], pair[1].anime_id))
    stage_ms["ranking"] = (time.perf_counter() - rank_started) * 1000.0

    rerank_started = time.perf_counter()
    reranked = _apply_diversity(scored, catalog_by_id, limit, config)
    stage_ms["reranking"] = (time.perf_counter() - rerank_started) * 1000.0

    # Final exclusion re-check: a source that ignored `excluded_ids` must not
    # leak an item into the response.
    results = [candidate.anime_id for candidate in reranked if candidate.anime_id not in blocked][:limit]

    total_ms = (time.perf_counter() - started) * 1000.0
    diagnostics: dict[str, Any] = {
        **routing.as_dict(),
        "path": "fast",
        "fast_path_version": config.version,
        "candidate_pool_size": len(candidates),
        "candidates_after_filters": len(filtered),
        "returned": len(results),
        "diversity_applied": config.diversity_strength > 0.0,
        "diversity_strength": config.diversity_strength,
        "diversity_window": config.diversity_window if config.diversity_strength > 0.0 else 0,
        "hard_filter_applied": allowed_ids is not None,
        "excluded_count": len(blocked),
        "stage_latency_ms": {name: round(value, 3) for name, value in stage_ms.items()},
        "total_latency_ms": round(total_ms, 3),
    }
    if retrieval is not None:
        diagnostics.update(retrieval.diagnostics())
    else:
        diagnostics["candidate_sources"] = {}
        diagnostics["tail_source_used"] = False

    return FastPathResult(
        anime_ids=results,
        candidates=[candidate for _score, candidate in scored[: max(limit * 4, 40)]],
        routing=routing,
        diagnostics=diagnostics,
    )


def _apply_diversity(
    scored: Sequence[tuple[float, Candidate]],
    catalog_by_id: Mapping[int, Mapping[str, Any]],
    limit: int,
    config: FastPathConfig,
) -> list[Candidate]:
    """Greedy diversity rerank over a bounded window.

    The window matters: the old hybrid rescanned `limit * 8` candidates for every
    output slot, which is quadratic in `limit`. Here the candidate window is
    fixed and small, so the cost does not grow with the requested result count.
    """
    if config.diversity_strength <= 0.0:
        return [candidate for _score, candidate in scored[:limit]]

    window = list(scored[: max(config.diversity_window, limit)])
    selected_items: list[Mapping[str, Any]] = []
    chosen: list[Candidate] = []
    while window and len(chosen) < limit:
        best_index = 0
        best_score = -float("inf")
        for index, (score, candidate) in enumerate(window):
            item = catalog_by_id.get(candidate.anime_id)
            penalty = _diversity_penalty(item, selected_items) if item else 0.0
            adjusted = score - config.diversity_strength * penalty
            if adjusted > best_score:
                best_score = adjusted
                best_index = index
        _score, candidate = window.pop(best_index)
        chosen.append(candidate)
        item = catalog_by_id.get(candidate.anime_id)
        if item is not None:
            selected_items.append(item)
    return chosen
