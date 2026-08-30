"""Lightweight multi-source candidate retrieval.

Retrieval is separated from ranking because the offline measurements showed the
two have very different bottlenecks. Recall of held-out positives at depth 300:

| Source | Recall@300 | Head | Mid-tail | Long-tail | p50 |
|---|---:|---:|---:|---:|---:|
| CountSketch | 0.5499 | 0.5577 | 0.0314 | 0.0000 | 5.14 ms |
| ALS | 0.7932 | 0.8100 | 0.0415 | 0.0000 | 9.71 ms |
| item-item | 0.6691 | 0.6800 | 0.0556 | **0.6667** | 6.56 ms |
| ALS + item-item | 0.7978 | — | — | — | 16.44 ms |

ALS dominates overall recall but retrieves nothing outside the head. item-item
is the only cheap source with long-tail reach. Combining them is therefore not
about beating ALS on aggregate recall -- it barely does -- but about restoring
tail coverage ALS structurally lacks.

Scores from different sources are not comparable, so they are never summed
directly. Each source's ranking is converted to a within-source rank score in
(0, 1], which is monotone in that source's own ordering and bounded
identically across sources.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class CandidateSource(Protocol):
    """A retrieval source: given positives, return ranked anime IDs."""

    def top_candidates(
        self,
        positive_ids: Sequence[int],
        limit: int,
        *,
        excluded_ids: Sequence[int] = (),
    ) -> list[int]: ...


@dataclass
class Candidate:
    """One retrieved item and every source that surfaced it."""

    anime_id: int
    sources: dict[str, float] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "anime_id": self.anime_id,
            "sources": {name: round(score, 6) for name, score in self.sources.items()},
            "ranks": dict(self.ranks),
        }

    @property
    def fused_score(self) -> float:
        """Best within-source rank score across the sources that found it.

        Taking the max rather than the sum keeps the value on the same bounded
        scale regardless of how many sources surfaced an item, so an item found
        by two weak sources cannot outrank an item ranked first by one strong
        source. Source agreement is preserved in `sources` for observability
        rather than being folded into the score.
        """
        return max(self.sources.values()) if self.sources else 0.0


@dataclass(frozen=True)
class RetrievalConfig:
    """Candidate pool sizes.

    Defaults are the smallest pool that retains most of the measured retrieval
    benefit: ALS at 300 captures 0.7932 recall against 0.8500 at 500, and adding
    100 item-item candidates supplies the tail reach ALS lacks for ~7 ms.
    """

    als_top_n: int = 300
    # Default OFF. Supplementing with item-item restores tail reach ALS lacks,
    # but costs 9.0% relative NDCG@10 (0.2588 -> 0.2355) and 4x the retrieval
    # latency (1.99 ms -> 8.18 ms). Enable it when catalog discovery is the
    # product goal; leave it off when top-10 precision is.
    item_item_top_m: int = 0
    countsketch_top_n: int = 300
    include_tail_source: bool = True
    version: str = "retrieval-v2"

    def __post_init__(self) -> None:
        if self.als_top_n < 1 or self.countsketch_top_n < 1:
            raise ValueError("primary candidate counts must be positive")
        if self.item_item_top_m < 0:
            raise ValueError("item_item_top_m cannot be negative")


@dataclass
class RetrievalResult:
    """Deduplicated candidates plus provenance and timing."""

    candidates: list[Candidate]
    source_counts: dict[str, int]
    duration_ms: float
    config_version: str
    tail_source_used: bool

    @property
    def anime_ids(self) -> list[int]:
        return [candidate.anime_id for candidate in self.candidates]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "candidate_pool_size": len(self.candidates),
            "candidate_sources": dict(self.source_counts),
            "tail_source_used": self.tail_source_used,
            "retrieval_config_version": self.config_version,
            "retrieval_duration_ms": round(self.duration_ms, 3),
        }


def _rank_score(rank: int, total: int) -> float:
    """Map a 0-based rank to a bounded, monotone score in (0, 1].

    Rank normalisation is used rather than min-max because raw ALS dot products
    and item-item cosine sums are on different scales with different
    distributions; min-max would be dominated by each source's outliers.
    """
    if total <= 0:
        return 0.0
    return float(total - rank) / float(total)


def retrieve_candidates(
    positive_ids: Sequence[int],
    sources: Mapping[str, CandidateSource],
    *,
    config: RetrievalConfig | None = None,
    excluded_ids: Sequence[int] = (),
    limits: Mapping[str, int] | None = None,
) -> RetrievalResult:
    """Fetch, deduplicate, and score candidates from every configured source.

    Sources are queried in the order given. An item found by several sources
    keeps one entry carrying every source's rank and score, so downstream code
    can see which channel surfaced it without re-running retrieval.
    """
    config = config or RetrievalConfig()
    default_limits = {
        "als": config.als_top_n,
        "item_item": config.item_item_top_m,
        "countsketch": config.countsketch_top_n,
    }
    resolved = {**default_limits, **(limits or {})}

    started = time.perf_counter()
    by_id: dict[int, Candidate] = {}
    order: list[int] = []
    source_counts: dict[str, int] = {}
    tail_source_used = False

    for name, source in sources.items():
        limit = int(resolved.get(name, config.als_top_n))
        if limit <= 0:
            continue
        ranked = source.top_candidates(positive_ids, limit, excluded_ids=excluded_ids)
        source_counts[name] = len(ranked)
        if name == "item_item" and ranked:
            tail_source_used = True
        total = len(ranked)
        for rank, anime_id in enumerate(ranked):
            anime_id = int(anime_id)
            candidate = by_id.get(anime_id)
            if candidate is None:
                candidate = Candidate(anime_id=anime_id)
                by_id[anime_id] = candidate
                order.append(anime_id)
            candidate.sources[name] = _rank_score(rank, total)
            candidate.ranks[name] = rank

    candidates = [by_id[anime_id] for anime_id in order]
    candidates.sort(key=lambda item: (-item.fused_score, item.anime_id))
    return RetrievalResult(
        candidates=candidates,
        source_counts=source_counts,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        config_version=config.version,
        tail_source_used=tail_source_used,
    )
