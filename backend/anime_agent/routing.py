"""Segment-aware collaborative routing.

Which collaborative source should serve a request depends on how much history
the user has. The offline evidence is specific about this:

| Segment | ALS vs CountSketch NDCG@10 | 95% CI | Conclusion |
|---|---|---|---|
| Sparse (1-4 positives)   | +13.9% | [-0.0259, +0.0732] | not demonstrated |
| Medium (5-19 positives)  | +45.9% | [+0.0224, +0.1375] | ALS better |
| Heavy (20+ positives)    | +97.9% | [+0.0898, +0.1715] | ALS better |

(Activity-balanced diagnostic, 100 users per segment, threshold 8.)

ALS is therefore the global default. An earlier policy routed sparse users to
CountSketch, but a direct production-architecture test later found that policy
reduced sparse-user NDCG@10 from 0.2003 to 0.1660 (n=30). Segment-aware routing
is retained as an opt-in experiment and is disabled by default.

The thresholds mirror the offline activity segmentation, but are applied to a
*live* profile: a session's known positives, not a training row. They are
configurable because the offline boundaries were chosen for evaluation
convenience, not from a product study.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Offline activity segmentation, reused so live routing and the diagnostics
# that justified it describe the same populations.
SPARSE_MAX_POSITIVES = 4
MEDIUM_MAX_POSITIVES = 19


class CollaborativeRoute(Enum):
    """Which collaborative source serves this request.

    A plain Enum rather than StrEnum: the project supports Python 3.10, where
    StrEnum does not exist. Callers use `.value` for serialization.
    """

    ALS = "als"
    SPARSE_FALLBACK = "sparse_fallback"
    NO_COLLABORATIVE = "no_collaborative"
    ALS_UNAVAILABLE_FALLBACK = "als_unavailable_fallback"


@dataclass(frozen=True)
class RoutingPolicy:
    """Configurable thresholds for collaborative routing.

    `medium_threshold` is the number of known positives at which ALS becomes the
    primary source. It defaults to one above the sparse ceiling, so the sparse
    segment defined offline is exactly the segment routed away from ALS.
    """

    medium_threshold: int = SPARSE_MAX_POSITIVES + 1
    # Default OFF. The production architecture benchmark measured routing
    # directly and it made sparse users *worse*: NDCG@10 0.1660 when routed to
    # CountSketch against 0.2003 under global ALS (n=30 sparse users). The
    # earlier "no demonstrated ALS gain for sparse users" finding meant the
    # interval included zero, not that CountSketch was better. Routing away from
    # ALS therefore costs relevance and buys only ~5% catalog coverage.
    #
    # The mechanism is retained and configurable so the decision stays testable
    # on a larger sparse sample.
    segment_aware: bool = False
    version: str = "routing-v2"

    def __post_init__(self) -> None:
        if self.medium_threshold < 1:
            raise ValueError("medium_threshold must be at least 1")


@dataclass(frozen=True)
class RoutingDecision:
    """The chosen route plus why, for the recommendation trace."""

    route: CollaborativeRoute
    known_positive_count: int
    reason: str
    policy_version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "collaborative_route": self.route.value,
            "known_positive_count": self.known_positive_count,
            "reason": self.reason,
            "policy_version": self.policy_version,
        }


def activity_segment(known_positive_count: int) -> str:
    """Segment name matching the offline activity diagnostics."""
    if known_positive_count <= 0:
        return "cold"
    if known_positive_count <= SPARSE_MAX_POSITIVES:
        return "sparse"
    if known_positive_count <= MEDIUM_MAX_POSITIVES:
        return "medium"
    return "heavy"


def choose_collaborative_route(
    positive_ids: Sequence[int],
    *,
    policy: RoutingPolicy | None = None,
    als_available: bool = True,
) -> RoutingDecision:
    """Pick the collaborative source for one request.

    `als_available` lets the caller degrade cleanly when the ALS artifact could
    not be loaded, rather than failing the request.
    """
    policy = policy or RoutingPolicy()
    count = len({int(value) for value in positive_ids})

    if count == 0:
        return RoutingDecision(
            CollaborativeRoute.NO_COLLABORATIVE,
            count,
            "no_known_positives",
            policy.version,
        )
    if not als_available:
        return RoutingDecision(
            CollaborativeRoute.ALS_UNAVAILABLE_FALLBACK,
            count,
            "als_artifact_unavailable",
            policy.version,
        )
    if not policy.segment_aware:
        return RoutingDecision(
            CollaborativeRoute.ALS,
            count,
            "segment_aware_routing_disabled",
            policy.version,
        )
    if count >= policy.medium_threshold:
        segment = activity_segment(count)
        return RoutingDecision(
            CollaborativeRoute.ALS,
            count,
            f"{segment}_activity_user",
            policy.version,
        )
    return RoutingDecision(
        CollaborativeRoute.SPARSE_FALLBACK,
        count,
        "als_sparse_gain_not_demonstrated",
        policy.version,
    )
