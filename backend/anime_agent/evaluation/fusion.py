"""Learn the hybrid recommender's channel-blend weights from held-out data.

The production scorer is a linear blend of ten normalized channel signals:

    score(item) = sum_c effective_weight[c] * signal[c](item)

Those weights have historically been hand-set constants. Because the scorer is
linear in the signals, the weights can be fitted directly rather than guessed,
and the fitted vector drops back into the same serving path with no change to
how a score is computed or explained.

The objective is pairwise, not pointwise. For a user with a held-out positive
`p` and a non-relevant candidate `n`, the model is trained to satisfy
`w . x_p > w . x_n`, optimising a RankNet-style logistic loss over the score
difference. This matches how the model is used -- ordering candidates -- and it
removes the need for an intercept or for calibrated probabilities.

Weights are constrained to be non-negative by projected gradient descent, which
keeps the learned vector inside the space the serving code already accepts
(`normalize_weights` clamps negatives to zero) and keeps every channel's
contribution interpretable as a share of the total.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..recommender import DEFAULT_CHANNEL_WEIGHTS, AnimeRecommender, normalize_weights
from .split import UserSplit

CHANNELS: tuple[str, ...] = tuple(DEFAULT_CHANNEL_WEIGHTS)
FUSION_ARTIFACT_VERSION = 1


@dataclass
class PairwiseDataset:
    """Feature differences for (positive, negative) candidate pairs."""

    differences: npt.NDArray[Any]
    users: int = 0
    positives_covered: int = 0
    positives_missed: int = 0
    channels: tuple[str, ...] = field(default_factory=lambda: CHANNELS)

    def __len__(self) -> int:
        return int(self.differences.shape[0])


def extract_channel_signals(
    recommender: AnimeRecommender,
    liked_ids: Sequence[int],
    excluded_ids: Sequence[int],
    *,
    shortlist: int,
) -> dict[int, npt.NDArray[Any]]:
    """Return per-channel signal vectors for a user's top-``shortlist`` candidates.

    Signals are read from the recommender's own score breakdown, so the features
    are exactly the numbers the production scorer blends -- there is no
    reimplementation to drift out of sync.
    """
    results = recommender.recommend(
        liked_ids=list(liked_ids),
        excluded_ids=list(excluded_ids),
        session_profile={},
        diversity_strength=0.0,
        exclude_related_series=False,
        limit=shortlist,
        # Reason strings are not features; skipping them avoids an expensive
        # per-item explain() pass over the whole shortlist.
        include_explanations=False,
        include_score_breakdown=True,
    )
    signals: dict[int, npt.NDArray[Any]] = {}
    for item in results:
        breakdown = (item.get("score_breakdown") or {}).get("channels") or {}
        if not breakdown:
            continue
        signals[int(item["id"])] = np.asarray(
            [float(breakdown.get(channel, {}).get("raw_score", 0.0)) for channel in CHANNELS],
            dtype=np.float64,
        )
    return signals


def build_pairwise_dataset(
    recommender: AnimeRecommender,
    users: Iterable[UserSplit],
    *,
    holdout: str = "validation",
    shortlist: int = 400,
    negatives_per_positive: int = 8,
    seed: int = 42,
) -> PairwiseDataset:
    """Assemble pairwise training rows from held-out positives.

    Candidates come from the recommender's own top-``shortlist`` shortlist for
    each user. Held-out positives that the shortlist never surfaces cannot form
    a pair and are counted in ``positives_missed`` rather than silently dropped:
    the fit optimises ordering *within* the retrieved region, which is the region
    the deployed ranker actually reorders.
    """
    if holdout not in {"validation", "test"}:
        raise ValueError("holdout must be 'validation' or 'test'")
    if shortlist < 2 or negatives_per_positive < 1:
        raise ValueError("shortlist must be at least 2 and negatives_per_positive positive")

    generator = np.random.default_rng(seed)
    rows: list[npt.NDArray[Any]] = []
    users_used = 0
    covered = 0
    missed = 0

    for user in users:
        relevant = set(user.validation_positive_ids if holdout == "validation" else user.test_positive_ids)
        if not relevant:
            continue
        known_nonpositive = [anime_id for anime_id, _rating in (*user.explicit_negative, *user.neutral, *user.ignored)]
        signals = extract_channel_signals(
            recommender,
            user.train_positive_ids,
            known_nonpositive,
            shortlist=shortlist,
        )
        if not signals:
            continue

        positive_ids = [anime_id for anime_id in signals if anime_id in relevant]
        negative_ids = [anime_id for anime_id in signals if anime_id not in relevant]
        missed += len(relevant) - len(positive_ids)
        if not positive_ids or not negative_ids:
            continue

        users_used += 1
        covered += len(positive_ids)
        negative_pool = np.asarray(negative_ids, dtype=np.int64)
        for positive_id in positive_ids:
            sampled = generator.choice(
                negative_pool,
                size=min(negatives_per_positive, len(negative_pool)),
                replace=False,
            )
            positive_vector = signals[positive_id]
            for negative_id in sampled.tolist():
                rows.append(positive_vector - signals[int(negative_id)])

    if not rows:
        raise ValueError("No usable (positive, negative) pairs were produced")

    return PairwiseDataset(
        differences=np.vstack(rows),
        users=users_used,
        positives_covered=covered,
        positives_missed=missed,
    )


def fit_pairwise_weights(
    dataset: PairwiseDataset,
    *,
    learning_rate: float = 0.5,
    iterations: int = 600,
    l2: float = 1e-3,
    seed: int = 42,
    initial_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit non-negative blend weights with projected gradient descent.

    Loss is ``mean(log(1 + exp(-w . d)))`` over pair differences ``d``, plus L2.
    After each step the weight vector is clipped at zero and renormalised to sum
    to one, so the optimiser only ever visits vectors the serving code accepts.

    ``initial_weights`` sets the starting point. It matters for a retired
    channel: starting a non-negative optimiser at exactly zero is not a fair
    test of whether that channel deserves weight, so a re-examination should
    start it above zero and let the data push it back down.
    """
    if learning_rate <= 0.0 or iterations < 1:
        raise ValueError("learning_rate must be positive and iterations at least one")

    differences = np.asarray(dataset.differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    start = dict(initial_weights) if initial_weights else dict(DEFAULT_CHANNEL_WEIGHTS)
    missing = set(CHANNELS).difference(start)
    if missing:
        raise ValueError("initial_weights is missing channels: " + ", ".join(sorted(missing)))
    weights = np.asarray([float(start[channel]) for channel in CHANNELS], dtype=np.float64)
    if np.any(weights < 0.0):
        raise ValueError("initial_weights must be non-negative")
    weights = weights + generator.normal(0.0, 1e-4, size=weights.shape)
    weights = np.maximum(weights, 0.0)
    weights /= max(weights.sum(), 1e-12)

    history: list[float] = []
    for _iteration in range(iterations):
        margins = differences @ weights
        # Stable sigmoid(-margin); this is the gradient coefficient of the loss.
        coefficients = np.where(
            margins >= 0.0,
            np.exp(-margins) / (1.0 + np.exp(-margins)),
            1.0 / (1.0 + np.exp(margins)),
        )
        gradient = -(differences * coefficients[:, None]).mean(axis=0) + l2 * weights
        weights = weights - learning_rate * gradient
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total <= 1e-12:
            raise ValueError("Projected gradient collapsed every channel weight to zero")
        weights /= total
        history.append(float(np.logaddexp(0.0, -(differences @ weights)).mean()))

    learned = {channel: float(weight) for channel, weight in zip(CHANNELS, weights.tolist(), strict=True)}
    return {
        "weights": normalize_weights(learned),
        "final_loss": history[-1],
        "initial_loss": history[0],
        "pairwise_accuracy": float(np.mean((differences @ weights) > 0.0)),
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "l2": float(l2),
        "initial_weights": {channel: float(start[channel]) for channel in CHANNELS},
        "pairs": int(len(dataset)),
        "users": int(dataset.users),
        "positives_covered": int(dataset.positives_covered),
        "positives_missed": int(dataset.positives_missed),
    }


def baseline_pairwise_accuracy(dataset: PairwiseDataset, weights: Mapping[str, float]) -> float:
    """Pairwise accuracy of a given weight vector, for a like-for-like comparison."""
    vector = np.asarray([float(weights.get(channel, 0.0)) for channel in CHANNELS], dtype=np.float64)
    total = vector.sum()
    if total > 0:
        vector = vector / total
    return float(np.mean((np.asarray(dataset.differences, dtype=np.float64) @ vector) > 0.0))


def save_fusion_artifact(result: Mapping[str, Any], output_path: Path, *, split_sha256: str) -> dict[str, Any]:
    """Write learned weights plus the evidence needed to audit them."""
    payload = {
        "artifact_version": FUSION_ARTIFACT_VERSION,
        "method": "pairwise RankNet-style logistic fit with non-negative projection",
        "channels": list(CHANNELS),
        "split_sha256": split_sha256,
        "trained_at_unix": int(time.time()),
        "baseline_weights": dict(DEFAULT_CHANNEL_WEIGHTS),
        **dict(result),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def load_fusion_weights(path: Path) -> dict[str, float]:
    """Load learned weights, rejecting artifacts that do not match this schema."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_version") != FUSION_ARTIFACT_VERSION:
        raise ValueError("Unsupported fusion artifact version")
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Fusion artifact does not contain a weights mapping")
    missing = set(CHANNELS).difference(weights)
    if missing:
        raise ValueError("Fusion artifact is missing channels: " + ", ".join(sorted(missing)))
    values = {channel: float(weights[channel]) for channel in CHANNELS}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("Fusion weights must be finite and non-negative")
    return normalize_weights(values)
