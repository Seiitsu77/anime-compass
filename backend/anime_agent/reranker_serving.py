"""Serve the second-stage reranker over an ALS candidate set.

The decision this implements: LambdaMART over the linear model, because its
gain is paired-significant and because the cost that actually matters -- the
17 MB feature-statistics artifact and the per-request feature construction --
is paid by *both* options. Choosing the simpler model would have saved 5 MB out
of 23 and 0.27 ms out of ~5; it would not have avoided the expensive part.

Two things this module is careful about:

**No train/serve skew.** The feature statistics shipped here are the same
train-only artifacts the reranker was fitted against. Recomputing them over all
production interactions would make them *fresher* and, without refitting, make
the model wrong -- the features would no longer be drawn from the distribution
its trees split on. Consistency beats recency for a frozen model.

**Fail to ALS, never to nothing.** Every failure path -- missing artifact,
checksum mismatch, LightGBM not installed, feature mismatch -- returns the ALS
order unchanged. A reranker that cannot load is a missing improvement, not an
outage, and the fast path predates it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .als_serving import sha256_file

logger = logging.getLogger("anime_compass.reranker")

RERANKER_ARTIFACT_VERSION = 1
# Bumped whenever a feature definition changes. An artifact whose feature list
# differs from the running code is refused rather than silently misinterpreted.
EXPECTED_FEATURE_COUNT = 18


class RerankerUnavailable(RuntimeError):
    """The reranker cannot serve. Callers fall back to the ALS order."""


class LearnedReranker:
    """A frozen LambdaMART booster plus the feature space it was trained on."""

    def __init__(
        self,
        booster: Any,
        feature_space: Any,
        *,
        metadata: Mapping[str, Any],
    ):
        self.booster = booster
        self.feature_space = feature_space
        self.metadata = dict(metadata)

    @property
    def index_by_id(self) -> dict[int, int]:
        return dict(self.feature_space.index_by_id)

    def rerank(
        self,
        profile_rows: Sequence[int],
        candidate_rows: Sequence[int],
        als_scores: npt.NDArray[np.float32],
    ) -> list[int]:
        """Return the candidate rows in learned order.

        This is a permutation and nothing else: it never adds, drops, or
        filters. Hard constraints and exclusions are the caller's job and are
        applied around this call, so a reranker cannot satisfy a required
        constraint approximately.
        """
        if not candidate_rows:
            return []
        features = self.feature_space.build(profile_rows, candidate_rows, als_scores)
        if features.shape[1] != EXPECTED_FEATURE_COUNT:
            raise RerankerUnavailable(
                f"feature count mismatch: built {features.shape[1]}, model expects {EXPECTED_FEATURE_COUNT}"
            )
        scores = np.asarray(self.booster.predict(features), dtype=np.float32)
        order = np.argsort(-scores, kind="stable")
        return [int(candidate_rows[index]) for index in order]

    def model_info(self) -> dict[str, Any]:
        return {
            "available": True,
            "method": "LightGBM LambdaMART over frozen ALS candidates",
            "trees": int(self.booster.num_trees()),
            "features": EXPECTED_FEATURE_COUNT,
            "artifact_version": RERANKER_ARTIFACT_VERSION,
            **{key: self.metadata.get(key) for key in ("artifact_sha256", "model_sha256", "training_source")},
        }


def load_reranker(
    feature_artifact: Path,
    model_artifact: Path,
    catalog: Sequence[Mapping[str, Any]],
    anime_ids: npt.NDArray[np.int64],
    *,
    expected_feature_sha256: str | None = None,
    expected_model_sha256: str | None = None,
) -> LearnedReranker:
    """Load and validate the reranker, or raise `RerankerUnavailable`.

    Validation mirrors the ALS loader deliberately: pinned checksums, a version
    check, and alignment against the catalog the caller is actually serving.
    """
    # Imported lazily so a deployment without the reranker never pays LightGBM's
    # ~650 ms import, and so a missing package degrades instead of crashing.
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - exercised by the env, not the suite
        raise RerankerUnavailable("lightgbm is not installed") from exc

    from .evaluation.reranking import RerankerFeatureSpace

    for path in (feature_artifact, model_artifact):
        if not path.exists():
            raise RerankerUnavailable(f"reranker artifact not found: {path.name}")

    if expected_feature_sha256:
        actual = sha256_file(feature_artifact)
        if actual != expected_feature_sha256:
            raise RerankerUnavailable(
                f"reranker feature checksum mismatch: expected {expected_feature_sha256[:16]}..., got {actual[:16]}..."
            )
    if expected_model_sha256:
        actual = sha256_file(model_artifact)
        if actual != expected_model_sha256:
            raise RerankerUnavailable(
                f"reranker model checksum mismatch: expected {expected_model_sha256[:16]}..., got {actual[:16]}..."
            )

    try:
        with np.load(feature_artifact, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            artifact_ids = np.asarray(payload["anime_ids"], dtype=np.int64)
            space = RerankerFeatureSpace.from_prepared(catalog, artifact_ids, payload)
    except (OSError, ValueError, KeyError) as exc:
        raise RerankerUnavailable(f"reranker feature artifact is unreadable: {type(exc).__name__}") from exc

    if metadata.get("artifact_version") != RERANKER_ARTIFACT_VERSION:
        raise RerankerUnavailable(f"unsupported reranker artifact version: {metadata.get('artifact_version')!r}")
    if not np.array_equal(artifact_ids, anime_ids):
        raise RerankerUnavailable(
            f"reranker feature artifact does not match the ALS item set ({len(artifact_ids)} vs {len(anime_ids)} items)"
        )

    try:
        booster = lgb.Booster(model_file=str(model_artifact))
    except Exception as exc:  # lightgbm raises its own error type
        raise RerankerUnavailable(f"reranker model failed to load: {type(exc).__name__}") from exc
    if booster.num_feature() != EXPECTED_FEATURE_COUNT:
        raise RerankerUnavailable(
            f"reranker model expects {booster.num_feature()} features, code builds {EXPECTED_FEATURE_COUNT}"
        )

    metadata["artifact_sha256"] = sha256_file(feature_artifact)
    metadata["model_sha256"] = sha256_file(model_artifact)
    return LearnedReranker(booster, space, metadata=metadata)


def try_load_reranker(*args: Any, **kwargs: Any) -> LearnedReranker | None:
    """Load the reranker, or return None after logging why not.

    The fast path uses this rather than `load_reranker`: a reranker that cannot
    load must degrade to ALS-only, not take the request down with it.
    """
    try:
        return load_reranker(*args, **kwargs)
    except RerankerUnavailable as exc:
        logger.warning(
            "reranker_unavailable",
            extra={"context": {"reason": str(exc), "action": "serving_als_order"}},
        )
        return None
