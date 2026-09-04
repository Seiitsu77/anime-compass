"""Production serving for the frozen implicit-ALS collaborative model.

This module is the *serving* half of ALS. It loads a numerical artifact and
scores users; it never trains. Nothing here imports SciPy, the evaluation
package, or the training code, so the FastAPI process carries no offline
dependency.

The artifact stores item factors only. A user vector is reconstructed at request
time by folding the user's positives into item space:

    x_u = (YᵀY + alpha · Y_uᵀY_u + reg · I)⁻¹ (alpha · Σ_{i∈u} y_i)

That is the same ridge solve the trainer performs for a user row, so a user who
was in the training set gets back (numerically) the factor that was learned for
them, and a user who was not gets a consistent one. Verified on 2,000 sampled
training users: cosine mean 0.9995 against the trained factors, top-20 ranking
overlap 0.9728.

Folding in rather than storing user factors is what lets the same artifact serve
a live session whose history the model has never seen.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

ALS_ARTIFACT_VERSION = 1

# An artifact declares which job it is for. The evaluation artifact withholds
# each user's held-out positives and backs every published metric; the
# production artifact trains on everything and is invalid for measuring.
# Serving the wrong one is silent and consequential in both directions, so the
# role is checked rather than assumed.
ARTIFACT_ROLE_EVALUATION = "evaluation"
ARTIFACT_ROLE_PRODUCTION = "production"
# Minimum catalog overlap before an artifact is considered usable. A small
# amount of drift is expected as the catalog gains titles; a large amount means
# the artifact belongs to a different catalog and must not be served.
MINIMUM_CATALOG_OVERLAP = 0.90


class ALSArtifactError(RuntimeError):
    """Raised when an ALS artifact cannot be served faithfully."""


class ALSArtifactRoleError(ALSArtifactError):
    """Raised when an artifact is for a different job than the caller wants."""


class ALSCatalogMismatchError(ALSArtifactError):
    """Raised when an artifact does not describe the active catalog.

    Separate from the generic error because this is the condition that must
    produce a high-severity degradation event rather than a quiet fallback: it
    means the served catalog moved out from under the model.
    """


class ALSCollaborativeIndex:
    """Serve implicit-ALS scores behind the collaborative-channel interface.

    Exposes the same three members the recommender already calls on its
    collaborative index -- ``profile_scores``, ``quality_score``, and
    ``model_info`` -- so it is a drop-in substitution.

    Bayesian quality statistics are not part of an ALS artifact. They are taken
    from a supplied ``quality_source`` (the CountSketch index in production), so
    swapping the collaborative signal never silently changes the quality
    channel.

    Instances are read-only after construction and safe to share across request
    threads: scoring allocates its own arrays and the cached Gram matrix is
    built once under a lock.
    """

    def __init__(
        self,
        anime_ids: np.ndarray,
        item_factors: np.ndarray,
        metadata: Mapping[str, Any],
        *,
        quality_source: Any | None = None,
    ):
        self.anime_ids = np.asarray(anime_ids, dtype=np.int64)
        self.item_factors = np.asarray(item_factors, dtype=np.float32)
        self.metadata = dict(metadata)
        self.quality_source = quality_source
        self.index_by_id = {int(value): index for index, value in enumerate(self.anime_ids.tolist())}
        self.alpha = float(self.metadata.get("alpha", 5.0))
        self.regularization = float(self.metadata.get("regularization", 0.05))
        self.dimensions = int(self.item_factors.shape[1])
        self._lock = threading.Lock()
        self._gramian: np.ndarray | None = None
        self._identity = np.eye(self.dimensions, dtype=np.float32)

    @classmethod
    def load(
        cls,
        path: Path,
        catalog: Sequence[Mapping[str, Any]],
        *,
        quality_source: Any | None = None,
        expected_artifact_sha256: str | None = None,
        expected_role: str | None = None,
        expected_catalog_ids_sha256: str | None = None,
    ) -> ALSCollaborativeIndex:
        """Load and validate an artifact, or raise.

        Validation is deliberately strict: serving a mismatched artifact would
        silently produce recommendations for the wrong catalog, which is worse
        than refusing to start.
        """
        path = Path(path)
        if not path.exists():
            raise ALSArtifactError(f"ALS artifact not found: {path}")

        if expected_artifact_sha256:
            actual = sha256_file(path)
            if actual != expected_artifact_sha256:
                raise ALSArtifactError(
                    f"ALS artifact checksum mismatch for {path.name}: "
                    f"expected {expected_artifact_sha256[:16]}..., got {actual[:16]}..."
                )

        with np.load(path, allow_pickle=False) as artifact:
            required = {"anime_ids", "item_factors", "metadata_json"}
            missing = required.difference(artifact.files)
            if missing:
                raise ALSArtifactError("ALS artifact is missing arrays: " + ", ".join(sorted(missing)))
            anime_ids = np.asarray(artifact["anime_ids"], dtype=np.int64)
            item_factors = np.asarray(artifact["item_factors"], dtype=np.float32)
            metadata = json.loads(str(artifact["metadata_json"].item()))

        if metadata.get("artifact_version") != ALS_ARTIFACT_VERSION:
            raise ALSArtifactError(
                f"Unsupported ALS artifact version: {metadata.get('artifact_version')!r} "
                f"(expected {ALS_ARTIFACT_VERSION})"
            )

        # Artifacts written before roles existed are evaluation builds.
        role = str(metadata.get("artifact_role") or ARTIFACT_ROLE_EVALUATION)
        if expected_role is not None and role != expected_role:
            raise ALSArtifactRoleError(
                f"ALS artifact role mismatch for {path.name}: expected {expected_role!r}, got {role!r}. "
                "Evaluation artifacts withhold held-out positives and must not serve production; "
                "production artifacts train on everything and must not produce holdout metrics."
            )
        if item_factors.ndim != 2 or item_factors.shape[0] != len(anime_ids):
            raise ALSArtifactError("ALS item factors are not aligned with anime IDs")
        if not len(anime_ids):
            raise ALSArtifactError("ALS artifact contains no items")
        if len(set(anime_ids.tolist())) != len(anime_ids):
            raise ALSArtifactError("ALS anime IDs must be unique")
        if not np.isfinite(item_factors).all():
            raise ALSArtifactError("ALS artifact contains non-finite factors")

        catalog_ids = {int(item["id"]) for item in catalog}
        overlap = sum(int(value) in catalog_ids for value in anime_ids.tolist())
        ratio = overlap / len(anime_ids)
        if ratio < MINIMUM_CATALOG_OVERLAP:
            raise ALSCatalogMismatchError(
                f"ALS artifact does not match the active catalog: {ratio:.1%} overlap "
                f"(minimum {MINIMUM_CATALOG_OVERLAP:.0%}); "
                f"artifact has {len(anime_ids)} items, catalog has {len(catalog_ids)}"
            )

        # A pinned catalog digest is exact where the overlap ratio is fuzzy: it
        # catches a catalog that drifted while still overlapping heavily. The
        # production artifact carries the digest of the catalog it was trained
        # against, so enforce it even when an operator did not duplicate the
        # value in an environment variable. An external pin, when supplied,
        # must also agree with the artifact metadata.
        artifact_catalog_ids_sha256 = str(metadata.get("catalog_ids_sha256") or "")
        if (
            expected_catalog_ids_sha256
            and artifact_catalog_ids_sha256
            and artifact_catalog_ids_sha256 != expected_catalog_ids_sha256
        ):
            raise ALSCatalogMismatchError(
                f"ALS artifact catalog pin mismatch for {path.name}: "
                f"deployment expects {expected_catalog_ids_sha256[:16]}..., "
                f"artifact declares {artifact_catalog_ids_sha256[:16]}..."
            )
        pinned_catalog_ids_sha256 = expected_catalog_ids_sha256 or artifact_catalog_ids_sha256
        if pinned_catalog_ids_sha256:
            actual_catalog = catalog_ids_digest(sorted(catalog_ids))
            if actual_catalog != pinned_catalog_ids_sha256:
                raise ALSCatalogMismatchError(
                    f"Catalog identity mismatch for {path.name}: "
                    f"expected {pinned_catalog_ids_sha256[:16]}..., got {actual_catalog[:16]}..."
                )

        metadata["artifact_sha256"] = sha256_file(path)
        metadata["catalog_overlap"] = round(ratio, 6)
        metadata["artifact_role"] = role
        return cls(anime_ids, item_factors, metadata, quality_source=quality_source)

    def _gram(self) -> np.ndarray:
        gramian = self._gramian
        if gramian is None:
            with self._lock:
                if self._gramian is None:
                    self._gramian = self.item_factors.T @ self.item_factors
                gramian = self._gramian
        return gramian

    def user_vector(self, positive_ids: Sequence[int]) -> np.ndarray | None:
        """Fold a user's positives into item space, or None if none are known."""
        rows = [self.index_by_id[int(value)] for value in positive_ids if int(value) in self.index_by_id]
        if not rows:
            return None
        liked = self.item_factors[np.asarray(rows, dtype=np.int64)]
        matrix = self._gram() + self.alpha * (liked.T @ liked) + self.regularization * self._identity
        target = self.alpha * liked.sum(axis=0)
        try:
            return np.linalg.solve(matrix, target).astype(np.float32)
        except np.linalg.LinAlgError:
            return None

    def profile_scores(
        self,
        positive_ids: Sequence[int] = (),
        negative_ids: Sequence[int] = (),
        explicit_ratings: Mapping[int, float] | None = None,
    ) -> dict[int, float]:
        """Score every catalog item for a user profile, normalized to [0, 1].

        `negative_ids` are deliberately not folded in. Implicit ALS has no
        principled place for negative confidence -- disliked titles are handled
        by exclusion at ranking time, not by pushing the user vector away.
        """
        liked = [int(value) for value in positive_ids]
        for anime_id, rating in (explicit_ratings or {}).items():
            if float(rating) >= 8.0:
                liked.append(int(anime_id))

        vector = self.user_vector(liked)
        if vector is None:
            return {}
        scores = self.item_factors @ vector
        np.maximum(scores, 0.0, out=scores)
        peak = float(scores.max()) if scores.size else 0.0
        if peak <= 0.0:
            return {}
        scores /= peak
        return {
            int(anime_id): float(score)
            for anime_id, score in zip(self.anime_ids.tolist(), scores.tolist(), strict=True)
            if score > 0.0
        }

    def raw_profile_scores(self, positive_ids: Sequence[int]) -> npt.NDArray[np.float32] | None:
        """Unnormalised item scores, in artifact row order.

        `profile_scores` clamps at zero and divides by the peak, which is fine
        for blending but wrong as a model input: the reranker was fitted on raw
        scores, and its trees split on absolute thresholds. Rescaling would
        leave the standardised feature intact and quietly move the raw one.
        """
        vector = self.user_vector(positive_ids)
        if vector is None:
            return None
        return np.asarray(self.item_factors @ vector, dtype=np.float32)

    def top_candidates(self, positive_ids: Sequence[int], limit: int, *, excluded_ids: Sequence[int] = ()) -> list[int]:
        """Return the highest-scoring `limit` items, excluding known ones.

        This is the retrieval entry point: it avoids materialising a full score
        dictionary, using a partial sort over the factor scores instead.
        """
        if limit <= 0:
            return []
        vector = self.user_vector(positive_ids)
        if vector is None:
            return []
        scores = self.item_factors @ vector
        blocked = {int(value) for value in excluded_ids}
        blocked.update(int(value) for value in positive_ids)
        if blocked:
            rows = [self.index_by_id[value] for value in blocked if value in self.index_by_id]
            if rows:
                scores[np.asarray(rows, dtype=np.int64)] = -np.inf
        take = min(limit, scores.shape[0])
        top = np.argpartition(-scores, kth=take - 1)[:take] if take < scores.shape[0] else np.arange(scores.shape[0])
        ordered = top[np.argsort(-scores[top], kind="stable")]
        return [int(self.anime_ids[index]) for index in ordered if np.isfinite(scores[index])]

    def quality_score(self, anime_id: int) -> float | None:
        if self.quality_source is None:
            return None
        return self.quality_source.quality_score(anime_id)

    def model_info(self) -> dict[str, Any]:
        return {
            "available": True,
            "method": "implicit-feedback ALS (frozen artifact, request-time fold-in)",
            "items": int(len(self.anime_ids)),
            "dimensions": self.dimensions,
            "factors": self.metadata.get("factors"),
            "alpha": self.alpha,
            "regularization": self.regularization,
            "iterations": self.metadata.get("iterations"),
            "artifact_version": ALS_ARTIFACT_VERSION,
            "artifact_role": self.metadata.get("artifact_role"),
            "artifact_sha256": self.metadata.get("artifact_sha256"),
            "split_sha256": self.metadata.get("split_sha256"),
            "ratings_sha256": self.metadata.get("ratings_sha256"),
            "catalog_ids_sha256": self.metadata.get("catalog_ids_sha256"),
            "training_source": self.metadata.get("training_source"),
            "ratings_used": self.metadata.get("ratings_used"),
            "valid_for_holdout_evaluation": not bool(self.metadata.get("not_valid_for_holdout_evaluation")),
        }

    @property
    def resident_array_bytes(self) -> int:
        return int(self.anime_ids.nbytes + self.item_factors.nbytes)


def catalog_ids_digest(anime_ids: Sequence[int]) -> str:
    """Hash a catalog ID set, matching the trainer's digest exactly."""
    import hashlib

    digest = hashlib.sha256()
    for value in anime_ids:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file's SHA-256.

    Duplicated from the evaluation package on purpose: serving must not import
    evaluation code.
    """
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
