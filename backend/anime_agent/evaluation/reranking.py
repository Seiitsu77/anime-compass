"""Second-stage reranking over a frozen ALS candidate set.

ALS already retrieves well: Recall@300 is 0.7932, so roughly four fifths of what
a user will like is somewhere in the top 300. That splits the problem in two,
and this module owns only the second half.

    ALS (frozen)  ->  candidates  ->  reranker  ->  order

Retrieval is not touched. The question is narrow and answerable: given that the
right items are usually already in the candidate set, does richer per-item and
per-user-item evidence order them better than the ALS score alone?

Leakage control is the whole game here, so every feature is built from one of
two sources and nothing else:

* **The user's train positives.** Never validation, never test. A profile
  feature that saw a held-out positive would make the reranker look brilliant
  and be worthless in production.
* **Train-only item statistics.** Popularity, rating count, rating mean, and
  Bayesian quality all come from the split's train-only artifacts, not from the
  catalog's all-time aggregates, which are computed over every user including
  held-out ones.

Descriptive catalog metadata -- genres, studios, type, source, year -- is used
directly. It describes the item, not the audience, so it carries no signal about
who liked what.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# Exact feature definitions, in the order the matrix stores them. The docstring
# on each is the definition of record; `docs/` quotes this list rather than
# restating it, so the two cannot disagree.
FEATURE_NAMES: tuple[str, ...] = (
    "als_score",  # raw ALS score for this (user, item), from the frozen artifact
    "als_score_z",  # that score standardised within this user's candidate list
    "als_rank_recip",  # 1 / (1 + zero-based rank) within the candidate list
    "genre_affinity",  # mean profile frequency of the candidate's genres
    "genre_coverage",  # fraction of the candidate's genres seen in the profile
    "genre_jaccard_max",  # best genre Jaccard against any single profile title
    "studio_affinity",  # fraction of profile titles sharing a studio
    "source_affinity",  # profile frequency of the candidate's source material
    "type_affinity",  # profile frequency of the candidate's format
    "year_gap",  # |candidate year - profile median year| / 10
    "year_close",  # 1 when that gap is within five years
    "item_item_max",  # best train-only item-item similarity to a profile title
    "item_item_sum5",  # sum of the five best such similarities
    "log_train_pop",  # log1p(train-only positive interactions for the item)
    "log_train_ratings",  # log1p(train-only rating count for the item)
    "train_rating_mean",  # train-only mean rating, centred on the global mean
    "train_bayes",  # train-only Bayesian quality score, centred
    "profile_size",  # log1p(number of train positives the user has)
)

N_FEATURES = len(FEATURE_NAMES)


def _tokens(values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        return (values.casefold(),)
    out: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("name") or value.get("title") or ""
        text = str(value).strip().casefold()
        if text:
            out.append(text)
    return tuple(out)


@dataclass(frozen=True)
class ItemAttributes:
    """Per-item descriptive metadata, indexed by ALS row."""

    genres: tuple[frozenset[str], ...]
    studios: tuple[frozenset[str], ...]
    source: tuple[str, ...]
    media_type: tuple[str, ...]
    year: npt.NDArray[np.float32]


class RerankerFeatureSpace:
    """Builds candidate features for one user at a time.

    Construction is the expensive part and happens once per evaluation run; the
    per-user call is a handful of vectorised lookups.
    """

    def __init__(
        self,
        anime_ids: npt.NDArray[np.int64],
        attributes: ItemAttributes,
        *,
        train_positive_count: npt.NDArray[np.float32],
        train_rating_count: npt.NDArray[np.float32],
        train_rating_mean: npt.NDArray[np.float32],
        train_bayes: npt.NDArray[np.float32],
        global_rating_mean: float,
        neighbor_indices: npt.NDArray[np.int32] | None = None,
        neighbor_scores: npt.NDArray[np.float32] | None = None,
    ):
        self.anime_ids = anime_ids
        self.index_by_id = {int(value): index for index, value in enumerate(anime_ids.tolist())}
        self.attributes = attributes
        self.log_train_pop = np.log1p(train_positive_count).astype(np.float32)
        self.log_train_ratings = np.log1p(train_rating_count).astype(np.float32)
        self.train_rating_mean = (train_rating_mean - global_rating_mean).astype(np.float32)
        self.train_bayes = (train_bayes - global_rating_mean).astype(np.float32)
        # Item-item similarity as a lookup from (item row) to {neighbour row: score}.
        self._neighbors: list[dict[int, float]] = []
        if neighbor_indices is not None and neighbor_scores is not None:
            for row in range(len(anime_ids)):
                self._neighbors.append(
                    {
                        int(neighbor): float(score)
                        for neighbor, score in zip(neighbor_indices[row], neighbor_scores[row], strict=False)
                        if score > 0
                    }
                )

    # ------------------------------------------------------------- loading

    @classmethod
    def from_artifacts(
        cls,
        catalog: Sequence[Mapping[str, Any]],
        anime_ids: npt.NDArray[np.int64],
        *,
        popularity_path: Path | None = None,
        quality_path: Path | None = None,
        item_item_path: Path | None = None,
    ) -> RerankerFeatureSpace:
        by_id = {int(item["id"]): item for item in catalog}
        rows = len(anime_ids)
        genres: list[frozenset[str]] = []
        studios: list[frozenset[str]] = []
        source: list[str] = []
        media_type: list[str] = []
        year = np.zeros(rows, dtype=np.float32)
        for index, anime_id in enumerate(anime_ids.tolist()):
            item = by_id.get(int(anime_id), {})
            genres.append(frozenset(_tokens(item.get("genres"))))
            studios.append(frozenset(_tokens(item.get("studios"))))
            source.append(str(item.get("source") or "").casefold())
            media_type.append(str(item.get("type") or "").casefold())
            year[index] = float(item.get("start_year") or 0.0)
        attributes = ItemAttributes(tuple(genres), tuple(studios), tuple(source), tuple(media_type), year)

        pop: npt.NDArray[np.float32] = np.zeros(rows, dtype=np.float32)
        if popularity_path and popularity_path.exists():
            with np.load(popularity_path, allow_pickle=False) as payload:
                pop = cls._align(payload["anime_ids"], payload["positive_count"], anime_ids)

        rating_count: npt.NDArray[np.float32] = np.zeros(rows, dtype=np.float32)
        rating_mean: npt.NDArray[np.float32] = np.zeros(rows, dtype=np.float32)
        bayes: npt.NDArray[np.float32] = np.zeros(rows, dtype=np.float32)
        global_mean = 0.0
        if quality_path and quality_path.exists():
            with np.load(quality_path, allow_pickle=False) as payload:
                rating_count = cls._align(payload["anime_ids"], payload["rating_count"], anime_ids)
                rating_mean = cls._align(payload["anime_ids"], payload["rating_mean"], anime_ids)
                bayes = cls._align(payload["anime_ids"], payload["bayesian_score"], anime_ids)
                meta = json.loads(str(payload["metadata_json"].item()))
                global_mean = float(meta.get("global_rating_mean", 0.0))

        neighbor_indices = neighbor_scores = None
        if item_item_path and item_item_path.exists():
            with np.load(item_item_path, allow_pickle=False) as payload:
                source_ids = payload["anime_ids"]
                if len(source_ids) == rows and bool(np.array_equal(source_ids, anime_ids)):
                    neighbor_indices = payload["neighbor_indices"]
                    neighbor_scores = payload["neighbor_scores"]

        return cls(
            anime_ids,
            attributes,
            train_positive_count=pop,
            train_rating_count=rating_count,
            train_rating_mean=rating_mean,
            train_bayes=bayes,
            global_rating_mean=global_mean,
            neighbor_indices=neighbor_indices,
            neighbor_scores=neighbor_scores,
        )

    @staticmethod
    def _align(
        source_ids: npt.NDArray[Any],
        values: npt.NDArray[Any],
        target_ids: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        lookup = {int(key): float(value) for key, value in zip(source_ids.tolist(), values.tolist(), strict=False)}
        return np.asarray([lookup.get(int(key), 0.0) for key in target_ids.tolist()], dtype=np.float32)

    # ------------------------------------------------------------ features

    def build(
        self,
        profile_rows: Sequence[int],
        candidate_rows: Sequence[int],
        als_scores: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Feature matrix for one user's candidate list.

        `profile_rows` are ALS rows for the user's **train** positives. Passing
        anything else here is the one mistake that would invalidate the whole
        experiment, so callers derive it from `UserSplit.train_positive_ids`.
        """
        n = len(candidate_rows)
        features = np.zeros((n, N_FEATURES), dtype=np.float32)
        if n == 0:
            return features

        attributes = self.attributes
        profile = list(profile_rows)
        profile_size = float(len(profile))

        genre_counts: dict[str, int] = {}
        studio_set: set[str] = set()
        source_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        profile_genres: list[frozenset[str]] = []
        years: list[float] = []
        for row in profile:
            item_genres = attributes.genres[row]
            profile_genres.append(item_genres)
            for genre in item_genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
            studio_set |= attributes.studios[row]
            source_counts[attributes.source[row]] = source_counts.get(attributes.source[row], 0) + 1
            type_counts[attributes.media_type[row]] = type_counts.get(attributes.media_type[row], 0) + 1
            if attributes.year[row] > 0:
                years.append(float(attributes.year[row]))
        median_year = float(np.median(years)) if years else 0.0
        divisor = max(profile_size, 1.0)

        # Item-item similarity to the profile, from the train-only artifact.
        profile_set = set(profile)
        best_sim = np.zeros(n, dtype=np.float32)
        top5_sim = np.zeros(n, dtype=np.float32)
        if self._neighbors:
            for position, row in enumerate(candidate_rows):
                shared = [score for neighbor, score in self._neighbors[row].items() if neighbor in profile_set]
                if shared:
                    shared.sort(reverse=True)
                    best_sim[position] = shared[0]
                    top5_sim[position] = float(sum(shared[:5]))

        mean_score = float(als_scores.mean()) if n else 0.0
        std_score = float(als_scores.std()) or 1.0
        log_profile = math.log1p(profile_size)

        for position, row in enumerate(candidate_rows):
            item_genres = attributes.genres[row]
            if item_genres:
                affinity = sum(genre_counts.get(genre, 0) for genre in item_genres) / (divisor * len(item_genres))
                coverage = sum(1 for genre in item_genres if genre in genre_counts) / len(item_genres)
                jaccard = 0.0
                for other in profile_genres:
                    union = item_genres | other
                    if union:
                        jaccard = max(jaccard, len(item_genres & other) / len(union))
            else:
                affinity = coverage = jaccard = 0.0

            candidate_year = float(attributes.year[row])
            gap = abs(candidate_year - median_year) / 10.0 if candidate_year > 0 and median_year > 0 else 0.0

            features[position] = (
                als_scores[position],
                (als_scores[position] - mean_score) / std_score,
                1.0 / (1.0 + position),
                affinity,
                coverage,
                jaccard,
                (len(attributes.studios[row] & studio_set) > 0) * 1.0,
                source_counts.get(attributes.source[row], 0) / divisor,
                type_counts.get(attributes.media_type[row], 0) / divisor,
                gap,
                1.0 if 0.0 < gap <= 0.5 else 0.0,
                best_sim[position],
                top5_sim[position],
                self.log_train_pop[row],
                self.log_train_ratings[row],
                self.train_rating_mean[row],
                self.train_bayes[row],
                log_profile,
            )
        return features


@dataclass
class StandardScaler:
    """Feature means and scales, fitted on training rows only."""

    mean: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]

    @classmethod
    def fit(cls, features: npt.NDArray[np.float32]) -> StandardScaler:
        mean = features.mean(axis=0).astype(np.float32)
        scale = features.std(axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return cls(mean, scale)

    def transform(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return ((features - self.mean) / self.scale).astype(np.float32)


class LinearReranker:
    """L2-regularised logistic regression over candidate features.

    Deliberately the simplest thing that could show signal. If a linear model
    over these features cannot beat the ALS score it is given as feature zero,
    the features do not carry incremental information and a stronger ranker
    would be fitting noise rather than finding structure.

    Serving is a scaler, a weight vector, and a dot product, so promoting it
    would add no runtime dependency to the NumPy-only path.
    """

    def __init__(self, weights: npt.NDArray[np.float32], bias: float, scaler: StandardScaler):
        self.weights = weights
        self.bias = bias
        self.scaler = scaler

    @classmethod
    def fit(
        cls,
        features: npt.NDArray[np.float32],
        labels: npt.NDArray[np.float32],
        *,
        l2: float = 1.0,
        epochs: int = 60,
        learning_rate: float = 0.5,
        seed: int = 20260902,
    ) -> LinearReranker:
        scaler = StandardScaler.fit(features)
        matrix = scaler.transform(features)
        rows, columns = matrix.shape
        weights = np.zeros(columns, dtype=np.float64)
        bias = 0.0
        # Positives are ~1% of rows, so weight them up rather than let the model
        # learn the majority class and stop.
        positive_rate = float(labels.mean()) or 1e-6
        weight_positive = (1.0 - positive_rate) / positive_rate
        sample_weight = np.where(labels > 0, weight_positive, 1.0).astype(np.float64)
        sample_weight /= sample_weight.mean()

        generator = np.random.default_rng(seed)
        order = np.arange(rows)
        batch = 8192
        for epoch in range(epochs):
            generator.shuffle(order)
            step = learning_rate / (1.0 + 0.05 * epoch)
            for start in range(0, rows, batch):
                index = order[start : start + batch]
                chunk = matrix[index].astype(np.float64)
                logits = chunk @ weights + bias
                predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                error = (predicted - labels[index]) * sample_weight[index]
                gradient = chunk.T @ error / len(index) + l2 * weights / rows
                weights -= step * gradient
                bias -= step * float(error.mean())
        return cls(weights.astype(np.float32), float(bias), scaler)

    def score(self, features: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return (self.scaler.transform(features) @ self.weights + self.bias).astype(np.float32)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "weights": [float(value) for value in self.weights],
            "bias": self.bias,
            "scaler_mean": [float(value) for value in self.scaler.mean],
            "scaler_scale": [float(value) for value in self.scaler.scale],
        }
