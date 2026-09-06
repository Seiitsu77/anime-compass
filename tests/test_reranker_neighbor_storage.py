"""Item-item neighbours stay in the form the artifact stores them.

`RerankerFeatureSpace` used to expand the neighbour arrays into 18,064 Python
dicts holding 3.1M entries. A memory profiling audit measured that at 313 MiB
against the 28 MiB the same numbers occupy as npz arrays -- an 11x expansion of
information the process already had, and the second largest consumer in a
1.98 GB process.

The dicts were replaced by a packed layout: two flat arrays plus a row-offset
index. That is a representation change and nothing else, so what these tests
guard is that the values reaching the feature matrix are unchanged, including
the one precondition the equivalence depends on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.anime_agent.evaluation.reranking import (
    FEATURE_NAMES,
    ItemAttributes,
    RerankerFeatureSpace,
)


def make_neighbors(rows: int, neighbors: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A top-K neighbour block, shaped the way the real artifact is shaped.

    Each row names distinct items, because it is the top K of a similarity row
    over distinct items. That matters: the packed layout keeps every entry,
    whereas the dict it replaced would have collapsed a repeated neighbour to
    its last score. The two agree exactly under this guarantee, which
    `test_the_production_artifact_has_distinct_neighbours_per_row` pins against
    the artifact actually shipped.
    """
    rng = np.random.default_rng(seed)
    indices = np.stack([rng.permutation(rows)[:neighbors] for _ in range(rows)]).astype(np.int32)
    scores = rng.random((rows, neighbors)).astype(np.float32)
    # A realistic artifact is sparse: most neighbour slots score zero.
    scores[rng.random((rows, neighbors)) < 0.4] = 0.0
    return indices, scores


def make_space(rows: int = 24, neighbors: int = 8, seed: int = 11) -> RerankerFeatureSpace:
    rng = np.random.default_rng(seed)
    anime_ids = np.arange(1, rows + 1, dtype=np.int64)
    attributes = ItemAttributes(
        genres=tuple(frozenset({"action"}) for _ in range(rows)),
        studios=tuple(frozenset({"madhouse"}) for _ in range(rows)),
        source=tuple("manga" for _ in range(rows)),
        media_type=tuple("tv" for _ in range(rows)),
        year=np.full(rows, 2010, dtype=np.float32),
    )
    indices, scores = make_neighbors(rows, neighbors, seed)
    return RerankerFeatureSpace(
        anime_ids,
        attributes,
        train_positive_count=rng.random(rows).astype(np.float32),
        train_rating_count=rng.random(rows).astype(np.float32),
        train_rating_mean=rng.random(rows).astype(np.float32),
        train_bayes=rng.random(rows).astype(np.float32),
        global_rating_mean=0.5,
        neighbor_indices=indices,
        neighbor_scores=scores,
    )


def test_neighbours_are_stored_as_arrays_not_dicts():
    space = make_space()
    assert not hasattr(space, "_neighbors"), "the per-item dict expansion must not come back"
    for name in ("_neighbor_offsets", "_neighbor_ids", "_neighbor_scores"):
        assert isinstance(getattr(space, name), np.ndarray)


def test_the_packed_rows_hold_exactly_the_positive_scored_neighbours():
    """The dict dropped zero scores; the packed layout must drop the same ones."""
    rows, neighbors = 24, 8
    indices, scores = make_neighbors(rows, neighbors, 11)

    space = make_space()
    offsets = space._neighbor_offsets
    assert offsets is not None and space._neighbor_ids is not None and space._neighbor_scores is not None
    assert int(offsets[-1]) == int((scores > 0).sum())

    for row in range(rows):
        expected = {int(n): float(s) for n, s in zip(indices[row], scores[row], strict=False) if s > 0}
        start, stop = int(offsets[row]), int(offsets[row + 1])
        packed = dict(
            zip(
                space._neighbor_ids[start:stop].tolist(),
                space._neighbor_scores[start:stop].tolist(),
                strict=False,
            )
        )
        assert packed == expected


def test_similarity_features_match_the_dict_implementation():
    """The exact computation the dict version performed, recomputed here."""
    space = make_space()
    rng = np.random.default_rng(3)
    rows = len(space.anime_ids)
    offsets = space._neighbor_offsets
    assert offsets is not None and space._neighbor_ids is not None and space._neighbor_scores is not None

    reference = [
        {
            int(n): float(s)
            for n, s in zip(
                space._neighbor_ids[int(offsets[row]) : int(offsets[row + 1])].tolist(),
                space._neighbor_scores[int(offsets[row]) : int(offsets[row + 1])].tolist(),
                strict=False,
            )
        }
        for row in range(rows)
    ]

    max_column = FEATURE_NAMES.index("item_item_max")
    sum5_column = FEATURE_NAMES.index("item_item_sum5")

    for _ in range(25):
        profile = rng.choice(rows, size=6, replace=False).tolist()
        candidates = rng.choice(rows, size=10, replace=False).tolist()
        features = space.build(profile, candidates, rng.standard_normal(10).astype(np.float32))

        profile_set = set(profile)
        for position, row in enumerate(candidates):
            shared = [score for neighbor, score in reference[row].items() if neighbor in profile_set]
            shared.sort(reverse=True)
            expected_max = shared[0] if shared else 0.0
            expected_sum5 = float(sum(shared[:5])) if shared else 0.0
            assert features[position, max_column] == np.float32(expected_max)
            assert features[position, sum5_column] == np.float32(expected_sum5)


def test_the_production_artifact_has_distinct_neighbours_per_row():
    """The precondition the packed layout relies on, checked against reality.

    If a future artifact ever repeated a neighbour within a row, the packed scan
    would count it twice where the old dict counted it once, and the similarity
    features would shift. Nothing else in the repository states this
    requirement, so it is stated here.
    """
    path = Path("data/processed/reranker_features.npz")
    if not path.exists():
        pytest.skip("reranker artifact is not present in this checkout")
    with np.load(path, allow_pickle=False) as payload:
        indices = payload["neighbor_indices"]
        scores = payload["neighbor_scores"]
    offenders = [
        row
        for row in range(indices.shape[0])
        if np.unique(indices[row][scores[row] > 0]).size != int((scores[row] > 0).sum())
    ]
    assert not offenders, f"rows repeat a neighbour id: {offenders[:5]}"


def test_an_absent_item_item_artifact_still_builds_features():
    """The artifact is optional; the packed arrays stay None without it."""
    space = make_space()
    bare = RerankerFeatureSpace(
        space.anime_ids,
        space.attributes,
        train_positive_count=np.zeros(len(space.anime_ids), dtype=np.float32),
        train_rating_count=np.zeros(len(space.anime_ids), dtype=np.float32),
        train_rating_mean=np.zeros(len(space.anime_ids), dtype=np.float32),
        train_bayes=np.zeros(len(space.anime_ids), dtype=np.float32),
        global_rating_mean=0.0,
    )
    assert bare._neighbor_offsets is None
    features = bare.build([0, 1], [2, 3], np.zeros(2, dtype=np.float32))
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
