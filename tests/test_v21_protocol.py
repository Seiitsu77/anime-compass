"""The evaluation protocol in experiments/v21_quality must not be able to leak.

A number from those scripts is only worth reading if a hidden positive can
never reach a profile, a feature, or an exclusion list, and if "production" in
the series-policy comparison is the rule the service really applies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.repositories.session_repository import SQLiteSessionRepository
from app.services.recommendation_service import RecommendationService
from backend.anime_agent.recommender import AnimeRecommender, series_key
from experiments.v21_quality.protocol import (
    SERIES_POLICIES,
    ProfileCase,
    apply_series_policy,
    assert_isolated,
    build_case,
    holdout_matched_subset,
    sample_profile,
)
from tests.conftest import anime

# ------------------------------------------------------------- profiles


def test_a_profile_is_deterministic_nested_and_independent_of_input_order():
    train = list(range(100, 160))
    three = sample_profile(7, train, 3)
    five = sample_profile(7, train, 5)
    twenty = sample_profile(7, list(reversed(train)), 20)

    assert len(three) == 3 and len(five) == 5 and len(twenty) == 20
    assert five[:3] == three and twenty[:5] == five, "a larger profile only ever adds Likes"
    assert sample_profile(7, train, 5) == five
    assert sample_profile(8, train, 5) != five, "different users draw different Likes"
    assert sorted(sample_profile(7, train, None)) == train
    assert sample_profile(7, train, 1000) == sample_profile(7, train, None)
    assert sample_profile(7, [], 5) == []


def test_a_case_keeps_hidden_positives_out_of_every_model_input():
    case = build_case(
        11,
        train_rows=range(0, 40),
        hidden_rows=[40, 41, 42],
        non_positive_rows=[50, 51],
        size=5,
    )
    assert len(case.profile) == 5
    assert set(case.profile) <= set(range(0, 40))
    assert case.hidden == {40, 41, 42}
    # What a model may be told to remove is the observed profile, and only that.
    assert case.excluded == frozenset(case.profile)
    assert not case.excluded & case.hidden
    # Likes the model trained on but the profile does not show are neither.
    assert case.unobserved_likes == frozenset(range(0, 40)) - set(case.profile)
    assert not case.unobserved_likes & case.hidden
    assert not case.unobserved_likes & case.excluded


@pytest.mark.parametrize(
    ("case", "exclusions", "message"),
    [
        (ProfileCase(1, (1, 2), frozenset({2, 9}), frozenset(), frozenset()), None, "in the profile"),
        (ProfileCase(1, (1,), frozenset({9}), frozenset(), frozenset()), [1, 9], "excluded from candidates"),
        (ProfileCase(1, (1,), frozenset({9}), frozenset(), frozenset({9})), None, "non-positive"),
        (ProfileCase(1, (1,), frozenset({9}), frozenset({9}), frozenset()), None, "training positive"),
    ],
)
def test_a_leak_in_any_role_is_refused(case, exclusions, message):
    with pytest.raises(AssertionError, match=message):
        assert_isolated(case, exclusions=exclusions)


def test_building_a_case_from_overlapping_sets_is_refused():
    with pytest.raises(AssertionError):
        build_case(3, train_rows=[1, 2, 3], hidden_rows=[3, 4], non_positive_rows=[], size=None)


# ------------------------------------------------------- series policies

# rows:      0     1     2     3     4     5     6     7
KEYS = ["a", "a", "b", "a", "c", "b", "d", "c"]
TYPES = ["TV", "Special", "TV", "Movie", "OVA", "Movie", "TV", "TV"]
RANKED = [1, 3, 2, 5, 4, 7, 6]  # row 0 is the Liked title, in franchise "a"
PROFILE = [0]


def test_each_series_policy_does_what_its_description_says():
    def run(policy, **kwargs):
        return apply_series_policy(RANKED, PROFILE, KEYS, policy, types=TYPES, **kwargs)

    assert run("none") == RANKED
    # Franchise "a" is Liked: production drops both of its remaining entries.
    assert run("production") == [2, 4, 6]
    # One per franchise, and the Liked franchise may contribute its best entry.
    assert run("dedupe") == [1, 2, 4, 6]
    # ...but only a series or a film: the special (row 1) is skipped for the film (row 3).
    assert run("dedupe_mainline") == [3, 2, 4, 6]
    assert run("cap2") == [1, 3, 2, 5, 4, 7, 6]
    # Batches of three: a franchise's second entry waits for the next batch.
    assert run("batch_dedupe", batch=3) == [1, 2, 4, 3, 5, 7, 6]

    for policy in SERIES_POLICIES:
        result = run(policy)
        assert len(result) == len(set(result)), f"{policy} repeated a title"
        assert set(result) <= set(RANKED)
        assert [RANKED.index(row) for row in result] == sorted(
            RANKED.index(row) for row in result
        ) or policy == "batch_dedupe"


def test_batch_dedupe_never_drops_a_title_and_keeps_each_batch_distinct():
    keys = ["x", "x", "x", "y", "y", "y", "z", "z", "z"]
    ranked = list(range(9))
    result = apply_series_policy(ranked, [], keys, "batch_dedupe", batch=3)
    assert result == [0, 3, 6, 1, 4, 7, 2, 5, 8]
    assert sorted(result) == ranked, "deferred, never dropped"
    for start in range(0, len(result), 3):
        batch = result[start : start + 3]
        assert len({keys[row] for row in batch}) == len(batch)


def test_batch_dedupe_tops_a_batch_up_when_distinct_franchises_run_out():
    keys = ["x"] * 5 + ["y"] * 3 + ["z"]
    ranked = list(range(9))
    result = apply_series_policy(ranked, [], keys, "batch_dedupe", batch=3)
    assert sorted(result) == ranked
    assert result[:3] == [0, 5, 8], "the first batch is one per franchise"
    # Only two franchises are left, so the second batch is filled in rank order.
    assert result[3:6] == [1, 6, 2]


def test_a_liked_franchise_can_be_rationed_per_batch():
    # Franchises a, b and c are Liked (rows 0-2); the ranked list leads with their sequels.
    keys = ["a", "b", "c", "a", "b", "c", "d", "e", "f", "g", "a"]
    profile = [0, 1, 2]
    ranked = [3, 4, 5, 6, 7, 8, 9, 10]

    one = apply_series_policy(ranked, profile, keys, "dedupe_liked1", batch=3)
    # Batch one takes a single sequel (row 3), then new franchises; the other
    # sequels wait, one per batch, and are never dropped.
    assert one == [3, 6, 7, 4, 8, 9, 5]
    for start in range(0, len(one), 3):
        assert sum(keys[row] in {"a", "b", "c"} for row in one[start : start + 3]) <= 1
    assert len({keys[row] for row in one}) == len(one), "still one entry per franchise"
    assert 10 not in one, "a second entry of franchise a is a duplicate, not a deferral"

    three = apply_series_policy(ranked, profile, keys, "dedupe_liked3", batch=3)
    assert three == [3, 4, 5, 6, 7, 8, 9]
    # With no Liked franchise in play the ration changes nothing.
    assert apply_series_policy(ranked, [], keys, "dedupe_liked1", batch=3) == apply_series_policy(
        ranked, [], keys, "dedupe", batch=3
    )


def test_an_unknown_policy_and_a_missing_format_are_errors():
    with pytest.raises(ValueError, match="unknown series policy"):
        apply_series_policy(RANKED, PROFILE, KEYS, "shuffle")
    with pytest.raises(ValueError, match="format"):
        apply_series_policy(RANKED, PROFILE, KEYS, "dedupe_mainline")


def test_the_production_policy_is_the_rule_the_service_applies(tmp_path: Path):
    """`production` here and `_collapse_series` there must not drift apart."""
    catalog = []
    for franchise in range(1, 9):
        for entry, suffix in enumerate(("", " Season 2", " Movie")):
            catalog.append(
                anime(
                    franchise * 10 + entry,
                    f"Saga{franchise} Chronicle{suffix}",
                    genres=["Action"],
                    synopsis="A synthetic franchise entry.",
                )
            )
    recommender = AnimeRecommender(catalog)
    sessions = SQLiteSessionRepository(f"sqlite:///{(tmp_path / 's.db').as_posix()}", retention_days=30)
    service = RecommendationService(recommender, sessions)

    ids = [int(item["id"]) for item in catalog]
    keys = [series_key(str(item["title"])) for item in catalog]
    row_of = {anime_id: row for row, anime_id in enumerate(ids)}
    liked = [10, 40]
    ranked_ids = [42, 11, 21, 20, 41, 31, 12, 50, 52, 22, 70, 71]

    served = service._collapse_series(ranked_ids, liked)
    evaluated = apply_series_policy([row_of[a] for a in ranked_ids], [row_of[a] for a in liked], keys, "production")
    assert [ids[row] for row in evaluated] == served == [21, 31, 50, 70]
    sessions.engine.dispose()


# --------------------------------------------------- watched-history rate


def test_non_positives_can_be_hidden_at_the_same_rate_as_positives():
    rows = list(range(1000, 1100))
    kept = holdout_matched_subset(5, rows, 0.8)
    assert len(kept) == 80
    assert kept <= set(rows)
    assert holdout_matched_subset(5, list(reversed(rows)), 0.8) == kept
    assert holdout_matched_subset(6, rows, 0.8) != kept
    # A larger known share only ever adds to what was known.
    assert holdout_matched_subset(5, rows, 0.5) <= kept
    assert holdout_matched_subset(5, rows, 1.0) == set(rows)
    assert holdout_matched_subset(5, rows, 0.0) == set()
    with pytest.raises(ValueError):
        holdout_matched_subset(5, rows, 1.2)


# ------------------------------------------------- the reserved users


def test_the_reserved_confirmation_set_is_untouched_and_disjoint():
    """Nothing may score the reserved users by accident.

    They are the untouched final test for the first model selection that needs
    one. Opening them is a deliberate, recorded act: change `reserved_status`
    in the population file, with a predeclared decision rule next to it.
    """
    import json

    from experiments.v21_quality.stack import INSPECTED, POPULATIONS, RERANKER_DIR, reserved_user_ids, spent_user_ids

    path = POPULATIONS / "small_profile_v1.json"
    if not path.exists():
        pytest.skip("the small-profile populations have not been defined in this checkout")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["reserved_status"] == "UNOPENED"

    reserved = set().union(*reserved_user_ids().values())
    development = {int(user_id) for ids in record["cohorts"].values() for user_id in ids}
    assert len(reserved) == sum(len(ids) for ids in record["reserved"].values()), "a user is reserved twice"
    assert not reserved & development, "a reserved user was also scored as a development user"

    if INSPECTED.exists() and (RERANKER_DIR / "confirm_user_ids.json.gz").exists():
        spent = set().union(*spent_user_ids().values())
        assert not reserved & spent, "a reserved user has already been scored by an experiment"
        assert development <= spent, "development users must be recorded as spent"
