"""The parts of the V2.1 evaluations that must be right for a number to mean anything.

Pure functions, unit-tested in tests/test_v21_protocol.py, shared by every
script in this directory:

* building a profile of K observed Likes while keeping every hidden positive
  out of the profile, out of the reranker's features, and out of the exclusion
  list;
* the series-filtering policies being compared, so "production" here is the
  same rule the service applies;
* hiding non-positive ratings at the same rate as positives, to measure what
  watched-history exclusion is worth without the split's asymmetry.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

BATCH = 18
MAINLINE_TYPES = frozenset({"tv", "movie"})
LIKED_FRANCHISE_CAPS = {"dedupe_liked1": 1, "dedupe_liked3": 3}

# Series policies under comparison. "production" is what the fast path serves.
SERIES_POLICIES: dict[str, str] = {
    "none": "no franchise filter",
    "production": "one entry per franchise, and nothing from a franchise the user already Liked",
    "dedupe": "one entry per franchise; a Liked franchise may contribute one further entry",
    "dedupe_mainline": "as dedupe, but a Liked franchise may contribute only a TV series or a film",
    "dedupe_liked1": "as dedupe, with at most one entry from Liked franchises in each batch of 18",
    "dedupe_liked3": "as dedupe, with at most three entries from Liked franchises in each batch of 18",
    "batch_dedupe": "one entry per franchise within each batch of 18; other entries wait for a later batch",
    "cap2": "at most two entries per franchise, Liked franchises included",
}


def stable_order(values: Iterable[int], salt: str) -> list[int]:
    """A deterministic pseudo-random order, independent of input order."""
    return sorted(
        values, key=lambda value: (hashlib.blake2b(f"{salt}:{value}".encode(), digest_size=8).digest(), value)
    )


def sample_profile(
    user_id: int, train_rows: Sequence[int], size: int | None, *, salt: str = "small-profile-v1"
) -> list[int]:
    """K observed Likes for one user, always the same K for the same user.

    Profiles nest: the K=3 profile is a prefix of the K=5 profile, so a larger
    K only ever adds information. `size=None` keeps the whole train profile.
    """
    ordered = stable_order(set(train_rows), f"{salt}:{user_id}")
    return ordered if size is None else ordered[: max(0, size)]


@dataclass(frozen=True)
class ProfileCase:
    """Everything one evaluated user contributes, with roles kept apart."""

    user_id: int
    # The only thing the models may see: profile, features, and exclusions.
    profile: tuple[int, ...]
    # What the ranking is scored against. Never an input to anything.
    hidden: frozenset[int]
    # Likes the model trained on but this profile does not show it. Neither an
    # input nor a target: they stay in the candidate list and count as misses.
    unobserved_likes: frozenset[int]
    # Titles the user rated below the positive threshold.
    non_positive: frozenset[int]

    @property
    def excluded(self) -> frozenset[int]:
        """What may be removed from candidates: the observed profile, nothing else."""
        return frozenset(self.profile)


def build_case(
    user_id: int,
    train_rows: Iterable[int],
    hidden_rows: Iterable[int],
    non_positive_rows: Iterable[int],
    size: int | None,
    *,
    salt: str = "small-profile-v1",
) -> ProfileCase:
    """Assemble one user's case and refuse it if a hidden positive leaked."""
    train = set(train_rows)
    hidden = frozenset(hidden_rows)
    profile = tuple(sample_profile(user_id, sorted(train), size, salt=salt))
    case = ProfileCase(
        user_id=user_id,
        profile=profile,
        hidden=hidden,
        unobserved_likes=frozenset(train.difference(profile)),
        non_positive=frozenset(non_positive_rows),
    )
    assert_isolated(case)
    return case


def assert_isolated(case: ProfileCase, *, exclusions: Iterable[int] | None = None) -> None:
    """Raise if a hidden positive is visible to a model in any form."""
    profile = set(case.profile)
    if profile & case.hidden:
        raise AssertionError(f"user {case.user_id}: a hidden positive is in the profile")
    if set(case.excluded) & case.hidden:
        raise AssertionError(f"user {case.user_id}: a hidden positive is in the exclusion list")
    if exclusions is not None and set(exclusions) & case.hidden:
        raise AssertionError(f"user {case.user_id}: a hidden positive was excluded from candidates")
    if case.non_positive & case.hidden:
        raise AssertionError(f"user {case.user_id}: a hidden positive is listed as a non-positive rating")
    if case.unobserved_likes & case.hidden:
        raise AssertionError(f"user {case.user_id}: a hidden positive is also a training positive")


def apply_series_policy(
    ranked: Sequence[int],
    profile: Iterable[int],
    keys: Sequence[str],
    policy: str,
    *,
    types: Sequence[str] | None = None,
    batch: int = BATCH,
) -> list[int]:
    """Filter a ranked list of rows under one series policy, preserving order."""
    if policy not in SERIES_POLICIES:
        raise ValueError(f"unknown series policy: {policy}")
    if policy == "none":
        return list(ranked)

    liked_series = {keys[row] for row in profile}
    if policy == "production":
        seen = set(liked_series)
        kept: list[int] = []
        for row in ranked:
            if keys[row] in seen:
                continue
            seen.add(keys[row])
            kept.append(row)
        return kept

    if policy in {"dedupe", "dedupe_mainline"}:
        if policy == "dedupe_mainline" and types is None:
            raise ValueError("dedupe_mainline needs each row's format")
        seen = set()
        kept = []
        for row in ranked:
            key = keys[row]
            if key in seen:
                continue
            if policy == "dedupe_mainline" and key in liked_series:
                assert types is not None
                if str(types[row]).casefold() not in MAINLINE_TYPES:
                    continue
            seen.add(key)
            kept.append(row)
        return kept

    if policy in LIKED_FRANCHISE_CAPS:
        # One per franchise across the list. Entries from Liked franchises are
        # rationed per batch; one over the ration waits for the next batch.
        cap = LIKED_FRANCHISE_CAPS[policy]
        seen = set()
        kept = []
        pending = list(ranked)
        while pending:
            chosen: list[int] = []
            deferred: list[int] = []
            from_liked = 0
            for row in pending:
                key = keys[row]
                if key in seen:
                    continue
                if len(chosen) >= batch:
                    deferred.append(row)
                elif key in liked_series and from_liked >= cap:
                    deferred.append(row)
                else:
                    seen.add(key)
                    chosen.append(row)
                    from_liked += key in liked_series
            if not chosen:
                break
            kept.extend(chosen)
            pending = deferred
        return kept

    if policy == "cap2":
        counts: dict[str, int] = {}
        kept = []
        for row in ranked:
            key = keys[row]
            if counts.get(key, 0) >= 2:
                continue
            counts[key] = counts.get(key, 0) + 1
            kept.append(row)
        return kept

    # batch_dedupe: fill one batch at a time. A second entry from a franchise
    # waits for the next batch, in its original order, and is never dropped.
    # If too few distinct franchises remain to fill a batch, the batch is topped
    # up in rank order rather than left short.
    kept = []
    pending = list(ranked)
    while pending:
        used: set[str] = set()
        chosen: list[int] = []
        deferred: list[int] = []
        for row in pending:
            if len(chosen) < batch and keys[row] not in used:
                used.add(keys[row])
                chosen.append(row)
            else:
                deferred.append(row)
        if len(chosen) < batch:
            top_up = deferred[: batch - len(chosen)]
            chosen.extend(top_up)
            deferred = deferred[len(top_up) :]
        kept.extend(chosen)
        pending = deferred
    return kept


def holdout_matched_subset(
    user_id: int, rows: Iterable[int], keep_fraction: float, *, salt: str = "watched-v1"
) -> set[int]:
    """The share of a user's non-positive ratings that would be known today.

    The split hides a fraction of each user's positives and none of their other
    ratings. Excluding *every* non-positive rating therefore assumes the system
    knows a user's whole history of things they did not love, while knowing
    only part of what they did. This keeps `keep_fraction` of the non-positives
    visible, chosen deterministically, so both kinds are known at the same rate.
    """
    if not 0.0 <= keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be between 0 and 1")
    ordered = stable_order(set(rows), f"{salt}:{user_id}")
    return set(ordered[: round(len(ordered) * keep_fraction)])


def franchise_counts(rows: Sequence[int], keys: Sequence[str]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[keys[row]] = counts.get(keys[row], 0) + 1
    return counts
