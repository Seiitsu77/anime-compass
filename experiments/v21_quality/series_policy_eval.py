"""Is the franchise filter too aggressive, and is "already watched" worth +0.045?

Two questions, answered on the same already-opened users as the baseline
diagnosis, with no production policy changed and no model trained.

1. Series policies. The feed applies one rule that does two different things:
   it keeps one entry per franchise in a list, and it removes every entry of a
   franchise the user already Liked. Those are separated here. Each policy is
   scored against all held-out positives, against the ones in a new franchise,
   and against the ones in an already-Liked franchise, so a gain in one cannot
   hide a loss in another. For sequels the filter currently removes, the
   user's full MAL list says whether they had in fact already watched them.

2. Watched-history exclusion. Removing everything a user rated below 8 was
   worth +0.045 NDCG@10. The split hides a share of each user's positives and
   none of their other ratings, so that figure assumes the system knows a
   user's whole history of things they did not love. Here the non-positives are
   hidden at the same rate as the positives.

    python experiments/v21_quality/series_policy_eval.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v21_quality.diagnose_baseline import mean_ci, paired  # noqa: E402
from experiments.v21_quality.protocol import (  # noqa: E402
    SERIES_POLICIES,
    apply_series_policy,
    assert_isolated,
    build_case,
    holdout_matched_subset,
)
from experiments.v21_quality.stack import RESULTS, WINDOW, load_stack, spent_user_ids  # noqa: E402
from scripts.evaluate_reranker import ndcg_at_k, recall_at_k  # noqa: E402

ARCHIVE = PROJECT_ROOT / "archive" / "animelist.csv"
STATUS_NAMES = {1: "currently watching", 2: "completed, unrated", 3: "on hold", 4: "dropped", 6: "plan to watch"}
ENGAGED = {1, 2, 3, 4}  # started in some form; Plan to Watch is intent, not viewing


def load_user_lists(user_ids: set[int]) -> dict[int, dict[int, tuple[int, int]]]:
    """Each user's full MAL list: anime_id -> (watching_status, rating)."""
    wanted = np.fromiter(user_ids, dtype=np.int64)
    lists: dict[int, dict[int, tuple[int, int]]] = {user_id: {} for user_id in user_ids}
    reader = pd.read_csv(
        ARCHIVE,
        chunksize=8_000_000,
        dtype={"user_id": "int32", "anime_id": "int32", "rating": "int16", "watching_status": "int16"},
        usecols=["user_id", "anime_id", "rating", "watching_status"],
    )
    for chunk in reader:
        rows = chunk[chunk["user_id"].isin(wanted)]
        for user_id, anime_id, rating, status in rows.itertuples(index=False, name=None):
            lists[int(user_id)][int(anime_id)] = (int(status), int(rating))
    return lists


def main() -> int:
    started = time.perf_counter()
    stack = load_stack()
    keys, types = stack.keys, stack.types
    users = sorted(spent_user_ids()["reranker_confirm"])
    lists = load_user_lists(set(users))
    print(f"loaded stack and {len(users)} MAL lists in {time.perf_counter() - started:.0f}s")

    per: dict[str, list[float]] = {}

    def put(name: str, value: float) -> None:
        per.setdefault(name, []).append(float(value))

    status_of: dict[str, Counter[str]] = {"liked_franchise": Counter(), "new_franchise": Counter()}
    scored = 0
    positives_in_liked = 0
    positives_total = 0
    for user_id in users:
        user = stack.store.get_user(user_id)
        if user is None:
            continue
        train = stack.rows(user.train_positive)
        test = set(stack.rows(user.test_positive))
        validation = set(stack.rows(user.validation_positive))
        rated_low = set(stack.rows((*user.explicit_negative, *user.neutral, *user.ignored)))
        if not train or not test:
            continue
        # Validation positives are hidden too: they must never be excluded either.
        case = build_case(user_id, train, test | validation, rated_low, None)
        profile = list(case.profile)
        scored += 1

        scores = stack.als_scores(profile)
        candidates = stack.top(scores, WINDOW)
        als_order = candidates
        lm_order = stack.rerank(profile, candidates, scores)
        liked_series = {keys[row] for row in profile}
        in_liked = {row for row in test if keys[row] in liked_series}
        new = test - in_liked
        positives_total += len(test)
        positives_in_liked += len(in_liked)

        # ------------------------------------------------ series policies
        for policy in SERIES_POLICIES:
            for arm, ranking in (("als", als_order), ("lambdamart", lm_order)):
                served = apply_series_policy(ranking, profile, keys, policy, types=types)
                prefix = f"{policy}.{arm}"
                put(f"{prefix}.ndcg@10", ndcg_at_k(served, test, 10))
                put(f"{prefix}.ndcg@18", ndcg_at_k(served, test, 18))
                put(f"{prefix}.recall@18", recall_at_k(served, test, 18))
                put(f"{prefix}.recall@54", recall_at_k(served, test, 54))
                # Per user, so a policy's effect on discovery is visible on its own.
                put(f"{prefix}.new_hits@18", len(new.intersection(served[:18])))
                put(f"{prefix}.liked_hits@18", len(in_liked.intersection(served[:18])))
                if new:
                    put(f"{prefix}.new_ndcg@10", ndcg_at_k(served, new, 10))
                    put(f"{prefix}.new_recall@18", recall_at_k(served, new, 18))
                if arm == "lambdamart":
                    head = served[:18]
                    counts = Counter(keys[row] for row in head)
                    put(f"{policy}.distinct_franchises@18", len(counts))
                    put(f"{policy}.liked_franchise_slots@18", sum(keys[row] in liked_series for row in head))
                    put(f"{policy}.max_per_franchise@18", max(counts.values()) if counts else 0)
                    put(f"{policy}.available_of_300", len(served))

        # --------------------- what the user had already done with each title
        user_list = lists.get(user_id, {})
        for row in apply_series_policy(lm_order, profile, keys, "dedupe", types=types)[:18]:
            anime_id = int(stack.anime_ids[row])
            if row in test or row in validation:
                label = "completed and liked (held out)"
            elif row in rated_low:
                label = "completed, rated below 8"
            else:
                status, _rating = user_list.get(anime_id, (0, 0))
                label = STATUS_NAMES.get(status, "not in the user's list")
            status_of["liked_franchise" if keys[row] in liked_series else "new_franchise"][label] += 1

        # ------------------------------------------ watched-history exclusion
        engaged = {
            stack.row_by_id[anime_id]
            for anime_id, (status, _rating) in user_list.items()
            if status in ENGAGED and anime_id in stack.row_by_id
        }
        # Anything rated 8+ is a positive of some split role and is not "watched, not loved".
        unrated_engagement = engaged - set(train) - test - validation - rated_low
        known_share = len(train) / (len(train) + len(test) + len(validation))
        arms = {
            "none": set(),
            "all_rated_below_8": set(rated_low),
            "matched_rated_below_8": holdout_matched_subset(user_id, rated_low, known_share),
            "matched_all_watched": holdout_matched_subset(user_id, rated_low | unrated_engagement, known_share),
            "all_watched": rated_low | unrated_engagement,
        }
        put("known_share_of_positives", known_share)
        put("rated_below_8", len(rated_low))
        put("unrated_engagement", len(unrated_engagement))
        for name, excluded in arms.items():
            assert_isolated(case, exclusions=excluded)
            arm_scores = stack.als_scores(profile, excluded)
            arm_candidates = stack.top(arm_scores, WINDOW)
            arm_order = stack.rerank(profile, arm_candidates, arm_scores)
            put(f"watched.{name}.recall@300", len(test.intersection(arm_candidates)) / len(test))
            put(f"watched.{name}.ndcg@10", ndcg_at_k(arm_order, test, 10))
            put(f"watched.{name}.ndcg@18", ndcg_at_k(arm_order, test, 18))
            put(f"watched.{name}.excluded", len(excluded))
            served = apply_series_policy(arm_order, profile, keys, "production")
            if new:
                put(f"watched.{name}.served_new_ndcg@10", ndcg_at_k(served, new, 10))

    def summary(name: str) -> dict[str, float]:
        return mean_ci(per[name])

    report: dict[str, Any] = {
        "protocol": {
            "users": scored,
            "user_population": "reranker experiment confirmation users (already opened once)",
            "candidates": WINDOW,
            "relevance": "held-out test positives; validation positives are hidden too and are never excluded",
            "policies": SERIES_POLICIES,
            "isolation": "assert_isolated() ran for every user and every exclusion arm",
        },
        "held_out_positives": {
            "total": positives_total,
            "in_a_liked_franchise": positives_in_liked,
            "share_in_a_liked_franchise": positives_in_liked / positives_total,
        },
        "series_policies": {},
        "sequel_status": {},
        "watched_history": {},
    }
    base = "production"
    for policy in SERIES_POLICIES:
        entry: dict[str, Any] = {"description": SERIES_POLICIES[policy]}
        for arm in ("als", "lambdamart"):
            prefix = f"{policy}.{arm}"
            entry[arm] = {
                metric: summary(f"{prefix}.{metric}")
                for metric in (
                    "ndcg@10",
                    "ndcg@18",
                    "recall@18",
                    "recall@54",
                    "new_hits@18",
                    "liked_hits@18",
                    "new_ndcg@10",
                    "new_recall@18",
                )
            }
        entry["reranker_gain_ndcg@10"] = paired(per[f"{policy}.als.ndcg@10"], per[f"{policy}.lambdamart.ndcg@10"])
        entry["reranker_gain_new_ndcg@10"] = paired(
            per[f"{policy}.als.new_ndcg@10"], per[f"{policy}.lambdamart.new_ndcg@10"]
        )
        entry["vs_production"] = {
            metric: paired(per[f"{base}.lambdamart.{metric}"], per[f"{policy}.lambdamart.{metric}"])
            for metric in (
                "ndcg@10",
                "ndcg@18",
                "recall@18",
                "recall@54",
                "new_hits@18",
                "new_ndcg@10",
                "new_recall@18",
            )
        }
        entry["list_shape"] = {
            name: summary(f"{policy}.{name}")
            for name in (
                "distinct_franchises@18",
                "liked_franchise_slots@18",
                "max_per_franchise@18",
                "available_of_300",
            )
        }
        report["series_policies"][policy] = entry

    for group, counter in status_of.items():
        total = sum(counter.values())
        report["sequel_status"][group] = {
            "recommendations": total,
            "per_user": total / scored,
            "share": {label: count / total for label, count in counter.most_common()},
        }

    report["watched_history"] = {
        "known_share_of_positives": summary("known_share_of_positives"),
        "titles_rated_below_8_per_user": summary("rated_below_8"),
        "other_watched_titles_per_user": summary("unrated_engagement"),
        "arms": {
            name: {
                "titles_excluded": summary(f"watched.{name}.excluded"),
                "recall@300": summary(f"watched.{name}.recall@300"),
                "ndcg@10": summary(f"watched.{name}.ndcg@10"),
                "ndcg@18": summary(f"watched.{name}.ndcg@18"),
                "served_new_ndcg@10": summary(f"watched.{name}.served_new_ndcg@10"),
                "gain_ndcg@10": paired(per["watched.none.ndcg@10"], per[f"watched.{name}.ndcg@10"]),
                "gain_ndcg@18": paired(per["watched.none.ndcg@18"], per[f"watched.{name}.ndcg@18"]),
                "gain_served_new_ndcg@10": paired(
                    per["watched.none.served_new_ndcg@10"], per[f"watched.{name}.served_new_ndcg@10"]
                ),
            }
            for name in ("none", "all_rated_below_8", "matched_rated_below_8", "matched_all_watched", "all_watched")
        },
    }
    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "series_policy_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def show(value: dict[str, float]) -> str:
        return f"{value['mean']:+.4f} [{value['ci_low']:+.4f},{value['ci_high']:+.4f}]"

    print(f"{scored} users, {report['runtime_seconds']}s")
    print(
        f"{'policy':16} {'NDCG@10':>8} {'NDCG@18':>8} {'R@18':>7} {'new n@10':>8} {'new hits':>8} {'liked hits':>10} {'franchises':>10} {'liked slots':>11}"
    )
    for policy, entry in report["series_policies"].items():
        lm, shape = entry["lambdamart"], entry["list_shape"]
        print(
            f"{policy:16} {lm['ndcg@10']['mean']:>8.4f} {lm['ndcg@18']['mean']:>8.4f} {lm['recall@18']['mean']:>7.4f} "
            f"{lm['new_ndcg@10']['mean']:>8.4f} {lm['new_hits@18']['mean']:>8.3f} {lm['liked_hits@18']['mean']:>10.3f} "
            f"{shape['distinct_franchises@18']['mean']:>10.2f} {shape['liked_franchise_slots@18']['mean']:>11.2f}"
        )
    for name, arm in report["watched_history"]["arms"].items():
        print(
            f"watched {name:24} excluded={arm['titles_excluded']['mean']:6.1f} ndcg@10={arm['ndcg@10']['mean']:.4f} gain {show(arm['gain_ndcg@10'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
