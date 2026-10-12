"""How good are recommendations for someone who has Liked 1, 3, 5, 10 or 20 titles?

The standing benchmark is dominated by long histories (a mean of 86 Likes; no
sparse user in the uniform 800). A new Anime Compass user has a handful. This
is a separate evaluation of exactly that case, on the frozen stack, with no
model trained or selected.

Two cohorts, because each answers a different doubt:

  truncated   Users with at least 20 train positives, shown to the models with
              only K of them. The same users at every K, so K is the only thing
              that changes. They are heavy users pretending to be new.
  natural     Users whose entire train profile really is K Likes. They are
              genuinely light users, with only two or three hidden positives
              each. K = 1 does not exist here: the split needs five positives.

What a model may see is the K observed Likes and nothing else. Hidden positives
(validation and test) are never in the profile, never in the features, and
never in the exclusion list; `protocol.assert_isolated` enforces it per user.
Train positives left out of a truncated profile are not targets and are not
excluded either: they stay in the list and count as misses, which is the
conservative reading.

Users. The truncated cohort reuses the reranker experiment's confirmation
users, already opened once. The natural cohort draws fresh users and records
them as spent. A separate confirmation set is reserved here and is NOT scored:
it stays untouched for the first model selection that needs it.

    python experiments/v21_quality/small_profile_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v21_quality.diagnose_baseline import gini, mean_ci, paired  # noqa: E402
from experiments.v21_quality.protocol import (  # noqa: E402
    BATCH,
    apply_series_policy,
    assert_isolated,
    build_case,
    stable_order,
)
from experiments.v21_quality.stack import POPULATIONS, RESULTS, WINDOW, Stack, load_stack, spent_user_ids  # noqa: E402
from scripts.evaluate_reranker import ndcg_at_k, recall_at_k  # noqa: E402

PROFILE_SIZES = (1, 3, 5, 10, 20)
NATURAL_SIZES = (3, 5, 10, 20)
DEPTHS = (300, 500, 1000, 2000)
CUTOFFS = (10, 18)
SERVED_POLICIES = ("production", "dedupe", "dedupe_liked3")
WINDOWS = 4
BATCHES_TRACKED = 12
POPULATION_FILE = POPULATIONS / "small_profile_v1.json"
SALT = "small-profile-population-v1"
DEV_NATURAL = 500
CONFIRM_NATURAL = 500
CONFIRM_TRUNCATED = 800
TRUNCATION_MINIMUM = 20


def define_populations(stack: Stack) -> dict[str, Any]:
    """Draw the development cohorts and reserve the confirmation set, once.

    Deterministic: hash order over users nothing has scored yet. The file is
    written the first time and only read afterwards, so a rerun scores the same
    users and the reservation cannot drift.
    """
    if POPULATION_FILE.exists():
        return json.loads(POPULATION_FILE.read_text(encoding="utf-8"))

    spent = set().union(*spent_user_ids().values())
    activity = stack.store.eligible_user_activity()
    untouched = {size: [] for size in NATURAL_SIZES}
    heavy: list[int] = []
    for user_id, train_count in activity:
        if user_id in spent:
            continue
        if train_count in untouched:
            untouched[train_count].append(user_id)
        if train_count >= TRUNCATION_MINIMUM:
            heavy.append(user_id)

    cohorts: dict[str, list[int]] = {}
    reserved: dict[str, list[int]] = {}
    taken: set[int] = set()
    for size in NATURAL_SIZES:
        ordered = stable_order(untouched[size], f"{SALT}:natural:{size}")
        cohorts[f"dev_natural_{size}"] = sorted(ordered[:DEV_NATURAL])
        reserved[f"confirm_natural_{size}"] = sorted(ordered[DEV_NATURAL : DEV_NATURAL + CONFIRM_NATURAL])
        taken.update(ordered[: DEV_NATURAL + CONFIRM_NATURAL])
    ordered = [user_id for user_id in stable_order(heavy, f"{SALT}:truncated") if user_id not in taken]
    reserved["confirm_truncated"] = sorted(ordered[:CONFIRM_TRUNCATED])

    everyone = [user_id for ids in (*cohorts.values(), *reserved.values()) for user_id in ids]
    assert len(everyone) == len(set(everyone)), "a user was drawn into two cohorts"
    assert not set(everyone) & spent, "a cohort includes a user an earlier experiment scored"
    record = {
        "version": 1,
        "salt": SALT,
        "split": "holdout_seed42_pos8.sqlite",
        "previously_spent_users": len(spent),
        "untouched_eligible_users_before": sum(1 for user_id, _count in activity if user_id not in spent),
        "cohorts_note": "Development cohorts. Scored by small_profile_eval.py; now spent, usable for diagnostics only.",
        "cohorts": cohorts,
        "reserved_note": (
            "Confirmation set. UNOPENED: no script has scored these users. Open once, for one model "
            "selection, against a decision rule written beforehand."
        ),
        "reserved": reserved,
        "reserved_status": "UNOPENED",
        "sha256": {
            name: hashlib.sha256(json.dumps(ids).encode()).hexdigest() for name, ids in {**cohorts, **reserved}.items()
        },
    }
    POPULATIONS.mkdir(parents=True, exist_ok=True)
    POPULATION_FILE.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


class Accumulator:
    """Per-user metrics and exposure counts for one (cohort, K) cell."""

    def __init__(self, items: int):
        self.items = items
        self.values: dict[str, list[float]] = {}
        self.exposure: dict[str, Counter[int]] = {}
        self.users = 0

    def put(self, name: str, value: float) -> None:
        self.values.setdefault(name, []).append(float(value))

    def expose(self, name: str, rows: list[int]) -> None:
        self.exposure.setdefault(name, Counter()).update(rows)


def score_case(stack: Stack, cell: Accumulator, case: Any) -> None:
    profile = list(case.profile)
    hidden = set(case.hidden)
    keys = stack.keys
    liked_series = {keys[row] for row in profile}
    new = {row for row in hidden if keys[row] not in liked_series}
    cell.users += 1
    cell.put("hidden_positives", len(hidden))
    cell.put("hidden_in_a_liked_franchise", len(hidden) - len(new))

    # --- retrieval: the profile is the only exclusion
    assert_isolated(case, exclusions=profile)
    scores = stack.als_scores(profile)
    deep = stack.top(scores, max(DEPTHS))
    for depth in DEPTHS:
        cell.put(f"recall@{depth}", len(hidden.intersection(deep[:depth])) / len(hidden))
    als_order = deep[:WINDOW]

    # --- the paged sequence, as the service builds it; window 0 is the top-300 rerank
    timings: dict[str, list[float]] = {}
    served, sizes, raw = stack.sequence(profile, windows=WINDOWS, policy="production", timings=timings)
    lm_order = raw[0]
    ranked_all = [row for window in raw for row in window]
    cell.put("latency_retrieval_ms", timings["retrieval_ms"][0])
    cell.put("latency_rerank_ms", timings["rerank_ms"][0])
    cell.put("latency_first_page_ms", timings["retrieval_ms"][0] + timings["rerank_ms"][0])
    cell.put("latency_all_windows_ms", sum(timings["retrieval_ms"]) + sum(timings["rerank_ms"]))

    for arm, ranking in (("als", als_order), ("lambdamart", lm_order)):
        for k in CUTOFFS:
            cell.put(f"{arm}.ndcg@{k}", ndcg_at_k(ranking, hidden, k))
            cell.put(f"{arm}.recall@{k}", recall_at_k(ranking, hidden, k))
            cell.expose(f"{arm}@{k}", ranking[:k])
        if case.unobserved_likes:
            known = hidden | case.unobserved_likes
            cell.put(f"{arm}.known_likes@10", len(known.intersection(ranking[:10])) / 10)

    # --- as served, under each series policy
    for policy in SERVED_POLICIES:
        for arm, ranking in (("als", als_order), ("lambdamart", lm_order)):
            shown = apply_series_policy(ranking, profile, keys, policy, types=stack.types)
            for k in CUTOFFS:
                cell.put(f"{policy}.{arm}.ndcg@{k}", ndcg_at_k(shown, hidden, k))
                cell.put(f"{policy}.{arm}.recall@{k}", recall_at_k(shown, hidden, k))
            cell.put(f"{policy}.{arm}.hits@18", len(hidden.intersection(shown[:18])))
            cell.put(f"{policy}.{arm}.new_hits@18", len(new.intersection(shown[:18])))
            if arm == "lambdamart":
                cell.put(f"{policy}.liked_franchise_slots@18", sum(keys[row] in liked_series for row in shown[:18]))
                cell.expose(f"{policy}@18", shown[:18])

    # --- how deep the feed goes, and what the depth is worth
    cell.put("depth.total", len(served))
    cell.put("depth.batches", len(served) / BATCH)
    position = 0
    for window, size in enumerate(sizes):
        chunk = served[position : position + size]
        position += size
        cell.put(f"depth.window{window}.titles", size)
        cell.put(f"depth.window{window}.hidden_found", len(hidden.intersection(chunk)) / len(hidden))
    for missing in range(len(sizes), WINDOWS):
        cell.put(f"depth.window{missing}.titles", 0)
        cell.put(f"depth.window{missing}.hidden_found", 0.0)
    for batch in range(BATCHES_TRACKED):
        chunk = served[batch * BATCH : (batch + 1) * BATCH]
        cell.put(f"depth.batch{batch + 1}.hidden_per_18", len(hidden.intersection(chunk)))
    cell.put("depth.hidden_found_in_sequence", len(hidden.intersection(served)) / len(hidden))
    # The same titles with no series filter, for the cost of the filter at depth.
    cell.put("depth.unfiltered_total", len(ranked_all))


def summarise(cell: Accumulator, stack: Stack) -> dict[str, Any]:
    def stat(name: str) -> dict[str, float] | None:
        return mean_ci(cell.values[name]) if name in cell.values else None

    def percentiles(name: str) -> dict[str, float]:
        data = np.asarray(cell.values[name])
        return {"p50": float(np.percentile(data, 50)), "p95": float(np.percentile(data, 95))}

    out: dict[str, Any] = {
        "users": cell.users,
        "hidden_positives_per_user": stat("hidden_positives"),
        "hidden_in_a_liked_franchise_per_user": stat("hidden_in_a_liked_franchise"),
        "retrieval": {f"recall@{depth}": stat(f"recall@{depth}") for depth in DEPTHS},
        "ranking": {},
        "as_served": {},
        "coverage": {},
        "latency_ms": {
            "als_fold_in_and_retrieval": percentiles("latency_retrieval_ms"),
            "lambdamart_features_and_scoring": percentiles("latency_rerank_ms"),
            "first_page": percentiles("latency_first_page_ms"),
            "rebuilding_all_four_windows": percentiles("latency_all_windows_ms"),
        },
        "depth": {
            "distinct_titles": stat("depth.total"),
            "batches_of_18": stat("depth.batches"),
            "minimum_titles": float(np.min(cell.values["depth.total"])),
            "p05_titles": float(np.percentile(cell.values["depth.total"], 5)),
            "per_window": {
                f"window_{window}": {
                    "titles": stat(f"depth.window{window}.titles"),
                    "share_of_hidden_positives_found": stat(f"depth.window{window}.hidden_found"),
                }
                for window in range(WINDOWS)
            },
            "hidden_positives_per_batch": {
                f"batch_{batch + 1}": stat(f"depth.batch{batch + 1}.hidden_per_18")["mean"]
                for batch in range(BATCHES_TRACKED)
            },
            "share_of_hidden_positives_in_whole_sequence": stat("depth.hidden_found_in_sequence"),
        },
    }
    for arm in ("als", "lambdamart"):
        out["ranking"][arm] = {
            **{f"{metric}@{k}": stat(f"{arm}.{metric}@{k}") for metric in ("ndcg", "recall") for k in CUTOFFS},
            "known_likes@10": stat(f"{arm}.known_likes@10"),
        }
    out["ranking"]["reranker_gain"] = {
        f"{metric}@{k}": paired(cell.values[f"als.{metric}@{k}"], cell.values[f"lambdamart.{metric}@{k}"])
        for metric in ("ndcg", "recall")
        for k in CUTOFFS
    }
    for policy in SERVED_POLICIES:
        out["as_served"][policy] = {
            "als_ndcg@10": stat(f"{policy}.als.ndcg@10"),
            "lambdamart_ndcg@10": stat(f"{policy}.lambdamart.ndcg@10"),
            "lambdamart_ndcg@18": stat(f"{policy}.lambdamart.ndcg@18"),
            "lambdamart_recall@10": stat(f"{policy}.lambdamart.recall@10"),
            "lambdamart_recall@18": stat(f"{policy}.lambdamart.recall@18"),
            "hits@18": stat(f"{policy}.lambdamart.hits@18"),
            "new_franchise_hits@18": stat(f"{policy}.lambdamart.new_hits@18"),
            "liked_franchise_slots@18": stat(f"{policy}.liked_franchise_slots@18"),
            "reranker_gain_ndcg@10": paired(
                cell.values[f"{policy}.als.ndcg@10"], cell.values[f"{policy}.lambdamart.ndcg@10"]
            ),
            "vs_production_ndcg@10": paired(
                cell.values["production.lambdamart.ndcg@10"], cell.values[f"{policy}.lambdamart.ndcg@10"]
            ),
            "vs_production_new_hits@18": paired(
                cell.values["production.lambdamart.new_hits@18"], cell.values[f"{policy}.lambdamart.new_hits@18"]
            ),
        }
    for name, counter in sorted(cell.exposure.items()):
        rows = np.fromiter(counter.keys(), dtype=np.int64)
        counts = np.zeros(stack.items)
        counts[rows] = np.fromiter(counter.values(), dtype=np.float64)
        slots = counts.sum()
        out["coverage"][name] = {
            "distinct_titles": int(len(rows)),
            "catalog_coverage": len(rows) / stack.items,
            "head_share": float(counts[stack.bucket == "head"].sum() / slots),
            "top_20_titles_share_of_slots": float(np.sort(counts)[-20:].sum() / slots),
            "exposure_gini": gini(counts),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--define-only", action="store_true", help="write the population file and stop")
    args = parser.parse_args()
    started = time.perf_counter()
    stack = load_stack()
    populations = define_populations(stack)
    if populations["reserved_status"] != "UNOPENED":
        print("note: the reserved confirmation set is no longer marked UNOPENED")
    if args.define_only:
        print({name: len(ids) for name, ids in {**populations["cohorts"], **populations["reserved"]}.items()})
        return 0

    report: dict[str, Any] = {
        "protocol": {
            "profile_sizes": list(PROFILE_SIZES),
            "relevance": "hidden positives: validation and test holdouts, never seen by any model",
            "profile": "K train positives in a fixed hash order; profiles nest across K",
            "exclusions": "the observed profile only",
            "isolation": "assert_isolated() ran for every user at every K",
            "candidates": WINDOW,
            "windows": WINDOWS,
            "policies": list(SERVED_POLICIES),
            "latency": "in-process, single thread, development machine; the serving API adds a few milliseconds",
            "populations_file": str(POPULATION_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "reserved_confirmation_set": {name: len(ids) for name, ids in populations["reserved"].items()},
            "reserved_status": populations["reserved_status"],
        },
        "truncated": {},
        "natural": {},
    }

    # ---------------------------------------------------------- truncated
    cells = {size: Accumulator(stack.items) for size in PROFILE_SIZES}
    for user_id in sorted(spent_user_ids()["reranker_confirm"]):
        user = stack.store.get_user(user_id)
        if user is None:
            continue
        train = stack.rows(user.train_positive)
        hidden = set(stack.rows(user.test_positive)) | set(stack.rows(user.validation_positive))
        if len(train) < TRUNCATION_MINIMUM or not hidden:
            continue
        low = set(stack.rows((*user.explicit_negative, *user.neutral, *user.ignored)))
        for size in PROFILE_SIZES:
            score_case(stack, cells[size], build_case(user_id, train, hidden, low, size))
    for size in PROFILE_SIZES:
        report["truncated"][str(size)] = summarise(cells[size], stack)
    print(f"truncated cohort: {cells[PROFILE_SIZES[0]].users} users, {time.perf_counter() - started:.0f}s")

    # ------------------------------------------------------------ natural
    for size in NATURAL_SIZES:
        cell = Accumulator(stack.items)
        for user_id in populations["cohorts"][f"dev_natural_{size}"]:
            user = stack.store.get_user(int(user_id))
            if user is None:
                continue
            train = stack.rows(user.train_positive)
            hidden = set(stack.rows(user.test_positive)) | set(stack.rows(user.validation_positive))
            if not train or not hidden:
                continue
            low = set(stack.rows((*user.explicit_negative, *user.neutral, *user.ignored)))
            # The whole train profile: these users really have this few Likes.
            score_case(stack, cell, build_case(int(user_id), train, hidden, low, None))
        report["natural"][str(size)] = summarise(cell, stack)
        print(f"natural cohort K={size}: {cell.users} users, {time.perf_counter() - started:.0f}s")

    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "small_profile_eval.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    for cohort in ("truncated", "natural"):
        print(f"\n{cohort}")
        print(
            f"{'K':>3} {'n':>5} {'R@300':>6} {'R@1000':>6} {'ALS n@10':>8} {'LM n@10':>8} {'gain':>24} {'served n@10':>11} {'R@18':>6} {'depth':>6} {'p50 ms':>7}"
        )
        for size, row in report[cohort].items():
            gain = row["ranking"]["reranker_gain"]["ndcg@10"]
            print(
                f"{size:>3} {row['users']:>5} {row['retrieval']['recall@300']['mean']:>6.3f} "
                f"{row['retrieval']['recall@1000']['mean']:>6.3f} {row['ranking']['als']['ndcg@10']['mean']:>8.4f} "
                f"{row['ranking']['lambdamart']['ndcg@10']['mean']:>8.4f} "
                f"{gain['mean']:+.4f} [{gain['ci_low']:+.4f},{gain['ci_high']:+.4f}] "
                f"{row['as_served']['production']['lambdamart_ndcg@10']['mean']:>11.4f} "
                f"{row['as_served']['production']['lambdamart_recall@18']['mean']:>6.3f} "
                f"{row['depth']['distinct_titles']['mean']:>6.0f} {row['latency_ms']['first_page']['p50']:>7.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
