"""How good are recommendations when the profile is as small as a real session's?

The offline population is dominated by heavy users (a mean of 86 liked titles),
and the uniform 800-user sample contains no sparse user at all. A live Anime
Compass profile is a handful of Likes. This measures the same frozen stack on
the same users with their profile cut down to K liked titles.

For each user and each K, the profile is a deterministic hash-ordered subset of
K train positives. Relevance is unchanged: the held-out test positives. So every
K is scored against the identical target, and K is the only thing that varies.

Also reported: the share of the top 10 that the user is known to like at all
(any rating >= 8 outside the K-title profile). That is closer to what a small
profile experiences, with one caveat stated in the output: the titles a user
liked but that were left out of the K-title profile were part of ALS training.

No model is trained or selected. Users are the reranker experiment's
confirmation users, already opened once; no untouched user is spent.

    python experiments/v21_quality/profile_size_curve.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.evaluation.reranking import RerankerFeatureSpace  # noqa: E402
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402
from backend.anime_agent.recommender import series_key  # noqa: E402
from experiments.v21_quality.diagnose_baseline import (  # noqa: E402
    ARTIFACTS,
    CANDIDATES,
    CATALOG,
    OUTPUT,
    RERANKER_DIR,
    SERVICE_FETCH,
    SERVICE_LIMIT,
    SPLIT,
    collapse_series,
    mean_ci,
    paired,
    top_rows,
)
from scripts.evaluate_reranker import FrozenALS, load_als, ndcg_at_k, recall_at_k  # noqa: E402

PROFILE_SIZES = (1, 2, 3, 5, 10, 20, 50)
MINIMUM_TRAIN_POSITIVES = 50  # every user must be able to supply the largest profile


def main() -> int:
    started = time.perf_counter()
    import lightgbm as lgb

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    anime_ids, factors, als_meta = load_als(ARTIFACTS / "als_train_only.npz")
    als = FrozenALS(factors, float(als_meta.get("alpha", 5.0)), float(als_meta.get("regularization", 0.05)))
    space = RerankerFeatureSpace.from_artifacts(
        catalog,
        anime_ids,
        popularity_path=ARTIFACTS / "popularity_train_only.npz",
        quality_path=ARTIFACTS / "countsketch_train_only.npz",
        item_item_path=ARTIFACTS / "item_item_train_only.npz",
    )
    booster = lgb.Booster(model_file=str(RERANKER_DIR / "lambdamart.txt"))
    row_by_id = space.index_by_id
    title_by_id = {int(item["id"]): str(item.get("title") or "") for item in catalog}
    keys = [series_key(title_by_id.get(int(anime_id), str(anime_id))) for anime_id in anime_ids.tolist()]

    populations = json.loads(gzip.open(RERANKER_DIR / "confirm_user_ids.json.gz").read())
    store = SplitStore(SPLIT)
    metrics: dict[int, dict[str, list[float]]] = {size: {} for size in PROFILE_SIZES}
    users = 0

    for user_id in sorted(int(value) for value in populations["confirm"]):
        user = store.get_user(user_id)
        if user is None:
            continue
        train = [row_by_id[a] for a in user.train_positive_ids if a in row_by_id]
        relevant = {row_by_id[a] for a, _r in user.test_positive if a in row_by_id}
        if len(train) < MINIMUM_TRAIN_POSITIVES or not relevant:
            continue
        users += 1
        all_likes = set(train) | relevant | {row_by_id[a] for a, _r in user.validation_positive if a in row_by_id}
        ordered = sorted(
            train, key=lambda row: hashlib.blake2b(f"profile-size-v1:{user_id}:{row}".encode(), digest_size=8).digest()
        )

        for size in PROFILE_SIZES:
            profile = ordered[:size]
            scores = als.scores(profile)
            scores[profile] = -np.inf
            top = top_rows(scores, CANDIDATES)
            als_order = [int(row) for row in top]
            features = space.build(profile, als_order, scores[top].astype(np.float32))
            learned = np.asarray(booster.predict(features), dtype=np.float32)
            lm_order = [als_order[i] for i in np.argsort(-learned, kind="stable")]
            other_likes = all_likes - set(profile)
            profile_series = {keys[row] for row in profile}
            reachable = {row for row in relevant if keys[row] not in profile_series}
            bucket = metrics[size]

            def put(name: str, value: float, _bucket: dict[str, list[float]] = bucket) -> None:
                _bucket.setdefault(name, []).append(float(value))

            put("recall@300", len(relevant.intersection(als_order)) / len(relevant))
            for arm, ranking in (("als", als_order), ("lambdamart", lm_order)):
                put(f"{arm}_ndcg@10", ndcg_at_k(ranking, relevant, 10))
                put(f"{arm}_recall@18", recall_at_k(ranking, relevant, 18))
                put(f"{arm}_known_likes_in_top10", len(other_likes.intersection(ranking[:10])) / 10)
                served = collapse_series(ranking[:SERVICE_FETCH], profile, keys)[:SERVICE_LIMIT]
                put(f"{arm}_served_known_likes_in_top10", len(other_likes.intersection(served[:10])) / 10)
                put(f"{arm}_served_ndcg@10", ndcg_at_k(served, reachable, 10) if reachable else 0.0)

    report: dict[str, Any] = {
        "protocol": {
            "users": users,
            "user_population": "reranker confirmation users with at least 50 train positives (already opened once)",
            "profile_sizes": list(PROFILE_SIZES),
            "profile_selection": "blake2b('profile-size-v1:<user>:<row>') order over the user's train positives",
            "relevance": "held-out test positives, identical for every profile size",
            "caveat": (
                "known_likes_in_top10 counts every title the user rated >= 8 outside the K-title profile. "
                "Train positives left out of the profile were part of ALS training, so this is an optimistic "
                "reading of small-profile quality, not a held-out one. ndcg and recall use held-out positives only."
            ),
        },
        "by_profile_size": {},
    }
    for size in PROFILE_SIZES:
        bucket = metrics[size]
        report["by_profile_size"][str(size)] = {
            "recall@300": mean_ci(bucket["recall@300"]),
            **{name: mean_ci(bucket[name]) for name in sorted(bucket) if name != "recall@300"},
            "reranker_gain_ndcg@10": paired(bucket["als_ndcg@10"], bucket["lambdamart_ndcg@10"]),
            "reranker_gain_served_ndcg@10": paired(bucket["als_served_ndcg@10"], bucket["lambdamart_served_ndcg@10"]),
            "reranker_gain_known_likes_in_top10": paired(
                bucket["als_known_likes_in_top10"], bucket["lambdamart_known_likes_in_top10"]
            ),
            "reranker_gain_served_known_likes_in_top10": paired(
                bucket["als_served_known_likes_in_top10"], bucket["lambdamart_served_known_likes_in_top10"]
            ),
        }
    report["runtime_seconds"] = round(time.perf_counter() - started, 1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "profile_size_curve.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{users} users, {report['runtime_seconds']}s")
    header = f"{'K':>3} {'recall@300':>10} {'ALS n@10':>9} {'LM n@10':>9} {'gain':>22} {'served gain':>22} {'likes/10 ALS':>12} {'LM':>6} {'served LM':>9}"
    print(header)
    for size in PROFILE_SIZES:
        row = report["by_profile_size"][str(size)]
        gain = row["reranker_gain_ndcg@10"]
        served = row["reranker_gain_served_ndcg@10"]
        print(
            f"{size:>3} {row['recall@300']['mean']:>10.4f} {row['als_ndcg@10']['mean']:>9.4f} "
            f"{row['lambdamart_ndcg@10']['mean']:>9.4f} "
            f"{gain['mean']:+.4f} [{gain['ci_low']:+.4f},{gain['ci_high']:+.4f}] "
            f"{served['mean']:+.4f} [{served['ci_low']:+.4f},{served['ci_high']:+.4f}] "
            f"{row['als_known_likes_in_top10']['mean']:>12.3f} {row['lambdamart_known_likes_in_top10']['mean']:>6.3f} "
            f"{row['lambdamart_served_known_likes_in_top10']['mean']:>9.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
