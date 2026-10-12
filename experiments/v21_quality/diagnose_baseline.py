"""Where does the current recommender lose quality? A read-only baseline diagnosis.

Nothing here trains, tunes, or selects a model. It scores the frozen
evaluation stack exactly as `scripts/evaluate_reranker.py` does:

    frozen train-only ALS  ->  top-300 candidates  ->  confirmed LambdaMART

and then takes the ranking apart: how much is lost at retrieval, how much at
ranking, who and what it is lost on, and what the product's own filters cost.

Users. The 800 confirmation users of the reranker experiment. They were opened
once for that decision and are therefore already spent; reusing them here
consumes no untouched user. They must not serve as a confirmation set again.

Run from the repository root:

    python experiments/v21_quality/diagnose_baseline.py
"""

from __future__ import annotations

import argparse
import gzip
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

from backend.anime_agent.evaluation.reranking import RerankerFeatureSpace  # noqa: E402
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402
from backend.anime_agent.recommender import series_key  # noqa: E402
from scripts.evaluate_reranker import FrozenALS, load_als, ndcg_at_k, recall_at_k  # noqa: E402

PERSONALIZED = PROJECT_ROOT / "data" / "evaluation" / "personalized"
SPLIT = PERSONALIZED / "splits" / "holdout_seed42_pos8.sqlite"
ARTIFACTS = PERSONALIZED / "artifacts" / "holdout_seed42_pos8"
CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
RERANKER_DIR = PERSONALIZED / "results" / "reranker"
OUTPUT = Path(__file__).resolve().parent / "results"

CANDIDATES = 300  # production retrieval depth (RETRIEVAL_ALS_TOP_N)
DEPTHS = (10, 18, 50, 100, 200, 300, 500, 1000, 2000)
CUTOFFS = (10, 18)  # 18 is one page of the For You feed
SERVICE_FETCH = 200  # top_k * 4 for the feed's limit of 50 with one_per_series
SERVICE_LIMIT = 50
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20261012
PUBLISHED = {"als_ndcg10": 0.2456, "lambdamart_ndcg10": 0.2828, "als_recall10": 0.2312, "lambdamart_recall10": 0.2537}


def mean_ci(values: list[float] | np.ndarray) -> dict[str, float]:
    """Mean with a percentile bootstrap interval over users."""
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    index = generator.integers(0, len(data), (BOOTSTRAP_ITERATIONS, len(data)))
    low, high = np.percentile(data[index].mean(axis=1), [2.5, 97.5])
    return {"mean": float(data.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(data))}


def paired(base: list[float], challenger: list[float]) -> dict[str, Any]:
    """Paired user-level bootstrap of challenger minus base."""
    delta = mean_ci(np.asarray(challenger, dtype=np.float64) - np.asarray(base, dtype=np.float64))
    delta["significant"] = bool(delta["ci_low"] > 0 or delta["ci_high"] < 0)
    return delta


def dcg_prefix(count: int) -> float:
    return float(sum(1.0 / np.log2(i + 2) for i in range(count)))


def oracle_ndcg(hits_in_candidates: int, relevant: int, k: int) -> float:
    """Best NDCG@k any reordering of a candidate set could reach."""
    ideal = dcg_prefix(min(k, relevant))
    return dcg_prefix(min(k, hits_in_candidates)) / ideal if ideal > 0 else 0.0


def gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=np.float64))
    if values.sum() <= 0:
        return 0.0
    n = len(values)
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * (cumulative / cumulative[-1]).sum()) / n)


def top_rows(scores: np.ndarray, depth: int) -> np.ndarray:
    take = min(depth, int(np.isfinite(scores).sum()))
    top = np.argpartition(-scores, take - 1)[:take]
    return top[np.argsort(-scores[top], kind="stable")]


def collapse_series(ranked: list[int], profile: list[int], keys: list[str]) -> list[int]:
    """`RecommendationService._collapse_series`, on artifact rows."""
    seen = {keys[row] for row in profile}
    kept: list[int] = []
    for row in ranked:
        key = keys[row]
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--rating-sample", type=int, default=30000, help="users for the rating-scale statistics")
    args = parser.parse_args()
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
    item_count = len(anime_ids)
    title_by_id = {int(item["id"]): str(item.get("title") or "") for item in catalog}
    keys = [series_key(title_by_id.get(int(anime_id), str(anime_id))) for anime_id in anime_ids.tolist()]

    with np.load(ARTIFACTS / "popularity_train_only.npz", allow_pickle=False) as payload:
        assert np.array_equal(payload["anime_ids"], anime_ids), "popularity artifact is not aligned with ALS"
        popularity = np.asarray(payload["positive_count"], dtype=np.float64)
    # Buckets as the published protocol defines them: rank by train positives,
    # ID as tie-break; head = top 20%, mid = next 30%, tail = bottom 50%.
    order = np.lexsort((anime_ids, -popularity))
    bucket = np.empty(item_count, dtype="<U4")
    bucket[order[: int(item_count * 0.2)]] = "head"
    bucket[order[int(item_count * 0.2) : int(item_count * 0.5)]] = "mid"
    bucket[order[int(item_count * 0.5) :]] = "tail"
    log_popularity = np.log1p(popularity)
    normalised_popularity = log_popularity / log_popularity.max()

    with np.load(ARTIFACTS / "item_item_train_only.npz", allow_pickle=False) as payload:
        neighbor_indices = np.asarray(payload["neighbor_indices"], dtype=np.int64)
        neighbor_scores = np.asarray(payload["neighbor_scores"], dtype=np.float32)

    populations = json.loads(gzip.open(RERANKER_DIR / "confirm_user_ids.json.gz").read())
    user_ids = sorted(int(value) for value in populations["confirm"])
    store = SplitStore(SPLIT)
    print(f"loaded stack in {time.perf_counter() - started:.1f}s; scoring {len(user_ids)} already-opened users")

    per_user: dict[str, list[float]] = {}

    def put(name: str, value: float) -> None:
        per_user.setdefault(name, []).append(float(value))

    segments: list[str] = []
    exposure: dict[str, Counter[int]] = {f"{arm}@{k}": Counter() for arm in ("als", "lambdamart") for k in CUTOFFS}
    relevant_by_bucket: Counter[str] = Counter()
    retrieved_by_bucket: Counter[str] = Counter()
    top10_by_bucket: Counter[str] = Counter()
    composition: dict[str, Counter[str]] = {f"{arm}@{k}": Counter() for arm in ("als", "lambdamart") for k in CUTOFFS}
    relevant_total = 0
    relevant_same_series = 0
    conditional: dict[str, Counter[str]] = {"als": Counter(), "lambdamart": Counter()}
    rank_of_retrieved: dict[str, list[int]] = {"als": [], "lambdamart": []}
    scored = 0

    for user_id in user_ids:
        user = store.get_user(user_id)
        if user is None:
            continue
        profile = [row_by_id[a] for a in user.train_positive_ids if a in row_by_id]
        relevant = {row_by_id[a] for a, _r in user.test_positive if a in row_by_id}
        if not profile or not relevant:
            continue
        scored += 1
        validation = {row_by_id[a] for a, _r in user.validation_positive if a in row_by_id}
        negatives = {row_by_id[a] for a, _r in user.explicit_negative if a in row_by_id}
        neutral = {row_by_id[a] for a, _r in (*user.neutral, *user.ignored) if a in row_by_id}
        n_relevant = len(relevant)
        segment = "sparse" if len(profile) <= 4 else "medium" if len(profile) <= 19 else "heavy"
        segments.append(segment)

        scores = als.scores(profile)
        scores[profile] = -np.inf  # the reranker protocol: only train positives are known
        deep = top_rows(scores, max(DEPTHS))
        deep_list = [int(row) for row in deep]
        als_order = deep_list[:CANDIDATES]
        candidate_scores = scores[deep[:CANDIDATES]].astype(np.float32)
        features = space.build(profile, als_order, candidate_scores)
        learned = np.asarray(booster.predict(features), dtype=np.float32)
        lm_order = [als_order[i] for i in np.argsort(-learned, kind="stable")]

        # --- retrieval at depth
        for depth in DEPTHS:
            put(f"recall@{depth}", len(relevant.intersection(deep_list[:depth])) / n_relevant)
        hits300 = len(relevant.intersection(als_order))
        put("all_relevant_retrieved@300", 1.0 if hits300 == n_relevant else 0.0)
        put("none_retrieved@300", 1.0 if hits300 == 0 else 0.0)

        # --- ranking, and the ceiling a perfect reranker would reach
        for k in CUTOFFS:
            put(f"als_ndcg@{k}", ndcg_at_k(als_order, relevant, k))
            put(f"als_recall@{k}", recall_at_k(als_order, relevant, k))
            put(f"lambdamart_ndcg@{k}", ndcg_at_k(lm_order, relevant, k))
            put(f"lambdamart_recall@{k}", recall_at_k(lm_order, relevant, k))
            put(f"oracle_recall@{k}", min(k, n_relevant) / n_relevant)
            for depth in (300, 500, 1000):
                hits = len(relevant.intersection(deep_list[:depth]))
                put(f"candidate_oracle_ndcg@{k}_depth{depth}", oracle_ndcg(hits, n_relevant, k))
                put(f"candidate_oracle_recall@{k}_depth{depth}", min(k, hits) / n_relevant)
        put("als_hit@10", 1.0 if relevant.intersection(als_order[:10]) else 0.0)
        put("lambdamart_hit@10", 1.0 if relevant.intersection(lm_order[:10]) else 0.0)

        # --- conditional ranking quality: given a positive WAS retrieved, where did it land?
        for arm, ranking in (("als", als_order), ("lambdamart", lm_order)):
            for position, row in enumerate(ranking):
                if row in relevant:
                    conditional[arm]["retrieved"] += 1
                    conditional[arm]["in_top10"] += position < 10
                    conditional[arm]["in_top18"] += position < 18
                    conditional[arm]["in_top50"] += position < 50
                    rank_of_retrieved[arm].append(position + 1)

        # --- exposure, popularity, and what the top of the list is made of
        profile_popularity = float(normalised_popularity[profile].mean())
        for arm, ranking in (("als", als_order), ("lambdamart", lm_order)):
            for k in CUTOFFS:
                head = ranking[:k]
                exposure[f"{arm}@{k}"].update(head)
                put(f"{arm}_popularity_lift@{k}", float(normalised_popularity[head].mean()) - profile_popularity)
                put(f"{arm}_mean_log_popularity@{k}", float(log_popularity[head].mean()))
                for row in head:
                    kind = (
                        "held_out_test_like"
                        if row in relevant
                        else "held_out_validation_like"
                        if row in validation
                        else "already_rated_5_or_lower"
                        if row in negatives
                        else "already_rated_6_or_7"
                        if row in neutral
                        else "never_rated"
                    )
                    composition[f"{arm}@{k}"][kind] += 1
        for row in relevant:
            relevant_by_bucket[str(bucket[row])] += 1
        for row in relevant.intersection(als_order):
            retrieved_by_bucket[str(bucket[row])] += 1
        for row in relevant.intersection(lm_order[:10]):
            top10_by_bucket[str(bucket[row])] += 1

        # --- hard exclusions: the published protocol removes everything the user ever rated
        known_other = negatives | neutral
        put("observed_non_positive_in_top300", len(known_other.intersection(als_order)))
        strict = scores.copy()
        if known_other:
            strict[list(known_other)] = -np.inf
        strict_order = [int(row) for row in top_rows(strict, CANDIDATES)]
        strict_features = space.build(profile, strict_order, strict[strict_order].astype(np.float32))
        strict_learned = np.asarray(booster.predict(strict_features), dtype=np.float32)
        strict_lm = [strict_order[i] for i in np.argsort(-strict_learned, kind="stable")]
        put("strict_recall@300", len(relevant.intersection(strict_order)) / n_relevant)
        for k in CUTOFFS:
            put(f"strict_als_ndcg@{k}", ndcg_at_k(strict_order, relevant, k))
            put(f"strict_lambdamart_ndcg@{k}", ndcg_at_k(strict_lm, relevant, k))
            put(f"strict_lambdamart_recall@{k}", recall_at_k(strict_lm, relevant, k))

        # --- one_per_series, exactly as the service applies it to the feed's request
        profile_series = {keys[row] for row in profile}
        unreachable = {row for row in relevant if keys[row] in profile_series}
        relevant_total += n_relevant
        relevant_same_series += len(unreachable)
        put("share_relevant_in_a_liked_series", len(unreachable) / n_relevant)
        survivors = collapse_series(lm_order[:SERVICE_FETCH], profile, keys)
        collapsed = survivors[:SERVICE_LIMIT]
        put("series_survivors_of_200", len(survivors))
        put("series_feed_shortfall", 1.0 if len(survivors) < SERVICE_LIMIT else 0.0)
        put("series_first_page_shortfall", 1.0 if len(survivors) < 18 else 0.0)
        put("series_survivors_of_300", len(collapse_series(lm_order, profile, keys)))
        reachable = relevant - unreachable
        als_collapsed = collapse_series(als_order[:SERVICE_FETCH], profile, keys)[:SERVICE_LIMIT]
        for arm, ranking in (("als", als_order), ("lambdamart", lm_order)):
            put(f"{arm}_top10_slots_in_a_liked_series", sum(keys[row] in profile_series for row in ranking[:10]) / 10)
        if reachable:
            # One-per-series can show a franchise once, so even a perfect
            # reranker scores one hit per franchise among the retrieved positives.
            showable = {keys[row] for row in reachable.intersection(als_order)}
            put("served_reachable_recall@300", len(reachable.intersection(als_order)) / len(reachable))
            for k in CUTOFFS:
                put(f"served_candidate_oracle_ndcg@{k}", oracle_ndcg(len(showable), len(reachable), k))
                put(f"served_als_ndcg@{k}", ndcg_at_k(als_collapsed, reachable, k))
                put(f"served_als_recall@{k}", recall_at_k(als_collapsed, reachable, k))
        for k in CUTOFFS:
            put(f"collapsed_als_ndcg@{k}", ndcg_at_k(als_collapsed, relevant, k))
            put(f"collapsed_ndcg@{k}", ndcg_at_k(collapsed, relevant, k))
            put(f"collapsed_recall@{k}", recall_at_k(collapsed, relevant, k))
            # Scored only on positives the product is allowed to show.
            put(f"reachable_users_flag@{k}", 1.0 if reachable else 0.0)
            if reachable:
                put(f"collapsed_reachable_ndcg@{k}", ndcg_at_k(collapsed, reachable, k))
                put(f"uncollapsed_reachable_ndcg@{k}", ndcg_at_k(lm_order, reachable, k))
                put(f"collapsed_reachable_recall@{k}", recall_at_k(collapsed, reachable, k))
                put(f"uncollapsed_reachable_recall@{k}", recall_at_k(lm_order, reachable, k))

        # --- retrieval-source combination at an equal candidate budget
        neighbour_scores = np.zeros(item_count, dtype=np.float32)
        for row in profile:
            np.add.at(neighbour_scores, neighbor_indices[row], neighbor_scores[row])
        neighbour_scores[profile] = -np.inf
        neighbour_order = [int(row) for row in top_rows(neighbour_scores, 1000) if neighbour_scores[row] > 0]
        for extra in (100, 200, 700):
            union = list(dict.fromkeys([*als_order, *neighbour_order]))[: CANDIDATES + extra]
            budget = CANDIDATES + extra
            put(f"union_recall@{budget}", len(relevant.intersection(union)) / n_relevant)
            put(f"als_recall_equal_budget@{budget}", len(relevant.intersection(deep_list[:budget])) / n_relevant)
            union_tail = [row for row in relevant.intersection(union) if bucket[row] != "head"]
            als_tail = [row for row in relevant.intersection(deep_list[:budget]) if bucket[row] != "head"]
            put(f"union_non_head_hits@{budget}", len(union_tail))
            put(f"als_non_head_hits@{budget}", len(als_tail))

    print(f"scored {scored} users in {time.perf_counter() - started:.1f}s")
    segment_array = np.asarray(segments)

    def summary(name: str) -> dict[str, float]:
        return mean_ci(per_user[name])

    def by_segment(name: str) -> dict[str, dict[str, float]]:
        values = np.asarray(per_user[name])
        return {
            segment: mean_ci(values[segment_array == segment])
            for segment in ("sparse", "medium", "heavy")
            if (segment_array == segment).any()
        }

    report: dict[str, Any] = {
        "protocol": {
            "split": SPLIT.name,
            "als_artifact": "als_train_only.npz (evaluation role, frozen)",
            "reranker": "results/reranker/lambdamart.txt (confirmed arm)",
            "candidates": CANDIDATES,
            "users": scored,
            "user_population": "reranker experiment confirmation users (already opened once)",
            "user_ids_sha256": hashlib.sha256(json.dumps(user_ids).encode()).hexdigest(),
            "known_items_excluded": "train positives only (reranker protocol); see `hard_exclusions` for the strict variant",
            "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "seed": BOOTSTRAP_SEED, "interval": "95% percentile"},
            "relevance": "held-out test positives (rating >= 8); held-out validation positives count as misses",
        },
        "sanity_check": {
            "als_ndcg10": float(np.mean(per_user["als_ndcg@10"])),
            "lambdamart_ndcg10": float(np.mean(per_user["lambdamart_ndcg@10"])),
            "als_recall10": float(np.mean(per_user["als_recall@10"])),
            "lambdamart_recall10": float(np.mean(per_user["lambdamart_recall@10"])),
            "published": PUBLISHED,
        },
        "retrieval": {
            **{f"recall@{depth}": summary(f"recall@{depth}") for depth in DEPTHS},
            "all_relevant_retrieved@300": summary("all_relevant_retrieved@300"),
            "none_retrieved@300": summary("none_retrieved@300"),
        },
        "ranking": {
            arm: {
                **{f"ndcg@{k}": summary(f"{arm}_ndcg@{k}") for k in CUTOFFS},
                **{f"recall@{k}": summary(f"{arm}_recall@{k}") for k in CUTOFFS},
                "hit@10": summary(f"{arm}_hit@10"),
            }
            for arm in ("als", "lambdamart")
        },
        "reranker_gain": {
            f"{metric}@{k}": paired(per_user[f"als_{metric}@{k}"], per_user[f"lambdamart_{metric}@{k}"])
            for metric in ("ndcg", "recall")
            for k in CUTOFFS
        },
        "ceiling": {},
        "conditional_ranking": {},
        "segments": {
            "counts": dict(Counter(segments)),
            "recall@300": by_segment("recall@300"),
            "als_ndcg@10": by_segment("als_ndcg@10"),
            "lambdamart_ndcg@10": by_segment("lambdamart_ndcg@10"),
            "lambdamart_recall@18": by_segment("lambdamart_recall@18"),
            "candidate_oracle_ndcg@10_depth300": by_segment("candidate_oracle_ndcg@10_depth300"),
        },
    }
    report["sanity_check"]["reproduces_published"] = bool(
        all(abs(report["sanity_check"][name] - value) < 5e-4 for name, value in PUBLISHED.items())
    )

    for k in CUTOFFS:
        actual = float(np.mean(per_user[f"lambdamart_ndcg@{k}"]))
        als_only = float(np.mean(per_user[f"als_ndcg@{k}"]))
        entry: dict[str, Any] = {
            "full_catalog_oracle_ndcg": 1.0,
            "oracle_recall": summary(f"oracle_recall@{k}"),
            "als_ndcg": als_only,
            "lambdamart_ndcg": actual,
        }
        for depth in (300, 500, 1000):
            ceiling = float(np.mean(per_user[f"candidate_oracle_ndcg@{k}_depth{depth}"]))
            entry[f"depth{depth}"] = {
                "candidate_oracle_ndcg": ceiling,
                "candidate_oracle_recall": float(np.mean(per_user[f"candidate_oracle_recall@{k}_depth{depth}"])),
                "lost_at_retrieval": 1.0 - ceiling,
            }
        ceiling300 = entry["depth300"]["candidate_oracle_ndcg"]
        entry["lost_at_ranking_given_300"] = ceiling300 - actual
        entry["share_of_candidate_ceiling_captured"] = actual / ceiling300 if ceiling300 else 0.0
        entry["extra_headroom_from_deeper_retrieval"] = {
            "300_to_500": entry["depth500"]["candidate_oracle_ndcg"] - ceiling300,
            "300_to_1000": entry["depth1000"]["candidate_oracle_ndcg"] - ceiling300,
        }
        report["ceiling"][f"@{k}"] = entry

    for arm in ("als", "lambdamart"):
        total = conditional[arm]["retrieved"]
        ranks = np.asarray(rank_of_retrieved[arm])
        report["conditional_ranking"][arm] = {
            "retrieved_positives": int(total),
            "share_in_top10": conditional[arm]["in_top10"] / total,
            "share_in_top18": conditional[arm]["in_top18"] / total,
            "share_in_top50": conditional[arm]["in_top50"] / total,
            "median_rank_within_300": float(np.median(ranks)),
            "mean_rank_within_300": float(ranks.mean()),
        }

    coverage: dict[str, Any] = {}
    for name, counter in exposure.items():
        rows = np.fromiter(counter.keys(), dtype=np.int64)
        counts = np.zeros(item_count)
        counts[rows] = np.fromiter(counter.values(), dtype=np.float64)
        slots = counts.sum()
        arm, cutoff = name.split("@")
        coverage[name] = {
            "distinct_items": int(len(rows)),
            "catalog_coverage": len(rows) / item_count,
            "head_share": float(counts[bucket == "head"].sum() / slots),
            "mid_share": float(counts[bucket == "mid"].sum() / slots),
            "tail_share": float(counts[bucket == "tail"].sum() / slots),
            "exposure_gini": gini(counts),
            "top_20_items_share_of_slots": float(np.sort(counts)[-20:].sum() / slots),
            "mean_log_popularity": summary(f"{arm}_mean_log_popularity@{cutoff}")["mean"],
            "popularity_lift_vs_profile": summary(f"{arm}_popularity_lift@{cutoff}"),
        }
    report["coverage_popularity"] = coverage
    report["item_buckets"] = {
        name: {
            "held_out_positives": int(relevant_by_bucket[name]),
            "share_of_held_out": relevant_by_bucket[name] / max(1, sum(relevant_by_bucket.values())),
            "retrieved@300": retrieved_by_bucket[name] / relevant_by_bucket[name] if relevant_by_bucket[name] else None,
            "in_lambdamart_top10": top10_by_bucket[name] / relevant_by_bucket[name]
            if relevant_by_bucket[name]
            else None,
            "catalog_items": int((bucket == name).sum()),
        }
        for name in ("head", "mid", "tail")
    }
    report["list_composition"] = {
        name: {kind: count / (scored * int(name.split("@")[1])) for kind, count in sorted(counter.items())}
        for name, counter in composition.items()
    }
    report["hard_exclusions"] = {
        "observed_non_positive_in_top300": summary("observed_non_positive_in_top300"),
        "recall@300": {
            "train_positives_excluded": summary("recall@300"),
            "all_observed_excluded": summary("strict_recall@300"),
        },
        "recall@300_delta": paired(per_user["recall@300"], per_user["strict_recall@300"]),
        **{
            f"lambdamart_ndcg@{k}": {
                "train_positives_excluded": summary(f"lambdamart_ndcg@{k}"),
                "all_observed_excluded": summary(f"strict_lambdamart_ndcg@{k}"),
                "delta": paired(per_user[f"lambdamart_ndcg@{k}"], per_user[f"strict_lambdamart_ndcg@{k}"]),
            }
            for k in CUTOFFS
        },
        "als_ndcg@10": {
            "train_positives_excluded": summary("als_ndcg@10"),
            "all_observed_excluded": summary("strict_als_ndcg@10"),
            "delta": paired(per_user["als_ndcg@10"], per_user["strict_als_ndcg@10"]),
        },
    }
    report["one_per_series"] = {
        "share_of_held_out_positives_in_a_liked_series": relevant_same_series / relevant_total,
        "per_user_share": summary("share_relevant_in_a_liked_series"),
        "survivors_of_200_fetched": {
            **summary("series_survivors_of_200"),
            "p05": float(np.percentile(per_user["series_survivors_of_200"], 5)),
            "min": float(np.min(per_user["series_survivors_of_200"])),
        },
        "survivors_of_300_candidates": summary("series_survivors_of_300"),
        "users_with_fewer_than_50_results": summary("series_feed_shortfall"),
        "users_with_fewer_than_18_results": summary("series_first_page_shortfall"),
        "top10_slots_in_a_liked_series": {
            "als": summary("als_top10_slots_in_a_liked_series"),
            "lambdamart": summary("lambdamart_top10_slots_in_a_liked_series"),
        },
        # What the feed actually serves: collapsed lists, scored on positives it may show.
        "as_served": {
            "users": int(len(per_user["served_reachable_recall@300"])),
            "reachable_recall@300": summary("served_reachable_recall@300"),
            **{
                f"@{k}": {
                    "als_ndcg": summary(f"served_als_ndcg@{k}"),
                    "lambdamart_ndcg": summary(f"collapsed_reachable_ndcg@{k}"),
                    "reranker_gain_ndcg": paired(
                        per_user[f"served_als_ndcg@{k}"], per_user[f"collapsed_reachable_ndcg@{k}"]
                    ),
                    "als_recall": summary(f"served_als_recall@{k}"),
                    "lambdamart_recall": summary(f"collapsed_reachable_recall@{k}"),
                    "reranker_gain_recall": paired(
                        per_user[f"served_als_recall@{k}"], per_user[f"collapsed_reachable_recall@{k}"]
                    ),
                    "candidate_oracle_ndcg": summary(f"served_candidate_oracle_ndcg@{k}"),
                    "benchmark_relevance_collapsed": {
                        "als_ndcg": summary(f"collapsed_als_ndcg@{k}"),
                        "lambdamart_ndcg": summary(f"collapsed_ndcg@{k}"),
                        "reranker_gain_ndcg": paired(
                            per_user[f"collapsed_als_ndcg@{k}"], per_user[f"collapsed_ndcg@{k}"]
                        ),
                    },
                }
                for k in CUTOFFS
            },
        },
        **{
            f"@{k}": {
                "ndcg_without_collapse": summary(f"lambdamart_ndcg@{k}"),
                "ndcg_with_collapse": summary(f"collapsed_ndcg@{k}"),
                "ndcg_delta": paired(per_user[f"lambdamart_ndcg@{k}"], per_user[f"collapsed_ndcg@{k}"]),
                "recall_delta": paired(per_user[f"lambdamart_recall@{k}"], per_user[f"collapsed_recall@{k}"]),
                "reachable_only": {
                    "users": int(sum(per_user[f"reachable_users_flag@{k}"])),
                    "ndcg_without_collapse": summary(f"uncollapsed_reachable_ndcg@{k}"),
                    "ndcg_with_collapse": summary(f"collapsed_reachable_ndcg@{k}"),
                    "ndcg_delta": paired(
                        per_user[f"uncollapsed_reachable_ndcg@{k}"], per_user[f"collapsed_reachable_ndcg@{k}"]
                    ),
                    "recall_delta": paired(
                        per_user[f"uncollapsed_reachable_recall@{k}"], per_user[f"collapsed_reachable_recall@{k}"]
                    ),
                },
            }
            for k in CUTOFFS
        },
    }
    report["source_combination"] = {
        f"budget_{CANDIDATES + extra}": {
            "als_alone": summary(f"als_recall_equal_budget@{CANDIDATES + extra}"),
            f"als_300_plus_item_item_{extra}": summary(f"union_recall@{CANDIDATES + extra}"),
            "delta": paired(
                per_user[f"als_recall_equal_budget@{CANDIDATES + extra}"],
                per_user[f"union_recall@{CANDIDATES + extra}"],
            ),
            "non_head_positives_retrieved": {
                "als_alone": float(np.sum(per_user[f"als_non_head_hits@{CANDIDATES + extra}"])),
                "union": float(np.sum(per_user[f"union_non_head_hits@{CANDIDATES + extra}"])),
            },
        }
        for extra in (100, 200, 700)
    }

    report["rating_scale"] = rating_scale_statistics(store, args.rating_sample)
    report["runtime_seconds"] = round(time.perf_counter() - started, 1)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "baseline_diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'baseline_diagnosis.json'} in {report['runtime_seconds']}s")
    print(json.dumps(report["sanity_check"], indent=2))
    return 0


def rating_scale_statistics(store: SplitStore, sample_size: int) -> dict[str, Any]:
    """What a fixed "8 or higher" threshold keeps and discards, per user.

    Descriptive statistics over raw ratings only. No model is scored on these
    users, so drawing them does not spend an evaluation user. The sample is
    recorded by digest so it can be reproduced.
    """
    all_ids = store.training_user_ids()
    keyed = sorted(all_ids, key=lambda uid: hashlib.blake2b(f"rating-scale-v1:{uid}".encode(), digest_size=8).digest())
    sample = sorted(keyed[:sample_size])

    share_positive: list[float] = []
    user_mean: list[float] = []
    totals = Counter()
    eligible_now = 0
    eligible_relative = 0
    gained_users = 0
    for user in store.iter_users_by_ids(sample):
        ratings = np.asarray(
            [
                rating
                for _anime, rating in (
                    *user.train_positive,
                    *user.validation_positive,
                    *user.test_positive,
                    *user.explicit_negative,
                    *user.neutral,
                    *user.ignored,
                )
            ],
            dtype=np.float64,
        )
        if len(ratings) < 5:
            totals["users_with_fewer_than_5_ratings"] += 1
            continue
        mean = float(ratings.mean())
        positive = ratings >= 8
        above_own_mean = ratings > mean
        share_positive.append(float(positive.mean()))
        user_mean.append(mean)
        totals["users"] += 1
        totals["ratings"] += len(ratings)
        totals["threshold_positives"] += int(positive.sum())
        totals["threshold_positives_not_above_own_mean"] += int((positive & ~above_own_mean).sum())
        totals["ratings_6_or_7"] += int(((ratings >= 6) & (ratings <= 7)).sum())
        totals["ratings_6_or_7_above_own_mean"] += int(((ratings >= 6) & (ratings <= 7) & above_own_mean).sum())
        totals["ratings_5_or_lower"] += int((ratings <= 5).sum())
        totals["ratings_below_own_mean_minus_1sd"] += int((ratings < mean - ratings.std()).sum())
        totals["relative_positives"] += int(above_own_mean.sum())
        totals["both"] += int((positive & above_own_mean).sum())
        totals["either"] += int((positive | above_own_mean).sum())
        has_now = int(positive.sum()) >= 5
        has_relative = int(above_own_mean.sum()) >= 5
        eligible_now += has_now
        eligible_relative += has_relative
        gained_users += (not has_now) and has_relative

    shares = np.asarray(share_positive)
    means = np.asarray(user_mean)
    users = max(1, totals["users"])
    return {
        "sample": {
            "requested": sample_size,
            "users_with_5_or_more_ratings": int(totals["users"]),
            "users_with_fewer_than_5_ratings": int(totals["users_with_fewer_than_5_ratings"]),
            "ratings": int(totals["ratings"]),
            "selection": "blake2b('rating-scale-v1:<user_id>') order over all training users",
            "user_ids_sha256": hashlib.sha256(json.dumps(sample).encode()).hexdigest(),
        },
        "share_of_a_users_ratings_that_are_8_or_higher": {
            "mean": float(shares.mean()),
            "deciles": [float(value) for value in np.percentile(shares, range(10, 100, 10))],
            "users_with_80pct_or_more": float((shares >= 0.8).mean()),
            "users_with_20pct_or_less": float((shares <= 0.2).mean()),
        },
        "user_mean_rating": {
            "mean": float(means.mean()),
            "deciles": [float(value) for value in np.percentile(means, range(10, 100, 10))],
            "users_with_mean_8_or_higher": float((means >= 8).mean()),
            "users_with_mean_below_6_5": float((means < 6.5).mean()),
        },
        "information_at_the_threshold": {
            "share_of_all_ratings_that_are_8_or_higher": totals["threshold_positives"] / totals["ratings"],
            "share_of_all_ratings_that_are_6_or_7": totals["ratings_6_or_7"] / totals["ratings"],
            "share_of_all_ratings_that_are_5_or_lower": totals["ratings_5_or_lower"] / totals["ratings"],
            "threshold_positives_at_or_below_the_users_own_mean": totals["threshold_positives_not_above_own_mean"]
            / max(1, totals["threshold_positives"]),
            "ratings_6_or_7_above_the_users_own_mean": totals["ratings_6_or_7_above_own_mean"]
            / max(1, totals["ratings_6_or_7"]),
            "jaccard_threshold_vs_above_own_mean": totals["both"] / max(1, totals["either"]),
            "ratings_more_than_1sd_below_own_mean": totals["ratings_below_own_mean_minus_1sd"] / totals["ratings"],
        },
        "eligibility": {
            "users_with_5_threshold_positives": eligible_now / users,
            "users_with_5_above_own_mean": eligible_relative / users,
            "users_gaining_a_usable_profile": gained_users / users,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
