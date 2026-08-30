"""Compare the old and new production recommendation architectures.

Same users, same split, same protocol. The arms are:

* ``hybrid_countsketch`` -- the previous default: ten-channel hybrid, CountSketch
  collaborative channel.
* ``fast_als`` -- the new default: ALS retrieval + lightweight ranking.
* ``fast_als_item_item`` -- the new default plus item-item tail supplementation.
* ``fast_als_diversity`` -- the new default with a bounded diversity rerank.
* ``fast_segment_routed`` -- segment-aware routing (sparse users -> CountSketch).
* ``fast_global_als`` -- routing disabled, everyone served by ALS.

The last two exist to test whether routing earns its complexity, rather than
assuming it does.

    python scripts/benchmark_production_architecture.py --users 800
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backend.anime_agent.als_serving import ALSCollaborativeIndex
from backend.anime_agent.collaborative import CollaborativeIndex
from backend.anime_agent.evaluation.collaborative_baselines import ItemItemModel
from backend.anime_agent.evaluation.metrics import (
    catalog_coverage,
    intra_list_diversity,
    mean_novelty,
    paired_bootstrap_aligned,
    popularity_bias,
    ranking_metrics,
)
from backend.anime_agent.evaluation.models import (
    compute_train_statistics,
    sanitize_catalog_with_training_statistics,
)
from backend.anime_agent.evaluation.split import SplitStore, UserSplit, select_evaluation_user_ids
from backend.anime_agent.fast_path import FastPathConfig, recommend_fast
from backend.anime_agent.recommender import AnimeRecommender
from backend.anime_agent.retrieval import RetrievalConfig
from backend.anime_agent.routing import RoutingPolicy, activity_segment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "splits" / "holdout_seed42_pos8.sqlite"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "artifacts" / "holdout_seed42_pos8"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "results" / "production_architecture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=800)
    parser.add_argument("--sample-seed", type=int, default=20260901)
    parser.add_argument("--exclude-user-ids", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip the slow hybrid arm.")
    return parser.parse_args()


def read_ids(path: Path | None) -> tuple[int, ...]:
    if path is None:
        return ()
    return tuple(
        sorted(
            {
                int(line.strip())
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
        )
    )


class CountSketchSource:
    """Adapt the CountSketch index to the retrieval source protocol."""

    def __init__(self, index: CollaborativeIndex):
        self.index = index

    def top_candidates(self, positive_ids, limit, *, excluded_ids=()):
        scores = self.index.profile_scores(positive_ids=list(positive_ids))
        blocked = {int(v) for v in excluded_ids} | {int(v) for v in positive_ids}
        ranked = sorted((a for a in scores if a not in blocked), key=lambda a: (-scores[a], a))
        return ranked[:limit]


class ItemItemSource:
    def __init__(self, model: ItemItemModel):
        self.model = model

    def top_candidates(self, positive_ids, limit, *, excluded_ids=()):
        blocked = {int(v) for v in excluded_ids} | {int(v) for v in positive_ids}
        subject = UserSplit(
            user_id=0,
            eligible=True,
            train_positive=tuple((int(v), 9) for v in positive_ids),
            validation_positive=(),
            test_positive=(),
            explicit_negative=(),
            neutral=(),
        )
        return [a for a in self.model.recommend(subject, limit + len(blocked)).anime_ids if a not in blocked][:limit]


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def main() -> None:
    args = parse_args()
    store = SplitStore(args.split)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_ids = [int(item["id"]) for item in catalog]

    statistics = compute_train_statistics(store, catalog_ids)
    counts = statistics.positive_counts_by_id()
    countsketch = CollaborativeIndex.load(args.artifacts_dir / "countsketch_train_only.npz", catalog)
    als = ALSCollaborativeIndex.load(args.artifacts_dir / "als_train_only.npz", catalog, quality_source=countsketch)
    item_item_path = args.artifacts_dir / "item_item_train_only.npz"
    item_item = ItemItemSource(ItemItemModel(item_item_path, catalog_ids, build_duration_seconds=0.0))
    countsketch_source = CountSketchSource(countsketch)

    evaluation_catalog = sanitize_catalog_with_training_statistics(catalog, statistics, countsketch)
    catalog_by_id = {int(item["id"]): item for item in evaluation_catalog}
    genres_by_id = {int(item["id"]): tuple(item.get("genres") or ()) for item in evaluation_catalog}
    hybrid = None if args.skip_hybrid else AnimeRecommender(evaluation_catalog, collaborative_index=countsketch)

    reserved = read_ids(args.exclude_user_ids)
    user_ids = select_evaluation_user_ids(
        store, limit=args.users, seed=args.sample_seed, strategy="uniform", excluded_user_ids=reserved
    )
    print(f"users: {len(user_ids)}  (excluded {len(reserved)})")

    always_on = FastPathConfig(routing=RoutingPolicy(medium_threshold=1), retrieval=RetrievalConfig(item_item_top_m=0))
    arms: dict[str, dict[str, Any]] = {
        "fast_als": {"config": always_on, "tail": None},
        "fast_als_item_item": {
            "config": FastPathConfig(
                routing=RoutingPolicy(medium_threshold=1), retrieval=RetrievalConfig(item_item_top_m=100)
            ),
            "tail": item_item,
        },
        "fast_als_diversity": {
            "config": FastPathConfig(
                routing=RoutingPolicy(medium_threshold=1),
                retrieval=RetrievalConfig(item_item_top_m=0),
                diversity_strength=0.3,
                diversity_window=30,
            ),
            "tail": None,
        },
        "fast_segment_routed": {"config": FastPathConfig(routing=RoutingPolicy()), "tail": None},
        "fast_global_als": {
            "config": FastPathConfig(routing=RoutingPolicy(segment_aware=False)),
            "tail": None,
        },
    }

    rows: list[dict[str, Any]] = []
    per_user: dict[str, dict[int, dict[str, float]]] = {}
    rankings_by_arm: dict[str, list[list[int]]] = {}
    histories: list[list[int]] = []
    segments: dict[int, str] = {}

    def score_ranking(name: str, user: UserSplit, ranking: list[int], elapsed_ms: float) -> None:
        metrics = ranking_metrics(ranking, set(user.test_positive_ids))
        per_user.setdefault(name, {})[user.user_id] = {
            "ndcg_at_10": metrics.ndcg_at_10,
            "recall_at_10": metrics.recall_at_10,
            "hit_rate_at_10": metrics.hit_rate_at_10,
            "ndcg_at_20": metrics.ndcg_at_20,
            "recall_at_20": metrics.recall_at_20,
            "latency_ms": elapsed_ms,
        }
        rankings_by_arm.setdefault(name, []).append(ranking)

    users = [u for u in store.iter_users_by_ids(user_ids) if u.test_positive_ids]
    for position, user in enumerate(users, start=1):
        positives = list(user.train_positive_ids)
        known = [a for a, _ in user.all_observed_training_ratings]
        histories.append(positives)
        segments[user.user_id] = activity_segment(len(positives))

        for name, spec in arms.items():
            started = time.perf_counter()
            result = recommend_fast(
                positives,
                catalog_by_id=catalog_by_id,
                als_source=als,
                fallback_source=countsketch_source,
                tail_source=spec["tail"],
                quality_lookup=als,
                excluded_ids=known,
                limit=args.limit,
                config=spec["config"],
            )
            score_ranking(name, user, result.anime_ids, (time.perf_counter() - started) * 1000.0)

        if hybrid is not None:
            started = time.perf_counter()
            rows_out = hybrid.recommend(
                liked_ids=positives,
                excluded_ids=[a for a, _ in (*user.explicit_negative, *user.neutral, *user.ignored)],
                session_profile={},
                diversity_strength=0.12,
                exclude_related_series=False,
                limit=args.limit,
                include_explanations=False,
            )
            score_ranking(
                "hybrid_countsketch",
                user,
                [int(r["id"]) for r in rows_out],
                (time.perf_counter() - started) * 1000.0,
            )
        if position % 100 == 0:
            print(f"  {position}/{len(users)}", flush=True)

    total_train = sum(counts.values())
    for name, values in per_user.items():
        ordered = [values[u.user_id] for u in users if u.user_id in values]
        rankings = rankings_by_arm[name]
        latencies = [row["latency_ms"] for row in ordered]
        row = {
            "arm": name,
            "users": len(ordered),
            **{
                key: round(float(np.mean([r[key] for r in ordered])), 6)
                for key in ("ndcg_at_10", "recall_at_10", "hit_rate_at_10", "ndcg_at_20", "recall_at_20")
            },
            "catalog_coverage": round(catalog_coverage(rankings, len(catalog_ids)), 6),
            "novelty_bits": round(mean_novelty(rankings, counts, catalog_size=len(catalog_ids)), 6),
            "popularity_bias": round(popularity_bias(rankings, histories, counts), 6),
            "intra_list_diversity": round(float(np.mean([intra_list_diversity(r, genres_by_id) for r in rankings])), 6),
            "p50_latency_ms": round(_percentile(latencies, 50), 3),
            "p95_latency_ms": round(_percentile(latencies, 95), 3),
            "total_train_positives": total_train,
        }
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "architecture_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Paired comparisons against the previous default, where it was measured.
    comparisons: list[dict[str, Any]] = []
    baseline = "hybrid_countsketch" if "hybrid_countsketch" in per_user else "fast_als"
    for name in per_user:
        if name == baseline:
            continue
        shared = sorted(set(per_user[name]) & set(per_user[baseline]))
        for metric in ("ndcg_at_10", "recall_at_10", "ndcg_at_20", "recall_at_20"):
            a = np.asarray([per_user[name][u][metric] for u in shared])
            b = np.asarray([per_user[baseline][u][metric] for u in shared])
            interval = paired_bootstrap_aligned(a - b, iterations=2000, seed=42)
            comparisons.append(
                {
                    "left": name,
                    "right": baseline,
                    "metric": metric,
                    "users": len(shared),
                    "difference": round(float(a.mean() - b.mean()), 6),
                    "relative": round(float((a.mean() - b.mean()) / b.mean()) if b.mean() else 0.0, 6),
                    "ci_lower": round(float(interval["ci_lower"]), 6),
                    "ci_upper": round(float(interval["ci_upper"]), 6),
                    "ci_excludes_zero": bool(interval["ci_lower"] > 0 or interval["ci_upper"] < 0),
                }
            )
    with (args.output_dir / "paired_bootstrap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    # Segment breakdown, to test whether routing earns its complexity.
    segment_rows: list[dict[str, Any]] = []
    for name in per_user:
        for segment in ("sparse", "medium", "heavy"):
            chosen = [u for u, s in segments.items() if s == segment and u in per_user[name]]
            if not chosen:
                continue
            segment_rows.append(
                {
                    "arm": name,
                    "segment": segment,
                    "users": len(chosen),
                    **{
                        key: round(float(np.mean([per_user[name][u][key] for u in chosen])), 6)
                        for key in ("ndcg_at_10", "recall_at_10", "ndcg_at_20")
                    },
                }
            )
    with (args.output_dir / "segment_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(segment_rows[0]))
        writer.writeheader()
        writer.writerows(segment_rows)

    print(f"\n{'arm':<24}{'NDCG@10':>9}{'R@10':>8}{'NDCG@20':>9}{'R@20':>8}{'Cov':>8}{'ILD':>8}{'p50':>10}{'p95':>10}")
    print("-" * 94)
    for row in sorted(rows, key=lambda r: -r["ndcg_at_10"]):
        print(
            f"{row['arm']:<24}{row['ndcg_at_10']:>9.4f}{row['recall_at_10']:>8.4f}{row['ndcg_at_20']:>9.4f}"
            f"{row['recall_at_20']:>8.4f}{row['catalog_coverage']:>8.4f}{row['intra_list_diversity']:>8.4f}"
            f"{row['p50_latency_ms']:>10.2f}{row['p95_latency_ms']:>10.2f}"
        )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
