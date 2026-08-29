"""Measure candidate retrieval recall separately from ranking.

The fusion work found that a large fraction of held-out positives never entered
a 300-item shortlist. If a relevant title is not retrieved, no reranker can
recover it, so retrieval recall bounds everything downstream.

This measures, for each candidate source, what fraction of a user's held-out
positives appear in the top-N candidates, at several depths, split by item
popularity bucket. Unions are evaluated by merging candidate lists.

    python scripts/evaluate_candidate_retrieval.py --users 400
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backend.anime_agent.collaborative import CollaborativeIndex
from backend.anime_agent.evaluation.collaborative_baselines import ALSModel, ItemItemModel
from backend.anime_agent.evaluation.metrics import build_item_popularity_buckets
from backend.anime_agent.evaluation.models import (
    compute_train_statistics,
    sanitize_catalog_with_training_statistics,
)
from backend.anime_agent.evaluation.split import SplitStore, UserSplit, select_evaluation_user_ids
from backend.anime_agent.recommender import AnimeRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "splits" / "holdout_seed42_pos8.sqlite"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "artifacts" / "holdout_seed42_pos8"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "results" / "candidate_retrieval"

DEPTHS = (100, 300, 500, 1000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--users", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=20260901)
    parser.add_argument("--exclude-user-ids", type=Path)
    parser.add_argument("--include-content", action="store_true", help="Also measure the content hybrid (slow).")
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


def _recall(candidates: Sequence[int], relevant: set[int], depth: int) -> float:
    if not relevant:
        return 0.0
    return len(set(candidates[:depth]) & relevant) / len(relevant)


def _bucket_recall(
    candidates: Sequence[int],
    relevant: set[int],
    depth: int,
    bucket_by_id: dict[int, str],
    bucket: str,
) -> float | None:
    subset = {a for a in relevant if bucket_by_id.get(int(a)) == bucket}
    if not subset:
        return None
    return len(set(candidates[:depth]) & subset) / len(subset)


def main() -> None:
    args = parse_args()
    store = SplitStore(args.split)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_ids = [int(item["id"]) for item in catalog]

    statistics = compute_train_statistics(store, catalog_ids)
    counts = statistics.positive_counts_by_id()
    bucket_by_id = build_item_popularity_buckets(np.asarray(sorted(counts)), counts)

    index = CollaborativeIndex.load(args.artifacts_dir / "countsketch_train_only.npz", catalog)
    als = ALSModel(args.artifacts_dir / "als_train_only.npz", catalog_ids, build_duration_seconds=0.0)
    item_item_path = args.artifacts_dir / "item_item_train_only.npz"
    item_item = (
        ItemItemModel(item_item_path, catalog_ids, build_duration_seconds=0.0) if item_item_path.exists() else None
    )

    content: AnimeRecommender | None = None
    if args.include_content:
        content = AnimeRecommender(
            sanitize_catalog_with_training_statistics(catalog, statistics, index),
            collaborative_index=None,
        )

    max_depth = max(DEPTHS)

    def countsketch_candidates(user: UserSplit) -> list[int]:
        scores = index.profile_scores(positive_ids=user.train_positive_ids)
        known = {a for a, _ in user.all_observed_training_ratings}
        ranked = sorted((a for a in scores if a not in known), key=lambda a: (-scores[a], a))
        return ranked[:max_depth]

    def als_candidates(user: UserSplit) -> list[int]:
        return als.recommend(user, max_depth).anime_ids

    def item_item_candidates(user: UserSplit) -> list[int]:
        assert item_item is not None
        return item_item.recommend(user, max_depth).anime_ids

    def content_candidates(user: UserSplit) -> list[int]:
        assert content is not None
        rows = content.recommend(
            liked_ids=list(user.train_positive_ids),
            session_profile={},
            diversity_strength=0.0,
            exclude_related_series=False,
            limit=max_depth,
            include_explanations=False,
        )
        return [int(row["id"]) for row in rows]

    sources: dict[str, Callable[[UserSplit], list[int]]] = {
        "countsketch": countsketch_candidates,
        "als": als_candidates,
    }
    if item_item is not None:
        sources["item_item"] = item_item_candidates
    if content is not None:
        sources["content"] = content_candidates

    reserved = read_ids(args.exclude_user_ids)
    user_ids = select_evaluation_user_ids(
        store, limit=args.users, seed=args.sample_seed, strategy="uniform", excluded_user_ids=reserved
    )
    print(f"users: {len(user_ids)}  sources: {list(sources)}")

    per_source: dict[str, list[list[int]]] = {name: [] for name in sources}
    relevants: list[set[int]] = []
    latencies: dict[str, list[float]] = {name: [] for name in sources}

    for position, user in enumerate(store.iter_users_by_ids(user_ids), start=1):
        relevant = set(user.test_positive_ids)
        if not relevant:
            continue
        relevants.append(relevant)
        for name, fetch in sources.items():
            started = time.perf_counter()
            per_source[name].append(fetch(user))
            latencies[name].append((time.perf_counter() - started) * 1000.0)
        if position % 100 == 0:
            print(f"  {position}/{len(user_ids)}", flush=True)

    # Unions are order-preserving interleaves, which is how a real multi-source
    # retriever merges: take from each source in turn until the depth is filled.
    def interleave(lists: Sequence[Sequence[int]], depth: int) -> list[int]:
        merged: list[int] = []
        seen: set[int] = set()
        for rank in range(max(len(x) for x in lists)):
            for source in lists:
                if rank < len(source):
                    value = int(source[rank])
                    if value not in seen:
                        seen.add(value)
                        merged.append(value)
                        if len(merged) >= depth:
                            return merged
        return merged

    unions: dict[str, tuple[str, ...]] = {}
    if "content" in sources:
        unions["als+content"] = ("als", "content")
    unions["als+countsketch"] = ("als", "countsketch")
    if item_item is not None:
        unions["als+item_item"] = ("als", "item_item")
    if "content" in sources and item_item is not None:
        unions["als+content+item_item"] = ("als", "content", "item_item")

    rows: list[dict[str, Any]] = []
    for name in list(sources) + list(unions):
        for depth in DEPTHS:
            recalls: list[float] = []
            head: list[float] = []
            mid: list[float] = []
            tail: list[float] = []
            for position, relevant in enumerate(relevants):
                if name in sources:
                    cands = per_source[name][position][:depth]
                else:
                    cands = interleave([per_source[s][position] for s in unions[name]], depth)
                recalls.append(_recall(cands, relevant, depth))
                for bucket, sink in (("head", head), ("mid_tail", mid), ("long_tail", tail)):
                    value = _bucket_recall(cands, relevant, depth, bucket_by_id, bucket)
                    if value is not None:
                        sink.append(value)
            latency = (
                np.asarray(latencies[name])
                if name in sources
                else np.sum([np.asarray(latencies[s]) for s in unions[name]], axis=0)
            )
            rows.append(
                {
                    "source": name,
                    "depth": depth,
                    "candidates": depth,
                    "recall": round(float(np.mean(recalls)), 6),
                    "head_recall": round(float(np.mean(head)), 6) if head else None,
                    "mid_tail_recall": round(float(np.mean(mid)), 6) if mid else None,
                    "long_tail_recall": round(float(np.mean(tail)), 6) if tail else None,
                    "p50_latency_ms": round(float(np.percentile(latency, 50)), 3),
                    "p95_latency_ms": round(float(np.percentile(latency, 95)), 3),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "retrieval_recall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "retrieval_recall.json").write_text(
        json.dumps({"users": len(relevants), "depths": list(DEPTHS), "rows": rows}, indent=2), encoding="utf-8"
    )

    print(f"\n{'source':<26}{'depth':>7}{'recall':>10}{'head':>9}{'mid':>9}{'tail':>9}{'p50 ms':>10}")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['source']:<26}{row['depth']:>7}{row['recall']:>10.4f}"
            f"{(row['head_recall'] or 0):>9.4f}{(row['mid_tail_recall'] or 0):>9.4f}"
            f"{(row['long_tail_recall'] or 0):>9.4f}{row['p50_latency_ms']:>10.2f}"
        )
    print(f"\nwrote {args.output_dir / 'retrieval_recall.csv'}")


if __name__ == "__main__":
    main()
