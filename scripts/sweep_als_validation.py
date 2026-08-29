"""Validation-only hyperparameter sweep for the implicit ALS baseline.

Every configuration is trained on train positives and scored against
**validation** positives. The test split is never read here, so selecting a
configuration from this sweep does not consume the test set.

The selection rule is fixed before the sweep runs: highest validation NDCG@10,
ties broken by validation Recall@10, then by fewer factors (prefer the simpler
model). Guardrail metrics are reported but do not select.

    python scripts/sweep_als_validation.py --grid data/evaluation/personalized/configs/als_sweep.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from backend.anime_agent.evaluation.collaborative_baselines import (
    ALSModel,
    build_als_artifact_from_split,
)
from backend.anime_agent.evaluation.metrics import (
    catalog_coverage,
    mean_novelty,
    popularity_bias,
    ranking_metrics,
)
from backend.anime_agent.evaluation.models import compute_train_statistics
from backend.anime_agent.evaluation.split import SplitStore, select_evaluation_user_ids

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "splits" / "holdout_seed42_pos8.sqlite"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "results" / "als_validation_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts-dir", type=Path, help="Where candidate ALS artifacts are written.")
    parser.add_argument("--grid", type=Path, required=True, help="JSON list of ALS configurations.")
    parser.add_argument("--validation-users", type=int, default=800)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--exclude-user-ids",
        type=Path,
        help="Newline-delimited user IDs reserved for confirmation; excluded from the validation pool.",
    )
    parser.add_argument("--recommendation-k", type=int, default=20)
    return parser.parse_args()


def read_user_ids(path: Path | None) -> tuple[int, ...]:
    if path is None:
        return ()
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return tuple(sorted(set(values)))


def evaluate_on_validation(
    model: ALSModel,
    store: SplitStore,
    user_ids: list[int],
    *,
    k: int,
    train_positive_counts: dict[int, int],
    catalog_size: int,
) -> dict[str, Any]:
    """Score a candidate against validation positives only."""
    rows: list[dict[str, Any]] = []
    rankings: list[list[int]] = []
    histories: list[list[int]] = []

    for user in store.iter_users_by_ids(user_ids):
        relevant = set(user.validation_positive_ids)
        if not relevant:
            continue
        ranking = model.recommend(user, k).anime_ids
        rankings.append(ranking)
        histories.append(list(user.train_positive_ids))
        metrics = ranking_metrics(ranking, relevant)
        rows.append(
            {
                "user_id": user.user_id,
                "ndcg_at_10": metrics.ndcg_at_10,
                "recall_at_10": metrics.recall_at_10,
                "hit_rate_at_10": metrics.hit_rate_at_10,
                "train_positive_count": len(user.train_positive),
            }
        )

    if not rows:
        raise ValueError("No validation users produced a ranking")

    def mean(name: str) -> float:
        return sum(float(row[name]) for row in rows) / len(rows)

    return {
        "validation_users": len(rows),
        "ndcg_at_10": mean("ndcg_at_10"),
        "recall_at_10": mean("recall_at_10"),
        "hit_rate_at_10": mean("hit_rate_at_10"),
        "catalog_coverage": catalog_coverage(rankings, catalog_size),
        "mean_novelty": mean_novelty(rankings, train_positive_counts, catalog_size=catalog_size),
        "popularity_bias": popularity_bias(rankings, histories, train_positive_counts),
    }


def main() -> None:
    args = parse_args()
    store = SplitStore(args.split)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_ids = [int(item["id"]) for item in catalog]
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    if not isinstance(grid, list) or not grid:
        raise SystemExit("--grid must be a non-empty JSON list of configurations")

    artifacts_dir = args.artifacts_dir or (args.output_dir / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reserved = read_user_ids(args.exclude_user_ids)
    if reserved:
        print(f"excluding {len(reserved):,} reserved user IDs from the validation pool")
    validation_user_ids = select_evaluation_user_ids(
        store,
        limit=args.validation_users,
        seed=args.sample_seed,
        strategy="uniform",
        excluded_user_ids=reserved,
    )
    assert not (set(validation_user_ids) & set(reserved))
    print(f"validation users: {len(validation_user_ids):,}")

    statistics = compute_train_statistics(store, catalog_ids)
    train_positive_counts = statistics.positive_counts_by_id()

    results: list[dict[str, Any]] = []
    for index, config in enumerate(grid, start=1):
        name = str(config.get("name") or f"als_{index}")
        artifact = artifacts_dir / f"{name}.npz"
        print(f"\n[{index}/{len(grid)}] {name}: {config}")
        started = time.perf_counter()
        metadata = build_als_artifact_from_split(
            store,
            catalog,
            artifact,
            factors=int(config.get("factors", 64)),
            iterations=int(config.get("iterations", 15)),
            regularization=float(config.get("regularization", 0.05)),
            alpha=float(config.get("alpha", 40.0)),
            cg_steps=int(config.get("cg_steps", 3)),
            seed=int(config.get("seed", 42)),
            confidence_mapping=str(config.get("confidence_mapping", "binary")),
        )
        train_seconds = time.perf_counter() - started
        model = ALSModel(artifact, catalog_ids, build_duration_seconds=train_seconds)
        scored = evaluate_on_validation(
            model,
            store,
            validation_user_ids,
            k=args.recommendation_k,
            train_positive_counts=train_positive_counts,
            catalog_size=len(catalog_ids),
        )
        row = {
            "name": name,
            "factors": metadata["factors"],
            "iterations": metadata["iterations"],
            "regularization": metadata["regularization"],
            "alpha": metadata["alpha"],
            "confidence_mapping": metadata.get("confidence_mapping", "binary"),
            "train_seconds": round(train_seconds, 2),
            **{key: round(float(value), 6) if isinstance(value, float) else value for key, value in scored.items()},
        }
        results.append(row)
        print(
            f"    val NDCG@10={row['ndcg_at_10']:.4f} Recall@10={row['recall_at_10']:.4f} "
            f"coverage={row['catalog_coverage']:.4f} ({train_seconds:.0f}s)"
        )

    # Predeclared selection: NDCG@10, then Recall@10, then prefer fewer factors.
    selected = sorted(results, key=lambda r: (-r["ndcg_at_10"], -r["recall_at_10"], r["factors"]))[0]

    payload = {
        "selection_rule": "max validation NDCG@10; ties by Recall@10, then fewer factors",
        "split": str(args.split),
        "validation_users": len(validation_user_ids),
        "reserved_user_count": len(reserved),
        "sample_seed": args.sample_seed,
        "test_split_read": False,
        "selected": selected,
        "candidates": results,
    }
    (args.output_dir / "sweep.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(results[0])
    with (args.output_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nselected: {selected['name']} (val NDCG@10={selected['ndcg_at_10']:.4f})")
    print(f"wrote {args.output_dir / 'sweep.json'}")


if __name__ == "__main__":
    main()
