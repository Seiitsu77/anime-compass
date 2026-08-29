from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from backend.anime_agent.evaluation.lightfm_training import (  # noqa: E402
    evaluate_lightfm_artifact_on_validation,
)
from backend.anime_agent.evaluation.models import compute_train_statistics  # noqa: E402
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402
from backend.anime_agent.lightfm_serving import LightFMServingIndex  # noqa: E402

DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "splits" / "holdout_seed42_pos8.sqlite"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "results" / "lightfm_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate exported LightFM artifacts and small popularity-penalty grids on validation positives only."
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help="Named exported LightFM artifact; repeat for controlled comparisons.",
    )
    parser.add_argument("--penalty-lambdas", default="0", help="Comma-separated non-negative lambda grid.")
    parser.add_argument("--validation-users", type=int, default=1_000)
    parser.add_argument("--activity-users-per-segment", type=int, default=100)
    parser.add_argument("--popularity-users-per-bucket", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Catalog JSON must contain a non-empty list")
    catalog = [dict(item) for item in payload if isinstance(item, dict)]
    if len(catalog) != len({int(item["id"]) for item in catalog}):
        raise ValueError("Catalog IDs must be unique")
    return catalog


def parse_artifacts(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODEL=PATH, received {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name.startswith("lightfm_") or not raw_path.strip() or name in result:
            raise ValueError(f"Invalid or duplicate artifact assignment: {value!r}")
        result[name] = Path(raw_path)
    return result


def parse_lambdas(value: str) -> tuple[float, ...]:
    lambdas = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not lambdas or len(lambdas) != len(set(lambdas)) or any(item < 0.0 for item in lambdas):
        raise ValueError("Penalty lambdas must be unique non-negative numbers")
    return tuple(sorted(lambdas))


def _summary_row(name: str, artifact: Path, result: dict[str, Any]) -> dict[str, Any]:
    primary = result["primary"]
    activity = result.get("activity_balanced") or {}
    popularity = result.get("popularity_stratified") or {}
    concentration = primary["popularity_concentration"]
    row: dict[str, Any] = {
        "model": name,
        "artifact": str(artifact),
        "popularity_penalty_lambda": result["popularity_penalty_lambda"],
        **{
            metric: primary[metric]
            for metric in (
                "ndcg_at_10",
                "recall_at_10",
                "hit_rate_at_10",
                "catalog_coverage",
                "novelty_bits",
                "popularity_bias",
                "intra_list_diversity",
                "recommended_normalized_popularity",
                "profile_normalized_popularity",
                "inference_latency_p50_ms",
                "inference_latency_p95_ms",
            )
        },
        **concentration,
    }
    for segment in ("sparse", "medium", "heavy"):
        values = (activity.get("user_segments") or {}).get(segment, {})
        row[f"{segment}_users"] = values.get("users", 0)
        for metric in ("ndcg_at_10", "recall_at_10", "hit_rate_at_10"):
            row[f"{segment}_{metric}"] = values.get(metric, 0.0)
    for bucket in ("head", "mid_tail", "long_tail"):
        values = (popularity.get("heldout_item_popularity") or {}).get(bucket, {})
        row[f"{bucket}_users"] = values.get("users", 0)
        row[f"{bucket}_recall_at_10"] = values.get("recall_at_10", 0.0)
        row[f"{bucket}_ndcg_at_10"] = values.get("ndcg_at_10", 0.0)
    return row


def main() -> None:
    args = parse_args()
    artifacts = parse_artifacts(args.artifact)
    lambdas = parse_lambdas(args.penalty_lambdas)
    catalog = load_catalog(args.catalog)
    store = SplitStore(args.split)
    statistics = compute_train_statistics(store, [int(item["id"]) for item in catalog])
    train_counts = statistics.positive_counts_by_id()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name, artifact_path in artifacts.items():
        index = LightFMServingIndex.load(artifact_path, catalog)
        for penalty_lambda in lambdas:
            result = evaluate_lightfm_artifact_on_validation(
                index,
                store,
                catalog,
                train_counts,
                validation_users=args.validation_users,
                activity_users_per_segment=args.activity_users_per_segment,
                popularity_users_per_bucket=args.popularity_users_per_bucket,
                seed=args.seed,
                popularity_penalty_lambda=penalty_lambda,
            )
            candidate_name = name if penalty_lambda == 0.0 else f"{name}_penalty_{penalty_lambda:g}"
            evaluations.append(
                {
                    "model": candidate_name,
                    "artifact": str(artifact_path),
                    "artifact_variant": index.metadata.get("variant"),
                    "selected_config": index.metadata.get("selected_config"),
                    "feature_summary": index.metadata.get("feature_summary"),
                    "training_duration_seconds": index.metadata.get("selected_training_duration_seconds"),
                    "total_search_duration_seconds": index.metadata.get("total_search_duration_seconds"),
                    "metrics": result,
                }
            )
            rows.append(_summary_row(candidate_name, artifact_path, result))
            print(
                f"{candidate_name}: validation NDCG@10={result['primary']['ndcg_at_10']:.4f}, "
                f"coverage={result['primary']['catalog_coverage']:.4f}, "
                f"sparse NDCG@10={(result['activity_balanced'] or {})['user_segments']['sparse']['ndcg_at_10']:.4f}"
            )

    payload = {
        "schema_version": 1,
        "selection_data": "validation positives only; test positives were not accessed",
        "split": str(args.split),
        "catalog": str(args.catalog),
        "seed": args.seed,
        "validation_users": args.validation_users,
        "activity_users_per_segment": args.activity_users_per_segment,
        "popularity_users_per_bucket": args.popularity_users_per_bucket,
        "evaluations": evaluations,
    }
    (args.output_dir / "validation_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    with (args.output_dir / "validation_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validation artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
