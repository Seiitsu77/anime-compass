from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.evaluation.runner import (  # noqa: E402
    EvaluationRunConfig,
    refresh_derived_outputs,
    run_personalized_evaluation,
)
from backend.anime_agent.evaluation.split import (  # noqa: E402
    FeedbackConfig,
    SplitConfig,
    SplitStore,
    build_split_store,
    split_store_matches,
)

DEFAULT_RATINGS = PROJECT_ROOT / "archive" / "rating_complete.csv"
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_EVALUATION_ROOT = PROJECT_ROOT / "data" / "evaluation" / "personalized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe, per-user held-out evaluation for configured popularity, "
            "CountSketch, LightFM, and current-hybrid models."
        )
    )
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--split",
        type=Path,
        help="Persistent SQLite split artifact (default is derived from split settings).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Train-only benchmark model artifacts (default is derived from the split name).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="JSON/CSV/Markdown output directory (default is sampled/ or full/).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-strategy",
        choices=("uniform", "stratified", "activity_stratified", "popularity_stratified"),
        default="uniform",
        help="Uniform is representative; stratified strategies are diagnostics.",
    )
    parser.add_argument(
        "--users-per-stratum",
        type=int,
        help="Diagnostic quota for each activity or popularity stratum; overrides total allocation.",
    )
    parser.add_argument(
        "--models",
        default="popularity,countsketch_cf,current_hybrid",
        help=(
            "Comma-separated models: popularity,countsketch_cf,current_hybrid,lightfm_id,lightfm_hybrid. "
            "Use the four collaborative models for the LightFM comparison."
        ),
    )
    parser.add_argument(
        "--lightfm-artifacts-dir",
        type=Path,
        help="Directory containing lightfm_id.npz and lightfm_hybrid.npz (default: <artifacts-dir>/lightfm).",
    )
    parser.add_argument("--positive-threshold", type=int, default=8)
    parser.add_argument("--neutral-min", type=int, default=6)
    parser.add_argument("--neutral-max", type=int, default=7)
    parser.add_argument("--negative-max", type=int, default=5)
    parser.add_argument("--minimum-positives", type=int, default=5)
    parser.add_argument(
        "--source-user-limit",
        type=int,
        help="Build a fast prefix split for pipeline smoke tests; omit for the full source dataset.",
    )
    parser.add_argument(
        "--max-evaluation-users",
        type=int,
        default=100,
        help="Deterministic user sample size; use 0 for every eligible user.",
    )
    parser.add_argument("--recommendation-k", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--countsketch-projections", type=int, default=3)
    parser.add_argument("--countsketch-width", type=int, default=128)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force-split", action="store_true")
    parser.add_argument("--force-model-rebuild", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument(
        "--refresh-output",
        type=Path,
        help="Regenerate CSV/Markdown views from an existing result directory without rerunning models.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Processed catalog not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Catalog JSON must contain a non-empty list")
    catalog = [dict(item) for item in payload if isinstance(item, dict)]
    ids = [int(item["id"]) for item in catalog]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog IDs must be unique")
    return catalog


def _default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    source_suffix = f"_users{args.source_user_limit}" if args.source_user_limit else ""
    split_name = f"holdout_seed{args.seed}_pos{args.positive_threshold}{source_suffix}"
    split_path = args.split or DEFAULT_EVALUATION_ROOT / "splits" / f"{split_name}.sqlite"
    artifacts_dir = args.artifacts_dir or DEFAULT_EVALUATION_ROOT / "artifacts" / split_name
    if args.users_per_stratum:
        scope = f"{args.sampling_strategy}_{args.users_per_stratum}_per_stratum"
    else:
        scope = "full" if args.max_evaluation_users == 0 else f"{args.sampling_strategy}_{args.max_evaluation_users}"
    output_dir = args.output_dir or DEFAULT_EVALUATION_ROOT / "results" / f"{split_name}_{scope}"
    return split_path, artifacts_dir, output_dir


def _print_summary(result: dict[str, Any]) -> None:
    print("")
    print("Model             NDCG@10  Recall@10  HR@10   p50 ms")
    for model in result["models"]:
        print(
            f"{model['model']:<17} "
            f"{model['ndcg_at_10']:.4f}   {model['recall_at_10']:.4f}     "
            f"{model['hit_rate_at_10']:.4f}  {model['engineering']['inference_latency_p50_ms']:.2f}"
        )


def main() -> None:
    args = parse_args()
    if args.refresh_output is not None:
        refresh_derived_outputs(args.refresh_output)
        print(f"Refreshed derived outputs in {args.refresh_output}")
        return
    feedback = FeedbackConfig(
        positive_threshold=args.positive_threshold,
        neutral_min=args.neutral_min,
        neutral_max=args.neutral_max,
        negative_max=args.negative_max,
    )
    split_config = SplitConfig(
        seed=args.seed,
        minimum_positives=args.minimum_positives,
        feedback=feedback,
    )
    split_path, artifacts_dir, output_dir = _default_paths(args)
    progress = None if args.quiet else lambda message: print(message, flush=True)

    catalog = load_catalog(args.catalog)
    catalog_ids = {int(item["id"]) for item in catalog}
    reusable = not args.force_split and split_store_matches(
        split_path,
        args.ratings,
        split_config,
        catalog_ids=catalog_ids,
        source_user_limit=args.source_user_limit,
    )
    if reusable:
        if progress is not None:
            progress(f"split: reusing {split_path}")
    else:
        if progress is not None:
            progress(f"split: building {split_path}")
        build_split_store(
            args.ratings,
            split_path,
            catalog_ids=catalog_ids,
            config=split_config,
            source_user_limit=args.source_user_limit,
            progress=progress,
        )

    store = SplitStore(split_path)
    if args.split_only:
        metadata = store.metadata()
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    run_config = EvaluationRunConfig(
        sample_seed=args.sample_seed,
        sampling_strategy=args.sampling_strategy,
        max_evaluation_users=args.max_evaluation_users,
        users_per_stratum=args.users_per_stratum,
        model_names=tuple(value.strip() for value in args.models.split(",") if value.strip()),
        recommendation_k=args.recommendation_k,
        bootstrap_iterations=args.bootstrap_iterations,
        countsketch_projections=args.countsketch_projections,
        countsketch_width=args.countsketch_width,
        force_model_rebuild=args.force_model_rebuild,
        progress_every=args.progress_every,
    )
    result = run_personalized_evaluation(
        store,
        catalog,
        catalog_path=args.catalog,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        config=run_config,
        lightfm_artifacts={
            name: (args.lightfm_artifacts_dir or artifacts_dir / "lightfm") / f"{name}.npz"
            for name in ("lightfm_id", "lightfm_hybrid")
        },
        progress=progress,
    )
    _print_summary(result)
    print(f"\nReport: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
