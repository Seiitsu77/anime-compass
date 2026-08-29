from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


def _read_user_ids(path: Path | None) -> tuple[int, ...]:
    """Load a newline-delimited user ID list, ignoring blanks and comments."""
    if path is None:
        return ()
    values: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            values.append(int(stripped))
    return tuple(sorted(set(values)))


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
        help=("Comma-separated built-ins or LightFM aliases matching lightfm_[a-z0-9_]+."),
    )
    parser.add_argument(
        "--lightfm-artifacts-dir",
        type=Path,
        help="Directory containing lightfm_id.npz and lightfm_hybrid.npz (default: <artifacts-dir>/lightfm).",
    )
    parser.add_argument(
        "--lightfm-artifact",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override an artifact path for a LightFM model alias; repeat as needed.",
    )
    parser.add_argument(
        "--lightfm-penalty",
        action="append",
        default=[],
        metavar="MODEL=LAMBDA",
        help="Apply a train-only normalized-log popularity penalty to a LightFM alias.",
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
    parser.add_argument(
        "--item-item-neighbors",
        type=int,
        default=200,
        help="Neighbours retained per item by the exact item-item baseline.",
    )
    parser.add_argument("--als-factors", type=int, default=64)
    parser.add_argument("--als-iterations", type=int, default=15)
    parser.add_argument("--als-regularization", type=float, default=0.05)
    parser.add_argument("--als-alpha", type=float, default=40.0)
    parser.add_argument(
        "--exclude-user-ids",
        type=Path,
        help="Newline-delimited user IDs to remove from the eligible pool before sampling.",
    )
    parser.add_argument(
        "--semantic-artifact",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "semantic_embeddings.npz",
        help="Optional semantic embedding artifact for the hybrid models.",
    )
    parser.add_argument(
        "--fusion-weights",
        type=Path,
        help="Learned channel weights JSON, required by the current_hybrid_learned model.",
    )
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


def _parse_assignments(values: list[str], *, numeric: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODEL=VALUE, received {value!r}")
        name, assigned = value.split("=", 1)
        name = name.strip()
        if not name.startswith("lightfm_") or not assigned.strip():
            raise ValueError(f"Invalid LightFM assignment: {value!r}")
        if name in result:
            raise ValueError(f"Duplicate LightFM assignment for {name}")
        result[name] = float(assigned) if numeric else Path(assigned)
    return result


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
        # An existing split that does not match the requested configuration is
        # almost always a caller mistake -- passing --split without the matching
        # --positive-threshold, for example. Silently rebuilding destroys the
        # artifact and orphans every model trained against it, so refuse unless
        # the caller explicitly asked to rebuild.
        if split_path.exists() and not args.force_split:
            existing = SplitStore(split_path).metadata()
            existing_feedback = existing.get("split_config", {}).get("feedback", {})
            details = [
                "Refusing to overwrite an existing split that does not match the requested settings.",
                f"  path                : {split_path}",
                f"  existing threshold  : {existing_feedback.get('positive_threshold')}",
                f"  requested threshold : {split_config.feedback.positive_threshold}",
                f"  existing seed       : {existing.get('split_config', {}).get('seed')}",
                f"  requested seed      : {split_config.seed}",
                "Pass matching settings, choose a different --split path, or pass "
                "--force-split to rebuild deliberately.",
            ]
            raise SystemExit("\n".join(details))
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

    model_names = tuple(value.strip() for value in args.models.split(",") if value.strip())
    artifact_overrides = _parse_assignments(args.lightfm_artifact)
    penalty_overrides = _parse_assignments(args.lightfm_penalty, numeric=True)
    run_config = EvaluationRunConfig(
        sample_seed=args.sample_seed,
        sampling_strategy=args.sampling_strategy,
        max_evaluation_users=args.max_evaluation_users,
        users_per_stratum=args.users_per_stratum,
        model_names=model_names,
        recommendation_k=args.recommendation_k,
        bootstrap_iterations=args.bootstrap_iterations,
        countsketch_projections=args.countsketch_projections,
        countsketch_width=args.countsketch_width,
        item_item_neighbors=args.item_item_neighbors,
        fusion_weights_path=str(args.fusion_weights) if args.fusion_weights else None,
        semantic_artifact_path=str(args.semantic_artifact) if args.semantic_artifact else None,
        excluded_user_ids=_read_user_ids(args.exclude_user_ids),
        als_factors=args.als_factors,
        als_iterations=args.als_iterations,
        als_regularization=args.als_regularization,
        als_alpha=args.als_alpha,
        force_model_rebuild=args.force_model_rebuild,
        progress_every=args.progress_every,
        lightfm_penalties=tuple(sorted(penalty_overrides.items())),
    )
    default_lightfm_dir = args.lightfm_artifacts_dir or artifacts_dir / "lightfm"
    lightfm_artifacts = {
        name: Path(artifact_overrides.get(name, default_lightfm_dir / f"{name}.npz"))
        for name in model_names
        if name.startswith("lightfm_")
    }
    result = run_personalized_evaluation(
        store,
        catalog,
        catalog_path=args.catalog,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        config=run_config,
        lightfm_artifacts=lightfm_artifacts,
        progress=progress,
    )
    _print_summary(result)
    print(f"\nReport: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
