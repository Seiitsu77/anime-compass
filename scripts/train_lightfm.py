from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.evaluation.lightfm_training import (  # noqa: E402
    LightFMCandidateConfig,
    LightFMSearchConfig,
    default_search_candidates,
    train_lightfm_challengers,
)
from backend.anime_agent.evaluation.models import compute_train_statistics  # noqa: E402
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402

DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "splits" / "holdout_seed42_pos8.sqlite"
DEFAULT_ARTIFACTS = (
    PROJECT_ROOT / "data" / "evaluation" / "personalized" / "artifacts" / "holdout_seed42_pos8" / "lightfm"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune LightFM-ID and LightFM-Hybrid on validation positives and export NumPy serving artifacts."
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--search-profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument(
        "--search-config",
        type=Path,
        help="Optional JSON list of candidate dictionaries; replaces the named search profile.",
    )
    parser.add_argument("--validation-users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Use 1 for reproducibility. Linux OpenMP builds can use more threads for exploratory runs.",
    )
    parser.add_argument("--studio-min-frequency", type=int, default=5)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("lightfm_id", "lightfm_hybrid"),
        default=("lightfm_id", "lightfm_hybrid"),
        help="Train both challengers by default; select one for a fixed sensitivity rerun.",
    )
    parser.add_argument(
        "--allow-single-loss",
        action="store_true",
        help="Allow a one-candidate fixed rerun. Do not use this for primary WARP/BPR selection.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Catalog JSON must contain a non-empty list")
    catalog = [dict(item) for item in payload if isinstance(item, dict)]
    ids = [int(item["id"]) for item in catalog]
    if len(ids) != len(set(ids)):
        raise ValueError("Catalog IDs must be unique")
    return catalog


def load_candidates(path: Path | None, profile: str) -> tuple[LightFMCandidateConfig, ...]:
    if path is None:
        return default_search_candidates(profile)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Search config must be a JSON list")
    return tuple(LightFMCandidateConfig(**dict(value)) for value in payload)


def main() -> None:
    args = parse_args()
    progress = None if args.quiet else lambda message: print(message, flush=True)
    store = SplitStore(args.split)
    catalog = load_catalog(args.catalog)
    split_metadata = store.metadata()
    if progress is not None:
        threshold = split_metadata.get("split_config", {}).get("feedback", {}).get("positive_threshold")
        progress(
            f"split: {split_metadata.get('users_after_filter', 0):,} eligible users, "
            f"positive threshold {threshold}, seed {split_metadata.get('split_config', {}).get('seed')}"
        )
        progress("statistics: aggregating train-only item counts")
    statistics = compute_train_statistics(store, [int(item["id"]) for item in catalog])
    search = LightFMSearchConfig(
        candidates=load_candidates(args.search_config, args.search_profile),
        validation_users=args.validation_users,
        seed=args.seed,
        num_threads=args.num_threads,
        studio_min_frequency=args.studio_min_frequency,
        require_both_losses=not args.allow_single_loss,
    )
    result = train_lightfm_challengers(
        store,
        catalog,
        artifacts_dir=args.artifacts_dir,
        search=search,
        train_positive_counts=statistics.positive_counts_by_id(),
        variants=args.variants,
        progress=progress,
    )
    print("")
    for variant in result["variants"]:
        selected = variant["metadata"]["selected_config"]
        metrics = variant["metadata"]["selected_validation_metrics"]
        print(
            f"{variant['variant']}: {selected['loss']} {selected['no_components']}d; "
            f"validation NDCG@10={metrics['ndcg_at_10']:.4f}, Recall@10={metrics['recall_at_10']:.4f}"
        )
        print(f"  artifact: {variant['artifact_path']}")
    print(f"Training manifest: {args.artifacts_dir / 'lightfm_training.json'}")


if __name__ == "__main__":
    main()
