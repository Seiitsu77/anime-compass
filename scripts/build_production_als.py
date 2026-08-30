"""Export a production ALS artifact from the frozen configuration.

The evaluation artifact is trained on the *train-only* split, which is correct
for benchmarking and wrong for production: it deliberately withholds each user's
held-out positives. Production should train on all available positive ratings
using the identical frozen hyperparameters.

This script does not tune anything. Hyperparameters are pinned to the frozen
reference and the script refuses to run with different ones, so a production
build can never silently diverge from the configuration the evidence describes.

    python scripts/build_production_als.py --ratings archive/rating_complete.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_RATINGS = PROJECT_ROOT / "archive" / "rating_complete.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "als_item_factors.npz"

# Frozen reference; see data/evaluation/personalized/FROZEN_als_reference.md
FROZEN_FACTORS = 128
FROZEN_ITERATIONS = 15
FROZEN_REGULARIZATION = 0.05
FROZEN_ALPHA = 5.0
FROZEN_CG_STEPS = 3
FROZEN_SEED = 42
FROZEN_CONFIDENCE = "binary"
FROZEN = {
    "factors": FROZEN_FACTORS,
    "iterations": FROZEN_ITERATIONS,
    "regularization": FROZEN_REGULARIZATION,
    "alpha": FROZEN_ALPHA,
    "cg_steps": FROZEN_CG_STEPS,
    "seed": FROZEN_SEED,
    "confidence_mapping": FROZEN_CONFIDENCE,
}
POSITIVE_THRESHOLD = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--from-split",
        type=Path,
        help="Build from an existing split store instead of raw ratings (benchmark parity).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Imported here so the module can be inspected without SciPy installed.
    from backend.anime_agent.evaluation.collaborative_baselines import build_als_artifact_from_split
    from backend.anime_agent.evaluation.split import SplitStore, sha256_file

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))

    if args.from_split is None:
        raise SystemExit(
            "Building directly from raw ratings is not implemented yet.\n"
            "Pass --from-split to build from an existing split store. A production build over\n"
            "the full rating file (no holdout) is the next step and must reuse the frozen\n"
            "hyperparameters recorded in FROZEN_als_reference.md."
        )

    store = SplitStore(args.from_split)
    print(f"source split : {args.from_split}")
    print(f"split sha256 : {sha256_file(args.from_split)}")
    print(f"frozen config: {FROZEN}")

    started = time.perf_counter()
    metadata = build_als_artifact_from_split(
        store,
        catalog,
        args.output,
        factors=FROZEN_FACTORS,
        iterations=FROZEN_ITERATIONS,
        regularization=FROZEN_REGULARIZATION,
        alpha=FROZEN_ALPHA,
        cg_steps=FROZEN_CG_STEPS,
        seed=FROZEN_SEED,
        confidence_mapping=FROZEN_CONFIDENCE,
    )
    duration = time.perf_counter() - started

    digest = sha256_file(args.output)
    print(f"\nbuilt in {duration:.0f}s -> {args.output}")
    print(f"  users        : {metadata['users_seen']:,}")
    print(f"  positive edges: {metadata['ratings_used']:,}")
    print(f"  artifact sha256: {digest}")
    print("\nPin this hash with ALS_EXPECTED_SHA256 so startup refuses an unverified artifact.")


if __name__ == "__main__":
    main()
