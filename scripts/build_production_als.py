"""Build the production ALS artifact from all historically available ratings.

Two ALS artifacts exist and must never be confused:

* the **evaluation** artifact, trained on a leakage-safe split that withholds
  each user's held-out positives. Every published holdout metric comes from it,
  so it stays byte-stable and is never rebuilt from full data.
* the **production** artifact, built here from every positive rating. It is
  strictly better for serving and strictly invalid for measuring, because any
  holdout scored against it would consist of interactions it already trained on.

The artifact is tagged ``artifact_role="production"`` and carries
``not_valid_for_holdout_evaluation``; the serving loader refuses an artifact
whose role does not match what the caller asked for.

Hyperparameters are pinned to the frozen validated configuration and are not
exposed as flags, so a production build cannot silently diverge from the
configuration the evidence describes.

    python scripts/build_production_als.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_RATINGS = PROJECT_ROOT / "archive" / "rating_complete.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "als_production_item_factors.npz"

# Frozen reference; see data/evaluation/personalized/FROZEN_als_reference.md
FROZEN_FACTORS = 128
FROZEN_ITERATIONS = 15
FROZEN_REGULARIZATION = 0.05
FROZEN_ALPHA = 5.0
FROZEN_CG_STEPS = 3
FROZEN_SEED = 42
POSITIVE_THRESHOLD = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--row-limit",
        type=int,
        help="Stop after N rating rows. For smoke tests only; never for a shipped artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Imported here so the module can be inspected without SciPy installed.
    from backend.anime_agent.als_serving import catalog_ids_digest, sha256_file
    from backend.anime_agent.evaluation.collaborative_baselines import build_production_als_artifact

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_ids = sorted(int(item["id"]) for item in catalog)

    print(f"ratings : {args.ratings}")
    print(f"catalog : {args.catalog} ({len(catalog_ids):,} items)")
    print(
        f"frozen  : factors={FROZEN_FACTORS} alpha={FROZEN_ALPHA} reg={FROZEN_REGULARIZATION} "
        f"iters={FROZEN_ITERATIONS} cg={FROZEN_CG_STEPS} seed={FROZEN_SEED} "
        f"threshold>={POSITIVE_THRESHOLD}"
    )
    if args.row_limit:
        print(f"WARNING: --row-limit {args.row_limit:,} set; this artifact is not shippable")

    started = time.perf_counter()
    metadata = build_production_als_artifact(
        args.ratings,
        catalog,
        args.output,
        positive_threshold=POSITIVE_THRESHOLD,
        factors=FROZEN_FACTORS,
        iterations=FROZEN_ITERATIONS,
        regularization=FROZEN_REGULARIZATION,
        alpha=FROZEN_ALPHA,
        cg_steps=FROZEN_CG_STEPS,
        seed=FROZEN_SEED,
        row_limit=args.row_limit,
        progress=lambda message: print(f"  {message}", flush=True),
    )
    duration = time.perf_counter() - started

    digest = sha256_file(args.output)
    size = args.output.stat().st_size
    print(f"\nbuilt in {duration:.0f}s -> {args.output}")
    print(f"  role                 : {metadata['artifact_role']}")
    print(f"  rows scanned         : {metadata['rows_scanned']:,}")
    print(f"  positive interactions: {metadata['ratings_used']:,}")
    print(f"  users                : {metadata['users_seen']:,}")
    print(f"  orphan positives     : {metadata['orphan_positive_rows']:,}")
    print(f"  artifact size        : {size / 1048576:.1f} MB")
    print(f"  artifact sha256      : {digest}")
    print(f"  ratings sha256       : {metadata['ratings_sha256']}")
    print(f"  catalog ids sha256   : {metadata['catalog_ids_sha256']}")
    if metadata["catalog_ids_sha256"] != catalog_ids_digest(catalog_ids):
        raise SystemExit("Catalog digest mismatch between the trainer and the serving hash function")

    print("\nPin these in the environment so startup refuses an unverified artifact:")
    resolved = args.output.resolve()
    try:
        shown = resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        shown = resolved.as_posix()
    print(f"  ALS_ARTIFACT_PATH={shown}")
    print(f"  ALS_EXPECTED_SHA256={digest}")
    print(f"  ALS_EXPECTED_CATALOG_IDS_SHA256={metadata['catalog_ids_sha256']}")


if __name__ == "__main__":
    main()
