"""Pack the reranker's serving artifact from the frozen training inputs.

The reranker needs five item statistics and an item-item neighbour matrix. In
evaluation those live across four separate split artifacts totalling 37 MB, most
of which is a CountSketch embedding matrix the reranker never touches. This
writes only the arrays the features actually read.

The statistics are copied, not recomputed. They are the train-only numbers the
model was fitted against; recomputing them over all production interactions
would make them fresher and, without refitting, make the model wrong, because
its trees split on thresholds learned from this distribution. For a frozen
model, consistency beats recency.

    python scripts/build_reranker_artifact.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.als_serving import sha256_file  # noqa: E402
from backend.anime_agent.reranker_serving import RERANKER_ARTIFACT_VERSION  # noqa: E402

SPLIT_ARTIFACTS = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "artifacts" / "holdout_seed42_pos8"
RESULTS = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "results" / "reranker"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "reranker_features.npz"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "processed" / "reranker_lambdamart.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with (
        np.load(SPLIT_ARTIFACTS / "countsketch_train_only.npz", allow_pickle=False) as quality,
        np.load(SPLIT_ARTIFACTS / "popularity_train_only.npz", allow_pickle=False) as popularity,
        np.load(SPLIT_ARTIFACTS / "item_item_train_only.npz", allow_pickle=False) as item_item,
    ):
        anime_ids = np.asarray(quality["anime_ids"], dtype=np.int64)
        if not np.array_equal(anime_ids, np.asarray(popularity["anime_ids"], dtype=np.int64)):
            raise SystemExit("popularity artifact item set does not match the quality artifact")
        if not np.array_equal(anime_ids, np.asarray(item_item["anime_ids"], dtype=np.int64)):
            raise SystemExit("item-item artifact item set does not match the quality artifact")
        source_meta = json.loads(str(quality["metadata_json"].item()))

        metadata = {
            "artifact_version": RERANKER_ARTIFACT_VERSION,
            "global_rating_mean": float(source_meta["global_rating_mean"]),
            "training_source": "personalized split train-only statistics (holdout_seed42_pos8)",
            "split_sha256": source_meta.get("split_sha256"),
            "items": int(len(anime_ids)),
            "neighbors_per_item": int(item_item["neighbor_indices"].shape[1]),
        }
        np.savez_compressed(
            args.output,
            anime_ids=anime_ids,
            positive_count=np.asarray(popularity["positive_count"], dtype=np.int32),
            rating_count=np.asarray(quality["rating_count"], dtype=np.int32),
            rating_mean=np.asarray(quality["rating_mean"], dtype=np.float32),
            bayesian_score=np.asarray(quality["bayesian_score"], dtype=np.float32),
            neighbor_indices=np.asarray(item_item["neighbor_indices"], dtype=np.int32),
            neighbor_scores=np.asarray(item_item["neighbor_scores"], dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata)),
        )

    # The booster is copied byte-for-byte; it is already the frozen model.
    args.model_output.write_bytes((RESULTS / "lambdamart.txt").read_bytes())

    feature_size = args.output.stat().st_size
    model_size = args.model_output.stat().st_size
    print(f"features  {args.output}  {feature_size / 1e6:6.2f} MB  sha256 {sha256_file(args.output)}")
    print(f"model     {args.model_output}  {model_size / 1e6:6.2f} MB  sha256 {sha256_file(args.model_output)}")
    print(f"items {metadata['items']:,}  neighbours/item {metadata['neighbors_per_item']}")
    print(f"built in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
