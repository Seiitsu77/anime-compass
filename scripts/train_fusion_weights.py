"""Fit the hybrid recommender's channel-blend weights from held-out positives.

Weights are fitted on *validation* users and scored once on *test* users, so the
reported improvement is not the number the optimiser was allowed to chase.

    python scripts/train_fusion_weights.py \
        --split data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite \
        --countsketch data/evaluation/personalized/artifacts/holdout_seed42_pos8/countsketch_train_only.npz \
        --train-users 400 --test-users 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.anime_agent.collaborative import CollaborativeIndex
from backend.anime_agent.evaluation.fusion import (
    CHANNELS,
    baseline_pairwise_accuracy,
    build_pairwise_dataset,
    fit_pairwise_weights,
    save_fusion_artifact,
)
from backend.anime_agent.evaluation.models import (
    compute_train_statistics,
    sanitize_catalog_with_training_statistics,
)
from backend.anime_agent.evaluation.runner import _load_semantic_index_for_evaluation
from backend.anime_agent.evaluation.split import SplitStore, sha256_file
from backend.anime_agent.recommender import (
    DEFAULT_CHANNEL_WEIGHTS,
    AnimeRecommender,
    experimental_semantic_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "fusion" / "learned_weights.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--countsketch", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--semantic-artifact",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "semantic_embeddings.npz",
        help="Optional semantic embedding artifact; without it that channel has no variance to fit.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-users", type=int, default=400, help="Validation users used to fit the weights.")
    parser.add_argument(
        "--test-users",
        type=int,
        default=0,
        help=(
            "Optional local sanity check on disjoint users. Leave at 0 when the real "
            "confirmation is a separate predeclared run, so the fit never sees it."
        ),
    )
    parser.add_argument(
        "--exclude-user-ids",
        type=Path,
        help="Newline-delimited user IDs reserved for confirmation; never used for fitting.",
    )
    parser.add_argument("--shortlist", type=int, default=400)
    parser.add_argument("--negatives-per-positive", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = SplitStore(args.split)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_ids = [int(item["id"]) for item in catalog]

    print("preparing train-only recommender")
    statistics = compute_train_statistics(store, catalog_ids)
    index = CollaborativeIndex.load(args.countsketch, catalog)
    evaluation_catalog = sanitize_catalog_with_training_statistics(catalog, statistics, index)
    semantic_index = (
        _load_semantic_index_for_evaluation(args.semantic_artifact, evaluation_catalog)
        if args.semantic_artifact
        else None
    )
    if semantic_index is None:
        print("warning: no semantic index; the semantic_embedding channel will have no variance to fit")
    recommender = AnimeRecommender(
        evaluation_catalog,
        collaborative_index=index,
        semantic_index=semantic_index,
    )

    reserved: set[int] = set()
    if args.exclude_user_ids:
        for line in args.exclude_user_ids.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                reserved.add(int(stripped))
        print(f"reserving {len(reserved):,} user IDs; they are never used for fitting")

    eligible = [user for user in store.iter_users(eligible_only=True) if user.user_id not in reserved]
    eligible.sort(key=lambda user: user.user_id)
    fit_users = eligible[: args.train_users]
    held_out = eligible[args.train_users : args.train_users + args.test_users] if args.test_users else []
    if not fit_users:
        raise SystemExit("Not enough eligible users to fit the weights")

    print(f"building pairwise rows from {len(fit_users)} validation users")
    train_set = build_pairwise_dataset(
        recommender,
        fit_users,
        holdout="validation",
        shortlist=args.shortlist,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
    )
    print(
        f"  pairs={len(train_set):,} users={train_set.users} covered={train_set.positives_covered} missed={train_set.positives_missed}"
    )

    # Start the retired semantic channel above zero so the fit can judge it on
    # evidence rather than inheriting the decision being re-examined.
    start = experimental_semantic_weights() if semantic_index is not None else dict(DEFAULT_CHANNEL_WEIGHTS)
    result = fit_pairwise_weights(
        train_set,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
        l2=args.l2,
        seed=args.seed,
        initial_weights=start,
    )

    result["reserved_user_count"] = len(reserved)
    result["semantic_index_available"] = semantic_index is not None
    # Per-channel score variance on the fit set. A channel with no variance
    # cannot be fitted, which is exactly how the semantic weight went unexamined.
    variances = train_set.differences.var(axis=0, ddof=1)
    result["channel_difference_variance"] = {
        channel: float(value) for channel, value in zip(CHANNELS, variances.tolist(), strict=True)
    }
    result["zero_variance_channels"] = sorted(
        channel for channel, value in result["channel_difference_variance"].items() if value <= 1e-12
    )

    learned_accuracy = hand_set_accuracy = None
    if held_out:
        print(f"local sanity check on {len(held_out)} disjoint users (not the confirmation set)")
        test_set = build_pairwise_dataset(
            recommender,
            held_out,
            holdout="test",
            shortlist=args.shortlist,
            negatives_per_positive=args.negatives_per_positive,
            seed=args.seed + 1,
        )
        learned_accuracy = baseline_pairwise_accuracy(test_set, result["weights"])
        hand_set_accuracy = baseline_pairwise_accuracy(test_set, DEFAULT_CHANNEL_WEIGHTS)
        result["test_pairs"] = len(test_set)
        result["test_users"] = test_set.users
        result["test_pairwise_accuracy_learned"] = learned_accuracy
        result["test_pairwise_accuracy_hand_set"] = hand_set_accuracy
        result["test_pairwise_accuracy_delta"] = learned_accuracy - hand_set_accuracy

    payload = save_fusion_artifact(result, args.output, split_sha256=sha256_file(store.path))

    print()
    print(f"{'channel':<20}{'hand-set':>12}{'learned':>12}{'delta':>12}")
    for channel, hand_set in DEFAULT_CHANNEL_WEIGHTS.items():
        learned = payload["weights"][channel]
        print(f"{channel:<20}{hand_set:>12.4f}{learned:>12.4f}{learned - hand_set:>+12.4f}")
    print()
    print(f"train pairwise accuracy : {result['pairwise_accuracy']:.4f}")
    print(f"semantic index available: {result['semantic_index_available']}")
    if result["zero_variance_channels"]:
        print(f"zero-variance channels  : {', '.join(result['zero_variance_channels'])}")
    if learned_accuracy is not None and hand_set_accuracy is not None:
        print(
            f"sanity  pairwise accuracy: hand-set {hand_set_accuracy:.4f} -> learned {learned_accuracy:.4f} "
            f"({learned_accuracy - hand_set_accuracy:+.4f})"
        )
    print(f"artifact: {args.output}")


if __name__ == "__main__":
    main()
