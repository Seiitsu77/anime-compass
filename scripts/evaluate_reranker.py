"""Does a learned second-stage ranker earn its place after ALS retrieval?

ALS retrieves well (Recall@300 = 0.7932), so this asks only whether richer
features order the candidate set better than the ALS score already does.
Retrieval is frozen: the same train-only ALS artifact generates candidates for
every arm, and nothing here retunes it.

Three disjoint user populations, none of which overlaps the 3,924 users any
earlier experiment has already scored:

    reranker training  ->  fits the model
    validation         ->  selects the model
    confirmation       ->  reported once, never tuned against

Run:

    python scripts/evaluate_reranker.py --train-users 2000 --val-users 500 --confirm-users 800
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.anime_agent.evaluation.reranking import (  # noqa: E402
    FEATURE_NAMES,
    LinearReranker,
    RerankerFeatureSpace,
)
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402

SPLIT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "splits" / "holdout_seed42_pos8.sqlite"
ARTIFACTS = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "artifacts" / "holdout_seed42_pos8"
CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
INSPECTED = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "metadata" / "inspected_user_ids.json"
OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "results" / "reranker"


def stable_pick(user_ids: list[int], count: int, salt: str) -> list[int]:
    """Deterministic hash-ordered selection, so a rerun picks the same users."""
    keyed = sorted(user_ids, key=lambda uid: hashlib.blake2b(f"{salt}:{uid}".encode(), digest_size=8).digest())
    return sorted(keyed[:count])


def load_als(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        anime_ids = np.asarray(payload["anime_ids"], dtype=np.int64)
        factors = np.asarray(payload["item_factors"], dtype=np.float32)
        metadata = json.loads(str(payload["metadata_json"].item()))
    return anime_ids, factors, metadata


class FrozenALS:
    """Request-time fold-in against the frozen train-only artifact."""

    def __init__(self, factors: np.ndarray, alpha: float, regularization: float):
        self.factors = factors
        self.alpha = alpha
        self.regularization = regularization
        self.gram = factors.T @ factors

    def scores(self, rows: list[int]) -> np.ndarray:
        if not rows:
            return np.zeros(self.factors.shape[0], dtype=np.float32)
        liked = self.factors[rows]
        dimensions = self.factors.shape[1]
        matrix = self.gram + self.alpha * (liked.T @ liked) + self.regularization * np.eye(dimensions, dtype=np.float32)
        target = self.alpha * liked.sum(axis=0)
        user = np.linalg.solve(matrix.astype(np.float64), target.astype(np.float64)).astype(np.float32)
        return self.factors @ user


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    gain = sum(1.0 / np.log2(i + 2) for i, item in enumerate(ranked[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return float(gain / ideal) if ideal > 0 else 0.0


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def paired_bootstrap(base: list[float], challenger: list[float], *, iterations: int = 2000, seed: int = 20260902):
    """Paired user-level bootstrap over the per-user metric difference."""
    diffs = np.asarray(challenger, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    generator = np.random.default_rng(seed)
    n = len(diffs)
    samples = np.array([diffs[generator.integers(0, n, n)].mean() for _ in range(iterations)])
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(diffs.mean()), float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-users", type=int, default=2000)
    parser.add_argument("--val-users", type=int, default=500)
    parser.add_argument("--confirm-users", type=int, default=800)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    anime_ids, factors, als_meta = load_als(ARTIFACTS / "als_train_only.npz")
    als = FrozenALS(factors, float(als_meta.get("alpha", 5.0)), float(als_meta.get("regularization", 0.05)))
    space = RerankerFeatureSpace.from_artifacts(
        catalog,
        anime_ids,
        popularity_path=ARTIFACTS / "popularity_train_only.npz",
        quality_path=ARTIFACTS / "countsketch_train_only.npz",
        item_item_path=ARTIFACTS / "item_item_train_only.npz",
    )
    row_by_id = space.index_by_id
    print(
        f"loaded catalog + frozen ALS ({factors.shape[0]} items, {factors.shape[1]} factors) "
        f"in {time.perf_counter() - started:.1f}s"
    )

    store = SplitStore(SPLIT)
    inspected = set(json.loads(INSPECTED.read_text(encoding="utf-8")))
    eligible = [uid for uid, count in store.eligible_user_activity() if count >= 5]
    pool = [uid for uid in eligible if uid not in inspected]
    print(f"eligible users {len(eligible):,}; already inspected {len(inspected):,}; fresh pool {len(pool):,}")

    train_ids = stable_pick(pool, args.train_users, "reranker-train-v1")
    rest = [uid for uid in pool if uid not in set(train_ids)]
    val_ids = stable_pick(rest, args.val_users, "reranker-val-v1")
    rest2 = [uid for uid in rest if uid not in set(val_ids)]
    confirm_ids = stable_pick(rest2, args.confirm_users, "reranker-confirm-v1")

    assert not (set(train_ids) & set(val_ids)), "train/val overlap"
    assert not (set(train_ids) & set(confirm_ids)), "train/confirm overlap"
    assert not (set(val_ids) & set(confirm_ids)), "val/confirm overlap"
    assert not (set(train_ids) | set(val_ids) | set(confirm_ids)) & inspected, "overlap with inspected users"
    print(f"disjoint populations: train={len(train_ids)} val={len(val_ids)} confirm={len(confirm_ids)}")

    def candidates_for(user_id: int) -> tuple[list[int], np.ndarray, list[int], set[int]] | None:
        user = store.get_user(user_id)
        if user is None:
            return None
        profile = [row_by_id[a] for a in user.train_positive_ids if a in row_by_id]
        relevant = {row_by_id[a] for a, _r in user.test_positive if a in row_by_id}
        if not profile or not relevant:
            return None
        scores = als.scores(profile)
        scores[profile] = -np.inf  # known items are never recommended
        take = min(args.candidates, len(scores))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top], kind="stable")]
        return profile, scores[top].astype(np.float32), [int(r) for r in top], relevant

    def collect(user_ids: list[int], label: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        features_all: list[np.ndarray] = []
        labels_all: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        tick = time.perf_counter()
        for position, user_id in enumerate(user_ids):
            built = candidates_for(user_id)
            if built is None:
                continue
            profile, scores, rows, relevant = built
            features = space.build(profile, rows, scores)
            labels = np.asarray([1.0 if row in relevant else 0.0 for row in rows], dtype=np.float32)
            features_all.append(features)
            labels_all.append(labels)
            records.append({"user_id": user_id, "rows": rows, "features": features, "relevant": relevant})
            if (position + 1) % 250 == 0:
                print(f"  {label}: {position + 1}/{len(user_ids)} users ({time.perf_counter() - tick:.0f}s)")
        return np.vstack(features_all), np.concatenate(labels_all), records

    print("\nbuilding candidate features...")
    train_features, train_labels, train_records = collect(train_ids, "train")
    print(f"train rows {train_features.shape[0]:,}  positives {int(train_labels.sum()):,} ({train_labels.mean():.3%})")
    _, _, val_records = collect(val_ids, "val")
    _, _, confirm_records = collect(confirm_ids, "confirm")

    print("\nfitting the linear reranker...")
    fit_started = time.perf_counter()
    model = LinearReranker.fit(train_features, train_labels)
    print(f"fitted in {time.perf_counter() - fit_started:.1f}s")
    ranked_weights = sorted(zip(FEATURE_NAMES, model.weights, strict=True), key=lambda p: -abs(p[1]))
    for name, weight in ranked_weights:
        print(f"    {name:20s} {weight:+.4f}")

    # ---------------------------------------------------------------- arms

    # A ranking-specific challenger. LightGBM is an offline training dependency
    # only; nothing in the serving path imports it.
    lambdamart = None
    val_features = np.vstack([record["features"] for record in val_records])
    val_labels = np.concatenate(
        [
            np.asarray([1.0 if row in record["relevant"] else 0.0 for row in record["rows"]], dtype=np.float32)
            for record in val_records
        ]
    )
    try:
        import lightgbm as lgb

        print("\nfitting the LambdaMART challenger...")
        fit_started = time.perf_counter()
        dataset = lgb.Dataset(
            train_features,
            label=train_labels,
            group=[len(record["rows"]) for record in train_records],
            feature_name=list(FEATURE_NAMES),
        )
        valid = lgb.Dataset(
            val_features,
            label=val_labels,
            group=[len(record["rows"]) for record in val_records],
            reference=dataset,
        )
        lambdamart = lgb.train(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [10],
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "lambdarank_truncation_level": 30,
                "verbosity": -1,
                "seed": 20260902,
            },
            dataset,
            num_boost_round=400,
            valid_sets=[valid],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        print(f"fitted in {time.perf_counter() - fit_started:.1f}s, {lambdamart.num_trees()} trees")
    except ImportError:
        print("\nlightgbm is not installed; skipping the LambdaMART arm")

    # An ablation that answers where the gain comes from: the same linear model
    # with the two item-item features zeroed out. Retrieval already declined
    # item-item as a *source*; this asks whether it earns a place as a *feature*.
    item_item_columns = [FEATURE_NAMES.index("item_item_max"), FEATURE_NAMES.index("item_item_sum5")]
    ablated_features = train_features.copy()
    ablated_features[:, item_item_columns] = 0.0
    ablated = LinearReranker.fit(ablated_features, train_labels)

    def linear_scorer(record: dict[str, Any]) -> np.ndarray:
        return model.score(record["features"])

    def ablated_scorer(record: dict[str, Any]) -> np.ndarray:
        features = record["features"].copy()
        features[:, item_item_columns] = 0.0
        return ablated.score(features)

    def lambdamart_scorer(record: dict[str, Any]) -> np.ndarray:
        assert lambdamart is not None
        return np.asarray(lambdamart.predict(record["features"]), dtype=np.float32)

    ARMS: dict[str, Any] = {"als": None, "linear": linear_scorer, "linear_no_item_item": ablated_scorer}
    if lambdamart is not None:
        ARMS["lambdamart"] = lambdamart_scorer

    def order_rows(record: dict[str, Any], scorer: Any) -> list[int]:
        rows = record["rows"]
        if scorer is None:
            return rows
        return [rows[i] for i in np.argsort(-scorer(record), kind="stable")]

    def evaluate(records: list[dict[str, Any]], scorer: Any) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {k: [] for k in ("ndcg10", "recall10", "hit10", "ndcg20", "recall20")}
        for record in records:
            rows = order_rows(record, scorer)
            relevant = record["relevant"]
            out["ndcg10"].append(ndcg_at_k(rows, relevant, 10))
            out["recall10"].append(recall_at_k(rows, relevant, 10))
            out["hit10"].append(hit_at_k(rows, relevant, 10))
            out["ndcg20"].append(ndcg_at_k(rows, relevant, 20))
            out["recall20"].append(recall_at_k(rows, relevant, 20))
        return out

    print("\n=== validation (model selection) ===")
    val_scores = {name: evaluate(val_records, scorer) for name, scorer in ARMS.items()}
    for name, values in val_scores.items():
        delta, low, high = paired_bootstrap(val_scores["als"]["ndcg10"], values["ndcg10"])
        print(f"  {name:20s} NDCG@10 {np.mean(values['ndcg10']):.4f}  delta {delta:+.4f} CI [{low:+.4f}, {high:+.4f}]")
    selected = max(
        (name for name in ARMS if name != "als"),
        key=lambda name: float(np.mean(val_scores[name]["ndcg10"])),
    )
    print(f"  selected on validation: {selected}")

    print("\n=== confirmation (reported once) ===")
    confirm_scores = {name: evaluate(confirm_records, scorer) for name, scorer in ARMS.items()}
    summary: dict[str, Any] = {"metrics": {}, "selected_arm": selected}
    for name, values in confirm_scores.items():
        summary["metrics"][name] = {}
        print(f"  -- {name}")
        for metric, series in values.items():
            mean = float(np.mean(series))
            delta, low, high = paired_bootstrap(confirm_scores["als"][metric], series)
            significant = bool(low > 0 or high < 0)
            summary["metrics"][name][metric] = {
                "mean": mean,
                "delta_vs_als": delta,
                "ci_low": low,
                "ci_high": high,
                "significant": significant,
            }
            flag = "" if name == "als" else ("  significant" if significant else "  includes zero")
            print(f"     {metric:9s} {mean:.4f}   delta {delta:+.4f} CI [{low:+.4f}, {high:+.4f}]{flag}")

    # Popularity concentration, measured on what each arm actually shows.
    pop = space.log_train_pop
    pop_order = np.argsort(-pop)
    head = set(pop_order[: int(0.2 * len(pop_order))].tolist())
    mid = set(pop_order[int(0.2 * len(pop_order)) : int(0.5 * len(pop_order))].tolist())

    def exposure(records: list[dict[str, Any]], scorer: Any) -> dict[str, float]:
        shown: list[int] = []
        for record in records:
            shown.extend(order_rows(record, scorer)[:10])
        return {
            "coverage": len(set(shown)) / len(pop),
            "distinct_items": float(len(set(shown))),
            "head_share": sum(1 for r in shown if r in head) / max(len(shown), 1),
            "mid_share": sum(1 for r in shown if r in mid) / max(len(shown), 1),
            "tail_share": sum(1 for r in shown if r not in head and r not in mid) / max(len(shown), 1),
            "mean_log_pop": float(np.mean([pop[r] for r in shown])),
        }

    summary["exposure"] = {name: exposure(confirm_records, scorer) for name, scorer in ARMS.items()}
    print("\n=== popularity / tail exposure at rank 10 ===")
    for name, values in summary["exposure"].items():
        print(
            f"  {name:20s} distinct {values['distinct_items']:6.0f}  coverage {values['coverage']:.4f}  "
            f"head {values['head_share']:.3f}  mid {values['mid_share']:.3f}  "
            f"tail {values['tail_share']:.3f}  mean_log_pop {values['mean_log_pop']:.3f}"
        )

    # Serving cost of the reranking stage alone, on top of retrieval.
    summary["latency_ms"] = {}
    for name, scorer in ARMS.items():
        if scorer is None:
            continue
        timings = []
        for record in confirm_records[:200]:
            tick = time.perf_counter()
            order_rows(record, scorer)
            timings.append((time.perf_counter() - tick) * 1000)
        timings.sort()
        summary["latency_ms"][name] = {"p50": timings[len(timings) // 2], "p95": timings[int(0.95 * len(timings))]}
        print(
            f"\n{name} rerank latency p50 {summary['latency_ms'][name]['p50']:.3f} ms  "
            f"p95 {summary['latency_ms'][name]['p95']:.3f} ms"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    summary["model"] = model.as_dict()
    if lambdamart is not None:
        model_path = args.output / "lambdamart.txt"
        lambdamart.save_model(str(model_path))
        summary["lambdamart"] = {
            "trees": int(lambdamart.num_trees()),
            "artifact_bytes": model_path.stat().st_size,
            "best_iteration": int(lambdamart.best_iteration or 0),
        }
    summary["linear_artifact_bytes"] = len(json.dumps(model.as_dict()).encode())
    summary["populations"] = {
        "train_user_count": len(train_ids),
        "val_user_count": len(val_ids),
        "confirm_user_count": len(confirm_ids),
        "inspected_excluded": len(inspected),
        "train_users_sha256": hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(),
        "val_users_sha256": hashlib.sha256(json.dumps(val_ids).encode()).hexdigest(),
        "confirm_users_sha256": hashlib.sha256(json.dumps(confirm_ids).encode()).hexdigest(),
    }
    summary["config"] = {
        "candidates": args.candidates,
        "als_factors": als_meta.get("factors"),
        "als_alpha": als_meta.get("alpha"),
        "split": SPLIT.name,
        "feature_names": list(FEATURE_NAMES),
    }
    summary["validation"] = {name: {"ndcg10": float(np.mean(values["ndcg10"]))} for name, values in val_scores.items()}
    (args.output / "reranker_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with gzip.open(args.output / "confirm_user_ids.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"train": train_ids, "val": val_ids, "confirm": confirm_ids}, handle)
    print(f"\nwrote {args.output / 'reranker_results.json'}")
    print(f"total {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
