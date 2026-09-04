"""Linear or LambdaMART? One decision, on already-frozen artifacts.

Both rerankers already beat ALS. That is settled and is not re-measured here.
The open question is narrower and was never actually tested: LambdaMART scored
0.2828 against Linear's 0.2725, but those came from two separate comparisons
*against ALS*, and a difference of differences is not a paired test. If the two
rerankers agree on most users, the gap can be real; if they disagree, it can be
noise. Only a direct paired comparison answers it.

Nothing is retrained. The linear weights come from `reranker_results.json`, the
trees from `lambdamart.txt`, and the confirmation users from
`confirm_user_ids.json.gz` -- all written by the previous run.

The second half of the script answers the question the previous run dodged: it
timed `model.predict()` alone, which is the cheapest stage of the pipeline and
therefore the least informative. Here every stage is timed, from profile
construction to the final top-N.

    python scripts/compare_rerankers.py
"""

from __future__ import annotations

import argparse
import gzip
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
    LinearReranker,
    RerankerFeatureSpace,
    StandardScaler,
)
from backend.anime_agent.evaluation.split import SplitStore  # noqa: E402
from scripts.evaluate_reranker import (  # noqa: E402
    FrozenALS,
    hit_at_k,
    load_als,
    ndcg_at_k,
    paired_bootstrap,
    recall_at_k,
)

SPLIT = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "splits" / "holdout_seed42_pos8.sqlite"
ARTIFACTS = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "artifacts" / "holdout_seed42_pos8"
CATALOG = PROJECT_ROOT / "data" / "processed" / "anime_catalog.json"
RESULTS = PROJECT_ROOT / "data" / "evaluation" / "personalized" / "results" / "reranker"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--latency-users", type=int, default=300)
    args = parser.parse_args()

    summary: dict[str, Any] = {}

    # ------------------------------------------------------------- load

    load_started = time.perf_counter()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    anime_ids, factors, als_meta = load_als(ARTIFACTS / "als_train_only.npz")
    als = FrozenALS(factors, float(als_meta["alpha"]), float(als_meta["regularization"]))
    space = RerankerFeatureSpace.from_artifacts(
        catalog,
        anime_ids,
        popularity_path=ARTIFACTS / "popularity_train_only.npz",
        quality_path=ARTIFACTS / "countsketch_train_only.npz",
        item_item_path=ARTIFACTS / "item_item_train_only.npz",
    )
    row_by_id = space.index_by_id
    print(f"loaded catalog + frozen ALS in {time.perf_counter() - load_started:.1f}s")

    frozen = json.loads((RESULTS / "reranker_results.json").read_text(encoding="utf-8"))
    payload = frozen["model"]
    linear_load_started = time.perf_counter()
    linear = LinearReranker(
        np.asarray(payload["weights"], dtype=np.float32),
        float(payload["bias"]),
        StandardScaler(
            np.asarray(payload["scaler_mean"], dtype=np.float32),
            np.asarray(payload["scaler_scale"], dtype=np.float32),
        ),
    )
    linear_load_ms = (time.perf_counter() - linear_load_started) * 1000

    import_started = time.perf_counter()
    import lightgbm as lgb

    lightgbm_import_ms = (time.perf_counter() - import_started) * 1000
    booster_started = time.perf_counter()
    booster = lgb.Booster(model_file=str(RESULTS / "lambdamart.txt"))
    booster_load_ms = (time.perf_counter() - booster_started) * 1000
    print(
        f"linear load {linear_load_ms:.2f} ms; lightgbm import {lightgbm_import_ms:.0f} ms; "
        f"booster load {booster_load_ms:.0f} ms"
    )

    with gzip.open(RESULTS / "confirm_user_ids.json.gz", "rt", encoding="utf-8") as handle:
        populations = json.load(handle)
    confirm_ids = populations["confirm"]
    print(f"frozen confirmation users: {len(confirm_ids)}")

    store = SplitStore(SPLIT)

    # ------------------------------------------- rebuild frozen candidates

    records: list[dict[str, Any]] = []
    for user_id in confirm_ids:
        user = store.get_user(user_id)
        if user is None:
            continue
        profile = [row_by_id[a] for a in user.train_positive_ids if a in row_by_id]
        relevant = {row_by_id[a] for a, _r in user.test_positive if a in row_by_id}
        if not profile or not relevant:
            continue
        scores = als.scores(profile)
        scores[profile] = -np.inf
        take = min(args.candidates, len(scores))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top], kind="stable")]
        rows = [int(r) for r in top]
        records.append(
            {
                "profile": profile,
                "rows": rows,
                "features": space.build(profile, rows, scores[top].astype(np.float32)),
                "relevant": relevant,
            }
        )
    print(f"rebuilt candidates for {len(records)} users\n")

    # ------------------------------------------- 1. direct paired comparison

    def order(record: dict[str, Any], scorer: Any) -> list[int]:
        if scorer is None:
            return record["rows"]
        values = scorer(record["features"])
        return [record["rows"][i] for i in np.argsort(-values, kind="stable")]

    arms: dict[str, Any] = {
        "als": None,
        "linear": linear.score,
        "lambdamart": lambda features: np.asarray(booster.predict(features), dtype=np.float32),
    }

    per_user: dict[str, dict[str, list[float]]] = {}
    for name, scorer in arms.items():
        metrics: dict[str, list[float]] = {k: [] for k in ("ndcg10", "recall10", "hit10", "ndcg20", "recall20")}
        for record in records:
            ranked = order(record, scorer)
            relevant = record["relevant"]
            metrics["ndcg10"].append(ndcg_at_k(ranked, relevant, 10))
            metrics["recall10"].append(recall_at_k(ranked, relevant, 10))
            metrics["hit10"].append(hit_at_k(ranked, relevant, 10))
            metrics["ndcg20"].append(ndcg_at_k(ranked, relevant, 20))
            metrics["recall20"].append(recall_at_k(ranked, relevant, 20))
        per_user[name] = metrics

    print("=== 1. LambdaMART vs Linear, paired on the same users ===")
    print("   (this is the comparison that was never run; not a difference of differences)")
    head_to_head: dict[str, Any] = {}
    for metric in ("ndcg10", "recall10", "ndcg20", "recall20"):
        linear_mean = float(np.mean(per_user["linear"][metric]))
        lgbm_mean = float(np.mean(per_user["lambdamart"][metric]))
        delta, low, high = paired_bootstrap(per_user["linear"][metric], per_user["lambdamart"][metric])
        significant = bool(low > 0 or high < 0)
        head_to_head[metric] = {
            "linear": linear_mean,
            "lambdamart": lgbm_mean,
            "delta": delta,
            "ci_low": low,
            "ci_high": high,
            "significant": significant,
        }
        print(
            f"  {metric:9s} linear {linear_mean:.4f}  lambdamart {lgbm_mean:.4f}  "
            f"delta {delta:+.4f}  CI [{low:+.4f}, {high:+.4f}]  "
            f"{'SIGNIFICANT' if significant else 'includes zero'}"
        )

    # How often do the two arms actually differ in the top 10?
    agreement = []
    for record in records:
        a = set(order(record, arms["linear"])[: args.limit])
        b = set(order(record, arms["lambdamart"])[: args.limit])
        agreement.append(len(a & b) / args.limit)
    head_to_head["top10_overlap_mean"] = float(np.mean(agreement))
    print(f"\n  mean top-{args.limit} overlap between the two rerankers: {np.mean(agreement):.3f}")
    summary["head_to_head"] = head_to_head

    # ------------------------------------ 2-3. full end-to-end serving cost

    print("\n=== 2. End-to-end serving latency (complete path, not model.predict) ===")
    sample = confirm_ids[: args.latency_users]
    profiles: list[list[int]] = []
    for user_id in sample:
        user = store.get_user(user_id)
        if user is None:
            continue
        rows = [row_by_id[a] for a in user.train_positive_ids if a in row_by_id]
        if rows:
            profiles.append(rows)

    def run_once(profile: list[int], scorer: Any) -> dict[str, float]:
        stages: dict[str, float] = {}
        tick = time.perf_counter()
        scores = als.scores(profile)
        scores[profile] = -np.inf
        stages["als_retrieval"] = (time.perf_counter() - tick) * 1000

        tick = time.perf_counter()
        take = min(args.candidates, len(scores))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top], kind="stable")]
        rows = [int(r) for r in top]
        stages["candidate_selection"] = (time.perf_counter() - tick) * 1000

        if scorer is None:
            tick = time.perf_counter()
            final = rows[: args.limit]
            stages["final_topn"] = (time.perf_counter() - tick) * 1000
            stages["feature_construction"] = 0.0
            stages["reranker_inference"] = 0.0
            stages["sort"] = 0.0
        else:
            tick = time.perf_counter()
            features = space.build(profile, rows, scores[top].astype(np.float32))
            stages["feature_construction"] = (time.perf_counter() - tick) * 1000

            tick = time.perf_counter()
            values = scorer(features)
            stages["reranker_inference"] = (time.perf_counter() - tick) * 1000

            tick = time.perf_counter()
            reordered = [rows[i] for i in np.argsort(-values, kind="stable")]
            stages["sort"] = (time.perf_counter() - tick) * 1000

            tick = time.perf_counter()
            final = reordered[: args.limit]
            stages["final_topn"] = (time.perf_counter() - tick) * 1000
        assert final is not None
        stages["total"] = sum(
            stages[key]
            for key in (
                "als_retrieval",
                "candidate_selection",
                "feature_construction",
                "reranker_inference",
                "sort",
                "final_topn",
            )
        )
        return stages

    latency: dict[str, Any] = {}
    for name, scorer in arms.items():
        for profile in profiles[:20]:  # warm caches and JIT-free numpy paths
            run_once(profile, scorer)
        runs = [run_once(profile, scorer) for profile in profiles]
        stage_names = list(runs[0])
        latency[name] = {
            stage: {
                "p50": percentile([run[stage] for run in runs], 0.50),
                "p95": percentile([run[stage] for run in runs], 0.95),
            }
            for stage in stage_names
        }
        total = latency[name]["total"]
        print(f"  {name:12s} total p50 {total['p50']:7.3f} ms   p95 {total['p95']:7.3f} ms")
    summary["latency_ms"] = latency

    print("\n=== 3. Stage-level latency (p50 ms) ===")
    stages_of_interest = (
        "als_retrieval",
        "candidate_selection",
        "feature_construction",
        "reranker_inference",
        "sort",
        "final_topn",
        "total",
    )
    header = "  stage                  " + "".join(f"{name:>14s}" for name in arms)
    print(header)
    for stage in stages_of_interest:
        row = f"  {stage:22s}" + "".join(f"{latency[name][stage]['p50']:14.3f}" for name in arms)
        print(row)

    # -------------------------------------------- 4. deployment cost

    print("\n=== 4. Deployment cost ===")
    import importlib.util

    def directory_bytes(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    lightgbm_spec = importlib.util.find_spec("lightgbm")
    lightgbm_dir = Path(lightgbm_spec.origin).parent if lightgbm_spec and lightgbm_spec.origin else None
    lightgbm_bytes = directory_bytes(lightgbm_dir) if lightgbm_dir else 0

    linear_artifact = len(json.dumps(payload).encode())
    lambdamart_artifact = (RESULTS / "lambdamart.txt").stat().st_size

    deployment: dict[str, dict[str, Any]] = {
        "linear": {
            "artifact_bytes": linear_artifact,
            "extra_dependencies": [],
            "extra_dependency_bytes": 0,
            "import_ms": 0.0,
            "model_load_ms": linear_load_ms,
        },
        "lambdamart": {
            "artifact_bytes": lambdamart_artifact,
            "extra_dependencies": ["lightgbm"],
            "extra_dependency_bytes": lightgbm_bytes,
            "import_ms": lightgbm_import_ms,
            "model_load_ms": booster_load_ms,
        },
    }
    for name, values in deployment.items():
        artifact_kb = float(values["artifact_bytes"]) / 1024
        deps_mb = float(values["extra_dependency_bytes"]) / 1e6
        print(
            f"  {name:12s} artifact {artifact_kb:8.1f} KB   extra deps {deps_mb:6.1f} MB   "
            f"import {float(values['import_ms']):6.1f} ms   load {float(values['model_load_ms']):6.2f} ms"
        )
    summary["deployment"] = deployment

    # Resident memory attributable to each option.
    try:
        import resource  # noqa: F401  (POSIX only)
    except ImportError:
        pass
    try:
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        summary["process_rss_bytes"] = int(counters.WorkingSetSize)
        print(f"  process RSS with both models loaded: {counters.WorkingSetSize / 1e6:.0f} MB")
    except (AttributeError, OSError):
        pass

    (RESULTS / "reranker_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'reranker_selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
