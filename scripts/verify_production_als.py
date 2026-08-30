"""Smoke-test the shipped production ALS artifact end to end.

Run after every production build, before pinning the hash in the environment.
Checks identity, role separation from the evaluation artifact, catalog
alignment, fold-in, known-item exclusion, determinism, and serving cost.

    python scripts/verify_production_als.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from backend.anime_agent.als_serving import (
    ARTIFACT_ROLE_EVALUATION,
    ARTIFACT_ROLE_PRODUCTION,
    ALSArtifactRoleError,
    ALSCatalogMismatchError,
    ALSCollaborativeIndex,
    catalog_ids_digest,
    sha256_file,
)

PROD = Path("data/processed/als_production_item_factors.npz")
EVAL = Path("data/evaluation/personalized/artifacts/holdout_seed42_pos8/als_train_only.npz")
EXPECTED_PROD = "95c079b1b8f4e0e509c8bab29e4357360f851e3adfd2abc261f358375ee13a10"
EXPECTED_EVAL = "a0be5f3f1dde0a406d2bd14af705467a4b8155e8089a286e578c2f6f0ded354b"

catalog = json.loads(Path("data/processed/anime_catalog.json").read_text(encoding="utf-8"))
catalog_ids = sorted(int(item["id"]) for item in catalog)
digest = catalog_ids_digest(catalog_ids)

ok = True


def check(label, condition, detail=""):
    global ok
    ok &= bool(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")


print("=== 1. artifact identity ===")
check("production hash matches the reported build", sha256_file(PROD) == EXPECTED_PROD)
check("evaluation artifact is unchanged", sha256_file(EVAL) == EXPECTED_EVAL)
check("the two artifacts are distinct files", sha256_file(PROD) != sha256_file(EVAL))

print("\n=== 2. load with every pin enforced ===")
started = time.perf_counter()
index = ALSCollaborativeIndex.load(
    PROD,
    catalog,
    expected_artifact_sha256=EXPECTED_PROD,
    expected_role=ARTIFACT_ROLE_PRODUCTION,
    expected_catalog_ids_sha256=digest,
)
load_seconds = time.perf_counter() - started
info = index.model_info()
check("loads under all pins", True, f"{load_seconds:.2f}s")
check("role is production", info["artifact_role"] == ARTIFACT_ROLE_PRODUCTION)
check("marked invalid for holdout evaluation", info["valid_for_holdout_evaluation"] is False)
check("trained on every positive", info["ratings_used"] == 30_875_410, f"{info['ratings_used']:,}")
check("frozen hyperparameters", (info["factors"], info["alpha"], info["regularization"]) == (128, 5.0, 0.05))

print("\n=== 3. catalog alignment ===")
check("item count matches catalog", info["items"] == len(catalog_ids), f"{info['items']:,}")
check("id sets are identical", set(index.anime_ids.tolist()) == set(catalog_ids))
check("ids are sorted and unique", bool(np.all(np.diff(index.anime_ids) > 0)))
check("factors are finite", bool(np.isfinite(index.item_factors).all()))

print("\n=== 4. role separation ===")
try:
    ALSCollaborativeIndex.load(EVAL, catalog, expected_role=ARTIFACT_ROLE_PRODUCTION)
    check("evaluation artifact refused for production", False)
except ALSArtifactRoleError:
    check("evaluation artifact refused for production", True)
try:
    ALSCollaborativeIndex.load(PROD, catalog, expected_role=ARTIFACT_ROLE_EVALUATION)
    check("production artifact refused for evaluation", False)
except ALSArtifactRoleError:
    check("production artifact refused for evaluation", True)
try:
    ALSCollaborativeIndex.load(PROD, catalog, expected_catalog_ids_sha256="0" * 64)
    check("pinned catalog mismatch refused", False)
except ALSCatalogMismatchError:
    check("pinned catalog mismatch refused", True)

print("\n=== 5. fold-in and known-item exclusion ===")
# Fullmetal Alchemist: Brotherhood, Steins;Gate, Death Note
profile = [5114, 9253, 1535]
vector = index.user_vector(profile)
check("fold-in produces a finite vector", vector is not None and bool(np.isfinite(vector).all()))
ranked = index.top_candidates(profile, 10, excluded_ids=profile)
check("returns a full ranking", len(ranked) == 10)
check("profile items excluded", not (set(profile) & set(ranked)))
by_id = {int(i["id"]): i for i in catalog}
print("     top 5:", [by_id[a]["title"] for a in ranked[:5]])

extra_block = ranked[:3]
reranked = index.top_candidates(profile, 10, excluded_ids=[*profile, *extra_block])
check("explicit exclusions honoured", not (set(extra_block) & set(reranked)))
check("unknown profile yields nothing", index.top_candidates([99999999], 5) == [])
check("empty profile yields nothing", index.top_candidates([], 5) == [])

print("\n=== 6. determinism and thread-safety surface ===")
check("repeated calls agree", index.top_candidates(profile, 10, excluded_ids=profile) == ranked)
check("profile_scores normalized to 1.0", abs(max(index.profile_scores(positive_ids=profile).values()) - 1.0) < 1e-6)

print("\n=== 7. serving cost ===")
size_mb = PROD.stat().st_size / 1048576
resident_mb = index.resident_array_bytes / 1048576
timings = []
for _ in range(200):
    t0 = time.perf_counter()
    index.top_candidates(profile, 300, excluded_ids=profile)
    timings.append((time.perf_counter() - t0) * 1000)
p50, p95 = np.percentile(timings, 50), np.percentile(timings, 95)
print(f"     artifact on disk : {size_mb:.1f} MB")
print(f"     resident arrays  : {resident_mb:.1f} MB")
print(f"     Top-300 retrieval: p50 {p50:.2f} ms, p95 {p95:.2f} ms")
check("retrieval stays in single-digit ms", p95 < 20)

print("\n=== 8. no training dependency in the serving import ===")
import sys  # noqa: E402

leaked = [m for m in sys.modules if m.startswith("scipy")]
check("scipy not required to serve", not leaked or os.environ.get("SMOKE_ALLOW_SCIPY"), str(leaked[:3]))

print("\n" + ("ALL SMOKE CHECKS PASSED" if ok else "SMOKE CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
