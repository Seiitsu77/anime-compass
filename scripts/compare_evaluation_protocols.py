"""Score one model under three evaluation protocols to show the metric's range.

Full-catalog NDCG@10 is not comparable to the sampled-negative NDCG@10 that most
published figures use. Model, split, and users are identical across all three
protocols here; only the protocol changes, so the spread is purely methodological.

Protocol C (sampled negatives) is DIAGNOSTIC ONLY. It is reported to make the
scale explicit and must never be used to select a model: Krichene and Rendle
(KDD 2020) show sampled negatives can reorder which model appears better.

    python scripts/compare_evaluation_protocols.py
"""

import json
from pathlib import Path

import numpy as np

from backend.anime_agent.evaluation.collaborative_baselines import ALSModel
from backend.anime_agent.evaluation.metrics import ndcg_at_k
from backend.anime_agent.evaluation.split import SplitStore

catalog = json.loads(Path("data/processed/anime_catalog.json").read_text(encoding="utf-8"))
catalog_ids = [int(i["id"]) for i in catalog]
store = SplitStore(Path("data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite"))
model = ALSModel(
    Path("data/evaluation/personalized/artifacts/holdout_seed42_pos8/als_train_only.npz"),
    catalog_ids,
    build_duration_seconds=0.0,
)
ids = [int(x) for x in Path("data/evaluation/personalized/metadata/als_confirmation_user_ids.txt").read_text().split()]

id_index = model.index_by_id
all_ids = np.asarray(model.anime_ids)
rng = np.random.default_rng(42)

A, B, C = [], [], []  # full-catalog / LOO full-catalog / LOO + 99 negatives
for u in store.iter_users_by_ids(ids):
    test = list(u.test_positive_ids)
    if not test:
        continue
    vec = model._user_vector(u.train_positive_ids)
    if vec is None:
        continue
    scores = model.item_factors @ vec
    known = {a for a, _ in u.all_observed_training_ratings}

    # A: our protocol - all test positives relevant, rank the whole catalog.
    order = np.argsort(-scores)
    ranked = [int(all_ids[i]) for i in order if int(all_ids[i]) not in known][:20]
    A.append(ndcg_at_k(ranked, set(test), 10))

    # B: leave-one-out, still ranking the whole catalog.
    target = int(rng.choice(test))
    others = set(test) - {target}
    ranked_b = [a for a in ranked if a not in others][:20]
    B.append(ndcg_at_k(ranked_b, {target}, 10))

    # C: leave-one-out against 99 sampled unseen negatives (He et al. 2017).
    excluded = known | set(test)
    pool = np.asarray([a for a in catalog_ids if a not in excluded])
    negs = rng.choice(pool, size=99, replace=False)
    cand = np.append(negs, target)
    rows = np.asarray([id_index[int(a)] for a in cand])
    cs = scores[rows]
    top = [int(cand[i]) for i in np.argsort(-cs)][:10]
    C.append(ndcg_at_k(top, {target}, 10))

print(f"users scored: {len(A)}\n")
print(f"{'protocol':<52}{'NDCG@10':>10}")
print("-" * 62)
print(f"{'A. all test positives, full 18,064-item catalog':<52}{np.mean(A):>10.4f}   <- what we report")
print(f"{'B. leave-one-out, full 18,064-item catalog':<52}{np.mean(B):>10.4f}")
print(f"{'C. leave-one-out, 1 positive + 99 sampled negatives':<52}{np.mean(C):>10.4f}   <- typical paper protocol")
