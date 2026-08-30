# Evaluation Protocol

## The primary protocol

Every headline number ranks each user's held-out favourites against the **full
18,064-item catalog**, counting **all** of their held-out positives.

```text
for each held-out user:
    profile   = their training positives
    relevant  = ALL of their held-out positives   (~10.6 on average)
    candidates= all 18,064 catalog items, minus anything already observed
    score     = NDCG@10 / Recall@10 over that ranking
```

This is what the product does: when someone asks for recommendations, the system
ranks the whole catalog, not a shortlist that is guaranteed to contain the
answer.

## Why the numbers look small

Published recommender NDCG@10 figures in the 0.5–0.7 range almost always come
from a **sampled-negative** protocol — hold out one interaction, sample 99 items
the user has not seen, rank those 100. The same ALS model, same users, same
split:

| Protocol | NDCG@10 |
|---|---:|
| **All test positives, full 18,064-item catalog** | **0.2588** |
| Leave-one-out, full catalog | 0.1775 |
| Leave-one-out + 99 sampled negatives | 0.8260 |

Nothing about the model changes between those rows. Picking one correct answer
from 100 candidates is a fundamentally easier task than picking ~10 specific
titles from 18,064.

The sampled-negative row is reported **only as a comparability diagnostic**. It
is never used to select a model, because Krichene & Rendle (KDD 2020) showed
sampled negatives do not merely shift metrics — they can reorder which model
appears better.

Reproduce all three with:

```bash
python scripts/compare_evaluation_protocols.py
```

## Interpreting Recall@10

32% of held-out users have more than ten held-out positives, and there are only
ten slots. The attainable ceiling is:

```text
oracle_recall_at_10 = mean( min(10, n_positives) / n_positives ) = 0.865
```

So ALS's Recall@10 of 0.2480 is **29% of the maximum attainable**, not 25% of
1.0. This normalisation is a reading aid; standard Recall@10 remains the
reported metric.

An oracle that places held-out positives first scores exactly NDCG@10 1.0000 and
Recall@10 0.8650, which confirms the metric implementation is correct. A seeded
random recommender scores 0.0008, confirming the floor.

## Sample discipline

Exploratory work burns users. By the end of the modelling phase, 3,706 distinct
users had been scored by some experiment. Reusing any of them would turn a
"held-out" claim into a restatement of numbers already seen.

So each confirmation run:

1. Excludes every user any earlier run scored.
2. Draws a fresh deterministic sample from what remains.
3. Asserts disjointness in code at sample time.
4. Records `sample_is_disjoint_from_excluded` in its run manifest.
5. Is opened **once**, against a decision rule frozen in writing beforehand.

Predeclaration documents live alongside the results:

- `data/evaluation/personalized/PREDECLARATION_als_confirmation.md`
- `data/evaluation/personalized/PREDECLARATION_fusion_confirmation.md`

Both record the frozen rule before the numbers existed, including the case where
the rule was not met and the change was rejected.

## Two model artifacts

| | Evaluation | Production |
|---|---|---|
| Trained on | split train positives (24,916,911) | all positives (30,875,410) |
| Withholds held-out positives | yes | no |
| Valid for holdout metrics | **yes** | **no** |
| Valid for serving | yes, but weaker | **yes** |

Substituting one for the other is silent and wrong in both directions: serving
the evaluation build ships a model trained on 19% fewer interactions, and
measuring against the production build scores it on interactions it already
trained on. The loader checks `artifact_role` and refuses a mismatch.

## What this evaluation cannot tell you

- **Real user satisfaction.** These are offline engineering proxies.
- **Next-item prediction.** The dataset has no interaction timestamps, so the
  holdout is a random split of each user's liked set. This measures preference
  reconstruction.
- **Post-2020 behaviour.** The interaction snapshot ends in 2020.
- **Tail quality at scale.** Long-tail conclusions rest on a balanced diagnostic
  of 100 users and 176 held-out tail items, not the natural population.

## Reports

Full decision records, each with its methodology and caveats:

| Report | Question |
|---|---|
| `results/collaborative_baselines_summary.md` | What does the CountSketch projection cost, and how does ALS compare? |
| `results/als_confirmation_summary.md` | Does tuned ALS pass its standalone gates? |
| `results/als_promotion_decision.md` | Complete evidence: thresholds, strata, tails, retrieval, hybrid substitution |
| `results/semantic_channel_summary.md` | Is the pretrained embedding channel worth its weight? |
| `results/learned_fusion_summary.md` | Can the blend weights be learned? |
| `results/metric_comparability.md` | Why these numbers are not comparable to published ones |
| `results/lightfm_challenger_summary.md` | Why LightFM was declined |

All under `data/evaluation/personalized/`.
