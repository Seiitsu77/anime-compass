# Learned Fusion Weights: Decision Report

## Decision

**Keep the hand-set channel weights. Confirmed on a predeclared, untouched
sample of 800 users.**

The learned blend is significantly *worse* than the constants it was meant to
replace, on both primary and secondary metrics, with intervals excluding zero:

| Metric | Hand-set | Learned | Delta | 95% CI | Excludes 0 |
|---|---:|---:|---:|---:|---|
| NDCG@10 | 0.1996 | 0.1814 | **−0.0181** (−9.1%) | [−0.0272, −0.0095] | yes |
| Recall@10 | 0.1873 | 0.1608 | **−0.0265** (−14.2%) | [−0.0370, −0.0163] | yes |
| Catalog coverage | 0.0509 | 0.0742 | +0.0233 (+45.8%) | — | — |

Against the [frozen decision rule](../PREDECLARATION_fusion_confirmation.md),
gates 1 and 2 fail and gate 3 passes. **The learned blend is not adopted.** The
confirmation set was opened once and is not reused.

This is reported as a result rather than discarded as a failure. "The weights are
hand-picked constants" was a real criticism of the model, and the honest answer
turns out to be that the constants are competitive, not that they were lazy.

### The one thing the learned blend does win

Catalog coverage improves 45.8% relative. The fit trades accuracy for breadth of
exposure. That trade was not what the predeclared rule optimised for, and it is
not adopted here — but it is a genuine finding: if a future product goal
prioritises catalog exposure over top-10 precision, this weight vector is a
starting point rather than a dead end.

### Methodology note: why this run supersedes the first one

The first fit reported a −0.0059 pairwise-accuracy delta on users drawn from the
same pool as earlier exploratory runs. This run fixed three problems with that
setup:

1. **The semantic channel is now live.** Previously it had no variance and could
   not be fitted; a third of the weight mass was effectively unfittable.
2. **The confirmation sample is genuinely untouched.** All 1,306 users any prior
   experiment had scored were excluded from the pool before sampling, and the
   fit additionally reserved the 800 confirmation users. Disjointness is asserted
   in the run manifest (`sample_is_disjoint_from_excluded: true`, overlap 0).
3. **The metric is the one the product cares about.** The first run reported
   pairwise accuracy, which is what the loss optimises. This run reports NDCG@10
   and Recall@10 with paired bootstrap intervals.

## Method

The production scorer is linear in its channel signals:

    score(item) = sum_c effective_weight[c] * signal[c](item)

so the weights can be fitted directly. The objective is pairwise, not pointwise:
for a user with a held-out positive `p` and a non-relevant candidate `n`, the fit
maximises `P(w · x_p > w · x_n)` under a RankNet-style logistic loss on the score
difference. That matches how the model is actually used — ordering candidates —
and avoids needing an intercept or calibrated probabilities.

Weights are constrained non-negative by projected gradient descent and
renormalised to sum to one after every step, so the optimiser only visits vectors
the serving path already accepts and each weight stays interpretable as a share
of the total.

Features are read from the recommender's own score breakdown, so they are exactly
the numbers the production scorer blends, with no reimplementation that could
drift.

**Fit and confirmation are on disjoint users, and the confirmation users were
never inspected by any earlier experiment.** Weights were fitted on validation
holdout positives only; the confirmation is a separate run of the full
evaluation harness on 800 predeclared users.

| | Fit set | Confirmation set |
|---|---:|---:|
| Users offered | 600 | 800 |
| Users contributing pairs | 571 | 800 evaluated |
| Pairs | 27,912 | — |
| Held-out positives covered | 3,489 | — |
| Held-out positives missed by the shortlist | 2,557 | — |
| Reserved from the fitting pool | 2,106 IDs | — |
| Overlap with previously inspected users | — | **0** |

The optimiser was started from `experimental_semantic_weights()`, i.e. with the
retired semantic channel restored to 0.14, so the fit could argue that weight
back up if the data supported it.

## Result

| Channel | Shipped | Fit started at | Fitted | Delta vs shipped |
|---|---:|---:|---:|---:|
| metadata | 0.1600 | 0.1600 | 0.1165 | −0.0435 |
| synopsis | 0.1000 | 0.1000 | 0.0111 | −0.0889 |
| lsa | 0.0400 | 0.0400 | 0.0000 | −0.0400 |
| semantic_embedding | 0.0000 | **0.1400** | **0.0294** | +0.0294 |
| dense | 0.0800 | 0.0800 | 0.0018 | −0.0782 |
| creator | 0.0500 | 0.0500 | 0.2404 | **+0.1904** |
| collaborative | 0.2200 | 0.2200 | 0.3393 | **+0.1193** |
| quality | 0.1300 | 0.1300 | 0.1123 | −0.0177 |
| session | 0.0500 | 0.0500 | 0.1492 | +0.0992 |
| novelty | 0.0300 | 0.0300 | 0.0000 | −0.0300 |

Fit-set pairwise accuracy was 0.7192. The optimiser fitted the training pairs;
it did not generalise, as the confirmation table above shows.

Per-channel difference variance is recorded in the artifact. `novelty` is the
only channel flagged as zero-variance, so its fitted 0.0 is uninformative;
`semantic_embedding` had genuine variance (0.0173) and was still pushed down.

## What the weights suggest, and what they do not

The direction of the shift is informative even though the fit lost. It moves
weight decisively toward **collaborative** (+0.14) and **creator** (+0.21)
signals and away from text-similarity channels. That is consistent with the
collaborative benchmark, where latent-factor models beat every content-based
signal by a wide margin.

Three caveats limit how far that reading can be pushed:

**The semantic channel is now live, and independently corroborates its
retirement.** In the refit the optimiser was deliberately started at the retired
channel's former weight of 0.14, so it could argue the weight back up. It did the
opposite: the fitted weight is **0.0294**, a 79% reduction, with a real feature
variance of 0.0173 (not zero). Because this fit is a different method on
different users from the
[semantic ablation](semantic_channel_summary.md), it is genuine convergent
evidence that 0.14 was too high.

`novelty` is now the channel flagged with zero variance, because signal
extraction runs with a neutral novelty preference. Its fitted 0.0 carries no
evidential value, and the run manifest records this rather than leaving it to be
inferred.

**43% of held-out positives never entered the shortlist.** 1,757 of 4,065
relevant items were outside the 300-candidate window, so the fit optimises
ordering *within* the retrieved region and says nothing about retrieval itself.
A model that ranks the retrieved set perfectly still misses those.

**One split, one threshold, one shortlist size.** No sensitivity analysis was run
over shortlist depth, negatives per positive, or positive threshold.

## Why the hand-set weights likely win

The most plausible explanation is regularisation by design. Ten weights fitted on
18k pairs drawn from 376 users is a small, correlated sample: the channels are
not independent (metadata, synopsis, LSA, and dense all derive from overlapping
text features), so the fit can shift mass between near-collinear channels without
improving the ranking, and then generalise slightly worse. The hand-set vector
spreads weight across channels in a way that behaves like a prior.

## What would change this

- ~~Rerun with the semantic artifact present.~~ **Done.** The channel was live
  for this fit and was still pushed from 0.14 down to 0.029.
- **Fit on more users.** 571 contributing users is still small; the cost is
  roughly 1.1 s per user, so several thousand is practical.
- **Vary the novelty preference during signal extraction**, so that channel has
  variance to be judged on instead of being flagged inert.
- **Add explicit regularisation toward the hand-set prior** rather than toward
  zero, which directly encodes "only move if the data justifies it."
- **Widen or remove the shortlist** so retrieval is not silently held fixed.
- **Score with ranking metrics, not just pairwise accuracy.** The runner supports
  this: `--models current_hybrid,current_hybrid_learned --fusion-weights <path>`
  produces NDCG/Recall with paired bootstrap intervals for both blends on
  identical users. Pairwise accuracy was used here because it is the quantity the
  loss optimises; it is not the quantity the product cares about.

## Reproduce

```powershell
python scripts/train_fusion_weights.py `
    --split data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite `
    --countsketch data/evaluation/personalized/artifacts/holdout_seed42_pos8/countsketch_train_only.npz `
    --train-users 400 --test-users 400 --shortlist 300 --iterations 800
```

Roughly 1.1 s per user after a 35 s setup. Extracting channel signals requests
the score breakdown without natural-language explanations, and passes
`diversity_strength=0.0`, which takes a non-quadratic path through the reranker.
Both matter: the naive call was 48.7 s per user, and the same call with greedy
diversity reranking at `limit=300` was 77 s per user.
