# V2.1 model experiment plan

Three controlled experiments, proposed and not run. Evidence for each is in
[`BASELINE_DIAGNOSIS.md`](BASELINE_DIAGNOSIS.md),
[`SMALL_PROFILE_EVAL.md`](SMALL_PROFILE_EVAL.md) and
[`SERIES_POLICY.md`](SERIES_POLICY.md).

The production ALS and LambdaMART artifacts stay as they are. Nothing below is
trained, promoted, or deployed without a reviewed result.

## What the evidence says to work on

| Finding | Where it bites | Experiment |
|---|---|---|
| With 1 to 10 Likes, broad-taste users have 29% to 50% of their hidden positives in the candidate set | Retrieval, small profiles | 1 |
| 47% of MAL list rows and 45% of ratings are never trained on; dropped titles are liked 6.7% of the time | Training data | 2 |
| The reranker shows no gain at 5 Likes or fewer, loses at 1, and promotes sequels the feed then removes | Ranking, as served | 3 |

Not on this list, deliberately: a larger or newer model family. For full
profiles the candidate ceiling is already 0.90 and the loss is in ranking; for
small profiles the cheap options have not been tried. See "Not scheduled".

## Rules every experiment follows

1. **Equal data, equal protocol.** The baseline is re-scored in the same run,
   on the same users, with the same known-item rule and candidate budget. A
   number from another report is never the comparison.
2. **Frozen split.** `holdout_seed42_pos8.sqlite`, seed 42, positives are
   ratings of 8 or higher. A challenger that changes what counts as a positive
   still reports against this definition as its primary metric.
3. **Hidden positives stay hidden.** Profiles, features and exclusion lists are
   built through `protocol.build_case`, which refuses a leak. Anything that
   reads `animelist.csv` drops held-out (user, title) pairs first; every
   held-out positive appears there as a completed row.
4. **An untouched confirmation set, opened once.**
   - Spent, diagnostics only: 9,224 users (3,924 from earlier experiments,
     3,300 from the reranker experiment, 2,000 small-profile development users).
   - **Reserved and unopened:** `populations/small_profile_v1.json` holds 800
     users for truncated profiles and 500 genuinely light users at each of
     K = 3, 5, 10, 20.
   - Training and validation populations are drawn from users that are in
     neither list (`stack.spent_user_ids`, `stack.reserved_user_ids`), with a
     new salt, and disjointness is asserted in code.
   - The reserved set serves the **first** model selection made against the
     small-profile evaluation. Opening it means setting `reserved_status` in
     that file, with the decision rule already written. A later experiment
     reserves its own.
5. **A decision rule written first**, in a `PREDECLARATION_<name>.md` next to
   the results, including what happens if it is not met.
6. **Paired user-level bootstrap**, 2,000 resamples, 95% interval, for every
   claimed difference. An interval that includes zero is "not demonstrated".
7. **Report the served condition and the small-profile condition**, not only
   the benchmark:
   - NDCG@10, NDCG@18, Recall@10, Recall@18 as served, under the production
     series policy and under any policy the experiment proposes;
   - the small-profile table at K = 1, 3, 5, 10, 20, both cohorts;
   - distinct titles, head share, and the top-20 titles' share of slots;
   - p50 and p95 latency per stage through the serving code at 3, 60 and 500
     Likes, artifact size, and cold-start cost.
8. **No sampled negatives for selection.**
9. **Offline metrics are proxies.** No result is described as an improvement in
   user satisfaction.

Default gates, unless a predeclaration tightens them:

| Gate | Requirement |
|---|---|
| Primary | The experiment's own primary metric: paired delta with a 95% interval above zero |
| No regression | Benchmark NDCG@10 and Recall@18 intervals not below zero for full profiles |
| Small profiles | No significant loss at any K, in either cohort |
| Concentration | Top-20 share of slots and distinct titles no worse than baseline by more than 10% relative |
| Latency | p95 at 60 Likes within +25% of baseline; p95 at 500 Likes under 150 ms |
| Reproducibility | Deterministic seed, checksummed artifact, role and catalog pins |

---

## Experiment 1. Small-profile retrieval

**Question.** Can more of what a new user will like be put in the candidate set
when the profile is 1 to 10 Likes, without changing anything for larger
profiles?

**Evidence.** Truncated profiles: Recall@300 is 0.29, 0.37, 0.42, 0.50 at K =
1, 3, 5, 10, and even 2,000 candidates reach only 0.53 to 0.76. Genuinely light
users are at 0.66 to 0.78 already, so the headroom is in broad tastes.

**Arms, at equal candidate budgets.** Nothing is trained in the first stage.

| Arm | Candidates |
|---|---|
| Baseline | ALS top 300 |
| Depth control | ALS top 500, and ALS top 1000 |
| Item-item union | ALS 300 + item-item neighbours of the K Liked titles, to 500 and 1000 |
| Popularity-prior union | ALS 300 + the most-liked titles among users who liked any of the K, to 500 |
| Size-gated | The best union below a profile-size threshold, ALS alone above it |

The depth controls are there so that a gain is attributable to the source and
not to the budget.

**Why item-item here is not the rejected experiment.** It was rejected as a
fused ranking source for all users (−9% NDCG@10). Here it only widens the
candidate set, only for small profiles, and contributes no score.

**Primary metric.** Recall at the candidate budget, truncated cohort, averaged
over K ∈ {1, 3, 5, 10}.

**Stage gate.** If no union beats ALS at the same budget with a significant
paired margin on validation users, stop. Recall that does not move cannot move
ranking.

**Second stage, only if the gate passes.** Rank the widened set. The reranker
refuses candidates it has no ALS score for, so every candidate is scored by ALS
first. Report as-served NDCG@10 at each K against the baseline, plus rerank
latency, which grows with the candidate count.

**Confirmation.** The reserved truncated and natural sets, opened once, for the
size-gated arm chosen on validation.

**Risks.** A wider set full of popular titles raises recall and concentration
together; the concentration gate is the check. A gain for truncated profiles
and none for natural ones is still a result worth having.

**Cost.** One to two days to the stage gate; three more beyond it.

## Experiment 2. Status-aware, confidence-weighted ALS

**Question.** Does training on what users started, finished without rating, or
dropped improve retrieval, most of all for small and medium profiles and for
mid-popularity titles?

**Evidence.** The models train on completed-and-rated titles, 52.8% of the MAL
list. Left out: 10.5M completed-unrated rows, 8.9M in progress or on hold, 4.3M
dropped. Like-rates when rated: in progress 62.5%, completed 54.0%, on hold
41.9%, dropped 6.7%.

**Arms.** Identical hyperparameters to the frozen reference (128 factors,
alpha 5.0, regularisation 0.05, 15 iterations, seed 42), so that only the
training data changes.

| Arm | Training matrix |
|---|---|
| Baseline | Train positives at confidence `1 + α`, **rebuilt in the same run** |
| 2a | + completed-unrated at reduced confidence |
| 2b | 2a + in-progress titles, confidence scaled by share of episodes watched |
| 2c | 2b, and dropped titles treated as known rather than unobserved |

Reduced-confidence weights are chosen on validation users only, from a small
predeclared grid.

**Leakage control, mandatory.** Every held-out validation and test positive is
a completed row in `animelist.csv`. The matrix builder drops those (user,
title) pairs, a unit test asserts none is present, and the artifact records the
split hash as the existing one does. Evaluation users' in-progress and dropped
rows are inputs like any other; their held-out positives are not.

**Primary metric.** Recall@300 on the small-profile evaluation, truncated
cohort, averaged over K. Secondary: full-profile Recall@300, and recall of
mid-popularity positives, which is 3.4% today.

**What it must not be judged on.** The current reranker was fitted to the
current ALS scores. Ranking metrics for a new retriever are reported with ALS
order and with a reranker refitted on the new scores, never with the old
reranker over new scores.

**Confirmation.** A newly reserved set, since the first is spent on experiment 1
or 3, whichever selects first.

**Cost.** About five minutes per ALS build plus a 2 GB scan per matrix; three
to four days in total. Retraining is why this is second.

## Experiment 3. A reranker trained for the list that is served

**Question.** Does a reranker fitted to the served policy and to small profiles
beat the current one as served, and stop being harmful at one Like?

**Evidence.**
- At 5 Likes or fewer the reranker has no measurable gain; at 1 it is
  significantly worse than ALS (−0.008 NDCG@10, truncated).
- 51% of its top-10 slots are in an already-Liked franchise, which the feed
  removes. Without that filter it is significantly worse than ALS at finding
  positives in new franchises (−0.016).
- It was fitted on full profiles averaging 86 Likes, with unfiltered labels.

**Arms.** Same 18 features, same LightGBM hyperparameters, same ALS candidates.

| Arm | Training change |
|---|---|
| Baseline | Current LambdaMART |
| ALS only | No reranker, as a floor |
| 3a | Labels and candidates as served: Liked-franchise entries removed, one per franchise |
| 3b | Profiles subsampled to K ∈ {1, 3, 5, 10, 20} alongside the full profile |
| 3c | 3a and 3b together |
| Size-routed | ALS order below a profile-size threshold, the best reranker above it |

**Dependency on the series policy.** 3a is fitted for one policy. If the policy
experiment in `SERIES_POLICY.md` changes what is served, 3a is refitted for
that policy; the arms are written so the policy is a parameter.

**Primary metric.** As-served NDCG@10 averaged over K ∈ {1, 3, 5, 10, 20} and
the full profile, equal weight, truncated and natural cohorts both reported.

**Extra gate.** No significant loss against ALS-only at K = 1 or K = 3. Today's
reranker fails this gate.

**Confirmation.** The reserved small-profile sets, if this experiment selects
before experiment 1 does.

**Why it is cheap.** No ALS retraining, fits in seconds, same artifact format,
no serving code path changes except the optional size route.

**Cost.** Two to three days.

## Order

Experiment 3 is the cheapest and addresses a measured harm (the reranker at one
Like), so it runs first. Experiment 1's first stage trains nothing and can run
alongside it. Experiment 2 follows, because it retrains ALS and its results
decide whether the reranker needs refitting again.

| Order | Experiment | Trains | First result |
|---|---|---|---|
| 1st | 3. Served-policy reranker | LightGBM, seconds | 2–3 days |
| alongside | 1. Small-profile retrieval, stage one | nothing | 1–2 days |
| then | 2. Status-aware ALS | ALS rebuilds | 3–4 days |

## Product experiments this plan depends on

These are not model experiments and offline numbers cannot settle them.

- **Series policy.** `dedupe_liked1` against `production`: one entry from a
  Liked franchise per batch instead of none. Offline it is +0.052 NDCG@10 for
  4% fewer new-franchise hits on full profiles, and +0.08 to +0.12 for light
  users. Whether users want the sequel shown is a question for an online
  comparison.
- **Already Watched uptake.** Now in the interface. Its ceiling offline is
  +0.035 NDCG@10 for low-rated history and +0.056 for all watched history, at a
  matched rate. What users actually mark decides how much of that is real.

## Not scheduled

- **LightGCN, a neural autoencoder, or a generative recommender.** None is
  assumed to help. For full profiles a perfect reorder of the existing 300
  candidates would score 0.90, so a stronger retriever does not address the
  loss. For small profiles, retrieval is the constraint, but experiments 1 and
  2 attack it for days of work rather than a new serving stack. A new family is
  revisited only if, after both, Recall@300 for truncated K ≤ 10 is still below
  0.55, and then it enters under the same rules, with its serving memory and
  latency stated before any training.
- **Synopsis-similarity fusion, LightFM-Hybrid.** Rejected earlier; no new
  hypothesis.
- **Per-user normalised preference strength.** Still plausible (16% of
  positives are at or below the rater's own mean) and shares a matrix builder
  with experiment 2. It follows experiment 2 rather than competing with it.
- **A Plan to Watch discovery target.** A measurement, not a model: 9.2% of
  new-franchise recommendations are already on the user's Plan to Watch list.
  Worth adding to the evaluation as a second lens when experiment 2 reports.
