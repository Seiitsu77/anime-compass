# V2.1 model experiment plan

Prioritised challengers, written before any of them is run. Evidence for each
priority is in [`BASELINE_DIAGNOSIS.md`](BASELINE_DIAGNOSIS.md). Nothing in this
plan has been executed, and no production artifact changes until a result has
been reviewed.

## Rules every challenger follows

These are the conditions of entry, not suggestions.

1. **Equal data, equal protocol.** The baseline is re-scored in the same run,
   on the same users, with the same known-item rule and candidate budget as the
   challenger. A number from another report is never the comparison.
2. **Frozen split.** `holdout_seed42_pos8.sqlite` (SHA-256 `a668114f…`), seed
   42, positives are ratings of 8 or higher. A challenger that changes what a
   positive is still reports against this definition as its primary metric.
3. **Three disjoint populations** drawn by `stable_pick` with a new salt:
   training, validation (all selection and early stopping), confirmation
   (opened once). Disjointness is asserted in code.
4. **An untouched confirmation set.** 7,224 users are already spent: the 3,924
   in `metadata/inspected_user_ids.json` plus the reranker experiment's 3,300
   in `results/reranker/confirm_user_ids.json.gz`. A confirmation set excludes
   all of them. 276,285 eligible users remain untouched.
5. **A decision rule written before the confirmation set is opened**, in a
   `PREDECLARATION_<name>.md` next to the results, including what happens if it
   is not met.
6. **Paired user-level bootstrap**, 2,000 resamples, 95% percentile interval,
   for every claimed difference. A delta whose interval includes zero is
   reported as not demonstrated.
7. **Report the served condition, not only the benchmark.** Every challenger
   reports, alongside the benchmark metrics:
   - NDCG@10, NDCG@18, Recall@10, Recall@18 **as served** (franchise filter on,
     scored on positives the feed may show);
   - the **profile-size curve** at K = 1, 3, 5, 10, 20, 50;
   - distinct items, head/mid/tail share, and the top-20 titles' share of
     slots at 10 and 18;
   - p50 and p95 latency per stage through the real serving code, at 3, 60 and
     500 Likes;
   - artifact size and cold-start cost.
8. **No sampled negatives** for selection (Krichene and Rendle, KDD 2020).
9. **Offline metrics are proxies.** No result is described as an improvement in
   user satisfaction.

Default promotion gates, unless a predeclaration tightens them:

| Gate | Requirement |
|---|---|
| Primary | As-served NDCG@10 paired delta with a 95% interval above zero |
| No regression | Benchmark NDCG@10 and Recall@18 intervals not below zero |
| Small profiles | No significant loss at any K ≤ 20 |
| Concentration | Top-20 share of slots and distinct items no worse than baseline by more than 10% relative |
| Latency | p95 at 60 Likes within +25% of baseline; p95 at 500 Likes under 150 ms |
| Reproducibility | Deterministic seed, checksummed artifact, role and catalog pins |

## Not being retried

- **Synopsis-similarity fusion.** Rejected; no new hypothesis here.
- **LightFM-Hybrid.** Rejected; metadata did not help and coverage halved.
- **Item-item as a fused ranking source.** Cost 9% NDCG@10. Challenger C below
  is a different hypothesis: candidates only, small profiles only, the
  reranker decides the order.

---

## Step 0. Adopt the served-condition measurements (no model change)

**Why first.** Two of the diagnosis's findings are about measurement, and every
later decision depends on them: the product-relevant NDCG@10 is 0.216 rather
than 0.283, and the uniform sample contains no small profile.

**Work.** Move `as_served` and the profile-size curve from
`experiments/v21_quality/` into the standing evaluation runner so every future
report carries them. Add a third relevance view that counts validation-holdout
likes as relevant for final reporting only (19% of top-10 slots are known likes
currently scored as misses); it is never used for selection.

**Cost.** About a day. No training, no users spent.

## A. A reranker trained for the list that is served

**Hypothesis.** The reranker was fitted on full profiles (mean 86 likes) with
unfiltered labels. Half of its top 10 is then removed by the franchise filter,
and it shows no held-out gain below 50 Likes. A reranker trained on
(1) franchise-collapsed candidates with showable labels and (2) profiles
subsampled to session size will beat the current one as served, most of all at
K ≤ 20.

**Why it ranks first among model changes.** It targets findings 1 and 3
together, needs no ALS retraining, fits in seconds, and changes no serving code
path: same features, same artifact format.

**Arms.** Current LambdaMART; retrained on collapsed labels only; retrained on
subsampled profiles only; both. Same 18 features, same hyperparameters.

**Data.** 2,000 fresh training users, each contributing its full profile plus
subsamples at K = 3, 10, 30; 500 validation; 800 confirmation.

**Primary metric.** As-served NDCG@10 averaged over K ∈ {3, 10, 20} and the full
profile, equal weight.

**Risks.** Label sparsity at small K; a model that helps small profiles and
hurts large ones, which the per-K gate catches.

**Cost.** One to two days.

## B. Explicit negative and "already seen" signal

**Hypothesis.** Knowing what a user has watched and not loved is worth +0.045
NDCG@10 as an exclusion alone. Using it also as a ranking signal (similarity of
a candidate to the user's low-rated titles) adds a further gain on top of
exclusion.

**Arms.** (1) Baseline. (2) Exclusion only: remove titles rated below 8 from
candidates, which is the published protocol and is already measured.
(3) Exclusion plus three reranker features: maximum item-item similarity to a
title the user rated 5 or lower, the same against 6-to-7 titles, and genre
overlap with the low-rated set.

**Equal-data baseline.** Arm 2, not arm 1. The question is what the features
add once exclusion is granted.

**Product dependency.** The live browser can supply this signal only through
Not Interested today. Offline evidence for arm 2 is the case for adding an
"Already Watched" action to the React frontend, which the legacy frontend had
as `seen_titles`. That is a product decision to take with this number in hand,
not a model change.

**Cost.** Two days offline. The frontend action is separate work.

## C. Wider retrieval for small profiles only

**Hypothesis.** With ten or fewer Likes, ALS top 300 contains under half of the
held-out positives, and item-item neighbours of those few titles recover
positives ALS misses. Adding item-item candidates for small profiles, leaving
the reranker to order them, raises as-served NDCG@10 at K ≤ 10 without
affecting large profiles.

**Why this is not the rejected experiment.** Item-item was rejected as a fused
ranking source across all users. Here it only widens the candidate set, only
below a profile-size threshold, and never contributes a score to the ranking.

**Arms at equal candidate budget.** ALS 300; ALS 500; ALS 300 + item-item 200,
for K ≤ 10 only. Depth 500 for ALS alone is included so that a gain is
attributable to the source and not to the budget.

**Stop condition.** If union recall at K ≤ 10 does not exceed ALS at the same
budget by a significant margin on validation, stop before fitting anything.

**Risks.** The reranker has never seen candidates from outside ALS top 300 and
refuses unknown rows by design; candidates must be scored by ALS as well. Rerank
latency grows with the candidate count.

**Cost.** One day to the stop condition; three more if it passes.

## D. Status-aware, confidence-weighted ALS

**Hypothesis.** Training on completed-and-rated titles discards 47% of the
list. Adding completed-but-unrated and in-progress titles at reduced
confidence, and treating dropped titles as known rather than unobserved,
improves retrieval for medium and small profiles and for mid-popularity titles.

**Evidence.** Dropped titles are liked 6.7% of the time; in-progress titles
62.5%; 10.5M completed titles carry no rating at all.

**Arms, identical hyperparameters to the frozen reference** (128 factors,
alpha 5.0, regularisation 0.05, 15 iterations, seed 42) so that only the data
changes:

| Arm | Confidence |
|---|---|
| Baseline | `1 + α` for train positives (rebuilt in the same run) |
| D1 | + completed-unrated at reduced confidence |
| D2 | D1 + currently watching, scaled by progress |
| D3 | D2, and dropped titles excluded from candidates as known items |

Reduced-confidence weights are chosen on validation users only.

**Leakage control, mandatory.** Every held-out validation and test positive is
also a completed row in `animelist.csv`. The training matrix must drop those
(user, title) pairs. A unit test asserts that no held-out pair is present, and
the artifact records the split hash as the existing one does.

**Equal-data baseline.** The frozen reference rebuilt in the same run from the
same row scan, not the stored artifact.

**Cost.** About 5 minutes per ALS build, a 2 GB scan per matrix, three to four
days in total. Retraining is the reason this sits below A to C.

## E. Per-user normalised preference strength

**Hypothesis.** A fixed threshold treats a 7 from a harsh rater and an 8 from
someone who gives everything 8 identically wrong. Graded confidence from a
per-user standardised rating ranks better than a binary positive.

**Evidence.** 16.1% of positives are at or below the rater's own mean; 19.6%
of 6-or-7 ratings are above it; for a quarter of users the threshold selects
80% or more of everything they rated.

**Arms.** Baseline binary confidence; `c = 1 + α·g(z)` with `g` a clipped,
monotone function of the user-standardised rating, positives unchanged; the
same with above-mean 6-or-7 ratings admitted as weak positives.

**Primary metric stays threshold 8**, so results remain comparable with every
earlier report. Relevance by the relative definition is reported as a secondary
view and never used for selection.

**Risk.** A per-user mean needs enough ratings to mean anything; users with
fewer than ten ratings keep the binary rule.

**Cost.** Three days. Shares the matrix builder with D.

## F. A discovery target from Plan to Watch (measurement)

**Hypothesis.** A recommender's job is to surface unseen titles a user would
choose. 27.9M Plan to Watch rows from 90% of users are exactly that, and they
are untouched by the current benchmark, which rewards reconstructing what was
already watched.

**Work.** Define "intent recall at K": the share of a user's planned titles in
the top K, with all watched titles excluded. Report it for the baseline first.
It is a second lens, not a replacement for held-out likes, and it mirrors the
product's Want to Watch.

**Caveat to keep visible.** A planned title is intent, not a confirmed like:
of the 1% that carry a rating, 61.8% are 8 or higher.

**Cost.** Two days. No model change.

## G. LightGCN or an autoencoder

**Not justified yet.** For full profiles the candidate ceiling is 0.90 and the
loss is in ranking; a stronger retriever does not address that. For small
profiles retrieval is the constraint, but A, C and D attack it at a fraction
of the cost.

**Entry condition.** Revisit only if, after C and D, recall at 300 for K ≤ 10
is still below 0.55 and the as-served gate has not been met. The first
candidate would then be an item-based autoencoder scored on the profile-size
curve, with the serving-memory cost of an 18,064 × 18,064 item matrix stated
up front.

---

## Order and expected value

| # | Work | Trains a model | Addresses | Cost |
|---|---|---|---|---|
| 0 | Served-condition measurements | no | measurement | 1 day |
| A | Reranker for the served list and small profiles | LightGBM, seconds | findings 1, 3 | 1–2 days |
| B | Negative and seen signal | LightGBM, seconds | finding 2 | 2 days |
| C | Wider retrieval for small profiles | no | finding 1 | 1–4 days |
| F | Plan to Watch discovery target | no | measurement | 2 days |
| D | Status-aware ALS | ALS rebuilds | findings 1, 5 | 3–4 days |
| E | Per-user normalised confidence | ALS rebuilds | finding 5 | 3 days |
| G | LightGCN or autoencoder | yes | gated on C, D | not scheduled |

Each confirmation consumes 800 untouched users. Running A through E end to end
uses roughly 4,000 of the 276,285 remaining.
