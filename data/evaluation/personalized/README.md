# Personalized Offline Evaluation

This benchmark is separate from the seven manually labelled catalog/agent cases in
[`../benchmark.json`](../benchmark.json). Those cases remain qualitative regression tests. This
pipeline measures per-user ranking on held-out positive ratings.

> Because interaction timestamps are unavailable, this evaluation measures preference
> reconstruction/generalization under a deterministic user-stratified random holdout, not
> chronological next-item prediction.

## Protocol

- Positive: rating `>= 8`; neutral: `6..7`; explicit negative: `<= 5`. The ranges are CLI-configurable,
  stored independently, and ratings outside configured ranges remain known/ignored rather than becoming
  unobserved items.
- Split seed: `42`. The split is independently shuffled per user using a stable hash of seed and user ID.
- Users with fewer than five positives are training-only. Users with 5–9 positives hold out one validation
  and one test item; 10–19 hold out one validation and two test items; users with 20+ hold out floor(10%)
  for each, with a minimum of one.
- The validation set is reserved for later tuning and is not used by any current model. Primary evaluation
  uses all test positives for binary relevance.
- Every candidate model sees the same catalog and user IDs. Exact training-known positives, neutral ratings,
  explicit negatives, and ignored ratings are excluded from rankings.
- Sampled runs default to a deterministic activity-stratified hash sample so sparse, medium, and heavy users
  are represented; `--sampling-strategy uniform` is also available. Full runs evaluate every eligible user.
- Popularity and item-popularity buckets use positive training counts only. CountSketch is rebuilt from
  observed training ratings only. Rating-derived catalog aggregates used by the hybrid are rebuilt or cleared
  from training data.

## Models

1. `popularity`: global positive training-interaction count, anime ID as tie-breaker.
2. `countsketch_cf`: the existing user-centred CountSketch item-similarity algorithm, trained on the split's
   observed training ratings; profiles use positive training items.
3. `current_hybrid`: the current production hybrid, invoked directly without HTTP, LLM explanations,
   semantic-network calls, or frontend work. Content metadata is retained; rating-derived aggregates are
   train-only.

No LightFM, LightGCN, SASRec, or other new model is included in this phase.

## Metric definitions

- Ranking metrics are calculated once per user and macro-averaged. Recall divides top-K hits by the number of
  that user's test positives. Binary NDCG uses logarithmic discount and an ideal ranking of up to K test
  positives. HitRate@10 is one when any test positive occurs in the first ten. MRR is the reciprocal rank of
  the first test positive within 20.
- Catalog coverage is unique recommended IDs divided by catalog size.
- Novelty is mean self-information in bits, `-log2((train_positive_count + 1) /
  (all_train_positives + catalog_size))`.
- Popularity bias is the per-user mean normalized-log popularity of recommendations minus that of the user's
  positive training history. Positive values mean recommendations skew more popular than the profile.
- Intra-list diversity is the mean pairwise Jaccard distance between catalog genre sets.
- User activity uses positive training interactions: sparse `1–4`, medium `5–19`, heavy `20+`. Sparse users
  remain possible because a five-positive user has three training positives after holdout.
- Item buckets rank the full catalog by positive training count with anime ID as a deterministic tie-breaker:
  head is the top 20%, mid-tail the next 30%, and long-tail the bottom 50%, including zero-count items.
- Main model differences use a paired user-level percentile bootstrap, never independent samples.

## Reproduce

From the repository root after installing `requirements-dev.txt`:

```powershell
# Tests
python -m pytest -q

# Fast pipeline check using only the first 500 source users and 3 evaluation users
python scripts/evaluate_personalized.py --source-user-limit 500 --max-evaluation-users 3 --bootstrap-iterations 100 --output-dir data/evaluation/personalized/results/pipeline_smoke

# Representative smoke: full training split, deterministic 10-user evaluation sample
python scripts/evaluate_personalized.py --max-evaluation-users 10 --bootstrap-iterations 2000 --output-dir data/evaluation/personalized/results/representative_smoke

# Full eligible-user evaluation (currently computationally expensive for the Python hybrid)
python scripts/evaluate_personalized.py --max-evaluation-users 0 --bootstrap-iterations 2000 --output-dir data/evaluation/personalized/results/full
```

The persistent split and train-only model artifacts are ignored by Git because they are large and reproducible.
Each result directory contains `results.json`; aggregate, segment, item-bucket, engineering, and paired-bootstrap
CSVs; compressed per-user metrics; `report.md`; and a checksummed `manifest.json`. Derived report/CSV views can
be recreated without inference using `python scripts/evaluate_personalized.py --refresh-output <result-dir>`.

## Interpretation limits

This can estimate personalized ranking generalization, collaborative/hybrid lift, sparse-user behavior,
popularity exposure, long-tail recovery, and offline engineering cost. It cannot establish chronological
next-anime prediction, online CTR or acceptance, causal satisfaction, or A/B-test impact. The ratings contain
selection and survivorship bias, and the random holdout lets a user's later franchise entries help reconstruct
earlier entries; those limitations should remain visible in any portfolio claim.
