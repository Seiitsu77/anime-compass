# Predeclaration: ALS Standalone Confirmation

**Written after the validation sweep completed and before the confirmation
sample was scored.** The configuration below was selected on validation
positives only; the test split has not been read for any ALS candidate.

## Scope

This confirms ALS **as a standalone collaborative model**. It does not authorise
integrating ALS into the production hybrid. Integration is a separate decision
that requires its own evidence.

## Selected configuration (frozen)

Selected by the sweep's predeclared rule — highest validation NDCG@10, ties by
Recall@10, then fewer factors — across 12 candidates over two validation-only
rounds:

| Parameter | Value |
|---|---|
| factors | 128 |
| iterations | 15 |
| regularization | 0.05 |
| alpha | 5.0 |
| cg_steps | 3 |
| seed | 42 |

Validation NDCG@10 0.2787, Recall@10 0.2547, coverage 0.0773.

## Sample definition (frozen)

| Property | Value |
|---|---|
| Split | `holdout_seed42_pos8.sqlite`, `rating >= 8` |
| Strategy | uniform |
| Seed | `20260901` |
| Size | 800 users |
| Excluded | the 2,906 IDs in `metadata/burned_user_ids.txt` |
| Frozen sample | `metadata/als_confirmation_user_ids.txt` |
| Overlap with any previously scored user | **0** |

The exclusion set includes every user scored by any prior run *and* the 800
validation users used to select these hyperparameters, so selection and
confirmation share no users.

## Comparison (frozen)

One run on the 800 confirmation users comparing three models on identical users:

- `popularity` — floor
- `countsketch_cf` — the production collaborative channel (incumbent)
- `als` — the selected configuration

Primary metric **NDCG@10**; secondary **Recall@10**. Paired bootstrap, 2,000
resamples, seed 42.

## Decision rule (frozen, written before seeing the result)

ALS passes its standalone confirmation **only if all four hold**:

1. NDCG@10 beats CountSketch and the paired 95% interval excludes zero.
2. Recall@10 does not regress with an interval excluding zero.
3. **Catalog coverage is at least 90% of CountSketch's.** This gate exists
   because the sweep revealed that tuning alpha for accuracy trades coverage
   away: coverage falls monotonically from 0.0947 at alpha=40 to 0.0502 at
   alpha=1. The earlier claim that ALS beats the incumbent on coverage came from
   an *untuned* configuration and does not survive tuning at 64 factors.
4. Popularity bias does not exceed CountSketch's.

The confirmation set is opened **once**. Failing the rule means reporting a
negative result, not re-tuning and re-opening.

## Still outstanding after this run

Passing the above is necessary, not sufficient, for promotion. These remain:

- Threshold 7 and 9 sensitivity. The LightFM lift vanished at threshold 9; ALS
  has not been checked.
- Activity-stratified diagnostic. Sparse-user regression was a specific LightFM
  failure and must be measured, not assumed.
- Item-popularity-stratified diagnostic (head / mid-tail / long-tail recovery).
- Intra-list diversity against a product threshold.

## Already established

- **Fold-in inference is faithful.** `ALSModel` exports item factors only and
  recomputes a user vector per request. Against the trained user factors for
  2,000 sampled users: cosine mean 0.9995 (min 0.9972, 100% above 0.95), top-20
  ranking overlap mean 0.9728 (99.4% at or above 0.90). Serving-time fold-in is
  therefore measuring what ALS actually learned, and generalises to users absent
  from training at no measurable accuracy cost.
