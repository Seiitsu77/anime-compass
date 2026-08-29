# ALS Standalone Confirmation: Result

## Outcome

**ALS passes standalone confirmation on all four predeclared gates.**

This authorises ALS as a validated standalone collaborative model. It does
**not** authorise integrating it into the production hybrid; that decision needs
its own evidence, and the diagnostics listed at the end remain outstanding.

## Result

800 users drawn at seed `20260901`, excluding all 2,906 users any prior run had
scored — including the 800 validation users used to select these
hyperparameters. Selection and confirmation share no users. Overlap with
previously scored users: **0**, asserted in the run manifest.

| Metric | CountSketch (incumbent) | ALS (tuned) | Delta | Relative | 95% CI | Excludes 0 |
|---|---:|---:|---:|---:|---:|---|
| NDCG@10 | 0.1641 | **0.2875** | +0.1234 | **+75.2%** | [+0.1084, +0.1384] | yes |
| Recall@10 | 0.1485 | **0.2618** | +0.1133 | **+76.3%** | [+0.0980, +0.1286] | yes |
| HR@10 | 0.6025 | **0.8175** | +0.2150 | +35.7% | — | — |
| Catalog coverage | 0.0744 | 0.0768 | +0.0024 | +3.3% | — | — |
| Popularity bias | 0.0546 | **0.0296** | −0.0250 | −45.8% | — | — |
| Intra-list diversity | **0.8262** | 0.8021 | −0.0241 | −2.9% | — | — |
| p50 rank latency | 7.03 ms | 6.91 ms | — | — | — | — |

| Gate | Result |
|---|---|
| 1. NDCG@10 beats incumbent, interval excludes zero | pass |
| 2. Recall@10 does not significantly regress | pass |
| 3. Coverage at least 90% of incumbent | pass (103.3%) |
| 4. Popularity bias no worse than incumbent | pass |

## The tuning finding that matters most

The original ALS number (NDCG@10 0.1841) came from stock defaults: 64 factors,
alpha 40. Those defaults were badly wrong for this dataset.

**Alpha dominates, and lower is better.** Validation NDCG@10 across the sweep:

| alpha (64 factors) | 100 | 40 | 10 | 5 | 2.5 | 1 |
|---|---:|---:|---:|---:|---:|---:|
| val NDCG@10 | 0.1486 | 0.2032 | 0.2485 | 0.2640 | **0.2733** | 0.2677 |
| val coverage | 0.1158 | 0.0947 | 0.0722 | 0.0640 | 0.0565 | 0.0502 |

Alpha is the implicit-feedback confidence scaling: `c_ui = 1 + alpha * r_ui`. A
high alpha tells the model to trust every observed positive very strongly. With
86 positives per user on average, alpha=40 over-weights an already dense signal
and drives the factors toward popularity.

**Accuracy and coverage move in opposite directions along alpha**, which is
exactly the trade that blocked LightFM. At 64 factors, tuning for NDCG would have
dropped coverage to 0.0565, well under the incumbent's 0.0744.

**Capacity resolves the trade.** Raising factors to 128 at alpha=5 gives the best
validation NDCG@10 (0.2787) *and* restores coverage to 0.0773. That is why the
selected configuration is `f128 / alpha=5` rather than the single-parameter
optimum `f64 / alpha=2.5`.

## Selected configuration

| Parameter | Value |
|---|---|
| factors | 128 |
| iterations | 15 |
| regularization | 0.05 |
| alpha | 5.0 |
| cg_steps | 3 |

Selected by validation NDCG@10 over 12 candidates in two rounds. The first round
selected at a grid boundary (alpha=10 was its lowest value), which is not a real
optimum, so the grid was extended downward. Both rounds read validation
positives only; the test split was untouched until the single confirmation run.

## Serving

`ALSModel` exports item factors only and recomputes a user vector per request by
folding the user's positives into item space. Against the trained user factors
for 2,000 sampled training users:

- cosine similarity mean **0.9995** (min 0.9972; 100% above 0.95)
- top-20 ranking overlap mean **0.9728** (99.4% at or above 0.90)

Fold-in is therefore a faithful reconstruction, not an approximation that changes
what is measured. It also means the model generalises to users absent from
training at no measurable accuracy cost, which a stored-user-factor design
cannot do. p50 rank latency is unchanged against the incumbent.

## Still outstanding before promotion

Passing standalone confirmation is necessary, not sufficient.

- **Threshold 7 and 9 sensitivity.** The LightFM lift vanished at threshold 9.
  ALS has not been checked, and a lift this large deserves more scepticism, not
  less.
- **Activity-stratified diagnostic.** Sparse-user regression was a specific
  LightFM failure. It must be measured rather than assumed from an aggregate.
- **Item-popularity-stratified diagnostic**, for head / mid-tail / long-tail
  recovery.
- **Intra-list diversity** is the one metric that regresses (−2.9%) and should be
  judged against a product threshold rather than accepted implicitly.
- **Hybrid integration is a separate decision.** These numbers compare ALS to the
  CountSketch *channel*, not to the full hybrid, which scored NDCG@10 0.1996 on
  its own confirmation sample using ten blended channels.

## Reproduce

```powershell
python scripts/sweep_als_validation.py `
    --grid data/evaluation/personalized/configs/als_sweep.json `
    --exclude-user-ids data/evaluation/personalized/metadata/reserved_user_ids.txt

python scripts/evaluate_personalized.py `
    --models popularity,countsketch_cf,als `
    --exclude-user-ids data/evaluation/personalized/metadata/burned_user_ids.txt `
    --sample-seed 20260901 --max-evaluation-users 800 `
    --output-dir data/evaluation/personalized/results/als_confirmation_users800
```
