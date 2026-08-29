# Predeclaration: Learned Fusion Confirmation

**Written before any metric was computed on the confirmation sample.** The
sample IDs were drawn and frozen first; the decision rule below was fixed before
the fit was run.

## Why a predeclaration is needed here

The learned-fusion question has already been examined once, and the semantic
channel has been examined twice. Across all prior runs, **1,306 distinct users
have been inspected**. Reusing any of them would turn a "held-out" claim into a
restatement of numbers already seen. The confirmation sample is therefore drawn
from users no experiment in this repository has ever scored.

## Sample definition (frozen)

| Property | Value |
|---|---|
| Split | `data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite` |
| Positive threshold | `rating >= 8` |
| Sampling strategy | uniform (representative) |
| Confirmation seed | `20260828` |
| Confirmation size | 800 users |
| Excluded from the pool | the 1,306 IDs in `metadata/inspected_user_ids.txt` |
| Frozen sample | `metadata/confirmation_user_ids.txt` |
| Reserved from fitting | `metadata/reserved_user_ids.txt` (inspected ∪ confirmation, 2,106 IDs) |

Disjointness is asserted in code at sample time and recorded in each run's
`manifest.json` as `experiment.sample_is_disjoint_from_excluded`.

## Fitting protocol (frozen)

- Weights are fitted **only** on validation-holdout positives, from users drawn
  after removing all 2,106 reserved IDs.
- The confirmation users are never passed to `build_pairwise_dataset`.
- The semantic channel is active for the fit, via
  `--semantic-artifact data/processed/semantic_embeddings.npz` and
  `experimental_semantic_weights()` as the optimiser's starting point, so the
  channel has real variance to be judged on.
- Per-channel difference variance is recorded. Any channel reported in
  `zero_variance_channels` was unfittable, and its resulting weight carries no
  evidential value.

## Confirmation protocol (frozen)

One run of `scripts/evaluate_personalized.py` on the 800 confirmation users,
comparing exactly two models on identical users:

- `current_hybrid` — the shipped blend, semantic weight 0.00
- `current_hybrid_learned` — the fitted blend from the step above

Primary metric: **NDCG@10**. Secondary: Recall@10. Both with 2,000-resample
paired bootstrap intervals, seed 42.

## Decision rule (frozen, written before seeing the result)

Adopt the learned blend **only if all three hold**:

1. NDCG@10 improves and the paired 95% interval excludes zero.
2. Recall@10 does not regress with an interval excluding zero.
3. Catalog coverage does not fall by more than 10% relative.

Otherwise the hand-set blend is retained and the result is published as a
negative one. **The confirmation set is opened exactly once.** If the rule is not
met, the response is to report it, not to re-fit and re-open.

## Semantic-specific expectation

The prior fit drove the semantic weight to zero, but that run is not evidence:
the channel was inactive and had no variance. With the channel live, a fitted
weight near zero would be **independent** corroboration of the retirement
decision in `results/semantic_channel_summary.md`. A fitted weight materially
above zero would be evidence *against* that decision and would require revisiting
it rather than being explained away.
