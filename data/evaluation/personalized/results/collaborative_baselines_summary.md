# Reference Collaborative Baselines: Decision Report

> **Update — ALS passed its standalone confirmation.** After a validation-only
> hyperparameter sweep (12 candidates, two rounds) and a run on 800 predeclared
> users that no earlier experiment had scored, the tuned configuration
> (128 factors, alpha 5.0) beats the incumbent by **+75.2% relative NDCG@10**
> and **+76.3% relative Recall@10**, both intervals excluding zero, while
> holding coverage at 103% of the incumbent and reducing popularity bias.
> See [ALS confirmation](als_confirmation_summary.md).
>
> **One claim below is corrected by that work.** The statement that ALS wins
> accuracy and exposure with "no trade to adjudicate" was based on an *untuned*
> configuration. The sweep shows coverage falls monotonically with alpha
> (0.0947 at alpha=40 down to 0.0502 at alpha=1), so tuning for accuracy at 64
> factors *does* cost coverage. Raising factors to 128 recovers it. The trade is
> real; it is resolved by capacity, not absent.

## Decision

**Promote implicit ALS to the production collaborative channel, pending a
predeclared confirmation run.**

This reverses the previous conclusion in scope, not in method. The earlier
[LightFM report](lightfm_challenger_summary.md) correctly declined to promote
LightFM-ID: it bought accuracy at the cost of catalog coverage, popularity
neutrality, and sparse-user retrieval. That decision was made against an
incomplete comparison set. Two reference baselines were missing, and adding them
changes the answer.

ALS clears every gate LightFM failed. It is the most accurate model measured, it
has the **highest** catalog coverage of any model including the incumbent, and it
has the **lowest** popularity bias of any model including the incumbent. There is
no accuracy-versus-exposure trade to adjudicate, because ALS wins both.

## Why these two baselines were needed

The production channel, CountSketch, is not a learned model. It is a single
streaming pass that user-centres each rating vector and projects the sparse user
dimension into signed hash buckets. Dot products then approximate adjusted-cosine
item similarity. Nothing is optimised; the projection is a compression device.

That framing raises two questions the previous benchmark could not answer:

1. **What does the projection cost?** `item_item_cosine` computes the same
   similarity exactly, from an identical residual transform. The gap between the
   two is the sketching error alone, with no confounding change of model family.
2. **How does a standard latent-factor model compare?** `als` is implicit-feedback
   alternating least squares (Hu, Koren, and Volinsky, 2008) — the default
   baseline for this task. Its absence was the largest hole in the prior
   evidence: a challenger was rejected without the field's reference point on
   the board.

Neither baseline adds a compiled dependency. Both are implemented in NumPy and
SciPy sparse and exported as NumPy arrays, matching how every other benchmark
artifact is served. ALS uses the conjugate-gradient formulation (Takács et al.,
2011), because an exact per-user solve is O(f³) and prohibitive at 290k users.

## Experiment design

Design, split, and feedback handling are unchanged from the LightFM report, so
the two sets of results are directly comparable.

- Split: deterministic per-user random positive holdout, `rating >= 8`, seed 42.
- Evaluation: deterministic uniform 1,000-user representative sample.
- Candidate catalog, known-item filtering, metrics, and held-out users are
  identical across all six models.
- `item_item_cosine` trains on all observed train ratings, exactly as CountSketch
  does. `als` trains on train positives only, the standard implicit formulation
  and the same feedback LightFM received. Explicit negatives and neutral ratings
  are never folded into the unobserved mass; they are excluded as known items at
  ranking time.
- Because timestamps are unavailable, this measures preference reconstruction,
  not chronological next-item prediction.

## Overall comparison

Deterministic uniform 1,000-user sample, full-data train-only split:

| Model | NDCG@10 | Recall@10 | HR@10 | Coverage | Pop. bias | ILD | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1023 | 0.0881 | 0.4420 | 0.0059 | 0.1297 | 0.8222 | 0.22 |
| CountSketch CF (incumbent) | 0.1534 | 0.1408 | 0.5840 | 0.0775 | 0.0579 | **0.8264** | 6.93 |
| Exact item-item cosine | 0.1650 | 0.1472 | 0.6010 | 0.0375 | 0.0898 | 0.8259 | 5.72 |
| **ALS** | **0.1841** | **0.1908** | **0.6810** | **0.0989** | **0.0377** | 0.7985 | 6.87 |
| LightFM-ID | 0.1833 | 0.1589 | 0.6430 | 0.0380 | 0.0867 | 0.7990 | 1.19 |
| LightFM-Hybrid | 0.1747 | 0.1483 | 0.6250 | 0.0421 | 0.0775 | 0.7770 | 1.16 |

Lower popularity bias is better; it measures how far recommendations skew more
popular than the user's own training profile.

## Paired bootstrap comparisons

2,000 paired user-level resamples, seed 42, on the same 1,000 users.

| Comparison | Metric | Absolute | Relative | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---|
| item-item − CountSketch | NDCG@10 | +0.0115 | +7.5% | [+0.0036, +0.0194] | yes |
| item-item − CountSketch | Recall@10 | +0.0064 | +4.6% | [−0.0021, +0.0149] | no |
| ALS − CountSketch | NDCG@10 | +0.0307 | +20.0% | [+0.0172, +0.0446] | yes |
| ALS − CountSketch | Recall@10 | +0.0500 | +35.5% | [+0.0364, +0.0638] | yes |
| LightFM-ID − ALS | NDCG@10 | −0.0007 | −0.4% | [−0.0133, +0.0121] | no |
| LightFM-ID − ALS | Recall@10 | −0.0319 | −16.7% | [−0.0448, −0.0187] | yes |
| LightFM-Hybrid − ALS | NDCG@10 | −0.0094 | −5.1% | [−0.0225, +0.0042] | no |
| LightFM-Hybrid − ALS | Recall@10 | −0.0425 | −22.3% | [−0.0561, −0.0294] | yes |

## What this establishes

**The CountSketch projection costs real accuracy.** Exact item-item cosine, on
identical inputs with an identical transform, is +7.5% relative NDCG@10 better
and the interval excludes zero. The sketch is not free. Its justification has to
rest on its build-time memory profile, not on being lossless.

**But exactness is not the main story.** Exact item-item recovers only about a
third of the gap to ALS, and it *halves* catalog coverage (0.0375 versus 0.0775)
because concentrated top-K neighbour lists retrieve a narrower slice of the
catalog than the sketch's diffuse similarity. The incumbent's coverage advantage
came partly from the noise the projection introduces — a real property, but an
accidental one rather than a designed one.

**ALS dominates.** It is statistically indistinguishable from LightFM-ID on
NDCG@10 while being decisively better on Recall@10 (+16.7% relative, interval
excludes zero), and it beats the incumbent on every headline metric at once:

- Accuracy: +20.0% relative NDCG@10, +35.5% relative Recall@10, both significant.
- Coverage: 0.0989 versus 0.0775, a 27.6% improvement over the incumbent.
- Popularity bias: 0.0377 versus 0.0579, a 34.9% reduction.

The guardrails that blocked LightFM — halved coverage, higher popularity skew —
are not merely satisfied by ALS, they move in the favourable direction. Intra-list
diversity is the one metric that regresses (0.7985 versus 0.8264, −3.4%), which
is consistent with a latent-factor model producing more thematically coherent
lists. That is a real trade and is noted rather than dismissed.

**Serving cost is unchanged.** ALS exports item factors only; a user vector is
recomputed at request time by folding their positives into item space, so the
model generalises to users absent from training and to live sessions. p50 rank
latency is 6.87 ms against the incumbent's 6.93 ms, and the resident artifact is
smaller.

## Why this is not yet a promotion

The same discipline that governed the LightFM decision applies here.

- This is one representative sample at one positive threshold. The LightFM lift
  disappeared at threshold 9; ALS has not yet been tested at 7 or 9.
- ALS hyperparameters (64 factors, 15 iterations, λ=0.05, α=40) were set from
  standard defaults, not selected on validation users. A validation-only sweep
  should run before promotion, and the test set must not be reopened during it.
- Activity-stratified and item-popularity-stratified diagnostics have not been
  run for ALS. Sparse-user regression was a specific LightFM failure and must be
  checked explicitly, not assumed from the aggregate.
- Intra-list diversity regresses and should be quantified against a product
  threshold rather than accepted implicitly.

The predeclared confirmation is therefore: **a validation-only hyperparameter
sweep, threshold-7/8/9 sensitivity, and the two stratified diagnostics, with the
test set opened once at the end.** If ALS holds, it replaces CountSketch.

## Reproduce

```powershell
python -m pip install -r requirements-evaluation.txt
python scripts/evaluate_personalized.py `
    --split data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite `
    --artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8 `
    --output-dir data/evaluation/personalized/results/baselines_fulltrain_uniform1000 `
    --models popularity,countsketch_cf,item_item_cosine,als,lightfm_id,lightfm_hybrid `
    --max-evaluation-users 1000 --sampling-strategy uniform
```

Build cost on the full split, measured on this run: exact item-item took roughly
three minutes for the 18,064-item similarity pass over 50.6M residual entries;
ALS took roughly twenty minutes for 15 iterations over 289k users.

## Why not LightGCN or another graph model

The obvious next question after ALS is whether to reach for a graph convolution
model such as LightGCN, NGCF, or a two-tower neural retriever. The evidence here
argues against it, for four reasons specific to this dataset and this project.

**1. The measured gap is not where a graph model helps.** LightGCN's advantage
over matrix factorisation comes from propagating signal across a sparse
interaction graph, which matters most for users and items with few edges. This
dataset is not sparse in that regime: 24.9M train positives across 289k users and
18k items, a mean of 86 positives per user, and only 6,343 of 289,601 eligible
users (2.2%) in the sparse 1-to-4 band. Published LightGCN gains over well-tuned
MF on dense benchmarks are typically in the low single digits of relative NDCG,
and they shrink further when the MF baseline is properly regularised. ALS has not
yet been tuned at all here.

**2. ALS is not yet at its own ceiling.** Its hyperparameters are stock defaults
(64 factors, 15 iterations, λ=0.05, α=40). A validation-only sweep over factors,
regularisation, and α will plausibly recover more than a graph model would add,
at a fraction of the cost and with no new failure modes. Tuning the baseline
before adopting a more complex family is the cheaper experiment and the one that
makes any later comparison meaningful.

**3. The binding constraint is evidence, not capacity.** The open questions are
threshold sensitivity, sparse-user behaviour, and item-tail coverage — all
questions about *generalisation and exposure*, not about model expressiveness.
Adding capacity does not answer them; it adds a second thing to validate. The
LightFM episode already demonstrated that a more expressive model can win on
NDCG and still be the wrong choice.

**4. The engineering cost is real and asymmetric.** LightGCN needs a GPU to train
at this scale in reasonable time, or many hours on CPU. That reintroduces exactly
the dependency problem the LightFM work ran into on this platform, and it
conflicts with a stated project property: artifacts are served from NumPy arrays
with no framework in the web process. ALS and item-item both preserve that
property. A graph model would need either a PyTorch runtime dependency or a
separate export-and-validate path.

**What would change this.** A graph or sequential model becomes worth trying if
(a) tuned ALS plateaus well below LightFM-ID on a metric that matters, (b) the
sparse-user stratum turns out to be the dominant failure mode after the
diagnostics run, or (c) interaction timestamps become available. Point (c) is the
strongest: the current split is a random positive holdout, not a chronological
one, so *sequential* models (SASRec, GRU4Rec) cannot even be evaluated fairly
here. If timestamps arrive, a sequential model is a better next step than a graph
model, because it would exploit information the current benchmark cannot see at
all rather than re-encoding information ALS already captures.
