# Linear or LambdaMART: Production Selection

## Decision

**LambdaMART.** Its advantage over the linear reranker is paired-significant on
three of four metrics, it is the only arm that preserves catalog coverage, and
its *incremental* cost over the linear model is small next to the cost both
options share.

## 1. The comparison that had not been run

The previous phase compared each reranker against ALS separately and inferred
that 0.2828 beat 0.2725. That is a difference of differences, not a paired test.
Run directly on the same 800 frozen confirmation users:

| Metric | Linear | LambdaMART | Δ | 95% CI | Verdict |
|---|---:|---:|---:|---|---|
| NDCG@10 | 0.2725 | 0.2828 | **+0.0103** | `[+0.0027, +0.0177]` | **significant** |
| Recall@10 | 0.2441 | 0.2537 | **+0.0096** | `[+0.0010, +0.0184]` | **significant** |
| NDCG@20 | 0.3011 | 0.3099 | **+0.0087** | `[+0.0022, +0.0150]` | **significant** |
| Recall@20 | 0.3519 | 0.3592 | +0.0073 | `[-0.0011, +0.0153]` | includes zero |

The gap is real, and it is modest: +3.8% relative NDCG@10. The two rerankers
agree on **76.7%** of top-10 slots, so they are genuinely different orderings
rather than one being a noisy copy of the other.

## 2. End-to-end serving latency

The previous phase reported `model.predict()` alone — 0.028 ms and 0.278 ms.
That was the cheapest stage of the pipeline and gave a misleading picture.
Measured across the complete path, interleaved so drift affects all arms equally:

**Demo-like profile (3 liked titles)**

| Stage | ALS only | + Linear | + LambdaMART |
|---|---:|---:|---:|
| ALS retrieval | 0.409 | 0.383 | 0.368 |
| Candidate selection | 0.111 | 0.101 | 0.114 |
| **Feature construction** | — | **4.650** | **4.793** |
| Reranker inference | — | 0.029 | 0.296 |
| Sort | — | 0.039 | 0.045 |
| **Total p50** | **0.528** | **5.286** | **5.725** |
| Total p95 | 0.965 | 8.535 | 9.469 |

**Evaluation-user profile (mean 86 liked titles)**

| Stage | ALS only | + Linear | + LambdaMART |
|---|---:|---:|---:|
| **Feature construction** | — | **14.570** | **13.738** |
| Reranker inference | — | 0.041 | 0.318 |
| **Total p50** | **0.613** | **15.278** | **14.945** |
| Total p95 | 0.975 | 45.632 | 45.362 |

Two things fall out of this.

**Feature construction is the entire cost, and it is shared.** It is the same
code for both arms; the small differences above are measurement noise. It scales
with profile size, because the genre-Jaccard and item-item lookups run over
every profile title: 4.7 ms at three titles, 14.6 ms at eighty-six.

**The choice between the two models is worth 0.27 ms.** Inference is 0.03 ms
against 0.30 ms. That is 5% of a demo-like request and 2% of a heavy one.

## 3. Deployment cost

Shared by both options — this is what actually moves the deployment:

| Item | Cost |
|---|---|
| Feature-statistics artifact | **17.36 MB** compressed |
| — of which item-item neighbours | ~14 MB (18,064 × 200 int32 + float32) |
| Artifact load | +108 ms cold start |
| Deployment payload | 14.0 MB → **31.4 MB** (+124%) |

Differential, LambdaMART over Linear:

| Item | Linear | LambdaMART | Δ |
|---|---:|---:|---:|
| Model artifact | 1.5 KB | 541.4 KB | +540 KB |
| Extra dependencies | none | lightgbm | +4.7 MB installed |
| Import cost | 0 ms | 650 ms | +650 ms cold start |
| Model load | 0.02 ms | 8.9 ms | +8.9 ms |
| Per-request inference | 0.029 ms | 0.296 ms | +0.27 ms |

Streamlit Cloud compatibility: LightGBM 4.7 ships manylinux/win wheels and needs
no system packages, so it installs cleanly. The import is lazy, so a deployment
with the reranker disabled never pays the 650 ms.

## 4. Why LambdaMART

Three reasons, in order of weight.

**The gain is supported, not inferred.** Paired-significant on NDCG@10,
Recall@10, and NDCG@20 against the linear model on the same users.

**It preserves coverage where Linear degrades it.** From the previous phase's
exposure analysis: ALS shows 1,023 distinct items across top-10 slots, LambdaMART
1,021, Linear only 870. The linear model buys part of its gain by concentrating
15% harder into fewer titles. NDCG does not see that, and for a recommender it
matters.

**Simplicity would not have bought much here.** The tie-breaker argument assumes
the simpler model avoids the complexity. It does not: both options need the same
17.4 MB feature artifact and the same 4.7–14.6 ms feature construction. Choosing
Linear saves 5.2 MB out of 22.6 and 0.27 ms out of 5.7. The complexity lives in
the feature pipeline, not in the model.

The honest cost: this ends "NumPy-only serving, no ML framework in the web
process", which was a stated property of the project. The README now says so.

## 5. Integration

Wired into the fast path, off by default:

```text
profile → ALS top-300 → hard filters and exclusions → learned reranker
        → final exclusion re-check → top-N
```

Filtering runs before reranking rather than after, which is cheaper and strictly
safer: the reranker only ever sees candidates that already cleared every hard
constraint, so it cannot resurrect an excluded item. The final exclusion
re-check that predates this work still runs afterwards.

Preserved unchanged: the ALS-only fallback, the agent architecture, the
static/dynamic prompt separation, constraint behaviour, and artifact role and
hash validation. `RERANKER_ENABLED=false` is the default; the fast path serves
the ALS order whenever the reranker is absent, unverified, or throws.

Verified end to end through `reranker_serving.load_reranker` on the frozen
confirmation users: NDCG@10 0.2456 → **0.2828** and Recall@10 0.2312 → **0.2537**,
reproducing the offline numbers exactly.

One correctness detail worth recording. `ALSCollaborativeIndex.profile_scores`
normalises to [0, 1], but the reranker was fitted on raw scores and its trees
split on absolute thresholds. Serving normalised scores would have left
`als_score_z` intact and silently moved `als_score` onto thresholds the model
never learned. `raw_profile_scores` exists so serving matches training exactly.

## 6. Evaluation honesty

The published ALS benchmark stays at **0.2588**, from the 800-user architecture
comparison. The reranker numbers come from a different 800 users and are labelled
by experiment. The two are not combined, and no relative gain is computed across
them.

A headline "old Hybrid versus ALS + reranker" claim would require rerunning the
old Hybrid on these confirmation users. That has not been done, so no such claim
is made.

## Reproduce

```bash
python scripts/compare_rerankers.py
python scripts/build_reranker_artifact.py
```
