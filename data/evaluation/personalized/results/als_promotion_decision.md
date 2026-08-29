# ALS Promotion Decision: Complete Evidence

## Decision

**Do not promote ALS as a drop-in replacement for CountSketch. Adopt
segment-aware routing instead, and simplify the architecture.**

ALS is decisively better at what it does, and decisively narrower. The evidence
does not support a single global substitution, and it does not support keeping
the current hybrid either.

Three findings drive this:

1. **ALS wins enormously on relevance, robustly across every relevance
   definition tested** (+59% to +87% relative NDCG@10 at thresholds 9, 8, 7;
   every interval excludes zero).
2. **ALS recommends head items exclusively.** In the balanced popularity
   diagnostic its mid-tail and long-tail exposure are both **0.0000**, versus
   CountSketch's 0.0508 and 0.0097. CountSketch's small tail recall (0.0113
   mid-tail) drops to exactly zero under ALS. Aggregate catalog coverage
   *improves*, which conceals this completely.
3. **The hybrid no longer earns its cost.** Hybrid-ALS scores NDCG@10 0.2629
   against standalone ALS at 0.2624 — a +0.2% difference whose interval
   includes zero — for **125× the latency** (947 ms versus 7.6 ms).

## 1. Frozen configuration

See [FROZEN_als_reference.md](../FROZEN_als_reference.md). 128 factors, alpha
5.0, regularization 0.05, 15 iterations, 3 CG steps, seed 42, trained on train
positives only. Artifact `a0be5f3f…`, split `a668114f…`, catalog `2ef54a71…`.
Threshold 7 and 9 variants use identical hyperparameters on their own splits.

## 2. Threshold sensitivity

800 confirmation users per threshold, disjoint from all previously scored users.

| Threshold | Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | Coverage | Novelty | Pop. bias | ILD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | random | 0.0012 | 0.0005 | 0.0125 | 0.0012 | 0.0014 | 0.5881 | 19.381 | −0.4969 | 0.9107 |
| 7 | popularity | 0.1171 | 0.0843 | 0.5062 | 0.1176 | 0.1182 | 0.0051 | 8.693 | 0.1284 | 0.8164 |
| 7 | CountSketch | 0.1531 | 0.1198 | 0.6212 | 0.1551 | 0.1639 | 0.0859 | 10.109 | 0.0455 | 0.8268 |
| 7 | **ALS** | **0.2862** | **0.2332** | **0.8350** | **0.3051** | **0.3372** | 0.0871 | 10.234 | 0.0382 | 0.7966 |
| 7 | oracle | 1.0000 | 0.7823 | 1.0000 | 1.0000 | 0.9291 | 0.1374 | 11.798 | −0.0533 | 0.8147 |
| 8 | random | 0.0008 | 0.0009 | 0.0075 | 0.0010 | 0.0012 | 0.5869 | 19.692 | −0.5343 | 0.9102 |
| 8 | popularity | 0.1087 | 0.1000 | 0.4713 | 0.1188 | 0.1442 | 0.0054 | 8.420 | 0.1313 | 0.8230 |
| 8 | CountSketch | 0.1516 | 0.1430 | 0.5775 | 0.1634 | 0.1993 | 0.0782 | 9.729 | 0.0540 | 0.8252 |
| 8 | **ALS** | **0.2624** | **0.2475** | **0.7788** | **0.2921** | **0.3518** | 0.0793 | 10.120 | 0.0309 | 0.8007 |
| 8 | oracle | 1.0000 | 0.8650 | 1.0000 | 1.0000 | 0.9674 | 0.1188 | 11.868 | −0.0723 | 0.8119 |
| 9 | random | 0.0003 | 0.0003 | 0.0025 | 0.0006 | 0.0013 | 0.5890 | 19.547 | −0.5629 | 0.9102 |
| 9 | popularity | 0.0924 | 0.1115 | 0.3588 | 0.1148 | 0.1786 | 0.0064 | 8.036 | 0.1351 | 0.8258 |
| 9 | CountSketch | 0.1668 | 0.1938 | 0.5325 | 0.1936 | 0.2720 | 0.0696 | 9.327 | 0.0568 | 0.8246 |
| 9 | **ALS** | **0.2652** | **0.2884** | **0.6963** | **0.2980** | **0.3853** | 0.0697 | 10.067 | 0.0119 | 0.8061 |
| 9 | oracle | 1.0000 | 0.9557 | 1.0000 | 1.0000 | 0.9915 | 0.0867 | 12.150 | −0.1144 | 0.8054 |

Random sits at the floor (0.0003–0.0012) and oracle at exactly 1.0000 NDCG@10,
which validates the metric implementation: a correct oracle must reach the
analytic ceiling, and it does.

### Paired bootstrap, ALS − CountSketch

| Threshold | Metric | Delta | Relative | 95% CI | Excludes 0 |
|---|---|---:|---:|---:|---|
| 7 | NDCG@10 | +0.1330 | +86.9% | [+0.1191, +0.1472] | yes |
| 7 | Recall@10 | +0.1134 | +94.7% | [+0.0991, +0.1278] | yes |
| 7 | NDCG@20 | +0.1501 | +96.8% | [+0.1378, +0.1622] | yes |
| 7 | Recall@20 | +0.1732 | +105.7% | [+0.1579, +0.1882] | yes |
| 8 | NDCG@10 | +0.1108 | +73.1% | [+0.0957, +0.1251] | yes |
| 8 | Recall@10 | +0.1046 | +73.1% | [+0.0877, +0.1201] | yes |
| 8 | NDCG@20 | +0.1287 | +78.7% | [+0.1149, +0.1418] | yes |
| 8 | Recall@20 | +0.1525 | +76.5% | [+0.1335, +0.1695] | yes |
| 9 | NDCG@10 | +0.0984 | +59.0% | [+0.0803, +0.1161] | yes |
| 9 | Recall@10 | +0.0946 | +48.8% | [+0.0722, +0.1160] | yes |
| 9 | NDCG@20 | +0.1044 | +53.9% | [+0.0863, +0.1218] | yes |
| 9 | Recall@20 | +0.1133 | +41.7% | [+0.0891, +0.1376] | yes |

**Robust.** The advantage narrows monotonically as the relevance definition
tightens (+86.9% → +73.1% → +59.0%) but never approaches zero. This is the
opposite of LightFM, whose lift vanished at threshold 9. Threshold 9 is not a
veto here.

## 3. Achievable Recall@10 (interpretability diagnostic only)

`oracle_recall_at_10 = min(K, n_test_positives) / n_test_positives`

| K | Mean achievable recall | Users capped by K |
|---|---:|---:|
| 10 | 0.8650 | 32.0% |
| 20 | 0.9674 | 11.2% |

| Model | Recall@10 | Normalized (diagnostic) |
|---|---:|---:|
| CountSketch | 0.1430 | 0.1673 |
| ALS | 0.2475 | 0.2868 |

Standard Recall@10 remains the reported metric. Normalized recall is shown only
so the raw value is not misread as "26% of achievable" when it is 29%.

## 4. Activity segmentation

**Natural population** (threshold 8 confirmation, 800 users):

| Segment | Users | CS NDCG@10 | ALS NDCG@10 | Rel | CS R@10 | ALS R@10 | CS N@20 | ALS N@20 | 95% CI | Excl 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sparse | 30 | 0.1392 | 0.1734 | +24.6% | 0.2667 | 0.3333 | 0.1470 | 0.1824 | [−0.0736, +0.1387] | no |
| Medium | 100 | 0.1714 | 0.2501 | +45.9% | 0.2200 | 0.3400 | 0.1883 | 0.2779 | [+0.0224, +0.1375] | yes |
| Heavy | 670 | 0.1492 | 0.2682 | +79.8% | 0.1259 | 0.2299 | 0.1604 | 0.2991 | [+0.1046, +0.1337] | yes |

**Activity-balanced diagnostic** (100 per segment), because n=30 sparse users in
the natural population cannot support a conclusion:

| Segment | Users | CS NDCG@10 | ALS NDCG@10 | Rel | CS R@10 | ALS R@10 | CS N@20 | ALS N@20 | 95% CI | Excl 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sparse | 100 | 0.1611 | 0.1836 | +13.9% | 0.2400 | 0.2900 | 0.1708 | 0.2037 | [−0.0259, +0.0732] | **no** |
| Medium | 100 | 0.1714 | 0.2501 | +45.9% | 0.2200 | 0.3400 | 0.1883 | 0.2779 | [+0.0224, +0.1375] | yes |
| Heavy | 100 | 0.1347 | 0.2666 | +97.9% | 0.1086 | 0.2261 | 0.1507 | 0.2971 | [+0.0898, +0.1715] | yes |

**The gain is strongly monotonic in user activity.** Sparse users show a positive
point estimate whose interval includes zero at n=100 in both samples: there is
**no demonstrated improvement and no regression** for sparse users. Medium and
heavy users improve decisively. This is a latent-factor model behaving exactly as
theory predicts — it needs interactions to place a user in factor space.

## 5. Item popularity and long-tail

**Balanced diagnostic** (quota 100 per bucket; 3,906 head / 792 mid-tail / 176
long-tail held-out items — adequate, unlike the natural population's 4 long-tail
users):

| Model | Bucket | Users | Items | Recall@10 | NDCG@10 | Exposure |
|---|---|---:|---:|---:|---:|---:|
| CountSketch | head | 118 | 3906 | 0.0604 | 0.1742 | 0.9395 |
| CountSketch | mid-tail | 100 | 792 | **0.0113** | 0.0144 | **0.0508** |
| CountSketch | long-tail | 100 | 176 | 0.0000 | 0.0000 | **0.0097** |
| **ALS** | head | 118 | 3906 | **0.1288** | **0.3083** | **1.0000** |
| **ALS** | mid-tail | 100 | 792 | **0.0000** | 0.0000 | **0.0000** |
| **ALS** | long-tail | 100 | 176 | 0.0000 | 0.0000 | **0.0000** |

| Model | top 1% | top 5% | top 10% | top 20% | Unique items | Gini | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch | 0.4622 | 0.7790 | 0.8618 | 0.9395 | 771 | 0.9816 | 0.0427 |
| ALS | 0.3109 | 0.7895 | 0.9756 | **1.0000** | 1052 | 0.9631 | 0.0582 |

**This is the finding that blocks a global substitution.** ALS places **100% of
its recommendation exposure inside the top 20% most popular items**. It is less
concentrated *within* the head than CountSketch (top-1% share 0.31 vs 0.46, Gini
0.963 vs 0.982, 1,052 unique items vs 771) — which is why aggregate coverage,
novelty, and popularity bias all *improve*. But it never leaves the head, and it
zeroes out the small mid-tail recall CountSketch had.

Reading catalog coverage alone would have produced exactly the wrong conclusion.

## 6. Diversity guardrail

| | CountSketch | ALS | Absolute | Relative |
|---|---:|---:|---:|---:|
| Intra-list diversity | 0.8252 | 0.8007 | −0.0245 | **−2.97%** |
| NDCG@10 | 0.1516 | 0.2624 | +0.1108 | +73.1% |

NDCG@10 gain 95% CI [+0.0957, +0.1251]. The trade is **4.5 NDCG points gained
per ILD point lost**. The regression is consistent across thresholds (−3.6%,
−3.0%, −2.2% at 7/8/9), so it is a stable property rather than noise.

**Verdict: acceptable.** A 3% intra-list diversity loss against a 73% relevance
gain is not a reason to reject, and it is cheaply correctable — the recommender
already has a greedy diversity reranker whose zero-diversity fast path is now
O(n) rather than O(n²), so applying it over a small Top-N costs little.

## 7. Weighted-confidence ALS

Validation-only mapping selection, 800 validation users, disjoint from all
confirmation samples. Frozen hyperparameters; only the confidence mapping varies.

| Mapping | Weights (8/9/10) | Val NDCG@10 | Recall@10 | HR@10 | Coverage | Pop. bias |
|---|---|---:|---:|---:|---:|---:|
| **binary** | 1 / 1 / 1 | **0.2904** | **0.2721** | **0.8037** | 0.0764 | 0.0284 |
| log | 1 / 1.585 / 2 | 0.2890 | 0.2701 | 0.7950 | 0.0778 | 0.0290 |
| sqrt | 1 / 1.414 / 1.732 | 0.2888 | 0.2684 | 0.7987 | 0.0776 | 0.0286 |
| linear | 1 / 2 / 3 | 0.2863 | 0.2672 | 0.7887 | **0.0790** | 0.0301 |

**Negative result: preserving rating magnitude does not help.** Binary wins, and
the ordering is monotonic — the more aggressively intensity is encoded, the worse
ranking gets, while coverage marginally improves. No confirmation run was needed
because the selected mapping *is* the already-confirmed frozen reference.

Explicit negatives and neutral ratings were never given negative confidence; they
remain excluded as known items at ranking time. Weights are strictly positive by
construction.

**A methodology note that matters.** The first version of this experiment
reported binary and linear as byte-identical. That was a bug, not a finding: the
conjugate-gradient solver read only the sparse matrix's column indices and never
its values, so every mapping trained the same model. It is fixed, and
`test_confidence_mapping_actually_changes_the_model` now fails if the weights
stop reaching the solver. Without the "identical numbers are suspicious" check,
this would have been published as a clean negative result for the wrong reason.

## 8. Candidate retrieval

300 users, held-out positives as targets, full-catalog candidate generation.

| Source | Recall@100 | Recall@300 | Recall@500 | Recall@1000 | p50 ms |
|---|---:|---:|---:|---:|---:|
| CountSketch | 0.3839 | 0.5499 | 0.6152 | 0.7077 | 5.14 |
| **ALS** | **0.6226** | **0.7932** | 0.8500 | 0.9027 | 9.71 |
| item-item | 0.3947 | 0.6691 | 0.8009 | 0.8983 | 6.56 |
| content | 0.3196 | 0.5047 | 0.5968 | 0.7485 | 730.74 |
| ALS + CountSketch | 0.6048 | 0.7813 | 0.8558 | 0.9240 | 14.93 |
| ALS + item-item | 0.6211 | 0.7978 | **0.8777** | **0.9438** | 16.44 |
| ALS + content + item-item | 0.5704 | 0.7640 | 0.8497 | 0.9277 | 747.60 |

By popularity bucket at depth 300: ALS head 0.8100 / mid-tail 0.0415 / long-tail
0.0000; CountSketch head 0.5577 / mid-tail 0.0314 / long-tail 0.0000; item-item
head 0.6800 / mid-tail 0.0556 / **long-tail 0.6667**.

**ALS improves Top-300 candidate recall by +44% relative** (0.5499 → 0.7932),
which directly addresses the fusion phase's finding that 43% of held-out
positives never reached the shortlist. Unions add little over ALS alone at
shallow depths and cost roughly double the latency; ALS + item-item is the only
union that helps at depth ≥ 500, and it is also the only source with any
long-tail reach.

**Content retrieval is not competitive**: worse recall than ALS at every depth
and 75× the latency.

## 9. Hybrid substitution

Controlled: identical users, split, catalog, filters, content channels, hand-set
weights, semantic weight 0, quality statistics, diversity settings. The **only**
change is the collaborative channel.

| Model | NDCG@10 | R@10 | HR@10 | NDCG@20 | R@20 | Coverage | Novelty | Pop. bias | ILD | p50 ms | p95 ms | Artifact MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch | 0.1516 | 0.1430 | 0.5775 | 0.1634 | 0.1993 | 0.0782 | 9.729 | 0.0540 | 0.8252 | 7.78 | 9.46 | 26.9 |
| **ALS standalone** | **0.2624** | **0.2475** | **0.7788** | 0.2921 | **0.3518** | 0.0793 | 10.120 | 0.0309 | 0.8007 | **7.55** | 9.73 | **9.0** |
| Hybrid + CountSketch | 0.1815 | 0.1682 | 0.6312 | 0.2000 | 0.2395 | 0.0563 | 9.759 | 0.0522 | 0.7396 | 947.77 | 1500.58 | 43.5 |
| **Hybrid + ALS** | **0.2629** | 0.2434 | 0.7762 | **0.2922** | 0.3485 | 0.0772 | 10.148 | **0.0292** | 0.7598 | 947.22 | 1476.55 | 16.6 |

**Hybrid-ALS − Hybrid-CountSketch:**

| Metric | Delta | Relative | 95% CI | Excludes 0 |
|---|---:|---:|---:|---|
| NDCG@10 | +0.0814 | +44.9% | [+0.0700, +0.0929] | yes |
| Recall@10 | +0.0752 | +44.7% | [+0.0618, +0.0879] | yes |
| NDCG@20 | +0.0921 | +46.1% | [+0.0818, +0.1023] | yes |
| Recall@20 | +0.1090 | +45.5% | [+0.0939, +0.1225] | yes |

ALS's advantage **survives embedding**. The hybrid's other nine channels dilute
it (73% standalone → 45% inside the hybrid) but do not erase it.

## 10. ALS standalone versus Hybrid-ALS

Same users, same protocol.

| Metric | ALS standalone | Hybrid-ALS | Delta | Relative | 95% CI | Excludes 0 |
|---|---:|---:|---:|---:|---:|---|
| NDCG@10 | 0.2624 | 0.2629 | +0.0005 | +0.2% | [−0.0069, +0.0077] | **no** |
| Recall@10 | 0.2475 | 0.2434 | −0.0041 | −1.7% | [−0.0128, +0.0043] | no |
| NDCG@20 | 0.2921 | 0.2922 | +0.0001 | +0.0% | [−0.0059, +0.0063] | no |
| Recall@20 | 0.3518 | 0.3485 | −0.0033 | −0.9% | [−0.0131, +0.0067] | no |

**Accuracy per additional millisecond: 5.8 × 10⁻⁷ NDCG/ms.** The hybrid costs
**+939.7 ms (125×)** for a NDCG@10 difference statistically indistinguishable
from zero, and it *loses* on Recall.

The hybrid also costs coverage (0.0772 vs 0.0793), ILD (0.7598 vs 0.8007), and
2.6× the resident memory.

Once the collaborative channel is strong, the other nine channels are not adding
measurable ranking value on this benchmark. That is an architecture finding, not
a tuning one.

**Important scope limit:** this benchmark measures *unconstrained personalized
ranking only*. The hybrid also enforces hard filters, entity constraints, and
explainability, which this benchmark does not score at all. "The hybrid adds no
ranking value" is not the same as "the hybrid adds no product value."

## 11. Sampled-negative diagnostic

See [metric comparability](metric_comparability.md). Same model, same users:

| Protocol | NDCG@10 |
|---|---:|
| **Primary selection protocol** — all test positives, full 18,064-item catalog | **0.2875** |
| Diagnostic — leave-one-out, full catalog | 0.1775 |
| **Comparability diagnostic only** — leave-one-out + 99 sampled negatives | 0.8260 |

No model in this phase was selected using the sampled-negative protocol.

## 12. Sanity references

| Model | NDCG@10 (threshold 8) | Role |
|---|---:|---|
| Random | 0.0008 | floor — validates the metric is not inflated |
| Popularity | 0.1087 | non-personalized baseline |
| CountSketch | 0.1516 | current production |
| ALS | 0.2624 | challenger |
| Oracle | 1.0000 | analytic ceiling — validates the metric implementation |

Oracle reaching exactly 1.0000 NDCG@10 and 0.8650 Recall@10 (the computed
achievable ceiling) confirms the metric code is correct. Oracle is not
deployable: it reads held-out labels.

## 13. Answers

| # | Question | Answer |
|---|---|---|
| 1 | ALS beats CountSketch at threshold 7? | **Yes**, +86.9% NDCG@10, CI excludes 0 |
| 2 | At threshold 8? | **Yes**, +73.1%, CI excludes 0 |
| 3 | At threshold 9? | **Yes**, +59.0%, CI excludes 0 |
| 4 | Improves sparse users? | **No demonstrated improvement, no regression.** +13.9% point estimate, CI includes 0 at n=100 |
| 5 | Improves medium users? | **Yes**, +45.9%, CI excludes 0 |
| 6 | Improves heavy users? | **Yes**, +97.9%, CI excludes 0 |
| 7 | Improves head retrieval? | **Yes**, Recall@300 head 0.5577 → 0.8100 |
| 8 | Improves mid-tail retrieval? | **No.** Balanced diagnostic recall 0.0113 → 0.0000; exposure 0.0508 → 0.0000 |
| 9 | Improves long-tail retrieval? | **No.** Both 0.0000 recall; exposure 0.0097 → 0.0000 |
| 10 | Is the ILD regression acceptable? | **Yes**, −2.97% against +73% relevance; 4.5 NDCG points per ILD point |
| 11 | Weighted-confidence improves binary? | **No.** Binary wins; all mappings monotonically worse |
| 12 | Materially improves Top-300 candidate recall? | **Yes**, +44% relative (0.5499 → 0.7932) |
| 13 | Improves the complete Hybrid? | **Yes**, +44.9% NDCG@10, CI excludes 0 |
| 14 | Does the Hybrid justify its latency vs standalone ALS? | **No.** +0.2% NDCG (CI includes 0) for 125× latency |
| 15 | Keep CountSketch as fallback / memory baseline? | **Yes**, as the tail-exposure and sparse-user arm, not merely as a fallback |
| 16 | Promote ALS to production? | **Not as a global replacement.** Promote via segment-aware routing |
| 17 | Is LightGCN justified? | **No.** See below |

## 14. Production recommendation

**Segment-aware routing, plus architectural simplification.**

- **Use ALS as the primary collaborative channel and as the candidate retriever**
  for medium and heavy users. That is where +46% to +98% relevance lives, and it
  raises Top-300 retrieval recall 44%.
- **Retain CountSketch** for tail exposure. It is the only cheap source that puts
  any exposure outside the head (5.1% mid-tail, 1.0% long-tail). If catalog
  discovery is a product goal, ALS alone regresses it to zero.
- **Do not route sparse users to ALS on current evidence.** The interval includes
  zero. Either keep CountSketch for them or run a larger sparse-user experiment
  before deciding.
- **Retrieve with ALS + item-item** if tail reach matters: it is the only union
  with long-tail recall (0.6667 at depth 300) and costs 16 ms.
- **Reconsider the hybrid.** It costs 125× latency for no measurable ranking gain
  once ALS is the collaborative channel. Keep it only for the capabilities this
  benchmark does not measure — hard filters, entity constraints, explanations —
  and consider running it *only* when a request carries such constraints, with
  ALS serving unconstrained recommendations directly.

Both models export framework-independent NumPy artifacts; ALS is 9.0 MB against
CountSketch's 26.9 MB, and neither adds a compiled runtime dependency to FastAPI.

## 15. Is LightGCN justified?

**No.** Not because it is unfashionable, and not because ALS is finished — but
because the concrete unresolved problems are not ones graph propagation solves.

The two real gaps are:

1. **Zero tail exposure.** ALS gives the entire top-20% of popularity 100% of
   exposure. LightGCN is a smoothing operator over the same interaction graph;
   propagation concentrates mass along high-degree paths and is, if anything,
   *more* popularity-biased than MF unless explicitly debiased. There is no
   reason to expect it to recover long-tail items that have almost no edges.
   The mechanisms that address this are exposure-aware reranking, popularity
   debiasing, and content-based retrieval for cold items — none of which is
   LightGCN.
2. **Sparse users show no significant gain.** This *is* the regime where graph
   propagation is theoretically motivated: a 1–4 interaction user benefits from
   multi-hop neighbourhood signal. But sparse users are **2.2% of the eligible
   population** here, and the current result is "no gain, no harm" rather than a
   failure. Adopting a GPU-dependent architecture, breaking the NumPy-only
   serving property, for a non-regressing 2.2% segment is not a defensible
   trade.

**The concrete condition that would justify it:** if a sparse/cold-start-focused
experiment showed ALS materially *regressing* against CountSketch for users with
1–4 interactions, on an adequately powered sample, **and** content-based and
item-item retrieval failed to close that gap, then multi-hop propagation would be
addressing a demonstrated failure mode rather than a hypothetical one.

The higher-value direction remains interaction timestamps. Without them this is
set completion, not next-item prediction, and no architecture recovers
information the data does not contain.

## 16. Methodological caveats

- **Sample disjointness.** Every confirmation sample excludes all previously
  scored users; disjointness is asserted at sample time and recorded in each
  manifest as `sample_is_disjoint_from_excluded`. 3,706 users are now burned.
- **Threshold samples are not shared.** Thresholds 7/8/9 have different eligible
  populations, so each has its own 800-user sample. Cross-threshold comparisons
  are between-sample, not paired. Within each threshold, ALS vs CountSketch is
  properly paired.
- **The threshold-8 number here (0.2624) differs from the earlier ALS
  confirmation (0.2875)** because the exclusion set grew between runs, producing
  a different sample. Both are valid independent confirmations; the paired
  comparison against CountSketch is consistent (+73.1% vs +75.2%).
- **Long-tail sample sizes.** The natural population had only 4 long-tail users
  and 4 held-out long-tail items — uninterpretable. All long-tail conclusions
  come from the balanced diagnostic (100 users, 176 items).
- **Balanced diagnostics are not population estimates.** Their unweighted
  aggregates must not be read as expected production performance.
- **Retrieval was measured on 300 users**, fewer than the ranking experiments.
- **One split family, one seed, one dataset snapshot ending 2020.**
- **No timestamps**, so this is preference reconstruction, not next-item
  prediction.
- **The hybrid comparison scores ranking only.** Hard filters, entity
  constraints, and explanation quality are not measured, and the hybrid exists
  substantially to provide them.
- **A split artifact was silently overwritten during this phase** (threshold 7
  rebuilt as threshold 8 by a CLI that defaulted the threshold). It was detected,
  regenerated, and verified against its recorded metadata; the CLI now refuses to
  overwrite a mismatched split unless `--force-split` is passed.

## 17. Reproduce

```powershell
# Frozen ALS reference and threshold variants (identical hyperparameters)
python scripts/sweep_als_validation.py --grid data/evaluation/personalized/configs/als_sweep.json `
    --exclude-user-ids data/evaluation/personalized/metadata/reserved_user_ids.txt
python scripts/sweep_als_validation.py --grid data/evaluation/personalized/configs/als_sweep_round2.json `
    --output-dir data/evaluation/personalized/results/als_validation_sweep_round2 `
    --exclude-user-ids data/evaluation/personalized/metadata/reserved_user_ids.txt

# Threshold sensitivity (7 / 8 / 9)
python scripts/evaluate_personalized.py --positive-threshold 7 `
    --split data/evaluation/personalized/splits/holdout_seed42_pos7.sqlite `
    --artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos7 `
    --output-dir data/evaluation/personalized/results/threshold7_confirmation_users800 `
    --models random,popularity,countsketch_cf,als,oracle `
    --exclude-user-ids data/evaluation/personalized/metadata/burned_user_ids.txt `
    --sample-seed 20260901 --max-evaluation-users 800

# Activity- and popularity-balanced diagnostics
python scripts/evaluate_personalized.py --models countsketch_cf,als `
    --sampling-strategy activity_stratified --users-per-stratum 100 `
    --output-dir data/evaluation/personalized/results/als_activity_balanced `
    --exclude-user-ids data/evaluation/personalized/metadata/burned_user_ids.txt --sample-seed 20260901
python scripts/evaluate_personalized.py --models countsketch_cf,als `
    --sampling-strategy popularity_stratified --users-per-stratum 100 `
    --output-dir data/evaluation/personalized/results/als_popularity_balanced `
    --exclude-user-ids data/evaluation/personalized/metadata/burned_user_ids.txt --sample-seed 20260901

# Weighted-confidence ALS (validation only)
python scripts/sweep_als_validation.py --grid data/evaluation/personalized/configs/als_confidence.json `
    --output-dir data/evaluation/personalized/results/als_confidence_sweep `
    --exclude-user-ids data/evaluation/personalized/metadata/burned_user_ids.txt

# Candidate retrieval frontier
python scripts/evaluate_candidate_retrieval.py --users 300 --include-content `
    --exclude-user-ids data/evaluation/personalized/metadata/burned_user_ids.txt

# Hybrid substitution and standalone-vs-hybrid
python scripts/evaluate_personalized.py `
    --models countsketch_cf,als,current_hybrid,current_hybrid_als --semantic-artifact "" `
    --output-dir data/evaluation/personalized/results/hybrid_substitution_users800 `
    --exclude-user-ids data/evaluation/personalized/metadata/burned_user_ids.txt `
    --sample-seed 20260901 --max-evaluation-users 800

# Protocol comparability diagnostic (never used for selection)
python scripts/compare_evaluation_protocols.py
```
