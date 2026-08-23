# LightFM Offline Challenger: Decision Report

## Decision

**Gather more evidence first; keep CountSketch as the production collaborative channel.**

LightFM-ID clears the primary threshold-8 accuracy gate on the representative 1,000-user sample: NDCG@10 is
0.1833 versus 0.1534 for CountSketch, a +19.51% relative lift with a paired 95% bootstrap interval of
[+0.0171, +0.0428] absolute. Recall@10 also improves by +0.0182 with an interval of [+0.0042, +0.0319].
Serving from exported NumPy arrays is fast.

It does not clear the complete promotion gate. Catalog coverage falls from 0.0775 to 0.0380, sparse-user NDCG@10
falls from 0.1319 to 0.0931 in the balanced diagnostic, and LightFM-ID/Hybrid recover none of the deliberately
sampled mid-tail or long-tail positives at 10. The lift also disappears at positive threshold 9. Metadata does
not repair these failures and lowers overall LightFM quality. Production code remains unchanged.

## Experiment design

- Primary feedback: positive `rating >= 8`, neutral `6..7`, explicit negative `<= 5`.
- Split: deterministic per-user random positive holdout, seed 42. Timestamps are unavailable, so this measures
  preference reconstruction/generalization rather than chronological next-item prediction.
- Training: positive training edges only for LightFM's standard implicit-ranking formulation. Explicit negative,
  neutral, and ignored ratings stay distinct and are excluded as known items during ranking; they are not silently
  converted to unobserved negatives.
- Selection: WARP and BPR were compared on validation positives only. The selected artifacts were opened on test
  positives once for this report.
- Evaluation A: deterministic uniform 1,000-user representative sample.
- Evaluation B: 100 sparse, 100 medium, and 100 heavy users; diagnostic only.
- Evaluation C: quota 100 for each head/mid-tail/long-tail stratum. Overlapping membership yields 119 unique users
  and selected memberships of 119 head, 100 mid-tail, and 100 long-tail; diagnostic only.
- Candidate catalog, known-item filtering, metrics, and held-out users are identical across models within each run.

## Overall collaborative comparison

Evaluation A — representative 1,000-user sample:

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty | Pop. bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1023 | 0.0881 | 0.4420 | 0.1149 | 0.1421 | 0.2153 | 0.0059 | 8.424 | 0.1297 | 0.8222 |
| CountSketch | 0.1534 | 0.1408 | 0.5840 | 0.1665 | 0.2005 | 0.2960 | **0.0775** | **9.639** | **0.0579** | **0.8264** |
| LightFM-ID | **0.1833** | **0.1589** | **0.6430** | **0.2025** | **0.2432** | **0.3480** | 0.0380 | 9.152 | 0.0867 | 0.7990 |
| LightFM-Hybrid | 0.1747 | 0.1483 | 0.6250 | 0.1917 | 0.2258 | 0.3227 | 0.0421 | 9.308 | 0.0775 | 0.7770 |

Positive popularity bias means recommendations skew more popular than the user's training profile. Both LightFM
variants are more popularity-biased than CountSketch. LightFM-ID's coverage is 51.0% lower than CountSketch's;
its ILD is 3.3% lower.

## Validation-only WARP/BPR screening

| Variant | Loss | Dimensions | Epochs | Fit time | Val NDCG@10 | Val Recall@10 | Val coverage | Val pop. bias |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 102.23s | **0.1988** | 0.1783 | 0.0275 | 0.0826 |
| LightFM-ID | BPR | 16 | 3 | 147.16s | 0.1980 | **0.1912** | **0.0612** | **0.0354** |
| LightFM-Hybrid | WARP | 16 | 3 | 265.00s | **0.1708** | **0.1611** | 0.0295 | 0.0735 |
| LightFM-Hybrid | BPR | 16 | 3 | 402.16s | 0.0765 | 0.0723 | **0.1033** | -0.2731 |

WARP was selected by the predeclared primary NDCG@10 rule. The ID-only WARP/BPR NDCG difference is only 0.0008,
while BPR has materially better validation Recall and coverage. This makes the chosen ID configuration fragile
and is one reason not to promote from this screen. The repository includes a larger validation-only search profile,
but it was not run after test inspection because doing so would turn the test set into an iterative tuning target.

## Paired bootstrap comparisons

Evaluation A, 2,000 paired user-level bootstrap resamples:

| Comparison | Metric | Absolute delta | Relative delta | 95% CI | Evidence in this evaluation |
|---|---|---:|---:|---:|---|
| LightFM-ID − CountSketch | NDCG@10 | +0.0299 | +19.51% | [+0.0171, +0.0428] | interval excludes 0 |
| LightFM-ID − CountSketch | Recall@10 | +0.0182 | +12.90% | [+0.0042, +0.0319] | interval excludes 0 |
| LightFM-Hybrid − CountSketch | NDCG@10 | +0.0213 | +13.86% | [+0.0074, +0.0351] | interval excludes 0 |
| LightFM-Hybrid − CountSketch | Recall@10 | +0.0075 | +5.32% | [-0.0067, +0.0215] | inconclusive |
| LightFM-Hybrid − LightFM-ID | NDCG@10 | -0.0087 | -4.73% | [-0.0184, +0.0008] | inconclusive |
| LightFM-Hybrid − LightFM-ID | Recall@10 | -0.0107 | -6.72% | [-0.0212, -0.0003] | evidence of a decrease |

## User activity

Evaluation B — activity-balanced diagnostic, exactly 100 users per training-positive activity bucket:

| Segment | Model | NDCG@10 | Recall@10 | HR@10 |
|---|---|---:|---:|---:|
| Sparse | CountSketch | **0.1319** | **0.2000** | **0.2000** |
| Sparse | LightFM-ID | 0.0931 | 0.1500 | 0.1500 |
| Sparse | LightFM-Hybrid | 0.0866 | 0.1400 | 0.1400 |
| Medium | CountSketch | **0.1242** | **0.1900** | 0.3100 |
| Medium | LightFM-ID | 0.1144 | 0.1850 | **0.3200** |
| Medium | LightFM-Hybrid | 0.1165 | 0.1550 | 0.2600 |
| Heavy | CountSketch | 0.1510 | 0.1247 | 0.6300 |
| Heavy | LightFM-ID | **0.2051** | **0.1469** | **0.7400** |
| Heavy | LightFM-Hybrid | 0.1870 | 0.1447 | 0.7200 |

The gain is concentrated among heavy users. Metadata does not help sparse users: LightFM-Hybrid is below both
LightFM-ID and CountSketch. The balanced aggregate is intentionally not interpreted as a population estimate.

## Item popularity

Evaluation C — popularity-stratified diagnostic. Buckets are derived from training-positive counts only:

| Bucket | Model | Qualifying users | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---|---:|---:|---:|---:|---:|
| Head | CountSketch | 119 | 3,856 | 0.0778 | 0.1567 | 0.9584 |
| Head | LightFM-ID | 119 | 3,856 | 0.0859 | **0.2202** | 1.0000 |
| Head | LightFM-Hybrid | 119 | 3,856 | **0.0925** | 0.2098 | 0.9979 |
| Mid-tail | CountSketch | 100 | 1,211 | **0.0026** | **0.0033** | 0.0311 |
| Mid-tail | LightFM-ID | 100 | 1,211 | 0.0000 | 0.0000 | 0.0000 |
| Mid-tail | LightFM-Hybrid | 100 | 1,211 | 0.0000 | 0.0000 | 0.0021 |
| Long-tail | CountSketch | 100 | 704 | **0.0101** | **0.0072** | 0.0105 |
| Long-tail | LightFM-ID | 100 | 704 | 0.0000 | 0.0000 | 0.0000 |
| Long-tail | LightFM-Hybrid | 100 | 704 | 0.0000 | 0.0000 | 0.0000 |

The diagnostic successfully increases tail evidence, but LightFM does not exploit it. Metadata improves neither
mid-tail nor long-tail recovery. A true new-item simulation was not forced: item-disjoint retraining and a separate
content-capable user/profile design are needed to make that experiment leakage-safe and interpretable.

## Threshold sensitivity

Each row is a separate 500-user representative run. The threshold-8-selected LightFM-ID WARP-16 configuration is
held fixed; no sensitivity result is used for tuning. Cohorts and relevance definitions change across rows, so only
within-row model deltas should be compared.

| Positive threshold | Eligible users | Train-positive edges | Matrix sparsity | CountSketch NDCG@10 | LightFM-ID NDCG@10 | Δ NDCG | 95% CI | CountSketch Recall@10 | LightFM-ID Recall@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 291,904 | 35,426,351 | 0.993651 | 0.1468 | 0.1889 | +0.0421 | [+0.0247, +0.0593] | 0.1143 | 0.1388 |
| 8 | 289,601 | 24,916,911 | 0.995524 | 0.1567 | 0.1948 | +0.0381 | [+0.0217, +0.0561] | 0.1484 | 0.1669 |
| 9 | 280,257 | 13,257,591 | 0.997600 | 0.1788 | 0.1793 | +0.0004 | [-0.0185, +0.0208] | 0.2120 | 0.2069 |

The LightFM-ID advantage is not robust to the strictest positive definition. At threshold 9, both NDCG and Recall
intervals include zero; the Recall point estimate is lower by 0.0051.

## Engineering comparison

Times are from the threshold-8 full training graph. LightFM offline total includes both WARP and BPR validation
candidates; selected fit is the winning WARP fit. Inference is NumPy-only and excludes HTTP, LLM, and rendering.
Peak RSS is whole trainer-process peak, not an isolated incremental allocation.

| Model | Selected fit/build | Offline total | Peak RSS | p50 | p95 | Array memory | Artifact |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch | 65.77s | 65.77s | 749.69 MiB | 7.28ms | 8.18ms | 26.87 MiB | 14.27 MiB |
| LightFM-ID | 102.23s | 297.61s | 1,339.75 MiB | 1.17ms | 1.34ms | 23.65 MiB | 19.97 MiB |
| LightFM-Hybrid | 265.00s | 715.28s | 1,345.67 MiB | 1.16ms | 1.31ms | 23.65 MiB | 19.96 MiB |

LightFM-ID serving is about 6.2 times faster than CountSketch in this benchmark and uses slightly less loaded array
memory, but its artifact is about 40% larger and its two-loss offline screen costs about 4.5 times as long. LightFM
is isolated to offline training; FastAPI loads only NumPy embeddings and biases and never imports `lightfm`.

## Metadata audit

LightFM-Hybrid uses sparse identity plus static catalog features: genres (98.0% complete), type (100%), source
(89.8%), studio (65.1%, rare studios collapsed), decade (98.9%), and content-rating classification (89.8%). Each
item row is L1-normalized. Outcome-derived score, rank, members, popularity, rating-count, favorites, watching
statistics, and collaborative availability fields are forbidden. Themes/demographics were empty in the processed
catalog; staff, characters, and voice actors were excluded for missingness/high cardinality in this first screen.

## Direct answers

1. **Does LightFM-ID outperform CountSketch?** Yes on threshold-8 representative ranking metrics, but not for
   sparse users or under threshold 9.
2. **Does metadata improve LightFM overall?** No. Hybrid has lower NDCG/Recall than ID; Recall's interval excludes 0
   in the negative direction.
3. **Does metadata help sparse users?** No; Hybrid is the weakest of the three personalized models in that slice.
4. **Does metadata help mid/long-tail retrieval?** No; both LightFM variants have zero Recall@10 in those strata.
5. **Popularity bias?** LightFM worsens it versus CountSketch; Hybrid partially reduces ID's bias but remains worse.
6. **Is the gain statistically convincing?** At threshold 8 overall, yes. Across thresholds and promotion
   guardrails, no.
7. **Serving cost?** LightFM NumPy scoring is faster and similarly sized in memory, with a larger artifact and higher
   offline training/search cost.
8. **Replace CountSketch?** Not yet. Keep CountSketch and gather more evidence.
9. **If a LightFM variant advances?** Advance LightFM-ID, not Hybrid, as the next offline challenger.
10. **Implement LightGCN next?** No. First run a predeclared larger confirmation, resolve the WARP/BPR coverage
    trade-off, improve sparse/tail objectives, and design a valid item-cold-start test. LightGCN would add complexity
    without evidence that graph depth addresses the observed failure mode.

## Limitations and suspicious results

- The search was deliberately small (two losses at one dimension/epoch setting), not a claim of exhaustive tuning.
- ID-only WARP barely beats BPR on validation NDCG while losing substantially on validation Recall/coverage.
- Evaluation C meets the quota in every bucket but has only 119 unique users because users can hold positives in
  multiple buckets; 118 are heavy users. Its per-bucket cuts are valid, its aggregate is not population-like.
- No interaction timestamps exist; random holdout can benefit from franchise-related titles watched at unknown times.
- The public training path should be Linux/WSL. The local Windows conda-forge WARP/BPR binary required an audited
  local RNG compatibility correction after an access violation; that local binary patch is not part of the repo.
- The current LightFM artifact should not be distributed as a production model until the Linux reproduction matches.

Detailed machine-readable outputs are in:

- [`lightfm_fulltrain_uniform1000`](lightfm_fulltrain_uniform1000/)
- [`lightfm_fulltrain_activity100_each`](lightfm_fulltrain_activity100_each/)
- [`lightfm_fulltrain_popularity100_each`](lightfm_fulltrain_popularity100_each/)
- [`lightfm_threshold_sensitivity`](lightfm_threshold_sensitivity/)
