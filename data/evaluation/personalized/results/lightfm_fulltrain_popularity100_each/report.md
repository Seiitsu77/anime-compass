# LightFM Offline Challenger Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation C — popularity-stratified diagnostic**.

Run scope: **diagnostic**; evaluated users: **119**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1292 | 0.0374 | 0.6387 | 0.1147 | 0.0586 | 0.3115 | 0.0055 | 8.493 | 0.2246 | 0.8231 |
| CountSketch CF | 0.1585 | 0.0631 | 0.7227 | 0.1504 | 0.0947 | 0.3294 | 0.0371 | 10.452 | 0.1089 | 0.8295 |
| LightFM-ID | 0.2161 | 0.0683 | 0.7983 | 0.2038 | 0.1120 | 0.4252 | 0.0312 | 9.499 | 0.1652 | 0.7946 |
| LightFM-Hybrid | 0.2050 | 0.0719 | 0.7647 | 0.1939 | 0.1146 | 0.3855 | 0.0359 | 9.808 | 0.1469 | 0.7810 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 0 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 0 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-ID | 0 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 0 | 0.0000 | 0.0000 | 0.0000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 1 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 1 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-ID | 1 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 1 | 0.0000 | 0.0000 | 0.0000 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 118 | 0.1303 | 0.0377 | 0.6441 |
| CountSketch CF | 118 | 0.1598 | 0.0636 | 0.7288 |
| LightFM-ID | 118 | 0.2179 | 0.0689 | 0.8051 |
| LightFM-Hybrid | 118 | 0.2068 | 0.0725 | 0.7712 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 119 | 3856 | 0.0450 | 0.1309 | 1.0000 |
| CountSketch CF | 119 | 3856 | 0.0778 | 0.1567 | 0.9584 |
| LightFM-ID | 119 | 3856 | 0.0859 | 0.2202 | 1.0000 |
| LightFM-Hybrid | 119 | 3856 | 0.0925 | 0.2098 | 0.9979 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 100 | 1211 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 100 | 1211 | 0.0026 | 0.0033 | 0.0311 |
| LightFM-ID | 100 | 1211 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 100 | 1211 | 0.0000 | 0.0000 | 0.0021 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 100 | 704 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 100 | 704 | 0.0101 | 0.0072 | 0.0105 |
| LightFM-ID | 100 | 704 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 100 | 704 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 20.80s | 20.80s | n/a | 0.40ms | 0.64ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 749.42 MiB | 7.63ms | 10.23ms | 26.87 MiB | 14.27 MiB |
| LightFM-ID | 102.23s | 297.61s | 1339.75 MiB | 1.22ms | 1.42ms | 23.65 MiB | 19.97 MiB |
| LightFM-Hybrid | 265.00s | 715.27s | 1345.67 MiB | 1.21ms | 1.43ms | 23.65 MiB | 19.96 MiB |

Whole-run process peak RSS: **749.42 MiB**.

## Validation-only LightFM model selection

The final configurations below were selected using validation positives only. Test positives were evaluated only after selection.

| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |
|---|---|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 0.1988 | 0.1783 |
| LightFM-Hybrid | WARP | 16 | 3 | 0.1708 | 0.1611 |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +2.92 percentage points; 95% paired-bootstrap CI [-0.30, +6.28] pp; relative delta +22.61%.
- Delta Recall@10: +2.57 percentage points; 95% paired-bootstrap CI [+0.79, +4.62] pp; relative delta +68.88%.

### LightFM-ID vs CountSketch CF

- Delta NDCG@10: +5.77 percentage points; 95% paired-bootstrap CI [+2.19, +9.31] pp; relative delta +36.39%.
- Delta Recall@10: +0.52 percentage points; 95% paired-bootstrap CI [-1.45, +2.33] pp; relative delta +8.23%.

### LightFM-Hybrid vs CountSketch CF

- Delta NDCG@10: +4.66 percentage points; 95% paired-bootstrap CI [+1.49, +7.71] pp; relative delta +29.39%.
- Delta Recall@10: +0.88 percentage points; 95% paired-bootstrap CI [-0.51, +2.26] pp; relative delta +13.98%.

### LightFM-Hybrid vs LightFM-ID

- Delta NDCG@10: -1.11 percentage points; 95% paired-bootstrap CI [-3.57, +1.31] pp; relative delta -5.13%.
- Delta Recall@10: +0.36 percentage points; 95% paired-bootstrap CI [-1.05, +1.70] pp; relative delta +5.31%.

## Interpretation

**Diagnostic sample:** its unweighted aggregate metrics are not estimates of whole-population performance. Use only the segment or popularity-stratum cuts that this evaluation was designed to inspect.

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 2.92 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **LightFM-ID versus CountSketch CF:** NDCG@10 is higher by 5.77 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM-Hybrid versus CountSketch CF:** NDCG@10 is higher by 4.66 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM-Hybrid versus LightFM-ID:** NDCG@10 is lower by 1.11 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **Latency:** Popularity has the lowest p50 in this run (0.40 ms).
- **Medium users:** Popularity has the highest sampled NDCG@10 (0.0000).
- **Heavy users:** LightFM-ID has the highest sampled NDCG@10 (0.2179).
- **Popularity/long tail:** inspect exposure together with held-out long-tail Recall@10; a model can appear novel simply because the train-only artifact has little evidence for many items.
- **Promotion gate:** LightFM remains an offline challenger. Replacement requires a positive paired interval, roughly 5% relative NDCG@10 lift, no material Recall/coverage/diversity regression, and acceptable serving cost.

## What this experiment can tell us

- Personalized ranking quality under deterministic random held-out positive interactions.
- Whether the current collaborative signal adds value over train-only popularity.
- Whether LightFM collaborative challengers add value over CountSketch when included in the run.
- How ranking quality changes with training-positive user activity.
- Recommendation exposure and recovery across train-defined item popularity buckets.

## What this experiment cannot tell us

- Chronological next-anime prediction because the source has no interaction timestamps.
- Real online click-through rate or recommendation acceptance.
- Causal user satisfaction.
- Production A/B-test performance without online traffic.

## Reproducibility

- Dataset SHA-256: `b60519348a90bd5e02c25355b374f7ca055a0637a237f0e163447953b13ffaa0`
- Split artifact SHA-256: `a668114f043a54dc7048dddc8d5290416579b0eda5abbddc14bc47065c970038`
- Catalog SHA-256: `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75`
- Bootstrap iterations: 2,000
- Evaluation duration: 146.45 seconds
