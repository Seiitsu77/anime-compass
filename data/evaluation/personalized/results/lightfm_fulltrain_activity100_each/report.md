# LightFM Offline Challenger Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation B — activity-balanced diagnostic**.

Run scope: **diagnostic**; evaluated users: **300**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.0869 | 0.0932 | 0.2533 | 0.1035 | 0.1512 | 0.1476 | 0.0042 | 8.339 | 0.1168 | 0.8289 |
| CountSketch CF | 0.1357 | 0.1716 | 0.3800 | 0.1503 | 0.2297 | 0.1865 | 0.0607 | 9.885 | 0.0255 | 0.8211 |
| LightFM-ID | 0.1375 | 0.1606 | 0.4033 | 0.1592 | 0.2511 | 0.2111 | 0.0250 | 9.114 | 0.0710 | 0.7900 |
| LightFM-Hybrid | 0.1300 | 0.1466 | 0.3733 | 0.1472 | 0.2140 | 0.1934 | 0.0287 | 9.281 | 0.0612 | 0.7596 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 100 | 0.0675 | 0.1000 | 0.1000 |
| CountSketch CF | 100 | 0.1319 | 0.2000 | 0.2000 |
| LightFM-ID | 100 | 0.0931 | 0.1500 | 0.1500 |
| LightFM-Hybrid | 100 | 0.0866 | 0.1400 | 0.1400 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 100 | 0.0835 | 0.1050 | 0.1900 |
| CountSketch CF | 100 | 0.1242 | 0.1900 | 0.3100 |
| LightFM-ID | 100 | 0.1144 | 0.1850 | 0.3200 |
| LightFM-Hybrid | 100 | 0.1165 | 0.1550 | 0.2600 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 100 | 0.1098 | 0.0746 | 0.4700 |
| CountSketch CF | 100 | 0.1510 | 0.1247 | 0.6300 |
| LightFM-ID | 100 | 0.2051 | 0.1469 | 0.7400 |
| LightFM-Hybrid | 100 | 0.1870 | 0.1447 | 0.7200 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 296 | 1500 | 0.0947 | 0.0882 | 1.0000 |
| CountSketch CF | 296 | 1500 | 0.1766 | 0.1384 | 0.9737 |
| LightFM-ID | 296 | 1500 | 0.1672 | 0.1412 | 1.0000 |
| LightFM-Hybrid | 296 | 1500 | 0.1513 | 0.1334 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 25 | 51 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 25 | 51 | 0.0000 | 0.0000 | 0.0242 |
| LightFM-ID | 25 | 51 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 25 | 51 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 1 | 1 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 1 | 1 | 0.0000 | 0.0000 | 0.0022 |
| LightFM-ID | 1 | 1 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 1 | 1 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 21.63s | 21.63s | n/a | 0.19ms | 0.27ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 749.83 MiB | 6.89ms | 7.80ms | 26.87 MiB | 14.27 MiB |
| LightFM-ID | 102.23s | 297.61s | 1339.75 MiB | 1.17ms | 1.33ms | 23.65 MiB | 19.97 MiB |
| LightFM-Hybrid | 265.00s | 715.27s | 1345.67 MiB | 1.17ms | 1.33ms | 23.65 MiB | 19.96 MiB |

Whole-run process peak RSS: **749.83 MiB**.

## Validation-only LightFM model selection

The final configurations below were selected using validation positives only. Test positives were evaluated only after selection.

| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |
|---|---|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 0.1988 | 0.1783 |
| LightFM-Hybrid | WARP | 16 | 3 | 0.1708 | 0.1611 |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +4.88 percentage points; 95% paired-bootstrap CI [+1.93, +7.93] pp; relative delta +56.13%.
- Delta Recall@10: +7.83 percentage points; 95% paired-bootstrap CI [+4.18, +11.68] pp; relative delta +84.05%.

### LightFM-ID vs CountSketch CF

- Delta NDCG@10: +0.18 percentage points; 95% paired-bootstrap CI [-2.77, +3.30] pp; relative delta +1.33%.
- Delta Recall@10: -1.09 percentage points; 95% paired-bootstrap CI [-5.17, +3.17] pp; relative delta -6.36%.

### LightFM-Hybrid vs CountSketch CF

- Delta NDCG@10: -0.57 percentage points; 95% paired-bootstrap CI [-3.73, +2.54] pp; relative delta -4.19%.
- Delta Recall@10: -2.50 percentage points; 95% paired-bootstrap CI [-6.70, +1.54] pp; relative delta -14.57%.

### LightFM-Hybrid vs LightFM-ID

- Delta NDCG@10: -0.75 percentage points; 95% paired-bootstrap CI [-3.23, +1.77] pp; relative delta -5.45%.
- Delta Recall@10: -1.41 percentage points; 95% paired-bootstrap CI [-4.63, +1.92] pp; relative delta -8.77%.

## Interpretation

**Diagnostic sample:** its unweighted aggregate metrics are not estimates of whole-population performance. Use only the segment or popularity-stratum cuts that this evaluation was designed to inspect.

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 4.88 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM-ID versus CountSketch CF:** NDCG@10 is higher by 0.18 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **LightFM-Hybrid versus CountSketch CF:** NDCG@10 is lower by 0.57 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **LightFM-Hybrid versus LightFM-ID:** NDCG@10 is lower by 0.75 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **Latency:** Popularity has the lowest p50 in this run (0.19 ms).
- **Sparse users:** CountSketch CF has the highest sampled NDCG@10 (0.1319).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.1242).
- **Heavy users:** LightFM-ID has the highest sampled NDCG@10 (0.2051).
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
- Evaluation duration: 71.98 seconds
