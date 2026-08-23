# LightFM Offline Challenger Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **1,000**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1023 | 0.0881 | 0.4420 | 0.1149 | 0.1421 | 0.2153 | 0.0059 | 8.424 | 0.1297 | 0.8222 |
| CountSketch CF | 0.1534 | 0.1408 | 0.5840 | 0.1665 | 0.2005 | 0.2960 | 0.0775 | 9.639 | 0.0579 | 0.8264 |
| LightFM-ID | 0.1833 | 0.1589 | 0.6430 | 0.2025 | 0.2432 | 0.3480 | 0.0380 | 9.152 | 0.0867 | 0.7990 |
| LightFM-Hybrid | 0.1747 | 0.1483 | 0.6250 | 0.1917 | 0.2258 | 0.3227 | 0.0421 | 9.308 | 0.0775 | 0.7770 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 17 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 17 | 0.0983 | 0.1765 | 0.1765 |
| LightFM-ID | 17 | 0.1001 | 0.1765 | 0.1765 |
| LightFM-Hybrid | 17 | 0.1025 | 0.1765 | 0.1765 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 169 | 0.0827 | 0.1183 | 0.2071 |
| CountSketch CF | 169 | 0.1371 | 0.2012 | 0.3195 |
| LightFM-ID | 169 | 0.1118 | 0.1834 | 0.3018 |
| LightFM-Hybrid | 169 | 0.1057 | 0.1509 | 0.2604 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 814 | 0.1085 | 0.0837 | 0.5000 |
| CountSketch CF | 814 | 0.1580 | 0.1275 | 0.6474 |
| LightFM-ID | 814 | 0.1999 | 0.1535 | 0.7236 |
| LightFM-Hybrid | 814 | 0.1905 | 0.1471 | 0.7101 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 998 | 9921 | 0.0889 | 0.1026 | 1.0000 |
| CountSketch CF | 998 | 9921 | 0.1431 | 0.1543 | 0.9931 |
| LightFM-ID | 998 | 9921 | 0.1623 | 0.1848 | 1.0000 |
| LightFM-Hybrid | 998 | 9921 | 0.1513 | 0.1761 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 162 | 343 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 162 | 343 | 0.0000 | 0.0000 | 0.0065 |
| LightFM-ID | 162 | 343 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 162 | 343 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 4 | 5 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 4 | 5 | 0.0000 | 0.0000 | 0.0004 |
| LightFM-ID | 4 | 5 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 4 | 5 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 21.52s | 21.52s | n/a | 0.22ms | 0.32ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 749.69 MiB | 7.28ms | 8.18ms | 26.87 MiB | 14.27 MiB |
| LightFM-ID | 102.23s | 297.61s | 1339.75 MiB | 1.17ms | 1.34ms | 23.65 MiB | 19.97 MiB |
| LightFM-Hybrid | 265.00s | 715.27s | 1345.67 MiB | 1.16ms | 1.31ms | 23.65 MiB | 19.96 MiB |

Whole-run process peak RSS: **749.69 MiB**.

## Validation-only LightFM model selection

The final configurations below were selected using validation positives only. Test positives were evaluated only after selection.

| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |
|---|---|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 0.1988 | 0.1783 |
| LightFM-Hybrid | WARP | 16 | 3 | 0.1708 | 0.1611 |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +5.11 percentage points; 95% paired-bootstrap CI [+3.95, +6.30] pp; relative delta +49.95%.
- Delta Recall@10: +5.27 percentage points; 95% paired-bootstrap CI [+3.92, +6.59] pp; relative delta +59.75%.

### LightFM-ID vs CountSketch CF

- Delta NDCG@10: +2.99 percentage points; 95% paired-bootstrap CI [+1.71, +4.28] pp; relative delta +19.51%.
- Delta Recall@10: +1.82 percentage points; 95% paired-bootstrap CI [+0.42, +3.19] pp; relative delta +12.90%.

### LightFM-Hybrid vs CountSketch CF

- Delta NDCG@10: +2.13 percentage points; 95% paired-bootstrap CI [+0.74, +3.51] pp; relative delta +13.86%.
- Delta Recall@10: +0.75 percentage points; 95% paired-bootstrap CI [-0.67, +2.15] pp; relative delta +5.32%.

### LightFM-Hybrid vs LightFM-ID

- Delta NDCG@10: -0.87 percentage points; 95% paired-bootstrap CI [-1.84, +0.08] pp; relative delta -4.73%.
- Delta Recall@10: -1.07 percentage points; 95% paired-bootstrap CI [-2.12, -0.03] pp; relative delta -6.72%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 5.11 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM-ID versus CountSketch CF:** NDCG@10 is higher by 2.99 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM-Hybrid versus CountSketch CF:** NDCG@10 is higher by 2.13 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM-Hybrid versus LightFM-ID:** NDCG@10 is lower by 0.87 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **Latency:** Popularity has the lowest p50 in this run (0.22 ms).
- **Sparse users:** LightFM-Hybrid has the highest sampled NDCG@10 (0.1025).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.1371).
- **Heavy users:** LightFM-ID has the highest sampled NDCG@10 (0.1999).
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
- Evaluation duration: 288.88 seconds
