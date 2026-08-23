# LightFM Offline Challenger Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **500**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.1567 | 0.1484 | 0.6160 | 0.1724 | 0.2125 | 0.2996 | 0.0543 | 9.613 | 0.0569 | 0.8268 |
| LightFM-ID | 0.1948 | 0.1669 | 0.6840 | 0.2135 | 0.2521 | 0.3823 | 0.0309 | 9.127 | 0.0855 | 0.8025 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 9 | 0.1462 | 0.2222 | 0.2222 |
| LightFM-ID | 9 | 0.0780 | 0.2222 | 0.2222 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 85 | 0.1285 | 0.1941 | 0.3294 |
| LightFM-ID | 85 | 0.1080 | 0.1765 | 0.3059 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 406 | 0.1628 | 0.1372 | 0.6847 |
| LightFM-ID | 406 | 0.2156 | 0.1637 | 0.7734 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 498 | 4838 | 0.1516 | 0.1581 | 0.9921 |
| LightFM-ID | 498 | 4838 | 0.1714 | 0.1971 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 70 | 149 | 0.0000 | 0.0000 | 0.0076 |
| LightFM-ID | 70 | 149 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 1 | 1 | 0.0000 | 0.0000 | 0.0003 |
| LightFM-ID | 1 | 1 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 65.77s | 65.77s | 755.09 MiB | 7.77ms | 8.76ms | 26.87 MiB | 14.27 MiB |
| LightFM-ID | 102.23s | 297.61s | 1339.75 MiB | 1.26ms | 1.66ms | 23.65 MiB | 19.97 MiB |

Whole-run process peak RSS: **755.09 MiB**.

## Validation-only LightFM model selection

The final configurations below were selected using validation positives only. Test positives were evaluated only after selection.

| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |
|---|---|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 0.1988 | 0.1783 |

## Paired statistical comparisons

### LightFM-ID vs CountSketch CF

- Delta NDCG@10: +3.81 percentage points; 95% paired-bootstrap CI [+2.17, +5.61] pp; relative delta +24.33%.
- Delta Recall@10: +1.85 percentage points; 95% paired-bootstrap CI [-0.15, +3.70] pp; relative delta +12.45%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **LightFM-ID versus CountSketch CF:** NDCG@10 is higher by 3.81 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** LightFM-ID has the lowest p50 in this run (1.26 ms).
- **Sparse users:** CountSketch CF has the highest sampled NDCG@10 (0.1462).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.1285).
- **Heavy users:** LightFM-ID has the highest sampled NDCG@10 (0.2156).
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
- Bootstrap iterations: 1,000
- Evaluation duration: 105.33 seconds
