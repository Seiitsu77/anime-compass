# LightFM Offline Challenger Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **100**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1009 | 0.0892 | 0.4400 | 0.1113 | 0.1325 | 0.2167 | 0.0037 | 8.882 | 0.2582 | 0.8228 |
| CountSketch CF | 0.1212 | 0.0962 | 0.4000 | 0.1334 | 0.1434 | 0.2384 | 0.0345 | 10.599 | 0.0404 | 0.8363 |
| LightFM-ID | 0.1014 | 0.0805 | 0.4400 | 0.1198 | 0.1532 | 0.2206 | 0.0054 | 9.000 | 0.2432 | 0.8215 |
| LightFM-Hybrid | 0.0761 | 0.0522 | 0.3300 | 0.0852 | 0.0985 | 0.1716 | 0.0076 | 9.778 | 0.1445 | 0.7700 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 2 | 0.1445 | 0.5000 | 0.5000 |
| CountSketch CF | 2 | 0.5000 | 0.5000 | 0.5000 |
| LightFM-ID | 2 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 2 | 0.0000 | 0.0000 | 0.0000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 17 | 0.0348 | 0.0588 | 0.1176 |
| CountSketch CF | 17 | 0.1210 | 0.1176 | 0.1765 |
| LightFM-ID | 17 | 0.0180 | 0.0294 | 0.0588 |
| LightFM-Hybrid | 17 | 0.0000 | 0.0000 | 0.0000 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 81 | 0.1138 | 0.0854 | 0.5062 |
| CountSketch CF | 81 | 0.1119 | 0.0817 | 0.4444 |
| LightFM-ID | 81 | 0.1213 | 0.0932 | 0.5309 |
| LightFM-Hybrid | 81 | 0.0939 | 0.0644 | 0.4074 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 100 | 917 | 0.0931 | 0.1027 | 1.0000 |
| CountSketch CF | 100 | 917 | 0.0972 | 0.1218 | 0.9400 |
| LightFM-ID | 100 | 917 | 0.0840 | 0.1025 | 1.0000 |
| LightFM-Hybrid | 100 | 917 | 0.0533 | 0.0762 | 0.9990 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 20 | 41 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 20 | 41 | 0.0000 | 0.0000 | 0.0500 |
| LightFM-ID | 20 | 41 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 20 | 41 | 0.0000 | 0.0000 | 0.0010 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 11 | 11 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 11 | 11 | 0.0000 | 0.0000 | 0.0100 |
| LightFM-ID | 11 | 11 | 0.0000 | 0.0000 | 0.0000 |
| LightFM-Hybrid | 11 | 11 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.04s | 0.04s | n/a | 0.18ms | 0.22ms | 0.41 MiB | 0.03 MiB |
| CountSketch CF | 0.23s | 0.23s | 749.52 MiB | 6.77ms | 7.52ms | 26.87 MiB | 0.92 MiB |
| LightFM-ID | 0.11s | 16.71s | 763.24 MiB | 1.18ms | 1.34ms | 1.35 MiB | 1.11 MiB |
| LightFM-Hybrid | 0.43s | 17.48s | 763.24 MiB | 1.18ms | 1.36ms | 1.35 MiB | 1.14 MiB |

Whole-run process peak RSS: **749.52 MiB**.

## Validation-only LightFM model selection

The final configurations below were selected using validation positives only. Test positives were evaluated only after selection.

| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |
|---|---|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 0.1287 | 0.1075 |
| LightFM-Hybrid | WARP | 16 | 3 | 0.1058 | 0.0869 |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +2.03 percentage points; 95% paired-bootstrap CI [-1.71, +5.95] pp; relative delta +20.07%.
- Delta Recall@10: +0.70 percentage points; 95% paired-bootstrap CI [-2.60, +3.95] pp; relative delta +7.87%.

### LightFM-ID vs CountSketch CF

- Delta NDCG@10: -1.98 percentage points; 95% paired-bootstrap CI [-6.48, +1.71] pp; relative delta -16.38%.
- Delta Recall@10: -1.58 percentage points; 95% paired-bootstrap CI [-5.38, +2.02] pp; relative delta -16.38%.

### LightFM-Hybrid vs CountSketch CF

- Delta NDCG@10: -4.51 percentage points; 95% paired-bootstrap CI [-8.93, -0.29] pp; relative delta -37.24%.
- Delta Recall@10: -4.41 percentage points; 95% paired-bootstrap CI [-8.64, -0.72] pp; relative delta -45.79%.

### LightFM-Hybrid vs LightFM-ID

- Delta NDCG@10: -2.53 percentage points; 95% paired-bootstrap CI [-4.96, +0.12] pp; relative delta -24.95%.
- Delta Recall@10: -2.83 percentage points; 95% paired-bootstrap CI [-4.97, -0.48] pp; relative delta -35.18%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

**Pipeline smoke only:** model artifacts were trained on a source-user prefix, so ranking values must not be compared with full-data experiments.

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 2.03 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **LightFM-ID versus CountSketch CF:** NDCG@10 is lower by 1.98 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **LightFM-Hybrid versus CountSketch CF:** NDCG@10 is lower by 4.51 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM-Hybrid versus LightFM-ID:** NDCG@10 is lower by 2.53 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **Latency:** Popularity has the lowest p50 in this run (0.18 ms).
- **Sparse users:** CountSketch CF has the highest sampled NDCG@10 (0.5000).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.1210).
- **Heavy users:** LightFM-ID has the highest sampled NDCG@10 (0.1213).
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

- Dataset SHA-256: `05b71cafcf9f527ac0bd60311d16563981bb7cedaf7da0e88f02e9abc1d1b58a`
- Split artifact SHA-256: `ac8ca73b4b31ab422fc3e2e63e54b08dc3b10fadb8ced8d3a0e6c14ef2625e14`
- Catalog SHA-256: `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75`
- Bootstrap iterations: 500
- Evaluation duration: 26.56 seconds
