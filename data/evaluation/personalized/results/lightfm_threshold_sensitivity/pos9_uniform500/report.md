# LightFM Offline Challenger Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **500**; positive threshold: **9**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.1788 | 0.2120 | 0.5360 | 0.2024 | 0.2858 | 0.2725 | 0.0490 | 9.261 | 0.0556 | 0.8254 |
| LightFM-ID | 0.1793 | 0.2069 | 0.5440 | 0.2069 | 0.2945 | 0.2866 | 0.0273 | 8.740 | 0.0871 | 0.8094 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 27 | 0.1095 | 0.2222 | 0.2222 |
| LightFM-ID | 27 | 0.1386 | 0.1852 | 0.1852 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 147 | 0.1888 | 0.2619 | 0.3673 |
| LightFM-ID | 147 | 0.1380 | 0.2245 | 0.3469 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 326 | 0.1801 | 0.1886 | 0.6380 |
| LightFM-ID | 326 | 0.2012 | 0.2007 | 0.6626 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 500 | 2685 | 0.2134 | 0.1794 | 0.9927 |
| LightFM-ID | 500 | 2685 | 0.2087 | 0.1800 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 37 | 69 | 0.0000 | 0.0000 | 0.0059 |
| LightFM-ID | 37 | 69 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 1 | 2 | 0.0000 | 0.0000 | 0.0014 |
| LightFM-ID | 1 | 2 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 37.60s | 37.60s | 755.11 MiB | 7.75ms | 8.56ms | 26.87 MiB | 14.32 MiB |
| LightFM-ID | 51.48s | 67.11s | 1002.82 MiB | 1.27ms | 1.65ms | 23.48 MiB | 19.80 MiB |

Whole-run process peak RSS: **755.11 MiB**.

## Validation-only LightFM model selection

The final configurations below were selected using validation positives only. Test positives were evaluated only after selection.

| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |
|---|---|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 0.1844 | 0.2148 |

## Paired statistical comparisons

### LightFM-ID vs CountSketch CF

- Delta NDCG@10: +0.04 percentage points; 95% paired-bootstrap CI [-1.85, +2.08] pp; relative delta +0.24%.
- Delta Recall@10: -0.51 percentage points; 95% paired-bootstrap CI [-3.01, +2.09] pp; relative delta -2.42%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **LightFM-ID versus CountSketch CF:** NDCG@10 is higher by 0.04 pp; the 95% interval includes zero, so the observed difference is not statistically conclusive.
- **Latency:** LightFM-ID has the lowest p50 in this run (1.27 ms).
- **Sparse users:** LightFM-ID has the highest sampled NDCG@10 (0.1386).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.1888).
- **Heavy users:** LightFM-ID has the highest sampled NDCG@10 (0.2012).
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
- Split artifact SHA-256: `e921d76c8ee3493c84435021372e6c1fab5f40d6af9d8b5b745e7d429f7d1d32`
- Catalog SHA-256: `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75`
- Bootstrap iterations: 1,000
- Evaluation duration: 117.57 seconds
