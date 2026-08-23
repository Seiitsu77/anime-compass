# LightFM Offline Challenger Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **500**; positive threshold: **7**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.1468 | 0.1143 | 0.5840 | 0.1555 | 0.1683 | 0.3087 | 0.0591 | 9.996 | 0.0480 | 0.8296 |
| LightFM-ID | 0.1889 | 0.1388 | 0.6700 | 0.1988 | 0.2140 | 0.3594 | 0.0381 | 9.439 | 0.0806 | 0.7932 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 10 | 0.1448 | 0.3000 | 0.3000 |
| LightFM-ID | 10 | 0.1743 | 0.4000 | 0.4000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 69 | 0.1106 | 0.1667 | 0.2464 |
| LightFM-ID | 69 | 0.0821 | 0.1594 | 0.2899 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 421 | 0.1528 | 0.1013 | 0.6461 |
| LightFM-ID | 421 | 0.2067 | 0.1293 | 0.7387 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 499 | 6759 | 0.1162 | 0.1475 | 0.9916 |
| LightFM-ID | 499 | 6759 | 0.1412 | 0.1895 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 111 | 303 | 0.0000 | 0.0000 | 0.0076 |
| LightFM-ID | 111 | 303 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 7 | 7 | 0.0000 | 0.0000 | 0.0008 |
| LightFM-ID | 7 | 7 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 35.38s | 35.38s | 755.33 MiB | 8.39ms | 9.42ms | 26.87 MiB | 14.17 MiB |
| LightFM-ID | 154.59s | 186.27s | 1514.84 MiB | 1.25ms | 1.59ms | 23.70 MiB | 20.03 MiB |

Whole-run process peak RSS: **755.33 MiB**.

## Validation-only LightFM model selection

The final configurations below were selected using validation positives only. Test positives were evaluated only after selection.

| Model | Loss | Dimensions | Epochs | Validation NDCG@10 | Validation Recall@10 |
|---|---|---:|---:|---:|---:|
| LightFM-ID | WARP | 16 | 3 | 0.1931 | 0.1375 |

## Paired statistical comparisons

### LightFM-ID vs CountSketch CF

- Delta NDCG@10: +4.21 percentage points; 95% paired-bootstrap CI [+2.47, +5.93] pp; relative delta +28.69%.
- Delta Recall@10: +2.46 percentage points; 95% paired-bootstrap CI [+0.68, +4.27] pp; relative delta +21.50%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **LightFM-ID versus CountSketch CF:** NDCG@10 is higher by 4.21 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** LightFM-ID has the lowest p50 in this run (1.25 ms).
- **Sparse users:** LightFM-ID has the highest sampled NDCG@10 (0.1743).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.1106).
- **Heavy users:** LightFM-ID has the highest sampled NDCG@10 (0.2067).
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
- Split artifact SHA-256: `32df74c44c074b0f562f50b69dd37c5b0f34cc3a0facd4fa49d7400c2697046d`
- Catalog SHA-256: `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75`
- Bootstrap iterations: 1,000
- Evaluation duration: 168.38 seconds
