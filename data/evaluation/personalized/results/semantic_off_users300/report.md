# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **300**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1061 | 0.0867 | 0.4333 | 0.1201 | 0.1465 | 0.2280 | 0.0045 | 8.431 | 0.1228 | 0.8223 |
| CountSketch CF | 0.1552 | 0.1541 | 0.6133 | 0.1720 | 0.2237 | 0.2843 | 0.0383 | 9.562 | 0.0561 | 0.8257 |
| Current Hybrid | 0.1935 | 0.1868 | 0.6567 | 0.2109 | 0.2582 | 0.3528 | 0.0340 | 9.635 | 0.0517 | 0.7478 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 82 | 0.9981 | 73919.3 | 0.9531 | 0.8303 |
| CountSketch CF | 0.6950 | 0.9377 | 0.9750 | 0.9965 | 692 | 0.9903 | 44133.7 | 0.8864 | 0.8303 |
| Current Hybrid | 0.6518 | 0.9433 | 0.9855 | 0.9965 | 614 | 0.9905 | 41856.9 | 0.8820 | 0.8303 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 5 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 5 | 0.0631 | 0.2000 | 0.2000 |
| Current Hybrid | 5 | 0.1840 | 0.4000 | 0.4000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 53 | 0.0756 | 0.1038 | 0.2075 |
| CountSketch CF | 53 | 0.1285 | 0.2075 | 0.3396 |
| Current Hybrid | 53 | 0.1681 | 0.2453 | 0.3774 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 242 | 0.1150 | 0.0847 | 0.4917 |
| CountSketch CF | 242 | 0.1630 | 0.1414 | 0.6818 |
| Current Hybrid | 242 | 0.1992 | 0.1696 | 0.7231 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 300 | 2946 | 0.0871 | 0.1062 | 1.0000 |
| CountSketch CF | 300 | 2946 | 0.1556 | 0.1557 | 0.9965 |
| Current Hybrid | 300 | 2946 | 0.1904 | 0.1955 | 0.9965 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 41 | 83 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 41 | 83 | 0.0000 | 0.0000 | 0.0035 |
| Current Hybrid | 41 | 83 | 0.0000 | 0.0000 | 0.0027 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| Current Hybrid | 0 | 0 | 0.0000 | 0.0000 | 0.0008 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 19.67s | 19.67s | n/a | 0.17ms | 0.33ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 749.52 MiB | 6.62ms | 8.02ms | 26.87 MiB | 14.27 MiB |
| Current Hybrid | 5.04s | 90.48s | n/a | 726.30ms | 1183.23ms | 43.48 MiB | 128.12 MiB |

Whole-run process peak RSS: **1337.44 MiB**.

### Current hybrid timing stages

The production recommender currently exposes combined candidate-generation/channel-scoring and diversity-reranking timers; it does not separately time each content channel.

| Stage | p50 | p95 |
|---|---:|---:|
| candidate generation and channel scoring | 567.58ms | 1018.56ms |
| diversity reranking | 143.72ms | 164.67ms |
| entity resolution | 0.00ms | 0.00ms |
| total | 719.34ms | 1176.11ms |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +4.91 percentage points; 95% paired-bootstrap CI [+3.18, +6.83] pp; relative delta +46.25%.
- Delta Recall@10: +6.74 percentage points; 95% paired-bootstrap CI [+4.60, +9.16] pp; relative delta +77.80%.

### Current Hybrid vs CountSketch CF

- Delta NDCG@10: +3.83 percentage points; 95% paired-bootstrap CI [+2.06, +5.75] pp; relative delta +24.66%.
- Delta Recall@10: +3.27 percentage points; 95% paired-bootstrap CI [+0.89, +5.56] pp; relative delta +21.23%.

### Current Hybrid vs Popularity

- Delta NDCG@10: +8.74 percentage points; 95% paired-bootstrap CI [+6.56, +11.15] pp; relative delta +82.33%.
- Delta Recall@10: +10.01 percentage points; 95% paired-bootstrap CI [+7.53, +12.79] pp; relative delta +115.55%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 4.91 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Current Hybrid versus CountSketch CF:** NDCG@10 is higher by 3.83 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Current Hybrid versus Popularity:** NDCG@10 is higher by 8.74 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** Popularity has the lowest p50 in this run (0.17 ms).
- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about 58.4 hours for 289,601 eligible users. This is a planning estimate, not a measured full-run duration.
- **Sparse users:** Current Hybrid has the highest sampled NDCG@10 (0.1840).
- **Medium users:** Current Hybrid has the highest sampled NDCG@10 (0.1681).
- **Heavy users:** Current Hybrid has the highest sampled NDCG@10 (0.1992).
- **Popularity/long tail:** inspect exposure together with held-out long-tail Recall@10; a model can appear novel simply because the train-only artifact has little evidence for many items.

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
- Evaluation duration: 351.70 seconds
