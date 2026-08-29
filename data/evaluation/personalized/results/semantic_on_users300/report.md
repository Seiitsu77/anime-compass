# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **300**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1061 | 0.0867 | 0.4333 | 0.1201 | 0.1465 | 0.2280 | 0.0045 | 8.431 | 0.1228 | 0.8223 |
| CountSketch CF | 0.1552 | 0.1541 | 0.6133 | 0.1720 | 0.2237 | 0.2843 | 0.0383 | 9.562 | 0.0561 | 0.8257 |
| Current Hybrid | 0.1763 | 0.1645 | 0.6233 | 0.1915 | 0.2329 | 0.3358 | 0.0335 | 9.764 | 0.0441 | 0.7366 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 82 | 0.9981 | 73919.3 | 0.9531 | 0.8303 |
| CountSketch CF | 0.6950 | 0.9377 | 0.9750 | 0.9965 | 692 | 0.9903 | 44133.7 | 0.8864 | 0.8303 |
| Current Hybrid | 0.6242 | 0.9363 | 0.9832 | 0.9965 | 606 | 0.9909 | 37976.2 | 0.8744 | 0.8303 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 5 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 5 | 0.0631 | 0.2000 | 0.2000 |
| Current Hybrid | 5 | 0.1262 | 0.2000 | 0.2000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 53 | 0.0756 | 0.1038 | 0.2075 |
| CountSketch CF | 53 | 0.1285 | 0.2075 | 0.3396 |
| Current Hybrid | 53 | 0.1770 | 0.2547 | 0.3774 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 242 | 0.1150 | 0.0847 | 0.4917 |
| CountSketch CF | 242 | 0.1630 | 0.1414 | 0.6818 |
| Current Hybrid | 242 | 0.1771 | 0.1440 | 0.6860 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 300 | 2946 | 0.0871 | 0.1062 | 1.0000 |
| CountSketch CF | 300 | 2946 | 0.1556 | 0.1557 | 0.9965 |
| Current Hybrid | 300 | 2946 | 0.1676 | 0.1776 | 0.9965 |

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
| Popularity | 19.99s | 19.99s | n/a | 0.18ms | 0.33ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 749.43 MiB | 6.68ms | 7.86ms | 26.87 MiB | 14.27 MiB |
| Current Hybrid | 5.25s | 91.00s | n/a | 750.29ms | 1206.11ms | 43.48 MiB | 128.12 MiB |

Whole-run process peak RSS: **1375.10 MiB**.

### Current hybrid timing stages

The production recommender currently exposes combined candidate-generation/channel-scoring and diversity-reranking timers; it does not separately time each content channel.

| Stage | p50 | p95 |
|---|---:|---:|
| candidate generation and channel scoring | 585.96ms | 1033.46ms |
| diversity reranking | 147.87ms | 167.58ms |
| entity resolution | 0.00ms | 0.00ms |
| total | 743.89ms | 1199.04ms |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +4.91 percentage points; 95% paired-bootstrap CI [+3.18, +6.83] pp; relative delta +46.25%.
- Delta Recall@10: +6.74 percentage points; 95% paired-bootstrap CI [+4.60, +9.16] pp; relative delta +77.80%.

### Current Hybrid vs CountSketch CF

- Delta NDCG@10: +2.11 percentage points; 95% paired-bootstrap CI [+0.24, +4.10] pp; relative delta +13.57%.
- Delta Recall@10: +1.04 percentage points; 95% paired-bootstrap CI [-1.46, +3.50] pp; relative delta +6.76%.

### Current Hybrid vs Popularity

- Delta NDCG@10: +7.01 percentage points; 95% paired-bootstrap CI [+4.70, +9.56] pp; relative delta +66.10%.
- Delta Recall@10: +7.78 percentage points; 95% paired-bootstrap CI [+5.20, +10.66] pp; relative delta +89.81%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 4.91 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Current Hybrid versus CountSketch CF:** NDCG@10 is higher by 2.11 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Current Hybrid versus Popularity:** NDCG@10 is higher by 7.01 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** Popularity has the lowest p50 in this run (0.18 ms).
- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about 60.4 hours for 289,601 eligible users. This is a planning estimate, not a measured full-run duration.
- **Sparse users:** Current Hybrid has the highest sampled NDCG@10 (0.1262).
- **Medium users:** Current Hybrid has the highest sampled NDCG@10 (0.1770).
- **Heavy users:** Current Hybrid has the highest sampled NDCG@10 (0.1771).
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
- Evaluation duration: 361.42 seconds
