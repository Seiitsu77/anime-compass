# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **800**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.1516 | 0.1430 | 0.5775 | 0.1634 | 0.1993 | 0.2940 | 0.0782 | 9.729 | 0.0540 | 0.8252 |
| LightFM Als | 0.2624 | 0.2475 | 0.7788 | 0.2921 | 0.3518 | 0.4341 | 0.0793 | 10.120 | 0.0309 | 0.8007 |
| Current Hybrid | 0.1815 | 0.1682 | 0.6312 | 0.2000 | 0.2395 | 0.3485 | 0.0563 | 9.759 | 0.0522 | 0.7396 |
| LightFM Current Hybrid Als | 0.2629 | 0.2434 | 0.7762 | 0.2922 | 0.3485 | 0.4440 | 0.0772 | 10.148 | 0.0292 | 0.7598 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.6579 | 0.9193 | 0.9625 | 0.9877 | 1,413 | 0.9853 | 41877.3 | 0.8765 | 0.8225 |
| LightFM Als | 0.4406 | 0.9235 | 0.9919 | 1.0000 | 1,432 | 0.9673 | 29463.1 | 0.8534 | 0.8225 |
| Current Hybrid | 0.6189 | 0.9234 | 0.9758 | 0.9944 | 1,017 | 0.9878 | 40052.8 | 0.8747 | 0.8225 |
| LightFM Current Hybrid Als | 0.4424 | 0.9088 | 0.9839 | 0.9993 | 1,394 | 0.9703 | 29666.1 | 0.8517 | 0.8225 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 30 | 0.1392 | 0.2667 | 0.2667 |
| LightFM Als | 30 | 0.1734 | 0.3333 | 0.3333 |
| Current Hybrid | 30 | 0.1788 | 0.3000 | 0.3000 |
| LightFM Current Hybrid Als | 30 | 0.1829 | 0.3333 | 0.3333 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 100 | 0.1714 | 0.2200 | 0.3700 |
| LightFM Als | 100 | 0.2501 | 0.3400 | 0.4900 |
| Current Hybrid | 100 | 0.2211 | 0.3000 | 0.4700 |
| LightFM Current Hybrid Als | 100 | 0.2429 | 0.3300 | 0.4800 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 670 | 0.1492 | 0.1259 | 0.6224 |
| LightFM Als | 670 | 0.2682 | 0.2299 | 0.8418 |
| Current Hybrid | 670 | 0.1757 | 0.1427 | 0.6701 |
| LightFM Current Hybrid Als | 670 | 0.2695 | 0.2265 | 0.8403 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 799 | 7557 | 0.1449 | 0.1524 | 0.9877 |
| LightFM Als | 799 | 7557 | 0.2518 | 0.2641 | 1.0000 |
| Current Hybrid | 799 | 7557 | 0.1704 | 0.1824 | 0.9944 |
| LightFM Current Hybrid Als | 799 | 7557 | 0.2481 | 0.2647 | 0.9993 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 132 | 268 | 0.0095 | 0.0032 | 0.0111 |
| LightFM Als | 132 | 268 | 0.0000 | 0.0000 | 0.0000 |
| Current Hybrid | 132 | 268 | 0.0076 | 0.0029 | 0.0053 |
| LightFM Current Hybrid Als | 132 | 268 | 0.0000 | 0.0000 | 0.0004 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 4 | 4 | 0.0000 | 0.0000 | 0.0012 |
| LightFM Als | 4 | 4 | 0.0000 | 0.0000 | 0.0000 |
| Current Hybrid | 4 | 4 | 0.0000 | 0.0000 | 0.0004 |
| LightFM Current Hybrid Als | 4 | 4 | 0.0000 | 0.0000 | 0.0003 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 65.77s | 65.77s | 749.16 MiB | 7.78ms | 9.46ms | 26.87 MiB | 14.27 MiB |
| LightFM Als | 268.39s | 268.39s | 1667.33 MiB | 7.55ms | 9.73ms | 8.96 MiB | 6.94 MiB |
| Current Hybrid | 6.64s | 98.55s | n/a | 947.77ms | 1500.58ms | 43.48 MiB | 128.12 MiB |
| LightFM Current Hybrid Als | 6.69s | 6.69s | n/a | 947.22ms | 1476.55ms | 16.61 MiB | 120.78 MiB |

Whole-run process peak RSS: **1667.33 MiB**.

### Current hybrid timing stages

The production recommender currently exposes combined candidate-generation/channel-scoring and diversity-reranking timers; it does not separately time each content channel.

| Stage | p50 | p95 |
|---|---:|---:|
| candidate generation and channel scoring | 735.78ms | 1295.21ms |
| diversity reranking | 197.80ms | 227.46ms |
| entity resolution | 0.00ms | 0.00ms |
| total | 939.89ms | 1491.90ms |

## Paired statistical comparisons

### Current Hybrid vs CountSketch CF

- Delta NDCG@10: +2.99 percentage points; 95% paired-bootstrap CI [+1.93, +4.15] pp; relative delta +19.71%.
- Delta Recall@10: +2.53 percentage points; 95% paired-bootstrap CI [+1.21, +3.81] pp; relative delta +17.67%.

### LightFM Als vs CountSketch CF

- Delta NDCG@10: +11.08 percentage points; 95% paired-bootstrap CI [+9.55, +12.50] pp; relative delta +73.07%.
- Delta Recall@10: +10.46 percentage points; 95% paired-bootstrap CI [+8.91, +11.99] pp; relative delta +73.13%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **Current Hybrid versus CountSketch CF:** NDCG@10 is higher by 2.99 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM Als versus CountSketch CF:** NDCG@10 is higher by 11.08 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** LightFM Als has the lowest p50 in this run (7.55 ms).
- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about 76.2 hours for 289,601 eligible users. This is a planning estimate, not a measured full-run duration.
- **Sparse users:** LightFM Current Hybrid Als has the highest sampled NDCG@10 (0.1829).
- **Medium users:** LightFM Als has the highest sampled NDCG@10 (0.2501).
- **Heavy users:** LightFM Current Hybrid Als has the highest sampled NDCG@10 (0.2695).
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
- Evaluation duration: 1914.17 seconds
