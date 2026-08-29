# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **800**; positive threshold: **9**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.0003 | 0.0003 | 0.0025 | 0.0006 | 0.0013 | 0.0008 | 0.5890 | 19.547 | -0.5629 | 0.9102 |
| Popularity | 0.0924 | 0.1115 | 0.3588 | 0.1148 | 0.1786 | 0.1646 | 0.0064 | 8.036 | 0.1351 | 0.8258 |
| CountSketch CF | 0.1668 | 0.1938 | 0.5325 | 0.1936 | 0.2720 | 0.2788 | 0.0696 | 9.327 | 0.0568 | 0.8246 |
| LightFM Als | 0.2652 | 0.2884 | 0.6963 | 0.2980 | 0.3853 | 0.4154 | 0.0697 | 10.067 | 0.0119 | 0.8061 |
| LightFM Oracle | 1.0000 | 0.9557 | 1.0000 | 1.0000 | 0.9915 | 1.0000 | 0.0867 | 12.150 | -0.1144 | 0.8054 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.0075 | 0.0451 | 0.0916 | 0.1905 | 10,639 | 0.5486 | 647.3 | 0.2496 | 0.8125 |
| Popularity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 116 | 0.9981 | 52697.8 | 0.9475 | 0.8125 |
| CountSketch CF | 0.7111 | 0.9384 | 0.9708 | 0.9889 | 1,258 | 0.9879 | 30960.2 | 0.8693 | 0.8125 |
| LightFM Als | 0.4194 | 0.9446 | 0.9959 | 1.0000 | 1,259 | 0.9690 | 16898.7 | 0.8244 | 0.8125 |
| LightFM Oracle | 0.2462 | 0.6271 | 0.7901 | 0.8642 | 1,566 | 0.9870 | 10474.0 | 0.6981 | 0.8125 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 32 | 0.0000 | 0.0000 | 0.0000 |
| Popularity | 32 | 0.0802 | 0.1875 | 0.1875 |
| CountSketch CF | 32 | 0.2497 | 0.4062 | 0.4062 |
| LightFM Als | 32 | 0.1444 | 0.2500 | 0.2500 |
| LightFM Oracle | 32 | 1.0000 | 1.0000 | 1.0000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 222 | 0.0000 | 0.0000 | 0.0000 |
| Popularity | 222 | 0.0782 | 0.1149 | 0.2027 |
| CountSketch CF | 222 | 0.1468 | 0.2095 | 0.3378 |
| LightFM Als | 222 | 0.2497 | 0.3243 | 0.5045 |
| LightFM Oracle | 222 | 1.0000 | 1.0000 | 1.0000 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 546 | 0.0004 | 0.0004 | 0.0037 |
| Popularity | 546 | 0.0989 | 0.1056 | 0.4322 |
| CountSketch CF | 546 | 0.1700 | 0.1750 | 0.6190 |
| LightFM Als | 546 | 0.2786 | 0.2761 | 0.8004 |
| LightFM Oracle | 546 | 1.0000 | 0.9351 | 1.0000 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 798 | 4290 | 0.0003 | 0.0003 | 0.1905 |
| Popularity | 798 | 4290 | 0.1133 | 0.0935 | 1.0000 |
| CountSketch CF | 798 | 4290 | 0.1954 | 0.1677 | 0.9889 |
| LightFM Als | 798 | 4290 | 0.2920 | 0.2670 | 1.0000 |
| LightFM Oracle | 798 | 4290 | 0.9564 | 0.9958 | 0.8642 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 66 | 128 | 0.0000 | 0.0000 | 0.3060 |
| Popularity | 66 | 128 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 66 | 128 | 0.0051 | 0.0045 | 0.0103 |
| LightFM Als | 66 | 128 | 0.0000 | 0.0000 | 0.0000 |
| LightFM Oracle | 66 | 128 | 0.6906 | 0.3560 | 0.0929 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 2 | 2 | 0.0000 | 0.0000 | 0.5035 |
| Popularity | 2 | 2 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 2 | 2 | 0.0000 | 0.0000 | 0.0008 |
| LightFM Als | 2 | 2 | 0.0000 | 0.0000 | 0.0000 |
| LightFM Oracle | 2 | 2 | 0.0000 | 0.0000 | 0.0428 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.00s | 0.00s | n/a | 2.28ms | 3.43ms | 0.00 MiB | 0.00 MiB |
| Popularity | 19.60s | 19.60s | n/a | 0.18ms | 0.41ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 37.05s | 37.05s | 749.08 MiB | 6.54ms | 7.55ms | 26.87 MiB | 14.32 MiB |
| LightFM Als | 272.98s | 272.98s | 749.08 MiB | 6.57ms | 7.61ms | 8.96 MiB | 6.65 MiB |
| LightFM Oracle | 0.00s | 0.00s | n/a | 0.02ms | 0.04ms | 0.00 MiB | 0.00 MiB |

Whole-run process peak RSS: **749.08 MiB**.

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +7.43 percentage points; 95% paired-bootstrap CI [+6.01, +9.01] pp; relative delta +80.42%.
- Delta Recall@10: +8.23 percentage points; 95% paired-bootstrap CI [+6.41, +10.09] pp; relative delta +73.88%.

### LightFM Als vs CountSketch CF

- Delta NDCG@10: +9.84 percentage points; 95% paired-bootstrap CI [+8.10, +11.54] pp; relative delta +59.03%.
- Delta Recall@10: +9.46 percentage points; 95% paired-bootstrap CI [+7.25, +11.60] pp; relative delta +48.82%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 7.43 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM Als versus CountSketch CF:** NDCG@10 is higher by 9.84 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** LightFM Oracle has the lowest p50 in this run (0.02 ms).
- **Sparse users:** LightFM Oracle has the highest sampled NDCG@10 (1.0000).
- **Medium users:** LightFM Oracle has the highest sampled NDCG@10 (1.0000).
- **Heavy users:** LightFM Oracle has the highest sampled NDCG@10 (1.0000).
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
- Split artifact SHA-256: `e921d76c8ee3493c84435021372e6c1fab5f40d6af9d8b5b745e7d429f7d1d32`
- Catalog SHA-256: `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75`
- Bootstrap iterations: 2,000
- Evaluation duration: 247.61 seconds
