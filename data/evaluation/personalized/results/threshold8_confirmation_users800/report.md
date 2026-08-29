# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **800**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.0008 | 0.0009 | 0.0075 | 0.0010 | 0.0012 | 0.0024 | 0.5869 | 19.692 | -0.5343 | 0.9102 |
| Popularity | 0.1087 | 0.1000 | 0.4713 | 0.1188 | 0.1442 | 0.2230 | 0.0054 | 8.420 | 0.1313 | 0.8230 |
| CountSketch CF | 0.1516 | 0.1430 | 0.5775 | 0.1634 | 0.1993 | 0.2940 | 0.0782 | 9.729 | 0.0540 | 0.8252 |
| LightFM Als | 0.2624 | 0.2475 | 0.7788 | 0.2921 | 0.3518 | 0.4341 | 0.0793 | 10.120 | 0.0309 | 0.8007 |
| LightFM Oracle | 1.0000 | 0.8650 | 1.0000 | 1.0000 | 0.9674 | 1.0000 | 0.1188 | 11.868 | -0.0723 | 0.8119 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.0064 | 0.0440 | 0.0935 | 0.1953 | 10,602 | 0.5508 | 1169.6 | 0.2882 | 0.8225 |
| Popularity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 98 | 0.9981 | 74489.0 | 0.9538 | 0.8225 |
| CountSketch CF | 0.6579 | 0.9193 | 0.9625 | 0.9877 | 1,413 | 0.9853 | 41877.3 | 0.8765 | 0.8225 |
| LightFM Als | 0.4406 | 0.9235 | 0.9919 | 1.0000 | 1,432 | 0.9673 | 29463.1 | 0.8534 | 0.8225 |
| LightFM Oracle | 0.2412 | 0.6157 | 0.8079 | 0.8892 | 2,146 | 0.9740 | 18797.8 | 0.7502 | 0.8225 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 30 | 0.0000 | 0.0000 | 0.0000 |
| Popularity | 30 | 0.0593 | 0.1333 | 0.1333 |
| CountSketch CF | 30 | 0.1392 | 0.2667 | 0.2667 |
| LightFM Als | 30 | 0.1734 | 0.3333 | 0.3333 |
| LightFM Oracle | 30 | 1.0000 | 1.0000 | 1.0000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 100 | 0.0000 | 0.0000 | 0.0000 |
| Popularity | 100 | 0.1056 | 0.1550 | 0.2800 |
| CountSketch CF | 100 | 0.1714 | 0.2200 | 0.3700 |
| LightFM Als | 100 | 0.2501 | 0.3400 | 0.4900 |
| LightFM Oracle | 100 | 1.0000 | 1.0000 | 1.0000 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 670 | 0.0010 | 0.0011 | 0.0090 |
| Popularity | 670 | 0.1114 | 0.0903 | 0.5149 |
| CountSketch CF | 670 | 0.1492 | 0.1259 | 0.6224 |
| LightFM Als | 670 | 0.2682 | 0.2299 | 0.8418 |
| LightFM Oracle | 670 | 1.0000 | 0.8389 | 1.0000 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 799 | 7557 | 0.0009 | 0.0007 | 0.1953 |
| Popularity | 799 | 7557 | 0.1017 | 0.1093 | 1.0000 |
| CountSketch CF | 799 | 7557 | 0.1449 | 0.1524 | 0.9877 |
| LightFM Als | 799 | 7557 | 0.2518 | 0.2641 | 1.0000 |
| LightFM Oracle | 799 | 7557 | 0.8661 | 0.9922 | 0.8892 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 132 | 268 | 0.0009 | 0.0010 | 0.3036 |
| Popularity | 132 | 268 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 132 | 268 | 0.0095 | 0.0032 | 0.0111 |
| LightFM Als | 132 | 268 | 0.0000 | 0.0000 | 0.0000 |
| LightFM Oracle | 132 | 268 | 0.5346 | 0.2689 | 0.0784 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 4 | 4 | 0.0000 | 0.0000 | 0.5011 |
| Popularity | 4 | 4 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 4 | 4 | 0.0000 | 0.0000 | 0.0012 |
| LightFM Als | 4 | 4 | 0.0000 | 0.0000 | 0.0000 |
| LightFM Oracle | 4 | 4 | 0.2500 | 0.0753 | 0.0324 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.00s | 0.00s | n/a | 2.37ms | 3.45ms | 0.00 MiB | 0.00 MiB |
| Popularity | 20.79s | 20.79s | n/a | 0.18ms | 0.42ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 748.81 MiB | 6.69ms | 7.96ms | 26.87 MiB | 14.27 MiB |
| LightFM Als | 268.39s | 268.39s | 748.81 MiB | 6.83ms | 8.02ms | 8.96 MiB | 6.94 MiB |
| LightFM Oracle | 0.00s | 0.00s | n/a | 0.02ms | 0.04ms | 0.00 MiB | 0.00 MiB |

Whole-run process peak RSS: **748.81 MiB**.

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +4.29 percentage points; 95% paired-bootstrap CI [+2.92, +5.64] pp; relative delta +39.45%.
- Delta Recall@10: +4.30 percentage points; 95% paired-bootstrap CI [+2.82, +5.73] pp; relative delta +42.96%.

### LightFM Als vs CountSketch CF

- Delta NDCG@10: +11.08 percentage points; 95% paired-bootstrap CI [+9.55, +12.50] pp; relative delta +73.07%.
- Delta Recall@10: +10.46 percentage points; 95% paired-bootstrap CI [+8.91, +11.99] pp; relative delta +73.13%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 4.29 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM Als versus CountSketch CF:** NDCG@10 is higher by 11.08 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
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
- Split artifact SHA-256: `a668114f043a54dc7048dddc8d5290416579b0eda5abbddc14bc47065c970038`
- Catalog SHA-256: `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75`
- Bootstrap iterations: 2,000
- Evaluation duration: 328.10 seconds
