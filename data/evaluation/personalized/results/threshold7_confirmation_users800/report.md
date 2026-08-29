# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **800**; positive threshold: **7**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.0012 | 0.0005 | 0.0125 | 0.0012 | 0.0014 | 0.0039 | 0.5881 | 19.381 | -0.4969 | 0.9107 |
| Popularity | 0.1171 | 0.0843 | 0.5062 | 0.1176 | 0.1182 | 0.2536 | 0.0051 | 8.693 | 0.1284 | 0.8164 |
| CountSketch CF | 0.1531 | 0.1198 | 0.6212 | 0.1551 | 0.1639 | 0.3272 | 0.0859 | 10.109 | 0.0455 | 0.8268 |
| LightFM Als | 0.2862 | 0.2332 | 0.8350 | 0.3051 | 0.3372 | 0.4874 | 0.0871 | 10.234 | 0.0382 | 0.7966 |
| LightFM Oracle | 1.0000 | 0.7823 | 1.0000 | 1.0000 | 0.9291 | 1.0000 | 0.1374 | 11.798 | -0.0533 | 0.8147 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.0076 | 0.0464 | 0.0981 | 0.2001 | 10,624 | 0.5505 | 1858.6 | 0.3334 | 0.8302 |
| Popularity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 92 | 0.9981 | 87089.7 | 0.9586 | 0.8302 |
| CountSketch CF | 0.6033 | 0.8876 | 0.9530 | 0.9846 | 1,552 | 0.9823 | 45742.3 | 0.8758 | 0.8302 |
| LightFM Als | 0.4514 | 0.8979 | 0.9838 | 1.0000 | 1,574 | 0.9657 | 38232.6 | 0.8685 | 0.8302 |
| LightFM Oracle | 0.2492 | 0.5706 | 0.8109 | 0.8979 | 2,482 | 0.9638 | 24013.2 | 0.7770 | 0.8302 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 28 | 0.0000 | 0.0000 | 0.0000 |
| Popularity | 28 | 0.1516 | 0.2143 | 0.2143 |
| CountSketch CF | 28 | 0.1418 | 0.2500 | 0.2500 |
| LightFM Als | 28 | 0.2492 | 0.4286 | 0.4286 |
| LightFM Oracle | 28 | 1.0000 | 1.0000 | 1.0000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 77 | 0.0000 | 0.0000 | 0.0000 |
| Popularity | 77 | 0.0781 | 0.0909 | 0.1429 |
| CountSketch CF | 77 | 0.1570 | 0.2208 | 0.3506 |
| LightFM Als | 77 | 0.2130 | 0.3247 | 0.4805 |
| LightFM Oracle | 77 | 1.0000 | 1.0000 | 1.0000 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| LightFM Random | 695 | 0.0014 | 0.0006 | 0.0144 |
| Popularity | 695 | 0.1200 | 0.0783 | 0.5583 |
| CountSketch CF | 695 | 0.1532 | 0.1033 | 0.6662 |
| LightFM Als | 695 | 0.2958 | 0.2152 | 0.8906 |
| LightFM Oracle | 695 | 1.0000 | 0.7494 | 1.0000 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 799 | 10391 | 0.0005 | 0.0012 | 0.2001 |
| Popularity | 799 | 10391 | 0.0858 | 0.1175 | 1.0000 |
| CountSketch CF | 799 | 10391 | 0.1216 | 0.1534 | 0.9846 |
| LightFM Als | 799 | 10391 | 0.2381 | 0.2878 | 1.0000 |
| LightFM Oracle | 799 | 10391 | 0.7845 | 0.9851 | 0.8979 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 198 | 515 | 0.0000 | 0.0000 | 0.2999 |
| Popularity | 198 | 515 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 198 | 515 | 0.0048 | 0.0026 | 0.0133 |
| LightFM Als | 198 | 515 | 0.0000 | 0.0000 | 0.0000 |
| LightFM Oracle | 198 | 515 | 0.4716 | 0.2479 | 0.0765 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| LightFM Random | 9 | 14 | 0.0000 | 0.0000 | 0.5000 |
| Popularity | 9 | 14 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 9 | 14 | 0.0000 | 0.0000 | 0.0021 |
| LightFM Als | 9 | 14 | 0.0000 | 0.0000 | 0.0000 |
| LightFM Oracle | 9 | 14 | 0.2222 | 0.1667 | 0.0256 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightFM Random | 0.00s | 0.00s | n/a | 2.79ms | 3.91ms | 0.00 MiB | 0.00 MiB |
| Popularity | 21.98s | 21.98s | n/a | 0.30ms | 0.61ms | 0.41 MiB | 0.06 MiB |
| CountSketch CF | 30.82s | 30.82s | 748.74 MiB | 7.03ms | 8.39ms | 26.87 MiB | 14.17 MiB |
| LightFM Als | 374.18s | 374.18s | 748.74 MiB | 6.70ms | 7.80ms | 8.96 MiB | 7.11 MiB |
| LightFM Oracle | 0.00s | 0.00s | n/a | 0.02ms | 0.04ms | 0.00 MiB | 0.00 MiB |

Whole-run process peak RSS: **748.74 MiB**.

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +3.61 percentage points; 95% paired-bootstrap CI [+2.21, +4.95] pp; relative delta +30.80%.
- Delta Recall@10: +3.55 percentage points; 95% paired-bootstrap CI [+2.16, +4.87] pp; relative delta +42.15%.

### LightFM Als vs CountSketch CF

- Delta NDCG@10: +13.30 percentage points; 95% paired-bootstrap CI [+11.92, +14.68] pp; relative delta +86.88%.
- Delta Recall@10: +11.34 percentage points; 95% paired-bootstrap CI [+10.00, +12.76] pp; relative delta +94.68%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 3.61 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM Als versus CountSketch CF:** NDCG@10 is higher by 13.30 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
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
- Split artifact SHA-256: `0217b60ba8968f97786a305e0ed8ea1e5b57a2febf4c74eee0cb3a06911b0abb`
- Catalog SHA-256: `2ef54a712f63eec2adc33f21bd431fc66ab097ba135ab440fd3d773f84668c75`
- Bootstrap iterations: 2,000
- Evaluation duration: 445.00 seconds
