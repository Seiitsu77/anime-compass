# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation B — activity-balanced diagnostic**.

Run scope: **diagnostic**; evaluated users: **300**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.1558 | 0.1895 | 0.4000 | 0.1699 | 0.2435 | 0.2145 | 0.0747 | 10.182 | 0.0186 | 0.8201 |
| LightFM Als | 0.2335 | 0.2854 | 0.5233 | 0.2596 | 0.3794 | 0.2984 | 0.0667 | 10.267 | 0.0137 | 0.7703 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.5705 | 0.8387 | 0.9160 | 0.9665 | 1,349 | 0.9759 | 37896.3 | 0.8497 | 0.8311 |
| LightFM Als | 0.4062 | 0.8882 | 0.9827 | 1.0000 | 1,204 | 0.9681 | 28072.5 | 0.8447 | 0.8311 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 100 | 0.1611 | 0.2400 | 0.2400 |
| LightFM Als | 100 | 0.1836 | 0.2900 | 0.2900 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 100 | 0.1714 | 0.2200 | 0.3700 |
| LightFM Als | 100 | 0.2501 | 0.3400 | 0.4900 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 100 | 0.1347 | 0.1086 | 0.5900 |
| LightFM Als | 100 | 0.2666 | 0.2261 | 0.7900 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 298 | 1352 | 0.1936 | 0.1582 | 0.9665 |
| LightFM Als | 298 | 1352 | 0.2901 | 0.2358 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 27 | 45 | 0.0000 | 0.0000 | 0.0288 |
| LightFM Als | 27 | 45 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 2 | 2 | 0.0000 | 0.0000 | 0.0047 |
| LightFM Als | 2 | 2 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 65.77s | 65.77s | 748.76 MiB | 7.72ms | 9.77ms | 26.87 MiB | 14.27 MiB |
| LightFM Als | 268.39s | 268.39s | 748.76 MiB | 7.08ms | 9.25ms | 8.96 MiB | 6.94 MiB |

Whole-run process peak RSS: **748.76 MiB**.

## Paired statistical comparisons

### LightFM Als vs CountSketch CF

- Delta NDCG@10: +7.77 percentage points; 95% paired-bootstrap CI [+4.84, +10.58] pp; relative delta +49.88%.
- Delta Recall@10: +9.58 percentage points; 95% paired-bootstrap CI [+5.67, +13.48] pp; relative delta +50.57%.

## Interpretation

**Diagnostic sample:** its unweighted aggregate metrics are not estimates of whole-population performance. Use only the segment or popularity-stratum cuts that this evaluation was designed to inspect.

- **LightFM Als versus CountSketch CF:** NDCG@10 is higher by 7.77 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** LightFM Als has the lowest p50 in this run (7.08 ms).
- **Sparse users:** LightFM Als has the highest sampled NDCG@10 (0.1836).
- **Medium users:** LightFM Als has the highest sampled NDCG@10 (0.2501).
- **Heavy users:** LightFM Als has the highest sampled NDCG@10 (0.2666).
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
- Evaluation duration: 81.87 seconds
