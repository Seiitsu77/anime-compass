# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation C — popularity-stratified diagnostic**.

Run scope: **diagnostic**; evaluated users: **119**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.1796 | 0.0542 | 0.7395 | 0.1608 | 0.0845 | 0.4259 | 0.0427 | 10.744 | 0.0977 | 0.8354 |
| LightFM Als | 0.2996 | 0.1029 | 0.8739 | 0.2850 | 0.1665 | 0.5196 | 0.0582 | 10.637 | 0.1040 | 0.8140 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 0.4622 | 0.7790 | 0.8618 | 0.9395 | 771 | 0.9816 | 30254.6 | 0.8165 | 0.7188 |
| LightFM Als | 0.3109 | 0.7895 | 0.9756 | 1.0000 | 1,052 | 0.9631 | 22705.1 | 0.8229 | 0.7188 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 1 | 0.0000 | 0.0000 | 0.0000 |
| LightFM Als | 1 | 0.0000 | 0.0000 | 0.0000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 3 | 0.2044 | 0.1667 | 0.3333 |
| LightFM Als | 3 | 0.2044 | 0.1667 | 0.3333 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| CountSketch CF | 115 | 0.1805 | 0.0517 | 0.7565 |
| LightFM Als | 115 | 0.3047 | 0.1021 | 0.8957 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 118 | 3906 | 0.0604 | 0.1742 | 0.9395 |
| LightFM Als | 118 | 3906 | 0.1288 | 0.3083 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 100 | 792 | 0.0113 | 0.0144 | 0.0508 |
| LightFM Als | 100 | 792 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| CountSketch CF | 100 | 176 | 0.0000 | 0.0000 | 0.0097 |
| LightFM Als | 100 | 176 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| CountSketch CF | 65.77s | 65.77s | 748.67 MiB | 8.76ms | 12.74ms | 26.87 MiB | 14.27 MiB |
| LightFM Als | 268.39s | 268.39s | 748.67 MiB | 9.00ms | 10.25ms | 8.96 MiB | 6.94 MiB |

Whole-run process peak RSS: **748.67 MiB**.

## Paired statistical comparisons

### LightFM Als vs CountSketch CF

- Delta NDCG@10: +12.00 percentage points; 95% paired-bootstrap CI [+7.99, +15.91] pp; relative delta +66.80%.
- Delta Recall@10: +4.87 percentage points; 95% paired-bootstrap CI [+3.41, +6.44] pp; relative delta +89.83%.

## Interpretation

**Diagnostic sample:** its unweighted aggregate metrics are not estimates of whole-population performance. Use only the segment or popularity-stratum cuts that this evaluation was designed to inspect.

- **LightFM Als versus CountSketch CF:** NDCG@10 is higher by 12.00 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** CountSketch CF has the lowest p50 in this run (8.76 ms).
- **Sparse users:** CountSketch CF has the highest sampled NDCG@10 (0.0000).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.2044).
- **Heavy users:** LightFM Als has the highest sampled NDCG@10 (0.3047).
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
- Evaluation duration: 129.02 seconds
