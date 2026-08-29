# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **800**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1116 | 0.0945 | 0.4713 | 0.1237 | 0.1447 | 0.2345 | 0.0060 | 8.430 | 0.1282 | 0.8227 |
| CountSketch CF | 0.1641 | 0.1485 | 0.6025 | 0.1797 | 0.2117 | 0.3225 | 0.0744 | 9.677 | 0.0546 | 0.8262 |
| LightFM Als | 0.2875 | 0.2618 | 0.8175 | 0.3172 | 0.3701 | 0.4795 | 0.0768 | 10.099 | 0.0296 | 0.8021 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 109 | 0.9981 | 74015.4 | 0.9532 | 0.8250 |
| CountSketch CF | 0.6697 | 0.9249 | 0.9689 | 0.9907 | 1,344 | 0.9860 | 42563.4 | 0.8796 | 0.8250 |
| LightFM Als | 0.4452 | 0.9271 | 0.9949 | 1.0000 | 1,388 | 0.9678 | 29625.5 | 0.8547 | 0.8250 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 15 | 0.1333 | 0.1333 | 0.1333 |
| CountSketch CF | 15 | 0.2262 | 0.3333 | 0.3333 |
| LightFM Als | 15 | 0.1347 | 0.3333 | 0.3333 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 114 | 0.0659 | 0.1096 | 0.1754 |
| CountSketch CF | 114 | 0.1755 | 0.2544 | 0.3684 |
| LightFM Als | 114 | 0.2357 | 0.3289 | 0.5175 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 671 | 0.1189 | 0.0911 | 0.5291 |
| CountSketch CF | 671 | 0.1608 | 0.1264 | 0.6483 |
| LightFM Als | 671 | 0.2997 | 0.2487 | 0.8793 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 799 | 8203 | 0.0957 | 0.1121 | 1.0000 |
| CountSketch CF | 799 | 8203 | 0.1491 | 0.1641 | 0.9907 |
| LightFM Als | 799 | 8203 | 0.2666 | 0.2896 | 1.0000 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 129 | 274 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 129 | 274 | 0.0114 | 0.0062 | 0.0080 |
| LightFM Als | 129 | 274 | 0.0000 | 0.0000 | 0.0000 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 5 | 7 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 5 | 7 | 0.0667 | 0.0363 | 0.0013 |
| LightFM Als | 5 | 7 | 0.0000 | 0.0000 | 0.0000 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 20.72s | 20.72s | n/a | 0.20ms | 0.38ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 749.46 MiB | 7.03ms | 8.26ms | 26.87 MiB | 14.27 MiB |
| LightFM Als | 268.39s | 268.39s | 749.46 MiB | 6.91ms | 8.20ms | 8.96 MiB | 6.94 MiB |

Whole-run process peak RSS: **749.46 MiB**.

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +5.25 percentage points; 95% paired-bootstrap CI [+3.93, +6.67] pp; relative delta +46.98%.
- Delta Recall@10: +5.40 percentage points; 95% paired-bootstrap CI [+3.92, +6.98] pp; relative delta +57.12%.

### LightFM Als vs CountSketch CF

- Delta NDCG@10: +12.34 percentage points; 95% paired-bootstrap CI [+10.90, +13.78] pp; relative delta +75.18%.
- Delta Recall@10: +11.33 percentage points; 95% paired-bootstrap CI [+9.81, +12.81] pp; relative delta +76.27%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **CountSketch CF versus Popularity:** NDCG@10 is higher by 5.25 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **LightFM Als versus CountSketch CF:** NDCG@10 is higher by 12.34 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** Popularity has the lowest p50 in this run (0.20 ms).
- **Sparse users:** CountSketch CF has the highest sampled NDCG@10 (0.2262).
- **Medium users:** LightFM Als has the highest sampled NDCG@10 (0.2357).
- **Heavy users:** LightFM Als has the highest sampled NDCG@10 (0.2997).
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
- Evaluation duration: 236.35 seconds
