# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Evaluation: **Evaluation A — representative uniform user sample**.

Run scope: **sampled**; evaluated users: **800**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current Hybrid | 0.1996 | 0.1873 | 0.6737 | 0.2155 | 0.2550 | 0.3660 | 0.0509 | 9.679 | 0.0519 | 0.7463 |
| LightFM Current Hybrid Learned | 0.1814 | 0.1608 | 0.6338 | 0.1958 | 0.2235 | 0.3461 | 0.0742 | 10.109 | 0.0266 | 0.7932 |

## Recommendation popularity concentration

Popularity ranks and profile comparisons use positive training interactions only. Exposure Gini includes every catalog item, including items that receive zero recommendations.

| Model | Top 1% share | Top 5% | Top 10% | Top 20% | Unique items | Exposure Gini | Avg train count | Rec profile popularity | User profile popularity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current Hybrid | 0.6351 | 0.9356 | 0.9826 | 0.9956 | 919 | 0.9891 | 41296.4 | 0.8794 | 0.8275 |
| LightFM Current Hybrid Learned | 0.5271 | 0.8904 | 0.9549 | 0.9814 | 1,340 | 0.9830 | 34846.7 | 0.8541 | 0.8275 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Current Hybrid | 15 | 0.2041 | 0.2667 | 0.2667 |
| LightFM Current Hybrid Learned | 15 | 0.2495 | 0.3333 | 0.3333 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Current Hybrid | 113 | 0.2490 | 0.3274 | 0.4956 |
| LightFM Current Hybrid Learned | 113 | 0.2229 | 0.2832 | 0.4425 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Current Hybrid | 672 | 0.1912 | 0.1620 | 0.7128 |
| LightFM Current Hybrid Learned | 672 | 0.1730 | 0.1364 | 0.6726 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Current Hybrid | 799 | 8504 | 0.1895 | 0.2006 | 0.9956 |
| LightFM Current Hybrid Learned | 799 | 8504 | 0.1627 | 0.1819 | 0.9814 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Current Hybrid | 126 | 475 | 0.0080 | 0.0040 | 0.0036 |
| LightFM Current Hybrid Learned | 126 | 475 | 0.0096 | 0.0067 | 0.0141 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Current Hybrid | 4 | 14 | 0.0000 | 0.0000 | 0.0008 |
| LightFM Current Hybrid Learned | 4 | 14 | 0.0000 | 0.0000 | 0.0046 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The current hybrid uses its ranking-only interface when present. LightFM latency uses exported NumPy arrays and does not include the native training dependency. Selected fit is the winning candidate's fit time; offline total includes all validation-search candidates. Peak RSS is the trainer process peak where available.

| Model | Selected fit/build | Offline total | Peak RSS | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current Hybrid | 5.29s | 92.42s | n/a | 789.07ms | 1284.44ms | 43.48 MiB | 128.12 MiB |
| LightFM Current Hybrid Learned | 5.31s | 5.31s | n/a | 785.00ms | 1275.13ms | 43.48 MiB | 128.12 MiB |

Whole-run process peak RSS: **1696.17 MiB**.

### Current hybrid timing stages

The production recommender currently exposes combined candidate-generation/channel-scoring and diversity-reranking timers; it does not separately time each content channel.

| Stage | p50 | p95 |
|---|---:|---:|
| candidate generation and channel scoring | 619.34ms | 1114.77ms |
| diversity reranking | 155.03ms | 174.34ms |
| entity resolution | 0.00ms | 0.00ms |
| total | 782.39ms | 1277.92ms |

## Paired statistical comparisons

### LightFM Current Hybrid Learned vs Current Hybrid

- Delta NDCG@10: -1.81 percentage points; 95% paired-bootstrap CI [-2.69, -0.92] pp; relative delta -9.08%.
- Delta Recall@10: -2.65 percentage points; 95% paired-bootstrap CI [-3.72, -1.58] pp; relative delta -14.15%.

## Interpretation

**This is a deterministic representative sample. The paired intervals quantify user-level uncertainty inside this sample; confirm borderline decisions on a predeclared larger sample rather than assuming a full-population run is necessary.**

- **LightFM Current Hybrid Learned versus Current Hybrid:** NDCG@10 is lower by 1.81 pp; the 95% interval excludes zero, providing evidence of a difference in this evaluation.
- **Latency:** LightFM Current Hybrid Learned has the lowest p50 in this run (785.00 ms).
- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about 63.5 hours for 289,601 eligible users. This is a planning estimate, not a measured full-run duration.
- **Sparse users:** LightFM Current Hybrid Learned has the highest sampled NDCG@10 (0.2495).
- **Medium users:** Current Hybrid has the highest sampled NDCG@10 (0.2490).
- **Heavy users:** Current Hybrid has the highest sampled NDCG@10 (0.1912).
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
- Evaluation duration: 1518.65 seconds
