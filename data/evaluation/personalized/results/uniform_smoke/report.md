# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Run scope: **sampled**; evaluated users: **100**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1035 | 0.0877 | 0.4200 | 0.1184 | 0.1367 | 0.2265 | 0.0039 | 8.415 | 0.1209 | 0.8219 |
| CountSketch CF | 0.1392 | 0.1462 | 0.5300 | 0.1594 | 0.2226 | 0.2541 | 0.0214 | 9.495 | 0.0572 | 0.8251 |
| Current Hybrid | 0.1506 | 0.1586 | 0.5500 | 0.1738 | 0.2413 | 0.2650 | 0.0210 | 9.547 | 0.0541 | 0.7488 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 0 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 0 | 0.0000 | 0.0000 | 0.0000 |
| Current Hybrid | 0 | 0.0000 | 0.0000 | 0.0000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 22 | 0.0785 | 0.1136 | 0.2273 |
| CountSketch CF | 22 | 0.1281 | 0.2273 | 0.3182 |
| Current Hybrid | 22 | 0.1335 | 0.2045 | 0.3182 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 78 | 0.1106 | 0.0803 | 0.4744 |
| CountSketch CF | 78 | 0.1423 | 0.1233 | 0.5897 |
| Current Hybrid | 78 | 0.1554 | 0.1457 | 0.6154 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 100 | 916 | 0.0885 | 0.1037 | 1.0000 |
| CountSketch CF | 100 | 916 | 0.1485 | 0.1398 | 0.9960 |
| Current Hybrid | 100 | 916 | 0.1607 | 0.1514 | 0.9975 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 17 | 34 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 17 | 34 | 0.0000 | 0.0000 | 0.0040 |
| Current Hybrid | 17 | 34 | 0.0000 | 0.0000 | 0.0020 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| Current Hybrid | 0 | 0 | 0.0000 | 0.0000 | 0.0005 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The hybrid uses its ranking-only interface, so deterministic explanation/result-payload construction is also excluded. Incremental build is the model's own stage; end-to-end includes required shared train-statistics/CountSketch stages.

| Model | Incremental build | End-to-end build | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 50.04s | 50.04s | 0.41ms | 0.54ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 9.65ms | 11.49ms | 26.87 MiB | 14.27 MiB |
| Current Hybrid | 7.85s | 123.66s | 1151.58ms | 1604.05ms | 43.48 MiB | 128.12 MiB |

Whole-run process peak RSS: **883.38 MiB**.

### Current hybrid timing stages

The production recommender currently exposes combined candidate-generation/channel-scoring and diversity-reranking timers; it does not separately time each content channel.

| Stage | p50 | p95 |
|---|---:|---:|
| candidate generation and channel scoring | 899.19ms | 1346.59ms |
| diversity reranking | 225.18ms | 267.84ms |
| entity resolution | 0.00ms | 0.00ms |
| total | 1137.12ms | 1590.32ms |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +3.56 percentage points; 95% paired-bootstrap CI [+0.43, +6.90] pp.
- Delta Recall@10: +5.85 percentage points; 95% paired-bootstrap CI [+1.69, +10.53] pp.

### Current Hybrid vs CountSketch CF

- Delta NDCG@10: +1.14 percentage points; 95% paired-bootstrap CI [-1.40, +3.76] pp.
- Delta Recall@10: +1.25 percentage points; 95% paired-bootstrap CI [-3.30, +5.61] pp.

## Interpretation

**This is a deterministic sampled run. Its findings are provisional; run the full eligible-user evaluation before using small differences to select a model.**

- **CountSketch versus popularity:** NDCG@10 is higher by 3.56 pp; the 95% interval excludes zero.
- **Hybrid versus CountSketch:** NDCG@10 is higher by 1.14 pp; the 95% interval includes zero.
- **Latency:** Popularity is fastest. Hybrid p50 is 1151.6 ms.
- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about 92.6 hours for 289,601 eligible users. This is a planning estimate, not a measured full-run duration.
- **Medium users:** Current Hybrid has the highest sampled NDCG@10 (0.1335).
- **Heavy users:** Current Hybrid has the highest sampled NDCG@10 (0.1554).
- **Popularity/long tail:** inspect exposure together with held-out long-tail Recall@10; a model can appear novel simply because the train-only artifact has little evidence for many items.
- **Next model:** the credible CountSketch-over-popularity lift justifies LightFM as the next offline challenger, not a production replacement. Promotion still requires a larger held-out gain with acceptable coverage, bias, and latency.

## What this experiment can tell us

- Personalized ranking quality under deterministic random held-out positive interactions.
- Whether the current collaborative signal adds value over train-only popularity.
- Whether the current hybrid adds value over its CountSketch collaborative channel.
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
- Evaluation duration: 215.49 seconds
