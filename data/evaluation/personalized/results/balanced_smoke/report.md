# Personalized Offline Recommendation Benchmark

> Because interaction timestamps are unavailable, this evaluation measures preference reconstruction/generalization under a deterministic user-stratified random holdout, not chronological next-item prediction.

Run scope: **sampled**; evaluated users: **30**; positive threshold: **8**; split seed: **42**.

## Ranking and beyond-accuracy results

| Model | NDCG@10 | Recall@10 | HR@10 | NDCG@20 | Recall@20 | MRR@20 | Coverage | Novelty (bits) | Popularity bias | ILD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.0975 | 0.0665 | 0.3000 | 0.1104 | 0.1442 | 0.2187 | 0.0027 | 8.344 | 0.1221 | 0.8278 |
| CountSketch CF | 0.1376 | 0.2186 | 0.4333 | 0.1504 | 0.2598 | 0.1599 | 0.0147 | 9.905 | 0.0300 | 0.8330 |
| Current Hybrid | 0.1673 | 0.2167 | 0.4333 | 0.1942 | 0.3349 | 0.2091 | 0.0158 | 9.925 | 0.0287 | 0.7323 |

## Performance by training-positive user activity

### Sparse users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 10 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 10 | 0.1315 | 0.2000 | 0.2000 |
| Current Hybrid | 10 | 0.1920 | 0.3000 | 0.3000 |

### Medium users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 10 | 0.0920 | 0.1000 | 0.2000 |
| CountSketch CF | 10 | 0.1635 | 0.3500 | 0.5000 |
| Current Hybrid | 10 | 0.1038 | 0.1500 | 0.3000 |

### Heavy users

| Model | Users | NDCG@10 | Recall@10 | HR@10 |
|---|---:|---:|---:|---:|
| Popularity | 10 | 0.2006 | 0.0995 | 0.7000 |
| CountSketch CF | 10 | 0.1177 | 0.1057 | 0.6000 |
| Current Hybrid | 10 | 0.2060 | 0.2001 | 0.7000 |

## Performance by held-out item popularity

### Head

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 28 | 186 | 0.0732 | 0.1051 | 1.0000 |
| CountSketch CF | 28 | 186 | 0.2390 | 0.1489 | 0.9650 |
| Current Hybrid | 28 | 186 | 0.2370 | 0.1816 | 0.9750 |

### Mid-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 6 | 12 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 6 | 12 | 0.0000 | 0.0000 | 0.0317 |
| Current Hybrid | 6 | 12 | 0.0000 | 0.0000 | 0.0200 |

### Long-Tail

| Model | Users with relevant items | Held-out items | Recall@10 | NDCG@10 | Recommendation exposure |
|---|---:|---:|---:|---:|---:|
| Popularity | 1 | 1 | 0.0000 | 0.0000 | 0.0000 |
| CountSketch CF | 1 | 1 | 0.0000 | 0.0000 | 0.0033 |
| Current Hybrid | 1 | 1 | 0.0000 | 0.0000 | 0.0050 |

The buckets are defined only from training positives: head is the top 20% of catalog items by count, mid-tail the next 30%, and long-tail the bottom 50%, including items with no training positives.

## Engineering metrics

Recommendation latency excludes HTTP, frontend rendering, LLM generation, and external APIs. Memory is the resident NumPy array footprint attributable to each loaded model; the run manifest separately records process peak RSS. The hybrid uses its ranking-only interface, so deterministic explanation/result-payload construction is also excluded. Incremental build is the model's own stage; end-to-end includes required shared train-statistics/CountSketch stages.

| Model | Incremental build | End-to-end build | p50 inference | p95 inference | Array memory | Artifact size |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 40.51s | 40.51s | 0.44ms | 0.66ms | 0.41 MiB | 0.05 MiB |
| CountSketch CF | 65.77s | 65.77s | 9.85ms | 11.21ms | 26.87 MiB | 14.27 MiB |
| Current Hybrid | 8.67s | 114.95s | 1193.57ms | 1800.56ms | 43.48 MiB | 128.12 MiB |

Whole-run process peak RSS was unavailable on this platform; per-model NumPy footprints remain reported.

### Current hybrid timing stages

The production recommender currently exposes combined candidate-generation/channel-scoring and diversity-reranking timers; it does not separately time each content channel.

| Stage | p50 | p95 |
|---|---:|---:|
| candidate generation and channel scoring | 911.38ms | 1510.20ms |
| diversity reranking | 242.28ms | 291.58ms |
| entity resolution | 0.00ms | 0.00ms |
| total | 1179.91ms | 1788.43ms |

## Paired statistical comparisons

### CountSketch CF vs Popularity

- Delta NDCG@10: +4.01 percentage points; 95% paired-bootstrap CI [-3.24, +13.69] pp.
- Delta Recall@10: +15.21 percentage points; 95% paired-bootstrap CI [+4.02, +29.11] pp.

### Current Hybrid vs CountSketch CF

- Delta NDCG@10: +2.97 percentage points; 95% paired-bootstrap CI [-3.78, +9.91] pp.
- Delta Recall@10: -0.19 percentage points; 95% paired-bootstrap CI [-13.25, +12.60] pp.

## Interpretation

**This is a deterministic sampled run. Its findings are provisional; run the full eligible-user evaluation before using small differences to select a model.**

**This activity-balanced sample intentionally gives each user segment equal quota. Aggregate metrics are unweighted and therefore are not estimates of whole-population performance; use the uniform sample for the primary aggregate comparison.**

- **CountSketch versus popularity:** NDCG@10 is higher by 4.01 pp; the 95% interval includes zero.
- **Hybrid versus CountSketch:** NDCG@10 is higher by 2.97 pp; the 95% interval includes zero.
- **Latency:** Popularity is fastest. Hybrid p50 is 1193.6 ms.
- **Full-run bottleneck:** a simple serial extrapolation from sampled hybrid p50 is about 96.0 hours for 289,601 eligible users. This is a planning estimate, not a measured full-run duration.
- **Sparse users:** Current Hybrid has the highest sampled NDCG@10 (0.1920).
- **Medium users:** CountSketch CF has the highest sampled NDCG@10 (0.1635).
- **Heavy users:** Current Hybrid has the highest sampled NDCG@10 (0.2060).
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
- Evaluation duration: 98.97 seconds
