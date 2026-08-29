# Recommender Proxy Evaluation

These offline metrics measure catalog consistency and recovery against a small, manually curated benchmark. They do not measure real user satisfaction.

Catalog: 18064 titles. Benchmark: 7 cases. K=10.

| model | hard filters | entity constraints | Hit Rate@K | genre recovery | coverage | diversity | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 1.000 | 1.000 | 0.200 | 0.847 | 0.002 | 0.754 | 227.4 | 592.3 |
| metadata_tfidf | 1.000 | 1.000 | 0.200 | 0.764 | 0.004 | 0.565 | 3223.8 | 3557.3 |
| synopsis_tfidf | 1.000 | 1.000 | 0.200 | 0.806 | 0.004 | 0.831 | 3198.6 | 3642.1 |
| lsa | 1.000 | 1.000 | 0.400 | 0.903 | 0.004 | 0.835 | 3207.3 | 3980.9 |
| pretrained_semantic | 1.000 | 1.000 | 0.200 | 0.903 | 0.004 | 0.750 | 3367.7 | 3748.1 |
| collaborative | 1.000 | 1.000 | 0.600 | 0.736 | 0.004 | 0.789 | 3533.3 | 4086.3 |
| final_hybrid | 1.000 | 1.000 | 0.800 | 0.903 | 0.004 | 0.728 | 3509.0 | 3894.6 |

## Metric Definitions

- Hard filters: fraction of returned titles satisfying explicit genre, format, score, year, episode, and exclusion constraints.
- Entity constraints: fraction of results with the required catalog relationship, measured only on entity cases.
- Hit Rate@K: fraction of labeled similarity cases where at least one manually expected title family appears in the top K.
- Genre recovery: fraction of expected genre labels present anywhere in each result list.
- Coverage: unique titles returned across the benchmark divided by catalog size.
- Diversity: mean pairwise genre Jaccard distance within each list.
- Latency: local recommender execution only; model loading and API/LLM latency are excluded.

The benchmark is intentionally a proxy. Its labels are small and subjective, so differences should be treated as engineering diagnostics rather than evidence of user preference quality.
