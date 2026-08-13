# Recommender Proxy Evaluation

These offline metrics measure catalog consistency and recovery against a small, manually curated benchmark. They do not measure real user satisfaction.

Catalog: 18064 titles. Benchmark: 7 cases. K=10.

| model | hard filters | entity constraints | Hit Rate@K | genre recovery | coverage | diversity | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 1.000 | 1.000 | 0.200 | 0.847 | 0.002 | 0.754 | 145.0 | 508.4 |
| metadata_tfidf | 1.000 | 1.000 | 0.200 | 0.764 | 0.004 | 0.565 | 3009.3 | 3414.0 |
| synopsis_tfidf | 1.000 | 1.000 | 0.200 | 0.806 | 0.004 | 0.831 | 3054.0 | 3356.1 |
| lsa | 1.000 | 1.000 | 0.400 | 0.903 | 0.004 | 0.835 | 3020.4 | 3445.0 |
| collaborative | 1.000 | 1.000 | 0.600 | 0.736 | 0.004 | 0.789 | 3072.3 | 3462.7 |
| final_hybrid | 1.000 | 1.000 | 1.000 | 0.903 | 0.004 | 0.737 | 3149.7 | 3462.0 |

## Metric Definitions

- Hard filters: fraction of returned titles satisfying explicit genre, format, score, year, episode, and exclusion constraints.
- Entity constraints: fraction of results with the required catalog relationship, measured only on entity cases.
- Hit Rate@K: fraction of labeled similarity cases where at least one manually expected title family appears in the top K.
- Genre recovery: fraction of expected genre labels present anywhere in each result list.
- Coverage: unique titles returned across the benchmark divided by catalog size.
- Diversity: mean pairwise genre Jaccard distance within each list.
- Latency: local recommender execution only; model loading and API/LLM latency are excluded.

The benchmark is intentionally a proxy. Its labels are small and subjective, so differences should be treated as engineering diagnostics rather than evidence of user preference quality.
