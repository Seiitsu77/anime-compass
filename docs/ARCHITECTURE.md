# Recommendation Architecture

## What changed, and why

The default ranking path used to be a ten-channel hybrid taking roughly a second
per request. It is now an ALS-first path taking single-digit milliseconds. This
was an evidence-driven complexity reduction, not a feature removal: the hybrid
still exists and still runs, for the requests that need what it does.

The measurements that drove it, all on the same 800 held-out users with
full-catalog ranking:

| Architecture | NDCG@10 | Recall@10 | p50 | p95 |
|---|---:|---:|---:|---:|
| Hybrid + CountSketch (previous default) | 0.1815 | 0.1682 | 924.6 ms | 1186.3 ms |
| **Fast ALS path (new default)** | **0.2588** | **0.2480** | **2.0 ms** | **2.7 ms** |

**+42.6% relative NDCG@10 at roughly 1/465th the latency**, paired 95% interval
[+0.0641, +0.0897], excluding zero.

Four findings produced this:

1. **ALS robustly beat CountSketch** as a collaborative model: +59% to +87%
   relative NDCG@10 across rating thresholds 9, 8, and 7, every interval
   excluding zero.
2. **The hybrid added nothing once ALS was the collaborative channel.**
   Substituting ALS into the hybrid gave 0.2629 against standalone ALS at
   0.2624 — an interval including zero — for 125× the latency.
3. **ALS alone has no tail reach.** Its recommendation exposure is 100% inside
   the top 20% most popular items; mid-tail and long-tail exposure are exactly
   zero. CountSketch and item-item retain small but non-zero tail exposure.
4. **The hybrid's remaining value is not ranking.** Hard filters, entity joins,
   explicit ordering, and evidence-bearing explanations were never scored by the
   personalized benchmark, so the benchmark cannot argue against them.

## Architecture

```text
                          User request
                               |
                               v
                     Resolve session profile
                               |
                               v
                    +----------------------+
                    |  Path policy          |   request semantics, not load
                    |  (path_policy.py)     |
                    +----------------------+
                       |                 |
        unconstrained  |                 |  entity / metadata / similarity
        personalized   |                 |  constraints present
                       v                 v
            +---------------------+   +-----------------------------+
            |   FAST PATH         |   |  CONSTRAINT-RICH PATH       |
            +---------------------+   +-----------------------------+
                       |                             |
                       v                             |
            Collaborative routing                    |
              (routing.py)                           |
                       |                             |
         +-------------+-------------+               |
         | segment_aware = False     |               |
         | (default: global ALS)     |               |
         +-------------+-------------+               |
                       |                             |
                       v                             |
              Candidate retrieval                    |
               (retrieval.py)                        |
                       |                             |
            ALS Top-300  [+ item-item Top-100        |
                          when tail reach matters]   |
                       |                             |
                       v                             |
              dedupe + provenance                    |
            within-source rank scores                |
                       |                             |
                       v                             v
                 hard filters                 entity resolution
              (exclusions, allowed)           exact catalog joins
                       |                      hard constraint filters
                       v                      multi-channel scoring
              lightweight ranking              hand-set fusion weights
           rank score + quality prior          diversity rerank
                       |                      explanations
                       v                             |
        optional bounded diversity rerank            |
                       |                             |
                       +-------------+---------------+
                                     |
                                     v
                              Top-N results
                                     |
                                     v
                     LLM presentation / explanation
```

## Routing rules

### Path selection (`path_policy.py`)

A request takes the **constraint-rich** path when it carries any of:

- required or preferred studios, staff, voice actors, or characters
- resolved entity mentions
- genre include/exclude, formats, `min_score`, year bounds, `max_episodes`
- reference titles (a "more like X" similarity request)
- free-text preference prose
- intent `rank_catalog`, `search`, or `details`

Otherwise it takes the **fast** path. Neither is labelled better; they answer
different questions.

### Collaborative source (`routing.py`)

Default: **global ALS** (`segment_aware = False`).

Segment-aware routing is implemented and configurable but **off by default**,
because measuring it showed it hurts:

| Segment | Users | Global ALS | Segment-routed | Hybrid+CountSketch |
|---|---:|---:|---:|---:|
| Sparse | 30 | **0.2003** | 0.1660 | 0.1788 |
| Medium | 100 | **0.2278** | 0.2278 | 0.2211 |
| Heavy | 670 | **0.2661** | 0.2661 | 0.1757 |

(NDCG@10.) The earlier "no demonstrated ALS gain for sparse users" finding meant
the confidence interval included zero, not that CountSketch was better. Routing
sparse users away from ALS costs them ~17% relative NDCG@10 and buys about 5%
catalog coverage. On 30 sparse users this is not conclusive either way, which is
exactly why the mechanism is retained and configurable rather than deleted.

CountSketch remains loaded at all times as:

- the sparse-user fallback when segment routing is enabled
- the tail-exposure source
- the quality-statistics source for the ALS channel
- the degradation path when the ALS artifact is missing or invalid

## Candidate retrieval configuration

| Setting | Default | Rationale |
|---|---:|---|
| `retrieval_als_top_n` | 300 | Recall@300 0.7932 vs 0.8500 at 500; the smaller pool keeps most of the benefit |
| `retrieval_item_item_top_m` | 0 | Opt-in: costs 9.0% relative NDCG@10 and 4× retrieval latency, buys tail reach |
| `routing_medium_threshold` | 5 | Matches the offline sparse/medium boundary; only used when segmenting is on |
| `routing_segment_aware` | false | Measured worse for sparse users than global ALS |
| `fast_path_diversity_strength` | 0.0 | Costs 5.6% relative NDCG@10; enable when catalog breadth matters |
| `fast_path_diversity_window` | 30 | Bounded window keeps the rerank linear in the window, not quadratic in `limit` |

Retrieval sources are queried independently and merged with deduplication.
Scores from different sources are **never summed directly** — an ALS dot product
and an item-item cosine sum are not comparable. Each source's ranking is
converted to a within-source rank score in (0, 1]; an item found by several
sources keeps the best of them, with every source's rank preserved for
observability.

## Serving the model

```text
offline training (SciPy, evaluation package)
        |
        v
  ALS item factors  ->  als_item_factors.npz  (9.0 MB float32)
        |
        v
production ALSCollaborativeIndex (NumPy only)
        |
        v
   request-time fold-in + scoring
```

The web process imports **only** `backend.anime_agent.als_serving`, which uses
NumPy and nothing else. No SciPy, no evaluation package, no training code enters
the FastAPI runtime. The index is read-only after construction and safe to share
across request threads.

Only item factors are stored. A user vector is reconstructed per request by
folding their positives into item space, so the model serves sessions whose
history it has never seen. Verified against the trained user factors on 2,000
sampled users: cosine mean 0.9995, top-20 ranking overlap 0.9728.

### Two ALS artifacts

| | Evaluation | Production |
|---|---|---|
| File | `als_train_only.npz` | `als_production_item_factors.npz` |
| `artifact_role` | `evaluation` | `production` |
| Trained on | split train positives (24,916,911) | all positives (30,875,410) |
| Withholds held-out positives | yes | no |
| Valid for holdout metrics | **yes** | **no** |
| Valid for serving | yes, but weaker | **yes** |
| Carries | `split_sha256` | `ratings_sha256`, `catalog_ids_sha256` |

They answer different questions and substituting one for the other is silent
and consequential in both directions: serving the evaluation build quietly ships
a model trained on 19% fewer interactions, and measuring against the production
build scores it on interactions it already trained on.

The loader therefore checks `artifact_role` against what the caller asked for.
Production defaults to `ALS_EXPECTED_ROLE=production`; the offline harness keeps
using the evaluation artifact and its published numbers are unaffected.

Rebuild production with:

```powershell
python scripts/build_production_als.py
```

Hyperparameters are pinned inside that script to the frozen validated
configuration and are not exposed as flags, so a production build cannot
silently diverge from the configuration the evidence describes.

### Artifact validation

On startup the artifact must have a supported version, aligned and unique IDs,
finite factors, and at least 90% catalog overlap. Failure behaviour:

| Condition | Severity | Behaviour |
|---|---|---|
| Artifact absent | warning | degrade to CountSketch |
| **Catalog mismatch** (overlap below 90%, or pinned digest differs) | **critical** | **refuse to start** |
| `ALS_EXPECTED_SHA256` set and mismatched | critical | **refuse to start** |
| `ALS_EXPECTED_CATALOG_IDS_SHA256` set and mismatched | critical | **refuse to start** |
| Any validation failure with `ALS_REQUIRE_VALID_ARTIFACT=true` | critical | **refuse to start** |
| Role mismatch, nothing pinned | high | log degradation event, degrade to CountSketch |

A catalog mismatch never degrades quietly. Falling back would keep serving
recommendations from a model that no longer describes the catalog, which is
worse than being down, so it escalates.

Every failure emits a single structured `als_artifact_degradation` event
carrying `severity`, `error_type`, `action` (`refusing_startup` or
`degraded_to_countsketch`), and which pins were set. Nothing trains or rebuilds
at startup.

## Observability

Every fast-path response carries a diagnostics block:

```json
{
  "recommendation_path": "fast",
  "collaborative_route": "als",
  "known_positive_count": 37,
  "reason": "segment_aware_routing_disabled",
  "candidate_pool_size": 300,
  "candidate_sources": {"als": 300},
  "tail_source_used": false,
  "diversity_applied": false,
  "hard_filter_applied": false,
  "stage_latency_ms": {"routing": 0.0, "retrieval": 1.4, "filtering": 0.2,
                       "ranking": 0.3, "reranking": 0.0},
  "artifact_versions": {
    "als_artifact_sha256": "...",
    "routing_config_version": "routing-v2",
    "retrieval_config_version": "retrieval-v2",
    "ranking_config_version": "fastpath-v1"
  }
}
```

Each returned item also carries `candidate_sources`, so which channel surfaced a
recommendation is observable without re-running retrieval. This supports
analysing ALS route acceptance, sparse-fallback frequency, hybrid invocation
rate, and per-source contribution.

## What did not change

- **Semantic channel stays retired** at weight 0.00, available only through
  `experimental_semantic_weights()`. Tests pin this.
- **Hand-set fusion weights stay** in the constraint-rich path. The learned
  blend lost its predeclared confirmation (NDCG@10 0.1814 vs 0.1996) and is not
  promoted.
- **Hard constraints stay hard.** Required entity constraints are never relaxed
  by the bounded replanner, and the fast path re-checks exclusions after ranking
  so a source that ignores them cannot leak an item through.
- **No new model family.** No LightGCN, SASRec, or BERT4Rec. The unresolved
  problems — zero tail exposure and no timestamps — are not ones those solve.
