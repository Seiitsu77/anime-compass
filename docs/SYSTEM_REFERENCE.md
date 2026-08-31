# System Reference

Detailed implementation notes moved out of the README so it stays readable.
For the architecture rationale see [ARCHITECTURE.md](ARCHITECTURE.md); for the
experiment record see the reports under
`data/evaluation/personalized/results/`.

## Agent Tool Flow

`AgentIntent` distinguishes six operations:

| Intent | Backend behavior |
|---|---|
| `search` | Relevance-oriented catalog/entity lookup |
| `rank_catalog` | Deterministic filtering and explicit sorting, independent of session taste |
| `recommend` | Preference-aware hybrid content recommendation |
| `details` | Catalog-grounded, spoiler-light anime introduction |
| `update_preferences` | Anonymous session feedback update |
| `conversation` | No catalog tool execution |

```text
user message
  -> Gemini/Ollama returns structured AgentIntent
  -> Pydantic validates and normalizes every field
  -> backend derives an allowlisted tool plan
  -> backend resolves entities and executes catalog joins/filters
  -> tool trace is validated against its typed contract
  -> provider may verbalize only the verified tool payload
  -> backend rejects ungrounded title lists
```

Available tools are `search_anime`, `search_entities`, `resolve_entity`, `rank_catalog`, `recommend_anime`, `get_anime_details`, and `update_session_preferences`.

### Bounded replanning

A strict filter chain can be over-constrained: "a 2015-only Madhouse isekai under 12 episodes rated above 8.5" is
a reasonable request that matches no catalog row. When a retrieval intent returns zero rows, the backend replans
**deterministically** rather than asking the model to retry, which would let it invent titles. Constraints are
dropped one at a time along a fixed ladder — score floor, episode cap, year bounds, formats, preferred entities,
excluded genres, required genres — and the tools re-execute. Two invariants hold:

- **Required entity constraints are never relaxed.** A request pinned to a voice actor, studio, staff member, or
  character is satisfied exactly or returns nothing. Relaxing those would break the grounding contract.
- **The loop is bounded** by `MAX_REPLAN_STEPS` (default 2, `0` disables), so cost stays predictable.

Every applied relaxation is recorded and returned on the response as `relaxations`, so the answer can state what
was loosened instead of silently widening the request.

For a request such as "recommend 7 anime with Yoshitsugu Matsuoka," the backend resolves the voice-actor record, joins its related anime IDs, filters to that verified subset, and only then ranks. Each result includes `matched_voice_actors`, character, language, entity ID, and relationship evidence.

## Recommendation Architecture

The default ranking path is **ALS-first and fast**. The ten-channel hybrid still exists and still runs, but only
for requests that need deterministic constraint handling. On the same 800 held-out users, full-catalog ranking:

| Architecture | NDCG@10 | Recall@10 | p50 | p95 |
|---|---:|---:|---:|---:|
| Hybrid + CountSketch (previous default) | 0.1815 | 0.1682 | 924.6 ms | 1186.3 ms |
| **Fast ALS path (current default)** | **0.2588** | **0.2480** | **2.0 ms** | **2.7 ms** |

**+42.6% relative NDCG@10 at roughly 1/465th the latency**, paired 95% CI `[+0.0641, +0.0897]`.

```text
request -> path policy -> [fast: ALS retrieval -> filters -> lightweight ranking]
                       -> [constraint-rich: entity joins -> hard filters -> hybrid channels]
```

A request takes the constraint-rich path when it names entities, metadata filters, reference titles, or free-text
preferences. Otherwise it takes the fast path. Neither path is "better" — they answer different questions, and
the personalized benchmark never scored what the rich path exists to do.

This was **evidence-driven complexity reduction**, not feature removal:

- ALS beat CountSketch robustly across rating thresholds 7, 8, and 9 (+59% to +87% relative NDCG@10).
- Substituting ALS into the hybrid gave no measurable ranking gain over standalone ALS (interval includes zero)
  for 125x the latency.
- ALS alone has **zero** mid-tail and long-tail exposure, so CountSketch and item-item are retained as
  complementary sources rather than deleted.

Full detail, routing rules, and configuration are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Recommendation Model

The constraint-rich path uses a **multi-channel hybrid collaborative/content recommender with session-based
personalization**. The fast default path uses ALS retrieval plus a lightweight rank-and-quality blend; the table
below describes the hybrid.

| Channel | Default weight | Source |
|---|---:|---|
| Metadata TF-IDF | 0.16 | genres, format, studios, people, catalog labels |
| Synopsis TF-IDF | 0.10 | synopsis terms and story themes |
| LSA | 0.04 | deterministic truncated decomposition of content features |
| Pretrained semantic | **0.00** | retired by default; optional `all-MiniLM-L6-v2` experiment |
| Hashed dense text | 0.08 | local token, character n-gram, and bigram features |
| Creator similarity | 0.05 | studios and selected staff roles |
| Collaborative | 0.22 | user-centred item vectors from completed-title ratings |
| Quality prior | 0.13 | Bayesian rating mean, aggregate score, members, and popularity |
| Session signal | 0.05 | anonymous likes, dislikes, watched titles, and preferences |
| Novelty | 0.03 | mainstream or less-famous preference |

Weights are renormalized across channels that are active for a request. Hard filters and required entity relationships are applied first.

> **The former 0.14 semantic weight was retired.** With the artifact built and the channel active, the hybrid
> lost 8.9% relative NDCG@10 and 11.9% relative Recall@10 on 300 held-out users, both intervals excluding zero.
> The code retains an explicit opt-in weight only so the hypothesis stays reproducible. See the
> [semantic channel report](../data/evaluation/personalized/results/semantic_channel_summary.md).

```text
effective_weight[c] = configured_weight[c] / sum(configured weights of active channels)
pre_diversity_score = sum(effective_weight[c] * normalized_score[c])
final_score = pre_diversity_score + diversity_adjustment
```

Every API recommendation exposes raw channel scores, configured and effective weights, weighted contributions, and the final diversity adjustment. The collaborative artifact uses randomized user-dimension projections to approximate adjusted-cosine item similarity; it is not mislabeled content LSA or matrix-factorization SVD.

### Collaborative Artifact

`rating_complete.csv` contains only completed-and-rated titles. Training user-centres each rating vector and projects the sparse user dimension into three independently signed CountSketch blocks. The web process loads only normalized item vectors and Bayesian item statistics, not the 818 MB interaction file.

```powershell
python scripts/build_collaborative_model.py
```

The generated artifact contains 18,064 aligned IDs, 384-dimensional vectors, and statistics learned from 56,726,861 in-scope ratings across 310,059 users. Startup validates version, shape, finite values, score ranges, unique IDs, and catalog overlap.

### Semantic Artifact

The optional semantic provider is `sentence-transformers/all-MiniLM-L6-v2`, pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. A semantic artifact stores normalized 384-dimensional vectors plus model, preprocessing, ID-map, and catalog-checksum metadata. Startup rejects stale or incompatible artifacts rather than silently using them.

The artifact is listed in `data/artifacts.manifest.json` as `required: false`, so a deployment whose dataset repo
does not carry it starts anyway and serves the remaining channels rather than failing.

Build or verify it with:

```powershell
python scripts/build_semantic_embeddings.py --force
python scripts/build_semantic_embeddings.py --verify-only --offline
```

## Evaluation

There are two deliberately separate evaluation layers:

- The existing seven-case benchmark protects catalog constraints, agent routing, and qualitative recommendation
  behavior.
- The personalized offline benchmark uses a deterministic, leakage-safe per-user positive holdout over the
  anonymous rating archive. It compares train-only popularity, CountSketch CF, exact item-item cosine, implicit
  ALS, LightFM-ID, and LightFM-Hybrid on identical users with NDCG/Recall/HR/MRR,
  coverage/novelty/bias/diversity, activity and long-tail diagnostics, threshold sensitivity, paired bootstrap
  intervals, and engineering costs.

The personalized methodology, commands, artifacts, and limitations are documented in
[personalized evaluation README](../data/evaluation/personalized/README.md). ALS has since been promoted to the
default personalized path; CountSketch and item-item remain optional comparison and tail-retrieval sources.

### Reading these numbers

**Full-catalog NDCG@10 is not comparable to published NDCG@10.** Most reported figures in the 0.5-0.7 range come
from a sampled-negative protocol: hold out one interaction, sample 99 unseen items, rank those 100. This project
ranks the entire 18,064-item catalog and counts every held-out positive, because that is what the product actually
does. The same ALS model on the same 800 users:

| Protocol | NDCG@10 |
|---|---:|
| All test positives, full 18,064-item catalog | **0.2875** (reported here) |
| Leave-one-out, full catalog | 0.1775 |
| Leave-one-out + 99 sampled negatives | **0.8260** (typical paper protocol) |

Nothing about the model changes between rows. Sampled negatives are reported only to make the scale explicit and
are never used for selection, since they can reorder which model appears better. In practical terms: **HR@10 is
0.8175**, so 82% of users get at least one held-out favourite in the top 10 of 18,064 titles, and Recall@10 of
0.2618 sits against a hard ceiling of 0.8556 because a third of users have more than ten held-out positives.
Full detail, and what genuinely limits the score, is in
[metric comparability](../data/evaluation/personalized/results/metric_comparability.md).

```powershell
python scripts/compare_evaluation_protocols.py
```

### Personalized collaborative benchmark

The primary result is a deterministic, representative 1,000-user sample from the full-data train-only split
(`rating >= 8`, seed 42). Candidate catalog, known-item filtering, metrics, and held-out users are identical
across models.

| Model | NDCG@10 | Recall@10 | HR@10 | Coverage | Pop. bias | ILD | p50 rank latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 0.1023 | 0.0881 | 0.4420 | 0.0059 | 0.1297 | 0.8222 | 0.22 ms |
| CountSketch CF (previous default) | 0.1534 | 0.1408 | 0.5840 | 0.0775 | 0.0579 | **0.8264** | 6.93 ms |
| Exact item-item cosine | 0.1650 | 0.1472 | 0.6010 | 0.0375 | 0.0898 | 0.8259 | 5.72 ms |
| **ALS (implicit)** | **0.1841** | **0.1908** | **0.6810** | **0.0989** | **0.0377** | 0.7985 | 6.87 ms |
| LightFM-ID | 0.1833 | 0.1589 | 0.6430 | 0.0380 | 0.0867 | 0.7990 | 1.19 ms |
| LightFM-Hybrid | 0.1747 | 0.1483 | 0.6250 | 0.0421 | 0.0775 | 0.7770 | 1.16 ms |

Lower popularity bias is better. Two findings drive the current decision:

- **The CountSketch projection is not free.** Exact item-item cosine uses an identical residual transform on
  identical inputs, so the only difference is the absence of the random projection. It is +7.5% relative NDCG@10
  better (paired 95% CI `[+0.0036, +0.0194]`). The sketch's justification is its build-time memory profile, not
  fidelity. Its coverage advantage turns out to come partly from the noise the projection introduces.
- **ALS clears every gate LightFM failed.** It beats the production model by +20.0% relative NDCG@10 and +35.5%
  relative Recall@10 (both intervals exclude zero), while *improving* catalog coverage by 27.6% and *reducing*
  popularity bias by 34.9%. It is statistically tied with LightFM-ID on NDCG@10 and decisively better on
  Recall@10 (+16.7% relative). Intra-list diversity regresses 3.4%, the one real trade.

**ALS has since been tuned and confirmed.** A validation-only sweep over 12 candidates found the stock-like
configuration was badly set: `alpha` dominates and lower is better (validation NDCG@10 rises from 0.1486 at
alpha=100 to 0.2733 at alpha=2.5), while coverage moves the opposite way, so raising factors to 128 was needed to keep both.
On the same validation users, the 64-factor, alpha-40 candidate scored 0.2032 and the selected 128-factor,
alpha-5 candidate scored 0.2787 (+37.2%).
On 800 predeclared users that no earlier run had scored, the selected configuration (128 factors, alpha 5.0)
beats the production collaborative channel by **+75.2% relative NDCG@10** and **+76.3% relative Recall@10**, both
intervals excluding zero, at 103% of its coverage and 46% lower popularity bias. Serving-time fold-in was verified
faithful (cosine 0.9995 against trained user factors).

Subsequent threshold-7/9, activity, tail, and production-architecture experiments completed the promotion work.
They showed a robust relevance gain, zero tail exposure for ALS, and no measurable ranking gain from keeping the
full hybrid on unconstrained requests. The shipped policy is therefore global ALS for the default path, with the
rich hybrid reserved for explicit constraints. See the [ALS confirmation report](../data/evaluation/personalized/results/als_confirmation_summary.md),
[promotion evidence](../data/evaluation/personalized/results/als_promotion_decision.md), and
[architecture rationale](ARCHITECTURE.md).

Offline training uses NumPy/SciPy implementations of exact blockwise adjusted-cosine similarity and
conjugate-gradient implicit ALS. The exported production artifact contains item factors only, so serving is
NumPy-only and does not load SciPy or a model-training framework.

```powershell
python -m pip install -r requirements-evaluation.txt
python scripts/evaluate_personalized.py --models popularity,countsketch_cf,item_item_cosine,als
```

### Learned fusion weights

The ten channel weights above are hand-set constants. Because the scorer is linear in its channel signals, they
can be fitted instead: `scripts/train_fusion_weights.py` optimises a RankNet-style pairwise logistic loss over
held-out positives, under a non-negativity projection so the learned vector stays in the space the serving path
already accepts.

Fitted on 400 validation users and scored **once** on 400 disjoint test users:

| Blend | Pairwise accuracy (held-out) |
|---|---:|
| Hand-set | **0.7284** |
| Learned | 0.7225 |

**The learned blend lost, so the hand-set weights stand.** Training loss did fall (0.6817 to 0.6727), so the fit
worked; it simply did not generalise better. The most likely reason is that ten weights fitted on 18k pairs from
376 users, over channels that are far from independent, can shift mass between near-collinear text signals without
improving the ranking.

The direction is still informative: the fit moves weight toward collaborative (+0.14) and creator (+0.21) signals
and away from text similarity, which matches what the collaborative benchmark shows. Two caveats bound that
reading — the semantic channel was inactive on the fitting machine, so its zeroed weight is an artifact of the run
rather than a verdict on pretrained embeddings, and 43% of held-out positives never entered the 300-item
shortlist, so the fit optimises ordering within the retrieved region rather than retrieval itself. Details and
follow-ups are in the [learned fusion report](../data/evaluation/personalized/results/learned_fusion_summary.md).

### Catalog/agent regression benchmark

The benchmark compares six actual configurations at `K=10`. One-channel runs set every unrelated channel weight to zero; the popularity baseline sorts the filtered candidate set directly.

| Model | Hard filters | Entity constraints | Hit Rate@10 | Genre recovery | Coverage | Diversity | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 1.000 | 1.000 | 0.200 | 0.847 | 0.002 | 0.754 | 227.4 |
| Metadata TF-IDF | 1.000 | 1.000 | 0.200 | 0.764 | 0.004 | 0.565 | 3223.8 |
| Synopsis TF-IDF | 1.000 | 1.000 | 0.200 | 0.806 | 0.004 | 0.831 | 3198.6 |
| LSA | 1.000 | 1.000 | 0.400 | **0.903** | 0.004 | 0.835 | 3207.3 |
| Pretrained semantic | 1.000 | 1.000 | 0.200 | **0.903** | 0.004 | 0.750 | 3367.7 |
| Collaborative | 1.000 | 1.000 | **0.600** | 0.736 | 0.004 | 0.789 | 3533.3 |
| Final hybrid | 1.000 | 1.000 | 0.800 | **0.903** | 0.004 | 0.728 | 3509.0 |

This table previously reported a 1.000 hybrid hit rate measured **without** the semantic channel, which had never
been built. With all ten channels active it is 0.800. At five labeled similarity cases that single flip is not
itself significant; the significant evidence is in the [semantic channel report](../data/evaluation/personalized/results/semantic_channel_summary.md).

These are offline engineering proxies, not measurements of user satisfaction. The seven-case benchmark is small and manually labeled. Full definitions, p95 latency, model metadata, and caveats are in [results.md](../data/evaluation/results.md) and [results.json](../data/evaluation/results.json).

Reproduce the table:

```powershell
python scripts/evaluate_recommender.py
```

## Actual Stack

- Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- NumPy, Sentence Transformers, PyTorch CPU
- SciPy for offline evaluation only; the web application never imports it
- SQLAlchemy 2.x and SQLite
- Gemini REST API and local Ollama/Gemma 3 adapters
- Plain HTML, CSS, and JavaScript
- Pytest, Ruff, MyPy, GitHub Actions
- Docker and Docker Compose

No React, Redis, Kafka, Kubernetes, authentication layer, GPU training, or open-web search is used.

## Local Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-api.txt
python -m pip install -e . --no-deps
Copy-Item .env.example .env
python run_app.py
```

Open `http://127.0.0.1:8000`. Set `LLM_PROVIDER=ollama` for local Gemma 3 or `LLM_PROVIDER=gemini` plus a backend-only `GEMINI_API_KEY` for a public deployment. Search and recommendation endpoints remain available when both LLM providers are offline.

### Runtime artifacts

The compact serving catalog and production ALS artifact used by Streamlit are committed under `data/processed`
and checksum-verified at startup. The much larger full catalog and optional research artifacts remain excluded
from Git; download them from a Hugging Face Dataset repository when running the complete FastAPI application:

```powershell
python scripts/download_artifacts.py --repo-id Seiitsu/anime-compass-data
```

Alternatively, place the CC0 Kaggle CSVs in the project and rebuild locally:

```powershell
python scripts/prepare_data.py
python scripts/build_collaborative_model.py
# Optional:
python scripts/build_semantic_embeddings.py --force
```

## API

Interactive docs are available at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | Parse intent, execute catalog tools, and return a grounded answer |
| `POST` | `/api/recommend` | Run a typed hybrid recommendation request |
| `POST` | `/api/search` | Paginated lexical/semantic search with filters and sorting |
| `POST` | `/api/rank` | Deterministically filter and sort by an explicit catalog field |
| `POST` | `/api/entities/search` | Resolve catalog entities |
| `GET` | `/api/anime/{anime_id}` | Return anime details and relationships |
| `GET/POST/DELETE` | `/api/session/{session_id}` | Manage anonymous session preferences |
| `GET` | `/api/model-info` | Inspect active channels, weights, and artifacts |
| `GET` | `/api/health` | Inspect API, catalog, database, provider, and embedding health |

## Tests And CI

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m compileall -q app backend scripts tests run_app.py
python -m ruff check app backend scripts tests run_app.py
python -m ruff format --check app backend scripts tests run_app.py
python -m mypy app backend scripts
python -m pytest -q
node --check frontend/app.js
```

Tests mock Gemini and Ollama, so CI requires no provider key and makes no paid API calls.

## Docker And Hugging Face Spaces

```powershell
docker build -t anime-compass .
docker run --rm -p 8000:7860 -e HF_DATASET_REPO=Seiitsu/anime-compass-data anime-compass
```

Or use `docker compose up --build`. The named volume stores only runtime SQLite data.

For a Hugging Face Docker Space, set `HF_DATASET_REPO=Seiitsu/anime-compass-data` as a Space variable. Missing artifacts are downloaded at startup and verified against `data/artifacts.manifest.json`. Add `LLM_PROVIDER=gemini` as a variable and `GEMINI_API_KEY` as a secret. Ollama remains the zero-cost local option but is not expected to run inside a small public Space.

The complete account and repository checklist is in [PUBLISHING.md](PUBLISHING.md).

## Data And Security

The primary catalog and interaction data derive from [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020), released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Legacy CC0 data is retained only as relationship/poster enrichment and a newer-title extension. See [DATASET_ATTRIBUTION.md](../DATASET_ATTRIBUTION.md).

Provider credentials are loaded only from ignored backend environment files. Request bodies, collection sizes, timeouts, CORS origins, rate limits, and output lengths are bounded. Error responses and frontend assets do not expose provider keys.

## Limitations

- Offline metrics cannot establish real user satisfaction; this portfolio project has no live traffic or A/B test.
- The interaction snapshot has no timestamps and ends in 2020, so the benchmark measures preference-set
  reconstruction rather than next-item prediction or current taste.
- ALS has strong full-catalog relevance but zero measured mid-tail and long-tail exposure. Item-item and
  CountSketch remain available for discovery-oriented experiments; the public demo optimizes top-10 relevance.
- The pretrained semantic channel is retired at weight 0.00 after a measured regression. The optional artifact and
  opt-in weight remain only to keep that negative result reproducible.
- The CountSketch projection measurably costs accuracy against exact item-item cosine (+7.5% relative NDCG@10 for
  the exact model). Its remaining justification is build-time memory, not fidelity.
- Learned fusion weights did not beat the hand-set constants on untouched users. The fit is limited by its
  candidate shortlist, and the novelty signal had zero variance in that run.
- Session personalization is anonymous, local, and not synchronized across devices.
- The archive stops at 2022 and has missing studio/staff data; retained enrichment extends coverage but is not a complete relationship graph.
- The 18,064-item Python scorer prioritizes transparency over low-latency vector-database retrieval.
- Public Gemini usage depends on the provider's current free-tier quota; deterministic features require no paid service.

## Resume Bullets

- Built a leakage-safe full-catalog recommendation benchmark with paired bootstrap intervals, threshold
  sensitivity, coverage, novelty, popularity-bias, diversity, and latency diagnostics; used it to replace the
  previous default with ALS at +42.6% NDCG@10 and roughly 465× lower p50 latency on the same 800 users.
- Built a local-first anime recommendation Agent with FastAPI, Pydantic-constrained LLM intent parsing, typed tool routing, catalog-grounded response validation, and bounded deterministic replanning that relaxes over-constrained filters without ever relaxing verified entity constraints, across Gemini and Ollama/Gemma 3 providers.
- Implemented an explainable hybrid recommender over 18,064 titles and 56.7M anonymous ratings using scalable user-centred collaborative projections, metadata/synopsis TF-IDF, LSA, creator signals, hard entity joins, diversity reranking, and session feedback.
- Built a reproducible data-quality and ablation pipeline plus 395 automated API, Agent, ranking,
  artifact-integrity, security, and transcript regression tests, with graceful deterministic fallback when LLM
  providers are unavailable.
- Found and fixed a silent model defect: a documented 0.14-weight embedding channel had never been built,
  manifested, or wired into any evaluation path, so every published metric described a ten-channel model measured
  as nine; building it showed the channel costs 8.9% relative NDCG@10, and the weight was retired on that evidence.
- Implemented exact blockwise adjusted-cosine item similarity and conjugate-gradient implicit ALS in NumPy/SciPy
  (no compiled dependency) to quantify what a CountSketch projection costs and to supply the standard latent-factor
  reference point, then challenged the hand-set hybrid weights with a non-negative pairwise RankNet-style fit;
  reported the negative result and retained the constants when the learned blend failed to generalise.

The source code is available under the [MIT License](../LICENSE). Dataset-derived artifacts retain their [CC0 attribution](../DATASET_ATTRIBUTION.md).
