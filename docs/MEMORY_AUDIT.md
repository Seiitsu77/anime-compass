# Memory Profiling Audit

Sections 1-17 are the audit as measured, before any change: they describe the process at
1.98 GB and no production code had been modified when they were taken. Section 18 records
the two optimisations that followed (A1 and B1) and their measured effect. B2 and B3
remain unimplemented, so sections 1-17 still describe them accurately.

All figures are process RSS (Windows working set via PSAPI), measured with
`backend/anime_agent/process_memory.py`. `tracemalloc` and `sys.getsizeof` are not used
for totals: they miss NumPy, LightGBM and torch buffers, and understate nested Python
structures.

Platform: Python 3.10.0, win32, 24 CPUs. MiB = 2^20, MB/GB = 10^6/10^9.

---

## 1. The real startup sequence

Taken from `lifespan` in `app/main.py`, not from documentation. Every step is eager;
nothing is lazy.

```
ensure_artifacts (only when HF_DATASET_REPO is set)
load_or_create_catalog          -> full catalog, 18,064 records
_load_semantic_index            -> only if embedding_provider == "sentence_transformers"
_load_collaborative_index       -> CountSketch
_load_als_index                 -> quality_source = the collaborative index
_load_reranker                  -> feature space + LambdaMART
AnimeRecommender                -> derived hybrid indexes
SQLiteSessionRepository
_build_providers                -> LLM clients
AgentOrchestrator               -> constructs its own EntityResolver
AppContainer(entity_resolver=EntityResolver(loaded_catalog))
```

`EMBEDDING_PROVIDER=sentence_transformers` is set in both `.env` and the Dockerfile, so
the semantic branch is live in production, not just locally.

## 2. The profiler

`scripts/profile_memory.py`, subcommands `stages`, `imports`, `artifacts`, `catalog`,
`arrays`, `requests`, `ab`, `all`. It is opt-in (a script, never imported by the app),
exposes nothing through the public API, and `build_staged_app()` mirrors the lifespan
order above so the staged numbers describe the real process. Component experiments run
in fresh subprocesses via `run_child()`, because allocators retain freed pages and
in-process teardown would understate every delta.

## 3. Staged startup, fresh process, 3 repetitions

Final RSS: **1886.0 / 1885.9 / 1886.1 MiB** (range 0.2 MiB). Startup peak 1908.0 MiB.

| Stage | RSS MiB | d stage | d base | ms |
|---|---:|---:|---:|---:|
| Python process baseline | 18.0 | 0.0 | 0.0 | 0 |
| Core imports (numpy, pydantic) | 30.2 | 12.1 | 12.1 | 70 |
| FastAPI + SQLAlchemy imports | 56.7 | 26.5 | 38.6 | 358 |
| `app.main` import | 75.2 | 18.5 | 57.1 | 273 |
| Settings | 75.2 | 0.0 | 57.1 | 21 |
| Full catalog loaded | 566.9 | **491.7** | 548.9 | 2092 |
| Semantic index (+ provider) | 954.2 | **387.3** | 936.1 | 5897 |
| Collaborative (CountSketch) | 981.8 | 27.6 | 963.7 | 419 |
| ALS production artifact | 992.2 | 10.5 | 974.2 | 390 |
| Reranker (features + LambdaMART) | 1354.5 | **362.2** | 1336.4 | 1082 |
| AnimeRecommender (hybrid indexes) | 1690.8 | **336.3** | 1672.7 | 7136 |
| Session repository (SQLite) | 1691.5 | 0.7 | 1673.5 | 479 |
| LLM providers | 1693.3 | 1.8 | 1675.3 | 462 |
| Agent orchestrator | 1787.0 | 93.6 | 1768.9 | 6001 |
| EntityResolver | 1886.0 | 99.0 | 1867.9 | 6572 |

Five stages account for 1839 MiB of 1886 MiB (97.5%).

## 4. Import-only cost - is torch resident in production?

**Yes.** Each measured in a fresh process, import only, no model loaded:

| Module | MiB | ms |
|---|---:|---:|
| sentence_transformers | 365.2 | 5036 |
| torch | 164.8 | 1265 |
| lightgbm | 119.7 | 1005 |
| fastapi | 20.2 | - |
| sqlalchemy | 16.6 | - |
| httpx | 12.0 | - |
| scipy | 10.4 | - |
| numpy | 9.9 | - |
| pydantic | 2.9 | - |

torch is pulled in transitively by sentence-transformers, which is active because
`EMBEDDING_PROVIDER=sentence_transformers`. It is **not** a stale import: `encode_query`
is called on the live request path at `backend/anime_agent/recommender.py:561` and
`:2291`, so the model is genuinely needed to embed free-text queries. Roughly 20% of the
process is a deep-learning runtime serving a 6-layer MiniLM.

## 5. Artifact disk size vs memory cost

| Artifact | Disk MB | RSS MiB | Ratio |
|---|---:|---:|---:|
| catalog (full) | 113.8 | **492.6** | 4.3x |
| catalog (serving) | 6.3 | 22.0 | 3.5x |
| semantic embeddings | 24.6 | 27.2 | 1.1x |
| collaborative | 14.3 | 27.8 | 1.9x |
| reranker features | 16.6 | 28.4 | 1.7x |
| ALS factors | 7.1 | 9.5 | 1.3x |
| lambdamart.txt | 0.5 | **114.3** | 229x |

The npz artifacts land near 1:1 - they stay as arrays. The two outliers are the JSON
catalog (parsed into Python objects) and `lambdamart.txt`, whose 114 MiB is almost
entirely the LightGBM import, not the 157-tree model.

## 6. Catalog decomposition

18,064 records, 43 distinct fields, 113.8 MB of JSON on disk becoming 491.7 MiB resident.

- Raw JSON text: 238 MiB transient. `load_or_create_catalog` uses `json.load(file)`,
  which frees the string before returning; startup peak exceeds final RSS by only
  22 MiB, so the allocator reuses those pages. **Not** a steady-state or peak item.
- `id -> record` map: **+1.1 MiB** for 18,064 entries. This proves the map stores
  references, not copies - there is no catalog duplication here.
- AnimeRecommender derived indexes: +358.2 MiB.
- EntityResolver indexes: +103.4 MiB.

## 7. NumPy / array audit

Every ndarray above 512 KiB reachable from a fully built application:

| Name | Shape | dtype | MB | owns data | contiguous |
|---|---|---|---:|---|---|
| collaborative.vectors | (18064, 384) | float32 | 27.75 | yes | C |
| semantic_index.matrix | (18064, 384) | float32 | 27.75 | yes | C |
| recommender.embedding_vectors | (18064, 192) | float32 | 13.87 | yes | C |
| als_index.item_factors | (18064, 128) | float32 | 9.25 | yes | C |
| recommender.svd_vectors | (18064, 48) | float32 | 3.47 | yes | C |

**Total 82.1 MB (78.3 MiB) - 4.6% of a 1772 MiB process.** No shared-buffer aliases, no
views, nothing non-contiguous.

The two 384-wide matrices are a coincidence of width, not a duplicate load: the
CountSketch projection and the MiniLM embeddings differ by 0.0528 mean absolute
difference, and `np.array_equal` is False. Index objects are correctly shared by
reference - `als.quality_source is collaborative_index`,
`recommender.collaborative_index is collaborative_index` and
`recommender.semantic_index is semantic_index` are all True.

**Arrays are not the problem.** Downcasting or quantising every array in the process
could recover at most 39 MB.

## 8. Reranker feature memory

`RerankerFeatureSpace.__init__` (`backend/anime_agent/evaluation/reranking.py`) expands
the neighbour arrays into `self._neighbors: list[dict[int, float]]`:

- 18,064 dicts holding 3,107,649 entries
- **312.9 MiB** of Python dict overhead
- versus **28.0 MiB** for the identical numeric payload as npz arrays
- **11.2x expansion of information already present in array form**

This is the single largest piece of pure representational overhead in the process.

## 9. Hybrid and semantic memory

From the A/B experiments in section 13: semantic costs 290.0 MiB, the hybrid path
337.2 MiB, the collaborative channel 30.0 MiB. The semantic figure is dominated by the
library runtime (365.2 MiB of import) rather than by data (27.2 MiB of embeddings).

## 10. Agent memory

The agent orchestrator stage adds 93.6 MiB, essentially all of it a second
EntityResolver (section 12); the orchestrator's own state is negligible. LLM providers
cost 1.8 MiB - thin HTTP clients.

**Ollama runs as a separate OS process.** It contributes nothing to this process's RSS.
Any Ollama model weights are outside every number in this report; a host sized from this
audit still needs separate headroom if Ollama is co-located.

## 11. Steady-state request memory and leak detection

| Workload | Delta |
|---|---|
| Steady state after warm-up | 1,982,943,232 B (1891.1 MiB) |
| small (3 liked) x20 | +585,728 B |
| medium (10 liked) x20 | 0 |
| heavy (199 liked) x10 | +8,192 B |
| constrained hybrid x10 | +56,811,520 B (54.2 MiB) |
| identical request x60 | **0** |
| Peak | 2,057,592,832 B (1962.3 MiB) |

**No leak.** The fast path is flat to within a page across 50 requests. The hybrid's
54.2 MiB is a one-time working set, and it is bounded: 20 *distinct* hybrid requests
added 15.7 MiB with decelerating growth (1.0, 0.5, 0.4 MB per 5 requests), and the
response cache is a fixed 128-entry `OrderedDict` LRU
(`app/services/recommendation_service.py:41-43`).

Steady state 1.983 GB matches the ~2 GB in the original question exactly.

## 12. Duplicate-load audit (proven by object identity)

**One real duplicate.** `EntityResolver` is constructed twice over the same catalog:
once inside `AnimeAgent.__init__` (`backend/anime_agent/agent.py`) and once by
`AppContainer` in `app/main.py`.

- distinct objects: `id` 2027364018080 vs 2028234939776
- 97.9 MiB + 99.0 MiB = **196.9 MiB for logically identical indexes**
- ~98 MiB is pure waste

Everything else checked clean: shared index objects (section 7), reference-only record
map (section 6), no aliased or duplicated arrays.

## 13. A/B component experiments

*MEMORY ATTRIBUTION EXPERIMENT ONLY.* Each configuration is a fresh subprocess built to
the AnimeRecommender stage. These measure what a component costs; they are not proposals
to remove it.

| Configuration | RSS MiB | Delta |
|---|---:|---:|
| full (baseline) | 1691.3 | - |
| no semantic | 1401.3 | **-290.0** |
| no reranker | 1322.4 | **-368.9** |
| no hybrid | 1354.1 | **-337.2** |
| no collaborative | 1661.3 | -30.0 |

## 14. Python-object expansion - where the memory actually goes

| Bucket | MiB | % |
|---|---:|---:|
| Catalog as Python objects | 491.7 | 26.1 |
| Semantic (torch + ST + MiniLM + matrix) | 387.3 | 20.5 |
| Reranker (LightGBM import + neighbour dicts) | 362.2 | 19.2 |
| AnimeRecommender derived indexes | 336.3 | 17.8 |
| EntityResolver (container copy) | 99.0 | 5.2 |
| Agent orchestrator (second EntityResolver) | 93.6 | 5.0 |
| Interpreter + framework imports | 75.2 | 4.0 |
| Collaborative CountSketch | 27.6 | 1.5 |
| ALS factors | 10.5 | 0.6 |
| SQLite + LLM providers | 2.5 | 0.1 |

Named, provable overhead:

| Overhead | MiB |
|---|---:|
| Catalog object expansion (492 MiB holding ~114 MB of JSON) | ~378 |
| Neighbour dicts (313 MiB holding 28 MiB of arrays) | ~285 |
| Duplicate EntityResolver | ~98 |
| **Total** | **~761 (40% of RSS)** |

The numeric payload the models actually compute on is 78.3 MiB - 4.2% of the process.
The dominant cost is the representation, not the data or the models.

## 15. Prioritised opportunities

### Class A - zero behaviour change, provable by object identity

**A1. Share one EntityResolver between AppContainer and AnimeAgent. -98 MiB.**
Both are built from the same catalog and produce identical indexes; only the
construction site differs. Verifiable by asserting the two attributes are the same
object and that recommendation and agent outputs are byte-identical before and after.

### Class B - real savings, require regression validation

**B1. Keep reranker neighbours as arrays instead of dicts. Up to -285 MiB.**
The npz already holds the values; `_neighbors` re-encodes them as 18,064 dicts.
Replacing lookup with a sorted-array search preserves the values but changes the lookup
path, so it needs a numerical-identity regression over the frozen confirmation users:
identical feature vectors, identical LambdaMART scores, identical ordering.

**B2. Load only the catalog fields the backend reads. Up to -380 MiB.**
43 fields are loaded; the serving catalog proved a compact subset costs 22.0 MiB rather
than 492.6 MiB. The backend needs more fields than Streamlit does (characters, voice
actors, synopsis), so this requires a field-usage audit and a full recommendation
regression. It is not the same change as the existing serving catalog.

**B3. Replace the torch runtime for query encoding. Up to -165 MiB.**
Only `encode_query` is needed at request time; document embeddings are precomputed. An
ONNX or tokenizer-only runtime could serve that without torch. This changes the
inference stack, so it needs embedding-equivalence testing - and it must reduce the
runtime cost of semantic search, never the capability.

### Class C - out of scope per the audit constraints

Quantising or downcasting model arrays (worth at most 39 MB anyway); ANN indexes;
replacing SQLite; disabling semantic or collaborative functionality to save RAM;
changing ALS, LambdaMART, ranking weights, thresholds or candidate generation.

## 16. Hosting thresholds

Peak is the binding constraint, not steady state: **2,057,592,832 B peak,
1,982,943,232 B steady.**

The unit matters here. Container platforms that advertise "2 GB" almost always mean
2 GiB (Docker `--memory=2g`, Render, Fly), so the peak is **1.916 GiB = 95.8% of a
2 GiB limit** - under it, but with 4.2% headroom. Against a decimal 2 x 10^9 limit it
is over.

| Limit | Verdict |
|---|---|
| 256 / 512 MB (Fly, Render free) | Impossible - under a third of what startup needs |
| 1 GB | Impossible today |
| 2 GiB | **Unsafe, not impossible.** 95.8% of the limit at peak; any load spike, allocator variance or fragmentation OOMs it. Not a tier to deploy on without headroom |
| 2 x 10^9 B (decimal) | Fails outright |
| 4 GB | Safe - 1.94 GB headroom at peak |

Measured after A1 and B1 landed (see section 18); B2 and B3 remain projections:

| Change | Peak | % of 2 GiB | Headroom |
|---|---|---:|---:|
| baseline | 2,057,592,832 B | 95.8% | 90 MB |
| A1 (measured) | 1,953,538,048 B | 91.0% | 194 MB |
| **A1 + B1 (measured)** | **1,625,829,376 B** | **75.7%** | **522 MB** |
| + B3 (projected) | ~1.46 GB | ~68% | ~690 MB |
| + B2 + B3 (projected) | ~1.08 GB | ~50% | ~1.07 GB |

## 17. Conclusion

**Where does the ~1.98 GB go?** Five stages account for 97.5%: the catalog as Python
objects (492 MiB), the sentence-transformers/torch runtime (387 MiB), the reranker
(362 MiB, of which 313 MiB is dict expansion of array data), the recommender's derived
indexes (336 MiB), and two copies of EntityResolver (197 MiB). Actual numeric model data
is 78.3 MiB - 4.2% of the process. The memory is spent on representation, not on models.

**Can it be reduced substantially without changing recommendation behaviour?** Yes.
About 761 MiB (40%) is named, measured overhead rather than information. One change is
provably free (A1, -98 MiB); the two largest (B1, -285 MiB; B2, -380 MiB) preserve the
same values and orderings but need regression evidence rather than an identity argument.

There is no leak, no array duplication, and no misconfiguration inflating the process.
Every large number traces to a deliberate design choice whose memory cost was simply
never measured.

### Quality gates

`compileall` pass, `ruff check` pass, `ruff format --check` 164 files formatted, `mypy`
clean on 83 files for both win32 and `--platform linux`, `pytest` 482 passed 2 skipped.
`git diff --stat HEAD` is empty over tracked files, so recommendation behaviour is
unchanged by construction: no production code path was touched.

---

## 18. Optimisation results (A1 and B1)

Both changes were required to produce byte-identical output, and both do. B2 and
B3 remain unimplemented.

### What changed

**A1** - `AnimeRecommender` now owns a lazily-built `entity_resolver`; the API
container and `AnimeAgent` both take that instance instead of constructing their own.
Lazy because the fast recommendation path never resolves an entity.

**B1** - `RerankerFeatureSpace` stores item-item neighbours as the artifact stores them:
three flat arrays (`_neighbor_offsets`, `_neighbor_ids`, `_neighbor_scores`) in place of
18,064 Python dicts. Zero-scored slots are dropped at construction, exactly as the dict
comprehension did, so the lookup scan visits the same entries in the same order.

### Evidence of equivalence

| Check | Result |
|---|---|
| Reranker feature matrix, 400 users x 300 candidates (120,000 rows x 18 features) | sha256 `41c95fa1...` before and after; `np.array_equal` and byte comparison both true |
| LambdaMART scores over those rows | sha256 `d3466f7b...` before and after |
| Resulting orderings | sha256 `c2708b52...` before and after |
| End-to-end fingerprint: 18 recommendation requests, 15 entity queries x 2 resolvers, 6 agent messages | sha256 `e3066d68...` pristine and after A1+B1; 0 substantive differences |
| Recommendation ID sequences (218 ids) | identical |
| Diagnostics decision fields (`learned_reranker_applied`, `recommendation_path`, `collaborative_route`, `hard_filter_applied`, `diversity_applied`, pool sizes, `tail_source_used`) | identical across all 18 requests |
| LambdaMART activation | fires on all 6 personalised requests, `recommendation_path=fast`, unchanged |
| Test suite | 491 passed, 2 skipped (9 new regression tests) |
| ruff / ruff format / mypy (win32 and `--platform linux`) | clean |

Only wall-clock `*_ms` fields differ between runs, as they must.

B1's equivalence relies on one precondition: no row of the item-item artifact repeats a
neighbour id. A dict would collapse a repeat to its last score; the packed layout keeps
both. The shipped artifact has zero such rows across all 18,064, and
`test_the_production_artifact_has_distinct_neighbours_per_row` now pins that.

### Memory

Staged startup, fresh processes, 3 repetitions each:

| | Baseline | After A1 | After A1 + B1 |
|---|---:|---:|---:|
| Reranker stage | 362.2 MiB | 362.5 MiB | **49.5 MiB** |
| EntityResolver stage | 99.0 MiB | **0.0 MiB** | **0.0 MiB** |
| Final RSS | 1886.0 MiB | 1786.6 MiB | **1473.6 MiB** |
| Startup peak | 1908.0 MiB | 1809.6 MiB | **1495.6 MiB** |

Live process under the same request workload:

| | Baseline | After A1 + B1 | Change |
|---|---:|---:|---:|
| Steady state | 1,982,943,232 B | 1,550,184,448 B | **-432.8 MB (-21.8%)** |
| Peak | 2,057,592,832 B | 1,625,829,376 B | **-431.8 MB (-21.0%)** |

Reranker load time also fell from 1082 ms to ~590 ms, since building 3.1M dict entries
was itself most of the cost.

Leak behaviour is unchanged: 60 identical requests grow 0 bytes, the hybrid's one-time
working set is 57.5 MB, and the 128-entry LRU still bounds it.

### Answer to the deployment question

**A 2 GiB deployment is now safely viable.** Peak occupies 75.7% of the limit with
522 MB of headroom, against 95.8% and 90 MB before. The remaining reduction to reach
1 GiB would need B2 and B3, which change the catalog load and the embedding runtime and
so require the regression work described in section 15.
