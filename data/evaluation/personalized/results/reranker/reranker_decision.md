# Two-Stage Reranking After ALS: Decision

## Question

ALS already retrieves well — Recall@100 0.6226, Recall@300 0.7932, Recall@500
0.8500. So this phase does not ask whether to replace retrieval. It asks one
narrower question: **given that the right items are usually already inside the
ALS candidate set, does richer evidence order them better than the ALS score
alone?**

## Outcome

**Yes, and materially.** A LightGBM LambdaMART reranker over the frozen ALS
top-300 improves NDCG@10 by **+0.0372 (+15.1% relative)** on 800 untouched
users, paired 95% CI `[+0.0276, +0.0465]`, while leaving catalog coverage
essentially unchanged.

The recommendation is to promote it, with one caveat stated plainly below: it
adds a LightGBM runtime dependency to a serving path that is currently
NumPy-only.

## Confirmation results

800 users, none of them ever scored by any earlier experiment. Same frozen
train-only ALS artifact generates the candidates for every arm; the arms differ
only in how they order those candidates.

| Arm | NDCG@10 | Δ vs ALS | 95% CI | Recall@10 | NDCG@20 | Recall@20 | HR@10 |
|---|---:|---:|---|---:|---:|---:|---:|
| **ALS standalone** | 0.2456 | — | — | 0.2312 | 0.2728 | 0.3293 | 0.7987 |
| Linear (no item-item) | 0.2572 | +0.0116 | `[+0.0037, +0.0196]` | 0.2362 | 0.2864 | 0.3435 | 0.8000 |
| Linear reranker | 0.2725 | +0.0269 | `[+0.0174, +0.0365]` | 0.2441 | 0.3011 | 0.3519 | 0.8025 |
| **LambdaMART** | **0.2828** | **+0.0372** | `[+0.0276, +0.0465]` | **0.2537** | **0.3099** | **0.3592** | 0.8187 |

Every NDCG and Recall delta above is significant at 95%. HR@10 is the one
metric whose interval includes zero for every arm (`[-0.0013, +0.0413]` for
LambdaMART), which is expected: hit rate is already 0.80 and saturating.

Selection happened on the 500 validation users, where the ordering was the same
(ALS 0.2377, linear 0.2613, LambdaMART 0.2754). The confirmation set was opened
once, after the arm was chosen.

## Where the gain comes from

The ablation is the interesting part. Dropping the two item-item features from
the linear model costs more than half its gain: **+0.0269 → +0.0116**.

That is worth stating carefully, because item-item was *rejected* earlier in
this project. As a **retrieval source** it restored tail reach but cost 9%
NDCG@10 and 4× latency, so it ships disabled. As a **reranking feature** over
candidates ALS already chose, the same signal is clearly useful. Those are not
contradictory findings: item-item is a poor candidate generator and a good
tie-breaker.

Linear model weights, largest first: `profile_size +0.67`, `als_score_z +0.47`,
`item_item_max +0.41`, `als_score +0.34`, `log_train_ratings +0.24`. Note that
`profile_size` is constant within a user's candidate list, so despite its
magnitude it cannot affect the linear model's ordering at all; it acts as a
per-user intercept. LambdaMART can use it as a gate, which is one plausible
reason it does better.

## Popularity and tail exposure

Measured over what each arm actually places in the top 10 across the 800
confirmation users.

| Arm | Distinct items | Coverage | Head share | Mid share | Tail share | Mean log popularity |
|---|---:|---:|---:|---:|---:|---:|
| ALS standalone | 1,023 | 0.0566 | 1.000 | 0.000 | 0.000 | 10.216 |
| Linear reranker | 870 | 0.0482 | 1.000 | 0.000 | 0.000 | 10.261 |
| Linear (no item-item) | 890 | 0.0493 | 1.000 | 0.000 | 0.000 | 10.305 |
| **LambdaMART** | **1,021** | **0.0565** | 0.999 | 0.001 | 0.000 | 10.249 |

This separates the two challengers more sharply than the ranking metrics do.
The **linear reranker buys its gain partly by concentrating further**: 15% fewer
distinct items reach a top-10 slot. **LambdaMART does not** — it holds coverage
at 1,021 distinct items against ALS's 1,023, and is the only arm that puts
anything at all outside the head.

Neither arm fixes the underlying problem. ALS puts ~100% of its exposure in the
most popular 20% of the catalog, and reranking within its candidate set cannot
undo that, because the candidates are already head-concentrated. This is a
ranking win, not a discovery win, and it should not be described as one.

## Latency and artifact size

| Arm | Rerank p50 | Rerank p95 | Artifact | Serving runtime |
|---|---:|---:|---:|---|
| Linear reranker | 0.028 ms | 0.031 ms | 1.5 KB JSON | NumPy only |
| LambdaMART | 0.278 ms | 0.658 ms | 554 KB text | LightGBM |

Against a ~2 ms fast path, LambdaMART adds roughly 14% at p50. That is
comfortably acceptable on its own. The real cost is the dependency: the serving
path is currently NumPy-only, with no SciPy, no training code, and no ML
framework in the web process, and promoting LambdaMART ends that.

The linear arm is the NumPy-only alternative at ~72% of the NDCG gain, but it
is also the arm that worsens concentration, so it is not simply a cheaper
version of the same thing.

## Candidate-depth sensitivity

Rerunning the whole experiment at top-500 instead of top-300:

| Arm | NDCG@10 Δ at 300 | NDCG@10 Δ at 500 |
|---|---:|---:|
| Linear | +0.0269 | +0.0258 |
| LambdaMART | +0.0372 | +0.0362 |

The result is not an artifact of candidate depth. Top-300 remains the better
operating point: at 500 LambdaMART's coverage falls to 0.0520 from 0.0565.

## Leakage controls

- **Candidates** come from `als_train_only.npz` (128 factors, alpha 5.0), the
  frozen train-only artifact. ALS was not retuned, and candidate generation and
  reranking were not changed together.
- **Profile features** use each user's `train_positive` interactions only.
  Validation and test positives are never passed to the feature builder; the
  function signature takes profile rows and cannot see anything else.
- **Item statistics** — popularity, rating count, rating mean, Bayesian
  quality — come from the split's train-only artifacts, not the catalog's
  all-time aggregates, which are computed over every user including held-out
  ones. `tests/test_reranking.py` asserts that adding `score`, `members`, and
  `favorites` to a catalog row changes no feature.
- **Descriptive metadata** (genres, studios, type, source, year) describes the
  item, not its audience, and carries no signal about who liked what.
- **Labels** are test positives, used only to fit and to score. No feature sees
  a label.

## User populations

3,924 distinct users had been scored by some earlier experiment. All three
populations here are drawn from the remaining 279,585 eligible users and are
mutually disjoint; the script asserts all four disjointness properties before
doing any work.

| Population | Users | Purpose |
|---|---:|---|
| Reranker training | 2,000 | fits the model |
| Validation | 500 | selects the arm, early-stops LambdaMART |
| Confirmation | 800 | reported once, never tuned against |

Training produced 600,000 candidate rows at a 2.614% positive rate.

The exact user IDs and their SHA-256 digests are recorded in
`reranker_results.json` and `confirm_user_ids.json.gz`.

## A caveat on the ALS baseline number

The ALS baseline here is 0.2456, against 0.2588 in the published architecture
comparison. Those are different user samples, and this experiment also excludes
only train positives from the candidate pool rather than every observed rating.
Both arms in every comparison above use the identical protocol, so the deltas
are sound; the absolute 0.2456 should not be quoted against the published
figure.

## Promotion criteria

| Criterion | Result |
|---|---|
| Positive paired NDCG@10 delta | **Yes** — +0.0372, CI excludes zero |
| No material Recall regression | **Yes** — Recall@10 +0.0226, Recall@20 +0.0299, both significant |
| Acceptable latency increase | **Yes** — +0.278 ms p50 on a ~2 ms path |
| No severe popularity/tail regression | **Yes** — coverage 0.0565 vs 0.0566 |
| Reproducible serving artifact | **Yes** — 554 KB, deterministic seed |
| No evaluation leakage | **Yes** — see leakage controls |
| Smallest practical runtime | **No** — adds LightGBM to a NumPy-only path |

Six of seven met. The seventh is a genuine trade-off rather than a failure, and
it is the decision to make before integrating.

## Reproduce

```bash
python scripts/evaluate_reranker.py --train-users 2000 --val-users 500 --confirm-users 800
python scripts/evaluate_reranker.py --candidates 500 --output data/evaluation/personalized/results/reranker_top500
```

Training needs `pip install -r requirements-evaluation.txt`. Nothing in the
serving path imports LightGBM today.
