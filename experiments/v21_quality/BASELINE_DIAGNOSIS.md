# V2.1 baseline quality diagnosis

Where the current recommender loses quality, measured on the frozen offline
protocol. Nothing here trains, tunes, or selects a model, and no production
artifact was touched.

Offline metrics are engineering proxies. NDCG on held-out ratings is not user
satisfaction, and nothing below should be read as a claim about it.

Three sections here were followed up and are refined elsewhere:

| Section | Refined in | What changed |
|---|---|---|
| 5, small profiles | [`SMALL_PROFILE_EVAL.md`](SMALL_PROFILE_EVAL.md) | A dedicated evaluation with a second, genuinely light cohort. The 0.27 to 0.44 recall below is the floor of a range, not the typical case |
| 6, hard exclusions | [`SERIES_POLICY.md`](SERIES_POLICY.md) | The +0.045 is +0.035 once low-rated history is hidden at the same rate as positives |
| 7, `one_per_series` | [`SERIES_POLICY.md`](SERIES_POLICY.md) | The flag is two rules. De-duplication costs 0.009; removing Liked franchises accounts for the rest |

## Protocol

| | |
|---|---|
| Split | `holdout_seed42_pos8.sqlite`, positives are ratings of 8 or higher |
| Retrieval | frozen train-only ALS (`als_train_only.npz`, evaluation role), top 300 |
| Ranking | the confirmed LambdaMART (`results/reranker/lambdamart.txt`) over those 300 |
| Users | the 800 confirmation users of the reranker experiment |
| Known items | train positives only, as in the reranker experiment |
| Intervals | 95% percentile bootstrap over users, 2,000 resamples, fixed seed; deltas are paired |

The 800 users were already opened once for the reranker decision, so reusing
them spends no untouched user. They must not be used as a confirmation set
again. The run reproduces the published numbers exactly before reporting
anything new:

| | Published | This run |
|---|---:|---:|
| ALS NDCG@10 | 0.2456 | 0.2456 |
| LambdaMART NDCG@10 | 0.2828 | 0.2828 |
| ALS Recall@10 | 0.2312 | 0.2312 |
| LambdaMART Recall@10 | 0.2537 | 0.2537 |

Reproduce with `python experiments/v21_quality/diagnose_baseline.py` (about 45 s).
Every number in this document is in `results/*.json`.

## 1. Headline metrics

| | ALS | ALS + LambdaMART | Paired gain |
|---|---:|---:|---|
| NDCG@10 | 0.2456 | 0.2828 `[0.268, 0.297]` | +0.0372 `[+0.028, +0.046]` |
| NDCG@18 | 0.2673 | 0.3061 `[0.292, 0.320]` | +0.0388 `[+0.031, +0.047]` |
| Recall@10 | 0.2312 | 0.2537 `[0.237, 0.270]` | +0.0226 `[+0.013, +0.031]` |
| Recall@18 | 0.3133 | 0.3453 `[0.327, 0.363]` | +0.0320 `[+0.020, +0.042]` |

18 is one page of the For You feed.

Retrieval recall of held-out positives by candidate depth:

| Depth | 10 | 18 | 50 | 100 | 200 | **300** | 500 | 1000 | 2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Recall | 0.231 | 0.313 | 0.480 | 0.607 | 0.734 | **0.791** `[0.776, 0.805]` | 0.858 | 0.910 | 0.944 |

31% of users have every held-out positive inside the top 300; 1.6% have none.

## 2. Retrieval ceiling against ranking quality

For each user, the best NDCG any reordering of the candidate set could reach,
against what the reranker actually reaches.

| | @10 | @18 |
|---|---:|---:|
| Perfect ranking of the whole catalog | 1.000 | 1.000 |
| Perfect ranking of ALS top 300 | 0.901 | 0.874 |
| Perfect ranking of ALS top 500 | 0.932 | 0.915 |
| Perfect ranking of ALS top 1000 | 0.953 | 0.944 |
| **ALS + LambdaMART, actual** | **0.283** | **0.306** |
| Lost at retrieval (depth 300) | 0.099 | 0.126 |
| Lost at ranking (within the 300) | 0.618 | 0.567 |

For full-history profiles the bottleneck is ranking, not retrieval. The
reranker captures 31% of what its candidate set allows. Deepening retrieval
from 300 to 1000 raises the ceiling by 0.05, of which a reranker working at
that efficiency would realise roughly a third.

Conditional on a held-out positive being retrieved (6,575 of them), where it
lands inside the 300:

| | In top 10 | In top 18 | In top 50 | Median rank |
|---|---:|---:|---:|---:|
| ALS | 20.0% | 29.1% | 51.8% | 47 |
| ALS + LambdaMART | 22.8% | 33.3% | 57.4% | 39 |

The perfect-ranking ceiling is deliberately loose. It is not attainable, for
reasons section 3 makes concrete.

## 3. What the top of the list is actually made of

Average composition of the LambdaMART top 10:

| | Share |
|---|---:|
| Held-out test like (counted as a hit) | 18.7% |
| Held-out **validation** like (liked, but counted as a miss) | 19.1% |
| Already rated 6 or 7 | 17.9% |
| Already rated 5 or lower | 1.0% |
| Never rated by this user | 43.2% |

Two consequences:

- **The metric understates precision by construction.** As many top-10 slots go
  to titles the user demonstrably liked but that sit in the validation holdout
  as go to scored hits. 38% of the top 10 are known likes; the metric credits
  half of them.
- **19% of the top 10 are titles the user has already watched and did not
  love.** The benchmark's published protocol excludes those; a live session
  cannot, because the browser only knows Likes. See section 6.

## 4. Coverage and popularity bias

| | Distinct items | Catalog coverage | Head share | Mid | Tail | Top-20 titles' share of slots |
|---|---:|---:|---:|---:|---:|---:|
| ALS top 10 | 1,023 | 5.66% | 100.0% | 0.0% | 0.0% | 12.1% |
| LambdaMART top 10 | 1,021 | 5.65% | 99.94% | 0.06% | 0.0% | 19.0% |
| ALS top 18 | 1,293 | 7.16% | 100.0% | 0.0% | 0.0% | 9.6% |
| LambdaMART top 18 | 1,387 | 7.68% | 99.92% | 0.08% | 0.0% | 15.2% |

Head is the most popular 20% of the catalog by train positives, mid the next
30%, tail the bottom 50%.

- Exposure is entirely head. Recommendations are also more popular than the
  profile that produced them (normalised log-popularity +0.048 for LambdaMART).
- LambdaMART keeps the number of distinct titles but concentrates harder on the
  very top: twenty titles take 19% of all top-10 slots, against 12% for ALS.
  The distinct-item count the promotion decision relied on does not show this.
- **The benchmark cannot measure long-tail quality.** Of 8,738 held-out
  positives, 96.96% are head items, 3.03% mid, and exactly one is tail. Recall
  at 300 is 77.5% for head positives and 3.4% for mid. No offline change here
  can be credited or blamed for tail behaviour on this protocol.

## 5. User segments

The uniform 800-user sample contains **no sparse user**: 663 heavy (20 or more
train positives) and 137 medium (5 to 19).

| | Medium (n=137) | Heavy (n=663) |
|---|---:|---:|
| Recall@300 | 0.796 | 0.790 |
| ALS NDCG@10 | 0.263 | 0.242 |
| LambdaMART NDCG@10 | 0.287 `[0.239, 0.340]` | 0.282 `[0.268, 0.297]` |
| LambdaMART Recall@18 | 0.449 | 0.324 |

A live profile is a handful of Likes, which this population does not represent.
`profile_size_curve.py` therefore re-scores the same stack on the same users
with the profile cut to K liked titles (474 users with at least 50 train
positives; relevance unchanged, so K is the only thing that varies):

| K likes | Recall@300 | ALS NDCG@10 | LambdaMART NDCG@10 | Reranker gain |
|---:|---:|---:|---:|---|
| 1 | 0.266 | 0.040 | 0.039 | −0.001 `[−0.008, +0.005]` |
| 2 | 0.303 | 0.046 | 0.045 | −0.001 `[−0.007, +0.006]` |
| 3 | 0.331 | 0.054 | 0.051 | −0.003 `[−0.010, +0.003]` |
| 5 | 0.369 | 0.060 | 0.058 | −0.003 `[−0.009, +0.004]` |
| 10 | 0.440 | 0.072 | 0.074 | +0.002 `[−0.004, +0.009]` |
| 20 | 0.520 | 0.090 | 0.087 | −0.003 `[−0.010, +0.003]` |
| 50 | 0.652 | 0.128 | 0.139 | **+0.012** `[+0.004, +0.019]` |

- **At session-sized profiles the conclusion of section 2 reverses.** With ten
  Likes, fewer than half of the held-out positives are in the candidate set at
  all. Retrieval, not ranking, is the binding constraint.
- **The reranker has no demonstrated held-out gain below 50 Likes.** Every
  interval at K ≤ 20 includes zero. It was fitted on users averaging 86 likes.

A more forgiving reading, with a caveat: the share of the served top 10 that the
user is known to like at all (any 8+ rating outside the K-title profile) is
0.42 for ALS and 0.44 for LambdaMART at K=3, and 0.53 against 0.57 at K=10; that
difference is significant from K=2 upward. The titles left out of a K-title
profile were part of ALS training, so this measure is optimistic and is not a
held-out result.

These are heavy users with a truncated profile, not genuinely sparse users. It
is the best proxy the data allows, and it is a proxy.

## 6. Hard exclusions

The question was whether exclusions needlessly shrink the candidate pool. They
do not; the measured problem is the opposite.

- Exclusions are applied before ALS takes its top 300, so the pool refills from
  the 18,064-title catalog and stays at 300.
- 32.3 of every 300 candidates are titles the user already rated below 8.
  Removing them, as the published protocol does, improves every metric:

| | Only likes known | Everything rated is known | Paired delta |
|---|---:|---:|---|
| Recall@300 | 0.791 | 0.812 | +0.021 `[+0.018, +0.025]` |
| ALS NDCG@10 | 0.246 | 0.289 | +0.044 `[+0.039, +0.049]` |
| LambdaMART NDCG@10 | 0.283 | 0.328 | +0.045 `[+0.040, +0.052]` |
| LambdaMART NDCG@18 | 0.306 | 0.348 | +0.042 `[+0.037, +0.047]` |

Knowing what a user has already seen is worth as much NDCG@10 as the reranker
itself. The published 0.2588 baseline assumes that knowledge; the live product
has it only for titles the user Liked, bookmarked, or marked Not Interested.
The legacy frontend had a "watched" action (`seen_titles`); the React frontend
currently has none.

## 7. `one_per_series`

The feed requests `one_per_series`, which keeps one entry per franchise and
drops every entry from a franchise the user already Liked.

- **Candidate availability is not a problem.** Of the 200 titles fetched for a
  50-title request, 141 survive on average (5th percentile 111, minimum 62). No
  user fell below 50, or below one page.
- **The benchmark and the product disagree about what counts.** 35.2% of
  held-out positives are in a franchise the user already liked. The benchmark
  credits them; the product never shows them.
- **The reranker leans on exactly those titles.** 51% of LambdaMART's top-10
  slots are in an already-liked franchise, against 38% for ALS.

| NDCG@10 | ALS | LambdaMART | Reranker gain |
|---|---:|---:|---|
| As benchmarked (no filter, all positives) | 0.246 | 0.283 | +0.037 `[+0.028, +0.046]` |
| As served (filter on, positives the feed may show; n=779) | 0.184 | 0.216 | **+0.032** `[+0.022, +0.043]` |

| Recall@18, as served | 0.278 | 0.305 | +0.027 `[+0.015, +0.040]` |
|---|---:|---:|---|

The reranker's gain survives the filter, which had not been measured. But the
number that describes the product is **0.216, not 0.283**, and retrieval is
weaker in this condition too: recall at 300 of showable positives is 0.733,
and the perfect-ranking ceiling is 0.828.

Whether hiding every sequel of a liked title is the right product behaviour is
a product question this data cannot answer. It does show the cost of the
current mismatch: the reranker is trained to rank titles the feed then removes.

## 8. Is the positive-only threshold losing information?

Rating-scale statistics over 28,554 users and 5.48M ratings. No model is scored
on these users.

| | |
|---|---:|
| Ratings that are 8 or higher (used as positives) | 54.6% |
| Ratings that are 6 or 7 (discarded: neither positive nor negative) | 34.9% |
| Ratings that are 5 or lower (discarded) | 10.5% |
| Mean share of a user's ratings that are 8+ | 63.4% |
| Users who rate 80% or more of their titles 8+ | 25.0% |
| Users whose mean rating is 8 or higher | 45.1% |
| Users who rate 20% or fewer of their titles 8+ | 2.3% |
| Threshold positives at or below that user's own mean | 16.1% |
| 6-or-7 ratings above that user's own mean | 19.6% |
| Jaccard, "8+" against "above own mean" | 0.744 |

- 45% of rating events carry no training signal at all.
- For a quarter of users "8 or higher" is nearly everything they rated, so the
  threshold does not separate what they loved from what they finished.
- One in six positives is a below-average title for the user who gave it, and
  one in five 6-or-7 ratings is an above-average one.
- It is not an eligibility problem: 98.9% of users have five threshold
  positives, and a relative definition would add 0.2%.

## 9. What MAL list status and progress would add

`rating_complete.csv` is exactly the completed-and-rated subset of
`animelist.csv` (57,633,278 rows in both). The models see 52.8% of the list.

| Status | Rows | Rated | Mean rating | 8+ when rated | 5 or lower when rated |
|---|---:|---:|---:|---:|---:|
| Completed | 68.1M | 84.6% | 7.51 | 54.0% | 11.0% |
| Plan to Watch | 27.9M | 1.0% | 7.87 | 61.8% | 10.8% |
| Currently Watching | 5.2M | 28.4% | 7.89 | 62.5% | 6.4% |
| Dropped | 4.3M | 45.6% | 4.81 | **6.7%** | **63.5%** |
| On Hold | 3.7M | 28.5% | 7.21 | 41.9% | 11.3% |

Signal the models never see: 10.5M completed-but-unrated titles, 8.9M started
and not finished, 4.3M dropped, 27.9M planned.

- **Dropped is a clean negative.** 6.7% liked, 63.5% rated 5 or lower. Even
  titles dropped after 26 or more episodes are liked only 20% of the time.
- **Completed-but-unrated is 10.5M implicit interactions** of unknown polarity,
  with a 54% base like-rate among their rated counterparts.
- **Plan to Watch is the product's Want to Watch.** 90% of users have such a
  list (median 33 titles). It is unwatched, self-declared intent, which makes
  it a discovery target the current benchmark does not have.
- 4.5% of users have no completed-and-rated title but do have other activity.

A leakage warning for anyone using this file: every held-out positive in the
split also appears in `animelist.csv` as a completed row. A status-aware
training matrix must drop held-out pairs by (user, title), or it trains on its
own test set.

## 10. Retrieval-source combination

ALS top 300 plus item-item candidates, against ALS alone at the same total
budget:

| Budget | ALS alone | ALS 300 + item-item | Paired delta | Non-head positives found |
|---|---:|---:|---|---|
| 400 | 0.833 | 0.853 | +0.020 `[+0.011, +0.030]` | 15 → 19 |
| 500 | 0.858 | 0.876 | +0.017 `[+0.007, +0.027]` | 29 → 37 |
| 1000 | 0.910 | 0.937 | +0.027 `[+0.018, +0.037]` | 80 → 112 |

A real but small recall gain for full profiles, where recall is not the
constraint. Whether it is larger for small profiles, where recall is the
constraint, has not been measured.

## 11. Serving latency

Measured through the running API on the development machine: fast path with
LambdaMART, 30 uncached requests per row after warm-up, server-side time.

| Likes in profile | p50 | p95 | LambdaMART stage p50 | Retrieval p50 |
|---:|---:|---:|---:|---:|
| 3 | 13 ms | 19 ms | 7.5 ms | 1.5 ms |
| 12 | 16 ms | 23 ms | 9.8 ms | 1.6 ms |
| 50 | 23 ms | 30 ms | 16.6 ms | 1.6 ms |
| 100 | 31 ms | 36 ms | 24.4 ms | 1.7 ms |
| 200 | 50 ms | 56 ms | 42.5 ms | 1.7 ms |
| 500 | 103 ms | 117 ms | 93.2 ms | 2.2 ms |

Sending 2,000 excluded IDs or 100 excluded titles with a 12-Like profile costs
nothing measurable (15.6 ms p50).

Reranker feature construction grows by roughly 0.17 ms per Like. Before V2.1
the request schema capped a profile at 50 Likes, which also capped this cost at
about 23 ms; the cap is now 500. Nothing changed for profiles of 50 or fewer.

## What limits quality, in order

1. **For session-sized profiles, retrieval.** Recall at 300 is 0.27 to 0.44
   with 1 to 10 Likes, and the reranker adds nothing measurable there.
2. **The system does not know what the user has already seen.** Worth +0.045
   NDCG@10 offline; 19% of the top 10 are already-watched, not-loved titles.
3. **The reranker is trained for a different list than the one served.** Half
   its top 10 is removed by the franchise filter, and the product-relevant
   NDCG@10 is 0.216.
4. **For full profiles, ranking inside the candidate set.** 31% of the
   candidate ceiling is captured; retrieval depth is not the issue.
5. **Half of the rating data and half of the list data are unused**: 45% of
   ratings, dropped titles, completed-but-unrated titles, and planned titles.
6. **The benchmark cannot see the tail, sparse users, or validation-holdout
   likes.** These are measurement limits, and they bound what any offline
   challenger can demonstrate.
