# Recommendation quality for profiles of 1 to 20 Likes

A separate evaluation of the case most new Anime Compass users are in. The
standing benchmark does not represent it: its users average 86 liked titles and
the uniform 800-user sample contains no sparse user.

Nothing is trained or selected here. Offline metrics are proxies and say
nothing directly about satisfaction.

Run with `python experiments/v21_quality/small_profile_eval.py` (about 15
minutes). Numbers are in `results/small_profile_eval.json`.

## Design

| | |
|---|---|
| Stack | Frozen train-only ALS, top 300, then the confirmed LambdaMART |
| Profile sizes | 1, 3, 5, 10, 20 observed Likes |
| Relevance | The user's hidden positives: validation and test holdouts |
| Known to the models | The K observed Likes. Nothing else |
| Intervals | 95% bootstrap over users; differences are paired |

### Two cohorts, because one is not enough

| Cohort | Who | What it represents | Weakness |
|---|---|---|---|
| **Truncated** | 663 users with 20+ train positives, shown with only K of them | A broad-taste fan at the start of a session | Heavy users pretending to be new |
| **Natural** | 500 users at each K whose whole train profile really is K Likes | A casual viewer | Two to four hidden positives each; no K=1, since the split needs five positives |

The same truncated users are scored at every K, and their profiles nest (the
K=3 profile is the first three Likes of the K=5 profile), so K is the only thing
that changes.

### Keeping hidden positives hidden

- The profile is drawn from train positives only, in a fixed hash order.
- The reranker's features are built from that profile and nothing else.
- The only titles removed from candidates are the profile itself.
- Train positives left out of a truncated profile are neither targets nor
  exclusions. They stay in the list and count as misses, which is the
  conservative reading; the models did train on them.
- `protocol.assert_isolated` checks all of this for every user at every K, and
  `tests/test_v21_protocol.py` checks that it refuses each kind of leak.

### Users

- Truncated cohort: the reranker experiment's confirmation users, already
  opened once.
- Natural cohort: 2,000 fresh users, now recorded as spent in
  `populations/small_profile_v1.json`.
- **Reserved and not scored:** 800 users for truncated profiles and 500 light
  users at each of K = 3, 5, 10, 20. This is the untouched confirmation set for
  the first model selection made against this evaluation. A test fails if it
  overlaps a spent user or stops being marked unopened.

## Results

### Retrieval: how much of what the user will like is in the candidate set

| K | Truncated: Recall@300 | @500 | @1000 | @2000 | Natural: Recall@300 | @500 | @1000 | @2000 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.287 | 0.360 | 0.462 | 0.534 | | | | |
| 3 | 0.370 | 0.450 | 0.552 | 0.623 | 0.660 | 0.713 | 0.763 | 0.799 |
| 5 | 0.420 | 0.504 | 0.608 | 0.675 | 0.754 | 0.797 | 0.847 | 0.870 |
| 10 | 0.503 | 0.586 | 0.692 | 0.756 | 0.778 | 0.817 | 0.861 | 0.889 |
| 20 | 0.599 | 0.687 | 0.781 | 0.842 | 0.814 | 0.860 | 0.906 | 0.927 |

**The earlier 0.27 to 0.44 was one end of a range, not the typical case.**
Genuinely light users are retrieved far better than truncated heavy users at
the same K: 0.66 against 0.37 at three Likes. A casual viewer's two hidden
positives are usually mainstream titles, which ALS finds from almost any
starting point. A broad-taste fan's twenty-five are not predictable from three.

A real new user is somewhere between the two. The truncated numbers are a
floor for someone with wide tastes; the natural ones describe someone whose
few Likes are most of what they watch.

For the truncated cohort retrieval is clearly the constraint: even 2,000
candidates hold only 62% of the hidden positives at K=3.

### Ranking: ALS against ALS + LambdaMART

Truncated cohort (25.6 hidden positives per user):

| K | ALS NDCG@10 | LambdaMART | Gain | ALS NDCG@18 | LambdaMART | ALS Recall@10 | LambdaMART | ALS Recall@18 | LambdaMART |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.082 | 0.073 | **−0.008** `[−0.015, −0.001]` | 0.079 | 0.073 | 0.042 | 0.039 | 0.059 | 0.057 |
| 3 | 0.098 | 0.095 | −0.003 `[−0.010, +0.004]` | 0.099 | 0.100 | 0.052 | 0.054 | 0.079 | 0.086 |
| 5 | 0.111 | 0.114 | +0.002 `[−0.005, +0.010]` | 0.111 | 0.116 | 0.061 | 0.062 | 0.090 | 0.096 |
| 10 | 0.140 | 0.145 | +0.005 `[−0.003, +0.012]` | 0.141 | 0.150 | 0.083 | 0.088 | 0.119 | 0.132 |
| 20 | 0.175 | 0.187 | **+0.011** `[+0.004, +0.019]` | 0.183 | 0.195 | 0.110 | 0.118 | 0.160 | 0.173 |

Natural cohort (2 hidden positives at K=3 and 5, 3 at K=10, 4 at K=20):

| K | ALS NDCG@10 | LambdaMART | Gain | ALS NDCG@18 | LambdaMART | ALS Recall@10 | LambdaMART | ALS Recall@18 | LambdaMART |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.213 | 0.211 | −0.003 `[−0.019, +0.014]` | 0.235 | 0.231 | 0.270 | 0.266 | 0.342 | 0.333 |
| 5 | 0.253 | 0.260 | +0.007 `[−0.009, +0.024]` | 0.273 | 0.284 | 0.319 | 0.334 | 0.380 | 0.412 |
| 10 | 0.257 | 0.284 | **+0.028** `[+0.014, +0.041]` | 0.287 | 0.308 | 0.303 | 0.335 | 0.387 | 0.401 |
| 20 | 0.301 | 0.325 | **+0.024** `[+0.012, +0.036]` | 0.331 | 0.360 | 0.323 | 0.359 | 0.398 | 0.447 |

- **With five Likes or fewer the reranker adds nothing measurable** in either
  cohort.
- **With one Like it is significantly worse than ALS alone** on truncated
  profiles.
- It starts to pay at ten Likes for light users and at twenty for broad ones.
- It was fitted on users averaging 86 Likes. Its profile features (genre
  affinity, studio affinity, item-item similarity to the profile) are computed
  from one to five titles here, which is very little to compute them from.

A less strict reading, for truncated profiles only: the share of the top 10
that the user is known to like at all, counting train positives the profile
did not show. It is optimistic, because the models trained on those titles.

| K | 1 | 3 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| ALS | 0.32 | 0.41 | 0.47 | 0.56 | 0.61 |
| LambdaMART | 0.32 | 0.44 | 0.50 | 0.59 | 0.64 |

So with three Likes, about four of the first ten recommendations are titles
this kind of user is known to like.

### As served: the franchise filter matters most for light users

NDCG@10 of LambdaMART under three series policies. `production` is what the
feed does today; see [`SERIES_POLICY.md`](SERIES_POLICY.md) for the policies.

Natural cohort:

| K | Unfiltered | `production` | `dedupe` | `dedupe` against `production` | New-franchise hits in 18, `dedupe` against `production` |
|---:|---:|---:|---:|---|---|
| 3 | 0.211 | **0.115** | 0.196 | +0.081 `[+0.062, +0.098]` | −0.026 `[−0.040, −0.014]` |
| 5 | 0.260 | **0.146** | 0.240 | +0.094 `[+0.075, +0.116]` | −0.018 `[−0.030, −0.008]` |
| 10 | 0.284 | **0.151** | 0.271 | +0.119 `[+0.099, +0.139]` | −0.070 `[−0.094, −0.048]` |
| 20 | 0.325 | **0.204** | 0.316 | +0.112 `[+0.093, +0.131]` | −0.108 `[−0.138, −0.080]` |

Truncated cohort:

| K | Unfiltered | `production` | `dedupe` | `dedupe` against `production` | New-franchise hits in 18, `dedupe` against `production` |
|---:|---:|---:|---:|---|---|
| 1 | 0.073 | 0.062 | 0.067 | +0.006 `[+0.002, +0.010]` | −0.030 `[−0.044, −0.018]` |
| 3 | 0.095 | 0.082 | 0.089 | +0.007 `[+0.002, +0.013]` | −0.078 `[−0.101, −0.057]` |
| 5 | 0.114 | 0.095 | 0.108 | +0.013 `[+0.007, +0.020]` | −0.140 `[−0.170, −0.112]` |
| 10 | 0.145 | 0.127 | 0.144 | +0.017 `[+0.009, +0.026]` | −0.262 `[−0.303, −0.223]` |
| 20 | 0.187 | 0.150 | 0.185 | +0.035 `[+0.025, +0.045]` | −0.477 `[−0.531, −0.424]` |

- **For light users, removing Liked franchises roughly halves measured
  NDCG@10** (0.211 to 0.115 at three Likes) and buys almost nothing: 3% to 10%
  fewer new-franchise hits is all `dedupe` gives up. Their hidden positives are
  disproportionately the next entry of something they already liked, and the
  filter removes exactly those from the top of the list.
- For broad-taste users the same change is a much smaller gain and a much
  larger cost in discovery, which grows with profile size (24% fewer
  new-franchise hits at K=20).
- The caveat from the policy document applies in full: a held-out positive in a
  Liked franchise is a sequel the user did go on to like. This data cannot say
  whether they had already seen it when they would have been shown it.

### Coverage

Distinct titles in the top 10 across each cohort, and how concentrated they
are. Every list is at least 98.9% head items.

| Cohort, K | ALS distinct | LambdaMART distinct | Catalog coverage (LambdaMART) | Top-20 titles' share of slots: ALS | LambdaMART |
|---|---:|---:|---:|---:|---:|
| Truncated, 1 | 1,453 | 1,826 | 10.1% | 8.5% | 24.3% |
| Truncated, 3 | 1,112 | 1,507 | 8.3% | 8.8% | 21.4% |
| Truncated, 5 | 997 | 1,362 | 7.5% | 9.6% | 19.2% |
| Truncated, 10 | 815 | 1,115 | 6.2% | 12.5% | 17.6% |
| Truncated, 20 | 740 | 936 | 5.2% | 15.3% | 20.6% |
| Natural, 3 | 991 | 1,189 | 6.6% | 12.4% | 29.8% |
| Natural, 5 | 932 | 1,114 | 6.2% | 12.9% | 25.4% |
| Natural, 10 | 807 | 940 | 5.2% | 15.1% | 21.4% |
| Natural, 20 | 788 | 881 | 4.9% | 14.6% | 22.0% |

LambdaMART does two things at once for small profiles: it brings more distinct
titles into someone's top 10, and it hands a much larger share of all slots to
the same twenty titles. For light users with three Likes, twenty titles take
30% of every top-10 slot. Small profiles get popular recommendations, and the
reranker makes them more uniform.

### Latency

In process, single thread, on the development machine. The HTTP layer adds a
few milliseconds; see [`PAGINATION_DEPTH.md`](PAGINATION_DEPTH.md) for the
serving path.

| K | First page p50 | p95 | Of which LambdaMART | Rebuilding all four windows p50 | p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.0 ms | 6.0 ms | 4.8 ms | 20 ms | 25 ms |
| 3 | 5.8 ms | 6.9 ms | 5.6 ms | 24 ms | 29 ms |
| 5 | 5.9 ms | 7.0 ms | 5.8 ms | 24 ms | 29 ms |
| 10 | 7.2 ms | 8.4 ms | 7.1 ms | 29 ms | 35 ms |
| 20 | 8.7 ms | 10.7 ms | 8.6 ms | 36 ms | 43 ms |

ALS fold-in and retrieval take 0.12 ms. Latency is not a constraint at these
profile sizes.

### Depth

Each profile gets about 810 distinct titles from four windows whatever its
size (5th percentile about 735, minimum 633). What that depth is worth, as the
share of hidden positives each window contributes:

| Cohort, K | Window 1 | Window 2 | Window 3 | Window 4 | Whole sequence |
|---|---:|---:|---:|---:|---:|
| Truncated, 3 | 28.1% | 7.9% | 3.5% | 2.5% | 41.9% |
| Truncated, 10 | 36.6% | 7.9% | 4.1% | 2.1% | 50.7% |
| Natural, 3 | 48.3% | 5.5% | 2.3% | 1.3% | 57.4% |
| Natural, 10 | 53.7% | 4.1% | 2.3% | 0.9% | 60.9% |

The first window, about twelve batches, holds most of what the sequence will
ever find. By the twelfth batch a truncated K=3 profile sees 0.28 hidden
positives per batch, against 1.2 in the first.

## What this means for a user with 1 to 10 Likes

1. **Expect recommendations to be popular and plausible rather than personal.**
   Light users are served reasonably (a third of their hidden positives in the
   first 18 before filtering); broad-taste users are not, until they Like more.
2. **The reranker is not helping them.** No measurable gain at five Likes or
   fewer, and a small loss at one. Serving ALS order below a profile-size
   threshold would cost nothing measured here and would save the reranker's
   latency; that is a routing experiment, not a retraining one.
3. **The franchise filter is the largest single lever for light users**, at
   +0.08 to +0.12 NDCG@10, with the caveat above about what a sequel means.
4. **Retrieval is the constraint for broad tastes.** That is where
   small-profile retrieval work should aim, and it is what the reserved
   confirmation set is for.
5. **Every Like matters a great deal at this size.** Recall@300 rises from 0.29
   to 0.50 between one and ten Likes for truncated profiles. Making it easy to
   Like, and to mark Already Watched, is worth more than any model change
   measured here.
