# How deep the paged feed goes, and what depth costs

Measurements for `POST /api/recommend/page`. Design and guarantees are in
`frontend/README.md` and `app/services/recommendation_sequence.py`; this
document is the numbers.

Two sources:

- **Serving depth and latency** come from `measure_feed_depth.py`, which drives
  the real `RecommendationService.page` with the production ALS factors and
  production LambdaMART, in process, for 150 real profiles at each size.
  Nothing there is scored against held-out labels.
- **Quality by depth** comes from `small_profile_eval.py`, on the evaluation
  artifacts, because the production models trained on the held-out titles.

## How a sequence is built

```
window 1   ALS ranks   1–300   → LambdaMART order → franchise filter →  ~219 titles
window 2   ALS ranks 301–600   → LambdaMART order → franchise filter →  ~203 titles
window 3   ALS ranks 601–900   → LambdaMART order → franchise filter →  ~196 titles
window 4   ALS ranks 901–1200  → LambdaMART order → franchise filter →  ~191 titles
```

Window 1 is exactly what production ranked before paging existed. Later
windows apply the same two stages to the next 300 titles by ALS score. Windows
are concatenated in ALS order, the franchise filter carries across them, and
nothing is shuffled.

## Depth: how many distinct ranked titles a profile gets

Production stack, four windows, production series policy.

| Likes | Distinct titles, mean | 5th percentile | Minimum | Maximum | Batches of 18 | Titles per window |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 807 | 731 | 691 | 931 | 44.9 | 220, 202, 195, 191 |
| 3 | 807 | 739 | 680 | 911 | 44.8 | 219, 202, 196, 191 |
| 5 | 807 | 739 | 661 | 910 | 44.8 | 219, 204, 195, 189 |
| 10 | 807 | 732 | 689 | 932 | 44.8 | 216, 204, 196, 192 |
| 20 | 806 | 738 | 658 | 926 | 44.8 | 214, 205, 196, 191 |
| 50 | 805 | 730 | 650 | 923 | 44.7 | 209, 205, 198, 192 |

- **About 807 titles, or 45 batches, whatever the profile size.** The lowest
  any of the 900 profiles reached was 650, which is 36 batches.
- The franchise filter removes about a third of each window (300 candidates to
  roughly 200), which is why 1,200 candidates become about 800 titles.
- Across every sequence read in full: **zero repeats**.
- A profile with no Likes gets a 54-title starter sequence, three batches.

Depth is a configuration value (`RECOMMENDATION_SEQUENCE_WINDOWS`), not a model
limit. Each further window adds about 190 titles.

## What the depth is worth

Share of a user's hidden positives that each window contributes (evaluation
stack, as served):

| Cohort, Likes | Window 1 | Window 2 | Window 3 | Window 4 | All four |
|---|---:|---:|---:|---:|---:|
| Truncated, 1 | 21.9% | 7.1% | 3.6% | 2.3% | 34.8% |
| Truncated, 3 | 28.1% | 7.9% | 3.5% | 2.5% | 41.9% |
| Truncated, 10 | 36.6% | 7.9% | 4.1% | 2.1% | 50.7% |
| Truncated, 20 | 41.1% | 8.0% | 3.6% | 2.1% | 54.9% |
| Natural, 3 | 48.3% | 5.5% | 2.3% | 1.3% | 57.4% |
| Natural, 10 | 53.7% | 4.1% | 2.3% | 0.9% | 60.9% |

Hidden positives per batch of 18, truncated profiles with three Likes:

| Batch | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hidden positives | 1.21 | 0.85 | 0.66 | 0.49 | 0.38 | 0.34 | 0.28 |

- **Quality falls steadily with depth, as it should for a ranked list.** The
  first batch is four times as dense as the twelfth.
- **The first window holds two thirds to nine tenths of everything the whole
  sequence will find.** Windows 3 and 4 together add about 6 points for broad
  tastes and 3 for light users.
- So depth beyond the first window is browsing room, not recommendation
  quality. Four windows is a reasonable ceiling; more would mostly add titles
  with a low chance of being wanted.

## Ranking trade-offs

**Against the unpaged path: none for the first window.** It is the same
computation, and a test asserts the first page equals the unpaged result.

**Windows against one large rerank.** A title at ALS rank 320 that LambdaMART
would rate highly appears after everything in window 1, about twelve batches
in. Reranking 600 or 1,200 candidates together would let it rise, but would
change the top of the list too: the reranker's features are relative to its
candidate list, and the reranker experiment already found that widening to 500
candidates did not improve NDCG@10 and reduced coverage. Windows keep page one
identical to what was validated.

**Stability under a change of exclusions.** The reranker scores a title
relative to the rest of its candidate list, so removing one title from a
300-candidate window nudges the others:

| Likes | Same titles in the first batch after one exclusion | Identical order |
|---:|---:|---:|
| 1 | 95.3% | 22.0% |
| 3 | 96.1% | 11.3% |
| 5 | 95.0% | 9.3% |
| 10 | 96.0% | 4.7% |
| 20 | 96.8% | 17.3% |
| 50 | 97.9% | 12.7% |

About 17 of 18 titles stay; their order usually does not. This is why a
sequence is frozen when it starts: marking a title Already Watched, Want to
Watch or Not Interested mid-browse fills that one slot from the same sequence
and does not re-rank the page under the user. The new exclusion takes effect
in the next sequence, on a new Like or a reload.

## Latency

`RecommendationService.page`, production stack, in process, on the development
machine. The HTTP layer adds a few milliseconds; in the browser a cached page
round-trips in about 8 ms and a ranking one in 16 to 55 ms.

| Likes | First page p50 | p95 | Page from cache p50 | Page that ranks a new window p50 |
|---:|---:|---:|---:|---:|
| 1 | 12.5 ms | 16.7 ms | 0.6 ms | 12.9 ms |
| 3 | 14.0 ms | 18.5 ms | 0.6 ms | 14.0 ms |
| 5 | 13.1 ms | 14.4 ms | 0.6 ms | 13.4 ms |
| 10 | 14.6 ms | 15.6 ms | 0.6 ms | 14.7 ms |
| 20 | 16.9 ms | 19.3 ms | 0.6 ms | 16.8 ms |
| 50 | 25.8 ms | 33.2 ms | 0.7 ms | 24.7 ms |

A sequence of 807 titles is read in 23 requests of 36. Four of them rank a
window; the other nineteen are served from the cached sequence.

**After a restart.** A fresh instance holds no sequence and rebuilds every
window up to the cursor. p50, by how deep the user was:

| Likes | Offset 36 | Offset 216 | Offset 432 | Offset 648 |
|---:|---:|---:|---:|---:|
| 1 | 12 ms | 25 ms | 36 ms | 49 ms |
| 10 | 14 ms | 29 ms | 43 ms | 57 ms |
| 20 | 17 ms | 33 ms | 49 ms | 65 ms |
| 50 | 26 ms | 51 ms | 75 ms | 99 ms |

All 3,600 pages resumed this way were identical to the pages the first
instance had served.

**Starter sequences are the slow case.** A profile with no Likes is ranked by
the hybrid, which scores the whole catalog on each call: about 1.8 s for each
18-title starter page that has to be ranked, and 6 ms once cached. Before the
starter sequence was chunked, the first screen took about 6.4 s. The remaining
1.6 s is the hybrid's own scoring pass and is outside this work.

## Request size

The request carries the profile snapshot and a cursor, never the titles shown.
In the browser, a profile with two Likes and one bookmark read all 793 titles
of its sequence with a request body of 446 to 447 bytes throughout. Before
paging, the same browse would have had to send every shown title as an
exclusion.

## Limits

- Sequences are deterministic for one deployment. Two instances on different
  CPU families could differ in the last bit of a floating-point score and swap
  two near-tied titles. The client's own no-repeat filter means that could
  drop or shift a title at a page boundary but never repeat one.
- A deployment that changes a model or its configuration invalidates open
  cursors by design; the client restarts the sequence and keeps what it has
  already shown out of it.
- Skipped titles are sent as exclusions when the next sequence starts. That
  list grows with Replace clicks, not with paging, and is cleared on reload.
