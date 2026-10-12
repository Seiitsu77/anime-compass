# Is the franchise filter too aggressive?

An evaluation of alternative series-filtering policies on the frozen stack.
**No production policy was changed.** Offline metrics are proxies; none of this
is a claim about user satisfaction.

Run with `python experiments/v21_quality/series_policy_eval.py`. Numbers are in
`results/series_policy_eval.json`. Users are the 800 confirmation users of the
reranker experiment, already opened once; candidates are the ALS top 300;
ranking is the confirmed LambdaMART unless stated. Intervals are 95% paired
bootstrap over users.

## What the feed does today

`one_per_series` is one flag that applies two different rules:

1. **De-duplication.** Keep the best-ranked entry of each franchise in a list.
2. **Liked-franchise removal.** Drop every entry of a franchise the user has
   already Liked: sequels, films, and specials alike.

They are not the same thing, and they do not cost the same.

There is also a third, wider rule on the hybrid path, which serves profiles
with no Likes: there the filter is seeded with **every excluded title**, so a
title marked Want to Watch, Already Watched or Not Interested hides its whole
franchise. `tests/test_recommendation_pagination.py` records that behaviour.

## The policies

| Policy | Rule |
|---|---|
| `none` | No franchise filter |
| `production` | One per franchise, and nothing from a Liked franchise |
| `dedupe_liked1` | One per franchise; at most **one** entry from Liked franchises per batch of 18 |
| `dedupe_liked3` | One per franchise; at most **three** such entries per batch |
| `dedupe` | One per franchise; Liked franchises unrestricted |
| `dedupe_mainline` | As `dedupe`, but a Liked franchise may contribute only a TV series or a film |
| `batch_dedupe` | One per franchise within a batch; other entries wait for a later batch |
| `cap2` | At most two entries per franchise |

`production` here is asserted equal to the service's own `_collapse_series` in
`tests/test_v21_protocol.py`.

## Results

35.1% of held-out positives (3,071 of 8,738) are in a franchise the user
already Liked. "New" below means a held-out positive in any other franchise.

| Policy | NDCG@10 | Recall@18 | Hits in 18 | New-franchise hits | Liked-franchise hits | Slots given to Liked franchises |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 0.283 | 0.345 | 2.74 | 1.13 | 1.61 | 7.8 |
| `cap2` | 0.283 | 0.342 | 2.72 | 1.15 | 1.57 | 7.4 |
| `dedupe` | 0.274 | 0.325 | 2.63 | 1.19 | 1.44 | 6.6 |
| `dedupe_mainline` | 0.271 | 0.318 | 2.58 | 1.22 | 1.36 | 6.2 |
| `dedupe_liked3` | 0.263 | 0.293 | 2.25 | 1.55 | 0.71 | 2.8 |
| `dedupe_liked1` | 0.234 | 0.255 | 1.97 | 1.67 | 0.30 | 1.0 |
| **`production`** | **0.182** | **0.206** | **1.75** | **1.75** | **0.00** | **0.0** |

Against `production`, paired:

| Policy | NDCG@10 | Recall@18 | New-franchise hits in 18 |
|---|---|---|---|
| `dedupe_liked1` | +0.052 `[+0.040, +0.065]` | +0.049 `[+0.039, +0.058]` | −0.074 `[−0.091, −0.056]` |
| `dedupe_liked3` | +0.081 `[+0.068, +0.095]` | +0.087 `[+0.075, +0.099]` | −0.198 `[−0.228, −0.169]` |
| `dedupe` | +0.093 `[+0.078, +0.107]` | +0.118 `[+0.105, +0.132]` | −0.551 `[−0.616, −0.486]` |

### What this says

- **De-duplication is nearly free.** Removing near-duplicate entries costs
  0.009 NDCG@10 (0.283 to 0.274) and raises new-franchise hits. That half of the
  rule is not the problem.
- **Liked-franchise removal accounts for almost the whole drop** from the
  benchmarked 0.283 to the served 0.182.
- **It is a trade, not a pure loss.** Removing Liked franchises frees about
  seven slots per page, and they are filled with titles from franchises the
  user has not engaged with. `production` finds 46% more new-franchise
  positives per page than `dedupe` (1.75 against 1.19), and fewer positives in
  total (1.75 against 2.63).
- **Rationing recovers most of one side for little of the other.** One
  Liked-franchise entry per batch gives +0.052 NDCG@10 and +13% total hits for
  4% fewer new-franchise hits. Three gives +0.081 and +29% for 11% fewer.
- `batch_dedupe` is identical to `dedupe` in the first batch by construction.
  Over three batches it reaches more positives (Recall@54 0.533 against 0.500),
  because a franchise's second entry returns later instead of being dropped.
- `dedupe_mainline`, allowing only a TV series or film from a Liked franchise,
  changes little: most of what gets recommended there is already mainline.

### The reranker behaves differently under each policy

| Policy | Reranker gain, all positives | Reranker gain, new-franchise positives |
|---|---|---|
| `production` | +0.025 `[+0.016, +0.034]` | +0.032 `[+0.022, +0.043]` |
| `dedupe_liked1` | +0.027 `[+0.017, +0.037]` | +0.015 `[+0.006, +0.024]` |
| `dedupe_liked3` | +0.030 `[+0.021, +0.040]` | +0.002 `[−0.006, +0.011]` |
| `dedupe` | +0.033 `[+0.024, +0.043]` | **−0.016** `[−0.025, −0.008]` |
| `none` | +0.037 `[+0.028, +0.046]` | **−0.014** `[−0.023, −0.006]` |

Without Liked-franchise removal, LambdaMART is significantly *worse* than ALS
at surfacing positives in new franchises: it was trained on labels where
sequels count, so it promotes them over everything else. The reranker helps
discovery only because the filter then strips those sequels out. Any change of
series policy should come with a reranker trained for that policy.

## "Liking Season 1 does not mean the user has watched Season 2"

For these users it usually does. Each recommended title in the top 18 under
`dedupe` was looked up in that user's full MAL list (`animelist.csv`):

| What the user had already done with it | Liked-franchise recommendations | New-franchise recommendations |
|---|---:|---:|
| Completed and liked (a held-out positive) | 44.3% | 21.2% |
| Completed, rated below 8 | 20.3% | 16.7% |
| Completed, no rating | 4.6% | 5.0% |
| Currently watching or on hold | 4.9% | 6.3% |
| Dropped | 0.8% | 1.6% |
| **Already engaged with, total** | **74.9%** | **50.8%** |
| On their Plan to Watch list | 4.5% | 9.2% |
| Not in their list at all | 20.6% | 40.0% |

- Three in four recommended sequels of a Liked title were already watched or
  started. When rated, they were liked 69% of the time, against 56% for
  new-franchise recommendations: high precision, mostly not news.
- One in four was something the user had not watched: 4.5% planned, 20.6% not
  in their list.
- A new-franchise recommendation is twice as likely to be unseen (49% against
  25%) and twice as likely to already be on the Plan to Watch list.

**The limit of this evidence.** These are MAL users who log what they watch,
with a median of over a hundred completed titles. For them a sequel of a
favourite is usually old news. An Anime Compass user who has Liked three titles
is not that user, and this dataset cannot say how often such a person has seen
the sequel. The measurement supports "sequels are precise and often redundant
for heavy users"; it does not support hiding them from everyone.

## Is the +0.045 from watched-history exclusion free of test-label leakage?

It contains no leaked label, but it is overstated.

**No label is leaked.** Every excluded title is one the user rated below 8, or
watched without rating. A held-out positive is a rating of 8 or higher, so the
two sets cannot overlap. `assert_isolated` checks this for every user and every
arm, validation positives included.

**The split is asymmetric.** It hides 19% of each user's positives and none of
their other ratings. Excluding *every* low-rated title therefore assumes the
system knows a user's complete history of things they did not love while
knowing only four fifths of what they did. Hiding the low-rated titles at the
same rate (80.7% known, chosen deterministically) gives the comparable figure:

| What the system knows the user has watched | Titles excluded | NDCG@10 | Gain | As-served gain, new franchises |
|---|---:|---:|---|---|
| Only their Likes | 0 | 0.283 | | |
| Everything rated below 8 (the earlier figure) | 91 | 0.328 | +0.045 `[+0.040, +0.052]` | +0.034 |
| Rated below 8, **at the matched rate** | 73 | 0.318 | **+0.035** `[+0.030, +0.040]` | +0.024 `[+0.019, +0.028]` |
| All watched titles, at the matched rate | 124 | 0.339 | +0.056 `[+0.050, +0.062]` | +0.041 |
| All watched titles | 154 | 0.360 | +0.078 `[+0.070, +0.085]` | +0.061 |

"All watched" adds the 63 titles a user had completed without rating, was
watching, had on hold, or had dropped.

- The like-for-like benefit of knowing low-rated history is **+0.035**, about
  three quarters of the +0.045 reported earlier.
- Knowing everything a user has watched, not only what they rated, is worth
  more than either: +0.056 at the matched rate.
- These are ceilings for the Already Watched action. They are reached only if a
  user tells the product everything they have seen, which nobody will. The
  realistic benefit is some fraction, and it grows with every title marked.

## Recommendation

Nothing here should change production on its own. What the evidence supports:

1. **Split the flag** into the two rules it contains, so each can be set and
   measured separately. De-duplication stays on.
2. **Test `dedupe_liked1` against `production` as a product experiment**, with
   a reranker retrained for it (experiment A in the plan). It is the smallest
   step that stops hiding every sequel, at the lowest cost to discovery.
3. **Make the hybrid path consistent with the fast path**, so that an Already
   Watched or Want to Watch title does not hide its franchise for a user with
   no Likes.
4. **Let the user settle it per title.** With Already Watched in the interface,
   a recommended sequel the user has seen is one click to remove, and that
   click also improves every later page.

The offline numbers cannot choose between more hits and more discovery. That is
a product decision, and it is better made with an online comparison than with
NDCG.
