# Why NDCG@10 Here Is Not Comparable To Published NDCG@10

## The short answer

Published recommender NDCG@10 figures in the 0.5–0.7 range almost always come
from a **sampled-negative** protocol: hold out one interaction, sample 99 items
the user has not seen, and rank those 100. We rank the **entire 18,064-item
catalog** and count **every** held-out positive. Those two numbers are not on the
same scale, and comparing them is a category error.

The same ALS model, on the same 800 confirmation users, from the same split:

| Protocol | NDCG@10 |
|---|---:|
| **A.** All test positives, full 18,064-item catalog | **0.2875** ← what this repo reports |
| **B.** Leave-one-out, full 18,064-item catalog | 0.1775 |
| **C.** Leave-one-out, 1 positive + 99 sampled negatives | **0.8260** ← the protocol behind most published numbers |

Under the protocol that produces the numbers people quote, this model scores
**0.826**. Under our protocol it scores 0.2875. Nothing about the model changed.

This is the well-documented sampled-metrics problem: Krichene and Rendle
(KDD 2020) showed that sampled negatives do not merely shift metrics, they can
reorder which model looks better. Protocol C is reported here only to make the
scale explicit — it is not used for any decision in this repository.

## Why we chose the harder protocol

Protocol A is what the product actually does. When a user asks for
recommendations, the system ranks the whole catalog, not a shortlist containing
a guaranteed answer. A metric computed against 99 random negatives measures a
task the system never performs.

The cost is that our headline numbers look small next to papers. The benefit is
that they mean something operationally.

## What the numbers actually say

On the 800 confirmation users, with ALS ranking all 18,064 items:

- **HR@10 = 0.8175.** For 82% of users, at least one of their held-out
  favourites appears in the top 10 of an 18,064-item catalog.
- **Recall@10 = 0.2618** against a hard ceiling of **0.8556**, because 33% of
  users have more than 10 held-out positives and only 10 slots exist. The model
  reaches 31% of the maximum attainable value, not 26% of 1.0.
- **NDCG@10 = 0.2875**, where the task is: given ~88 titles a user liked, place
  ~10.6 *specific* other titles they liked into the top 10 of 18,064.

## What genuinely limits the score

Protocol accounts for most of the gap to published figures. These are the real
constraints that remain.

**1. No interaction timestamps (largest data limitation).** The holdout is a
random split of each user's liked set, so the task is *set completion*, not
next-item prediction. Sequence, recency, and trend signals — the things that
drive most modern gains — are unavailable. Protocol B (0.1775) shows how hard
even single-item prediction is without them. With timestamps, a sequential model
(SASRec, GRU4Rec) could exploit information the current benchmark cannot see at
all. This is the single highest-value data acquisition for this project.

**2. Rating magnitude is discarded.** Feedback is binarised at `rating >= 8`, so
an 8 and a 10 are identical to the model, and ALS trains on positives only. The
archive has explicit 1–10 ratings; a weighted-confidence or explicit-feedback
formulation could use that signal. The evaluation would need to change alongside
it, since graded relevance would raise NDCG's ceiling.

**3. Many held-out positives per user, ten slots.** Recall@10 cannot exceed
0.8556 in this population. Reporting Recall@20 or NDCG@20 alongside would give a
fairer view of the same model.

**4. Single-stage full-catalog scoring.** There is no retrieval-then-rerank
architecture. A two-stage design — cheap recall of a few hundred candidates,
then an expensive reranker — is how production systems get both quality and
latency, and it is the natural place for a learned reranker to earn its keep.

**5. The snapshot ends in 2020**, so there are no post-2020 interactions to learn
from and no temporal dynamics.

## What actually moved the number

Tuning, not architecture. The stock ALS defaults (64 factors, alpha 40) scored
NDCG@10 0.1841. A validation-only sweep found alpha was badly mis-set and that
capacity mattered; at 128 factors and alpha 5.0 the same model family scores
0.2875 — a **+56% relative improvement from hyperparameters alone**, and +75%
over the production CountSketch channel.

That ordering is the lesson: the baseline was untuned long before it was
under-powered. Reaching for a more complex model family before exhausting the
simple one would have attributed a tuning win to an architecture change.

## Reproduce

The protocol comparison is a standalone script; it reads the confirmation users
and the selected ALS artifact and computes all three protocols in one pass.
Protocol C is diagnostic only and must not be used for model selection.
