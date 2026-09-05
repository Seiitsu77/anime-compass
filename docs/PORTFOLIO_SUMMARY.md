# Portfolio Summary

Three versions of the same story, for three different audiences. All numbers are
reproducible from this repository.

---

## One line — resume bullet

> Built and shipped a personalized anime recommender over 57M+ historical
> ratings; replaced a ten-channel hybrid with a tuned implicit-ALS retrieval
> path after controlled offline experiments, improving NDCG@10 by 42.6% (paired
> 95% CI [+0.064, +0.090]) while cutting p50 recommendation latency from ~925 ms
> to ~2 ms.

Shorter variant if space is tight:

> Personalized recommender over 57M ratings: +42.6% NDCG@10 and ~465× lower
> latency by replacing a ten-channel hybrid with tuned implicit ALS, then a
> further +15.1% NDCG@10 from a LambdaMART reranker over the ALS candidate set,
> each chosen on its own controlled benchmark with predeclared decision rules.

---

## 30 seconds — recruiter screen

I built an anime recommendation system on a public dataset of 57 million
ratings. The production model is implicit-feedback ALS trained on 30.9 million
positive interactions across 18,000 titles.

The part I would point at is the process rather than the model. I started from a
ten-channel hybrid recommender and ran a series of controlled offline
experiments to work out what was actually earning its place. The answer was:
much less than was there. Replacing the hybrid's collaborative channel with a
properly tuned ALS model improved ranking quality by 42.6% and made
recommendations about 465 times faster, because the other nine channels turned
out to add no measurable ranking value once the collaborative signal was strong.

Several things I tried did not work — pretrained sentence embeddings, a learned
fusion model, and routing sparse users to a different model all measured worse
than the simpler alternative. Those are written up alongside the successes,
because deciding what to remove was most of the work.

There is a live demo where you can pick a few titles and get recommendations in
about two milliseconds, with an explanation of which of your picks each result
resembles.

---

## Two minutes — MLE / AI engineering interview

**Problem and data.** Anime Recommendation Database 2020: 57.6M ratings from
310k users over 18,064 titles. No interaction timestamps, which matters — it
makes this preference reconstruction, not next-item prediction, and rules out
sequential models like SASRec on a fair evaluation.

**Evaluation first.** Before touching models I built a leakage-safe harness: a
deterministic per-user positive holdout, full-catalog ranking against every
held-out positive, and paired bootstrap confidence intervals. I deliberately did
not use the common one-positive-versus-99-sampled-negatives protocol. The same
model scores 0.826 under sampled negatives and 0.2588 under full-catalog
ranking — a category difference, not a model difference — and Krichene & Rendle
showed sampled negatives can reorder which model looks better. I report the
sampled figure only as a comparability diagnostic.

**Baselines before architecture.** The incumbent collaborative channel was a
CountSketch projection — a compression trick, not a learned model. Two baselines
were missing, so I added them: exact adjusted-cosine item-item, which isolates
what the projection costs, and implicit ALS, the standard latent-factor
reference. ALS won. Critically, the initial ALS numbers came from stock
hyperparameters; a validation-only sweep found `alpha` was badly mis-set, and
tuning moved validation NDCG@10 from 0.2032 to 0.2787 on the same 800 users
(+37.2%). The baseline was untuned long before it was under-powered.

**Discipline around confirmation.** Every confirmation sample excludes all users
any earlier experiment scored — 3,706 by the end — with disjointness asserted in
code and recorded in each run manifest. Decision rules were frozen in writing
before the confirmation set was opened, once. ALS passed a four-gate rule:
significant NDCG@10 and Recall@10 gains, coverage within 90% of the incumbent,
and no worse popularity bias.

**The architecture finding.** Substituting ALS into the full hybrid gave NDCG@10
0.2629 against standalone ALS at 0.2624 — an interval including zero — for 125×
the latency. So the hybrid stopped being the default and became the path for
constraint-rich requests only, where it does something the benchmark never
scored: entity resolution, exact catalog joins, and grounded explanations.

**Negative results.** Pretrained synopsis embeddings carried the second-largest
weight in the blend and had never actually been built; when I built and measured
them they cost 8.9% NDCG@10, so the weight was retired. A learned linear fusion
model generalised worse than hand-set weights on untouched users. Segment-aware
routing measured worse for sparse users than global ALS. Each was rejected on a
predeclared rule.

**Production engineering.** Serving is NumPy-only — no SciPy, no training code
in the web process. Only item factors ship; a user vector is reconstructed per
request by folding their positives into item space, verified faithful against
the trained factors at cosine 0.9995, which means the same 7 MB artifact serves
sessions the model has never seen. Evaluation and production use separate
artifacts with distinct roles, and the loader refuses to serve one for the
other's job. Catalog mismatches refuse startup rather than silently degrading.

**Honest limits.** No timestamps, a snapshot ending in 2020, no real users, and
ALS puts 100% of its recommendation exposure inside the top 20% most popular
items — it has no tail reach on its own, which is why the item-item source is
retained as an option. These are documented rather than papered over.

---

## The reranker, briefly

ALS retrieves well — Recall@300 is 0.7932 — so the open question was ordering,
not retrieval. A LambdaMART reranker over the frozen ALS top-300 improved
NDCG@10 from 0.2456 to 0.2828 on 800 users no earlier experiment had touched,
paired 95% CI `[+0.0276, +0.0465]`.

Two details make it worth telling. Over half the gain comes from item-item
similarity features — a signal this project had already *rejected* as a
candidate generator, where it cost 9% NDCG for tail reach. It is a poor
retriever and a good tie-breaker, and those are not contradictory. And the
choice of LambdaMART over a simpler linear model was not made on accuracy
alone: both need the same 17 MB feature artifact and the same per-request
feature construction, so the simpler model would have saved 5 MB of 23 and
0.27 ms of 5.7 while concentrating the catalog 15% harder.

Two percentages, two samples: **+42.6%** (Hybrid to ALS) and **+15.1%** (ALS to
LambdaMART) come from different confirmation sets and are never combined.

## Talking points by role

**Data Science** — evaluation design, protocol comparability, paired bootstrap
intervals, predeclared decision rules, and reporting negative results.

**MLE** — hyperparameter sweeps that isolate one factor, artifact integrity and
role separation, fold-in serving, latency work grounded in profiling (a
quadratic reranker and a coupled explanation pass were the actual costs).

**AI Engineering** — schema-constrained LLM intent parsing with typed tool
routing, bounded deterministic replanning that never relaxes verified entity
constraints, and grounded explanations computed from learned item-factor
similarity rather than generated text.

---

## What I would do next

Not implemented, deliberately — this is where the project stops:

- Acquire interaction timestamps, which would make sequential models evaluable.
- Address ALS's zero tail exposure with exposure-aware reranking rather than a
  larger model.
- Widen the sparse-user experiment; the current n=30 is not conclusive.
