# Semantic Channel: Decision Report

## Decision

**The semantic channel's 0.14 weight is not justified. Reduce or remove it.**

The pretrained-embedding channel carried the second-largest weight in the
advertised blend (0.14 of 1.00) and had **never been active in a single
published measurement**. Building the artifact and turning it on shows the
channel does not merely fail to help — at its configured weight it makes the
hybrid measurably worse.

On 300 identical held-out users, with every other setting fixed:

| Metric | Semantic off | Semantic on | Delta | 95% CI | Excludes 0 |
|---|---:|---:|---:|---:|---|
| NDCG@10 | 0.1935 | 0.1763 | **−0.0172** (−8.9%) | [−0.0270, −0.0076] | yes |
| Recall@10 | 0.1868 | 0.1645 | **−0.0223** (−11.9%) | [−0.0357, −0.0095] | yes |
| HR@10 | 0.6567 | 0.6233 | −0.0333 | [−0.0667, +0.0000] | no |

Two independent benchmarks agree. The seven-case catalog/agent benchmark's
final-hybrid Hit Rate@10 fell from 1.000 to 0.800 when the channel was enabled,
though at n=5 that alone would be noise.

## How this went unnoticed

The channel was documented, weighted, and wired — but never built.

- `data/processed/semantic_embeddings.npz` did not exist.
- It was **absent from `data/artifacts.manifest.json`**, so no deployment would
  have downloaded it either. A Hugging Face Space would have run nine channels.
- The published catalog/agent benchmark recorded the failure verbatim:
  `"semantic_model": {"available": false, "reason": "RuntimeError"}`.
- The three offline evaluation paths (`runner.py`'s two hybrid builders and
  `train_fusion_weights.py`) constructed `AnimeRecommender` **without** passing a
  semantic index at all, so the channel was structurally inert there regardless
  of whether the artifact existed.

The consequence: every number ever published for this project described a
ten-channel model but was produced by a nine-channel one. The weight table in the
README was aspirational, not measured.

## What changed

- Built the artifact: 18,064 titles, 384 dimensions, `all-MiniLM-L6-v2` pinned to
  revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. 73 seconds on CPU.
- Added it to `data/artifacts.manifest.json` with `required: false`, and taught
  `ensure_artifacts` to honour that flag. Optional artifacts that are missing or
  fail checksum are now skipped with a warning instead of aborting startup, which
  matches the application's existing graceful degradation.
- Passed a semantic index into both hybrid builders in the evaluation runner and
  into the fusion trainer, behind a `--semantic-artifact` flag.
- Verified the artifact loads against the *sanitized* evaluation catalog: the
  content checksum is over `id` plus story text, and train-only sanitization only
  clears rating-derived aggregates, so the checksums match exactly.

## Is this leakage?

No. Semantic vectors are derived from synopsis text, not from ratings. That is
the same argument that already justifies the metadata TF-IDF, synopsis TF-IDF,
and LSA channels, all of which read the same content. No held-out positive
influences the embeddings.

## Why the channel likely hurts

The most plausible explanation is **redundancy plus misallocation**. Three
existing channels (metadata TF-IDF 0.16, synopsis TF-IDF 0.10, LSA 0.04) already
read the same synopsis text. A general-purpose sentence encoder over the same
field adds little independent signal while consuming 0.14 of the weight budget —
weight that was, in effect, taken from the channels that do carry signal.

The single-channel result supports this: `pretrained_semantic` alone scores Hit
Rate@10 of 0.200 on the catalog benchmark, tied with the two TF-IDF channels and
well below LSA (0.400) and collaborative (0.600). It is not a strong channel on
its own, and it is not adding a new view of the item.

Two corroborating observations:

- Beyond accuracy, the channel is close to neutral: catalog coverage is flat
  (−0.0004), popularity bias improves slightly (−0.0076), novelty improves
  slightly (+0.128 bits), intra-list diversity worsens slightly (−0.0112). It is
  not buying exposure benefits in exchange for the accuracy it costs.
- The [learned fusion fit](learned_fusion_summary.md) also drove this weight to
  zero. That run is not independent evidence, because the channel was inactive
  and therefore had no variance to fit — but the fit should now be repeated with
  the channel live, where a genuine downward pull would be meaningful.

## Recommendation

1. **Do not ship 0.14.** On this evidence the honest default is to set the
   semantic weight to 0 and redistribute, or to justify a much smaller value.
2. **Re-run the learned fusion fit with the channel active.** It is now a real
   feature with real variance. If the fit independently drives it toward zero,
   that is convergent evidence from a different method.
3. **If the channel is worth keeping, change what it encodes.** Embedding the
   same synopsis text that three other channels already read is the likely
   defect. Encoding something they do not — combined title, themes, and
   demographic context, or a review-derived summary — would give it an
   independent view. That is a modelling change, not a weight change.
4. **Keep the artifact and the wiring regardless.** The channel must be
   measurable to be dismissed, and it now is.

## Reproduce

```powershell
python scripts/build_semantic_embeddings.py --force --offline --catalog data/processed/anime_catalog.json

# Semantic on
python scripts/evaluate_personalized.py `
    --split data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite `
    --artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8 `
    --output-dir data/evaluation/personalized/results/semantic_on_users300 `
    --models popularity,countsketch_cf,current_hybrid `
    --max-evaluation-users 300 --semantic-artifact data/processed/semantic_embeddings.npz

# Semantic off, same deterministic users
python scripts/evaluate_personalized.py `
    --split data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite `
    --artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8 `
    --output-dir data/evaluation/personalized/results/semantic_off_users300 `
    --models popularity,countsketch_cf,current_hybrid `
    --max-evaluation-users 300 --semantic-artifact ""
```

## Limitations

- 300 users, one split, one threshold. The NDCG and Recall intervals exclude
  zero, but this has not been checked at thresholds 7 or 9, nor on activity or
  tail strata.
- Only one encoder and one text schema were tested. This result is about
  `all-MiniLM-L6-v2` over `anime-story-v1` text at weight 0.14; it is not a
  general claim about pretrained embeddings for this catalog.
- The weight itself was never tuned, because the channel was never active. A
  smaller non-zero weight has not been swept.
