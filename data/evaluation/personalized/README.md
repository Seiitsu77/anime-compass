# Personalized Offline Evaluation

This benchmark is separate from the seven manually labelled catalog/agent cases in
[`../benchmark.json`](../benchmark.json). Those cases remain qualitative regression tests. This
pipeline measures per-user ranking on held-out positive ratings.

> Because interaction timestamps are unavailable, this evaluation measures preference
> reconstruction/generalization under a deterministic user-stratified random holdout, not
> chronological next-item prediction.

## Protocol

- Positive: rating `>= 8`; neutral: `6..7`; explicit negative: `<= 5`. The ranges are CLI-configurable,
  stored independently, and ratings outside configured ranges remain known/ignored rather than becoming
  unobserved items.
- Split seed: `42`. The split is independently shuffled per user using a stable hash of seed and user ID.
- Users with fewer than five positives are training-only. Users with 5–9 positives hold out one validation
  and one test item; 10–19 hold out one validation and two test items; users with 20+ hold out floor(10%)
  for each, with a minimum of one.
- LightFM configuration selection uses validation positives only. Primary evaluation uses test positives for
  binary relevance only after selection; test metrics are never fed back into training.
- Every candidate model sees the same catalog and user IDs. Exact training-known positives, neutral ratings,
  explicit negatives, and ignored ratings are excluded from rankings.
- Sampled runs default to a deterministic uniform hash sample for population-level aggregate estimates. The
  activity-balanced sample uses `--sampling-strategy activity_stratified --users-per-stratum N`; its aggregate is
  not population-weighted. The separate `popularity_stratified` diagnostic obtains users with held-out relevant
  head, mid-tail, and long-tail items. Popularity buckets are always computed from training positives only.
- Popularity and item-popularity buckets use positive training counts only. CountSketch is rebuilt from
  observed training ratings only. Rating-derived catalog aggregates used by the hybrid are rebuilt or cleared
  from training data.

## Models

1. `popularity`: global positive training-interaction count, anime ID as tie-breaker.
2. `countsketch_cf`: the existing user-centred CountSketch item-similarity algorithm, trained on the split's
   observed training ratings; profiles use positive training items.
3. `current_hybrid`: the current production hybrid, invoked directly without HTTP, LLM explanations,
   deterministic explanation/result-payload construction, semantic-network calls, or frontend work. The new
   ranking-only interface preserves filters, scores, and diversity ordering while returning IDs only. Content
   metadata is retained; rating-derived aggregates are train-only. Production behavior remains the default.
4. `lightfm_id`: LightFM with user identity, anime identity, and positive training interactions only.
5. `lightfm_hybrid`: LightFM identity features plus sparse, static catalog genre, type, source, frequency-filtered
   studio, decade, and content-rating features. Outcome-derived fields and high-cardinality people data are excluded.
6. `item_item_cosine`: exact adjusted-cosine item similarity over the same user-centred residuals CountSketch
   uses, computed blockwise so the full 18k-by-18k matrix is never materialised, with the top 200 neighbours per
   item retained. This is the reference that isolates what the CountSketch projection costs, because the only
   difference between the two is the random projection.
7. `als`: implicit-feedback alternating least squares (Hu, Koren, and Volinsky, 2008) with the conjugate-gradient
   solver (Takács et al., 2011), trained on train positives only. Item factors are exported; a user vector is
   recomputed at request time by folding their positives into item space, so the model generalises to users absent
   from training.
8. `current_hybrid_learned`: the production hybrid with channel weights fitted from held-out data instead of
   hand-set, via `scripts/train_fusion_weights.py`. Requires `--fusion-weights`.

LightFM is an offline-only training dependency. Exported user/item representations and biases are validated and
served with NumPy; the FastAPI runtime does not import LightFM. The item-item and ALS baselines need only NumPy
and SciPy sparse (`requirements-evaluation.txt`) and add no compiled dependency. LightGCN is not implemented; see
the [collaborative baselines decision report](results/collaborative_baselines_summary.md) for why a graph model is
not the next step.

## LightFM challenger results

- [`results/lightfm_challenger_summary.md`](results/lightfm_challenger_summary.md): decision report, final tables,
  confidence intervals, engineering results, threshold sensitivity, and limitations.
- [`results/lightfm_fulltrain_uniform1000/report.md`](results/lightfm_fulltrain_uniform1000/report.md): Evaluation A,
  representative 1,000-user comparison.
- [`results/lightfm_fulltrain_activity100_each/report.md`](results/lightfm_fulltrain_activity100_each/report.md):
  Evaluation B, 100 sparse/medium/heavy users each.
- [`results/lightfm_fulltrain_popularity100_each/report.md`](results/lightfm_fulltrain_popularity100_each/report.md):
  Evaluation C, quota of 100 qualifying users per item-popularity stratum.

LightFM-ID improves representative threshold-8 NDCG@10 from 0.1534 to 0.1833 (+19.51%; paired 95% CI
[+0.0171, +0.0428]) and Recall@10 from 0.1408 to 0.1589. It is not promoted: coverage is roughly halved, sparse
users and tail retrieval regress, metadata does not help, and the NDCG lift disappears at threshold 9. CountSketch
remains the production collaborative channel while LightFM-ID remains an offline challenger.

## Metric definitions

- Ranking metrics are calculated once per user and macro-averaged. Recall divides top-K hits by the number of
  that user's test positives. Binary NDCG uses logarithmic discount and an ideal ranking of up to K test
  positives. HitRate@10 is one when any test positive occurs in the first ten. MRR is the reciprocal rank of
  the first test positive within 20.
- Catalog coverage is unique recommended IDs divided by catalog size.
- Novelty is mean self-information in bits, `-log2((train_positive_count + 1) /
  (all_train_positives + catalog_size))`.
- Popularity bias is the per-user mean normalized-log popularity of recommendations minus that of the user's
  positive training history. Positive values mean recommendations skew more popular than the profile.
- Intra-list diversity is the mean pairwise Jaccard distance between catalog genre sets.
- Beyond-accuracy metrics use the configured top-20 recommendation lists by default. Serialization timing covers
  the compact offline ranking IDs, not a production HTTP payload.
- User activity uses positive training interactions: sparse `1–4`, medium `5–19`, heavy `20+`. Sparse users
  remain possible because a five-positive user has three training positives after holdout.
- Item buckets rank the full catalog by positive training count with anime ID as a deterministic tie-breaker:
  head is the top 20%, mid-tail the next 30%, and long-tail the bottom 50%, including zero-count items.
- Main model differences use a paired user-level percentile bootstrap, never independent samples.
- Split schema v3 retained a legacy `train_positive_sparsity` key that actually stores density. New readers and
  reports expose unambiguous `train_positive_density` and `train_positive_matrix_sparsity` fields while preserving
  the old key so existing split artifacts remain reusable.

## LightFM environment

Use Linux or WSL for training. LightFM 1.17 is isolated from `requirements.txt` and the web runtime:

```bash
python3.10 -m venv .lightfm-env
source .lightfm-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lightfm.txt
```

Conda/micromamba alternative:

```bash
conda env create -f environment-lightfm.yml
conda run -n anime-lightfm python -m pytest tests/test_lightfm_challenger.py -q
```

Native Windows builds can lack OpenMP and vary by compiler/package build. Before a real Windows run, execute the
native LightFM score-consistency test above in a disposable environment. Prefer WSL/Linux if WARP/BPR crashes.

## Reproduce

From the repository root. Commands below assume the normal project environment is active for evaluation and the
isolated `anime-lightfm` environment is available for training.

```powershell
# Main tests (normal application environment)
.\.venv\Scripts\python.exe -m pytest -q

# LightFM native/export consistency (isolated offline environment)
conda run -n anime-lightfm python -m pytest tests/test_lightfm_challenger.py -q

# Validation-only WARP/BPR training and NumPy export on the full threshold-8 split
conda run -n anime-lightfm python scripts/train_lightfm.py --split data/evaluation/personalized/splits/holdout_seed42_pos8.sqlite --artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8/lightfm --search-profile smoke --validation-users 300 --num-threads 1

# Fast source-prefix smoke: build split, train artifacts, then evaluate
.\.venv\Scripts\python.exe scripts/evaluate_personalized.py --source-user-limit 500 --split-only
conda run -n anime-lightfm python scripts/train_lightfm.py --split data/evaluation/personalized/splits/holdout_seed42_pos8_users500.sqlite --artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8_users500/lightfm --search-profile smoke --validation-users 100 --num-threads 1
.\.venv\Scripts\python.exe scripts/evaluate_personalized.py --source-user-limit 500 --max-evaluation-users 100 --models popularity,countsketch_cf,lightfm_id,lightfm_hybrid --lightfm-artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8_users500/lightfm --bootstrap-iterations 500 --output-dir data/evaluation/personalized/results/lightfm_smoke_users500_uniform100

# Evaluation A — representative sample
.\.venv\Scripts\python.exe scripts/evaluate_personalized.py --sampling-strategy uniform --max-evaluation-users 1000 --models popularity,countsketch_cf,lightfm_id,lightfm_hybrid --lightfm-artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8/lightfm --bootstrap-iterations 2000 --output-dir data/evaluation/personalized/results/lightfm_fulltrain_uniform1000 --progress-every 100

# Evaluation B — 100 users per training-positive activity bucket
.\.venv\Scripts\python.exe scripts/evaluate_personalized.py --sampling-strategy activity_stratified --users-per-stratum 100 --models popularity,countsketch_cf,lightfm_id,lightfm_hybrid --lightfm-artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8/lightfm --bootstrap-iterations 2000 --output-dir data/evaluation/personalized/results/lightfm_fulltrain_activity100_each --progress-every 100

# Evaluation C — 100 qualifying users per train-defined item-popularity bucket
.\.venv\Scripts\python.exe scripts/evaluate_personalized.py --sampling-strategy popularity_stratified --users-per-stratum 100 --models popularity,countsketch_cf,lightfm_id,lightfm_hybrid --lightfm-artifacts-dir data/evaluation/personalized/artifacts/holdout_seed42_pos8/lightfm --bootstrap-iterations 2000 --output-dir data/evaluation/personalized/results/lightfm_fulltrain_popularity100_each --progress-every 100
```

The persistent split and train-only model artifacts are ignored by Git because they are large and reproducible.
Each result directory contains `results.json`; aggregate, segment, item-bucket, engineering, and paired-bootstrap
CSVs; compressed per-user metrics; `report.md`; and a checksummed `manifest.json`. Derived report/CSV views can
be recreated without inference using `python scripts/evaluate_personalized.py --refresh-output <result-dir>`.
The committed [`metadata/split_seed42_pos8.json`](metadata/split_seed42_pos8.json) records the full archive split's
checksums, counts, configuration, build time, and successful accounting audit without committing the 221 MiB SQLite file.
Its 906,417 out-of-candidate rating rows map to the 1,348 raw adult titles intentionally excluded by the catalog
pipeline; they are reported as filtered rows, not silently treated as unobserved feedback.

## Interpretation limits

This can estimate personalized ranking generalization, collaborative-model lift, sparse-user behavior,
popularity exposure, long-tail recovery, and offline engineering cost. It cannot establish chronological
next-anime prediction, online CTR or acceptance, causal satisfaction, or A/B-test impact. The ratings contain
selection and survivorship bias, and the random holdout lets a user's later franchise entries help reconstruct
earlier entries; those limitations should remain visible in any portfolio claim.
