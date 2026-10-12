# V2.1 recommendation-quality work

Offline analysis of the current recommender, kept apart from the application so
that no experiment can reach a production artifact by accident. Nothing here
trains, tunes, or selects a model.

| Document | Question |
|---|---|
| [`BASELINE_DIAGNOSIS.md`](BASELINE_DIAGNOSIS.md) | What limits quality today? |
| [`SERIES_POLICY.md`](SERIES_POLICY.md) | Is the franchise filter too aggressive, and is "already watched" really worth +0.045? |
| [`SMALL_PROFILE_EVAL.md`](SMALL_PROFILE_EVAL.md) | How good is the experience with 1 to 20 Likes? |
| [`PAGINATION_DEPTH.md`](PAGINATION_DEPTH.md) | How deep does the paged feed go, and what does depth cost? |
| [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) | Which controlled experiments next, and under what rules? |

## Scripts

| Script | What it does | Runtime |
|---|---|---|
| `diagnose_baseline.py` | Retrieval ceiling against ranking quality, coverage, segments, exclusions | ~45 s |
| `profile_size_curve.py` | The first, quick profile-size curve (superseded by `small_profile_eval.py`) | ~30 s |
| `profile_animelist_status.py` | Status and progress signal in `archive/animelist.csv` | ~20 s |
| `series_policy_eval.py` | Series-filter policies, sequel status in users' MAL lists, watched-history exclusion at a matched rate | ~2 min |
| `small_profile_eval.py` | Profiles of 1, 3, 5, 10, 20 Likes: retrieval, ranking, coverage, series policies, depth, latency | ~15 min |
| `measure_feed_depth.py` | Depth and latency of the paged feed through the real serving code and production artifacts | ~5 min |

Shared code:

- `protocol.py`: building a K-Like profile with hidden positives kept out of
  every model input, the series policies, and matched-rate history sampling.
  Unit-tested in `tests/test_v21_protocol.py`.
- `stack.py`: the frozen evaluation stack, and the lists of spent and reserved
  users.

Run from the repository root with the project environment, for example:

```powershell
.\.venv\Scripts\python.exe experiments\v21_quality\small_profile_eval.py
```

They need the frozen split and train-only artifacts under
`data/evaluation/personalized/`, the full catalog, the raw archive, and
`lightgbm` (`requirements-evaluation.txt`). Output goes to `results/*.json`.

## Which artifacts are read

- Every quality number comes from the **evaluation** artifacts: the train-only
  ALS factors and the confirmed LambdaMART from the reranker experiment.
- `measure_feed_depth.py` is the one script that loads the **production**
  artifacts, read-only, to measure serving depth and latency. It scores nothing
  against held-out labels, because the production models trained on them.
- Nothing writes to `data/processed/`.

## Users

A user an experiment has scored is spent: usable for diagnostics, never again
as an untouched test.

| Set | Users | Status |
|---|---:|---|
| Earlier experiments (`metadata/inspected_user_ids.json`) | 3,924 | spent |
| Reranker experiment: train, validation, confirmation | 3,300 | spent |
| `populations/small_profile_v1.json`, development cohorts | 2,000 | spent by `small_profile_eval.py` |
| `populations/small_profile_v1.json`, reserved | 2,800 | **UNOPENED** |

Model scoring in the diagnosis and policy scripts uses the reranker
experiment's 800 confirmation users, already opened once.
`diagnose_baseline.py` reproduces that experiment's published numbers as its
first step and records whether it did.

The reserved set is 800 users for truncated profiles and 500 genuinely light
users at each of 3, 5, 10 and 20 Likes. No script scores it. It is the
untouched confirmation set for the first model selection that needs one; see
rule 4 in the plan. `tests/test_v21_protocol.py` fails if it overlaps any spent
user or is no longer marked unopened.
