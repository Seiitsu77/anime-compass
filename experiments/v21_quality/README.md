# V2.1 recommendation-quality diagnosis

Read-only analysis of the current recommender, kept apart from the application
so that no experiment can reach a production artifact by accident.

- [`BASELINE_DIAGNOSIS.md`](BASELINE_DIAGNOSIS.md): what limits quality today, with the numbers.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md): prioritised challengers and the rules they must follow.

## Scripts

| Script | What it does | Runtime |
|---|---|---|
| `diagnose_baseline.py` | Retrieval ceiling against ranking quality, coverage, segments, exclusions, `one_per_series`, rating scale | ~45 s |
| `profile_size_curve.py` | The same stack with each profile cut to K liked titles | ~30 s |
| `profile_animelist_status.py` | Status and progress signal in `archive/animelist.csv` | ~20 s |

Run from the repository root with the project environment:

```powershell
.\.venv\Scripts\python.exe experiments\v21_quality\diagnose_baseline.py
.\.venv\Scripts\python.exe experiments\v21_quality\profile_size_curve.py
.\.venv\Scripts\python.exe experiments\v21_quality\profile_animelist_status.py
```

They need the frozen split and train-only artifacts under
`data/evaluation/personalized/`, the full catalog, the raw archive, and
`lightgbm` (`requirements-evaluation.txt`). Output goes to `results/*.json`.

## What these scripts do not do

- They never read or write `data/processed/`. The production ALS artifact and
  the production reranker are not loaded.
- They train nothing and select nothing.
- They spend no untouched evaluation user. Model scoring uses the 800
  confirmation users of the reranker experiment, which were already opened
  once. `diagnose_baseline.py` reproduces that experiment's published numbers
  as its first step and records whether it did.

Those 800 users are now diagnostic users. Do not use them to confirm a
challenger; see rule 4 in the plan.
