# Anime Recommendation System

A production-style personalized anime recommender built on 57M+ historical ratings. Pick a few titles you
like and it ranks the full ~18,000-item catalog against your profile in about two milliseconds, using a tuned
implicit-feedback ALS model trained on 30.9M positive interactions. The interesting part is not only the model — it
is the evidence trail behind it: the major model and routing decisions were made with controlled offline experiments,
including predeclared confirmation rules, and several promising ideas were measured, rejected, and documented.

## Live Demo

https://anime-compass-xijwv9hjuqvoskonhrwtus.streamlit.app/

No login, no account, no API key. Load an example profile, click Recommend, and inspect the model results and
architecture from the same page.

## Why This Project Is Interesting

- **A real model-selection story, not a single notebook.** Popularity → CountSketch → exact item-item → LightFM →
  tuned ALS, each compared on identical users with paired bootstrap intervals.
- **Negative results are first-class.** Pretrained embeddings, learned fusion, and segment-aware routing were all
  tested and all rejected on evidence. They are documented, not hidden.
- **The final architecture is simpler than what it replaced**, and that simplification was measured rather than
  asserted: the previous ten-channel hybrid added no measurable ranking value once ALS was strong.
- **The evaluation protocol is deliberately hard.** Full-catalog ranking against every held-out positive, not the
  common one-positive-versus-99-sampled-negatives shortcut.
- **Production concerns are handled**: artifact checksums, role separation between evaluation and serving models,
  fail-loud catalog validation, and a NumPy-only serving path.

## Key Results

| | |
|---|---|
| Ratings scanned | **57,633,278** |
| Positive interactions in the production model | **30,875,410** |
| NDCG@10 vs the previous production architecture | **+42.6%** (paired 95% CI `[+0.0641, +0.0897]`) |
| Recommendation latency | **~465× lower** (924.6 ms → 2.0 ms p50) |
| Evaluation protocol | Full ~18,000-item catalog, all held-out positives |

Primary comparison, same 800 held-out users, same protocol:

| Architecture | NDCG@10 | Recall@10 | NDCG@20 | Recall@20 | p50 |
|---|---:|---:|---:|---:|---:|
| Old Hybrid + CountSketch | 0.1815 | 0.1682 | 0.2000 | 0.2395 | 924.6 ms |
| **Fast production ALS** | **0.2588** | **0.2480** | **0.2889** | **0.3572** | **2.0 ms** |

NDCG@10 of 0.2588 is a ranking-quality score against the full catalog, not an accuracy percentage.

## Demo

The demo is a single Streamlit page:

1. **Landing** — headline, three metrics, one-paragraph explanation.
2. **Pick titles** — four one-click example profiles, plus catalog search.
3. **Recommendations** — poster cards with year, type, episodes, score, genres, and a grounded explanation
   naming which of your titles the result actually resembles.
4. **Under the hood** — collapsible sections for architecture, model results, evaluation protocol, rejected
   experiments, and deployment health.

## Architecture

```text
                 User preferences
                        |
                        v
                  Request router
                    /         \
           simple request    constrained request
                 |                    |
                 v                    v
          Production ALS          Rich Hybrid
            fast path           constraint path
                 \                    /
                  v                  v
                    Recommendations
                          |
                          v
                     Explanations
```

ALS serves the default personalized path. The multi-channel hybrid runs only when a request carries explicit
constraints — a studio, a voice actor, a year window — that need exact catalog joins. Full detail in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Modeling

| Model | NDCG@10 | Recall@10 | Decision |
|---|---:|---:|---|
| Random | 0.0008 | 0.0009 | Floor reference |
| Popularity | 0.1087 | 0.1000 | Baseline |
| CountSketch (previous production) | 0.1516 | 0.1430 | Replaced |
| Exact item-item cosine | 0.1650 | 0.1472 | Retained as an optional retrieval source |
| LightFM-ID | 0.1833 | 0.1589 | Rejected — halved coverage, failed threshold-9 robustness |
| **Tuned ALS** | **0.2624** | **0.2475** | **Promoted** |
| Oracle (analytic ceiling) | 1.0000 | 0.8650 | Not deployable; validates the metric |

Exact item-item and LightFM-ID were measured on an earlier 1,000-user sample; the rest come from 800-user
confirmation samples. Populations differ, so cross-row gaps are indicative rather than paired.

Tuning mattered: on the same 800 validation users, the stock-like 64-factor, `alpha=40` configuration scored
0.2032 NDCG@10, while the selected 128-factor, `alpha=5` configuration scored 0.2787 — a **37.2% validation
gain** before the untouched confirmation set was opened.

## Evaluation

Every headline number ranks each user's held-out favourites against the **full ~18,000-item catalog**. This is
substantially harder than the common protocol of one positive against 99 sampled negatives, where the same model
scores NDCG@10 **0.826**. That figure is reported only as a comparability diagnostic and never used to select a
model.

- ~10.6 held-out positives per user, so Recall@10 has a hard ceiling of **0.865**.
- Confirmation samples exclude every user any earlier experiment scored; disjointness is asserted at sample time.
- Decision rules are frozen in writing before the confirmation set is opened, once.
- Evaluation and production use **separate model artifacts**. The evaluation artifact withholds held-out
  positives; the production artifact trains on everything and is refused for measurement.

See [docs/EVALUATION.md](docs/EVALUATION.md).

## What Did Not Work

| Experiment | Result | Decision |
|---|---|---|
| Pretrained synopsis embeddings | −8.9% NDCG@10, −11.9% Recall@10 on held-out users | Retired from the default path |
| Learned linear fusion | Generalised worse than the hand-set blend (0.1814 vs 0.1996) | Rejected |
| Segment-aware routing | Sparse-user NDCG@10 fell from 0.2003 to 0.1660 | Disabled by default |
| Item-item supplementation | Restored tail reach, cost 9% NDCG@10 and 4× latency | Kept optional |
| Rich Hybrid as the default | No measurable ranking gain over ALS alone, 125× the latency | Constraint-rich requests only |

Each had a predeclared decision rule. Reporting them is the point: the architecture is what survived.

## Engineering

- **395 tests**, ruff, ruff-format, and mypy across `app`, `backend`, and `scripts`, all enforced in CI.
- **Artifact integrity**: SHA-256 pinning, catalog-digest pinning, and role separation between the evaluation and
  production models. A catalog mismatch refuses startup rather than silently serving a stale model.
- **NumPy-only serving.** No SciPy, no training code, and no ML framework in the web process.
- **Reproducibility**: dataset, split, catalog, and artifact hashes recorded for every published number.
- Known limitation: the interaction snapshot ends in 2020, so this measures preference reconstruction rather
  than next-item prediction. There are no timestamps.

## Run Locally

Windows PowerShell:

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

`requirements.txt` is the demo's complete runtime, and it is what Streamlit Community Cloud installs
automatically.

The demo needs two artifacts that are too large for Git. Fetch them into `data/processed/`:

```bash
python scripts/download_artifacts.py --repo-id Seiitsu/anime-compass-data --artifact als_production_item_factors.npz --artifact anime_catalog_serving.json
```

Or set `ALS_ARTIFACT_URL` and `SERVING_CATALOG_URL` (see [Deployment](#deployment)) and let the app
download them on first run. Either way they are checksum-verified against `data/artifacts.manifest.json`
before use. No API key is needed.

The serving catalog holds only the ten fields the demo reads, which is 6.6 MB
instead of 119 MB for identical recommendations. The demo falls back to the full
catalog when only that is present, which is what a development checkout usually has.

The complete FastAPI/agent application has a separate runtime and needs the full
catalog artifacts described in [`docs/SYSTEM_REFERENCE.md`](docs/SYSTEM_REFERENCE.md):

```powershell
python -m pip install -r requirements-api.txt
python run_app.py
```

To rebuild the two demo artifacts from the raw CC0 files instead of downloading them, install the
offline dependencies and run:

```bash
python -m pip install -r requirements-evaluation.txt
python scripts/prepare_data.py
python scripts/build_production_als.py
python scripts/verify_production_als.py
python scripts/build_serving_catalog.py
```

## Deployment

**Streamlit Community Cloud.** The demo imports the recommendation core directly rather than calling FastAPI over
HTTP — both would run in the same process, so the extra hop would add a failure mode and nothing else.

The deployment payload is two files, about 14 MB in total:

| File | Size | SHA-256 |
|---|---:|---|
| `als_production_item_factors.npz` | 7.4 MB | `95c079b1b8f4e0e509c8bab29e4357360f851e3adfd2abc261f358375ee13a10` |
| `anime_catalog_serving.json` | 6.6 MB | `4b11cf3e5f3d015ed071232e569068248af709e7979e561194b451e3ac716bf3` |

Both are hosted in a public Hugging Face Dataset repo and fetched over https at startup, then verified
before use. Only the two URLs go in Streamlit secrets: the expected checksums are already pinned in
`data/artifacts.manifest.json`, and the model embeds the exact catalog-ID digest it was trained against, so
a changed catalog is rejected even when more than 90% of IDs still overlap.

1. Push this repository to GitHub.
2. Upload both files to a public Hugging Face Dataset repo — this project uses
   [`Seiitsu/anime-compass-data`](https://huggingface.co/datasets/Seiitsu/anime-compass-data);
   see [docs/PUBLISHING.md](docs/PUBLISHING.md).
3. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing at `streamlit_app.py`,
   on Python 3.12. Dependencies come from `requirements.txt` automatically.
4. Set two secrets:

```toml
ALS_ARTIFACT_URL = "https://huggingface.co/datasets/Seiitsu/anime-compass-data/resolve/main/als_production_item_factors.npz"
SERVING_CATALOG_URL = "https://huggingface.co/datasets/Seiitsu/anime-compass-data/resolve/main/anime_catalog_serving.json"
```

If either file cannot be fetched or fails verification, the demo shows a clear model-unavailable state. It never
falls back to a weaker model while presenting itself as the benchmarked one.

The compact catalog is a **deployment** artifact, not a model change: it preserves every MAL ID and so leaves the
pinned catalog identity digest unchanged, and the equivalence is enforced by tests rather than asserted. The
FastAPI service and the offline evaluation harness keep using the full catalog, which the constraint-rich Hybrid
needs for entity joins.

## Project Structure

```text
streamlit_app.py                  portfolio demo entry point
backend/anime_agent/
  als_serving.py                  production ALS index (NumPy only)
  fast_path.py                    default recommendation path
  retrieval.py  routing.py        candidate retrieval and source selection
  showcase.py                     headless service behind the demo
  artifact_bootstrap.py           fetch and verify the deployment payload
  recommender.py                  multi-channel hybrid (constraint-rich path)
  evaluation/                     offline benchmark harness
app/                              FastAPI service
scripts/                          build, evaluate, verify, migrate, compact
docs/                             architecture, evaluation, portfolio summary
data/evaluation/personalized/     experiment reports and decision records
tests/                            395 tests
```

## Data And License

Derived from [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020),
released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). See
[DATASET_ATTRIBUTION.md](DATASET_ATTRIBUTION.md). Source code is [MIT](LICENSE).

This is a portfolio project. There are no real users and no production traffic behind these numbers.
