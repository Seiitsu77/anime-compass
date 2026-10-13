# Publishing Anime Compass

This covers the technical demo and the artifacts. The web application itself
runs on Google Cloud Run and is documented in
[CLOUD_RUN_DEPLOYMENT.md](CLOUD_RUN_DEPLOYMENT.md).

The demo's deployment uses GitHub for source, a public Hugging Face Dataset
repository for the serving artifacts, and Streamlit Community Cloud for the
public page. Keeping the artifacts out of Git keeps the repository reviewable;
they are checksum-verified after download, so a wrong or partial file is never
served.

## 1. GitHub repository

Use the public repository `anime-compass` for source, tests, evaluation reports,
and documentation. Keep the MIT license for source code; the dataset-derived
artifacts retain the CC0 attribution in
[`DATASET_ATTRIBUTION.md`](../DATASET_ATTRIBUTION.md).

Before pushing, verify that credentials and raw data are absent:

```powershell
git status
git ls-files .env archive data/raw
git grep -l -I -E "BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|AIza|AQ\."
```

The last two commands should not list a credential or a raw CSV. The scan
matches this file itself, because the pattern is written here; any other hit is
worth investigating. `git status` should show no file under `data/processed`
other than `.gitkeep`.

## 2. Hugging Face Dataset repository

Create a public Dataset repo named `anime-compass-data`, licensed **CC0-1.0**
because the processed files derive from the CC0 source dataset. Upload the two
files the demo needs:

```powershell
python -m pip install -U huggingface_hub
huggingface-cli login
huggingface-cli upload Seiitsu/anime-compass-data data/processed/als_production_item_factors.npz als_production_item_factors.npz --repo-type=dataset
huggingface-cli upload Seiitsu/anime-compass-data data/processed/anime_catalog_serving.json anime_catalog_serving.json --repo-type=dataset
```

Their sizes and SHA-256 values are pinned in
[`data/artifacts.manifest.json`](../data/artifacts.manifest.json), so the demo
rejects a file that does not match. Confirm both resolve URLs return 200 before
deploying:

```text
https://huggingface.co/datasets/Seiitsu/anime-compass-data/resolve/main/als_production_item_factors.npz
https://huggingface.co/datasets/Seiitsu/anime-compass-data/resolve/main/anime_catalog_serving.json
```

## 3. Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io) with the
   GitHub account that owns the repository.
2. Create an app from the `main` branch.
3. Set the entry point to `streamlit_app.py` and choose Python 3.12.
4. Add these two secrets, then deploy. `requirements.txt` supplies the complete
   two-package direct runtime, and the checksums come from the manifest, so no
   other secret is needed.

   ```toml
   ALS_ARTIFACT_URL = "https://huggingface.co/datasets/Seiitsu/anime-compass-data/resolve/main/als_production_item_factors.npz"
   SERVING_CATALOG_URL = "https://huggingface.co/datasets/Seiitsu/anime-compass-data/resolve/main/anime_catalog_serving.json"
   ```
5. Load an example profile, request recommendations, and open **Deployment
   health**. It should report `Production ALS`, a verified artifact, 18,064
   catalog items, and `Fast path ready: yes`.

After deployment, add the public URL to the README and the GitHub repository's
About panel. A short screenshot or GIF is useful, but only after it reflects the
actual deployed page.

### What the demo downloads

The demo imports the recommendation core directly rather than calling FastAPI
over HTTP: both would run in the same process, so the extra hop would add a
failure mode and nothing else. Its payload is four files, about 31 MB:

| File | Size | SHA-256 |
|---|---:|---|
| `als_production_item_factors.npz` | 7.4 MB | `95c079b1b8f4e0e509c8bab29e4357360f851e3adfd2abc261f358375ee13a10` |
| `anime_catalog_serving.json` | 6.6 MB | `4b11cf3e5f3d015ed071232e569068248af709e7979e561194b451e3ac716bf3` |
| `reranker_features.npz` | 17.4 MB | `83cb272bf3ba6e3b5d5185313ab73d952c783c443f8a2f65b4c3c02d2f8e0b82` |
| `reranker_lambdamart.txt` | 0.6 MB | `78787f8a4c1a79c3fa3c28a2bfe3c52c474b73ea0b6d3a1b4f440930f5127dec` |

The checksums are pinned in `data/artifacts.manifest.json`, and the model embeds
the digest of the catalog IDs it was trained against, so a changed catalog is
rejected even when most IDs still overlap. Without the reranker files the demo
still works and reports `Reranker: degraded to ALS`. If the ALS artifact or the
catalog cannot be fetched or fails verification, the demo shows a clear
model-unavailable state rather than falling back to a weaker model while
presenting itself as the benchmarked one.

The compact catalog holds only the ten fields the demo reads, 6.6 MB instead of
119 MB for identical recommendations. It is a deployment artifact, not a model
change: it preserves every MAL ID, and the equivalence is enforced by tests.

## 4. The rest of the artifacts

The same Dataset repo can carry the artifacts the full FastAPI/agent application
needs — the 119 MB catalog and the optional evaluation files. Never upload `.env`
or an API key. The existing downloader validates every file against the manifest:

```powershell
python scripts/download_artifacts.py --repo-id Seiitsu/anime-compass-data
```

A failed download or checksum leaves the model unavailable rather than silently
serving a different recommender.

## 5. Portfolio checklist

- Pin the GitHub repository and add topics such as `recommendation-system`,
  `collaborative-filtering`, `streamlit`, `fastapi`, `information-retrieval`,
  and `machine-learning`.
- Link the Streamlit app from GitHub and the resume.
- Keep claims tied to committed reports: offline metrics are not evidence of
  real-user satisfaction or state-of-the-art performance.
- Be ready to explain the data split, full-catalog protocol, model-role
  separation, artifact verification, negative experiments, and the lack of
  interaction timestamps.
