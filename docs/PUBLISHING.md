# Publishing Anime Compass

The portfolio deployment uses GitHub for source, a public Hugging Face Dataset
repository for the two serving artifacts, and Streamlit Community Cloud for the
public demo. Keeping the artifacts out of Git keeps the repository reviewable;
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
huggingface-cli upload YOUR_HF_USERNAME/anime-compass-data data/processed/als_production_item_factors.npz als_production_item_factors.npz --repo-type=dataset
huggingface-cli upload YOUR_HF_USERNAME/anime-compass-data data/processed/anime_catalog_serving.json anime_catalog_serving.json --repo-type=dataset
```

Their sizes and SHA-256 values are pinned in
[`data/artifacts.manifest.json`](../data/artifacts.manifest.json), so the demo
rejects a file that does not match. Confirm both resolve URLs return 200 before
deploying:

```text
https://huggingface.co/datasets/YOUR_HF_USERNAME/anime-compass-data/resolve/main/als_production_item_factors.npz
https://huggingface.co/datasets/YOUR_HF_USERNAME/anime-compass-data/resolve/main/anime_catalog_serving.json
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
   ALS_ARTIFACT_URL = "https://huggingface.co/datasets/YOUR_HF_USERNAME/anime-compass-data/resolve/main/als_production_item_factors.npz"
   SERVING_CATALOG_URL = "https://huggingface.co/datasets/YOUR_HF_USERNAME/anime-compass-data/resolve/main/anime_catalog_serving.json"
   ```
5. Load an example profile, request recommendations, and open **Deployment
   health**. It should report `Production ALS`, a verified artifact, 18,064
   catalog items, and `Fast path ready: yes`.

After deployment, add the public URL to the README and the GitHub repository's
About panel. A short screenshot or GIF is useful, but only after it reflects the
actual deployed page.

## 4. The rest of the artifacts

The same Dataset repo can carry the artifacts the full FastAPI/agent application
needs — the 119 MB catalog and the optional evaluation files. Never upload `.env`
or an API key. The existing downloader validates every file against the manifest:

```powershell
python scripts/download_artifacts.py --repo-id YOUR_HF_USERNAME/anime-compass-data
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
