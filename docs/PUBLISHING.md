# Publishing Anime Compass

Use GitHub for source code and a public Hugging Face Dataset repository for the generated catalog artifacts. This keeps the Git repository reviewable while making local runs and Docker Spaces reproducible.

## 1. Create the repositories

Create these public repositories after your accounts are ready:

| Platform | Suggested name | Purpose |
|---|---|---|
| GitHub | `anime-compass` | Source, tests, CI, screenshots, and documentation |
| Hugging Face Dataset | `anime-compass-data` | Generated catalog and collaborative-model artifacts |
| Hugging Face Space | `anime-compass` | Public Docker demo |

Choose **CC0-1.0** for the Dataset repository because the processed files derive from the CC0 source dataset. Keep the source-code repository under the included MIT license.

## 2. Upload the runtime artifacts

Install the Hugging Face CLI, authenticate, and upload the two ignored files:

```powershell
python -m pip install -U huggingface_hub
hf auth login
hf upload YOUR_HF_USERNAME/anime-compass-data data/processed/anime_catalog.json anime_catalog.json --repo-type dataset
hf upload YOUR_HF_USERNAME/anime-compass-data data/processed/collaborative_embeddings.npz collaborative_embeddings.npz --repo-type dataset
```

Do not commit those generated files to GitHub. The catalog is about 114 MiB, and the collaborative index is about 14 MiB. Anime Compass validates both size and SHA-256 against `data/artifacts.manifest.json`.

Test a clean download before publishing:

```powershell
python scripts/download_artifacts.py --repo-id YOUR_HF_USERNAME/anime-compass-data
```

The command reports `verified` without downloading when the local files already match.

## 3. Publish the source to GitHub

Review the staged files, make the first commit, then connect the new empty repository:

```powershell
git status
git add .
git commit -m "Build explainable anime recommendation agent"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/anime-compass.git
git push -u origin main
```

Before the first push, confirm that `.env`, `anime_catalog.json`, `collaborative_embeddings.npz`, and the raw `archive/*.csv` files are absent from `git ls-files`. Add a repository description, topics such as `recommendation-system`, `collaborative-filtering`, `fastapi`, `llm-agent`, `information-retrieval`, and `machine-learning`, plus the deployed Space URL.

## 4. Deploy the Docker Space

Create a Docker Space named `anime-compass`, then copy the GitHub source into the Space repository. Configure these Space variables:

```text
HF_DATASET_REPO=YOUR_HF_USERNAME/anime-compass-data
HF_DATASET_REVISION=main
LLM_PROVIDER=gemini
COLLABORATIVE_ENABLED=true
EMBEDDING_PROVIDER=none
```

Add `GEMINI_API_KEY` as a **secret**, never as a variable or committed file. At container startup, the app downloads and checksum-verifies missing artifacts from the public Dataset repository. Search, filters, ranking, recommendations, and details still work through deterministic fallbacks if the LLM provider is unavailable.

After deployment, check:

- `/api/ready` returns `200`.
- `/api/health` reports the catalog, database, providers, and collaborative index.
- Search pagination plus exact studio/format and include/exclude filters work.
- The Agent returns catalog-grounded titles and degrades gracefully when the provider is unavailable.
- The Space URL is linked from the GitHub About panel and README.

## 5. Finish the portfolio presentation

Add one short demo GIF or a current screenshot, pin the GitHub repository, and use claims you can defend in an interview. The offline evaluation is intentionally labeled as a small engineering proxy; do not describe it as evidence of user satisfaction or state-of-the-art recommendation quality.
