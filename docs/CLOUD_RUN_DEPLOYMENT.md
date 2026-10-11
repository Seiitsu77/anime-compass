# Deploying to Google Cloud Run

The production image is `Dockerfile.cloudrun`, built by Cloud Build and deployed
by `scripts/deploy_cloudrun.ps1`. It uses the pinned CPU container dependencies, so the
dependency stack is the one already proven: CPU-only `torch==2.14.0+cpu`,
`sentence-transformers==5.7.0`, `transformers==5.16.1`, and `libgomp1` for
LightGBM.

## What gets deployed

| Setting | Value |
|---|---|
| Region | `us-central1` |
| Service | `anime-compass` |
| Image | `us-central1-docker.pkg.dev/PROJECT/anime-compass/anime-compass:<short SHA>`, deployed by digest |
| CPU / memory | 1 vCPU / 4 GiB |
| Instances | min 0, max 1 |
| Concurrency | 4 requests per instance (Cloud Run's default is 80) |
| Startup CPU boost | on |
| Container port | 8080 |
| Startup probe | `GET /api/ready`, initial delay 2s, timeout 2s, period 2s, failure threshold 60 (122s budget) |
| Environment | `LLM_PROVIDER=gemini`, `HF_DATASET_REPO=Seiitsu/anime-compass-data` |
| Secret | `GEMINI_API_KEY` from Secret Manager `gemini-api-key`, pinned version (default `1`; see [Rotating the Gemini key](#rotating-the-gemini-key)) |
| Runtime identity | `anime-compass-run@PROJECT.iam.gserviceaccount.com` |
| Access | public, unauthenticated |

`/api/ready` answers **503 while the recommender initializes** and 200 only once
it can serve. That is what the startup probe waits for, so Cloud Run never routes
traffic to a half-initialized instance. `/api/health` answers as soon as the HTTP
server is up.

## Prerequisites

- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed and on `PATH`.
- A project with billing enabled.
- Project Owner is the simplest. Otherwise you need permission to enable APIs,
  create service accounts and act as them, create Artifact Registry
  repositories, manage Secret Manager secrets and their IAM, submit Cloud Build
  builds (including writing to their staging bucket), and deploy Cloud Run
  services.

## One-time setup

Run these once in Windows PowerShell, from any directory.

```powershell
# 1. Authenticate and select the project
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Enable the required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# 3. Artifact Registry repository for the images
gcloud artifacts repositories create anime-compass --repository-format=docker --location=us-central1 --description="Anime Compass container images"

# 4. Dedicated runtime service account
gcloud iam service-accounts create anime-compass-run --display-name="Anime Compass Cloud Run runtime"
```

Create the secret. The key is read without echoing and written with **no
trailing newline**: the application sends it verbatim as a request header, so a
stray newline would break every Gemini call.

```powershell
# 5. Secret Manager secret holding the Gemini API key
$secure = Read-Host -Prompt 'Gemini API key' -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$file = [IO.Path]::GetTempFileName()
try {
    [IO.File]::WriteAllText($file, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim())
    gcloud secrets create gemini-api-key --replication-policy=automatic "--data-file=$file"
}
finally {
    Remove-Item -LiteralPath $file -Force
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
}
```

This creates version `1` of the secret, which is the version the deploy script
pins by default.

Grant the runtime account access to **that one secret only** -- not project-wide:

```powershell
# 6. Least-privilege secret access
$project = gcloud config get-value project
gcloud secrets add-iam-policy-binding gemini-api-key --member="serviceAccount:anime-compass-run@$project.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

Never put the key in source control, in `--set-env-vars`, or in a committed file.

## Deploy

From the repository root, with changes committed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1
```

The script checks the project, the repository, the service account, that the
pinned secret version exists and is enabled, and that your gcloud supports
`--startup-probe` -- all **before** building, so a setup gap fails in seconds. It
refuses to deploy uncommitted changes, because the image tag is the commit SHA;
`-AllowDirty` overrides that and tags the image `<sha>-dirty`. It stops at the
first failed command and ends by printing the service, `/api/health` and
`/api/ready` URLs.

## Rotating the Gemini key

The service pins a specific secret version rather than `latest`. Cloud Run
resolves a secret exposed as an environment variable when each instance starts,
so `latest` would let a rotated key reach new instances while running ones keep
the old value. A pinned version gives every instance of a revision the same key
and makes rotation an explicit deploy.

1. Add the new key as a new version, newline-free as before:

   ```powershell
   $secure = Read-Host -Prompt 'New Gemini API key' -AsSecureString
   $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
   $file = [IO.Path]::GetTempFileName()
   try {
       [IO.File]::WriteAllText($file, [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim())
       gcloud secrets versions add gemini-api-key "--data-file=$file"
   }
   finally {
       Remove-Item -LiteralPath $file -Force
       [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
   }
   gcloud secrets versions list gemini-api-key
   ```

   `versions add` prints the new version number; `versions list` shows every
   version and its state.

2. Set the `$GeminiSecretVersion` default in the `param` block of
   `scripts/deploy_cloudrun.ps1` to that number, commit, and deploy. Committing
   records which key version production runs alongside the code that deploys
   it. For a one-off deploy without editing the script:

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\deploy_cloudrun.ps1 -GeminiSecretVersion 2
   ```

3. Verify the new revision as below.

4. Only then disable the old version -- and only once no revision you might roll
   back to still pins it, because a revision whose pinned version is disabled
   cannot start new instances.

   ```powershell
   gcloud secrets versions disable 1 --secret=gemini-api-key
   ```

## Verify

```powershell
$url = 'https://YOUR-SERVICE-URL'

Invoke-RestMethod "$url/api/ready"                         # status: ready
(Invoke-RestMethod "$url/api/health").components.reranker  # LambdaMART active: 157 trees, 18 features
(Invoke-RestMethod "$url/api/health").agent.provider       # gemini

$fast = Invoke-RestMethod -Method Post "$url/api/recommend" -ContentType 'application/json' -Body '{"liked_ids":[5114,9253,11061],"top_k":12}'
$fast.diagnostics.recommendation_path                      # fast
$fast.diagnostics.learned_reranker_applied                 # True

python scripts/smoke_e2e.py --base-url $url --with-agent   # waits for readiness, then 7 cases
```

## If the build fails with PERMISSION_DENIED

Newer projects may run builds as the Compute Engine default service account,
which can lack permission to read the uploaded source or push the image. Grant
it the Cloud Build builder role, then re-run the script:

```powershell
$project = gcloud config get-value project
$number = (gcloud projects describe $project --format=json | ConvertFrom-Json).projectNumber
gcloud projects add-iam-policy-binding $project --member="serviceAccount:$number-compute@developer.gserviceaccount.com" --role="roles/cloudbuild.builds.builder"
```

## Rolling back

```powershell
gcloud run revisions list --service=anime-compass --region=us-central1
gcloud run services update-traffic anime-compass --region=us-central1 --to-revisions=REVISION_NAME=100
```

Each revision keeps the secret version it was deployed with, so rolling back
also rolls back the key -- which is why an old version stays enabled while a
rollback target still uses it.

## Known behaviour worth knowing

- **Cold starts.** With `min-instances=0`, the first request after the service
  scales to zero waits while the catalog and models load, and `/api/ready`
  answers 503 `service_warming` until then.
  One observed load on 2026-10-11 took 107.94s to receive the page; a subsequent
  page request took 0.09s. The instance reported 93.98s of initialization,
  including 51.29s for semantic embeddings and 18.23s for the recommender.
  The CSS response took 0.11s; the frontend uses plain CSS, with no Tailwind
  runtime. These are individual diagnostic observations, not latency percentiles.
  Keeping a minimum instance warm can reduce this wait but introduces idle
  billing; see [Cloud Run minimum instances](https://docs.cloud.google.com/run/docs/configuring/min-instances).
- **Concurrency is 4.** An instance handles up to four requests at once, and with
  `max-instances=1` that is four across the whole service; further requests wait
  for a free slot and can be rejected if none frees up in time. The value is
  deliberately conservative for 1 vCPU and a memory-resident ML stack. Tune it
  later from Cloud Run's metrics -- request latency, container CPU and memory
  utilization, and instance count -- raising it only while CPU and memory keep
  headroom and latency stays acceptable.
- **`/tmp` is memory-backed.** The SQLite session store lives there and counts
  against the 4 GiB limit. Sessions do not survive an instance being replaced.
- **Rate limiting sees the proxy.** The per-client limiter keys on the TCP peer
  address. Behind Cloud Run's front end that may be shared by all visitors, so
  the 60 requests/minute limit can act as one global limit. Trusting
  `X-Forwarded-For` wholesale would let clients spoof their way past it, so this
  has deliberately not been changed.
- **Gemini failure.** Gemini is the only LLM provider. If it fails, the agent
  falls back to the deterministic catalog parser. Personalized recommendations
  still use ALS and its available reranker without requiring text generation.
