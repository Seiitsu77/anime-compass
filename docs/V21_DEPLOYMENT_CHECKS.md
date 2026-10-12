# Anime Compass V2.1 deployment checks

Verified on 2026-10-12. Status: ready for deployment; no blocking failures found.
No Cloud Run revision or production traffic was changed.

## Candidate

- Runtime source commit: `b7ee94e`, on local branch `codex/v21-production`.
- Clean release checkout: `.tools/v21-production-check`.
- Cloud Build: [successful candidate build](https://console.cloud.google.com/cloud-build/builds/f0debdbb-d715-42f0-8773-3518b73c04b8?project=937277373885).
- Image tag: `us-central1-docker.pkg.dev/project-4ea00490-f490-4348-86b/anime-compass/anime-compass:v21-check-b7ee94e`.
- Tested immutable digest: `sha256:10b02aac3af16c4d799414932833f5c7e7a477bd79d08cda61b283ba2cb5c917`.

Deploy the tested digest when approving the release. The main working folder
contains unrelated changes; do not use `-AllowDirty` to include them accidentally.
The isolated release checkout excludes those changes.

## Verification

| Check | Result |
|---|---|
| Locked frontend installation, Node tests and Vite production build | Passed; 77 frontend tests |
| Complete backend suite on isolated release | 559 passed, 13 skipped |
| Production static integration and API checks alongside the working tree | 37 passed |
| Type checking, lint, formatting and deployment-script syntax | Passed |
| Chromium using compiled React assets and the same FastAPI source | Passed |
| Linux image build and credential-free container checks in Cloud Build | Passed |

Chromium used a fresh browser profile and synthetic test data. It verified 54
unique recommendations across three batches, Replace, Already Watched, hash
navigation, reload, mobile layout, and legacy localStorage preservation. No
runtime errors were observed; development diagnostics were absent.

The Cloud Build check used the actual catalog and frozen production artifacts,
with no network or Gemini credentials. It verified ALS, LambdaMART, semantic
embeddings, readiness, the compiled asset URLs, non-overlapping cursor pages,
cursor replay, exclusions, stale/malformed cursor rejection, and the existing
recommendation, search and preferences API contracts. Backend tests also cover
cache loss/restart recovery and pagination exhaustion.

Existing API schemas, recommendation sequence code, ranking implementation and
the localStorage key/migration code were not changed by this preparation.
Third-party Gemini generation was mocked in local tests; no paid generation
request was made.

## Production integration

The Cloud Run image now builds React in a Node 22 stage and copies its output
to `frontend-react/dist`. FastAPI selects that directory through `FRONTEND_DIR`.
API routes are registered before the static mount. HTML is revalidated and
content-hashed assets have immutable caching; missing API/assets remain 404.
CI builds React before testing FastAPI static integration, and Cloud Build
refuses to publish an image if its production container check fails.

## Configuration preserved

- Service: `anime-compass`, region `us-central1`.
- Existing service URLs and hash navigation remain unchanged, preserving the
  browser origin and `localStorage.animeCompassSession`.
- Runtime: 1 vCPU, 4 GiB, concurrency 4, port 8080, startup CPU boost enabled.
- Revision scaling: minimum 0, maximum 1; existing service-level ceiling remains unchanged.
- Startup probe: `/api/ready`, 2s initial delay, 2s timeout, 2s period, 60 failures.
- Existing runtime service account, public access and Secret Manager remain unchanged.
- Live secret binding is `GEMINI_API_KEY=gemini-api-key:2`; version 2 is enabled.
  The helper's old default of version 1 was corrected to 2. Environment and
  secret updates preserve other existing entries. No secret values were read.
- Live revision remains `anime-compass-00003-kfx`.

Cold starts and ephemeral SQLite storage remain characteristics of the current
configuration. Browser storage remains the persistent source for user feedback.
No unrelated uncommitted files were edited or included in the candidate.
