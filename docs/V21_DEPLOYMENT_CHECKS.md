# Anime Compass V2.1 deployment checks

Superseded on 2026-10-18 by V2.1.1, which fixes franchise constraints in agent
answers: see [V211_DEPLOYMENT_CHECKS.md](V211_DEPLOYMENT_CHECKS.md). The revision
below is now the rollback target.

Verified on 2026-10-12. Status: the corrected candidate below passed every check
at 0% of traffic and was then promoted. Production is 100% on
`anime-compass-v21-94d6c10`; `anime-compass-00003-kfx` is retained for rollback:

    gcloud run services update-traffic anime-compass --region=us-central1 --to-revisions=anime-compass-00003-kfx=100

After promotion the deployment checker passed against the production URL and 22
of 22 live Gemini turns there behaved correctly.

## Candidate

- Runtime source commit: `94d6c10`, on local branch `codex/v21-production`.
- Clean release checkout: `.tools/v21-production-check`.
- Cloud Build: [successful candidate build](https://console.cloud.google.com/cloud-build/builds/51af7fa0-f314-4a95-b28b-49fb4464f22f?project=937277373885).
- Image tag: `us-central1-docker.pkg.dev/project-4ea00490-f490-4348-86b/anime-compass/anime-compass:v21-check-94d6c10`.
- Tested immutable digest: `sha256:daab1cb7a5d758a26263576f97a054efddee262a9dde8ffbde85be9ae7995266`.
- Cloud Run revision: `anime-compass-v21-94d6c10`, tag `v21-94d6c10`, also
  reachable at `https://v21-94d6c10---anime-compass-b2c6mriqrq-uc.a.run.app`.

Deploy the tested digest when approving the release. The main working folder
contains unrelated changes; do not use `-AllowDirty` to include them accidentally.
The isolated release checkout excludes those changes.

### From the serving image back to its source

The chain was checked link by link on 2026-10-12, not inferred from the tag:

1. Cloud Run serves `anime-compass-v21-94d6c10`, whose image is the digest above.
2. That digest is the recorded output of Cloud Build `51af7fa0-f314-4a95-b28b-49fb4464f22f`.
3. The build's source archive, as recorded in its provenance, has SHA-256
   `5e0602bd402bf9ba945d6731cde653a37008f627cb45263b466a9c0709baa222`.
   The archive was downloaded and hashes to exactly that.
4. All 149 files in the archive are identical to commit `94d6c10`, and it holds
   nothing else. Tracked files it leaves out are the ones `.gcloudignore`
   excludes: tests, docs, experiments and evaluation data.

Commit `94d6c10` is tagged `v2.1.0`. Later commits on `main` change documentation
and repository layout and are not what the serving image was built from.

Do not deploy the first candidate, `b7ee94e` /
`sha256:10b02aac3af16c4d799414932833f5c7e7a477bd79d08cda61b283ba2cb5c917`
(revision `anime-compass-v21-b7ee94e`). It carries the agent failures described
next, as does the revision serving production today.

## Agent correction

The first candidate's controlled rollout found that "Introduce Cowboy Bebop
without spoilers." was answered with "I updated this session's preferences".
Gemini had classified the request correctly; the backend's own normalization
read the substring "without" as negative feedback and rerouted it to a session
write, and the only tool-contract check ran after that write. With Gemini the
write carried an empty patch. With Gemini unavailable the same message stored
Cowboy Bebop in `excluded_titles`.

Commit `6c1cba8` fixes the routing and adds a backend write gate: a session
preference is written only when the user's own message states that kind of
preference, decided before any tool runs. Live testing against Gemini found and
fixed three further defects in the same path: HTTP 500 for "Tell me more about
the second one" with no previous list; every "anime similar to X" intent being
rejected because Gemini words the relation as `similar_to`; and a positional
follow-up describing the wrong title on the rule path. Commit `94d6c10` caches
the catalog title scan so the gate adds no latency.

| Check | Result |
|---|---|
| Complete backend suite on isolated release | 713 passed, 13 skipped (154 new) |
| Type checking, lint and formatting | Passed |
| Frontend tests and Vite production build | Passed; 77 frontend tests |
| Linux image build and credential-free container checks in Cloud Build | Passed |
| `scripts/check_production_deployment.py` against the candidate URL | Passed |
| Live Gemini requests against the candidate URL, fresh sessions | 64 of 64 turns correct |

The live run repeated each information request three to five times and checked
intent, tool trace, answer and the stored session before and after every turn.
No information, search or recommendation request changed a stored preference;
"I like", "I've already watched" and "Don't recommend" each stored exactly the
field they state. Gemini returned HTTP 503 on 19 of the 64 turns; those were
served by the deterministic fallback and behaved identically.

Known and unchanged: a correct Gemini answer that does not repeat the title
verbatim is rejected by the grounding check and the catalog answer is served
instead (seen on every "who are the main characters" request), and a chat
recommendation takes about 40 seconds on one vCPU.

## Verification of the build preparation

These results are from the first candidate, `b7ee94e`. The frontend source and
build configuration have not changed since. The Chromium run was not repeated
for the corrected candidate; the other rows were, with the results above.

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

This section describes the deployed image, built from `v2.1.0`. On `main` the
React source has since moved from `frontend-react/` to `frontend/` and the
legacy vanilla frontend has been removed; the mechanism is unchanged.

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
