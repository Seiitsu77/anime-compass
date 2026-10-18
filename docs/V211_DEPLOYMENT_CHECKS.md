# Anime Compass V2.1.1 deployment checks

Verified and promoted on 2026-10-18. Production is 100% on
`anime-compass-franchise-753c4f3`. `anime-compass-v21-94d6c10`, the previous
production revision, is retained for rollback:

    gcloud run services update-traffic anime-compass --region=us-central1 --to-revisions=anime-compass-v21-94d6c10=100

## Release

- Runtime source commit: `753c4f3`, branch `fix/franchise-constraints`, tagged `v2.1.1`.
- Clean release checkout: `.tools/franchise-fix` (no untracked files, no local data).
- Cloud Build: [`94a64a05-7f0d-4cdf-bd90-8216e7fbcc6c`](https://console.cloud.google.com/cloud-build/builds/94a64a05-7f0d-4cdf-bd90-8216e7fbcc6c?project=937277373885), build and in-build container check both passed.
- Image tag: `us-central1-docker.pkg.dev/project-4ea00490-f490-4348-86b/anime-compass/anime-compass:franchise-753c4f3`.
- Immutable digest: `sha256:d3176d926b8cde3f6d9343b47f5d1ecc821dd415471ade011667ded4d24ba1f0`.
- Cloud Run revision: `anime-compass-franchise-753c4f3`, tag `franchise-753c4f3`, also
  reachable at `https://franchise-753c4f3---anime-compass-b2c6mriqrq-uc.a.run.app`.

The revision was deployed at 0% traffic with the previous revision's settings
and described before testing. Every field matched production except the image:
1 CPU, 4 GiB, concurrency 4, timeout 300 s, max one instance, CPU throttling
and startup boost on, the same service account, `LLM_PROVIDER` and
`HF_DATASET_REPO`, `GEMINI_API_KEY` from secret version 2, and the startup probe
on `/api/ready`.

### From the serving image back to its source

Checked link by link on 2026-10-18:

1. Cloud Run serves `anime-compass-franchise-753c4f3`, whose image is the digest above.
2. That digest is the recorded output of Cloud Build `94a64a05-...`, and the
   registry resolves the `franchise-753c4f3` tag to the same digest.
3. The build's source archive has SHA-256
   `60c94009dd0e2371ed3755ab3a979d7f5a7b545d6b412a15c9ea19f373d6b507`, as
   recorded in the build's provenance. The archive was downloaded and hashes to
   exactly that.
4. Its 146 files are all tracked at `753c4f3` and match it: 10 byte for byte,
   136 differing only in CRLF line endings from the Windows checkout. Nothing
   else is in it. The 334 tracked files it leaves out are the ones
   `.gcloudignore` excludes: tests, docs, experiments, evaluation data, CI
   configuration and `.env.example` files.

## The failure

"Hey man, recommend me some Gundam movies. I want the year after 2020." returned
movies from 2021 onward that were not Gundam.

Gemini parsed the request as `intent=recommend, catalog_query="Gundam",
formats=[Movie], min_year=2021`. `recommend_anime` ignores `catalog_query`, so
the recommender received only the format and the year, plus the whole message
as a soft similarity query, and nothing downstream knew the subject was fixed.
The answer was built from well-ranked 2021 movies. When Gemini was unavailable,
the rule fallback also failed to read "movies" as a format.

The fix makes a named franchise a hard constraint (see
[SYSTEM_REFERENCE.md](SYSTEM_REFERENCE.md#franchise-constraints)): membership is
read from catalog titles and aliases, Gemini's proposal is verified against the
message and the catalog, the family filter runs before every ranking channel,
the final eligibility gate re-checks every row after replanning, and franchise
requests are neither relaxed nor padded.

### Constraint trace on the release, exact request

Intent, arguments and candidate counts from the candidate URL; the answer from
production after promotion, where every returned ID was checked again:

| Stage | Value |
|---|---|
| Gemini intent | `catalog_query="Gundam"`, `formats=[Movie]`, `min_year=2021`, no `required_franchises` |
| Backend franchise resolution | `required_title_families=[gundam]`, from the message and from `catalog_query` |
| `recommend_anime` arguments | `required_title_families=[gundam]`, `formats=[Movie]`, `min_year=2021` |
| Candidates | 138 Gundam catalog entries, 8 after the Movie and year filters |
| Replanning | none |
| Final eligibility gate | nothing to drop |
| Answer, production | Gemini, grounded: the 8 Gundam movies from 2021 to 2025 |

## The audit

The same pipeline was audited for every kind of hard constraint against the
previous production revision. Fixed in this release:

| Constraint | Before | Now |
|---|---|---|
| Franchise ("Gundam movies") | not modelled | hard, before ranking and at the final gate |
| Franchise exclusion ("but not Gundam") | not modelled | hard, before ranking and at the final gate |
| Follow-up "what about TV series instead?" | franchise lost on the rule fallback | kept |
| "Detective Conan movies" | "Detective" became a genre filter: 0 of 79 entries | the 36 movies are eligible |
| "Fate movies" | similarity results including non-Fate titles | strict `Fate/` membership |
| "Kana Hanazawa" | also required the voice actor "Kana": 0 results | her credits only |
| "directed by Makoto Shinkai" | also required his one voice credit: 1 of 9 titles | his directed works |
| Year windows, single years, "films", "OVAs" | dropped by the rule fallback | read |
| Empty result | "Try adding a genre, title, or format" | names the constraints that matched nothing |

Studio, character, episode ceiling and title exclusion were already correct.

## Checks

| Check | Result |
|---|---|
| Backend suite on the release commit | 811 passed, 15 skipped (63 new in `tests/test_constraint_regressions.py`) |
| Ruff, ruff format, mypy | Passed |
| GitHub CI on `753c4f3` | All five jobs passed |
| Cloud Build, including the credential-free container check | Passed |
| `scripts/check_production_deployment.py`, candidate and production | Passed, 5 of 5, both |
| `scripts/smoke_e2e.py --with-agent`, candidate | Passed, 7 of 7 |
| `scripts/evaluate_agent_constraints.py`, rule fallback only, in-process | 28 of 28 turns |
| Same, live Gemini, in-process | 28 of 28 turns; no ineligible ID in any of 40 attempts |
| Same, live Gemini, candidate URL | 28 of 28 turns; no ineligible ID in any of 42 attempts (265 rows) |

The suite checks every returned ID against the catalog with an oracle written
independently of the code under test, and checks every attempt, including those
the rule fallback answered (see [EVALUATION.md](EVALUATION.md#agent-constraint-suite)).
On the candidate URL Gemini parsed 27 of the 28 final answers; the other turn
fell back on all three attempts and its fallback answer met every constraint.
Of the 15 fallback attempts, 10 were Gemini HTTP 503, 3 an open circuit breaker
after those failures and 2 timeouts.

The first live in-process run is not counted: the runner compared
`parser_mode` with `gemini`, which the API reports as `structured_llm`, so it
retried every turn and recorded only the last attempt. Commit `753c4f3` reads
the provider from the debug trace instead.

## Known and unchanged

- Gemini returned HTTP 503 or timed out on 12 of 42 attempts during the candidate run.
  Those requests are served by the rule fallback, which meets the same constraints.
- "Anime similar to Gundam" mostly returns other Gundam titles. That is allowed
  for a similarity request but is a weak answer.
- "Anime with Lelouch Lamperouge" returns a single title through Gemini.
- No ALS or LambdaMART artifact changed in this release.
