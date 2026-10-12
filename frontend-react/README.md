# Anime Compass V2 — React + Vite prototype

A separate, **not deployed** white-and-lavender frontend prototype, built after reviewing the existing `Seiitsu77/anime-compass` GitHub source. The existing FastAPI API, Gemini Agent and ALS/LambdaMART ranking are reused. No model artifact is changed by this frontend; the V2.1 work did change request handling in the API, listed under [V2.1](#v21-feedback-semantics-and-correctness).

## Run locally (PowerShell)

Requirements: Node.js 20.19+ or 22.12+ and npm.

```powershell
cd "C:\Study\Toy project"
# Extract the downloaded archive as frontend-react/ (or copy this folder there).
cd frontend-react
npm install
npm run dev
```

Then open `http://127.0.0.1:5173`. Vite forwards `/api` to the existing Cloud Run service. The first remote response may be slow after Cloud Run scales to zero. For a local FastAPI server, create `.env` using `.env.example` and change `API_PROXY_TARGET` to `http://127.0.0.1:8000`; restart Vite. Never place GEMINI_API_KEY or other server secrets in Vite environment variables.

```powershell
npm test
npm run build
```

The result of `npm run build` is in `dist/`. Do not deploy this prototype without integration testing.

## Pages

- **Home**: clean hero, restrained search, three primary filters, real catalog popularity sorting (not a fabricated live trending chart), recommendations, Agent CTA.
- **Discover**: full `/api/search`, filtering, paging, details and Like button.
- **For You**: My Collection (Liked, Want to Watch, Not Interested), up to 18 personalized `/api/recommend` picks with Replace, Not Interested and Next Batch, and profile JSON backup import/export.
- **Agent**: `/api/agent` with `session_id` and previous messages passed as `history` for follow-up context.

## Existing browser Likes compatibility

The original vanilla frontend stores an object in `localStorage.animeCompassSession` with fields `sessionId`, `liked`, and `preferences`. This app keeps those existing fields under **the same key** and adds two optional ones the legacy frontend ignores: `watchlist` (Want to Watch) and `notInterested`. Legacy session data still loads, and unit tests cover compatibility in both directions.

**Important**: `localhost:5173` and the live Cloud Run URL have different browser origins and thus different localStorage. Existing likes will automatically be available only when this build is eventually served from the *same existing Cloud Run host*. During local development you can export the JSON from the old browser's DevTools or from a previous React session, then import it using **For You → Import backup**. Before ever switching a production page, export your current Likes backup.

## Data honesty / limitations

- No fabricated posters, anime records, logged-in account, watch progress, current-season charts, weekly popularity, streaming pricing, or geographic availability.
- The default catalog listing sorts by available catalog popularity (`/api/search`); it is deliberately called **Popular in the catalog**.
- No Region or Pricing control because the existing search API does not support those hard filters.
- The Agent is a separate page, not an always-open sidebar; conversation history is retained while navigating pages in the current tab, but not persisted across reloads.
- The original backend session repository uses SQLite and may be ephemeral in Cloud Run, so the browser session object remains the persistent source for Likes.
- This is an integration-ready frontend prototype. V2.1 was exercised end to end in a browser against a local backend running the real catalog, ALS artifact and LambdaMART reranker (`scripts/dev_e2e_backend.py`).

## Production adoption checklist

1. Copy the project into `frontend-react/` in the existing monorepo on a **new branch**.
2. Run `npm install`, `npm test`, `npm run build`, and test all flows in a real browser against the **same backend revision** used in production.
3. Test search, like/unlike, persisted Likes, recommendations, missing/failed API, Agent follow-up, filtering, mobile layout and keyboard controls.
4. Back up the old `frontend/` assets; run Vite build in Cloud Build/Docker or CI, then copy `frontend-react/dist/*` to the FastAPI-served `frontend/` directory. Preserve `/api` routing and SPA hash-based navigation.
5. Confirm the Cloud Run URL remains unchanged, so production `localStorage.animeCompassSession` survives.
6. After acceptance, deploy a **new revision** and retain rollback options.

## Files

- `src/App.jsx` pages, data loading and navigation
- `src/components/` card, filtering, navbar, Agent
- `src/lib/api.js` same-origin FastAPI integration
- `src/lib/session.js` legacy browser Likes schema preservation and the four feedback types
- `src/lib/recommendationFeed.js` the browsing sequence: no repeats, no stale results, Skip without dislike
- `src/lib/strings.js` every user-facing string, in English
- `src/lib/diagnostics.js` development-only recommendation diagnostics
- `src/styles.css` responsive white-and-lavender design tokens and styles

No unnecessary animation dependency is installed. Blur Text and Pill Nav now animate using a small in-project React Bits-inspired implementation, with reduced-motion support.

## Cold-start handling

The frontend polls `/api/ready` before initially loading results or recommendations. It refreshes `/api/health` once the backend is ready and uses its real metadata and Agent availability. When sending an Agent message, the frontend first synchronizes the saved browser profile to `/api/session/preferences`, then passes recent conversation history to `/api/agent`.

## V2.1: feedback semantics and correctness

The interface is English only. Every user-facing string lives in
`src/lib/strings.js`; `tests/english.test.js` fails if a component inlines copy
or any source file contains non-English script. Anime titles and other catalog
metadata are shown exactly as the catalog provides them.

### Four kinds of feedback

| Action | What it means | Stored | Sent to the recommender |
|---|---|---|---|
| **Like** | Positive preference | `liked` | `liked_ids`: the only input to the ALS profile |
| **Want to Watch** | Future intent | `watchlist` | An exclusion. Never a like, never "seen", never a rating |
| **Not Interested** | Explicit negative feedback | `notInterested`, plus the legacy `disliked_titles` and `excluded_titles` | An exclusion, and a disliked title for the server-side profile |
| **Replace** (Skip) | "Not this one right now" | Nothing | An exclusion for the current page session only |

- **Replace** used to call `rejectAnime()`, which wrote a permanent dislike.
  It now records nothing: the title is skipped until the page is reloaded and
  another eligible recommendation takes its place.
- **Not Interested** is the separate, explicit action. It can be undone from
  the notice or from My Collection → Not Interested.
- Only a change to the Liked set recomputes recommendations, because only Likes
  move the model. Want to Watch, Not Interested and Replace fill the slot of the
  card in place, so a bookmark no longer sends the user back to the first
  batch.

### Guarantees, and where they are enforced

- **A Liked title is never recommended.** Enforced by the API (profile titles
  are excluded on every path) and again by the eligibility filter in the client.
- **Next Batch never repeats.** Every title shown in a sequence is sent as an
  exclusion and filtered again on the client, so a server that ignored the
  exclusions still could not cause a repeat. When the pool is exhausted the
  page says so.
- **A stale response cannot overwrite a newer one.** Each profile change starts
  a new epoch in `recommendationFeed.js`; any response that returns in a later
  epoch than it started is dropped. A burst of Likes is debounced into one
  request.
- **Nothing is truncated silently.** `/api/health` advertises the request
  limits (`ranking.request_limits`: 500 Likes, 2,000 excluded IDs). If a profile
  exceeds a limit, the most recent Likes are sent, the rest are excluded so
  they still cannot be recommended, and the page states how many were used.
  Against a backend that predates the advertised limits the client falls back
  to 50 and 100.
- **Profile sync is ordered.** At most one `/api/session/preferences` request is
  in flight and the newest profile always wins. Only fields the API accepts are
  sent, so a restored backup with an unknown key no longer makes every sync
  fail.

### API changes that came with it

Request handling only. No model artifact, feature, or ranking code changed.

- `liked_ids` limit 50 → 500, `excluded_ids` 100 → 2,000, `excluded_titles`
  100 → 500. The old client sent the *oldest* 50 Likes, so the 51st Like and
  every later one never changed a recommendation.
- The ALS fast path now honours title-based exclusions (`excluded_titles`,
  `seen_titles`, and the stored `disliked_titles`) by exact title match. It used
  to read IDs only, so a legacy "dislike" was dropped by the Hybrid path and
  ignored on the default one.
- A profile sync no longer erases the conversation context the agent keeps
  (`last_recommendations`), which follow-ups such as "more like the second one"
  depend on.
- Unranked titles (`popularity` or `rank` of 0) sort last instead of first.

### Development diagnostics

In `npm run dev` the app logs each recommendation response to the console and
keeps the last 50 in `window.__animeCompassDiagnostics`: profile size, path,
whether the reranker ran, and what entered and left the list. Vite removes all
of it from a production build.

To see whether a Like moved the ALS user vector, run the backend with
`RECOMMENDATION_DEBUG_DIAGNOSTICS=true`. The response then carries a digest of
the folded-in user vector and the leading candidate IDs before and after
reranking. It is off by default and must stay off in production.

```powershell
.\.venv\Scripts\python.exe scripts\dev_e2e_backend.py   # real models, debug on, no external calls
# frontend-react\.env.local:  API_PROXY_TARGET=http://127.0.0.1:8011
npm run dev
```

### Unchanged from the earlier V2.1 work

- **18 at a time:** `BATCH_SIZE=18`, from a ranked reserve of up to 50. Home
  shows six preview cards; For You shows up to eighteen. Batches are the next
  best results from the recommender, in order, not a random reshuffle.
- **Compact collection:** each My Collection tab shows at most 12 titles until
  expanded with Show More.
- **Motion:** a staggered blur-in headline and a sliding navigation pill,
  written in-project and respecting `prefers-reduced-motion`.

If you roll back to the **old vanilla frontend** and change Likes there, it
rewrites `animeCompassSession` without `watchlist` and `notInterested`; export a
backup first. Negative feedback survives that round trip, because it is also
stored in the legacy title fields. This frontend creates no server-side account
and no multi-device collection.
