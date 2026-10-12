/**
 * Developer-only recommendation diagnostics.
 *
 * Answers one question while developing: after a Like, did the profile, the
 * ALS user vector, and the ranking actually change? Nothing here runs in a
 * production build. `createDiagnosticsLog(false)` records nothing and installs
 * no global, and the caller passes `import.meta.env.DEV`, which Vite compiles
 * to `false` for production.
 *
 * The user-vector digest and the pre-rerank candidate IDs come from the
 * backend's `diagnostics.debug` block, which exists only when the server runs
 * with RECOMMENDATION_DEBUG_DIAGNOSTICS=true. Without it the comparison still
 * reports the profile size and the ranking change, from the response itself.
 */
const HISTORY_LIMIT = 50;

export function summarizeRecommendation(payload) {
  const diagnostics = payload?.diagnostics || {};
  const debug = diagnostics.debug || {};
  return {
    path: diagnostics.recommendation_path || diagnostics.path || null,
    route: diagnostics.collaborative_route || null,
    knownPositives: diagnostics.known_positive_count ?? null,
    candidatePool: diagnostics.candidate_pool_size ?? null,
    excluded: diagnostics.excluded_count ?? null,
    rerankerApplied: diagnostics.learned_reranker_applied ?? null,
    profile: diagnostics.profile || null,
    userVector: debug.als_user_vector_sha256_16 || null,
    userVectorNorm: debug.als_user_vector_norm ?? null,
    alsTopIds: Array.isArray(debug.als_retrieved_top_ids) ? debug.als_retrieved_top_ids : null,
    rerankedTopIds: Array.isArray(debug.reranked_top_ids) ? debug.reranked_top_ids : null,
    resultIds: (Array.isArray(payload?.results) ? payload.results : []).map(item => Number(item.id)),
    accounting: payload?.accounting || null,
    timingMs: payload?.timing_ms ?? null,
  };
}

const sameList = (a, b) => a.length === b.length && a.every((value, index) => value === b[index]);

/** What moved between two responses. `null` means "the server did not say". */
export function compareRecommendations(previous, next, depth = 18) {
  if (!previous || !next) return null;
  const before = previous.resultIds.slice(0, depth);
  const after = next.resultIds.slice(0, depth);
  const beforeSet = new Set(before);
  const afterSet = new Set(after);
  const kept = after.filter(id => beforeSet.has(id)).length;
  return {
    profileSizeChanged: previous.knownPositives !== next.knownPositives,
    userVectorChanged: previous.userVector && next.userVector ? previous.userVector !== next.userVector : null,
    alsCandidatesChanged: previous.alsTopIds && next.alsTopIds ? !sameList(previous.alsTopIds, next.alsTopIds) : null,
    rankingChanged: !sameList(before, after),
    overlap: after.length ? kept / after.length : null,
    entered: after.filter(id => !beforeSet.has(id)),
    left: before.filter(id => !afterSet.has(id)),
  };
}

export function createDiagnosticsLog(enabled, { sink = globalThis.console, target = globalThis } = {}) {
  if (!enabled) return { enabled: false, record() { return null; }, entries: [] };
  const entries = [];
  let lastReset = null;
  // Inspectable from the browser console during development.
  if (target) target.__animeCompassDiagnostics = entries;
  return {
    enabled: true,
    entries,
    record(event, payload, extra = {}) {
      const summary = summarizeRecommendation(payload);
      // Only whole-profile recomputes are comparable; a refill answers a
      // different question (what comes after the titles already shown).
      const change = event === 'profile' ? compareRecommendations(lastReset, summary) : null;
      if (event === 'profile') lastReset = summary;
      const entry = { event, at: new Date().toISOString(), ...extra, summary, change };
      entries.push(entry);
      if (entries.length > HISTORY_LIMIT) entries.shift();
      sink?.debug?.('[anime-compass] recommend', entry);
      return entry;
    },
  };
}
