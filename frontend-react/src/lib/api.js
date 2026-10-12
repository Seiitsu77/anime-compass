/** Same-origin API client. Vite proxies /api locally; FastAPI serves it in production. */
import { S } from './strings.js';
import { legacyHiddenTitles, legacySeenTitles } from './session.js';

export class ApiError extends Error {
  constructor(message, status, code) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

export async function api(path, { signal, ...options } = {}) {
  const response = await fetch(path, {
    ...options,
    signal,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  });
  let payload;
  try {
    payload = await response.json();
  } catch {
    // A proxy or platform error page, not an API response: the service is down or restarting.
    const describe = response.status >= 500 ? S.errors.serviceUnavailable : S.errors.unreadableResponse;
    throw new ApiError(describe(response.status), response.status);
  }
  if (!response.ok) {
    const error = payload.error;
    const message = typeof error === 'object' ? error?.message : error;
    if (response.status === 429) throw new ApiError(S.recommendations.rateLimited, 429, 'rate_limit_exceeded');
    throw new ApiError(message || S.errors.requestFailed(response.status), response.status, error?.code);
  }
  return payload;
}

export function searchCatalog(filters = {}, signal) {
  return api('/api/search', {
    method: 'POST',
    signal,
    body: JSON.stringify({
      query: filters.query || '',
      include_genres: filters.genre ? [filters.genre] : [],
      formats: filters.format ? [filters.format] : [],
      min_year: filters.year ? Number(filters.year) : null,
      max_year: filters.year ? Number(filters.year) : null,
      min_score: filters.minScore ? Number(filters.minScore) : null,
      max_episodes: filters.maxEpisodes ? Number(filters.maxEpisodes) : null,
      required_studios: filters.studio ? [filters.studio] : [],
      sort_by: filters.sort || 'popularity',
      offset: filters.offset || 0,
      top_k: filters.top_k || 24,
      semantic: Boolean(filters.query),
    }),
  });
}

/**
 * What /api/recommend accepts per request. These are the ceilings of a backend
 * that predates `ranking.request_limits` in /api/health; a newer backend
 * advertises larger ones and `resolveLimits` adopts them. Starting from the
 * conservative values keeps this build working against either.
 */
export const LEGACY_REQUEST_LIMITS = Object.freeze({
  liked_ids: 50,
  excluded_ids: 100,
  excluded_titles: 100,
  seen_titles: 200,
  results: 50,
});
// Liked titles ride along only as a readable echo of liked_ids; the IDs are
// what the model consumes, so the echo stays small whatever the profile size.
const PROFILE_TITLE_ECHO = 50;

export function resolveLimits(health) {
  const advertised = health?.ranking?.request_limits;
  const limits = { ...LEGACY_REQUEST_LIMITS };
  if (advertised && typeof advertised === 'object') {
    for (const key of Object.keys(limits)) {
      const value = Number(advertised[key]);
      if (Number.isInteger(value) && value > 0) limits[key] = value;
    }
  }
  return limits;
}

/**
 * Cursor paging, if the backend offers it. `/api/health` advertises
 * `ranking.pagination`; a backend without that block is paged the old way, by
 * excluding everything already shown.
 */
export function resolvePagination(health) {
  const advertised = health?.ranking?.pagination;
  const batchSize = Number(advertised?.batch_size);
  if (!advertised || typeof advertised.endpoint !== 'string' || !Number.isInteger(batchSize) || batchSize <= 0) return null;
  return {
    endpoint: advertised.endpoint,
    batchSize,
    windowSize: Number(advertised.window_size) || null,
    maxWindows: Number(advertised.max_windows) || null,
  };
}

const positiveIds = (values) => [...new Set((values || []).map(Number).filter(id => Number.isInteger(id) && id > 0))];
const lastN = (values, count) => (values.length > count ? values.slice(values.length - count) : values);

/**
 * Build the /api/recommend body and an account of anything that did not fit.
 *
 * Nothing is dropped silently: `accounting` reports every truncation so the UI
 * can say so. When a list must be cut, the most recent entries win, because a
 * Like the user just made has to reach the model.
 *
 * Exclusion priority, highest first: likes that did not fit in `liked_ids`,
 * Not Interested, Already Watched, Want to Watch, skipped in this browsing
 * sequence, and then already-shown titles from newest to oldest. Likes that are sent in
 * `liked_ids` need no exclusion entry; the server never recommends a profile
 * title.
 */
export function buildRecommendRequest(session, options = {}) {
  const limits = options.limits || LEGACY_REQUEST_LIMITS;
  const { body, accounting } = buildSequenceRequest(session, options);
  return { body: { ...body, limit: Math.min(options.limit || limits.results, limits.results) }, accounting };
}

/**
 * The profile snapshot that defines one recommendation sequence.
 *
 * With cursor paging this body is built once, when a sequence starts, and sent
 * unchanged with every page; the server rebuilds the same ranked list from it.
 * `shownIds` is therefore only used by the exclusion fallback. Without it the
 * exclusion list holds what the user chose to exclude and nothing that grows
 * with paging.
 */
export function buildSequenceRequest(session, options = {}) {
  const limits = options.limits || LEGACY_REQUEST_LIMITS;
  const liked = (session.liked || []).filter(item => Number.isInteger(Number(item.id)) && Number(item.id) > 0);
  const likedSent = lastN(liked, limits.liked_ids);
  const likedOverflow = liked.slice(0, liked.length - likedSent.length);
  const likedIds = positiveIds(likedSent.map(item => item.id));
  const sentLikes = new Set(likedIds);

  const shownNewestFirst = positiveIds(options.shownIds).reverse();
  const wantedExclusions = positiveIds([
    ...likedOverflow.map(item => item.id),
    ...(session.notInterested || []).map(item => item.id),
    // Already Watched is an exclusion and nothing more: not a Like, not a dislike.
    ...(session.watched || []).map(item => item.id),
    ...(session.watchlist || []).map(item => item.id),
    ...(options.skippedIds || []),
    ...shownNewestFirst,
  ]).filter(id => !sentLikes.has(id));
  const excludedIds = wantedExclusions.slice(0, limits.excluded_ids);

  const hiddenTitles = legacyHiddenTitles(session);
  const excludedTitles = lastN(hiddenTitles, limits.excluded_titles);
  // Watched titles that carry an ID are excluded by ID above; only title-only
  // entries from the legacy frontend still need to travel as titles.
  const seen = legacySeenTitles(session);
  const seenTitles = lastN(seen, limits.seen_titles);
  const preferences = session.preferences || {};

  const body = {
    session_id: session.sessionId,
    liked_ids: likedIds,
    excluded_ids: excludedIds,
    excluded_titles: excludedTitles,
    seen_titles: seenTitles,
    // Hidden and seen titles travel once, at the top level. Disliked titles are
    // repeated here because the Hybrid also uses them as a negative signal.
    session_profile: {
      liked_titles: lastN(likedSent, PROFILE_TITLE_ECHO).map(item => item.title),
      disliked_titles: lastN((preferences.disliked_titles || []).filter(Boolean), PROFILE_TITLE_ECHO),
      preferred_genres: preferences.preferred_genres || [],
      excluded_genres: preferences.excluded_genres || [],
      temporary_ratings: preferences.temporary_ratings || {},
      previous_reference_titles: preferences.previous_reference_titles || [],
    },
    one_per_series: true,
  };
  const accounting = {
    likedTotal: liked.length,
    likedSent: likedIds.length,
    likedDropped: liked.length - likedSent.length,
    excludedTotal: wantedExclusions.length,
    excludedSent: excludedIds.length,
    excludedDropped: wantedExclusions.length - excludedIds.length,
    titlesDropped: (hiddenTitles.length - excludedTitles.length) + (seen.length - seenTitles.length),
  };
  return { body, accounting };
}

/** One page of a server-ranked sequence. `body` comes from buildSequenceRequest. */
export function recommendPage(body, { cursor = null, limit, signal, endpoint = '/api/recommend/page' } = {}) {
  return api(endpoint, { method: 'POST', signal, body: JSON.stringify({ ...body, cursor, limit }) });
}

/** Returns the API payload with the request `accounting` attached. */
export async function recommend(session, signal, options = {}) {
  // The model is authoritative for ordering; the client can only remove ineligible rows.
  const { body, accounting } = buildRecommendRequest(session, options);
  const payload = await api('/api/recommend', { method: 'POST', signal, body: JSON.stringify(body) });
  return { ...payload, accounting };
}

// Mirrors SessionPreferenceUpdate. That model forbids unknown keys, so the
// profile is sent field by field: spreading `session.preferences` meant one
// unrecognised key in a restored backup made every sync fail with a 422.
const SYNC_LIST_LIMITS = Object.freeze({
  liked_titles: 500,
  disliked_titles: 500,
  seen_titles: 1000,
  excluded_titles: 1000,
  preferred_genres: 100,
  excluded_genres: 100,
  previous_reference_titles: 50,
});

export function buildPreferencePayload(session) {
  const preferences = session.preferences || {};
  const lists = { ...preferences, liked_titles: (session.liked || []).map(item => item.title) };
  const payload = { session_id: session.sessionId, replace: true };
  for (const [field, limit] of Object.entries(SYNC_LIST_LIMITS)) {
    const values = Array.isArray(lists[field]) ? lists[field].filter(value => typeof value === 'string' && value.trim()) : [];
    payload[field] = lastN(values, limit);
  }
  const ratings = {};
  for (const [title, rating] of Object.entries(preferences.temporary_ratings || {})) {
    const value = Number(rating);
    if (title.trim() && Number.isFinite(value) && value >= 1 && value <= 10) ratings[title.trim()] = value;
  }
  payload.temporary_ratings = ratings;
  return payload;
}

export function syncPreferences(session, signal) {
  return api('/api/session/preferences', {
    method: 'POST',
    signal,
    body: JSON.stringify(buildPreferencePayload(session)),
  });
}

/**
 * Serialise profile syncs so the server always ends on the newest profile.
 *
 * Each sync replaces the stored profile. Fired independently, two quick clicks
 * could arrive out of order and leave the server holding the older one. Here at
 * most one request is in flight; anything queued behind it collapses into a
 * single follow-up carrying the latest state.
 */
export function createPreferenceSync(send = syncPreferences) {
  let inFlight = null;
  let pending = null;
  async function drain() {
    while (pending) {
      const next = pending;
      pending = null;
      try { await send(next); } catch { /* localStorage remains the source of truth */ }
    }
    inFlight = null;
  }
  return {
    /** Queue the latest profile. Resolves once the server has caught up with it. */
    push(session) {
      pending = session;
      if (!inFlight) inFlight = drain();
      return inFlight;
    },
    /** Resolves when nothing is queued or in flight. */
    idle() {
      return inFlight || Promise.resolve();
    },
  };
}
