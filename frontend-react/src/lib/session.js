/**
 * Browser profile, stored under the legacy vanilla frontend's key.
 *
 * The legacy object is `{sessionId, liked, preferences}`. This app keeps those
 * three fields exactly and adds two optional ones the legacy frontend ignores:
 * `watchlist` (Want to Watch) and `notInterested`.
 *
 * Four kinds of feedback, deliberately not interchangeable:
 *   Like            positive preference; the only input to the ALS profile.
 *   Want to Watch   future intent; removed from recommendations, never treated
 *                   as watched or as a positive rating.
 *   Not Interested  explicit negative feedback, persisted here.
 *   Skip / Replace  not stored at all. It belongs to the current browsing
 *                   sequence (see recommendationFeed.js) and ends with it.
 */
export const SESSION_STORAGE_KEY = 'animeCompassSession';

const PREFERENCE_LIST_FIELDS = [
  'disliked_titles',
  'seen_titles',
  'excluded_titles',
  'preferred_genres',
  'excluded_genres',
  'previous_reference_titles',
];

export function defaultPreferences() {
  return {
    disliked_titles: [],
    seen_titles: [],
    excluded_titles: [],
    preferred_genres: [],
    excluded_genres: [],
    temporary_ratings: {},
    previous_reference_titles: [],
  };
}
function makeSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
const sameId=(a,b)=>Number(a?.id)===Number(b?.id);
export const titleKey=(value)=>String(value??'').trim().toLocaleLowerCase();
function validAnime(item) {
  return item && Number.isInteger(Number(item.id)) && Number(item.id)>0 && typeof item.title === 'string';
}
function cleanAnimeList(input) {
  if (!Array.isArray(input)) return [];
  const seen = new Set();
  return input.filter(item => {
    if (!validAnime(item) || seen.has(Number(item.id))) return false;
    seen.add(Number(item.id)); return true;
  });
}
function cleanPreferences(raw) {
  // Unknown keys are kept: a backup written by another version must round-trip.
  const preferences = { ...defaultPreferences(), ...(raw && typeof raw === 'object' ? raw : {}) };
  for (const field of PREFERENCE_LIST_FIELDS) {
    preferences[field] = Array.isArray(preferences[field])
      ? preferences[field].filter(value => typeof value === 'string' && value.trim())
      : [];
  }
  const ratings = preferences.temporary_ratings;
  preferences.temporary_ratings = ratings && typeof ratings === 'object' && !Array.isArray(ratings) ? ratings : {};
  return preferences;
}
export function normalizeSession(value, sessionIdFactory = makeSessionId) {
  const raw = value && typeof value === 'object' ? value : {};
  const liked = cleanAnimeList(raw.liked);
  const likedIds = new Set(liked.map(item=>Number(item.id)));
  // Older versions never wrote these fields. They must remain optional.
  const watchlist = cleanAnimeList(raw.watchlist || raw.wantToWatch)
    .filter(item=>!likedIds.has(Number(item.id)));
  const savedIds = new Set([...likedIds, ...watchlist.map(item=>Number(item.id))]);
  const notInterested = cleanAnimeList(raw.notInterested)
    .filter(item=>!savedIds.has(Number(item.id)));
  return {
    sessionId: typeof raw.sessionId === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(raw.sessionId)
      ? raw.sessionId : sessionIdFactory(),
    liked,
    watchlist,
    notInterested,
    preferences: cleanPreferences(raw.preferences),
  };
}
export function loadSession() {
  try { return normalizeSession(JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || '{}')); }
  catch { return normalizeSession({}); }
}
/** Returns false when storage is unavailable or full, instead of throwing mid-render. */
export function persistSession(session) {
  try {
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify({
      sessionId: session.sessionId,
      liked: session.liked,
      watchlist: session.watchlist || [],
      notInterested: session.notInterested || [],
      preferences: session.preferences,
    }));
    return true;
  } catch { return false; }
}

const without=(list,item)=>(list||[]).filter(anime=>!sameId(anime,item));
const withoutTitle=(entries,title)=>(entries||[]).filter(value=>titleKey(value)!==titleKey(title));
const withTitle=(entries,title)=>(entries||[]).some(value=>titleKey(value)===titleKey(title)) ? entries : [...(entries||[]),title.trim()];

/** Drop one title's negative feedback, in both the ID list and the legacy title lists. */
function forgetNotInterested(session,item) {
  const listed=(session.notInterested||[]).some(anime=>sameId(anime,item));
  const titled=[...(session.preferences.disliked_titles||[]),...(session.preferences.excluded_titles||[])]
    .some(value=>titleKey(value)===titleKey(item.title));
  if (!listed && !titled) return session;
  return {
    ...session,
    notInterested: without(session.notInterested,item),
    preferences: {
      ...session.preferences,
      disliked_titles: withoutTitle(session.preferences.disliked_titles,item.title),
      excluded_titles: withoutTitle(session.preferences.excluded_titles,item.title),
    },
  };
}
export function toggleLiked(session, item) {
  const exists = session.liked.some(anime=>sameId(anime,item));
  if (exists) return { ...session, liked: without(session.liked,item) };
  // An explicit Like overrides an earlier bookmark or rejection of the same title.
  const cleared=forgetNotInterested(session,item);
  return { ...cleared, liked: [...cleared.liked,item], watchlist: without(cleared.watchlist,item) };
}
export function toggleWatchlist(session,item) {
  const exists=(session.watchlist||[]).some(anime=>sameId(anime,item));
  if (exists) return { ...session, watchlist: without(session.watchlist,item) };
  const cleared=forgetNotInterested(session,item);
  return { ...cleared, watchlist: [...(cleared.watchlist||[]),item], liked: without(cleared.liked,item) };
}
/**
 * Persist explicit negative feedback. The title is also written to the legacy
 * `disliked_titles` / `excluded_titles` lists, which is what the server-side
 * profile, the agent, and the vanilla frontend read.
 */
export function markNotInterested(session,item) {
  if (!validAnime(item)) return session;
  const already=(session.notInterested||[]).some(anime=>sameId(anime,item));
  return {
    ...session,
    liked: without(session.liked,item),
    watchlist: without(session.watchlist,item),
    notInterested: already ? session.notInterested : [...(session.notInterested||[]),item],
    preferences: {
      ...session.preferences,
      disliked_titles: withTitle(session.preferences.disliked_titles,item.title),
      excluded_titles: withTitle(session.preferences.excluded_titles,item.title),
    },
  };
}
export function clearNotInterested(session,item) {
  return forgetNotInterested(session,item);
}
/** Title-only negative feedback written before IDs were stored. */
export function legacyHiddenTitles(session) {
  const known=new Set((session.notInterested||[]).map(item=>titleKey(item.title)));
  const seen=new Set();
  return [...(session.preferences.disliked_titles||[]),...(session.preferences.excluded_titles||[])]
    .filter(title=>{
      const key=titleKey(title);
      if (!key || known.has(key) || seen.has(key)) return false;
      seen.add(key); return true;
    });
}
export function clearHiddenTitle(session,title) {
  return {
    ...session,
    preferences: {
      ...session.preferences,
      disliked_titles: withoutTitle(session.preferences.disliked_titles,title),
      excluded_titles: withoutTitle(session.preferences.excluded_titles,title),
    },
  };
}
/** IDs that must never be recommended: saved, bookmarked, or rejected. */
export function blockedIds(session) {
  return new Set([...session.liked,...(session.watchlist||[]),...(session.notInterested||[])].map(x=>Number(x.id)));
}
/** Titles that must never be recommended, for entries that carry no ID. */
export function blockedTitleKeys(session) {
  const preferences=session.preferences||{};
  return new Set([
    ...(preferences.excluded_titles||[]),
    ...(preferences.disliked_titles||[]),
    ...(preferences.seen_titles||[]),
  ].map(titleKey).filter(Boolean));
}
