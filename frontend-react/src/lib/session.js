/**
 * Browser profile, stored under the legacy vanilla frontend's key.
 *
 * The legacy object is `{sessionId, liked, preferences}`. This app keeps those
 * three fields exactly and adds three optional ones the legacy frontend
 * ignores: `watchlist` (Want to Watch), `watched` (Already Watched) and
 * `notInterested`.
 *
 * Five kinds of feedback, deliberately not interchangeable:
 *
 *   Like             positive preference; the only input to the ALS profile.
 *   Want to Watch    future intent; removed from recommendations, never
 *                    treated as watched or as a positive rating.
 *   Already Watched  seen, with no judgement attached; removed from
 *                    recommendations, never a Like and never negative feedback.
 *   Not Interested   explicit negative feedback, persisted here.
 *   Skip / Replace   not stored at all. It belongs to the current browsing
 *                    sequence (see recommendationFeed.js) and ends with it.
 *
 * A title is in at most one of the four stored lists: the latest action wins,
 * and undoing an action leaves the title with no state. `FEEDBACK` below is the
 * whole contract in one table; tests/feedback.test.js checks every row of it.
 */
export const SESSION_STORAGE_KEY = 'animeCompassSession';

/**
 * What each stored feedback type means downstream.
 *   list           the session field holding the titles
 *   legacyTitles   legacy `preferences` title lists it is mirrored into, which
 *                  is what the server profile, the agent, and the vanilla
 *                  frontend read
 *   profileInput   sent as `liked_ids`, so it moves the recommender
 *   negative       sent as a disliked title
 */
export const FEEDBACK = Object.freeze({
  liked: { list: 'liked', legacyTitles: [], profileInput: true, negative: false },
  wantToWatch: { list: 'watchlist', legacyTitles: [], profileInput: false, negative: false },
  alreadyWatched: { list: 'watched', legacyTitles: ['seen_titles'], profileInput: false, negative: false },
  notInterested: { list: 'notInterested', legacyTitles: ['disliked_titles', 'excluded_titles'], profileInput: false, negative: true },
});
const STORED_LISTS = Object.values(FEEDBACK).map(kind => kind.list);
const LEGACY_TITLE_FIELDS = ['seen_titles', 'disliked_titles', 'excluded_titles'];

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
  // Older versions never wrote the optional lists. A title found in two lists
  // keeps the first, in this order.
  const taken = new Set();
  const exclusive = (input) => cleanAnimeList(input).filter(item => {
    if (taken.has(Number(item.id))) return false;
    taken.add(Number(item.id)); return true;
  });
  const liked = exclusive(raw.liked);
  const watchlist = exclusive(raw.watchlist || raw.wantToWatch);
  const watched = exclusive(raw.watched);
  const notInterested = exclusive(raw.notInterested);
  return {
    sessionId: typeof raw.sessionId === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(raw.sessionId)
      ? raw.sessionId : sessionIdFactory(),
    liked,
    watchlist,
    watched,
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
      watched: session.watched || [],
      notInterested: session.notInterested || [],
      preferences: session.preferences,
    }));
    return true;
  } catch { return false; }
}

const has=(list,item)=>(list||[]).some(anime=>sameId(anime,item));
const without=(list,item)=>(list||[]).filter(anime=>!sameId(anime,item));
const withoutTitle=(entries,title)=>(entries||[]).filter(value=>titleKey(value)!==titleKey(title));
const withTitle=(entries,title)=>(entries||[]).some(value=>titleKey(value)===titleKey(title)) ? entries : [...(entries||[]),title.trim()];

/** Remove every stored trace of one title: all four lists and the legacy title lists. */
function forget(session,item) {
  const next={...session,preferences:{...session.preferences}};
  for (const list of STORED_LISTS) next[list]=without(session[list],item);
  for (const field of LEGACY_TITLE_FIELDS) next.preferences[field]=withoutTitle(session.preferences[field],item.title);
  return next;
}
/** Give one title exactly one kind of feedback, replacing whatever it had. */
function place(session,item,kind) {
  if (!validAnime(item)) return session;
  const next=forget(session,item);
  next[kind.list]=[...next[kind.list],item];
  for (const field of kind.legacyTitles) next.preferences[field]=withTitle(next.preferences[field],item.title);
  return next;
}
const toggle=(session,item,kind)=>has(session[kind.list],item) ? forget(session,item) : place(session,item,kind);

export const toggleLiked=(session,item)=>toggle(session,item,FEEDBACK.liked);
export const toggleWatchlist=(session,item)=>toggle(session,item,FEEDBACK.wantToWatch);
/** Seen it. Hides the title from recommendations and says nothing else about it. */
export const toggleWatched=(session,item)=>toggle(session,item,FEEDBACK.alreadyWatched);
export const markWatched=(session,item)=>place(session,item,FEEDBACK.alreadyWatched);
export const clearWatched=(session,item)=>(has(session.watched,item) ? forget(session,item) : session);
/** Persist explicit negative feedback. */
export const markNotInterested=(session,item)=>place(session,item,FEEDBACK.notInterested);
export function clearNotInterested(session,item) {
  const titled=[...(session.preferences.disliked_titles||[]),...(session.preferences.excluded_titles||[])]
    .some(value=>titleKey(value)===titleKey(item.title));
  return has(session.notInterested,item) || titled ? forget(session,item) : session;
}
/** Which stored feedback a title has, as a FEEDBACK key, or null. */
export function feedbackOf(session,item) {
  for (const [name,kind] of Object.entries(FEEDBACK)) if (has(session[kind.list],item)) return name;
  return null;
}

const titlesWithoutEntry=(titles,entries)=>{
  const known=new Set((entries||[]).map(item=>titleKey(item.title)));
  const seen=new Set();
  return titles.filter(title=>{
    const key=titleKey(title);
    if (!key || known.has(key) || seen.has(key)) return false;
    seen.add(key); return true;
  });
};
/** Title-only negative feedback written before IDs were stored. */
export function legacyHiddenTitles(session) {
  return titlesWithoutEntry(
    [...(session.preferences.disliked_titles||[]),...(session.preferences.excluded_titles||[])],
    session.notInterested,
  );
}
/** Title-only "watched" entries written by the legacy frontend. */
export function legacySeenTitles(session) {
  return titlesWithoutEntry(session.preferences.seen_titles||[],session.watched);
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
export function clearSeenTitle(session,title) {
  return {...session,preferences:{...session.preferences,seen_titles:withoutTitle(session.preferences.seen_titles,title)}};
}
/** IDs that must never be recommended: liked, bookmarked, watched, or rejected. */
export function blockedIds(session) {
  return new Set(STORED_LISTS.flatMap(list=>session[list]||[]).map(x=>Number(x.id)));
}
/** Titles that must never be recommended, for entries that carry no ID. */
export function blockedTitleKeys(session) {
  const preferences=session.preferences||{};
  return new Set(LEGACY_TITLE_FIELDS.flatMap(field=>preferences[field]||[]).map(titleKey).filter(Boolean));
}
