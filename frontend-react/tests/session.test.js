import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SESSION_STORAGE_KEY, blockedIds, blockedTitleKeys, clearHiddenTitle, clearNotInterested, defaultPreferences,
  legacyHiddenTitles, loadSession, markNotInterested, normalizeSession, persistSession, toggleLiked, toggleWatchlist,
} from '../src/lib/session.js';

function withStorage(storage, run) {
  const original=globalThis.localStorage;
  globalThis.localStorage=storage;
  try { return run(); } finally { globalThis.localStorage=original; }
}

test('retains existing vanilla browser storage key', () => {
  assert.equal(SESSION_STORAGE_KEY, 'animeCompassSession');
});

test('migrates old session without deleting likes or other preferences', () => {
  const old = { sessionId: 'legacy_123', liked: [{id: 42, title: 'Frieren', image_url: 'https://example.test/42.jpg'}], preferences: { seen_titles: ['Cowboy Bebop'], custom_flag: 'keep' } };
  const next = normalizeSession(old);
  assert.deepEqual(next.liked, old.liked);
  assert.deepEqual(next.preferences.seen_titles, ['Cowboy Bebop']);
  assert.equal(next.preferences.custom_flag, 'keep');
});

test('unknown or malformed saved state does not crash', () => {
  assert.deepEqual(normalizeSession(null, ()=>'fixed').liked, []);
  assert.equal(normalizeSession({sessionId:'?!'}, ()=>'fixed').sessionId, 'fixed');
  const repaired=normalizeSession({preferences:{excluded_titles:null,seen_titles:'oops',disliked_titles:[7,'Kept',''],temporary_ratings:[]}},()=>'fixed');
  assert.deepEqual(repaired.preferences.excluded_titles,[]);
  assert.deepEqual(repaired.preferences.seen_titles,[]);
  assert.deepEqual(repaired.preferences.disliked_titles,['Kept']);
  assert.deepEqual(repaired.preferences.temporary_ratings,{});
});

test('liking and unliking do not mutate the original array', () => {
  const start = normalizeSession({liked:[]}, ()=>'fixed');
  const liked = toggleLiked(start, {id:1,title:'Test'});
  assert.equal(start.liked.length, 0);
  assert.equal(liked.liked.length, 1);
  assert.equal(toggleLiked(liked, {id:1,title:'Test'}).liked.length, 0);
});

test('legacy session gains empty Want to Watch and Not Interested lists',()=>{
  const old=normalizeSession({sessionId:'persisted',liked:[{id:3,title:'Liked'}]},()=> 'unused');
  assert.deepEqual(old.watchlist,[]);
  assert.deepEqual(old.notInterested,[]);
  assert.ok(blockedIds(old).has(3));
});

test('Want to Watch is independent from likes, mutually exclusive, and persisted',()=>{
  const initial=normalizeSession({sessionId:'s',liked:[{id:1,title:'A'}]},()=> 'unused');
  const watch=toggleWatchlist(initial,{id:2,title:'B'});
  assert.equal(watch.liked.length,1);
  assert.equal(watch.watchlist.length,1);
  // Bookmarking is future intent: it is not "seen" and not negative feedback.
  assert.equal(watch.preferences.seen_titles.length,0);
  assert.equal(watch.preferences.excluded_titles.length,0);
  assert.equal(watch.preferences.disliked_titles.length,0);
  assert.deepEqual(watch.preferences.temporary_ratings,{});
  const moved=toggleLiked(watch,{id:2,title:'B'});
  assert.deepEqual(moved.liked.map(x=>x.id),[1,2]);
  assert.deepEqual(moved.watchlist,[]);
  const toWatch=toggleWatchlist(moved,{id:1,title:'A'});
  assert.deepEqual(toWatch.liked.map(x=>x.id),[2]);
  assert.deepEqual(toWatch.watchlist.map(x=>x.id),[1]);
  let saved;
  withStorage({setItem:(key,val)=>saved={key,data:JSON.parse(val)}},()=>assert.equal(persistSession(toWatch),true));
  assert.equal(saved.key,'animeCompassSession');
  assert.deepEqual(saved.data.watchlist.map(x=>x.title),['A']);
});

test('Not Interested is persisted negative feedback and is idempotent',()=>{
  const start=normalizeSession({sessionId:'s',liked:[]},()=> 'unused');
  const first=markNotInterested(start,{id:14,title:'Bad Match'});
  assert.deepEqual(first.notInterested,[{id:14,title:'Bad Match'}]);
  // Written to the legacy fields too: that is what the server profile and the agent read.
  assert.deepEqual(first.preferences.disliked_titles,['Bad Match']);
  assert.deepEqual(first.preferences.excluded_titles,['Bad Match']);
  const again=markNotInterested(first,{id:14,title:'Bad Match'});
  assert.equal(again.notInterested.length,1);
  assert.equal(again.preferences.excluded_titles.length,1);
  assert.deepEqual(start.preferences.excluded_titles,[]);
  assert.deepEqual(start.notInterested,[]);
  assert.ok(blockedIds(first).has(14));

  let saved;
  withStorage({setItem:(_key,val)=>saved=JSON.parse(val)},()=>persistSession(first));
  const reloaded=withStorage({getItem:()=>JSON.stringify(saved)},()=>loadSession());
  assert.deepEqual(reloaded.notInterested.map(x=>x.id),[14]);
  assert.deepEqual(reloaded.preferences.disliked_titles,['Bad Match']);
});

test('Not Interested can be undone and is overridden by a Like or a bookmark',()=>{
  const start=normalizeSession({sessionId:'s',liked:[{id:1,title:'Keep'}]},()=> 'unused');
  const rejected=markNotInterested(start,{id:14,title:'Bad Match'});
  const undone=clearNotInterested(rejected,{id:14,title:'Bad Match'});
  assert.deepEqual(undone.notInterested,[]);
  assert.deepEqual(undone.preferences.disliked_titles,[]);
  assert.deepEqual(undone.preferences.excluded_titles,[]);

  const liked=toggleLiked(rejected,{id:14,title:'Bad Match'});
  assert.deepEqual(liked.liked.map(x=>x.id),[1,14]);
  assert.deepEqual(liked.notInterested,[]);
  assert.deepEqual(liked.preferences.disliked_titles,[]);

  const bookmarked=toggleWatchlist(rejected,{id:14,title:'Bad Match'});
  assert.deepEqual(bookmarked.watchlist.map(x=>x.id),[14]);
  assert.deepEqual(bookmarked.notInterested,[]);

  // Rejecting a saved title removes it from the saved lists.
  const flipped=markNotInterested(start,{id:1,title:'Keep'});
  assert.deepEqual(flipped.liked,[]);
  assert.deepEqual(flipped.notInterested.map(x=>x.id),[1]);
});

test('title-only negative feedback from the legacy frontend is still honoured and removable',()=>{
  const legacy=normalizeSession({sessionId:'s',liked:[],preferences:{disliked_titles:['Old Dislike'],excluded_titles:['Old Dislike','Hidden Only'],seen_titles:['Watched Once']}},()=> 'unused');
  assert.deepEqual(legacyHiddenTitles(legacy),['Old Dislike','Hidden Only']);
  assert.deepEqual([...blockedTitleKeys(legacy)].sort(),['hidden only','old dislike','watched once']);
  const withId=markNotInterested(legacy,{id:9,title:'New Dislike'});
  assert.deepEqual(legacyHiddenTitles(withId),['Old Dislike','Hidden Only']);
  const cleared=clearHiddenTitle(withId,'old dislike');
  assert.deepEqual(legacyHiddenTitles(cleared),['Hidden Only']);
  assert.deepEqual(cleared.preferences.disliked_titles,['New Dislike']);
});

test('what this app saves is still readable by the legacy vanilla frontend',()=>{
  const session=markNotInterested(
    toggleWatchlist(toggleLiked(normalizeSession({sessionId:'legacy-abc'},()=> 'unused'),{id:1,title:'A'}),{id:2,title:'B'}),
    {id:3,title:'C'},
  );
  let saved;
  withStorage({setItem:(_key,val)=>saved=JSON.parse(val)},()=>persistSession(session));
  // Exactly what frontend/app.js loadSessionState() does.
  const legacyState={
    sessionId:saved.sessionId,
    liked:Array.isArray(saved.liked)?saved.liked:[],
    preferences:{...defaultPreferences(),...(saved.preferences||{})},
  };
  assert.equal(legacyState.sessionId,'legacy-abc');
  assert.deepEqual(legacyState.liked,[{id:1,title:'A'}]);
  assert.deepEqual(Object.keys(legacyState.preferences).sort(),Object.keys(defaultPreferences()).sort());
  assert.deepEqual(legacyState.preferences.disliked_titles,['C']);
  assert.deepEqual(legacyState.preferences.excluded_titles,['C']);
  assert.deepEqual(Object.keys(saved).sort(),['liked','notInterested','preferences','sessionId','watchlist']);
});

test('a storage failure is reported instead of thrown',()=>{
  const session=normalizeSession({sessionId:'s'},()=> 'unused');
  const failed=withStorage({setItem:()=>{throw new Error('QuotaExceededError');}},()=>persistSession(session));
  assert.equal(failed,false);
});
