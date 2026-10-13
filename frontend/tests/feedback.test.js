/**
 * The feedback contract, row by row.
 *
 * Five things a user can say about a title, and exactly what each one stores
 * and sends. If a row here changes, the product's meaning of that action has
 * changed.
 *
 *   Action           Stored in       Recommender input        Server profile
 *   Like             liked           liked_ids (moves model)  liked_titles
 *   Want to Watch    watchlist       excluded_ids             nothing
 *   Already Watched  watched         excluded_ids             seen_titles
 *   Not Interested   notInterested   excluded_ids             disliked_titles, excluded_titles
 *   Replace / Skip   nothing         excluded for this page   nothing
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FEEDBACK, blockedIds, clearNotInterested, clearSeenTitle, clearWatched, feedbackOf, legacySeenTitles, loadSession,
  markNotInterested, markWatched, normalizeSession, persistSession, toggleLiked, toggleWatched, toggleWatchlist,
} from '../src/lib/session.js';
import { buildPreferencePayload, buildSequenceRequest } from '../src/lib/api.js';
import { createCursorSource, createRecommendationFeed } from '../src/lib/recommendationFeed.js';

const LIMITS={liked_ids:500,excluded_ids:2000,excluded_titles:500,seen_titles:500,results:50};
const TITLE={id:77,title:'Subject Title'};
const blank=()=>normalizeSession({sessionId:'s',liked:[{id:1,title:'Anchor Like'}]},()=> 'unused');
const APPLY={
  liked:(session,item)=>toggleLiked(session,item),
  wantToWatch:(session,item)=>toggleWatchlist(session,item),
  alreadyWatched:(session,item)=>markWatched(session,item),
  notInterested:(session,item)=>markNotInterested(session,item),
};
const UNDO={
  liked:(session,item)=>toggleLiked(session,item),
  wantToWatch:(session,item)=>toggleWatchlist(session,item),
  alreadyWatched:(session,item)=>clearWatched(session,item),
  notInterested:(session,item)=>clearNotInterested(session,item),
};
const KINDS=Object.keys(FEEDBACK);
const LISTS=KINDS.map(kind=>FEEDBACK[kind].list);
const LEGACY=['seen_titles','disliked_titles','excluded_titles'];
const mentions=(value)=>JSON.stringify(value).includes(TITLE.title);
const reload=(session)=>{
  const original=globalThis.localStorage;let saved;
  globalThis.localStorage={setItem:(_key,value)=>{saved=value;},getItem:()=>saved};
  try { persistSession(session); return loadSession(); } finally { globalThis.localStorage=original; }
};

/** Everything observable about one title: where it is stored and what is sent. */
function observe(session) {
  const {body}=buildSequenceRequest(session,{limits:LIMITS});
  const sync=buildPreferencePayload(session);
  return {
    lists:LISTS.filter(list=>session[list].some(item=>item.id===TITLE.id)),
    legacy:LEGACY.filter(field=>session.preferences[field].includes(TITLE.title)),
    blocked:blockedIds(session).has(TITLE.id),
    sentAsLike:body.liked_ids.includes(TITLE.id),
    sentAsExclusion:body.excluded_ids.includes(TITLE.id),
    sentAsDislike:body.session_profile.disliked_titles.includes(TITLE.title),
    sentAsTitle:body.excluded_titles.includes(TITLE.title)||body.seen_titles.includes(TITLE.title),
    syncLiked:sync.liked_titles.includes(TITLE.title),
    syncSeen:sync.seen_titles.includes(TITLE.title),
    syncDisliked:sync.disliked_titles.includes(TITLE.title),
    syncExcluded:sync.excluded_titles.includes(TITLE.title),
  };
}

const EXPECTED={
  liked:{lists:['liked'],legacy:[],blocked:true,sentAsLike:true,sentAsExclusion:false,sentAsDislike:false,sentAsTitle:false,syncLiked:true,syncSeen:false,syncDisliked:false,syncExcluded:false},
  wantToWatch:{lists:['watchlist'],legacy:[],blocked:true,sentAsLike:false,sentAsExclusion:true,sentAsDislike:false,sentAsTitle:false,syncLiked:false,syncSeen:false,syncDisliked:false,syncExcluded:false},
  alreadyWatched:{lists:['watched'],legacy:['seen_titles'],blocked:true,sentAsLike:false,sentAsExclusion:true,sentAsDislike:false,sentAsTitle:false,syncLiked:false,syncSeen:true,syncDisliked:false,syncExcluded:false},
  notInterested:{lists:['notInterested'],legacy:['disliked_titles','excluded_titles'],blocked:true,sentAsLike:false,sentAsExclusion:true,sentAsDislike:true,sentAsTitle:false,syncLiked:false,syncSeen:false,syncDisliked:true,syncExcluded:true},
};
const NOTHING={lists:[],legacy:[],blocked:false,sentAsLike:false,sentAsExclusion:false,sentAsDislike:false,sentAsTitle:false,syncLiked:false,syncSeen:false,syncDisliked:false,syncExcluded:false};

for(const kind of KINDS) {
  test(`${kind}: stores and sends exactly what the contract says`,()=>{
    const start=blank();
    const applied=APPLY[kind](start,TITLE);
    assert.deepEqual(observe(applied),EXPECTED[kind]);
    assert.equal(feedbackOf(applied,TITLE),kind);
    // The table in session.js and the behaviour agree.
    assert.deepEqual(EXPECTED[kind].lists,[FEEDBACK[kind].list]);
    assert.deepEqual([...EXPECTED[kind].legacy].sort(),[...FEEDBACK[kind].legacyTitles].sort());
    assert.equal(EXPECTED[kind].sentAsLike,FEEDBACK[kind].profileInput);
    assert.equal(EXPECTED[kind].sentAsDislike,FEEDBACK[kind].negative);
    // It survives a reload, and the action never mutated its input.
    assert.deepEqual(observe(reload(applied)),EXPECTED[kind]);
    assert.deepEqual(observe(start),NOTHING);
    // Applying it twice changes nothing more, and undoing it leaves no trace at all.
    if(kind==='alreadyWatched'||kind==='notInterested')assert.deepEqual(APPLY[kind](applied,TITLE),applied);
    assert.deepEqual(observe(UNDO[kind](applied,TITLE)),NOTHING);
    // The anchor Like is untouched throughout: feedback on one title never edits another.
    assert.deepEqual(applied.liked.filter(item=>item.id===1),[{id:1,title:'Anchor Like'}]);
  });
}

test('only a Like is sent as a profile input; the other three never reach liked_ids',()=>{
  let session=blank();
  session=toggleWatchlist(session,{id:10,title:'Planned'});
  session=markWatched(session,{id:11,title:'Seen'});
  session=markNotInterested(session,{id:12,title:'Rejected'});
  session=toggleLiked(session,{id:13,title:'Loved'});
  const {body}=buildSequenceRequest(session,{limits:LIMITS});
  assert.deepEqual(body.liked_ids,[1,13]);
  assert.deepEqual(body.excluded_ids,[12,11,10]);
  assert.deepEqual(body.session_profile.liked_titles,['Anchor Like','Loved']);
  assert.deepEqual(body.session_profile.disliked_titles,['Rejected']);
});

test('Already Watched is not a dislike: no negative field ever names it',()=>{
  const session=markWatched(blank(),TITLE);
  const {body}=buildSequenceRequest(session,{limits:LIMITS});
  const sync=buildPreferencePayload(session);
  assert.equal(mentions(body.session_profile),false,'not in the profile echo, liked or disliked');
  assert.equal(mentions(sync.disliked_titles),false);
  assert.equal(mentions(sync.excluded_titles),false);
  assert.equal(mentions(sync.liked_titles),false);
  assert.deepEqual(sync.seen_titles,[TITLE.title]);
  assert.deepEqual(session.notInterested,[]);
  assert.deepEqual(session.preferences.temporary_ratings,{});
});

test('a title holds one kind of feedback at a time: the latest action wins',()=>{
  for(const first of KINDS)for(const second of KINDS) {
    if(first===second)continue;
    const moved=APPLY[second](APPLY[first](blank(),TITLE),TITLE);
    assert.deepEqual(observe(moved),EXPECTED[second],`${first} then ${second}`);
    assert.deepEqual(observe(UNDO[second](moved,TITLE)),NOTHING,`undoing ${second} after ${first} must not restore ${first}`);
  }
});

test('the Already Watched toggle turns on and off',()=>{
  const on=toggleWatched(blank(),TITLE);
  assert.deepEqual(observe(on),EXPECTED.alreadyWatched);
  assert.deepEqual(observe(toggleWatched(on,TITLE)),NOTHING);
});

test('Replace stores nothing and is not any of the four kinds of feedback',async()=>{
  const session=blank();
  const before=JSON.stringify(session);
  const calls=[];
  const feed=createRecommendationFeed({source:createCursorSource({
    buildRequest:(current,{skippedIds})=>buildSequenceRequest(current,{skippedIds,limits:LIMITS}),
    requestPage:async(body,{cursor})=>{
      calls.push(body);
      const offset=cursor?Number(cursor):0;
      const results=Array.from({length:36},(_,i)=>({id:100+offset+i,title:`Anime ${100+offset+i}`}));
      return {results,sequence:{has_more:true,next_cursor:String(offset+36)}};
    },
  })});
  const first=await feed.start(session);
  const target=first.visible[4];
  const result=await feed.skip(session,target);
  assert.notEqual(result.visible[4].id,target.id);
  assert.equal(JSON.stringify(session),before,'the profile is byte-for-byte unchanged');
  assert.equal(feedbackOf(session,target),null);
  assert.equal(blockedIds(session).has(target.id),false);
  assert.equal(JSON.stringify(buildPreferencePayload(session)).includes(target.title),false);
  // It is remembered for this page only, and sent with the next sequence as an exclusion.
  assert.deepEqual(feed.state().skippedIds,[target.id]);
  await feed.start(toggleLiked(session,{id:5,title:'New Like'}));
  assert.deepEqual(calls.at(-1).excluded_ids,[target.id]);
  assert.deepEqual(calls.at(-1).session_profile.disliked_titles,[]);
  // A reload forgets it.
  assert.deepEqual(createRecommendationFeed({source:{open:async()=>({results:[],hasMore:false,state:{}}),more:async()=>({results:[],hasMore:false,state:{}})}}).state().skippedIds,[]);
});

test('watched titles from the legacy frontend are honoured by title and can be removed',()=>{
  const legacy=normalizeSession({sessionId:'s',liked:[],preferences:{seen_titles:['Old Watched','Another Seen']}},()=> 'unused');
  assert.deepEqual(legacySeenTitles(legacy),['Old Watched','Another Seen']);
  assert.deepEqual(buildSequenceRequest(legacy,{limits:LIMITS}).body.seen_titles,['Old Watched','Another Seen']);
  // A new entry with an ID is excluded by ID and is not repeated as a title.
  const mixed=markWatched(legacy,TITLE);
  const {body}=buildSequenceRequest(mixed,{limits:LIMITS});
  assert.deepEqual(body.seen_titles,['Old Watched','Another Seen']);
  assert.deepEqual(body.excluded_ids,[TITLE.id]);
  assert.deepEqual(legacySeenTitles(mixed),['Old Watched','Another Seen']);
  assert.deepEqual(legacySeenTitles(clearSeenTitle(mixed,'old watched')),['Another Seen']);
  assert.deepEqual(buildPreferencePayload(mixed).seen_titles,['Old Watched','Another Seen',TITLE.title]);
});

test('a stored profile that lists one title twice is repaired to one kind of feedback',()=>{
  const repaired=normalizeSession({sessionId:'s',liked:[TITLE],watchlist:[TITLE],watched:[TITLE,{id:2,title:'Only Watched'}],notInterested:[TITLE,{id:2,title:'Only Watched'}]},()=> 'unused');
  assert.equal(feedbackOf(repaired,TITLE),'liked');
  assert.deepEqual(repaired.watched,[{id:2,title:'Only Watched'}]);
  assert.deepEqual(repaired.notInterested,[]);
  assert.deepEqual(normalizeSession({sessionId:'s',liked:[]},()=> 'unused').watched,[],'older sessions gain an empty list');
});
