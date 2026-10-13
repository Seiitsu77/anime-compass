import test from 'node:test';
import assert from 'node:assert/strict';
import {
  LEGACY_REQUEST_LIMITS, api, buildPreferencePayload, buildRecommendRequest, createPreferenceSync,
  recommend, resolveLimits, searchCatalog, syncPreferences,
} from '../src/lib/api.js';
import { markNotInterested, normalizeSession, toggleLiked, toggleWatchlist } from '../src/lib/session.js';

const anime=(id)=>({id,title:`Anime ${id}`});
const profile=(likes=0)=>normalizeSession({sessionId:'s',liked:Array.from({length:likes},(_,i)=>anime(i+1))},()=> 'unused');
async function withFetch(handler, run) {
  const original=globalThis.fetch;
  globalThis.fetch=handler;
  try { return await run(); } finally { globalThis.fetch=original; }
}
const ok=(payload)=>({ok:true,status:200,json:async()=>payload});

test('catalog search posts correct schema fields', async () => {
  let body;
  await withFetch(async (url,options)=>{
    assert.equal(url,'/api/search'); body=JSON.parse(options.body);
    return ok({results:[],total:0});
  },async()=>{
    await searchCatalog({query:'Gundam',genre:'Mecha',format:'Movie',year:'2022',sort:'relevance'});
  });
  assert.equal(body.sort_by,'relevance');
  assert.deepEqual(body.include_genres,['Mecha']);
  assert.deepEqual(body.formats,['Movie']);
  assert.equal(body.min_year,2022);
  assert.equal(body.max_year,2022);
});

test('personalized recommendation passes liked ids, not reference titles',async()=>{
  let body;
  const payload=await withFetch(async(url,options)=>{
    assert.equal(url,'/api/recommend');body=JSON.parse(options.body);
    return ok({results:[]});
  },()=>recommend({sessionId:'some-id',liked:[{id:1,title:'Example'}],preferences:{excluded_titles:[],seen_titles:[]}}));
  assert.deepEqual(body.liked_ids,[1]);
  assert.equal(body.reference_titles,undefined);
  assert.equal(body.one_per_series,true);
  assert.equal(body.session_profile.liked_titles[0],'Example');
  // No filter fields: anything here would route the request off the ALS + LambdaMART fast path.
  for (const field of ['include_genres','exclude_genres','formats','min_score','free_text_preferences','query'])
    assert.equal(body[field],undefined,`${field} must not be sent`);
  assert.equal(payload.accounting.likedSent,1);
});

test('exclusions cover Want to Watch, Not Interested, skipped and shown, without repeating sent likes',()=>{
  let session=profile(1);
  session=toggleWatchlist(session,anime(3));
  session=markNotInterested(session,anime(4));
  const {body,accounting}=buildRecommendRequest(session,{shownIds:[5,6],skippedIds:[7]});
  assert.equal(body.top_k,undefined);
  assert.equal(body.limit,50);
  assert.deepEqual(body.liked_ids,[1]);
  // Priority order: Not Interested, Want to Watch, skipped, then shown (newest first).
  assert.deepEqual(body.excluded_ids,[4,3,7,6,5]);
  assert.equal(accounting.excludedDropped,0);
});

test('Want to Watch is never sent as a like, as seen, or as negative feedback',()=>{
  const session=toggleWatchlist(profile(2),anime(9));
  const {body}=buildRecommendRequest(session);
  assert.deepEqual(body.liked_ids,[1,2]);
  assert.ok(body.excluded_ids.includes(9));
  assert.deepEqual(body.seen_titles,[]);
  assert.deepEqual(body.excluded_titles,[]);
  assert.ok(!body.session_profile.liked_titles.includes('Anime 9'));
  assert.deepEqual(body.session_profile.disliked_titles,[]);
  assert.deepEqual(body.session_profile.temporary_ratings,{});
  const synced=buildPreferencePayload(session);
  assert.ok(!JSON.stringify(synced).includes('Anime 9'),'a bookmark must not reach the server-side profile');
});

test('a skipped title is an exclusion for this request only, never negative feedback',()=>{
  const session=profile(2);
  const {body}=buildRecommendRequest(session,{skippedIds:[30]});
  assert.ok(body.excluded_ids.includes(30));
  assert.deepEqual(body.excluded_titles,[]);
  assert.deepEqual(body.session_profile.disliked_titles,[]);
  assert.ok(!JSON.stringify(buildPreferencePayload(session)).includes('30'));
});

test('Not Interested is sent by ID and as a disliked title',()=>{
  const session=markNotInterested(profile(2),anime(40));
  const {body}=buildRecommendRequest(session);
  assert.ok(body.excluded_ids.includes(40));
  assert.deepEqual(body.session_profile.disliked_titles,['Anime 40']);
  // The ID already covers it; the title list is for entries that have no ID.
  assert.deepEqual(body.excluded_titles,[]);
  const synced=buildPreferencePayload(session);
  assert.deepEqual(synced.disliked_titles,['Anime 40']);
  assert.deepEqual(synced.excluded_titles,['Anime 40']);
});

test('legacy title-only dislikes are sent as excluded titles',()=>{
  const session=normalizeSession({sessionId:'s',liked:[anime(1)],preferences:{disliked_titles:['Old One'],excluded_titles:['Old One','Hidden'],seen_titles:['Seen']}},()=> 'unused');
  const {body}=buildRecommendRequest(session);
  assert.deepEqual(body.excluded_titles,['Old One','Hidden']);
  assert.deepEqual(body.seen_titles,['Seen']);
});

test('more than 50 likes: nothing is ignored silently, and the newest likes win',()=>{
  const session=profile(73);
  const legacy=buildRecommendRequest(session,{limits:LEGACY_REQUEST_LIMITS});
  assert.equal(legacy.body.liked_ids.length,50);
  // The most recent likes are kept. The previous build sent the oldest 50, so
  // the 51st like and every later one never changed a recommendation.
  assert.equal(legacy.body.liked_ids[0],24);
  assert.equal(legacy.body.liked_ids.at(-1),73);
  assert.deepEqual(legacy.accounting,{likedTotal:73,likedSent:50,likedDropped:23,excludedTotal:23,excludedSent:23,excludedDropped:0,titlesDropped:0});
  // Likes that did not fit must still never be recommended.
  for(let id=1;id<=23;id++)assert.ok(legacy.body.excluded_ids.includes(id),`liked ${id} must be excluded`);

  const limits=resolveLimits({ranking:{request_limits:{liked_ids:500,excluded_ids:2000,excluded_titles:500,seen_titles:500,results:50}}});
  const modern=buildRecommendRequest(session,{limits});
  assert.equal(modern.body.liked_ids.length,73);
  assert.equal(modern.accounting.likedDropped,0);
  assert.deepEqual(modern.body.excluded_ids,[]);
  assert.equal(modern.body.session_profile.liked_titles.length,50,'the title echo stays bounded');

  const liked=toggleLiked(session,anime(900));
  assert.equal(buildRecommendRequest(liked,{limits:LEGACY_REQUEST_LIMITS}).body.liked_ids.at(-1),900,'a new like always reaches the model');
});

test('more than 100 shown titles: truncation is reported and feedback outranks history',()=>{
  let session=toggleWatchlist(profile(3),anime(1000));
  session=markNotInterested(session,anime(1001));
  const shownIds=Array.from({length:180},(_,i)=>100+i);
  const legacy=buildRecommendRequest(session,{shownIds,skippedIds:[1002],limits:LEGACY_REQUEST_LIMITS});
  assert.equal(legacy.body.excluded_ids.length,100);
  assert.deepEqual(legacy.body.excluded_ids.slice(0,3),[1001,1000,1002]);
  assert.equal(legacy.body.excluded_ids[3],279,'newest shown title first');
  assert.equal(legacy.accounting.excludedTotal,183);
  assert.equal(legacy.accounting.excludedDropped,83);

  const limits=resolveLimits({ranking:{request_limits:{excluded_ids:2000}}});
  const modern=buildRecommendRequest(session,{shownIds,skippedIds:[1002],limits});
  assert.equal(modern.body.excluded_ids.length,183);
  assert.equal(modern.accounting.excludedDropped,0);
});

test('request limits come from the backend when advertised, else the safe legacy values',()=>{
  assert.deepEqual(resolveLimits(null),LEGACY_REQUEST_LIMITS);
  assert.deepEqual(resolveLimits({ranking:{}}),LEGACY_REQUEST_LIMITS);
  const mixed=resolveLimits({ranking:{request_limits:{liked_ids:500,excluded_ids:'many',results:-1,unknown:9}}});
  assert.equal(mixed.liked_ids,500);
  assert.equal(mixed.excluded_ids,LEGACY_REQUEST_LIMITS.excluded_ids);
  assert.equal(mixed.results,LEGACY_REQUEST_LIMITS.results);
  assert.equal(mixed.unknown,undefined);
});

test('profile sync sends only fields the API accepts, so a restored backup cannot break it',async()=>{
  const session=normalizeSession({sessionId:'s',liked:[anime(1)],watchlist:[anime(2)],preferences:{custom_flag:'keep',seen_titles:['Seen'],temporary_ratings:{Good:9,Bad:'x',Huge:42}}},()=> 'unused');
  let body;
  await withFetch(async(url,options)=>{assert.equal(url,'/api/session/preferences');body=JSON.parse(options.body);return ok({profile:{}});},()=>syncPreferences(session));
  assert.equal(body.replace,true);
  assert.equal(body.session_id,'s');
  assert.equal(body.custom_flag,undefined);
  assert.equal(body.watchlist,undefined);
  assert.deepEqual(body.liked_titles,['Anime 1']);
  assert.deepEqual(body.seen_titles,['Seen']);
  assert.deepEqual(body.temporary_ratings,{Good:9});
  const allowed=new Set(['session_id','replace','liked_titles','disliked_titles','seen_titles','excluded_titles','preferred_genres','excluded_genres','previous_reference_titles','temporary_ratings']);
  assert.deepEqual(Object.keys(body).filter(key=>!allowed.has(key)),[]);
});

test('profile sync is serialised and the latest profile always wins',async()=>{
  const sent=[];const gates=[];
  const send=(session)=>new Promise(resolve=>{sent.push(session.tag);gates.push(resolve);});
  const sync=createPreferenceSync(send);
  const first=sync.push({tag:'A'});
  sync.push({tag:'B'});
  const last=sync.push({tag:'C'});
  assert.deepEqual(sent,['A'],'only one request is in flight');
  gates[0]();
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(sent,['A','C'],'the superseded profile B is never sent');
  gates[1]();
  await Promise.all([first,last,sync.idle()]);
  assert.deepEqual(sent,['A','C']);
});

test('a failed sync does not block later ones',async()=>{
  let calls=0;
  const sync=createPreferenceSync(async()=>{calls+=1;if(calls===1)throw new Error('offline');});
  await sync.push({tag:'A'});
  await sync.push({tag:'B'});
  assert.equal(calls,2);
});

test('API errors surface the server message, and rate limiting gets plain wording',async()=>{
  await withFetch(async()=>({ok:false,status:422,json:async()=>({error:{code:'validation_error',message:'Request validation failed'}})}),async()=>{
    await assert.rejects(api('/api/recommend'),error=>error.status===422&&error.code==='validation_error'&&error.message==='Request validation failed');
  });
  await withFetch(async()=>({ok:false,status:429,json:async()=>({error:{code:'rate_limit_exceeded',message:'Too many requests'}})}),async()=>{
    await assert.rejects(api('/api/recommend'),error=>error.status===429&&/wait a moment/.test(error.message));
  });
});
