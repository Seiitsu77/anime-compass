import test from 'node:test';
import assert from 'node:assert/strict';
import { createRecommendationFeed } from '../src/lib/recommendationFeed.js';
import { BATCH_SIZE } from '../src/lib/recommendationPool.js';
import { LEGACY_REQUEST_LIMITS, buildRecommendRequest } from '../src/lib/api.js';
import { markNotInterested, normalizeSession, toggleLiked, toggleWatchlist } from '../src/lib/session.js';

const MODERN_LIMITS={liked_ids:500,excluded_ids:2000,excluded_titles:500,seen_titles:500,results:50};
const anime=(id)=>({id,title:`Anime ${id}`});
const profile=(likedIds=[])=>normalizeSession({sessionId:'s',liked:likedIds.map(anime)},()=> 'unused');
const ids=(items)=>items.map(item=>item.id);
const tick=()=>new Promise(resolve=>setImmediate(resolve));

/** A ranked catalog behind the real request builder, so limits are exercised too. */
function fakeServer({total=400,limits=MODERN_LIMITS,honorExclusions=true,rank=(session)=>(session.liked.length%2?-1:1)}={}) {
  const calls=[];
  const fetchPage=async(session,{shownIds,skippedIds,limit})=>{
    const {body,accounting}=buildRecommendRequest(session,{shownIds,skippedIds,limit,limits});
    calls.push(body);
    const blocked=new Set(honorExclusions?[...body.liked_ids,...body.excluded_ids]:[]);
    const direction=rank(session);
    const ranked=Array.from({length:total},(_,i)=>anime(direction>0?i+1:total-i));
    return {results:ranked.filter(item=>!blocked.has(item.id)).slice(0,body.limit),accounting};
  };
  return {fetchPage,calls};
}
/** A fetch whose responses are released by hand, to order them deliberately. */
function manualServer() {
  const pending=[];
  const fetchPage=(session,options)=>new Promise((resolve,reject)=>pending.push({session,options,resolve,reject}));
  return {fetchPage,pending};
}
const page=(from,count=50)=>({results:Array.from({length:count},(_,i)=>anime(from+i))});

test('first batch is 18 unique titles and never includes anything saved or rejected',async()=>{
  let session=profile([2,4]);
  session=toggleWatchlist(session,anime(1));
  session=markNotInterested(session,anime(3));
  const feed=createRecommendationFeed({fetchPage:fakeServer({honorExclusions:false}).fetchPage});
  const result=await feed.start(session);
  assert.equal(result.applied,true);
  assert.equal(result.visible.length,BATCH_SIZE);
  assert.equal(new Set(ids(result.visible)).size,BATCH_SIZE);
  for(const blocked of [1,2,3,4])assert.ok(!ids(result.visible).includes(blocked));
});

test('Next Batch never repeats a title, across refills and past 100 shown',async()=>{
  const server=fakeServer();
  const session=profile([2,4]);
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const seen=[...ids((await feed.start(session)).visible)];
  for(let batch=0;batch<12;batch++) {
    const result=await feed.nextBatch(session);
    assert.equal(result.applied,true);
    assert.equal(result.visible.length,BATCH_SIZE);
    for(const id of ids(result.visible))assert.ok(!seen.includes(id),`title ${id} was shown twice`);
    seen.push(...ids(result.visible));
  }
  assert.equal(seen.length,13*BATCH_SIZE);
  assert.equal(new Set(seen).size,seen.length);
  assert.ok(seen.length>100);
  assert.ok(server.calls.at(-1).excluded_ids.length>100,'every shown title is sent as an exclusion');
});

test('a server that ignores exclusions still cannot cause a repeat',async()=>{
  const feed=createRecommendationFeed({fetchPage:fakeServer({total:60,honorExclusions:false}).fetchPage});
  const session=profile([]);
  const seen=[...ids((await feed.start(session)).visible)];
  let last;
  for(let batch=0;batch<6;batch++) {
    last=await feed.nextBatch(session);
    if(!last.changed)break;
    for(const id of ids(last.visible))assert.ok(!seen.includes(id));
    seen.push(...ids(last.visible));
  }
  // The pool ran dry: the feed says so rather than cycling back to shown titles.
  assert.equal(last.changed,false);
  assert.equal(last.exhausted,true);
  assert.equal(new Set(seen).size,seen.length);
  assert.ok(seen.length<=50);
});

test('with the legacy 100-exclusion limit the shortfall is reported and nothing repeats',async()=>{
  const server=fakeServer({limits:LEGACY_REQUEST_LIMITS});
  const session=profile([2,4]);
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const seen=[...ids((await feed.start(session)).visible)];
  let reported=0;
  for(let batch=0;batch<14;batch++) {
    const result=await feed.nextBatch(session);
    if(!result.changed)break;
    for(const id of ids(result.visible))assert.ok(!seen.includes(id));
    seen.push(...ids(result.visible));
    reported=Math.max(reported,result.accounting?.excludedDropped||0);
  }
  assert.equal(new Set(seen).size,seen.length);
  assert.ok(seen.length>100);
  assert.ok(reported>0,'dropped exclusions are surfaced, not hidden');
});

test('Replace skips in place, records no negative feedback, and the title stays away',async()=>{
  const server=fakeServer();
  const session=profile([2,4]);
  const before=JSON.stringify(session);
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const first=await feed.start(session);
  const target=first.visible[5];
  const result=await feed.skip(session,target);
  assert.equal(result.changed,true);
  assert.equal(result.visible.length,BATCH_SIZE);
  assert.notEqual(result.visible[5].id,target.id);
  assert.equal(result.replacement.id,result.visible[5].id,'the new title takes the same slot');
  assert.deepEqual(ids(result.visible).filter((_,i)=>i!==5),ids(first.visible).filter((_,i)=>i!==5));
  // The profile is untouched: Skip is not a dislike and is not persisted.
  assert.equal(JSON.stringify(session),before);
  assert.deepEqual(session.preferences.disliked_titles,[]);
  assert.deepEqual(session.notInterested,[]);

  const later=new Set(ids(result.visible));
  for(let batch=0;batch<4;batch++)for(const id of ids((await feed.nextBatch(session)).visible))later.add(id);
  assert.ok(!later.has(target.id),'a skipped title must not return later in the sequence');
  assert.ok(server.calls.at(-1).excluded_ids.includes(target.id),'sent as an exclusion for this sequence');

  // A profile change starts a new sequence; the skipped title still does not come straight back.
  const liked=toggleLiked(session,anime(399));
  const restarted=await feed.start(liked);
  assert.ok(!ids(restarted.visible).includes(target.id));
  assert.ok(server.calls.at(-1).excluded_ids.includes(target.id));
  assert.deepEqual(liked.preferences.disliked_titles,[]);
});

test('a skipped title stays hidden across a profile change even if the server ignores exclusions',async()=>{
  // Same ranking for every profile, and no exclusion handling: only the feed can keep it out.
  const server=fakeServer({honorExclusions:false,rank:()=>1});
  const session=profile([390]);
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const target=(await feed.start(session)).visible[2];
  await feed.skip(session,target);
  const restarted=await feed.start(toggleLiked(session,anime(391)));
  assert.equal(restarted.visible.length,BATCH_SIZE);
  assert.ok(!ids(restarted.visible).includes(target.id));
  feed.unskip(target);
  assert.ok(ids((await feed.start(session)).visible).includes(target.id),'unskip makes it eligible again');
});

test('a skipped title returns after a reload, because nothing was persisted',async()=>{
  const session=profile([2,4]);
  const feed=createRecommendationFeed({fetchPage:fakeServer().fetchPage});
  const target=(await feed.start(session)).visible[0];
  await feed.skip(session,target);
  // A reload builds a new feed from the stored session.
  const reloaded=createRecommendationFeed({fetchPage:fakeServer().fetchPage});
  const restored=normalizeSession(JSON.parse(JSON.stringify(session)),()=> 'unused');
  assert.ok(ids((await reloaded.start(restored)).visible).includes(target.id));
});

test('Not Interested and Want to Watch fill the slot in place without a recompute',async()=>{
  const server=fakeServer();
  let session=profile([2,4]);
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const first=await feed.start(session);
  const requests=server.calls.length;

  const rejected=first.visible[0];
  session=markNotInterested(session,rejected);
  const afterReject=await feed.drop(session,rejected);
  assert.equal(afterReject.visible.length,BATCH_SIZE);
  assert.ok(!ids(afterReject.visible).includes(rejected.id));

  const saved=afterReject.visible[3];
  session=toggleWatchlist(session,saved);
  const afterSave=await feed.drop(session,saved);
  assert.equal(afterSave.visible.length,BATCH_SIZE);
  assert.ok(!ids(afterSave.visible).includes(saved.id));
  assert.equal(server.calls.length,requests,'both came from the reserve: no extra request');
  assert.equal(new Set(ids(afterSave.visible)).size,BATCH_SIZE);

  // Unlike Skip, Not Interested survives a reload.
  const reloaded=createRecommendationFeed({fetchPage:fakeServer().fetchPage});
  const restored=normalizeSession(JSON.parse(JSON.stringify(session)),()=> 'unused');
  const again=await reloaded.start(restored);
  assert.ok(!ids(again.visible).includes(rejected.id));
  assert.ok(!ids(again.visible).includes(saved.id));
});

test('a stale profile response never overwrites the newest one',async()=>{
  const server=manualServer();
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const older=feed.start(profile([1]));
  const newer=feed.start(profile([1,2]));
  // The newer request answers first; the older one arrives late.
  server.pending[1].resolve(page(500));
  const applied=await newer;
  server.pending[0].resolve(page(100));
  const late=await older;
  assert.equal(applied.applied,true);
  assert.equal(late.applied,false);
  assert.deepEqual(feed.state().visibleIds,ids(page(500).results).slice(0,BATCH_SIZE));
});

test('a Next Batch that was in flight when the profile changed is discarded',async()=>{
  const server=manualServer();
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const session=profile([1]);
  const started=feed.start(session);
  server.pending[0].resolve(page(100,20));
  await started;
  const batch=feed.nextBatch(session);
  await tick();
  assert.equal(server.pending.length,2,'the reserve was too small, so a refill is in flight');
  const liked=toggleLiked(session,anime(7));
  const restarted=feed.start(liked);
  server.pending[2].resolve(page(700));
  assert.equal((await restarted).applied,true);
  server.pending[1].resolve(page(300));
  assert.equal((await batch).applied,false);
  assert.deepEqual(feed.state().visibleIds,ids(page(700).results).slice(0,BATCH_SIZE));
  assert.ok(!feed.state().shownIds.some(id=>id>=300&&id<400),'nothing from the stale response was recorded as shown');
});

test('rapid repeated clicks are applied in order with no duplicates and no lost slot',async()=>{
  const session=profile([2,4]);
  const feed=createRecommendationFeed({fetchPage:fakeServer().fetchPage});
  const first=await feed.start(session);
  const targets=first.visible.slice(0,6);
  // Six different cards, plus a double click on the first, fired without waiting.
  const results=await Promise.all([...targets.map(item=>feed.skip(session,item)),feed.skip(session,targets[0])]);
  assert.ok(results.every(result=>result.applied));
  assert.equal(results.at(-1).changed,false,'the double click is a no-op');
  const final=feed.state().visibleIds;
  assert.equal(final.length,BATCH_SIZE);
  assert.equal(new Set(final).size,BATCH_SIZE);
  for(const item of targets)assert.ok(!final.includes(item.id));
  assert.equal(new Set(results.slice(0,6).map(result=>result.replacement.id)).size,6);
});

test('rapid clicks that outrun the reserve refill once per need and stay unique',async()=>{
  const session=profile([2,4]);
  const server=fakeServer();
  const feed=createRecommendationFeed({fetchPage:server.fetchPage,poolSize:20});
  let visible=(await feed.start(session)).visible;
  const everShown=new Set(ids(visible));
  for(let round=0;round<3;round++) {
    const results=await Promise.all(visible.slice(0,5).map(item=>feed.skip(session,item)));
    visible=results.at(-1).visible;
    for(const result of results) {
      assert.ok(!everShown.has(result.replacement.id),'a replacement must be new to this sequence');
      everShown.add(result.replacement.id);
    }
    assert.equal(new Set(ids(visible)).size,BATCH_SIZE);
  }
});

test('actions are ignored while a new profile is loading, then work again',async()=>{
  const server=manualServer();
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const session=profile([1]);
  const started=feed.start(session);
  assert.equal((await feed.nextBatch(session)).ignored,true);
  server.pending[0].resolve(page(100));
  await started;
  assert.equal((await feed.nextBatch(session)).changed,true);
});

test('an aborted or failed load neither applies nor locks the feed',async()=>{
  const server=manualServer();
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const session=profile([1]);
  const controller=new AbortController();
  const aborted=feed.start(session,{signal:controller.signal});
  controller.abort();
  server.pending[0].reject(Object.assign(new Error('aborted'),{name:'AbortError'}));
  assert.equal((await aborted).applied,false);

  const failed=feed.start(session);
  server.pending[1].reject(new Error('HTTP 500'));
  const failure=await failed;
  assert.equal(failure.applied,true);
  assert.equal(failure.error.message,'HTTP 500');

  const recovered=feed.start(session);
  server.pending[2].resolve(page(100));
  assert.equal((await recovered).visible.length,BATCH_SIZE);
  assert.equal(feed.state().resetting,false);
});

test('a failed refill keeps the current batch and reports the error',async()=>{
  let fail=false;
  const server=fakeServer();
  const feed=createRecommendationFeed({fetchPage:(session,options)=>fail?Promise.reject(new Error('offline')):server.fetchPage(session,{...options,limit:20})});
  const session=profile([2,4]);
  const first=await feed.start(session);
  fail=true;
  const result=await feed.nextBatch(session);
  assert.equal(result.changed,false);
  assert.equal(result.error.message,'offline');
  assert.deepEqual(ids(result.visible),ids(first.visible));
});

test('73 likes and 150 shown titles together: every like is sent and nothing repeats',async()=>{
  const likedIds=Array.from({length:73},(_,i)=>300+i);
  const session=profile(likedIds);
  const server=fakeServer();
  const feed=createRecommendationFeed({fetchPage:server.fetchPage});
  const seen=[...ids((await feed.start(session)).visible)];
  assert.equal(server.calls[0].liked_ids.length,73);
  for(let batch=0;batch<8;batch++)seen.push(...ids((await feed.nextBatch(session)).visible));
  assert.ok(seen.length>=150);
  assert.equal(new Set(seen).size,seen.length);
  assert.ok(!seen.some(id=>likedIds.includes(id)),'a liked title was recommended');
  assert.ok(server.calls.every(call=>call.liked_ids.length===73));
});
