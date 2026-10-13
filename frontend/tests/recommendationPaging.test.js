import test from 'node:test';
import assert from 'node:assert/strict';
import { createAutoSource, createCursorSource, createExclusionSource, createRecommendationFeed } from '../src/lib/recommendationFeed.js';
import { BATCH_SIZE, SEQUENCE_PAGE_SIZE } from '../src/lib/recommendationPool.js';
import { buildRecommendRequest, buildSequenceRequest, recommendPage, resolvePagination } from '../src/lib/api.js';
import { markNotInterested, normalizeSession, toggleLiked, toggleWatchlist } from '../src/lib/session.js';

const LIMITS={liked_ids:500,excluded_ids:2000,excluded_titles:500,seen_titles:500,results:50};
const anime=(id)=>({id,title:`Anime ${id}`});
const profile=(likedIds=[])=>normalizeSession({sessionId:'s',liked:likedIds.map(anime)},()=> 'unused');
const ids=(items)=>items.map(item=>item.id);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const encode=(value)=>Buffer.from(JSON.stringify(value)).toString('base64url');
const decode=(cursor)=>JSON.parse(Buffer.from(cursor,'base64url').toString());

/**
 * A stand-in for /api/recommend/page with the same contract: the body defines a
 * ranked sequence, the cursor is a position in it, and nothing else is kept.
 * Every call rebuilds the sequence from the body alone, exactly as a freshly
 * restarted instance would have to.
 */
function pagedServer({depth=400,version='v1'}={}) {
  const server={calls:[],version,depth};
  const rank=(body)=>{
    const blocked=new Set([...body.liked_ids,...body.excluded_ids]);
    // Any deterministic order will do; it must depend on the profile and the deployment.
    const shift=(body.liked_ids.length*7+server.version.length*13)%server.depth;
    return Array.from({length:server.depth},(_,i)=>anime(((i+shift)%server.depth)+1)).filter(item=>!blocked.has(item.id));
  };
  const fingerprint=(body)=>`${server.version}|${JSON.stringify(body)}`;
  server.sequenceFor=(body)=>rank(body);
  server.requestPage=async(body,{cursor,limit})=>{
    server.calls.push({body:JSON.parse(JSON.stringify(body)),cursor,limit});
    let offset=0;
    if(cursor) {
      const position=decode(cursor);
      if(position.f!==fingerprint(body))throw Object.assign(new Error('sequence expired'),{status:409,code:'sequence_expired'});
      offset=position.o;
    }
    const sequence=rank(body);
    const results=sequence.slice(offset,offset+limit);
    const end=offset+results.length;
    const hasMore=end<sequence.length;
    return {results,sequence:{offset,returned:results.length,has_more:hasMore,next_cursor:hasMore?encode({f:fingerprint(body),o:end}):null,source:'fast'}};
  };
  return server;
}
function cursorFeed(server,options={}) {
  const source=createCursorSource({
    buildRequest:(session,{skippedIds})=>buildSequenceRequest(session,{skippedIds,limits:LIMITS}),
    requestPage:server.requestPage,
  });
  return createRecommendationFeed({source,...options});
}

test('batches of 18 come from one server sequence and the request never grows',async()=>{
  const server=pagedServer();
  const session=toggleWatchlist(profile([2,4]),anime(50));
  const feed=cursorFeed(server);
  const shown=[...ids((await feed.start(session)).visible)];
  for(let batch=0;batch<11;batch++) {
    const result=await feed.nextBatch(session);
    assert.equal(result.visible.length,BATCH_SIZE);
    shown.push(...ids(result.visible));
  }
  assert.equal(shown.length,12*BATCH_SIZE);
  assert.equal(new Set(shown).size,shown.length,'a title was shown twice');
  // Exactly the server order: nothing reordered, skipped over, or invented.
  assert.deepEqual(shown,ids(server.sequenceFor(server.calls[0].body)).slice(0,shown.length));

  // 216 titles were shown, and no request ever mentions one of them.
  const first=server.calls[0].body;
  for(const call of server.calls) {
    assert.deepEqual(call.body,first,'the profile snapshot is sent unchanged with every page');
    assert.deepEqual(call.body.excluded_ids,[50],'only what the user excluded, never what was shown');
    assert.equal(call.body.shown_ids,undefined);
    assert.equal(call.limit,SEQUENCE_PAGE_SIZE);
  }
  assert.equal(server.calls.length,6,'two batches per request: 216 titles in six requests');
  assert.equal(server.calls[0].cursor,null);
  assert.ok(server.calls.slice(1).every(call=>typeof call.cursor==='string'));
});

test('Replace and Next Batch read the same sequence, so neither can skip or repeat',async()=>{
  const server=pagedServer();
  const session=profile([2,4]);
  const feed=cursorFeed(server);
  const everything=[...ids((await feed.start(session)).visible)];
  const replaced=[];
  for(let round=0;round<4;round++) {
    for(const slot of [0,7,17]) {
      const target=feed.state().visibleIds[slot];
      const result=await feed.skip(session,anime(target));
      assert.equal(result.visible.length,BATCH_SIZE);
      assert.equal(result.visible[slot].id,result.replacement.id,'the replacement takes the vacated slot');
      everything.push(result.replacement.id);
      replaced.push(target);
    }
    everything.push(...ids((await feed.nextBatch(session)).visible));
  }
  const sequence=ids(server.sequenceFor(server.calls[0].body));
  assert.equal(new Set(everything).size,everything.length);
  // Every title surfaced, in the order it surfaced, is a gapless prefix of the sequence.
  assert.deepEqual(everything,sequence.slice(0,everything.length));
  assert.equal(everything.length,5*BATCH_SIZE+12);
  // Skipped titles are behind the cursor; none had to be sent to the server.
  assert.ok(server.calls.every(call=>call.body.excluded_ids.length===0));
  assert.deepEqual(feed.state().skippedIds,replaced);
});

test('the end of the sequence is reported, with a partial last batch and no wrap-around',async()=>{
  const server=pagedServer({depth:47});
  const session=profile([2,4]);
  const feed=cursorFeed(server);
  const first=await feed.start(session);
  const second=await feed.nextBatch(session);
  const third=await feed.nextBatch(session);
  const fourth=await feed.nextBatch(session);
  assert.deepEqual([first.visible.length,second.visible.length,third.visible.length],[18,18,9]);
  assert.equal(third.partial,true);
  assert.equal(third.exhausted,true);
  assert.equal(fourth.changed,false);
  assert.equal(fourth.exhausted,true);
  assert.deepEqual(ids(fourth.visible),ids(third.visible),'the last batch stays on screen');
  const all=[...ids(first.visible),...ids(second.visible),...ids(third.visible)];
  assert.equal(new Set(all).size,45);

  // Replace at the end removes the card instead of repeating a title.
  const gone=await feed.skip(session,third.visible[0]);
  assert.equal(gone.replacement,null);
  assert.equal(gone.visible.length,8);
});

test('a new Like starts a new sequence with a new snapshot',async()=>{
  const server=pagedServer();
  const session=profile([2,4]);
  const feed=cursorFeed(server);
  const first=await feed.start(session);
  await feed.nextBatch(session);
  const skipped=feed.state().visibleIds[3];
  await feed.skip(session,anime(skipped));
  const liked=toggleLiked(session,first.visible[0]);
  const restarted=await feed.start(liked);

  const opening=server.calls.at(-1);
  assert.equal(opening.cursor,null,'the old cursor is not reused');
  assert.deepEqual(opening.body.liked_ids,[2,4,first.visible[0].id]);
  // The skipped title is the only extra exclusion: bounded by clicks, not by paging.
  assert.deepEqual(opening.body.excluded_ids,[skipped]);
  assert.ok(!ids(restarted.visible).includes(skipped));
  assert.ok(!ids(restarted.visible).includes(first.visible[0].id));
  assert.deepEqual(ids(restarted.visible),ids(server.sequenceFor(opening.body)).slice(0,BATCH_SIZE));
});

test('a page for an older profile that arrives late is dropped',async()=>{
  const pending=[];
  const source=createCursorSource({
    buildRequest:(session,{skippedIds})=>buildSequenceRequest(session,{skippedIds,limits:LIMITS}),
    requestPage:(body,options)=>new Promise(resolve=>pending.push({body,options,resolve})),
  });
  const reply=(from,cursor)=>({results:Array.from({length:36},(_,i)=>anime(from+i)),sequence:{has_more:true,next_cursor:cursor}});
  const feed=createRecommendationFeed({source});
  const session=profile([1]);
  const started=feed.start(session);
  pending[0].resolve(reply(100,'c1'));
  await started;
  await feed.nextBatch(session);                       // served from the reserve
  const inFlight=feed.nextBatch(session);              // needs the next page
  await tick();
  assert.equal(pending.length,2);
  assert.equal(pending[1].options.cursor,'c1');
  const restarted=feed.start(toggleLiked(session,anime(7)));
  pending[2].resolve(reply(700,'n1'));
  assert.equal((await restarted).applied,true);
  pending[1].resolve(reply(300,'c2'));
  assert.equal((await inFlight).applied,false);
  assert.deepEqual(feed.state().visibleIds,Array.from({length:18},(_,i)=>700+i));
  assert.equal(feed.state().cursor,'n1','the stale response did not move the new cursor');
  assert.ok(!feed.state().shownIds.some(id=>id>=300&&id<400));
});

test('when a deployment invalidates the cursor the feed restarts without repeating anything',async()=>{
  const server=pagedServer();
  const session=profile([2,4]);
  const feed=cursorFeed(server);
  const shown=[...ids((await feed.start(session)).visible)];
  shown.push(...ids((await feed.nextBatch(session)).visible));
  server.version='v2-new-model';                        // the next page request will be refused with 409
  for(let batch=0;batch<5;batch++) {
    const result=await feed.nextBatch(session);
    assert.equal(result.applied,true);
    assert.equal(result.error,undefined);
    assert.equal(result.visible.length,BATCH_SIZE);
    for(const id of ids(result.visible))assert.ok(!shown.includes(id),`title ${id} was repeated after the restart`);
    shown.push(...ids(result.visible));
  }
  assert.equal(feed.state().restarts,1);
  assert.equal(new Set(shown).size,shown.length);
  assert.ok(server.calls.some(call=>call.cursor===null&&server.calls.indexOf(call)>0),'a new sequence was opened');
});

test('a title blocked elsewhere mid-sequence is filtered and the batch still fills',async()=>{
  const server=pagedServer();
  let session=profile([2,4]);
  const feed=cursorFeed(server);
  await feed.start(session);
  const upcoming=ids(server.sequenceFor(server.calls[0].body)).slice(18,54);
  // Bookmarked and rejected from the Discover page: they were never on screen here.
  for(const id of upcoming.slice(0,10))session=toggleWatchlist(session,anime(id));
  for(const id of upcoming.slice(10,14))session=markNotInterested(session,anime(id));
  const next=await feed.nextBatch(session);
  assert.equal(next.visible.length,BATCH_SIZE);
  for(const id of upcoming.slice(0,14))assert.ok(!ids(next.visible).includes(id));
  // The cursor keeps its place: the snapshot is not rebuilt for a mid-sequence change.
  assert.deepEqual(server.calls.at(-1).body,server.calls[0].body);
});

test('more than 50 Likes and more than 100 shown titles together',async()=>{
  const likedIds=Array.from({length:73},(_,i)=>300+i);
  const server=pagedServer();
  const session=profile(likedIds);
  const feed=cursorFeed(server);
  const shown=[...ids((await feed.start(session)).visible)];
  for(let batch=0;batch<9;batch++)shown.push(...ids((await feed.nextBatch(session)).visible));
  assert.equal(shown.length,180);
  assert.equal(new Set(shown).size,180);
  assert.ok(!shown.some(id=>likedIds.includes(id)));
  assert.ok(server.calls.every(call=>call.body.liked_ids.length===73&&call.body.excluded_ids.length===0));
});

test('rapid Replace clicks across a page boundary stay ordered and unique',async()=>{
  const server=pagedServer();
  const session=profile([2,4]);
  const feed=cursorFeed(server);
  const first=await feed.start(session);
  // The reserve holds 18. Eighteen clicks fired in one tick empty it exactly...
  const emptied=await Promise.all(first.visible.map(item=>feed.skip(session,item)));
  assert.ok(emptied.every(result=>result.applied&&result.replacement));
  assert.equal(server.calls.length,1,'all eighteen came from the reserve');
  // ...and six more, again without waiting, must read the next page once between them.
  const crossing=await Promise.all(feed.state().visibleIds.slice(0,6).map(id=>feed.skip(session,anime(id))));
  assert.ok(crossing.every(result=>result.applied&&result.replacement));
  assert.equal(server.calls.length,2,'one request served all six');
  const surfaced=[...ids(first.visible),...emptied.map(r=>r.replacement.id),...crossing.map(r=>r.replacement.id)];
  assert.deepEqual(surfaced,ids(server.sequenceFor(server.calls[0].body)).slice(0,42),'click order is sequence order');
  assert.equal(new Set(feed.state().visibleIds).size,BATCH_SIZE);
});

test('the request builder separates the sequence snapshot from legacy paging',()=>{
  const session=markNotInterested(toggleWatchlist(profile([1,2]),anime(9)),anime(8));
  const snapshot=buildSequenceRequest(session,{skippedIds:[7],limits:LIMITS});
  assert.deepEqual(snapshot.body.liked_ids,[1,2]);
  assert.deepEqual(snapshot.body.excluded_ids,[8,9,7]);
  assert.equal(snapshot.body.limit,undefined,'the page size belongs to the page request');
  assert.equal(snapshot.body.one_per_series,true);
  // The exclusion fallback still adds shown titles and a limit.
  const legacy=buildRecommendRequest(session,{skippedIds:[7],shownIds:[20,21],limits:LIMITS});
  assert.deepEqual(legacy.body.excluded_ids,[8,9,7,21,20]);
  assert.equal(legacy.body.limit,50);
});

test('recommendPage posts the snapshot with a cursor and a page size',async()=>{
  const original=globalThis.fetch;let seen;
  globalThis.fetch=async(url,options)=>{seen={url,body:JSON.parse(options.body)};return {ok:true,status:200,json:async()=>({results:[],sequence:{has_more:false}})};};
  try {
    await recommendPage({session_id:'s',liked_ids:[1],excluded_ids:[]},{cursor:'abc',limit:36});
    assert.equal(seen.url,'/api/recommend/page');
    assert.deepEqual(seen.body,{session_id:'s',liked_ids:[1],excluded_ids:[],cursor:'abc',limit:36});
    await recommendPage({liked_ids:[]},{limit:18,endpoint:'/api/v2/page'});
    assert.equal(seen.url,'/api/v2/page');
    assert.equal(seen.body.cursor,null);
  } finally {globalThis.fetch=original;}
});

test('cursor paging is used only when the backend advertises it',async()=>{
  assert.equal(resolvePagination(null),null);
  assert.equal(resolvePagination({ranking:{}}),null);
  assert.equal(resolvePagination({ranking:{pagination:{endpoint:'/api/recommend/page',batch_size:0}}}),null);
  assert.deepEqual(resolvePagination({ranking:{pagination:{endpoint:'/api/recommend/page',batch_size:18,window_size:300,max_windows:4}}}),
    {endpoint:'/api/recommend/page',batchSize:18,windowSize:300,maxWindows:4});

  let advertised=false;
  const server=pagedServer();
  const legacyCalls=[];
  const source=createAutoSource({
    available:()=>advertised,
    cursor:createCursorSource({buildRequest:(session,{skippedIds})=>buildSequenceRequest(session,{skippedIds,limits:LIMITS}),requestPage:server.requestPage}),
    exclusion:createExclusionSource({fetchPage:async(session,options)=>{
      legacyCalls.push(options);
      const blocked=new Set([...session.liked.map(x=>x.id),...options.shownIds]);
      return {results:Array.from({length:300},(_,i)=>anime(i+1)).filter(x=>!blocked.has(x.id)).slice(0,options.limit)};
    }}),
  });
  const feed=createRecommendationFeed({source});
  const session=profile([2,4]);

  // An older backend: page by exclusion, as before.
  const old=[...ids((await feed.start(session)).visible)];
  for(let batch=0;batch<4;batch++)old.push(...ids((await feed.nextBatch(session)).visible));
  assert.equal(feed.state().source,'exclusion');
  assert.equal(new Set(old).size,5*BATCH_SIZE);
  assert.ok(legacyCalls.at(-1).shownIds.length>0);
  assert.equal(server.calls.length,0);

  // The backend now advertises paging: the next sequence uses the cursor.
  advertised=true;
  await feed.start(session);
  await feed.nextBatch(session);
  await feed.nextBatch(session);
  assert.equal(feed.state().source,'cursor');
  assert.ok(server.calls.length>=2);
});
