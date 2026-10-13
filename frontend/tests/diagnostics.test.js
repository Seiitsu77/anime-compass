import test from 'node:test';
import assert from 'node:assert/strict';
import { compareRecommendations, createDiagnosticsLog, summarizeRecommendation } from '../src/lib/diagnostics.js';

const response=(resultIds,{positives,vector,alsTop}={})=>({
  results:resultIds.map(id=>({id,title:`Anime ${id}`})),
  timing_ms:12.5,
  diagnostics:{
    recommendation_path:'fast',collaborative_route:'als',known_positive_count:positives,
    candidate_pool_size:300,learned_reranker_applied:true,excluded_count:positives,
    ...(vector?{debug:{als_user_vector_sha256_16:vector,als_user_vector_norm:1.5,als_retrieved_top_ids:alsTop,reranked_top_ids:resultIds}}:{}),
  },
});

test('production builds record nothing and expose no global',()=>{
  const target={};
  const messages=[];
  const log=createDiagnosticsLog(false,{sink:{debug:(...args)=>messages.push(args)},target});
  assert.equal(log.record('profile',response([1,2,3],{positives:2,vector:'aaaa'})),null);
  assert.equal(log.entries.length,0);
  assert.equal(messages.length,0);
  assert.equal('__animeCompassDiagnostics' in target,false);
});

test('a Like shows up as a changed profile, user vector, candidates and ranking',()=>{
  const target={};
  const messages=[];
  const log=createDiagnosticsLog(true,{sink:{debug:(...args)=>messages.push(args)},target});
  log.record('profile',response([1,2,3,4],{positives:2,vector:'aaaa1111',alsTop:[1,2,3,4]}),{likedCount:2});
  // A refill between two profile loads must not become the comparison baseline.
  assert.equal(log.record('refill',response([9,8,7],{positives:2,vector:'aaaa1111',alsTop:[9,8,7]})).change,null);
  const after=log.record('profile',response([3,5,6,1],{positives:3,vector:'bbbb2222',alsTop:[3,5,6,1]}),{likedCount:3});

  assert.deepEqual(after.change,{
    profileSizeChanged:true,userVectorChanged:true,alsCandidatesChanged:true,rankingChanged:true,
    overlap:0.5,entered:[5,6],left:[2,4],
  });
  assert.equal(after.summary.path,'fast');
  assert.equal(after.summary.rerankerApplied,true);
  assert.equal(target.__animeCompassDiagnostics,log.entries);
  assert.equal(messages.length,3);
});

test('without the server debug block the vector comparison is unknown, not false',()=>{
  const before=summarizeRecommendation(response([1,2,3],{positives:2}));
  const after=summarizeRecommendation(response([1,2,3],{positives:2}));
  const change=compareRecommendations(before,after);
  assert.equal(change.userVectorChanged,null);
  assert.equal(change.alsCandidatesChanged,null);
  assert.equal(change.rankingChanged,false);
  assert.equal(change.overlap,1);
});
