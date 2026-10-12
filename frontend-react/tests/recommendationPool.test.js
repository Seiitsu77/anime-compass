import test from 'node:test';
import assert from 'node:assert/strict';
import { BATCH_SIZE,POOL_SIZE,uniqueEligible,takeFromPool } from '../src/lib/recommendationPool.js';

test('up to 18 unique results, even if the server returns a liked anime in the top pool',()=>{
  const data=Array.from({length:50},(_,i)=>({id:i+1,title:`Anime ${i+1}`}));
  const eligible=uniqueEligible([data[0],...data,data[1]],new Set([1,4]));
  assert.equal(POOL_SIZE,50);
  assert.equal(BATCH_SIZE,18);
  const first=eligible.slice(0,BATCH_SIZE);
  assert.equal(first.length,18);
  assert.ok(first.every(x=>x.id!==1&&x.id!==4));
  assert.equal(new Set(first.map(x=>x.id)).size,18);
});

test('replace from reserve never repeats currently shown or previously surfaced titles',()=>{
  const initial=[{id:3,title:'Already shown'},{id:4,title:'Forbidden'},{id:5,title:'Available'}];
  const result=takeFromPool(initial,[{id:1,title:'Visible'}],new Set([3]),new Set([4]),1);
  assert.deepEqual(result.chosen.map(x=>x.id),[5]);
  assert.deepEqual(result.rest.map(x=>x.id),[3,4]);
});

test('next batches do not repeat each other',()=>{
  const data=Array.from({length:50},(_,i)=>({id:i+1,title:`Anime ${i+1}`}));
  const first=data.slice(0,BATCH_SIZE);
  const seen=new Set(first.map(x=>x.id));
  const second=uniqueEligible(data,new Set(),seen).slice(0,BATCH_SIZE);
  assert.equal(second.length,BATCH_SIZE);
  assert.equal(new Set([...first,...second].map(x=>x.id)).size,2*BATCH_SIZE);
});
