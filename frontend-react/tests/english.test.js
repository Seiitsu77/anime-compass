import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { S } from '../src/lib/strings.js';

const root=join(dirname(fileURLToPath(import.meta.url)),'..');
function walk(directory) {
  return readdirSync(directory).flatMap(name=>{
    const path=join(directory,name);
    return statSync(path).isDirectory()?walk(path):[path];
  });
}
const sources=[...walk(join(root,'src')),join(root,'index.html')].filter(path=>/\.(jsx?|css|html)$/.test(path));
const read=(path)=>readFileSync(path,'utf8');
const name=(path)=>relative(root,path).replaceAll('\\','/');

// Any script that is not Latin: CJK, kana, hangul, full-width forms, Cyrillic, and so on.
const NON_LATIN_SCRIPT=/[\u0370-\u1FFF\u2E80-\u9FFF\uA000-\uD7FF\uF900-\uFAFF\uFE30-\uFE4F\uFF00-\uFFEF]/u;
// Typographic punctuation and symbols the English copy is allowed to use.
const ALLOWED_SYMBOLS=/[…·–—“”‘’★✦]/gu;

function leaves(value,path,out=[]) {
  if(typeof value==='string')out.push([path,value]);
  else if(typeof value==='function') {
    // Exercise both grammatical numbers; the arguments are sample data, not copy.
    for(const args of [[1,2,3],[2,5,7],['Sample Title','detail']])leaves(value(...args),`${path}()`,out);
  } else if(Array.isArray(value))value.forEach((entry,index)=>leaves(entry,`${path}[${index}]`,out));
  else if(value&&typeof value==='object')for(const [key,entry] of Object.entries(value))leaves(entry,path?`${path}.${key}`:key,out);
  return out;
}
const copy=leaves(S,'');

test('no source file contains non-English script',()=>{
  for(const path of sources) {
    const match=read(path).match(NON_LATIN_SCRIPT);
    assert.equal(match,null,`${name(path)} contains non-English text: ${match?.[0]}`);
  }
});

test('every string in the localization module is English',()=>{
  assert.ok(copy.length>150,'the module should hold the whole interface');
  for(const [path,text] of copy) {
    assert.ok(text.trim().length>0,`${path} is empty`);
    const stripped=text.replace(ALLOWED_SYMBOLS,'');
    assert.ok(/^[\x20-\x7E]*$/.test(stripped),`${path} is not plain English: ${text}`);
    assert.ok(!/undefined|NaN|\[object/.test(text),`${path} renders a placeholder: ${text}`);
  }
});

test('the agreed terminology is used exactly',()=>{
  assert.equal(S.actions.replace,'Replace');
  assert.equal(S.forYou.nextBatch,'Next Batch');
  assert.equal(S.actions.like,'Like');
  assert.equal(S.actions.liked,'Liked');
  assert.equal(S.actions.wantToWatch,'Want to Watch');
  assert.equal(S.actions.notInterested,'Not Interested');
  assert.equal(S.actions.alreadyWatched,'Already Watched');
  assert.match(S.forYou.tabAlreadyWatched(3),/^Already Watched \(3\)$/);
  assert.match(S.actions.replaceHint,/^Skip /);
  assert.equal(S.nav.forYou,'For You');
  assert.equal(S.nav.discover,'Discover');
  assert.equal(S.forYou.collectionTitle,'My Collection');
  assert.equal(S.filters.moreFilters,'More Filters');
  assert.equal(S.discover.showMore,'Show More');
  assert.match(S.forYou.showMore(4),/^Show More/);
  assert.equal(S.forYou.showLess,'Show Less');
});

test('wording matches behaviour: Skip is temporary, Not Interested is lasting',()=>{
  assert.match(S.recommendations.skipped('X'),/not marked Not Interested/);
  assert.match(S.actions.replaceHint,/for now/);
  assert.match(S.recommendations.markedNotInterested('X'),/will not be recommended again/);
  assert.match(S.forYou.emptyWantBody,/not treated as watched or Liked/);
  // Already Watched must say what it is not, so it cannot be read as a rating.
  assert.match(S.actions.alreadyWatchedHint,/not a Like and not Not Interested/);
  assert.match(S.recommendations.markedWatched('X'),/does not count as a Like or as Not Interested/);
  // Replace must never be described as a rejection.
  assert.doesNotMatch(`${S.actions.replaceHint} ${S.actions.replaceLabel('X')}`,/not interested|dislike/i);
});

test('terms are not paraphrased elsewhere in the copy',()=>{
  const banned=[/\bwatchlist\b/i,/favou?rites?\b/i,/\bsaved for later\b/i,/\bswap\b/i,/\bdislike/i,/\bload more\b/i,/\bcollapse\b/i,/\bhide filters\b/i];
  for(const [path,text] of copy) {
    // "favorite" is the hero headline's ordinary English word, not a feature name.
    if(path==='home.headline')continue;
    for(const pattern of banned)assert.doesNotMatch(text,pattern,`${path} paraphrases a fixed term: ${text}`);
  }
});

test('components take their copy from the localization module',()=>{
  const components=sources.filter(path=>path.endsWith('.jsx'));
  assert.ok(components.length>=6);
  for(const path of components) {
    const source=read(path);
    const attribute=source.match(/\b(?:aria-label|title|placeholder|alt)="[^"]*[A-Za-z][^"]*"/);
    assert.equal(attribute,null,`${name(path)} hard-codes an attribute: ${attribute?.[0]}`);
    // Literal words between a tag's ">" and the next "<" are inline copy.
    const text=source.match(/(?<![=-])>[^<>{}=&|?:;()]*[A-Za-z]{2,}[^<>{}=&|?:;()]*</);
    assert.equal(text,null,`${name(path)} hard-codes visible text: ${text?.[0]}`);
  }
});
