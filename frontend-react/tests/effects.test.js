import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root=join(dirname(fileURLToPath(import.meta.url)),'..');
const walk=(directory)=>readdirSync(directory).flatMap(name=>{
  const path=join(directory,name);
  return statSync(path).isDirectory()?walk(path):[path];
});

test('no effect returns the value of an expression',()=>{
  // `useEffect(()=>node.scrollIntoView())` returns whatever the call returns. In
  // current Chromium that is a Promise, React treats it as a clean-up, and the
  // whole app unmounts with "destroy is not a function" (the Agent page did).
  for(const path of walk(join(root,'src')).filter(file=>file.endsWith('.jsx'))) {
    const match=readFileSync(path,'utf8').match(/useEffect\(\s*(?:async\s*)?\([^)]*\)\s*=>\s*[^{\s][^\n]{0,60}/);
    assert.equal(match,null,`${relative(root,path)} has an effect with an implicit return: ${match?.[0]}`);
  }
});
