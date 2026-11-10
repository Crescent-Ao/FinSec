import test from 'node:test';
import assert from 'node:assert/strict';
import {runValidatedAudit} from './agent_restart.mjs';

const validate=async text=>{const value=JSON.parse(text);if(!Array.isArray(value.findings))throw new Error('Expected findings object');return value;};
test('Both arms receive an explicit object schema and bounded public-format repair',async()=>{
  for(const mode of ['model','pi']){
    let calls=0;
    const r=await runValidatedAudit({mode,prompt:'public source only',generate:async messages=>{
      assert.match(messages[0].content,/\{"findings":\[/);
      return calls++===0?'[]':'{"findings":[]}';
    },executeTool:async()=>({}),validate});
    assert.equal(r.result.model_calls,2);assert.equal(r.repairs,1);assert.equal(r.firstPassValid,false);assert.equal(r.formatError,null);
  }
});
test('Repeated invalid answers consume at most four model requests',async()=>{
  let calls=0;
  const r=await runValidatedAudit({prompt:'public source',mode:'model',generate:async()=>{calls++;return 'bad';},executeTool:async()=>({}),validate});
  assert.equal(calls,4);assert.equal(r.result.model_calls,4);assert.ok(r.formatError);
});
test('Pi executes a genuine model-requested tool and finalizes using the same local model',async()=>{
  let calls=0,tools=0;
  const r=await runValidatedAudit({prompt:'public source',mode:'pi',generate:async()=>calls++===0?'{"tool":"static_scan","arguments":{}}':'{"findings":[]}',
    executeTool:async()=>{tools++;return {candidates:[]};},validate});
  assert.equal(tools,1);assert.equal(r.result.model_calls,2);assert.equal(r.formatError,null);
});
test('The last permitted Pi round explicitly requires a final answer',async()=>{
  let calls=0;
  const r=await runValidatedAudit({prompt:'public source',mode:'pi',maxTurns:2,generate:async messages=>{
    if(calls++===0)return '{"tool":"read_source","arguments":{}}';
    assert.match(messages[0].content,/最后一次模型调用/);return '{"findings":[]}';
  },executeTool:async()=>({source:'public'}),validate});
  assert.equal(r.result.model_calls,2);assert.equal(r.formatError,null);
});
