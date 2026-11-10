import test from 'node:test';
import assert from 'node:assert/strict';
import {runAudit} from './agent.mjs';

test('Pi executes the model-requested tool and returns its result to the local model',async()=>{
  let calls=0;
  const result=await runAudit({prompt:'sample',generate:async messages=>{
    if(calls++===0)return '{"tool":"search_code","arguments":{"query":"md5"}}';
    assert.match(messages.at(-1).content,/matched-source/);return '{"findings":[]}';
  },executeTool:async(name,args)=>{assert.equal(name,'search_code');assert.equal(args.query,'md5');return {matches:['matched-source']};}});
  assert.equal(result.error,null);assert.equal(result.model_calls,2);assert.equal(result.trace.filter(x=>x.kind==='tool').length,1);
});

test('Tool recursion stops at the shared model call budget',async()=>{
  const result=await runAudit({prompt:'sample',maxTurns:2,generate:async()=>'{"tool":"read_source","arguments":{}}',executeTool:async()=>({code:'safe'})});
  assert.equal(result.model_calls,2);assert.match(result.error,/BUDGET_EXHAUSTED/);
});

test('Unknown tools cannot invoke the tool bridge',async()=>{
  let calls=0,executed=false;
  const result=await runAudit({prompt:'sample',generate:async()=>calls++===0?'{"tool":"shell","arguments":{"cmd":"anything"}}':'{"findings":[]}',executeTool:async()=>{executed=true;return {};}});
  assert.equal(executed,false);assert.equal(result.error,null);
});

test('Invalid tool arguments are handled by Pi before execution',async()=>{
  let calls=0,executed=false;
  const result=await runAudit({prompt:'sample',generate:async()=>calls++===0?'{"tool":"search_code","arguments":{}}':'{"findings":[]}',executeTool:async()=>{executed=true;return {};}});
  assert.equal(executed,false);assert.equal(result.error,null);
});
