import test from 'node:test';
import assert from 'node:assert/strict';
import {parseNativeAction,wireMessages,runValidatedAudit} from './agent_described_4096.mjs';

test('Qwen XML parameters preserve multiline Python and typed arrays',()=>{
  const call=parseNativeAction('<tool_call>\n<function=run_python_test>\n<parameter=test_code>\nassert 2+3==5\nprint("ok")\n</parameter>\n</function>\n</tool_call>');
  assert.equal(call.name,'run_python_test');assert.equal(call.arguments.test_code,'assert 2+3==5\nprint("ok")');
  const rules=parseNativeAction('<tool_call><function=lookup_rules><parameter=rule_ids>["AES_ECB"]</parameter></function></tool_call>');
  assert.deepEqual(rules.arguments.rule_ids,['AES_ECB']);
});
test('Native assistant calls and role=tool results survive serialization',()=>{
  const messages=wireMessages({systemPrompt:'test',messages:[{role:'assistant',content:[{type:'toolCall',id:'x',name:'run_command',arguments:{command:'echo ok'}}]},
    {role:'toolResult',toolCallId:'x',toolName:'run_command',content:[{type:'text',text:'actual result'}]}]});
  assert.equal(messages[1].tool_calls[0].function.name,'run_command');assert.equal(messages[2].role,'tool');assert.equal(messages[2].content,'actual result');
});
test('Pi actually receives full tool schemas in the generation callback',async()=>{
  let calls=0;
  const result=await runValidatedAudit({mode:'pi',prompt:'demo',generate:async(messages,_tokens,tools)=>{
    assert.ok(tools.some(t=>t.function.name==='run_command'));
    if(calls++===0)return '<tool_call><function=run_command><parameter=command>echo ok</parameter></function></tool_call>';
    assert.equal(messages.at(-1).role,'tool');return '{"findings":[]}';
  },executeTool:async()=>({exit_code:0,stdout:'ok'}),validate:async text=>JSON.parse(text)});
  assert.equal(result.formatError,null);assert.equal(result.result.model_calls,2);
});
test('Plain model receives no schemas and no tool instructions',async()=>{
  await runValidatedAudit({mode:'model',prompt:'demo',generate:async(messages,_tokens,tools)=>{
    assert.equal(tools,undefined);assert.doesNotMatch(messages[0].content,/run_command|run_python_test/);return '{"findings":[]}';
  },executeTool:async()=>{throw new Error('No tools');},validate:async text=>JSON.parse(text)});
});
