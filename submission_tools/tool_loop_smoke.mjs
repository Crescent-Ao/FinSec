// A scripted response sequence verifies real Pi wiring and actual sandbox execution.
// This is an integration check, not a new model-quality measurement.
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import readline from 'node:readline';
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {runValidatedAudit} from '../pi_crypto_agent/agent_described_4096.mjs';
const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const out=path.join(root,'runs/tool_smoke');fs.mkdirSync(out,{recursive:true});
const spec=path.join(out,'unused_model_spec.json');fs.writeFileSync(spec,JSON.stringify({path:'unused-in-tool-check',precision:'nf4'}));
const child=spawn(process.env.FINSEC_PYTHON??'python3.11',['-u',path.join(root,'pi_crypto_agent/native_worker_described_4096.py'),'--spec',spec],
 {cwd:root,stdio:['pipe','pipe','inherit'],env:{...process.env}});
let seq=0;const pending=new Map();readline.createInterface({input:child.stdout}).on('line',line=>{const r=JSON.parse(line),p=pending.get(r.id);pending.delete(r.id);r.error?p.reject(new Error(r.error)):p.resolve(r.result)});
child.on('exit',code=>{for(const p of pending.values())p.reject(new Error('Worker exited '+code));pending.clear()});
const rpc=data=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});child.stdin.write(JSON.stringify({...data,id})+'\n')});
const publicCase={file:'ecb_demo.py',code:'from Crypto.Cipher import AES\ndef encrypt(data):\n    return AES.new(b"k"*16,AES.MODE_ECB).encrypt(data)\n',knowledge:[]};
let calls=0,actual;
try{
 const audit=await runValidatedAudit({mode:'pi',prompt:'Verify the supplied Python implementation.',maxTurns:3,
  generate:async()=>calls++===0?'<tool_call><function=run_python_test><parameter=test_code>from audited_source import encrypt\nc = encrypt(b"a"*32)\nassert c[:16] == c[16:]\nprint("ECB repeated blocks: verified against actual function")</parameter></function></tool_call>':'{"findings":[]}',
  executeTool:async(name,args)=>{actual=await rpc({op:'tool',name,arguments:args,public:publicCase});return actual},
  validate:async text=>JSON.parse(text)});
 assert.equal(audit.result.model_calls,2);assert.equal(actual.exit_code,0);assert.match(actual.stdout,/verified against actual function/);
 fs.writeFileSync(path.join(out,'actual_pi_tool_check.json'),JSON.stringify({scope:'scripted responses + genuine Pi execution, no LLM inference',
  model_inference_performed:false,external_api_calls:0,input:publicCase,tool_return:actual,result:audit.result},null,2));
 console.log('PASS: Pi → native tool bridge → real Python sandbox → actual stdout → Pi result. No model or API loaded.');
}finally{child.stdin.end();child.kill()}
