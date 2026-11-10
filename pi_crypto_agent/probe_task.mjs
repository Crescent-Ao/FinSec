import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {spawn} from 'node:child_process';
import {runValidatedAudit} from './agent_native.mjs';

const root=path.resolve(import.meta.dirname,'..');
const output=path.join(root,'outputs/security_hardening_20261109_v1/Pi_actual_task_probe');
fs.mkdirSync(output,{recursive:true});
const process_=spawn(path.join(root,'.venv-eval/bin/python'),['-u',path.join(root,'scripts/probe_pi_native_bridge.py')],{cwd:root,stdio:['pipe','pipe','inherit']});
let next=0;const pending=new Map();
readline.createInterface({input:process_.stdout}).on('line',line=>{
  const d=JSON.parse(line),p=pending.get(d.id);pending.delete(d.id);if(p)d.error?p.reject(new Error(d.error)):p.resolve(d.result);
});
process_.on('exit',code=>{for(const p of pending.values())p.reject(new Error(`Bridge exited${code}`));pending.clear();});
const rpc=data=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});process_.stdin.write(JSON.stringify({...data,id})+'\n');});
const knowledge=JSON.parse(fs.readFileSync(path.join(root,'evaluation_data/stage2_agent_holdout_20261107_v1/knowledge.json'),'utf8'));
const cases=[
 {id:'independent_ECB_probe',file:'payment.py',code:'from Crypto.Cipher import AES\n\ndef encrypt_account(payload: bytes, key: bytes):\n    return AES.new(key, AES.MODE_ECB).encrypt(payload)\n'},
 {id:'independent_GCM_probe',file:'payment_safe.py',code:'from Crypto.Cipher import AES\n\ndef encrypt_account(payload: bytes, key: bytes):\n    cipher = AES.new(key, AES.MODE_GCM)\n    ciphertext, tag = cipher.encrypt_and_digest(payload)\n    return cipher.nonce, ciphertext, tag\n'},
];
const results=[];
try{
 const identity=await rpc({op:'identity'});
 for(const c of cases){
  const publicCase={...c,knowledge};
  const prompt='请审计以下生产支付账户数据保护模块的密码学误用。本次任务要求：自主选择工具，编写并实际执行至少一项与代码行为相关的测试，查看真实输出后给出审计结论。不要只解释如何测试，也不要伪造执行结果。测试的具体方法由你选择。最终只返回规定的findings对象；测试过程记录于工具轨迹。\n'+JSON.stringify(publicCase);
  const start=Date.now();
  const audit=await runValidatedAudit({mode:'pi',prompt,maxTurns:12,maxTokens:512,
   generate:async(messages,max_new_tokens,tools)=>(await rpc({op:'generate',messages,max_new_tokens,tools})).text,
   executeTool:(name,arguments_)=>rpc({op:'tool',name,arguments:arguments_,public:publicCase}),
   validate:text=>rpc({op:'validate',text,public:publicCase})});
  const tools=audit.result.trace.filter(x=>x.kind==='tool');
  const executions=tools.filter(x=>['run_command','run_python_test'].includes(x.name));
  const record={case:c,model_identity:identity,...audit,seconds:(Date.now()-start)/1000,
   model_requested_tools:tools.map(x=>x.name),actual_execution_calls:executions.length,
   paid_API_calls:0,scope:'Independent benign application probe; execution requested but no tool name/test/answer supplied to model'};
  results.push(record);fs.writeFileSync(path.join(output,c.id+'.json'),JSON.stringify(record,null,2)+'\n');
  console.log(JSON.stringify({case:c.id,tools:record.model_requested_tools,executions:executions.length,format_error:audit.formatError,findings:audit.findings}));
 }
 fs.writeFileSync(path.join(output,'result.json'),JSON.stringify({models:'safetySFT41 NF4',cases:results,paid_API_calls:0},null,2)+'\n');
}finally{process_.stdin.end();process_.kill();}
