import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import readline from 'node:readline';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { runValidatedAudit } from './agent_native_4096.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.dirname(here);
const args = Object.fromEntries(Array.from({length:(process.argv.length-2)/2}, (_,i)=>[process.argv[2+2*i].replace(/^--/,''),process.argv[3+2*i]]));
for (const name of ['spec','cases','knowledge','output']) if (!args[name]) throw new Error(`Missing --${name}`);
const output = path.resolve(args.output); fs.mkdirSync(output, {recursive:true});
const hash = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const json = p => JSON.parse(fs.readFileSync(p,'utf8'));
const rows = p => fs.readFileSync(p,'utf8').trim().split('\n').filter(Boolean).map(s=>JSON.parse(s));
const write = (name,value) => { const p=path.join(output,name);fs.writeFileSync(p+'.tmp',JSON.stringify(value,null,2)+'\n');fs.renameSync(p+'.tmp',p); };
const cases = rows(args.cases), knowledge = json(args.knowledge);
if (cases.length !== 82 || new Set(cases.map(c=>c.id)).size !== 82) throw new Error('Require the frozen82 unique cases');
const worker = spawn(args.python ?? path.join(root,'.venv-eval/bin/python'), ['-u',path.join(here,'native_worker_4096.py'),'--spec',path.resolve(args.spec)],
  {cwd:root,stdio:['pipe','pipe','inherit'],env:{...process.env,HF_HUB_OFFLINE:'1',TOKENIZERS_PARALLELISM:'false',OMP_NUM_THREADS:'4'}});
let seq=0; const pending = new Map();
const lines = readline.createInterface({input:worker.stdout});
lines.on('line',line=>{
  try { const d=JSON.parse(line), p=pending.get(d.id);if (!p) throw new Error('Unknown bridge response');pending.delete(d.id);d.error?p.reject(new Error(d.error)):p.resolve(d.result); }
  catch(e) { for(const p of pending.values())p.reject(e);pending.clear(); }
});
worker.on('exit',code=>{for(const p of pending.values())p.reject(new Error(`Native worker exited ${code}`));pending.clear();});
const rpc = data => new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});worker.stdin.write(JSON.stringify({...data,id})+'\n');});
let identity;
try {
  identity = await rpc({op:'identity'});
  const plan={protocol:'pi-crypto-native-tools-4096output-v6',model_identity:identity,cases_sha256:hash(args.cases),knowledge_sha256:hash(args.knowledge),
    cases:82,modes:['pi'],max_model_calls:12,max_tokens_per_call:4096,greedy:true,external_model_calls:0,
    dependency_lock_sha256:hash(path.join(here,'package-lock.json')),
    source_hashes:Object.fromEntries(['agent_native_4096.mjs','native_worker_4096.py','evaluate_native_4096_pi_only.mjs','sandbox.py','workspace_tools.py'].map(n=>[n,hash(path.join(here,n))])),
    comparison_scope:'Pi-only evaluation: up to12 calls,4096 output tokens per call,16384 context. Model baselines are reused historical1024-output results, with protocol/call-budget differences recorded in baseline_reuse_manifest.json. This is not an equal-budget framework ablation; initial static/rule preparation is recorded separately.',
    corpus_scope:'Frozen authored/synthetic file-level holdout; not independently expert-certified or a real-project benchmark'};
  const fingerprint=crypto.createHash('sha256').update(JSON.stringify(plan)).digest('hex');
  const savedPlan=path.join(output,'plan.json');
  if(fs.existsSync(savedPlan)&&JSON.stringify(json(savedPlan))!==JSON.stringify({...plan,fingerprint}))throw new Error('Cannot resume changed model/data/code/protocol');
  write('plan.json',{...plan,fingerprint});
  for(const mode of plan.modes){
    const recordPath=path.join(output,`${mode}.predictions.jsonl`);
    const saved=fs.existsSync(recordPath)?rows(recordPath):[];
    const done=new Map(saved.map(r=>[r.id,r]));
    if(done.size!==saved.length||saved.some(r=>r.fingerprint!==fingerprint||!cases.some(c=>c.id===r.id)))throw new Error('Invalid cached predictions');
    for(const c of cases){
      if(done.has(c.id))continue;
      const publicCase={id:c.id,file:c.file,code:c.code,knowledge};
      const preparation=[];
      if(mode==='pi') for(const name of ['static_scan','lookup_rules']) preparation.push({kind:'framework_tool',name,result:await rpc({op:'tool',name,arguments:{},public:publicCase})});
      const prompt=JSON.stringify({file:c.file,source_with_lines:c.code.split('\n').map((s,i)=>`${i+1}: ${s}`).join('\n'),local_crypto_rules:knowledge,framework_preparation:preparation});
      const start=Date.now();let result,findings=[],formatError=null,firstPassValid=false,repairs=0;
      try {
        const audit=await runValidatedAudit({mode,prompt,maxTurns:12,
          generate:async(messages,max_new_tokens,tools)=>(await rpc({op:'generate',messages,max_new_tokens,tools})).text,
          executeTool:(name,arguments_)=>rpc({op:'tool',name,arguments:arguments_,public:publicCase}),
          validate:text=>rpc({op:'validate',text,public:publicCase})});
        ({result,findings,formatError,firstPassValid,repairs}=audit);
        if(result.error)formatError=result.error;
      } catch(e){formatError=String(e);}
      const record={id:c.id,fingerprint,findings,format_valid:formatError===null,format_error:formatError,
        first_pass_format_valid:firstPassValid,format_repairs:repairs,framework_preparation:preparation,
        elapsed_seconds:(Date.now()-start)/1000,result:result??null};
      fs.appendFileSync(recordPath,JSON.stringify(record)+'\n');done.set(c.id,record);
      write('status.json',{state:'evaluating',mode,completed:done.size,total:cases.length,external_model_calls:0,updated:new Date().toISOString()});
    }
    let tp=0,fp=0,fn=0,exact=0,invalid=0,seconds=0,calls=0,toolCalls=0,benign=0,benignFP=0,firstValid=0,repairs=0,frameworkCalls=0;
    const key=f=>JSON.stringify([f.rule_id,f.line]);
    for(const c of cases){
      const r=done.get(c.id),truth=new Set(c.expected.map(key)),found=new Set(r.findings.map(key));
      const hits=[...found].filter(k=>truth.has(k)).length;
      tp+=hits;fp+=found.size-hits;fn+=truth.size-hits;
      exact+=Number(r.format_valid&&hits===truth.size&&hits===found.size);invalid+=Number(!r.format_valid);
      if(truth.size===0){benign++;benignFP+=Number(found.size>0);}
      firstValid+=Number(r.first_pass_format_valid);repairs+=r.format_repairs??0;frameworkCalls+=r.framework_preparation?.length??0;
      seconds+=r.elapsed_seconds;calls+=r.result?.model_calls??0;toolCalls+=r.result?.trace.filter(x=>x.kind==='tool').length??0;
    }
    const precision=tp/(tp+fp)||0,recall=tp/(tp+fn)||0;
    write(`${mode}.summary.json`,{model_identity:identity,mode,cases:82,true_positive:tp,false_positive:fp,false_negative:fn,
      precision,recall,f1:2*precision*recall/(precision+recall)||0,case_exact_match:exact/82,format_failures:invalid,
      first_pass_format_valid_rate:firstValid/82,format_repairs:repairs,framework_tool_calls:frameworkCalls,format_valid_rate:1-invalid/82,benign_cases:benign,benign_false_positive_cases:benignFP,
      benign_false_positive_rate:benign?benignFP/benign:0,total_seconds:seconds,model_calls:calls,tool_calls:toolCalls,external_model_calls:0,fingerprint});
  }
  write('status.json',{state:'completed',cases:82,modes:['pi'],model_identity:identity,external_model_calls:0,updated:new Date().toISOString()});
}catch(e){write('status.json',{state:'failed',reason:String(e),external_model_calls:0});throw e;}
finally{worker.stdin.end();worker.kill();}
