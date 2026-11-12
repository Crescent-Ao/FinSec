import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {runValidatedAudit} from '../pi_crypto_agent/agent_described_4096.mjs';

const root=path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const opts=Object.fromEntries(Array.from({length:(process.argv.length-2)/2},(_,i)=>[process.argv[i*2+2].replace(/^--/,''),process.argv[i*2+3]]));
for(const n of ['file','model-dir','python','output'])if(!opts[n])throw new Error(`Missing --${n}`);
const code=fs.readFileSync(opts.file,'utf8');if(Buffer.byteLength(code)>65536)throw new Error('Source exceeds64KiB');
fs.mkdirSync(opts.output,{recursive:true});
const spec={path:opts['model-dir'],precision:'nf4',system_prompt:''};if(opts.adapter)spec.adapter=opts.adapter;
const specPath=path.join(opts.output,'spec.json');fs.writeFileSync(specPath,JSON.stringify(spec,null,2));
const knowledge=JSON.parse(fs.readFileSync(opts.knowledge ?? path.join(root,'knowledge/crypto_rules.json'),'utf8'));
const publicCase={id:'live-'+Date.now(),file:path.basename(opts.file),code,knowledge};
const worker=spawn(opts.python,['-u',path.join(root,'pi_crypto_agent/native_worker_described_4096.py'),'--spec',specPath],
  {cwd:root,stdio:['pipe','pipe','inherit'],env:{...process.env,HF_HUB_OFFLINE:'1',TOKENIZERS_PARALLELISM:'false',OMP_NUM_THREADS:'4'}});
let next=0;const pending=new Map();
readline.createInterface({input:worker.stdout}).on('line',line=>{
  try{const x=JSON.parse(line),p=pending.get(x.id);if(!p)return;pending.delete(x.id);x.error?p.reject(new Error(x.error)):p.resolve(x.result);}
  catch(e){for(const p of pending.values())p.reject(e);pending.clear();}
});
worker.on('exit',code=>{for(const p of pending.values())p.reject(new Error(`Local worker exited ${code}`));pending.clear();});
worker.on('error',e=>{for(const p of pending.values())p.reject(e);pending.clear();});
const rpc=data=>new Promise((resolve,reject)=>{const id=++next;pending.set(id,{resolve,reject});worker.stdin.write(JSON.stringify({...data,id})+'\n');});
const events=[];
try{
 const identity=await rpc({op:'identity'});
 const prepared=[];
 for(const name of ['static_scan','lookup_rules'])prepared.push({kind:'framework_tool',name,result:await rpc({op:'tool',name,arguments:{},public:publicCase})});
 const prompt=JSON.stringify({file:publicCase.file,source_with_lines:code.split('\n').map((s,i)=>`${i+1}: ${s}`).join('\n'),local_crypto_rules:knowledge,framework_preparation:prepared});
 const start=Date.now();
 const audit=await runValidatedAudit({mode:'pi',prompt,maxTurns:Number(opts['max-turns']??12),
  generate:async(messages,max_new_tokens,tools)=>{const answer=await rpc({op:'generate',messages,max_new_tokens,tools});events.push({kind:'model',raw:answer.text});console.log(`Model round ${events.filter(e=>e.kind==='model').length}`);return answer.text;},
  executeTool:async(name,arguments_)=>{const result=await rpc({op:'tool',name,arguments:arguments_,public:publicCase});events.push({kind:'tool',name,arguments:arguments_,result});console.log('Tool executed:',name);return result;},
  validate:text=>rpc({op:'validate',text,public:publicCase})});
 const report={model:opts.adapter?path.basename(opts.adapter):'base_nf4',precision:'nf4',model_identity:identity,file:publicCase.file,input_source:code,findings:audit.findings,
  format_error:audit.formatError,framework_preparation:prepared,elapsed_seconds:(Date.now()-start)/1000,
  result:audit.result,events,external_model_calls:0};
 fs.writeFileSync(path.join(opts.output,'audit.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify({file:report.file,findings:report.findings,format_error:report.format_error,record:path.join(opts.output,'audit.json')},null,2));
}finally{worker.stdin.end();worker.kill();}
