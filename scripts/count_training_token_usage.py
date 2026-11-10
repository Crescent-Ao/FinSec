"""CPU-only accounting of frozen training data and successful GRPO-lineage logs.

Dataset serialization counts are not advertised as GPU FLOPs or all-run totals.
"""
import copy
import datetime
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fineval_pipeline.core import digest,rows,write_json
from scripts.validate_stage2_delivery import native_messages

REPORT=ROOT/'reports/training_token_usage_20261109'
SOURCE=REPORT/'source'


def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True)
    def encoded(row,generation=False):
        return len(tokenizer.apply_chat_template(native_messages(row['messages']),tools=row.get('tools') or None,
            return_dict=False,enable_thinking=False,add_generation_prompt=generation))
    datasets=SOURCE/'datasets'
    stages={}
    for stage,filename in [('original_safety_sft','sft_train.jsonl'),('original_safety_sft_long','sft_long_train.jsonl'),
        ('supplemental_safety_sft41',str(ROOT/'outputs/security_hardening_20261109_v1/phase3_corrected_ingestion/data/sft_train.jsonl'))]:
        path=Path(filename) if filename.startswith('/') else datasets/filename
        data=rows(path);lengths=[encoded(row) for row in data]
        stages[stage]={'rows':len(data),'canonical_full_sequence_tokens':sum(lengths),'maximum_length':max(lengths),
            'source':str(path),'sha256':digest(path)}
        print(stage,stages[stage]['canonical_full_sequence_tokens'],flush=True)
    data=rows(datasets/'dpo_train.jsonl');chosen=[];rejected=[];prompts=[]
    for row in data:
        chosen.append(encoded(row))
        alternative=copy.deepcopy(row)
        alternative['messages'][-1]={'role':'assistant','content':row['rejected_response']}
        rejected.append(encoded(alternative))
        prompt=copy.deepcopy(row);prompt['messages']=prompt['messages'][:-1]
        prompts.append(encoded(prompt,True))
    stages['dpo557']={'preference_pairs':len(data),'chosen_full_tokens':sum(chosen),'rejected_full_tokens':sum(rejected),
        'canonical_both_branches_tokens':sum(chosen)+sum(rejected),'prompt_tokens_once':sum(prompts),
        'counts_prompt_in_each_branch':True,'reference_forward_tokens_not_added':True,
        'source':str(datasets/'dpo_train.jsonl'),'sha256':digest(datasets/'dpo_train.jsonl')}
    data=rows(datasets/'grpo_train.jsonl');prompt_tokens=sum(encoded(row,True) for row in data)
    stages['grpo']={'unique_training_prompts':len(data),'canonical_prompt_tokens_once':prompt_tokens,
        'nominal_num_generations':4,'canonical_prompt_tokens_x4':prompt_tokens*4,
        'private_reference_answers_counted_in_prompt':False,'source':str(datasets/'grpo_train.jsonl'),
        'sha256':digest(datasets/'grpo_train.jsonl')}
    histories={};logged=[];physical_discarded=[]
    for label,keep in [('grpo_old',lambda step:step<=60),('grpo_resumed',lambda step:step>=61)]:
        directory=SOURCE/label
        for record in rows(directory/'logging.jsonl'):
            key=record.get('global_step/max_steps')
            if not key:continue
            step=int(key.split('/')[0])
            if keep(step):histories[step]=record
            else:physical_discarded.append({'step':step,'mean_completion_length':record.get('completions/mean_length')})
        for record in rows(directory/'completions.jsonl'):
            step=int(record['step'][0])
            if keep(step):
                if len(record['completion'])!=len(record['prompt']):raise ValueError('Incomplete logged rollout record')
                logged.extend(zip([step]*len(record['completion']),record['prompt'],record['completion']))
    if set(histories)!=set(range(1,110)):raise ValueError('GRPO successful-lineage steps1..109 are not complete')
    logged_prompt=sum(len(tokenizer.encode(p,add_special_tokens=False)) for _,p,_ in logged)
    logged_completion=sum(len(tokenizer.encode(c,add_special_tokens=False)) for _,_,c in logged)
    estimate_full=sum(float(h['completions/mean_length'])*48 for h in histories.values())
    estimate_partial=estimate_full-24*float(histories[109]['completions/mean_length'])
    stages['grpo'].update(optimizer_steps=109,effective_batch=48,logged_completion_records=len(logged),
        logged_prompt_text_tokens=logged_prompt,logged_completion_text_tokens=logged_completion,
        log_scope='Swift deque retains24 of up to48 rollouts per optimizer step; these are not all generated tokens',
        successful_lineage_completion_token_estimate_lower=estimate_partial,
        successful_lineage_completion_token_estimate_upper=estimate_full,
        estimate_assumptions='mean completion-mask length x48 per full update; final update24..48 completions. Not an exact token meter.',
        discarded_unsaved_steps61_to69=len(physical_discarded),
        discarded_run_and_replayed_steps_not_in_effective_lineage=True)
    # Published frozen-record count from the completed stage1 full-data run.
    stages['finance_sft']={'rows':547247,'canonical_full_sequence_tokens':296760476,'epochs':1,
        'source':'reports/stage1_financial_finetuning_report.md; merged_full_ddp3 frozen token metadata',
        'sampler_padding_extra_rows':1,'sampler_padding_extra_tokens_not_in_this_sum':True}
    result={'updated_china':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'model':'Qwen3.5-9B native tokenizer, thinking disabled','stages':stages,
        'scope':'Frozen selected-stage data per one epoch, including oversampled rows present in each frozen file; canonical serialization, not padded device execution.',
        'excluded':['Failed capacity pilots, discarded financial parameter studies, invalid-reward GRPO runs',
            'Teacher/API generation tokens, attack/evaluation tokens, reference-policy extra forwards/backward FLOPs',
            'Padding and empty-thinking/scaffolding differences in actual Swift encoding'],
        'tokenizer_files':{p.name:digest(p) for p in (ROOT/'models/Qwen3.5-9B').iterdir() if p.name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']}}
    write_json(REPORT/'token_usage.json',result)
    print('ACCOUNTING_SAVED',str(REPORT/'token_usage.json'),flush=True)


if __name__=='__main__':main()
