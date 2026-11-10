"""Freeze the user's stage-1 distillation, guard evaluation data and family splits."""
import hashlib
import json
import re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

from transformers import AutoTokenizer

from validate_pilot_data import SimilarityIndex,load_fineval,normalized

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'fin_security_competition'
DEST=Path('/mnt/sda/ps/qwen_lora_experiments')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources=['track1_fin_academic/stage1_fin_sft_part1.jsonl','track2_accounting_econ/stage1_fin_sft_part2.jsonl',
             'fineval_benchmark_suite/stage1_fineval_mcq_booster.jsonl']
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B',local_files_only=True)
    official,n=load_fineval(ROOT/'evaluation_data/FinEval/academic');assert n==4661
    local=SimilarityIndex()
    for r in [json.loads(x) for x in (SOURCE/'fineval_benchmark_suite/fineval_sim_test_benchmark.jsonl').read_text().splitlines()]:
        local.add(r['id'],normalized(r['question']+' '.join(r[k] for k in 'ABCD')))
    protected=set()
    for p in SOURCE.rglob('*non_member*.jsonl'):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            if 'messages' in r:protected.add(normalized(' '.join(m['content'] for m in r['messages'] if m['role']=='user')))
    candidates=[];excluded=[];seen={}
    for relative in sources:
        for line_no,line in enumerate((SOURCE/relative).read_text().splitlines(),1):
            original=json.loads(line);meta=original.get('meta',{});messages=original['messages']
            if not messages or messages[-1]['role']!='assistant' or not any(m['role']=='user' for m in messages):
                raise ValueError('Malformed SFT conversation: '+relative)
            if any(m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str) or not m['content'].strip() for m in messages):
                raise ValueError('Invalid SFT message')
            identity=original.get('id') or meta.get('id') or relative.split('/')[0]+':'+str(line_no)
            group=original.get('topic_id') or meta.get('point_id') or re.sub(r'_B\d+_[A-Za-z]+$','',identity)
            user=' '.join(m['content'] for m in messages if m['role']=='user')
            text=normalized(user);tokens=len(tokenizer.apply_chat_template(messages,return_dict=False,enable_thinking=False))
            record={'id':identity,'group_id':relative.split('/')[0]+':'+group,
                    'source_group_id':relative.split('/')[0]+':'+group,'domain':meta.get('subject') or meta.get('category') or original.get('domain','finance'),
                    'messages':messages,'source':{'file':relative,'line':line_no,'sha256':sha(SOURCE/relative)},
                    'review_status':'user_distilled_not_expert_verified','tokens':tokens}
            reason=None;matches=[]
            if text in protected:reason='exact_nonmember_prompt_overlap'
            elif text in seen:reason='duplicate_prompt'
            elif tokens>4000:reason='deferred_over_4000_tokens_no_truncation'
            else:
                matches=official.search(text,protected=True)
                if matches:reason='possible_official_FinEval_overlap'
                else:
                    matches=local.search(text,protected=True)
                    if matches:reason='possible_local_holdout_overlap'
            if reason:
                excluded.append({'id':identity,'source':relative,'reason':reason,'tokens':tokens,'matches':matches})
                continue
            seen[text]=identity;candidates.append(record)
    # Cluster near-duplicate prompts before splitting; do not merely shuffle rows.
    parent=list(range(len(candidates)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(a)]=find(b)
    groups=defaultdict(list)
    for i,r in enumerate(candidates):groups[r['group_id']].append(i)
    for indices in groups.values():
        for i in indices[1:]:union(indices[0],i)
    index=SimilarityIndex()
    for i,r in enumerate(candidates):
        text=normalized(' '.join(m['content'] for m in r['messages'] if m['role']=='user'),True)
        for match in index.search(text,split=str(i),group=str(i)):union(i,int(match['reference']))
        index.add(str(i),text,split=str(i),group=str(i))
    families=defaultdict(list)
    for i,r in enumerate(candidates):families[find(i)].append(r)
    run=DEST/('distilled_finance_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));run.mkdir(parents=True)
    splits=defaultdict(list)
    ordered=sorted(families.values(),key=lambda rr:hashlib.sha256(('42:'+min(r['group_id'] for r in rr)).encode()).hexdigest())
    # Deterministic complete-family holdouts, roughly 10% validation and 10% privacy.
    validation_target=max(1,round(len(candidates)*0.1));privacy_target=validation_target
    for family in ordered:
        split='validation' if len(splits['validation'])<validation_target else 'privacy_nonmembers' if len(splits['privacy_nonmembers'])<privacy_target else 'train'
        gid='distilled_family_'+hashlib.sha256('|'.join(sorted(r['id'] for r in family)).encode()).hexdigest()[:16]
        for r in family:r['group_id']=gid;r['split']=split;splits[split].append(r)
    # Check original source families as well as the derived near-duplicate clusters.
    prior_ids=set();prior_source_groups=set();prior_groups=set()
    for records in splits.values():
        ids={r['id'] for r in records};source_groups={r['source_group_id'] for r in records};clusters={r['group_id'] for r in records}
        assert len(ids)==len(records),'Duplicate source IDs'
        assert not prior_ids.intersection(ids),'Cross-split IDs'
        assert not prior_source_groups.intersection(source_groups),'Cross-split source families'
        assert not prior_groups.intersection(clusters),'Cross-split clusters'
        prior_ids.update(ids);prior_source_groups.update(source_groups);prior_groups.update(clusters)
    manifest={'scope':'first-stage Chinese distillation; source claims not independently expert-verified','seed':42,
              'preparation_script_sha256':sha(Path(__file__).resolve()),
              'source_files':{s:sha(SOURCE/s) for s in sources},'base_model_manifest_sha256':sha(ROOT/'reports/model_manifest.json'),
              'excluded':excluded,'counts_by_exclusion':dict(Counter(x['reason'] for x in excluded)),
              'official_guard_questions':n,'guard_limit':'Lexical checks cannot exclude semantic paraphrases or pretraining contamination.',
              'max_length':4096,'token_filter':4000,'splits':{},'privacy_policy':'Original nonmember/test/canary/safety/agent files excluded. Actual train IDs alone define members.'}
    for split in ('train','validation','privacy_nonmembers'):
        records=sorted(splits[split],key=lambda r:r['id']);assert records
        path=run/(split+'.records.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        entry={'rows':len(records),'ids':[r['id'] for r in records],'groups':sorted({r['group_id'] for r in records}),
               'source_groups':sorted({r['source_group_id'] for r in records}),
               'source_sha256':sha(path),'records_path':str(path),'max_tokens':max(r['tokens'] for r in records),
               'source_counts':dict(Counter(r['source']['file'] for r in records))}
        if split!='privacy_nonmembers':
            training=run/(split+'.jsonl');training.write_text(''.join(json.dumps({'messages':r['messages']},ensure_ascii=False)+'\n' for r in records))
            entry['training_file_sha256']=sha(training)
        manifest['splits'][split]=entry
    (run/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (ROOT/'reports/latest_distilled_run.txt').write_text(str(run)+'\n')
    (ROOT/'reports/distilled_data_quality.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'run':str(run),'splits':{s:e['rows'] for s,e in manifest['splits'].items()},'excluded':manifest['counts_by_exclusion']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
