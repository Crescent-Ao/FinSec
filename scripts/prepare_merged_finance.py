"""Freeze the full previously audited corpus plus a newly uploaded SFT bundle.

Heldouts have priority over training, including old public prompts underlying
uploaded distilled heldouts. New/old near families use the previous 32-perm
MinHash policy. Old inputs are read-only; all derived files live in a new run.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.prepare_full_finance import signature, band_keys, lexical_similarity, digest, write_json, length_bin
from scripts.validate_pilot_data import normalized, load_fineval

NEW = []; BANDS = []; EXACT = {}; TOKENIZER = None; OFFICIAL = None


def initialize():
    global TOKENIZER, OFFICIAL
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    TOKENIZER = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B', local_files_only=True)
    OFFICIAL, count = load_fineval(ROOT/'evaluation_data/FinEval/academic')
    assert count == 4661


def validate_new(batch):
    result = []
    for r in batch:
        messages = r['messages']; text = ' '.join(m['content'] for m in messages)
        if any(re.search(r'<\|[^>]*\|>|\x00', m['content']) for m in messages):
            result.append((None, r['id'], 'embedded_control_token')); continue
        # Explicit think blocks preserve reasoning in Swift and the HF template.
        ids = TOKENIZER.apply_chat_template(messages, tokenize=True, return_dict=False, enable_thinking=False)
        if len(ids) > 4000:
            result.append((None, r['id'], 'over_4000_tokens_deferred_without_truncation')); continue
        if OFFICIAL.search(normalized(text), protected=True):
            result.append((None, r['id'], 'possible_official_FinEval_overlap')); continue
        r['tokens'] = len(ids)
        result.append((r, r['id'], None))
    return result


def matches(prompt):
    masked = re.sub(r'[0-9]+', '#', prompt)
    found = set(EXACT.get(prompt, [])); candidates = set()
    value = signature(masked)
    for band, key in zip(BANDS, band_keys(value)): candidates.update(band.get(key, []))
    current = np.frombuffer(value, dtype='<u4')
    for i in candidates-found:
        if np.count_nonzero(current == NEW[i]['signature']) < 16: continue
        if lexical_similarity(masked, NEW[i]['masked']) >= .85: found.add(i)
    return sorted(found)


def scan_batch(batch):
    return [(identity, group, split, matches(prompt), prompt in EXACT) for identity, group, split, prompt in batch]


def streams(old):
    for split in ('train', 'validation', 'privacy_nonmembers'):
        with (old/(split+'.records.jsonl')).open() as f:
            for line in f: yield split, json.loads(line)


def main():
    global NEW, BANDS, EXACT
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); p.add_argument('--old', type=Path, required=True)
    p.add_argument('--bundle', type=Path, default=ROOT/'data-distill'); p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(); run = a.run.resolve(); old = a.old.resolve(); bundle = a.bundle.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run/'manifest.json').exists(): raise RuntimeError('Refusing to replace a frozen merged corpus')
    import fcntl
    lock = (run/'preparation.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    uploaded_manifest = json.loads((bundle/'manifest.json').read_text())
    for f in uploaded_manifest['files']:
        path = bundle/f['path']; assert path.resolve().is_relative_to(bundle)
        assert path.stat().st_size == f['bytes'] and digest(path) == f['sha256'], str(path)
    original = json.loads((old/'manifest.json').read_text())
    for split, e in original['splits'].items():
        assert digest(old/(split+'.records.jsonl')) == e['source_sha256']
        if split != 'privacy_nonmembers': assert digest(old/(split+'.jsonl')) == e['training_file_sha256']
    inputs = []; known = set(); counts = Counter(); excluded = []
    for filename, split in [('sft_train.jsonl', 'train'), ('sft_holdout.jsonl', 'privacy_nonmembers')]:
        path = bundle/'sft'/filename; checksum = digest(path)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            r = json.loads(line); known.add(re.sub(r'_cot$', '', r['id']))
            messages = [dict(m) for m in r['messages']]
            assert [m['role'] for m in messages] == ['user', 'assistant']
            assert all(isinstance(m['content'], str) and m['content'].strip() for m in messages)
            reason = r.get('reasoning_content', '').strip()
            if reason:
                assert '<think>' not in reason and '</think>' not in reason
                messages[-1]['content'] = '<think>\n'+reason+'\n</think>\n\n'+messages[-1]['content'].strip()
            inputs.append({'id':'upload:'+r['id'], 'source_group_id':'upload:'+re.sub(r'_cot$', '', r['id']),
                           'source_name':'upload_'+r['source'], 'domain':r.get('subject','finance'), 'messages':messages,
                           'fixed_split':split, 'lang':r.get('lang'), 'has_reasoning':bool(reason),
                           'review_status':'user_distillation_teacher_scores_not_expert_verified',
                           'source':{'file':str(path),'line':number,'sha256':checksum}})
    path = bundle/'sft/synthetic_mcq.jsonl'; checksum = digest(path)
    for number, line in enumerate(path.read_text().splitlines(), 1):
        r = json.loads(line)
        if r['id'] in known: counts['raw_synthetic_already_represented_by_SFT_family'] += 1; continue
        if r.get('status') != 'ok' or r.get('verify_match') is not True:
            counts['raw_synthetic_unverified_deferred'] += 1; continue
        answer = r['answer']; assert answer in 'ABCD' and len(answer) == 1
        question = r['question']+'\n'+'\n'.join(k+'. '+r[k] for k in 'ABCD')+'\n请给出正确选项。'
        reason = r.get('reasoning','').strip()
        content = ('<think>\n'+reason+'\n</think>\n\n' if reason else '')+'答案是：'+answer
        inputs.append({'id':'upload:'+r['id'], 'source_group_id':'upload:'+r['id'], 'source_name':'upload_synth',
                       'domain':r['subject'], 'messages':[{'role':'user','content':question},{'role':'assistant','content':content}],
                       'fixed_split':'train','lang':'zh','has_reasoning':bool(reason),
                       'review_status':'teacher_synthesis_independent_resolve_not_expert_verified',
                       'source':{'file':str(path),'line':number,'sha256':checksum}})
    counts['uploaded_stage1_candidates'] = len(inputs)
    # Even rejected heldouts remain protected by their original family ID.
    heldout_ids = {r['source_group_id'] for r in inputs if r['fixed_split'] != 'train'}
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('fork'), initializer=initialize) as pool:
        for result in pool.map(validate_new, [inputs[i:i+32] for i in range(0,len(inputs),32)]):
            for row, identity, reason in result:
                if reason: excluded.append({'id':identity,'reason':reason}); counts[reason] += 1
                else: NEW.append({'record':row,'prompt':normalized(row['messages'][0]['content'])})
    retained_ids={v['record']['id'] for v in NEW}
    for r in inputs:
        if r['fixed_split']=='privacy_nonmembers' and r['id'] not in retained_ids:
            NEW.append({'record':dict(r,tokens=0),'prompt':normalized(r['messages'][0]['content']),'guard_only':True})
    assert NEW
    BANDS = [defaultdict(list) for _ in range(8)]; EXACT = defaultdict(list)
    parent = list(range(len(NEW))); protected = [2 if v['record']['fixed_split']=='privacy_nonmembers' or v['record']['source_group_id'] in heldout_ids else 0 for v in NEW]
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[max(a,b)]=min(a,b); protected[min(a,b)]=max(protected[a],protected[b])
    families = {}; conversations = set(); duplicate_new = set()
    for i,v in enumerate(NEW):
        r=v['record']; group=r['source_group_id']
        if group in families: union(i,families[group])
        else: families[group]=i
        for other in matches(v['prompt']): union(i,other)
        fingerprint=hashlib.sha256(json.dumps(r['messages'],sort_keys=True,ensure_ascii=False).encode()).digest()
        if fingerprint in conversations: duplicate_new.add(i)
        conversations.add(fingerprint)
        v['masked']=re.sub(r'[0-9]+','#',v['prompt']); sig=signature(v['masked']);v['signature']=np.frombuffer(sig,dtype='<u4')
        EXACT[v['prompt']].append(i)
        for band,key in zip(BANDS,band_keys(sig)):band[key].append(i)
    anchors={}; replaced=set(); scanned=0
    def batches():
        batch=[]
        for split,r in streams(old):
            prompt=normalized(' '.join(m['content'] for m in r['messages'] if m['role']=='user'))
            batch.append((r['id'],r['group_id'],split,prompt))
            if len(batch)==128:yield batch;batch=[]
        if batch:yield batch
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('fork')) as pool:
        # Bound prefetch: Executor.map otherwise buffers the entire 1.8GB ledger.
        iterator=iter(batches());pending=[]
        for _ in range(a.workers*2):
            batch=next(iterator,None)
            if batch:pending.append(pool.submit(scan_batch,batch))
        while pending:
            for identity,group,split,found,exact in pending.pop(0).result():
                scanned+=1
                if found:
                    first=found[0]
                    for other in found[1:]:union(first,other)
                    if group in anchors:union(first,anchors[group])
                    else:anchors[group]=first
                    root=find(first);protected[root]=max(protected[root],{'train':0,'validation':1,'privacy_nonmembers':2}[split])
                    if exact and split=='train':replaced.add(identity)
            batch=next(iterator,None)
            if batch:pending.append(pool.submit(scan_batch,batch))
            if scanned%12800==0:
                write_json(run/'preparation_progress.json',{'scanned_old_rows':scanned,'linked_old_families':len(anchors),'uploaded_candidates':len(NEW)})
                print('cross-corpus checked',scanned,flush=True)
    entries={s:{'rows':0,'ids':[],'groups':set(),'source_groups':set(),'source_counts':Counter(),'tokens':0,'max_tokens':0} for s in ('train','validation','privacy_nonmembers')}
    records={s:(run/(s+'.records.jsonl')).open('w') for s in entries}
    training={s:(run/(s+'.jsonl')).open('w') for s in ('train','validation')}
    strata={s:defaultdict(list) for s in ('train','privacy_nonmembers')};longest=None
    def output(r,split,group):
        nonlocal longest
        r=dict(r,split=split,group_id=group);e=entries[split]
        records[split].write(json.dumps(r,ensure_ascii=False)+'\n')
        if split in training:training[split].write(json.dumps({'messages':r['messages']},ensure_ascii=False)+'\n')
        e['rows']+=1;e['ids'].append(r['id']);e['groups'].add(group);e['source_groups'].add(r['source_group_id'])
        e['source_counts'][r['source_name']]+=1;e['tokens']+=r['tokens'];e['max_tokens']=max(e['max_tokens'],r['tokens'])
        if split in strata:strata[split][(r['source_name'],length_bin(r['tokens']))].append(r['id'])
        if split=='train' and (longest is None or r['tokens']>longest['tokens']):longest=r
    def assignment(i):
        root=find(i)
        return ('train','validation','privacy_nonmembers')[protected[root]],'merged_family_'+hashlib.sha256(NEW[root]['record']['id'].encode()).hexdigest()[:24]
    for original_split,r in streams(old):
        split,group=assignment(anchors[r['group_id']]) if r['group_id'] in anchors else (original_split,r['group_id'])
        if original_split=='train' and split!='train':counts['old_training_rows_moved_to_protected_holdout']+=1
        if r['id'] in replaced:
            counts['old_exact_prompt_replaced_by_uploaded_distillation']+=1;continue
        output(r,split,group)
    for i,v in enumerate(NEW):
        if v.get('guard_only'):continue
        if i in duplicate_new:counts['exact_duplicate_uploaded_conversation']+=1;continue
        split,group=assignment(i);output(v['record'],split,group)
    for handle in list(records.values())+list(training.values()):handle.close()
    seen=set();groups=set()
    for split,e in entries.items():
        assert len(set(e['ids']))==e['rows'] and not seen.intersection(e['ids'])
        assert not groups.intersection(e['groups']),'Cross-split merged family'
        seen.update(e['ids']);groups.update(e['groups']);e['groups']=sorted(e['groups']);e['source_groups']=sorted(e['source_groups']);e['source_counts']=dict(e['source_counts'])
        e.update(source_sha256=digest(run/(split+'.records.jsonl')),records_path=str(run/(split+'.records.jsonl')))
        if split in training:e['training_file_sha256']=digest(run/(split+'.jsonl'))
    member_ids=[];nonmember_ids=[]
    order=lambda value:hashlib.sha256(('mia42:'+value).encode()).digest()
    for key,private in sorted(strata['privacy_nonmembers'].items()):
        members=strata['train'].get(key,[]);size=min(len(members),len(private))
        member_ids.extend(sorted(members,key=order)[:size]);nonmember_ids.extend(sorted(private,key=order)[:size])
    assert len(member_ids)>=1000
    write_json(run/'mia_selection.json',{'member_ids':member_ids,'nonmember_ids':nonmember_ids,'matched_on':'source and token length bins; seed42'})
    (run/'longest_probe.jsonl').write_text(json.dumps({'messages':longest['messages']},ensure_ascii=False)+'\n')
    write_json(run/'rejections.json',excluded)
    manifest={'data_scale':'full_filtered','scope':'Full previously eligible corpus plus all eligible uploaded stage1/SFT/synthetic families, without row caps or oversampling.',
              'seed':42,'max_length':4096,'token_filter':4000,'splits':entries,'counts':dict(counts),'official_guard_questions':4661,
              'old_manifest':str(old/'manifest.json'),'old_manifest_sha256':digest(old/'manifest.json'),
              'uploaded_manifest':str(bundle/'manifest.json'),'uploaded_manifest_sha256':digest(bundle/'manifest.json'),
              'preparation_script_sha256':digest(Path(__file__)),'privacy_selection_sha256':digest(run/'mia_selection.json'),
              'privacy_policy':'All previous heldouts and uploaded heldouts stay out of training; near families assigned to the most protected split.',
              'duplicate_policy':'Replace old exact user prompts with uploaded distillation; retain distinct direct/reasoning variants inside one family; remove identical conversations.',
              'reasoning_policy':'Explicit canonical think blocks in assistant content; no reasoning field silently dropped.',
              'excluded_components':['sft_all duplicate aggregate','safety stage2','Agent and benchmark files'],
              'limitations':['Teacher quality scores are not expert verification.','MinHash lexical matching is approximate and cannot rule out semantic or pretraining contamination.','No true source-document IDs exist for all public records.']}
    write_json(run/'manifest.json',manifest)
    for name in ['manifest.json','mia_selection.json','train.jsonl','validation.jsonl','train.records.jsonl','validation.records.jsonl','privacy_nonmembers.records.jsonl']:(run/name).chmod(0o444)
    print(json.dumps({'splits':{s:{k:e[k] for k in ('rows','tokens','source_counts')} for s,e in entries.items()},'counts':dict(counts),'privacy_pairs':len(member_ids)},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
