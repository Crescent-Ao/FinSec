"""Audit and freeze every eligible financial SFT record, without a row cap.

CPU workers tokenize and check protected references. The parent clusters numeric
templates and lexical near duplicates before assigning whole-family holdouts.
Classifiers are selection signals, not answer-correctness verifiers.
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sqlite3
import struct
import sys
from array import array
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import xxhash
from datasketch import MinHash
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_pilot_data import SimilarityIndex, arithmetic, load_fineval, normalized
RESOURCE = Path('/mnt/sda/ps/qwen_lora_resources/huggingface')
OFFICIAL = PRIVATE = TOKENIZER = None
RESERVED = re.compile(r'<\|(?:im_start|im_end|endoftext|tool_call|tool_response)\|>')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path); temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def signature(text):
    shingles = {text[i:i+5].encode() for i in range(max(1, len(text)-4))}
    m = MinHash(num_perm=32, seed=42, hashfunc=xxhash.xxh32_intdigest)
    m.update_batch(list(shingles))
    return m.hashvalues.astype('<u4').tobytes()


def band_keys(value):
    return [xxhash.xxh64_intdigest(value[i:i+16], seed=i) for i in range(0, 128, 16)]


def missing_context_reason(prompt):
    if re.search(r'\b(?:attached|uploaded)\s+(?:image|chart|spreadsheet|screenshot)\b', prompt, re.I):
        return 'text_only_record_requires_attachment'
    if re.search(r'\bwhich of the following\b', prompt, re.I):
        choices = re.findall(r'(?:^|[\s(])(?:[A-Da-d]|[1-9]|I{1,3}|IV)[.)\]:]\s*\S', prompt)
        bullets = re.findall(r'(?:^|\n)\s*[-*•]\s+\S', prompt)
        if len(choices)+len(bullets) < 2: return 'multiple_choice_options_not_explicit_deferred'
    return None


def init_worker():
    global TOKENIZER
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    TOKENIZER = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-9B', local_files_only=True)


def process_batch(batch):
    rendered = []
    for row in batch:
        rendered.append(TOKENIZER.apply_chat_template(row['messages'], tokenize=False, enable_thinking=False))
    encoded = TOKENIZER(rendered, add_special_tokens=False, return_attention_mask=False)['input_ids']
    results = []
    for row, ids in zip(batch, encoded):
        reason = None
        messages = row['messages']
        prompt = normalized(' '.join(m['content'] for m in messages if m['role'] == 'user'))
        full = normalized(' '.join(m['content'] for m in messages))
        if len(ids) > 4000: reason = 'over_4000_tokens_deferred_without_truncation'
        elif not prompt or len(ids) < 10: reason = 'empty_or_degenerate_example'
        elif any(RESERVED.search(m['content']) or '\x00' in m['content'] for m in messages): reason = 'embedded_control_token_or_nul'
        elif missing_context_reason(' '.join(m['content'] for m in messages if m['role'] == 'user')):
            reason = missing_context_reason(' '.join(m['content'] for m in messages if m['role'] == 'user'))
        elif OFFICIAL.search(full, protected=True): reason = 'possible_FinEval_overlap'
        elif row.get('fixed_split') == 'train' or not row.get('fixed_split'):
            if PRIVATE.search(full, protected=True) or PRIVATE.search(re.sub(r'[0-9]+', '#', full), protected=True):
                reason = 'protected_holdout_overlap'
        if reason:
            results.append((None, {'id': row['id'], 'source': row['source_name'], 'reason': reason})); continue
        row['tokens'] = len(ids)
        row['review_status'] = 'automated_selection_not_expert_verified'
        masked = re.sub(r'[0-9]+', '#', prompt)
        results.append(({'record': row, 'prompt': prompt, 'masked': masked,
                         'signature': signature(masked)}, None))
    return results


def local_records():
    manifest = json.loads((ROOT/'reports/distilled_data_quality.json').read_text())
    memberships = {i: split for split, entry in manifest['splits'].items() for i in entry['ids']}
    for relative, expected in manifest['source_files'].items():
        path = ROOT/'fin_security_competition'/relative
        if digest(path) != expected: raise ValueError('Local distilled source changed: '+relative)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            r = json.loads(line); meta = r.get('meta', {})
            identity = r.get('id') or meta.get('id') or relative.split('/')[0]+':'+str(line_number)
            split = memberships.get(identity)
            if split is None: continue  # Previously deferred; still over the identical 4K policy.
            if meta.get('answer'):
                found = re.match(r'^\s*([ABCD])', r['messages'][-1]['content'])
                if not found or found[1] != meta['answer']: raise ValueError('MCQ label mismatch: '+identity)
            group = r.get('topic_id') or meta.get('point_id') or re.sub(r'_B\d+_[A-Za-z]+$', '', identity)
            yield {'id': 'distilled:'+identity, 'source_group_id': 'distilled:'+relative.split('/')[0]+':'+group,
                   'source_name': 'distilled', 'domain': meta.get('subject') or r.get('domain', 'finance'),
                   'messages': r['messages'], 'fixed_split': split,
                   'source': {'file': str(path), 'line': line_number, 'sha256': expected}}
    for split in ('train', 'validation', 'privacy_nonmembers'):
        path = ROOT/'data/pilot_luna_v1'/(split+'.candidates.jsonl'); checksum = digest(path)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            r = json.loads(line)
            for check in r.get('checks', []):
                from decimal import Decimal
                expected = Decimal(check['expected'])
                tolerance = min(Decimal(check['tolerance']), Decimal(1).scaleb(expected.as_tuple().exponent)/2)
                if abs(arithmetic(check['expression'])-expected) > tolerance:
                    raise ValueError('Luna arithmetic mismatch: '+r['id'])
            yield {'id': 'luna:'+r['id'], 'source_group_id': 'luna:'+r['group_id'], 'source_name': 'luna',
                   'domain': r['domain'], 'messages': r['messages'], 'fixed_split': split,
                   'source': {'file': str(path), 'line': line_number, 'sha256': checksum}}


def build_guards(locals_):
    global OFFICIAL, PRIVATE
    OFFICIAL, count = load_fineval(ROOT/'evaluation_data/FinEval/academic')
    if count != 4661: raise ValueError('Incomplete official FinEval guard')
    PRIVATE = SimilarityIndex()
    def add(identity, messages):
        prompt = ' '.join(m['content'] for m in messages if m['role'] == 'user')
        PRIVATE.add(identity, normalized(prompt))
        PRIVATE.add(identity+':numeric_template', normalized(prompt, True))
    for r in locals_:
        if r['fixed_split'] != 'train': add(r['id'], r['messages'])
    source = ROOT/'fin_security_competition'
    for path in source.rglob('*non_member*.jsonl'):
        for i, line in enumerate(path.read_text().splitlines()):
            r = json.loads(line)
            if r.get('messages'): add(str(path.relative_to(source))+':'+str(i), r['messages'])
    path = source/'fineval_benchmark_suite/fineval_sim_test_benchmark.jsonl'
    for line in path.read_text().splitlines():
        r = json.loads(line)
        add('local_test:'+r['id'], [{'role': 'user', 'content': r['question']+' '.join(r[k] for k in 'ABCD')}])
    return count


def public_records(counts, rejections, verified, resource_manifest):
    manifest = json.loads(resource_manifest.read_text())
    for name in ('fin-cfa-graphgen', 'fin-glaive'):
        entry = manifest['repositories']['whoisjiji/'+name]
        file = next(f for f in entry['files'] if f['file'].endswith('.jsonl'))
        path = Path(file['file']); h = hashlib.sha256()
        with path.open('rb') as f:
            for line_number, raw in enumerate(f, 1):
                h.update(raw); r = json.loads(raw); counts[name+':raw'] += 1
                identity = name+':'+str(line_number)
                scores = [r.get(k) for k in ('is_fin_score', 'is_selfcont_score', 'is_verif_score')]
                reason = None
                if any(not isinstance(s, (int, float)) or not math.isfinite(s) or not 0 <= s <= 1 for s in scores): reason = 'invalid_classifier_score'
                elif scores[0] <= .99 or scores[1] <= .9: reason = 'financial_or_selfcontainment_below_threshold'
                elif any(not isinstance(r.get(k), str) or not r[k].strip() for k in ('user', 'assistant')): reason = 'missing_prompt_or_answer'
                elif name == 'fin-glaive' and (not isinstance(r.get('reasoning'), str) or not r['reasoning'].strip()): reason = 'missing_released_reasoning'
                if reason:
                    counts[name+':'+reason] += 1
                    rejections.write(json.dumps({'id': identity, 'source': name, 'reason': reason})+'\n'); continue
                answer = r['assistant'].strip()
                if name == 'fin-glaive': answer = 'Reasoning:\n'+r['reasoning'].strip()+'\n\nFinal answer:\n'+answer
                yield {'id': identity, 'source_name': name, 'source_group_id': identity, 'domain': 'international_finance',
                       'messages': [{'role': 'user', 'content': r['user'].strip()}, {'role': 'assistant', 'content': answer}],
                       'scores': dict(zip(('financial', 'selfcontained', 'verifiable'), scores)),
                       'source': {'file': str(path), 'line': line_number, 'sha256': file['sha256'], 'revision': entry['revision']}}
        if h.hexdigest() != file['sha256']: raise ValueError('Full public source checksum mismatch: '+name)
        verified[name] = {'sha256': h.hexdigest(), 'rows': counts[name+':raw'], 'revision': entry['revision']}


def lexical_similarity(a, b):
    aa, bb = SimilarityIndex.shingles(a), SimilarityIndex.shingles(b)
    return len(aa & bb)/max(1, len(aa | bb))


def length_bin(n):
    return next((str(v) for v in (256, 512, 1024, 2048) if n <= v), '4000')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resource-manifest',type=Path,default=ROOT/'reports/finance_resource_downloads.json')
    args = parser.parse_args(); run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    import fcntl
    preparation_lock = (run/'preparation.lock').open('a')
    fcntl.flock(preparation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preparation_sha = digest(Path(__file__))
    if (run/'manifest.json').exists(): raise ValueError('Refusing to overwrite a frozen dataset')
    locals_ = list(local_records()); guard_count = build_guards(locals_)
    counts = Counter(); verified = {}
    db_path = run/'preparation.sqlite'
    if db_path.exists(): db_path.unlink()  # Only this run's incomplete preparation scratch.
    db = sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE records (n INTEGER PRIMARY KEY, record TEXT, masked TEXT, prompt TEXT, valid INTEGER DEFAULT 1)')
    parent = array('I'); bands = [defaultdict(list) for _ in range(8)]
    exact = {}; source_groups = {}; template_groups = {}; fixed = {}; source = {}; tokens = {}; identities = {}; signatures = []
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    def batches():
        batch = []
        from itertools import chain
        for row in chain(locals_, public_records(counts, rejection_file, verified, args.resource_manifest)):
            batch.append(row)
            if len(batch) == 32: yield batch; batch = []
        if batch: yield batch
    def consume(result):
        for value, error in result:
            if error:
                counts[error['source']+':'+error['reason']] += 1; rejection_file.write(json.dumps(error)+'\n'); continue
            r, prompt, masked, sig = value['record'], value['prompt'], value['masked'], value['signature']
            key = hashlib.sha256(prompt.encode()).digest()
            if key in exact:
                old = exact[key]
                previous = json.loads(db.execute('SELECT record FROM records WHERE n=?', (old,)).fetchone()[0])
                same = normalized(previous['messages'][-1]['content']) == normalized(r['messages'][-1]['content'])
                reason = 'duplicate_prompt' if same else 'conflicting_answers_for_same_prompt'
                if not same: db.execute('UPDATE records SET valid=0 WHERE n=?', (old,))
                counts[r['source_name']+':'+reason] += 1
                rejection_file.write(json.dumps({'id': r['id'], 'reason': reason, 'reference': previous['id']})+'\n'); continue
            n = len(parent); parent.append(n); exact[key] = n
            current_signature = np.frombuffer(sig, dtype='<u4'); signatures.append(current_signature)
            db.execute('INSERT INTO records(n,record,masked,prompt) VALUES(?,?,?,?)', (n, json.dumps(r, ensure_ascii=False), masked, prompt))
            group = r['source_group_id']
            if group in source_groups: union(n, source_groups[group])
            else: source_groups[group] = n
            template = hashlib.sha256(masked.encode()).digest()
            if template in template_groups: union(n, template_groups[template])
            else: template_groups[template] = n
            keys = band_keys(sig); candidates = set()
            for index, key in zip(bands, keys): candidates.update(index.get(key, []))
            for other in sorted(candidates):
                if find(n) == find(other): continue
                if np.count_nonzero(current_signature == signatures[other]) < 16: continue
                text = db.execute('SELECT masked FROM records WHERE n=?', (other,)).fetchone()[0]
                if lexical_similarity(masked, text) >= .85: union(n, other)
            for index, key in zip(bands, keys): index[key].append(n)
            if r.get('fixed_split'): fixed[n] = r['fixed_split']
            source[n] = r['source_name']; tokens[n] = r['tokens']; identities[n] = r['id']
            counts[r['source_name']+':eligible_before_family_split'] += 1
            if len(parent) % 2000 == 0:
                db.commit(); write_json(run/'preparation_progress.json', {'accepted': len(parent), 'counts': dict(counts)})
                print('accepted', len(parent), 'raw_public', sum(v for k, v in counts.items() if k.endswith(':raw')), flush=True)
    with (run/'rejections.jsonl').open('w') as rejection_file:
        tasks = iter(batches())
        # Bounded submission and source-order consumption make selection deterministic.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork'), initializer=init_worker) as pool:
            pending = []
            for _ in range(args.workers*2):
                batch = next(tasks, None)
                if batch is None: break
                pending.append(pool.submit(process_batch, batch))
            while pending:
                consume(pending.pop(0).result())
                batch = next(tasks, None)
                if batch is not None: pending.append(pool.submit(process_batch, batch))
    db.commit()
    valid = {r[0] for r in db.execute('SELECT n FROM records WHERE valid=1')}
    families = defaultdict(list)
    for n in sorted(valid): families[find(n)].append(n)
    assigned = {}; strata = defaultdict(list); excluded_families = 0
    for root, members in families.items():
        specified = {fixed[n] for n in members if n in fixed}
        if len(specified) > 1:
            counts['mixed_fixed_split_family_records_quarantined'] += len(members); excluded_families += 1; continue
        if specified: assigned[root] = specified.pop()
        else:
            n = members[0]; strata[(source[n], length_bin(tokens[n]))].append(root)
    public_total = sum(len(families[root]) for roots in strata.values() for root in roots)
    for stratum, roots in sorted(strata.items()):
        size = sum(len(families[root]) for root in roots)
        target = max(1, math.ceil(2000*size/max(1, public_total)))
        ordered = sorted(roots, key=lambda root: hashlib.sha256(('42:'+identities[min(families[root])]).encode()).digest())
        val = private = 0
        for root in ordered:
            if val < target: assigned[root] = 'validation'; val += len(families[root])
            elif private < target: assigned[root] = 'privacy_nonmembers'; private += len(families[root])
            else: assigned[root] = 'train'
    handles = {s: (run/(s+'.records.jsonl')).open('w') for s in ('train', 'validation', 'privacy_nonmembers')}
    training_handles = {s: (run/(s+'.jsonl')).open('w') for s in ('train', 'validation')}
    entries = {s: {'rows': 0, 'ids': [], 'groups': set(), 'source_groups': set(), 'source_counts': Counter(), 'tokens': 0, 'max_tokens': 0} for s in handles}
    train_by_stratum = defaultdict(list); nonmember_by_stratum = defaultdict(list); longest = None
    for n, text in db.execute('SELECT n,record FROM records WHERE valid=1 ORDER BY n'):
        root = find(n)
        if root not in assigned: continue
        split = assigned[root]; r = json.loads(text)
        r['group_id'] = 'full_family_'+hashlib.sha256(identities[root].encode()).hexdigest()[:24]; r['split'] = split
        handles[split].write(json.dumps(r, ensure_ascii=False)+'\n')
        if split in training_handles: training_handles[split].write(json.dumps({'messages': r['messages']}, ensure_ascii=False)+'\n')
        e = entries[split]; e['rows'] += 1; e['ids'].append(r['id']); e['groups'].add(r['group_id']); e['source_groups'].add(r['source_group_id'])
        e['source_counts'][r['source_name']] += 1; e['tokens'] += r['tokens']; e['max_tokens'] = max(e['max_tokens'], r['tokens'])
        stratum = (r['source_name'], length_bin(r['tokens']))
        if split == 'train': train_by_stratum[stratum].append(r['id'])
        elif split == 'privacy_nonmembers': nonmember_by_stratum[stratum].append(r['id'])
        if split != 'privacy_nonmembers' and (longest is None or r['tokens'] > longest['tokens']): longest = r
    for handle in list(handles.values())+list(training_handles.values()): handle.close()
    all_ids = set(); all_groups = set(); all_source_groups = set()
    for split, e in entries.items():
        if not e['rows']: raise ValueError('Empty split: '+split)
        if len(set(e['ids'])) != e['rows'] or all_ids.intersection(e['ids']): raise ValueError('Overlapping/duplicate IDs')
        if all_groups.intersection(e['groups']) or all_source_groups.intersection(e['source_groups']): raise ValueError('Cross-split family leakage')
        all_ids.update(e['ids']); all_groups.update(e['groups']); all_source_groups.update(e['source_groups'])
        e['groups'] = sorted(e['groups']); e['source_groups'] = sorted(e['source_groups']); e['source_counts'] = dict(e['source_counts'])
        path = run/(split+'.records.jsonl'); e.update(records_path=str(path), source_sha256=digest(path))
        if split in training_handles: e['training_file_sha256'] = digest(run/(split+'.jsonl'))
    selected_members = []; selected_nonmembers = []; unmatched = {}
    for stratum, nonmembers in sorted(nonmember_by_stratum.items()):
        members = train_by_stratum.get(stratum, [])
        size = min(len(members), len(nonmembers))
        if not size: unmatched[str(stratum)] = len(nonmembers); continue
        order = lambda identity: hashlib.sha256(('mia42:'+identity).encode()).digest()
        selected_members.extend(sorted(members, key=order)[:size]); selected_nonmembers.extend(sorted(nonmembers, key=order)[:size])
    if len(selected_nonmembers) < 1000: raise ValueError('Too few matched nonmembers for full-data privacy evaluation')
    selection = {'member_ids': selected_members, 'nonmember_ids': selected_nonmembers,
                 'matched_on': 'source and length bins <=256/512/1024/2048/4000 tokens; fixed seed 42', 'unmatched_nonmember_strata': unmatched}
    write_json(run/'mia_selection.json', selection)
    (run/'longest_probe.jsonl').write_text(json.dumps({'messages': longest['messages']}, ensure_ascii=False)+'\n')
    manifest = {'data_scale': 'full_filtered', 'scope': 'All eligible records from both complete public pools and isolated local financial sources; no row cap or oversampling.',
                'seed': 42, 'preparation_script_sha256': preparation_sha, 'source_files': verified, 'counts': dict(counts),
                'local_input_inventory': {'distilled_raw_stage1': 435, 'distilled_previously_isolated_candidates': 411,
                                          'distilled_previously_deferred_over_4000_tokens': 24, 'luna_total': 500,
                                          'local_candidates_and_protected_holdouts_loaded': len(locals_),
                                          'stage2_safety_stage3_original_nonmembers_canaries_agent_and_benchmarks': 'excluded from financial training'},
                'splits': entries, 'max_length': 4096, 'token_filter': 4000, 'official_guard_questions': guard_count,
                'quality_policy': {'financial_score_gt': .99, 'selfcontained_score_gt': .9, 'verifiable_threshold': None,
                                   'near_family_jaccard_gte': .85, 'near_candidate_method': '32-permutation MinHash, 8 bands of 4, estimated Jaccard >=0.5 prefilter; exact Jaccard verification',
                                   'no_truncation': True, 'conflicting_exact_answers': 'quarantine all records sharing the normalized prompt'},
                'privacy_selection': str(run/'mia_selection.json'), 'privacy_selection_sha256': digest(run/'mia_selection.json'),
                'holdout_policy': 'Keep historical local heldouts isolated; reserve approximately 2000 public records per holdout, whole families stratified by source and length.',
                'limitations': ['Classifier scores and syntax checks do not establish factual or calculation correctness.',
                               'Explicit missing choices/attachment dependencies are conservatively deferred; these heuristics can reject valid prose-only examples.',
                               'Lexical/MinHash checks are approximate; semantic paraphrases, translations and pretraining contamination may remain.',
                               'No source document identifiers are released for GraphGen; true document-level separation cannot be established.',
                               'English data will dominate this one-pass corpus; Chinese FinEval may regress.'],
                'scratch_database': str(db_path), 'mixed_fixed_split_families_quarantined': excluded_families}
    write_json(run/'manifest.json', manifest)
    write_json(ROOT/'reports/full_finance_data_quality.json', {k: v for k, v in manifest.items() if k != 'splits'} | {'splits': {s: {k: v for k, v in e.items() if k not in ('ids', 'groups', 'source_groups')} for s, e in entries.items()}})
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
    print(json.dumps({'splits': {s: {'rows': e['rows'], 'tokens': e['tokens'], 'sources': e['source_counts']} for s, e in entries.items()}, 'privacy_pairs': len(selected_members)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
