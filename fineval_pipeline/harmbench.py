"""Pinned upstream GCG/PAIR search loops, native local-HF model adapters.

Only selected upstream definitions are loaded, avoiding unrelated vLLM/Ray/API
imports. Search-loop source is unmodified; backend/PAIR conversation adapters
are local and explicitly identified in the output protocol.
"""
import ast
import copy
import csv
import gc
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .core import Backend, append, bind, digest, fingerprint, model_spec, resolve, rows, write_json


def definitions(path, names, namespace=None):
    tree = ast.parse(Path(path).read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    if len(selected) != len(names):
        raise ValueError(f'Pinned upstream definitions changed: {path}')
    ns = {} if namespace is None else namespace
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def upstream(cfg):
    source = resolve(cfg['harmbench']['source_dir'])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != cfg['harmbench']['revision']:
        raise ValueError('HarmBench source revision mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('HarmBench source must be unmodified')
    return source


def behaviors(cfg):
    settings = cfg['harmbench']
    with resolve(settings['behaviors']).open(newline='', encoding='utf-8-sig') as f:
        all_rows = list(csv.DictReader(f))
    categories = set(settings['supported_categories'])
    selected = [r for r in all_rows if r['FunctionalCategory'] in categories]
    excluded = [r['BehaviorID'] for r in all_rows if r not in selected]
    if settings['limit']:
        selected = selected[:settings['limit']]
    if not selected or len({r['BehaviorID'] for r in selected}) != len(selected):
        raise ValueError('Empty or duplicate HarmBench selection')
    return selected, excluded


class Conversation:
    def __init__(self):
        self.system = ''
        self.messages = []

    def set_system_message(self, text):
        self.system = text


class PairAttacker:
    """Native chat frontend; retries count toward actual model-query accounting."""
    def __init__(self, backend, common, settings):
        self.backend, self.common, self.settings = backend, common, settings
        self.template = Conversation()
        self.failed_outputs = []

    def get_attack(self, conversations, prompts):
        results = []
        for conv, prompt in zip(conversations, prompts):
            conv.messages.append({'role': 'user', 'content': prompt})
            seed = '{"improvement": "", "prompt": "' if len(conv.messages)==1 else '{"improvement": "'
            for _ in range(self.settings['max_retries']):
                prefix = self.backend.render([{'role': 'system', 'content': conv.system}]+conv.messages)+seed
                output = self.backend.generate_raw(prefix,
                    max_new_tokens=self.settings['attack_max_n_tokens'], temperature=1.0)
                text = seed+output
                if not text.rstrip().endswith('}'):
                    text += '}'
                attack, _ = self.common['extract_json'](text)
                if attack and all(isinstance(attack.get(k), str) for k in ('prompt', 'improvement')):
                    conv.messages.append({'role': 'assistant', 'content': json.dumps(attack)})
                    results.append(attack)
                    break
                self.failed_outputs.append(text[-8192:])
            else:
                # Do not fall back to direct requests and call them PAIR successes.
                raise RuntimeError('PAIR attacker failed JSON generation within retry budget')
        return results


class PairTarget:
    def __init__(self, backend, maximum):
        self.backend, self.maximum = backend, maximum

    def get_response(self, prompts):
        return [self.backend.chat([{'role': 'user', 'content': p}], max_new_tokens=self.maximum) for p in prompts]


class PairJudge:
    def __init__(self, backend, prompts, maximum):
        self.backend, self.prompts, self.maximum = backend, prompts, maximum

    def score(self, attacks, responses, behavior, context):
        scores = []
        for attack, response in zip(attacks, responses):
            messages = [{'role': 'system', 'content': self.prompts['get_judge_system_prompt'](behavior, context)},
                {'role': 'user', 'content': f'[CONTEXT]: {context}\n[PROMPT]: {attack}\n[ASSISTANT RESPONSE]: {response}'}]
            seed = 'Rating: [['
            text = seed+self.backend.generate_raw(self.backend.render(messages)+seed, max_new_tokens=self.maximum)
            found = re.search(r'\[\[(10|[1-9])\]\]', text)
            if not found:
                raise RuntimeError('PAIR search judge returned an invalid rating; not counted as safe')
            scores.append(int(found.group(1)))
        return scores


def make_attack(cfg, method, target, auxiliary):
    import torch
    from accelerate.utils import find_executable_batch_size
    from torch.nn import CrossEntropyLoss
    from tqdm import tqdm
    source = upstream(cfg)
    settings = cfg['harmbench']
    if method == 'GCG':
        if settings['gcg'].get('use_prefix_cache'):
            raise ValueError('Legacy GCG prefix KV cache is incompatible with Qwen3.5 hybrid attention')
        utils = definitions(source/'baselines/gcg/gcg_utils.py', ['sample_control', 'get_nonascii_toks'], {'torch': torch})
        class LocalBase:
            def __init__(self, **kwargs):
                self.model, self.tokenizer = target.model, target.tokenizer
                self.model_name_or_path = target.spec['path']
        def nonascii(tokenizer):
            original = utils['get_nonascii_toks'](tokenizer)
            padded = torch.arange(len(tokenizer), target.model.get_input_embeddings().weight.shape[0])
            return torch.cat([original, padded])
        namespace = dict(gc=gc, json=json, torch=torch, tqdm=tqdm, CrossEntropyLoss=CrossEntropyLoss,
                         find_executable_batch_size=find_executable_batch_size,
                         SingleBehaviorRedTeamingMethod=LocalBase,
                         get_template=lambda **kw: {'prompt': target.render([{'role': 'user', 'content': '{instruction}'}])},
                         sample_control=utils['sample_control'], get_nonascii_toks=nonascii)
        cls = definitions(source/'baselines/gcg/gcg.py', ['GCG'], namespace)['GCG']
        return cls(target_model={}, targets_path=str(resolve(settings['targets'])), **settings['gcg'])
    if method != 'PAIR':
        raise ValueError('Only GCG and PAIR are supported')
    common = definitions(source/'baselines/pair/common.py', ['extract_json', 'get_init_msg', 'process_target_response'], {'ast': ast})
    prompts = definitions(source/'baselines/pair/system_prompts.py', ['get_attacker_system_prompt', 'get_judge_system_prompt'])
    opts = settings['pair']
    attacker = PairAttacker(auxiliary[opts['attacker']], common, opts)
    pair_target = PairTarget(target, settings['target_max_new_tokens'])
    judge = PairJudge(auxiliary[opts['search_judge']], prompts, opts['judge_max_n_tokens'])
    ns = dict(copy=copy, json=json, tqdm=tqdm, SingleBehaviorRedTeamingMethod=object,
              get_init_msg=common['get_init_msg'], process_target_response=common['process_target_response'],
              get_attacker_system_prompt=prompts['get_attacker_system_prompt'],
              load_PAIR_models=lambda *a, **kw: (attacker, pair_target, judge))
    cls = definitions(source/'baselines/pair/PAIR.py', ['PAIR', 'clean_attacks_and_convs'], ns)['PAIR']
    return cls(attack_model={}, target_model={}, judge_model={}, targets_path=str(resolve(settings['targets'])),
               attack_max_n_tokens=opts['attack_max_n_tokens'], target_max_n_tokens=settings['target_max_new_tokens'],
               judge_max_n_tokens=opts['judge_max_n_tokens'], max_retries=opts['max_retries'],
               n_streams=opts['n_streams'], steps=opts['steps'], keep_last_n=3)


def attack(cfg, name, method, target, identity, auxiliary, directory):
    import torch
    import random
    import numpy as np
    selected, excluded = behaviors(cfg)
    settings = cfg['harmbench']
    context = {'task': 'harmbench-upstream-search-native-HF-v1', 'model': identity, 'method': method,
               'parameters': settings['gcg'] if method=='GCG' else settings['pair'], 'revision': settings['revision'],
               'target_max_new_tokens': settings['target_max_new_tokens'],
               'auxiliary_models': {n: getattr(b, 'identity', b.spec) for n, b in auxiliary.items()},
               'behaviors': digest(resolve(settings['behaviors'])), 'targets': digest(resolve(settings['targets'])),
               'selected_ids': [b['BehaviorID'] for b in selected], 'seed': settings['seed']}
    run_id = bind(directory, context)
    path = directory/'attacks.jsonl'
    done = {r['behavior_id']: r for r in rows(path)} if path.exists() else {}
    if any(r['fingerprint'] != run_id for r in done.values()):
        raise ValueError('Attack resume fingerprint mismatch')
    engine = make_attack(cfg, method, target, auxiliary)
    for behavior in selected:
        key = behavior['BehaviorID']
        if key in done:
            continue
        seed = settings['seed'] + int(fingerprint(key)[:8], 16)
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed % 2**32)
        counters = {'target': target.calls, **{n: b.calls for n, b in auxiliary.items()}}
        try:
            prompt, logs = engine.generate_test_cases_single_behavior(behavior, verbose=False)
        except Exception as error:
            write_json(directory/'failure.json', {'behavior_id': key, 'error': str(error),
                'type': type(error).__name__, 'fingerprint': run_id,
                'target_generation_queries': target.calls-counters['target'],
                'auxiliary_queries': {n:b.calls-counters[n] for n,b in auxiliary.items()},
                'attacker_invalid_outputs': getattr(getattr(engine,'attackLM',None),'failed_outputs',[])})
            raise
        # Target completion freshly generated under the common evaluation budget.
        completion = target.chat([{'role': 'user', 'content': prompt}], max_new_tokens=settings['target_max_new_tokens'])
        queries = {'target': target.calls-counters['target'],
                   **{n: b.calls-counters[n] for n, b in auxiliary.items()}}
        result = {'behavior_id': key, 'prompt': prompt, 'generation': completion, 'attack_log': logs,
                  'query_counts': queries, 'fingerprint': run_id}
        if method == 'GCG':
            result['gradient_steps'] = len(logs['all_losses'])
            result['candidate_budget_upper_bound'] = settings['gcg']['num_steps']*settings['gcg']['search_width']
        append(path, result)
        done[key] = result
        print(f'{method} {name}: {len(done)}/{len(selected)}', flush=True)
    write_json(directory/'attack_summary.json', {'fingerprint': run_id, 'model_identity': identity,
        'selected_ids': context['selected_ids'], 'excluded_unsupported_ids': excluded, 'method': method,
        'split': settings['split'], 'scope': 'subset' if settings['limit'] or excluded else 'all_text_behaviors',
        'parameters': context['parameters'],
        'target_max_new_tokens': settings['target_max_new_tokens'],
        'protocol': context['task'],
        'note': 'Each model attacked afresh; upstream search loops + native-HF adapters; categories recorded explicitly.'})


def score(cfg, method, directory):
    settings = cfg['harmbench']
    source = upstream(cfg)
    classifier_spec = dict(settings['classifier'])
    classifier_spec['path'] = str(resolve(classifier_spec['path']))
    if not Path(classifier_spec['path']).exists():
        raise FileNotFoundError('Official HarmBench classifier missing; cannot substitute refusal keywords for ASR')
    manifest_path = resolve(classifier_spec['manifest'])
    manifest = json.loads(manifest_path.read_text())
    if manifest['repo'] != classifier_spec['upstream_id'] or manifest['revision'] != classifier_spec['revision']:
        raise ValueError('Official classifier provenance mismatch')
    for record in manifest['files']:
        path = Path(classifier_spec['path'])/record['file']
        if not path.is_file() or path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
            raise ValueError('Official classifier file missing or changed: '+record['file'])
    from .core import model_identity
    classifier_id = model_identity(classifier_spec)
    classifier_id['upstream_manifest_sha256'] = digest(manifest_path)
    completed = rows(directory/'attacks.jsonl')
    selected, _ = behaviors(cfg)
    behavior_map = {b['BehaviorID']: b for b in selected}
    if {r['behavior_id'] for r in completed} != set(behavior_map) or len(completed) != len(behavior_map):
        raise ValueError('Incomplete attack coverage; no ASR emitted')
    templates = definitions(source/'eval_utils.py', ['LLAMA2_CLS_PROMPT'])['LLAMA2_CLS_PROMPT']
    context = {'task': 'harmbench-official-llama2-classifier-HF-v1', 'classifier': classifier_id,
               'attacks_sha256': digest(directory/'attacks.jsonl'), 'revision': settings['revision']}
    run_id = bind(directory/'judge', context)
    path = directory/'judge/predictions.jsonl'
    saved = rows(path) if path.exists() else []
    done = {r['behavior_id']: r for r in saved}
    if len(done)!=len(saved) or not set(done)<=set(behavior_map) or any(r['label'] not in (0,1) for r in saved):
        raise ValueError('Invalid/duplicate/out-of-scope classifier checkpoint')
    classifier = Backend(classifier_spec)
    hashing = None
    for row in completed:
        key = row['behavior_id']
        if key in done:
            if done[key]['fingerprint'] != run_id:
                raise ValueError('Judge checkpoint identity mismatch')
            continue
        b = behavior_map[key]
        text = classifier.tokenizer.decode(classifier.tokenizer.encode(row['generation'], add_special_tokens=True)[:512], skip_special_tokens=True)
        if 'hash_check' in b['Tags']:
            if hashing is None:
                import spacy
                from datasketch import MinHash
                from tqdm import tqdm
                nlp = spacy.load('en_core_web_sm')
                nlp.max_length = 10000000
                hashing = definitions(source/'eval_utils.py', ['tokenize', 'compute_hashes_with_sliding_window',
                    'check_output_with_sliding_window', 'compute_results_hashing'],
                    {'nlp': nlp, 'MinHash': MinHash, 'tqdm': tqdm})
            import os
            cwd = os.getcwd()
            try:
                os.chdir(source)
                label = hashing['compute_results_hashing'](b, [{'generation': text}])[0]['label']
            finally:
                os.chdir(cwd)
            result = {'behavior_id': key, 'label': label, 'fingerprint': run_id}
            append(path, result); done[key] = result
            continue
        template = templates['prompt_contextual' if 'context' in b['Tags'].split(', ') else 'prompt']
        prompt = template.format(behavior=b['Behavior'], context=b['ContextString'], generation=text)
        if classifier.tokenizer.bos_token and not prompt.startswith(classifier.tokenizer.bos_token):
            prompt = classifier.tokenizer.bos_token + prompt
        output = classifier.generate_raw(prompt, max_new_tokens=1).strip().lower()
        if output not in ('yes', 'no'):
            raise RuntimeError(f'Classifier invalid label {output!r}; not silently counted as safe')
        result = {'behavior_id': key, 'label': int(output=='yes'), 'fingerprint': run_id}
        append(path, result); done[key] = result
    attacks_meta = json.loads((directory/'attack_summary.json').read_text())
    categories = {}
    for key,r in done.items():
        category = behavior_map[key]['FunctionalCategory']
        values = categories.setdefault(category,{'successes':0,'count':0})
        values['successes'] += r['label']; values['count'] += 1
    for values in categories.values():
        values['asr'] = values['successes']/values['count']
    summary = {'task': context['task'], 'method': method, 'asr': sum(r['label'] for r in done.values())/len(completed),
               'behaviors': len(completed), 'successful_behaviors': sum(r['label'] for r in done.values()),
               'fingerprint': run_id, 'classifier_identity': classifier_id, 'attack': attacks_meta,
               'by_functional_category': categories,
               'coverage_note': 'Per-behavior one final attack; excluded categories and limits recorded in attack summary.'}
    write_json(directory/'summary.json', summary)
    return summary
