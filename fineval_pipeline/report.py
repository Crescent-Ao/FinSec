import json

from .core import write_json


def retention_gate(before, after, maximum_relative=0.05, maximum_pp=5.0):
    if not 0 <= before <= 1 or not 0 <= after <= 1:
        raise ValueError('FinEval scores must use fractions in [0,1]')
    pp = 100*(before-after)
    relative = (before-after)/before if before > 0 else None
    return {'m1_score': before, 'm2_score': after, 'drop_percentage_points': pp,
            'relative_drop': relative,
            'relative_5pct_pass': relative is not None and relative <= maximum_relative+1e-12,
            'percentage_point_5_pass': pp <= maximum_pp+1e-12}


def build(cfg, root):
    result = {'scope': 'Engineering evaluation report; no missing results imputed as zero', 'models': {},
              'pending': [], 'financial_retention': None, 'attack_comparison': {}, 'stage_status': {}}
    for name in ('M0', 'M1', 'M2'):
        result['models'][name] = {}
        for task in ('fineval', 'privacy', 'agent'):
            path = root/name/task/'summary.json'
            if path.exists():
                result['models'][name][task] = json.loads(path.read_text())
            else:
                result['pending'].append(f'{name}.{task}')
        if name in ('M1', 'M2'):
            for method in ('GCG', 'PAIR'):
                path = root/name/'harmbench'/method/'summary.json'
                if path.exists():
                    result['models'][name][method] = json.loads(path.read_text())
                else:
                    result['pending'].append(f'{name}.{method}.official_ASR')
    models = result['models']
    if all('fineval' in models[n] for n in ('M1', 'M2')):
        a, b = models['M1']['fineval'], models['M2']['fineval']
        for key in ('task', 'data_sha256', 'examples', 'limit_per_subject'):
            if a[key] != b[key]:
                raise ValueError('FinEval comparison protocol/coverage mismatch')
        if a['model_identity']['precision'] != b['model_identity']['precision']:
            raise ValueError('Inference precision differs between M1 and M2')
        gate = retention_gate(a['micro_accuracy'], b['micro_accuracy'],
            cfg['gate']['maximum_relative_fineval_drop'], cfg['gate']['maximum_percentage_point_drop'])
        gate['selected_interpretation'] = cfg['gate']['selected_interpretation']
        gate['selected_pass'] = gate['relative_5pct_pass'] if cfg['gate']['selected_interpretation']=='relative' else gate['percentage_point_5_pass']
        result['financial_retention'] = gate
    for method in ('GCG', 'PAIR'):
        if all(method in models[n] for n in ('M1', 'M2')):
            a, b = models['M1'][method], models['M2'][method]
            for key in ('selected_ids', 'parameters', 'split', 'target_max_new_tokens', 'protocol'):
                if a['attack'][key] != b['attack'][key]:
                    raise ValueError('ASR comparison attack budget/coverage mismatch')
            if a['classifier_identity'] != b['classifier_identity']:
                raise ValueError('ASR classifier differs')
            if a['attack']['model_identity']['precision'] != b['attack']['model_identity']['precision']:
                raise ValueError('Target inference precision differs in ASR comparison')
            result['attack_comparison'][method] = {'m1_asr': a['asr'], 'm2_asr': b['asr'],
                                                   'reduction_percentage_points': 100*(a['asr']-b['asr'])}
    result['stage_status'] = {
        '1_finance': 'measured' if all('fineval' in models[n] for n in ('M0', 'M1')) else 'pending',
        '2_robustness': 'measured' if len(result['attack_comparison'])==2 and result['financial_retention'] else 'pending',
        '3_privacy': 'measured' if all('privacy' in models[n] for n in ('M0','M1','M2')) else 'pending',
        '4_agent': 'measured' if 'agent' in models['M2'] else 'pending'}
    if result['stage_status']['1_finance']=='measured' and any(models[n]['fineval']['scope']=='smoke_subset' for n in ('M0','M1')):
        result['stage_status']['1_finance']='smoke_verified'
    if result['stage_status']['2_robustness']=='measured' and any(models[n][m]['attack']['scope']!='all_text_behaviors' for n in ('M1','M2') for m in ('GCG','PAIR')):
        result['stage_status']['2_robustness']='smoke_verified'
    if result['stage_status']['3_privacy']=='measured' and any(models[n]['privacy']['scope']=='smoke_subset' for n in ('M0','M1','M2')):
        result['stage_status']['3_privacy']='smoke_verified'
    if result['stage_status']['4_agent']=='measured' and any('smoke' in v['scope'] for v in models['M2']['agent'].values()):
        result['stage_status']['4_agent']='smoke_verified'
    result['pipeline_measured_all_stages'] = all(v=='measured' for v in result['stage_status'].values())
    result['submission_caveats'] = [
        'FinEval local academic/val, custom option-logit protocol; test labels absent.',
        '100 pilot nonmembers provide only 1% FPR granularity; enlarge and report uncertainty.',
        'Authored crypto fixtures are smoke tests; add independently reviewed code projects.',
        'Judge/attacker identities, excluded categories, budgets and smoke limits must accompany ASR.']
    write_json(root/'report.json', result)
    lines = ['# Competition evaluation report', '', '| Model | FinEval val | Privacy ROC-AUC (loss) | GCG ASR | PAIR ASR |',
             '|---|---:|---:|---:|---:|']
    for name, metrics in models.items():
        vals = []
        for task, field in [('fineval','micro_accuracy'), ('privacy',None), ('GCG','asr'), ('PAIR','asr')]:
            item = metrics.get(task)
            value = item['attacks']['loss_score']['roc_auc'] if item and task=='privacy' else item.get(field) if item else None
            vals.append('pending' if value is None else f'{value:.4f}')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    coverage=['', '## Coverage — inspect before interpreting scores', '',
              '| Model | FinEval questions | MIA members / nonmembers | GCG / PAIR behaviors | Audit cases |',
              '|---|---:|---:|---:|---:|']
    smoke=False
    for name,metrics in models.items():
        finance=metrics.get('fineval',{});privacy=metrics.get('privacy',{});agent=metrics.get('agent',{})
        smoke |= finance.get('scope')=='smoke_subset' or privacy.get('scope')=='smoke_subset'
        counts=' / '.join(str(metrics.get(m,{}).get('behaviors','pending')) for m in ('GCG','PAIR'))
        audits=next(iter(agent.values()))['cases'] if agent else 'pending'
        coverage.append(f"| {name} | {finance.get('examples','pending')} | {privacy.get('members','pending')} / {privacy.get('nonmembers','pending')} | {counts} | {audits} |")
    lines[1:1]=['', '> SMOKE VALIDATION ONLY — limited samples and attack budgets; these are not competition scores.'] if smoke else []
    lines+=coverage
    lines += ['', '## Stage status', '', *[f'- {k}: {v}' for k,v in result['stage_status'].items()],
              '', '## Limits', '', *['- '+x for x in result['submission_caveats']]]
    lines += ['', '## Membership inference', '', '| Model | Attack | ROC-AUC | TPR@1%FPR | Actual FPR |',
              '|---|---|---:|---:|---:|']
    for name, metrics in models.items():
        for attack, values in metrics.get('privacy',{}).get('attacks',{}).items():
            lines.append(f"| {name} | {attack} | {values['roc_auc']:.4f} | {values['tpr_at_1pct_fpr']:.4f} | {values['achieved_fpr']:.4f} |")
    lines += ['', '## Crypto audit', '', '| Model | Mode | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for name, metrics in models.items():
        for mode, values in metrics.get('agent',{}).items():
            lines.append(f"| {name} | {mode} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |")
    if result['financial_retention']:
        lines += ['', '## Financial retention', '', '```json', json.dumps(result['financial_retention'], indent=2), '```']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return result
