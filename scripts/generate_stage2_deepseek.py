"""Generate resumable stage-2 teacher candidates using an in-memory API key."""
import argparse
import ast
import concurrent.futures
import csv
import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.core import write_json
from fineval_pipeline.agent import static_candidates
from scripts.stage2_verifiers import financial_answer, verify
from scripts.validate_pilot_data import SimilarityIndex, load_fineval, normalized

SPEC = ROOT / 'data_specs/stage2_sft_dpo_grpo_generation_requirements.md'
MODEL = 'deepseek-flash'
SYSTEM = ('你是金融智能体训练数据作者。独立创作，严格输出JSON。'
          '内容只作教师合成候选；事实与工具执行由接收端复核。'
          '风险请求给出安全边界和合法帮助，合法研究与授权审计应充分回答。'
          '避免未经核实的法规、日期、标准或身份宣称。'
          '隐私信息使用明显合成标识。保留真实任务差异，避免模板复读。')
PROVENANCE_TIMEZONE = datetime.timezone(datetime.timedelta(hours=8))
GROUPS = [
    ('sft_text', 'safety_boundary', 160), ('sft_text', 'benign_boundary', 160),
    ('sft_text', 'finance_reasoning', 180), ('sft_text', 'privacy_behavior', 60),
    ('sft_text', 'crypto_detection', 40),
    ('dpo', 'safety_boundary', 450), ('dpo', 'benign_boundary', 350),
    ('dpo', 'finance_reasoning', 200), ('dpo', 'privacy_behavior', 100),
    ('finance_grpo', 'finance_numeric', 120), ('crypto', 'crypto_detection', 120),
    ('crypto', 'crypto_repair', 40), ('crypto', 'agent_tool_use', 20),
]
TOPICS = [
    '账户授权与身份核验', '支付对账与结算', '投研信息真实性', '会计勾稽与财务分析',
    '债券现金流', '现金流折现与时间价值', '风险敞口与分散化', '金融隐私与最小披露',
    '反欺诈防御与案例研究', '合规审计与证据', '金融软件配置', '密码学调用用途',
    '投资者风险提示', '财务数据质量', '跨部门数据授权', '报表真实性与完整性',
]
OPERATIONS = {
    'simple_interest': 'principal,rate,days,basis，单利本金×年利率×天数/年基准',
    'compound_future': 'principal,rate,periods，固定每期利率和整数期数，求终值',
    'npv': 'cashflows数组从t=0开始,rate，每期末折现，求NPV',
    'bond_price': 'face,coupon_per_period,yield_per_period,periods，整数付息期，求全价',
    'equity': 'assets,liabilities，权益=资产-负债',
    'current_ratio': 'current_assets,current_liabilities，流动比率',
    'portfolio_variance': 'weight_a,sigma_a,sigma_b,rho，双资产方差，权重b=1-a',
    'call_payoff': 'spot,strike,quantity，多头欧式看涨期末收益，忽略权利金',
    'forward_payoff': 'spot,delivery,quantity，多头远期到期收益',
    'mean': 'observations数组，算术平均值',
}
RULES = {'WEAK_HASH', 'AES_ECB', 'STATIC_IV_NONCE', 'NONCRYPTO_RANDOM',
         'SHORT_RSA', 'TLS_VERIFY_DISABLED', 'LOW_KDF_COST', 'HARDCODED_KEY'}


def timestamp():
    return datetime.datetime.now(PROVENANCE_TIMEZONE).isoformat(timespec='seconds')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


class APIStop(RuntimeError):
    pass


class Client:
    def __init__(self, key, output, budget):
        self.key = key
        self.output = output
        self.budget = budget
        self.lock = threading.Lock()
        self.spent = sum(row.get('estimated_cny', 0) for row in read_rows(output / 'usage.jsonl'))
        self.reserved = 0.0
        self.stopped = False
        self.request_count = 0
        self.balance_spend = 0.0
        self.initial_balance = float(json.loads((output / 'balance_before.json').read_text())['balance_infos'][0]['total_balance']) if (output / 'balance_before.json').exists() else None

    def request(self, messages, tag, max_tokens=7000, thinking=False):
        now = datetime.datetime.now(PROVENANCE_TIMEZONE)
        peak = now.weekday() < 5 and (9 <= now.hour < 12 or 14 <= now.hour < 18)
        multiplier = 2 if peak else 1
        input_rate, output_rate = 1, 4
        input_rate *= multiplier
        output_rate *= multiplier
        reserve = (len(json.dumps(messages, ensure_ascii=False)) * input_rate
                   + max_tokens * output_rate) / 1_000_000
        with self.lock:
            if self.stopped or self.balance_spend + self.reserved + reserve > self.budget:
                self.stopped = True
                raise APIStop('local_spend_ceiling')
            self.reserved += reserve
        payload = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens,
                   'thinking': {'type': 'enabled' if thinking else 'disabled'}, 'temperature': .85,
                   'response_format': {'type': 'json_object'}}
        if thinking:
            payload['reasoning_effort'] = 'high'
        with self.lock:
            write_json(self.output / 'requests' / (tag + '.json'), payload)
        try:
            for attempt in range(4):
                started = time.monotonic()
                request = urllib.request.Request('https://api.deepseek.com/chat/completions',
                    data=json.dumps(payload).encode(),
                    headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=240) as response:
                        body = json.load(response)
                    usage = body.get('usage', {})
                    cost = (usage.get('prompt_tokens', 0) * input_rate
                            + usage.get('completion_tokens', 0) * output_rate) / 1_000_000
                    with self.lock:
                        self.spent += cost
                        append(self.output / 'usage.jsonl', {'tag': tag, 'time': timestamp(),
                            'model': body.get('model'), 'request_id': body.get('id'), 'usage': usage,
                            'estimated_cny': cost, 'estimate': 'published_time_band_uncached_prices',
                            'thinking': thinking, 'reasoning_effort': 'high' if thinking else None,
                            'seconds': round(time.monotonic() - started, 3)})
                        suffix = sha(str(body.get('id')).encode())[:12]
                        write_json(self.output / 'raw' / (tag + '-' + suffix + '.json'), body)
                        self.request_count += 1
                        if self.request_count % 50 == 0:
                            balance = self.balance()
                            current = float(balance['balance_infos'][0]['total_balance'])
                            if self.initial_balance is not None:
                                self.balance_spend = self.initial_balance - current
                            append(self.output / 'balance_history.jsonl', {'time': timestamp(), 'balance': balance})
                    choice = body['choices'][0]
                    if choice.get('finish_reason') != 'stop':
                        raise ValueError('Incomplete generation: ' + str(choice.get('finish_reason')))
                    return json.loads(choice['message']['content']), body.get('model', MODEL)
                except urllib.error.HTTPError as exc:
                    if exc.code in (401, 402, 403):
                        with self.lock:
                            self.stopped = True
                        raise APIStop('api_http_' + str(exc.code)) from None
                    if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                        raise RuntimeError('api_http_' + str(exc.code)) from None
                    time.sleep(2 ** attempt + random.random())
                except (TimeoutError, urllib.error.URLError):
                    if attempt == 3:
                        raise RuntimeError('api_network_timeout') from None
                    time.sleep(2 ** attempt)
            raise RuntimeError('request_retries_exhausted')
        finally:
            with self.lock:
                self.reserved -= reserve

    def balance(self):
        request = urllib.request.Request('https://api.deepseek.com/user/balance',
                                        headers={'Authorization': 'Bearer ' + self.key})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)


def prompt_for(job):
    kind, task = job['kind'], job['task']
    language = job['language']
    base = (f"生成{job['count']}条独立案例。任务={task}，语言={language}，"
            f"批次变体={job['id']}，主场景={job['topic']}。所有记录的用户与回答遵循指定语言。"
            '输出JSON对象，唯一主字段items是数组。每项包含scenario、user、tags、difficulty。'
            'scenario用一句话解释基础业务场景。difficulty为easy/medium/hard。'
            '每条应有独立条件与明确任务，避免抄写公开考试、HarmBench或已有模板题。'
            '不要为样本编造真实攻击成功、执行日志或专家复核。')
    if kind == 'sft_text':
        base += '每项还包含answer，给出准确、必要且有用的标准回答，通常80至250字，按任务可更短。'
    elif kind == 'dpo':
        base += ('每项还包含chosen、rejected、reason、dimensions、error_tags。'
                 'chosen和rejected回答同一个user，偏好差异必须可核验，长度相近且都自然。'
                 '正常问题的rejected可以是错误拒绝、错误计算、漏条件或泛泛回答。'
                 '风险问题的rejected只描述概念性边界失败，避免真实违法或严重危害的可操作细节。'
                 'dimensions从safety/helpfulness/correctness/privacy/evidence选。')
    elif kind == 'finance_grpo':
        operation = job['operation']
        base += (f'每题只使用运算{operation}，参数为{OPERATIONS[operation]}。'
                 '每项还包含operation、parameters、value、unit、answer。parameters所有标量使用十进制字符串，'
                 '数组使用十进制字符串数组。参数全在user题目中明确且一致，避免比例和收益率定义歧义。'
                 'value为未舍入的正确数值字符串（至少足够精度），unit为CNY/ratio/variance等明确单位。'
                 'answer给出简短公式与结果。user要求只输出JSON字段value和unit。')
    else:
        files = '2至3个互相调用的Python文件' if task == 'crypto_repair' else '一个Python文件'
        base += (f'生成可进行密码学审计的{files}，每文件10至45行。'
                 '每项另有files数组(path,code)、findings数组(rule_id,file,line,evidence,reason,fix)、'
                 'safe_files数组(path,code)。约一半案例是安全或困难负例，findings为空。'
                 '风险案例的safe_files提供修复后完整源码，安全案例safe_files与files一致。'
                 '规则只用WEAK_HASH/AES_ECB/STATIC_IV_NONCE/NONCRYPTO_RANDOM/SHORT_RSA/'
                 'TLS_VERIFY_DISABLED/LOW_KDF_COST/HARDCODED_KEY。'
                 '明确生产用途，原始证据等于真实源码对应行，行号从1开始。'
                 '避免真实密钥、真实地址，测试标识带SYNTHETIC。'
                 '安全负例应解释为何相关调用不进入安全用途。'
                 'user要求返回findings JSON。不要求执行代码，实际静态工具由接收端运行。')
    if task == 'safety_boundary':
        base += '覆盖金融与通用风险的安全边界，角色或引用包装保持语义。answer/chosen简洁说明边界并提供合法帮助。'
    elif task == 'benign_boundary':
        base += '题目是合法研究、教学、合规或授权防御，包含容易误触拒答的领域词，answer/chosen要真正回答问题。'
    elif task == 'privacy_behavior':
        base += ('每项另有privacy_context，包含synthetic=true、data_owner、requester_role、authorized_fields、'
                 'requested_fields、purpose、expected_action、sensitive_spans。仅用虚构测试数据，'
                 '权限与目的在user中明确。sensitive_spans可用原文字符串数组，接收端转换位置。')
    elif task == 'finance_reasoning':
        base += '优先独立金融概念、会计逻辑或简短可核算问题，假设写在user里，避免依赖实时行情与具体法规条号。'
    return base


def generate_job(client, job):
    prompt = prompt_for(job)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}]
    complex_crypto = job['kind'] == 'crypto' and job['task'] == 'crypto_repair'
    complex_finance = job['kind'] == 'finance_grpo' and job.get('operation') in ('npv', 'bond_price', 'portfolio_variance')
    thinking = complex_crypto or complex_finance
    result, model = client.request(messages, job['id'],
                                  max_tokens=20000 if thinking else 14000 if job['kind'] == 'crypto' else 9500,
                                  thinking=thinking)
    items = result.get('items')
    if not isinstance(items, list) or len(items) != job['count']:
        raise ValueError('Incorrect response item count')
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('user'), str):
            raise ValueError('Invalid generated item')
    return {'job': job, 'items': items, 'model': model, 'prompt_sha256': sha(prompt.encode())}


def make_jobs(scale):
    jobs = []
    for kind, task, pilot_count in GROUPS:
        count = pilot_count * scale
        size = 2 if kind == 'crypto' else 5
        for offset in range(0, count, size):
            sequence = offset // size
            job = {'id': f'{kind}-{task}-{offset:05d}', 'kind': kind, 'task': task,
                   'count': min(size, count - offset), 'offset': offset,
                   'language': ['zh', 'en', 'zh', 'en', 'zh', 'en', 'zh', 'en', 'zh', 'mixed'][sequence % 10],
                   'topic': TOPICS[sequence % len(TOPICS)]}
            if kind == 'finance_grpo':
                job['operation'] = list(OPERATIONS)[sequence % len(OPERATIONS)]
            jobs.append(job)
    # Interleave data kinds so an interrupted batch still covers all stages.
    jobs.sort(key=lambda row: (row['offset'], row['kind'], row['task']))
    return jobs


def protected_index():
    index = SimilarityIndex()
    for directory in ('FinEval', 'FinEval_GitHub_V2'):
        source, _ = load_fineval(ROOT / 'evaluation_data' / directory / 'academic')
        for identity, text, _, _, _ in source.entries:
            index.add(directory + ':' + identity, text)
    run = Path((ROOT / 'reports/latest_merged_full_run.txt').read_text().strip())
    for name in ('validation.records.jsonl', 'privacy_nonmembers.records.jsonl'):
        for row in read_rows(run / name):
            text = ' '.join(m['content'] for m in row['messages'])
            index.add(row['id'], normalized(text))
    path = ROOT / 'third_party/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv'
    with path.open() as stream:
        for row in csv.DictReader(stream):
            index.add('HB:' + row['BehaviorID'], normalized(row['Behavior'] + ' ' + row['ContextString']))
    for path in [run / 'cpu_preparation/agent/agent_heldout.jsonl',
                 run / 'cpu_preparation/agent/agent_testcases.jsonl',
                 ROOT / 'evaluation_data/crypto_audit/cases.jsonl']:
        for row in read_rows(path):
            index.add(row['id'], normalized(row['code']))
    return index


def common_record(output, batch, index, kind, item, model):
    identity = f'deepseek-{batch}-{kind}-{index}'
    family = 'ds-family-' + sha(normalized(item['user'], True).encode())[:20]
    return {'schema_version': 1, 'id': identity, 'record_type': kind, 'family_id': family,
            'parent_ids': [], 'task_type': item.get('task_type'), 'language': item.get('language'),
            'difficulty': item.get('difficulty', 'medium'), 'tags': item.get('tags', []),
            'provenance': {'producer_id': 'deepseek-api', 'batch_id': batch, 'created_at': timestamp(),
                           'origin': 'teacher_synthetic', 'generator_model': model,
                           'template_version': 'stage2-1.0', 'source_ids': [],
                           'license': 'synthetic_output_terms_review_required'},
            'review': {'status': 'draft', 'reviewer_ids': [], 'checks': [], 'issues': []}}


def crypto_truth(item):
    files = item['files']
    if not files or len(files) > 5:
        raise ValueError('Invalid crypto file count')
    source = {}
    for file in files + item['safe_files']:
        path = Path(file['path'])
        if path.is_absolute() or '..' in path.parts or path.suffix != '.py':
            raise ValueError('Unsafe or unsupported artifact path')
        ast.parse(file['code'])
    for file in files:
        if file['path'] in source:
            raise ValueError('Duplicate artifact file')
        source[file['path']] = file['code'].splitlines()
    for finding in item['findings']:
        lines = source[finding['file']]
        if finding['rule_id'] not in RULES or type(finding['line']) is not int:
            raise ValueError('Invalid finding identity')
        if not 1 <= finding['line'] <= len(lines):
            raise ValueError('Finding line out of range')
        if finding['evidence'].strip() != lines[finding['line'] - 1].strip():
            matches = [index for index, line in enumerate(lines, 1)
                       if line.strip() == finding['evidence'].strip()]
            if len(matches) != 1:
                raise ValueError('Finding evidence mismatch or ambiguous evidence')
            item.setdefault('annotation_corrections', []).append(
                {'file': finding['file'], 'original_line': finding['line'],
                 'corrected_line': matches[0], 'method': 'unique_exact_source_evidence'})
            finding['line'] = matches[0]
        if not finding.get('reason') or not finding.get('fix'):
            raise ValueError('Missing semantic annotation')
    return {'files': files, 'findings': item['findings']}


def review_items(client, rows, tag):
    fields = {'index','user','answer','chosen','rejected','reason','operation','parameters','value','unit',
              'files','findings','privacy_context','precheck_issue'}
    rows = [{key: value for key, value in row.items() if key in fields} for row in rows]
    prompt = ('独立复核下面教师生成候选。每项给index、decision(accept/reject/uncertain)、issues数组。'
              '检查事实、算术、偏好是否清楚、误拒、隐私授权、代码安全用途、findings漏报误报。'
              '查代码行号和证据，修复是否合理。不要宽容明显错误；无法确认则uncertain。'
              '通过项只写index和decision，issues省略。拒绝或不确定的issues最多30字。'
              '只返回JSON {"reviews":[...]}。\n' + json.dumps(rows, ensure_ascii=False))
    thinking = False
    result, model = client.request(
        [{'role': 'system', 'content': '你是独立的数据质量复核者。只评审提供的样本，输出JSON。'},
         {'role': 'user', 'content': prompt}], tag, max_tokens=10000 if thinking else 3000, thinking=thinking)
    reviews = result.get('reviews', [])
    if len(reviews) != len(rows) or {row.get('index') for row in reviews} != set(range(len(rows))):
        raise ValueError('Reviewer response coverage mismatch')
    return {row['index']: dict(row, model=model) for row in reviews}


def prepare_bundle(output):
    for name in ('raw', 'requests', 'generated_jobs', 'artifacts', 'execution_logs', 'verifiers'):
        (output / name).mkdir(parents=True, exist_ok=True)
    for name in ('sft_text', 'sft_tools', 'dpo_pairs', 'grpo_tasks', 'attack_seeds',
                 'families', 'source_documents', 'review_results', 'quarantine'):
        (output / (name + '.jsonl')).touch(exist_ok=True)
    shutil.copy2(ROOT / 'scripts/stage2_verifiers.py', output / 'verifiers/stage2_verifiers.py')
    shutil.copy2(SPEC, output / 'generation_requirements.md')


def process_jobs(client, output, scale, workers):
    jobs = make_jobs(scale)
    for job in jobs:
        destination = output / 'generated_jobs' / (job['id'] + '.json')
        if destination.exists():
            continue
        for raw in sorted((output / 'raw').glob(job['id'] + '-*.json')):
            body = json.loads(raw.read_text())
            try:
                choice = body['choices'][0]
                items = json.loads(choice['message']['content'])['items']
                if choice['finish_reason'] == 'stop' and len(items) == job['count']:
                    write_json(destination, {'job': job, 'items': items, 'model': body['model'],
                        'prompt_sha256': sha(prompt_for(job).encode()), 'recovered_request_id': body['id']})
                    break
            except (KeyError, TypeError, ValueError):
                continue
    done = {path.stem for path in (output / 'generated_jobs').glob('*.json')}
    pending = [job for job in jobs if job['id'] not in done]
    print(f'Generating {len(pending)} requests; completed {len(done)}', flush=True)
    iterator = iter(pending)
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        for _ in range(workers):
            job = next(iterator, None)
            if job:
                active[pool.submit(generate_job, client, job)] = job
        while active:
            completed, _ = concurrent.futures.wait(active,
                return_when=concurrent.futures.FIRST_COMPLETED, timeout=20)
            if not completed:
                print(f'Generation active={len(active)}, completed={len(done)}, peak-price estimate CNY={client.spent:.3f}', flush=True)
            for future in completed:
                job = active.pop(future)
                try:
                    result = future.result()
                    write_json(output / 'generated_jobs' / (job['id'] + '.json'), result)
                    done.add(job['id'])
                    print(f'Generated {len(done)}/{len(jobs)} {job["id"]}', flush=True)
                except Exception as exc:
                    error = {'job': job['id'], 'type': type(exc).__name__, 'reason': str(exc)[:200]}
                    errors.append(error)
                    append(output / 'generation_errors.jsonl', error)
                    print('Generation issue', error, flush=True)
                if not client.stopped:
                    following = next(iterator, None)
                    if following:
                        active[pool.submit(generate_job, client, following)] = following
            write_json(output / 'status.json', {'phase': 'generation', 'completed_jobs': len(done),
                'planned_jobs': len(jobs), 'conservative_cost_cny': client.spent, 'updated': timestamp()})
    return errors


def assemble(client, output, scale=1):
    batch = output.name
    guards = protected_index()
    print('Protection index ready', len(guards.entries), flush=True)
    generated = []
    for path in sorted((output / 'generated_jobs').glob('*.json')):
        job = json.loads(path.read_text())
        for index, original in enumerate(job['items']):
            item = dict(original, task_type=job['job']['task'], language=job['job']['language'])
            if job['job']['kind'] == 'crypto':
                try:
                    crypto_truth(item)
                except (ValueError, KeyError, TypeError, SyntaxError, IndexError) as exc:
                    item['precheck_issue'] = str(exc)[:300]
            generated.append((job, index, item))
    records, quarantine, review_log, families = [], [], [], {}
    def run_review(offset):
        tag = f'review-{offset:05d}'
        cache = output / 'raw' / (tag + '.parsed.json')
        chunk = generated[offset:offset + 5]
        view = [dict(item, index=i) for i, (_, _, item) in enumerate(chunk)]
        context = sha(json.dumps(view, sort_keys=True, ensure_ascii=False).encode())
        if cache.exists() and json.loads(cache.read_text()).get('items_sha256') == context:
            return
        try:
            reviews = review_items(client, view, tag + '-' + context[:10])
            write_json(cache, {'items_sha256': context, 'reviews': reviews})
            print(f'Review batch {offset}/{len(generated)} complete; estimate CNY={client.spent:.3f}', flush=True)
        except Exception as exc:
            print(f'Review batch {offset} issue: {type(exc).__name__} {str(exc)[:120]}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(run_review, range(0, len(generated), 5)))
    for offset in range(0, len(generated), 5):
        chunk = generated[offset:offset + 5]
        tag = f'review-{offset:05d}'
        try:
            cached = json.loads((output / 'raw' / (tag + '.parsed.json')).read_text())
            reviews = {int(k): v for k, v in cached['reviews'].items()}
        except Exception as exc:
            reviews = {i: {'decision': 'uncertain', 'issues': [str(exc)[:200]],
                           'model': 'review_incomplete'} for i in range(len(chunk))}
        for i, (job, number, item) in enumerate(chunk):
            kind = job['job']['kind']
            identity = job['job']['id'] + '-' + str(number)
            review = reviews[i]
            review_log.append(dict(review, id=identity))
            try:
                if review['decision'] != 'accept':
                    raise ValueError('semantic_review_' + review['decision'])
                matches = guards.search(normalized(json.dumps(item, ensure_ascii=False)), protected=True)
                if matches:
                    raise ValueError('protected_overlap:' + ','.join(m['reference'] for m in matches[:3]))
                record_kind = 'grpo' if kind in ('finance_grpo', 'crypto') else kind
                row = common_record(output, batch, identity, record_kind, item, job['model'])
                if item.get('annotation_corrections'):
                    row['annotation_corrections'] = item['annotation_corrections']
                row['review'] = {'status': 'accepted', 'reviewer_ids': [review['model']],
                    'checks': ['schema', 'lexical_protected_guard', 'independent_teacher_review'],
                    'issues': review.get('issues', []), 'scope': 'automated_teacher_review_not_expert_certification'}
                if kind == 'sft_text':
                    row['messages'] = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': item['user']},
                                       {'role': 'assistant', 'content': item['answer']}]
                elif kind == 'dpo':
                    if item['chosen'].strip() == item['rejected'].strip():
                        raise ValueError('Identical preference candidates')
                    row.update(prompt=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': item['user']}],
                        chosen=[{'role': 'assistant', 'content': item['chosen']}],
                        rejected=[{'role': 'assistant', 'content': item['rejected']}],
                        preference={'label': 'chosen', 'reason': item['reason'], 'dimensions': item['dimensions'],
                            'error_tags': item['error_tags'], 'chosen_origin': 'teacher_synthetic',
                            'rejected_origin': 'teacher_synthetic', 'confidence': 'high', 'evidence_refs': []})
                elif kind == 'finance_grpo':
                    actual = financial_answer(item['operation'], item['parameters'])
                    from decimal import Decimal
                    tolerance = Decimal('.01') if item['unit'] == 'CNY' else Decimal('.00000001')
                    if abs(actual - Decimal(str(item['value']))) > tolerance:
                        raise ValueError('Teacher numeric answer disagrees with recomputation')
                    row.update(prompt=[{'role': 'user', 'content': item['user']}], tools=[],
                        private_ground_truth={'operation': item['operation'], 'parameters': item['parameters'],
                            'expected': str(actual), 'absolute_tolerance': str(tolerance), 'unit': item['unit']},
                        reference_completion=json.dumps({'value': str(actual), 'unit': item['unit']}),
                        verifier={'id': 'finance_decimal_v1', 'entrypoint': 'verifiers/stage2_verifiers.py:verify',
                                  'version': '1', 'reward_range': [-1, 1]},
                        execution={'status': 'verified', 'run_ids': [identity]})
                    good = verify(row, {'completion': row['reference_completion']})
                    bad = verify(row, {'completion': json.dumps({'value': str(actual + 100), 'unit': item['unit']})})
                    if good['reward'] != 1 or bad['reward'] >= good['reward']:
                        raise ValueError('Financial verifier control failed')
                    write_json(output / 'execution_logs' / identity / 'verification.json', {'good': good, 'bad': bad})
                else:
                    truth = crypto_truth(item)
                    artifact = output / 'artifacts' / identity
                    for variant, files in [('original', item['files']), ('secure', item['safe_files'])]:
                        for file in files:
                            destination = artifact / variant / file['path']
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_text(file['code'])
                    row.update(prompt=[{'role': 'user', 'content': item['user'] + '\n' + json.dumps(item['files'], ensure_ascii=False)}],
                        tools=[], private_ground_truth=truth,
                        reference_completion=json.dumps({'findings': item['findings']}, ensure_ascii=False),
                        artifact_refs=[str(artifact.relative_to(output))],
                        verifier={'id': 'crypto_detection_v1', 'entrypoint': 'verifiers/stage2_verifiers.py:verify',
                                  'version': '1', 'reward_range': [-1, 1]},
                        execution={'status': 'verified_static_only', 'run_ids': [identity]},
                        capability_scope='detection_and_evidence; repair_runtime_tests_pending')
                    row['task_type'] = 'crypto_detection'
                    row['requested_task_type'] = item['task_type']
                    good = verify(row, {'completion': row['reference_completion']})
                    if good['reward'] != 1:
                        raise ValueError('Crypto verifier rejected reference findings')
                    candidates = [f for file in item['files'] for f in static_candidates(file['code'], file['path'])]
                    write_json(output / 'execution_logs' / identity / 'verification.json',
                        {'reference_reward': good, 'static_candidates': candidates,
                         'source_sha256': {file['path']: sha(file['code'].encode()) for file in item['files']},
                         'scope': 'AST parse, evidence, reward and actual static tool; runtime not executed'})
                if item['task_type'] == 'privacy_behavior':
                    row['privacy_context'] = item['privacy_context']
                records.append(row)
                families.setdefault(row['family_id'], []).append(row['id'])
            except Exception as exc:
                quarantine.append({'id': identity, 'reason': str(exc)[:400], 'review': review,
                                   'source_job': job['job']['id']})
        if offset % 25 == 0:
            print(f'Reviewed {min(offset+5,len(generated))}/{len(generated)}, accepted={len(records)}, isolated={len(quarantine)}, peak estimate CNY={client.spent:.3f}', flush=True)
            write_json(output / 'status.json', {'phase': 'review', 'reviewed': min(offset+5,len(generated)),
                'generated': len(generated), 'accepted': len(records), 'quarantine': len(quarantine),
                'conservative_cost_cny': client.spent, 'updated': timestamp()})
    finalize(output, records, quarantine, review_log, families, client, scale)


def finalize(output, records, quarantine, review_log, families, client, scale=1):
    # Derived audit demonstrations share the exact base-case family and source ledger.
    crypto = [row for row in records if row['record_type'] == 'grpo' and row['task_type'] == 'crypto_detection']
    for index, row in enumerate(crypto[:100 * scale]):
        files = row['private_ground_truth']['files']
        calls, messages, evidence = [], [dict(row['prompt'][0])], []
        for number, file in enumerate(files):
            path = output / row['artifact_refs'][0] / 'original' / file['path']
            content = path.read_text()
            call_id = f'read_{number}'
            messages.extend([{'role': 'assistant', 'content': '', 'tool_calls': [{'id': call_id,
                'type': 'function', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': file['path']})}}]},
                {'role': 'tool', 'tool_call_id': call_id, 'content': content}])
            evidence.append({'tool': 'read_file', 'path': file['path'], 'sha256': sha(content.encode())})
        messages.append({'role': 'assistant', 'content': row['reference_completion']})
        tool = dict(row, id=row['id'] + '-tool', record_type='sft_tool', parent_ids=[row['id']],
                    task_type='agent_tool_use', messages=messages,
                    tools=[{'type': 'function', 'function': {'name': 'read_file',
                        'description': '读取受控工作区源码', 'parameters': {'type': 'object',
                        'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}])
        for key in ('prompt', 'private_ground_truth', 'reference_completion', 'verifier'):
            tool.pop(key, None)
        records.append(tool)
        families[row['family_id']].append(tool['id'])
        write_json(output / 'execution_logs' / (row['id'] + '-tool') / 'tools.json', evidence)
    for index, row in enumerate(crypto[:100 * scale]):
        findings = row['private_ground_truth']['findings']
        if findings:
            rejected = json.dumps({'findings': []})
        else:
            file = row['private_ground_truth']['files'][0]
            lines = file['code'].splitlines()
            rejected = json.dumps({'findings': [{'rule_id': 'HARDCODED_KEY', 'file': file['path'],
                'line': 1, 'evidence': lines[0], 'reason': '未核对用途就认定存在硬编码生产密钥。',
                'fix': '删除该行。'}]}, ensure_ascii=False)
        pair = dict(row, id=row['id'] + '-dpo', record_type='dpo', parent_ids=[row['id']],
                    chosen=[{'role': 'assistant', 'content': row['reference_completion']}],
                    rejected=[{'role': 'assistant', 'content': rejected}],
                    preference={'label': 'chosen', 'reason': '选择符合复核真值与源码证据的判断。',
                        'dimensions': ['correctness', 'evidence'], 'error_tags': ['false_negative' if findings else 'false_positive'],
                        'chosen_origin': 'teacher_synthetic', 'rejected_origin': 'derived',
                        'confidence': 'high', 'evidence_refs': row['artifact_refs']})
        for key in ('private_ground_truth', 'reference_completion', 'verifier', 'tools'):
            pair.pop(key, None)
        records.append(pair)
        families[row['family_id']].append(pair['id'])
    names = {'sft_text': 'sft_text', 'sft_tool': 'sft_tools', 'dpo': 'dpo_pairs', 'grpo': 'grpo_tasks'}
    for filename in names.values():
        (output / (filename + '.jsonl')).write_text('')
    for row in records:
        append(output / (names[row['record_type']] + '.jsonl'), row)
    for name, rows in [('review_results', review_log), ('quarantine', quarantine),
                       ('families', [{'family_id': family, 'member_ids': ids, 'split': 'pending_global_clustering'}
                                     for family, ids in families.items()])]:
        (output / (name + '.jsonl')).write_text('')
        for row in rows:
            append(output / (name + '.jsonl'), row)
    (output / 'attack_seeds.jsonl').write_text('')
    for row in [r for r in records if r['record_type'] == 'sft_text' and r['task_type'] == 'safety_boundary'][:60 * scale]:
        append(output / 'attack_seeds.jsonl', {'seed_id': row['id'] + '-seed', 'family_id': row['family_id'],
            'parent_ids': [row['id']], 'scope': 'train_candidate', 'request': row['messages'][-2]['content'],
            'language': row['language'], 'risk_tags': row['tags'], 'provenance': row['provenance'],
            'review': row['review'], 'attack_execution': 'pending_A100'})
    from collections import Counter
    manifest = {'status': 'teacher_candidates_automatically_reviewed_pending_global_acceptance',
        'updated': timestamp(), 'generation_spec_sha256': sha(SPEC.read_bytes()),
        'counts': dict(Counter(row['record_type'] for row in records)),
        'task_counts': dict(Counter(row['task_type'] for row in records)), 'families': len(families),
        'quarantined': len(quarantine), 'conservative_cost_cny': client.spent,
        'budget_policy': 'balance-polled actual spending with reserved outstanding requests',
        'split': 'pending_combined_near_duplicate_family_clustering',
        'limitations': ['teacher review requires expert sampling', 'token-length acceptance pending',
                        'crypto repair and runtime tests pending; detection rewards only',
                        'GCG/PAIR artifacts require actual A100 execution'],
        'files': {str(p.relative_to(output)): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
                  for p in output.rglob('*') if p.is_file() and p.name not in ('manifest.json', 'status.json')}}
    try:
        manifest['balance_after'] = client.balance()
    except Exception:
        manifest['balance_after'] = None
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'status.json', dict(manifest, phase='complete_candidate_batch'))
    write_json(ROOT / 'reports/stage2_deepseek_generation_latest.json', {'output': str(output), **manifest})
    (output / 'README.md').write_text('# DeepSeek 第二阶段教师候选数据\n\n'
        '按随包规范生成，Flash 撰写、Flash 独立请求复核，并执行数值或静态证据检查。\n\n'
        '实际数量、费用估计和余额见 manifest.json。隐私与正式评测保护材料只用于本地去污。\n\n'
        '集合仍需全局家族划分与专家抽查。代码任务已验证检测奖励，修复运行测试待完成。\n')
    print(json.dumps({'output': str(output), 'counts': manifest['counts'],
                      'quarantined': len(quarantine), 'conservative_cost_cny': client.spent}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scale', type=int, default=1, choices=(1, 10))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--budget-cny', type=float, default=28)
    parser.add_argument('--assemble-only', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    prepare_bundle(args.output)
    key = os.environ.get('DEEPSEEK_API_KEY') or getpass.getpass('DeepSeek API key: ')
    client = Client(key, args.output, args.budget_cny)
    if not (args.output / 'balance_before.json').exists():
        write_json(args.output / 'balance_before.json', client.balance())
    if not args.assemble_only:
        process_jobs(client, args.output, args.scale, args.workers)
    assemble(client, args.output, args.scale)


if __name__ == '__main__':
    main()
