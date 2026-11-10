"""DeepSeek transport with a durable, process-shared prepaid request budget."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
from zoneinfo import ZoneInfo

from .core import write_json


class BudgetExceeded(RuntimeError): pass
class APITransportError(RuntimeError): pass
class APIConfigurationError(APITransportError): pass


def rates(at=None):
    at=at or datetime.datetime.now(ZoneInfo('Asia/Shanghai'))
    # Treat statutory holidays conservatively as weekdays without a calendar dependency.
    peak=at.weekday()<5 and (9<=at.hour<12 or 14<=at.hour<18)
    multiplier=2 if peak else 1
    return {'hit':.02*multiplier,'miss':1.*multiplier,'output':4.*multiplier}


def usage_cost(usage,prices):
    prompt=int(usage['prompt_tokens']);completion=int(usage['completion_tokens'])
    hit=int(usage.get('prompt_cache_hit_tokens',0))
    miss=int(usage.get('prompt_cache_miss_tokens',prompt-hit))
    if min(prompt,completion,hit,miss)<0 or hit+miss!=prompt:
        raise ValueError('Inconsistent API token accounting')
    return (hit*prices['hit']+miss*prices['miss']+completion*prices['output'])/1_000_000


class BudgetLedger:
    def __init__(self,root,limit=25.):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'ledger.json';self.lock=self.root/'ledger.lock'
        with self.lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if self.path.exists():
                if json.loads(self.path.read_text())['limit_cny']!=limit:
                    raise ValueError('Existing campaign budget differs; refusing implicit spending increase')
            else:
                write_json(self.path,{'limit_cny':limit,'spent_cny':0.,'pending':{},'settled_requests':0,
                    'by_variant':{},'policy':'reserve peak-price input-byte/token upper bound before every request; retain uncertain charges'})

    @staticmethod
    def committed(state):
        return state['spent_cny']+sum(x['upper_cny'] for x in state['pending'].values())

    def reserve(self,upper,variant,role):
        if upper<0:raise ValueError('Negative reservation')
        with self.lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);state=json.loads(self.path.read_text())
            if state.get('stop_reason'):raise BudgetExceeded(state['stop_reason'])
            if state.get('allowed_variants') is not None and variant not in state['allowed_variants']:
                raise BudgetExceeded('This API authorization is limited to the requested PAIR target variants')
            if state.get('allowed_roles') is not None and role not in state['allowed_roles']:
                raise BudgetExceeded('This API authorization does not cover that task role')
            if state['limit_cny'] is not None and self.committed(state)+upper>state['limit_cny']+1e-12:
                raise BudgetExceeded('Shared DeepSeek campaign spending ceiling reached')
            key=uuid.uuid4().hex
            state['pending'][key]={'upper_cny':upper,'variant':variant,'role':role,'time':time.time()}
            write_json(self.path,state)
            return key

    def settle(self,key,cost,metadata=None):
        if cost<0:raise ValueError('Negative charge')
        with self.lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX);state=json.loads(self.path.read_text())
            reservation=state['pending'].pop(key)
            state['spent_cny']+=cost;state['settled_requests']+=1
            variant=reservation['variant'];state['by_variant'][variant]=state['by_variant'].get(variant,0.)+cost
            if cost>reservation['upper_cny']+1e-10:
                state['stop_reason']='Actual request charge exceeded the conservative reservation'
            write_json(self.path,state)
            with (self.root/'usage.jsonl').open('a') as log:
                log.write(json.dumps({'reservation_id':key,'variant':variant,'role':reservation['role'],
                    'estimated_cny':cost,**(metadata or {})},ensure_ascii=False)+'\n')
            if state.get('stop_reason'):raise BudgetExceeded(state['stop_reason'])

    def snapshot(self):
        with self.lock.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_SH);state=json.loads(self.path.read_text())
        return {**state,'committed_upper_cny':self.committed(state),
                'remaining_unreserved_cny':None if state['limit_cny'] is None else max(0.,state['limit_cny']-self.committed(state))}


class DeepSeek:
    """Official PAIR LanguageModel interface, using the authorized API endpoint."""
    def __init__(self,variant='pilot',model='deepseek-flash',**kwargs):
        self.model_name=model;self.variant=variant
        limit=os.environ.get('HB_API_BUDGET_CNY','25')
        self.ledger=BudgetLedger(os.environ['HB_API_BUDGET_ROOT'],None if limit.lower()=='none' else float(limit))
        key_file=os.environ.get('HB_DEEPSEEK_KEY_FILE')
        self.key=Path(key_file).read_text().strip() if key_file else os.environ.get('DEEPSEEK_API_KEY')
        if not self.key:raise ValueError('DeepSeek credential is unavailable')

    def is_initialized(self):return True

    def generate(self,messages,max_n_tokens,temperature):
        role='search_judge' if max_n_tokens<=64 else 'attacker'
        payload={'model':self.model_name,'messages':messages,'max_tokens':max_n_tokens,
            'temperature':temperature,'thinking':{'type':'disabled'},'stream':False}
        if role=='attacker':payload['response_format']={'type':'json_object'}
        # Byte-level upper bound plus generous message framing. Reserve peak prices
        # even at off-peak times, and never release a timeout's uncertain charge.
        input_upper=len(json.dumps(messages,ensure_ascii=False).encode())+64*len(messages)+256
        upper=(input_upper*2+max_n_tokens*8)/1_000_000
        for attempt in range(3):
            token=self.ledger.reserve(upper,self.variant,role)
            started=time.monotonic();prices=rates()
            request=urllib.request.Request('https://api.deepseek.com/chat/completions',
                data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+self.key})
            try:
                with urllib.request.urlopen(request,timeout=120) as response:body=json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code in (401,402,403):
                    self.ledger.settle(token,0.,{'http_status':exc.code})
                    raise APIConfigurationError('DeepSeek credential/balance error: HTTP '+str(exc.code)) from None
                if exc.code==429:
                    self.ledger.settle(token,0.,{'http_status':429})
                    if attempt<2:time.sleep(2**attempt);continue
                # Other errors/timeouts retain the reservation, including retries
                # whose remote billing state cannot be established.
                raise APITransportError('DeepSeek HTTP '+str(exc.code)) from None
            except (TimeoutError,urllib.error.URLError,ValueError):
                raise APITransportError('DeepSeek request failed; its possible charge remains reserved') from None
            write_json(self.ledger.root/'responses'/(token+'.json'),{'variant':self.variant,
                'role':role,'request':payload,'response':body})
            try:
                billed_prices={k:max(v,rates()[k]) for k,v in prices.items()}
                cost=usage_cost(body['usage'],billed_prices)
            except (KeyError,TypeError,ValueError):
                raise APITransportError('DeepSeek usage missing/invalid; request charge remains reserved') from None
            self.ledger.settle(token,cost,{'usage':body['usage'],'prices_cny_per_million':billed_prices,
                'response_model':body.get('model'),'request_id':body.get('id'),
                'finish_reason':body.get('choices',[{}])[0].get('finish_reason'),
                'seconds':time.monotonic()-started,'thinking':False})
            content=body['choices'][0]['message'].get('content')
            if not isinstance(content,str):raise APITransportError('DeepSeek returned no text content')
            return content
        raise APITransportError('DeepSeek rate-limit retries exhausted')

    def batched_generate(self,prompts_list,max_n_tokens,temperature,**kwargs):
        if len(prompts_list)<=1:
            return [self.generate(x,max_n_tokens,temperature) for x in prompts_list]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(2,len(prompts_list))) as pool:
            return list(pool.map(lambda x:self.generate(x,max_n_tokens,temperature),prompts_list))
