"""Loopback/SSH transport for the unchanged official PAIR target interface."""
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
import urllib.parse
import uuid


class TargetRPCError(RuntimeError):pass


def validate_dispatch(dispatch):
    groups=dispatch['groups']
    if dispatch.get('layout') in ['single_GRPO_one_A100','single_GRPO_one_RTX4090']:
        if len(groups)!=1 or set(groups[0])!={'grpo'} or not isinstance(groups[0]['grpo'],str) or not groups[0]['grpo']:
            raise ValueError('Expected exactly one explicit GRPO target endpoint')
        return
    if dispatch.get('layout')=='two_targets_two_RTX4090':
        if len(groups)!=1 or set(groups[0])!={'finance','hardened'}:
            raise ValueError('Expected one independent financial/hardened pair')
        endpoints=list(groups[0].values())
        if any(not isinstance(x,str) or not x for x in endpoints) or len(set(endpoints))!=2:
            raise ValueError('Two local targets require separate explicit endpoints')
        return
    if dispatch.get('layout')=='four_targets_one_A100':
        if len(groups)!=2 or any(set(group)!={'finance','hardened'} for group in groups):
            raise ValueError('Expected two independent financial/hardened target pairs')
        endpoints=[value for group in groups for value in group.values()]
        if any(value is None for value in endpoints) or len(set(endpoints))!=4:
            raise ValueError('Four A100 replicas require four separate endpoints')
        return
    if len(groups)!=2 or any(set(group)!={'base_nf4','dpo'} for group in groups):
        raise ValueError('Expected two independent pairs of PAIR target models')
    slots=[value for group in groups for value in group.values()]
    endpoints=[value for value in slots if value is not None]
    if len(set(endpoints))!=3 or slots.count(None) not in (0,1):
        raise ValueError('Expected three RTX4090 endpoints and at most one A100 model')
    ownership={}
    for group in groups:
        for variant,endpoint in group.items():
            if endpoint is None:continue
            if endpoint in ownership and ownership[endpoint]!=variant:raise ValueError('An endpoint cannot serve two model variants')
            ownership[endpoint]=variant


class TargetRPC:
    def __init__(self,endpoint,variant,token_file,expected):
        self.endpoint=endpoint.rstrip('/');self.variant=variant
        address=urllib.parse.urlsplit(self.endpoint)
        if address.scheme!='http' or address.hostname not in ('127.0.0.1','localhost') or address.username or address.path:
            raise ValueError('PAIR target RPC must use an SSH loopback endpoint')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.token=Path(token_file).read_text().strip()
        self.expected=expected
        health=self.request('/health')
        if health['variant']!=variant or health['model_identity']!=expected['model_identity']:
            raise ValueError('PAIR target endpoint has another model or variant')
        if health['software']!=expected['software'] or not health['is_loaded_in_4bit']:
            raise ValueError('PAIR target endpoint precision or software differs')
        if health['upstream_hf_source_sha256']!=expected['hf_source_sha256']:
            raise ValueError('PAIR target endpoint uses another generation implementation')
        self.health=health

    def request(self,path,payload=None):
        data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None
        req=urllib.request.Request(self.endpoint+path,data=data,headers={
            'Content-Type':'application/json','Authorization':'Bearer '+self.token})
        try:
            with self.opener.open(req,timeout=1800) as response:result=json.load(response)
        except (urllib.error.URLError,TimeoutError) as exc:
            raise TargetRPCError('PAIR target RPC failed; no local-model fallback: '+type(exc).__name__) from None
        if result.get('error'):raise TargetRPCError('PAIR target RPC: '+result['error'])
        return result

    def batched_generate(self,full_prompts_list,max_n_tokens,temperature,stop_tokens=None):
        payload={'request_id':uuid.uuid4().hex,'variant':self.variant,'prompts':full_prompts_list,
                 'max_n_tokens':max_n_tokens,'temperature':temperature,'stop_tokens':stop_tokens or []}
        result=self.request('/generate',payload)
        outputs=result['outputs']
        if len(outputs)!=len(full_prompts_list) or not all(isinstance(x,str) for x in outputs):
            raise TargetRPCError('PAIR target RPC returned incomplete generations')
        return outputs

    def is_initialized(self):return True

    def shutdown(self):return self.request('/shutdown',{})


def validate_payload(payload,variant):
    if payload.get('variant')!=variant:raise ValueError('Wrong target variant')
    prompts=payload.get('prompts')
    if not isinstance(prompts,list) or not 1<=len(prompts)<=3 or not all(isinstance(p,str) for p in prompts):
        raise ValueError('Expected one to three official PAIR target prompts')
    if payload.get('temperature')!=0 or type(payload.get('max_n_tokens')) is not int or not 1<=payload['max_n_tokens']<=512:
        raise ValueError('PAIR target must retain greedy generation and the 512-token ceiling')
    key=payload.get('request_id','')
    if len(key)!=32 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Invalid request ID')
    return hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
