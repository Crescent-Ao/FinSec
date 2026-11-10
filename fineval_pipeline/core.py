import hashlib
import json
import math
import os
import struct
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_WEIGHT_IDENTITIES = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg['_config_sha256'] = digest(path)
    return cfg


def resolve(path):
    if not path:
        raise ValueError('Required path is not configured')
    p = Path(os.path.expandvars(str(path))).expanduser()
    return p if p.is_absolute() else ROOT/p


def model_spec(cfg, name, use_export=True):
    spec = dict(cfg['models'][name])
    protocol = cfg.get('experiment_protocol', {})
    required_precision = protocol.get('required_target_precision')
    if required_precision and spec.get('precision', 'bf16') != required_precision:
        raise ValueError(f'{name} target precision must match the experiment protocol: {required_precision}')
    if protocol.get('adapter_loading') == 'unmerged' and spec.get('evaluation_export'):
        raise ValueError('This experiment requires the original quantized base plus its unmerged adapter')
    spec['path'] = str(resolve(spec.get('path')))
    if spec.get('adapter'):
        spec['adapter'] = str(resolve(spec['adapter']))
    if spec.get('stage') == 'safety':
        if spec.get('parent') != 'M1' or not spec.get('safety_manifest'):
            raise ValueError('M2 requires parent M1 and a safety_manifest recording actual enhancement and isolation')
        plan = json.loads(resolve(spec['safety_manifest']).read_text())
        if plan.get('parent') != 'M1' or plan.get('nonmembers_excluded') is not True:
            raise ValueError('M2 safety lineage/nonmember isolation is not established')
    if use_export and spec.get('evaluation_export'):
        export = resolve(spec['evaluation_export'])
        if not (export/'export_manifest.json').is_file():
            raise ValueError(f'Evaluation export not prepared for {name}; run export-model first')
        exported=json.loads((export/'export_manifest.json').read_text())
        if exported['source_identity']!=model_identity(spec):
            raise ValueError(f'Evaluation export no longer matches the registered source model: {name}')
        spec['path'] = str(export)
        spec.pop('adapter',None)
    return spec


def weight_identity(path):
    path = Path(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix == '.safetensors' or p.name.endswith('.json') or p.name.endswith('.jinja')))
    if not any(p.suffix == '.safetensors' for p in files):
        raise ValueError(f'No safetensors in {path}')
    key=(str(path),tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ino) for p in files))
    if key in _WEIGHT_IDENTITIES:
        return _WEIGHT_IDENTITIES[key]
    # Full content digests: equal index files alone do not establish equal weights.
    identity=fingerprint([(p.name, digest(p)) for p in files])
    _WEIGHT_IDENTITIES[key]=identity
    return identity


def model_identity(spec):
    identity = {'weights': weight_identity(spec['path']), 'precision': spec.get('precision', 'bf16'),
                'system_prompt': spec.get('system_prompt', '')}
    if spec.get('adapter'):
        identity['adapter'] = weight_identity(spec['adapter'])
    return identity


def parameter_count(path):
    count = 0
    for p in sorted(Path(path).glob('*.safetensors')):
        with p.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(length))
        count += sum(math.prod(v['shape']) for k, v in header.items() if k != '__metadata__')
    return count


def bind(directory, context):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'run.json'
    identity = fingerprint(context)
    record = {'fingerprint': identity, 'context': context}
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError(f'Resume refused: model/data/protocol/config changed: {directory}')
    write_json(path, record)
    return identity


class Backend:
    """Local HF inference only; also exposes differentiable inputs for GCG."""
    def __init__(self, spec):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU or external-model fallback')
        self.spec = spec
        self.torch = torch
        self.calls = 0
        self.tokenizer = AutoTokenizer.from_pretrained(spec['path'], local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        config = AutoConfig.from_pretrained(spec['path'], local_files_only=True)
        cls = AutoModelForImageTextToText if config.model_type == 'qwen3_5' else AutoModelForCausalLM
        kwargs = dict(local_files_only=True, dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa')
        if spec.get('precision', 'bf16') == 'nf4':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        elif spec.get('precision', 'bf16') != 'bf16':
            raise ValueError('precision must be bf16 or nf4')
        self.model = cls.from_pretrained(spec['path'], **kwargs)
        if spec.get('adapter'):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, spec['adapter'])
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        if hasattr(self.model.config, 'get_text_config'):
            self.model.config.get_text_config().use_cache = False

    def render(self, messages):
        messages = list(messages)
        if self.spec.get('system_prompt') and not any(m['role']=='system' for m in messages):
            messages.insert(0, {'role': 'system', 'content': self.spec['system_prompt']})
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False)

    def encode(self, text, max_length=4096):
        encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        if encoded['input_ids'].shape[-1] > max_length:
            raise ValueError('Context limit exceeded; refusing silent truncation')
        return encoded.to('cuda:0')

    def generate_raw(self, text, max_new_tokens=256, temperature=0.0, max_length=8192):
        encoded = self.encode(text, max_length-max_new_tokens)
        self.calls += 1
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0, use_cache=True,
                      pad_token_id=self.tokenizer.pad_token_id)
        if temperature > 0:
            kwargs['temperature'] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **kwargs)
        return self.tokenizer.decode(output[0, encoded['input_ids'].shape[-1]:], skip_special_tokens=True)

    def chat(self, messages, **kwargs):
        return self.generate_raw(self.render(messages), **kwargs)

    def options(self, prompt):
        encoded = self.encode(self.render([{'role': 'user', 'content': prompt}]))
        with self.torch.inference_mode():
            logits = self.model(**encoded, logits_to_keep=1, use_cache=False).logits[0, -1]
        scores = {}
        for label in 'ABCD':
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError('Option label is not one token')
            scores[label] = float(logits[ids[0]].float())
        return max(scores, key=scores.get), scores

    def answer_logprobs(self, messages, max_length=4096):
        import torch.nn.functional as F
        if messages[-1]['role'] != 'assistant':
            raise ValueError('Membership sample must end with assistant answer')
        prefix = self.render(messages[:-1])
        # One canonical scoring serialization across M0/M1/M2; EOS excluded.
        text = prefix + messages[-1]['content']
        encoded = self.tokenizer(text, return_tensors='pt', add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.pop('offset_mapping')[0].tolist()
        positions = [i for i, (a, b) in enumerate(offsets) if a >= len(prefix) and b > a and i > 0]
        if not positions or encoded['input_ids'].shape[-1] > max_length:
            raise ValueError('Empty answer or sample too long; no truncation permitted for MIA')
        encoded = encoded.to('cuda:0')
        with self.torch.inference_mode():
            base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
            # Text-only Qwen3.5: project 32 answer positions at a time instead of
            # materializing 4096 x 248320 vocabulary logits on a 24 GiB card.
            text_model = getattr(getattr(base, 'model', None), 'language_model', None)
            chunked_head = getattr(base.config, 'model_type', None) == 'qwen3_5' and text_model is not None
            if chunked_head:
                hidden = text_model(**encoded, use_cache=False).last_hidden_state[0]
                head = base.get_output_embeddings()
            else:
                logits = self.model(**encoded, use_cache=False).logits[0]
            values = []
            for start in range(0, len(positions), 32):
                idx = self.torch.tensor(positions[start:start+32], device='cuda:0')
                selected = head(hidden[idx-1]) if chunked_head else logits[idx-1]
                lp = -F.cross_entropy(selected.float(), encoded['input_ids'][0, idx], reduction='none')
                values.extend(lp.cpu().tolist())
        return values
