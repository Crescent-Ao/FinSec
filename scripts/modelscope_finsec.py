"""Download the published FinSec adapter or run its canonical NF4 inference."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = 'aiops11/Qwen3.5-9B-FinSec'
REVISION = 'finsec-nf4-v1.0.0'


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate_adapter(directory):
    directory = Path(directory)
    release = json.loads((directory / 'finsec_release.json').read_text())
    adapter = release['adapter']
    for filename, expected in [(adapter['file'], adapter['sha256']), (adapter['config_file'], adapter['config_sha256'])]:
        target = (directory / filename).resolve()
        if target.parent != directory.resolve() or digest(target) != expected:
            raise ValueError('Published adapter integrity check failed: ' + filename)
    if release['format'] != 'unmerged_peft_lora' or release['loading']['quantization'] != 'bitsandbytes_nf4':
        raise ValueError('This loader expects the original NF4 base plus an unmerged adapter')
    return release


def download(directory, revision=REVISION, base_directory=None):
    from modelscope_hub import HubApi
    api = HubApi(endpoint='https://modelscope.cn')
    directory = Path(directory)
    api.download_repo(REPO, 'model', revision=revision, local_dir=directory,
                      allow_patterns=['adapter_config.json', 'adapter_model.safetensors', 'finsec_release.json'])
    release = validate_adapter(directory)
    if base_directory:
        base = release['base_model']
        api.download_repo(base['modelscope_id'], 'model', revision=base['modelscope_revision'],
                          local_dir=base_directory, allow_patterns=[row['file'] for row in base['files']])
        verify_base(base_directory, release)
    return release


def verify_base(directory, release):
    directory = Path(directory)
    for row in release['base_model']['files']:
        target = (directory / row['file']).resolve()
        if directory.resolve() not in target.parents or digest(target) != row['sha256']:
            raise ValueError('Base model differs from the evaluated checkpoint: ' + row['file'])


def infer(adapter_directory, base_directory, prompt, max_new_tokens=256, verify=False):
    release = validate_adapter(adapter_directory)
    if verify:
        verify_base(base_directory, release)
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('Canonical NF4 inference requires a supported CUDA environment')
    tokenizer = AutoTokenizer.from_pretrained(base_directory, local_files_only=True)
    base = AutoModelForImageTextToText.from_pretrained(
        base_directory, local_files_only=True, dtype=torch.bfloat16, device_map={'': 0},
        attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(base, adapter_directory).eval()
    model.requires_grad_(False)
    torch.manual_seed(42)
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda:0')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                use_cache=True, pad_token_id=tokenizer.pad_token_id)
    answer = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return {'model': REPO, 'version': release['version'], 'precision': 'nf4',
            'unmerged_adapter': True, 'is_loaded_in_4bit': bool(getattr(base, 'is_loaded_in_4bit', False)),
            'answer': answer, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('download')
    fetch.add_argument('--output', default='models/FinSec')
    fetch.add_argument('--revision', default=REVISION)
    fetch.add_argument('--base-output', help='Also download the pinned ~19GB base model')
    run = sub.add_parser('infer')
    run.add_argument('--adapter-dir', required=True)
    run.add_argument('--base-dir', required=True)
    run.add_argument('--prompt', default='请解释债券久期与利率变化之间的关系。')
    run.add_argument('--max-new-tokens', type=int, default=256)
    run.add_argument('--verify-base', action='store_true')
    args = parser.parse_args()
    if args.command == 'download':
        release = download(args.output, args.revision, args.base_output)
        print(json.dumps({'adapter_directory': str(Path(args.output).resolve()), 'version': release['version'],
                          'integrity_verified': True, 'base_downloaded': bool(args.base_output)}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(infer(args.adapter_dir, args.base_dir, args.prompt, args.max_new_tokens, args.verify_base), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
