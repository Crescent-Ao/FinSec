"""JSONL bridge: Pi executes tools; this process owns one local NF4 model.

Only the active case's public source and rules are sent here, never its labels.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fineval_pipeline.agent import parse_findings, static_candidates
from fineval_pipeline.core import Backend, model_identity


def tool_result(name, args, public):
    if name in {"workspace_info","list_files","read_file","write_file","replace_text","run_command"}:
        from pi_crypto_agent.workspace_tools import workspace_tool
        return workspace_tool(name,args,public)
    if name == "run_python_test":
        from pi_crypto_agent.sandbox import run_python_test
        return run_python_test(args, public)
    filename, code, rules = public['file'], public['code'], public['knowledge']
    if name == 'read_source':
        return {'file': filename, 'source_with_lines': '\n'.join(f'{i}: {s}' for i, s in enumerate(code.splitlines(), 1))}
    if name == 'static_scan':
        return {'candidates': static_candidates(code, filename), 'scope': 'candidates require model judgment'}
    if name == 'search_code':
        query = args['query']
        if not isinstance(query, str) or not 1 <= len(query) <= 100:
            raise ValueError('Search requires a bounded literal string')
        return {'matches': [{'file': filename, 'line': i, 'text': s} for i, s in enumerate(code.splitlines(), 1) if query.lower() in s.lower()][:50]}
    if name == 'lookup_rules':
        selected = args.get('rule_ids', [])
        return {'rules': [r for r in rules if not selected or r['id'] in selected]}
    raise ValueError('Unknown read-only audit tool')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec['precision'] != 'nf4':
        raise ValueError('Pi comparison requires the original NF4 protocol')
    backend = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            op = request['op']
            with contextlib.redirect_stdout(sys.stderr):
                if op == 'generate':
                    if backend is None:
                        backend = Backend(spec)
                    rendered=backend.tokenizer.apply_chat_template(request['messages'],tools=request.get('tools') or None,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    text = backend.generate_raw(rendered,max_new_tokens=request['max_new_tokens'],max_length=8192)
                    result = {'text': text}
                elif op == 'identity':
                    result = model_identity(spec)
                elif op == 'tool':
                    result = tool_result(request['name'], request.get('arguments', {}), request['public'])
                elif op == 'validate':
                    public = request['public']
                    result = {'findings': parse_findings(request['text'], public['file'], public['code'], {r['id'] for r in public['knowledge']})}
                else:
                    raise ValueError('Unknown bridge operation')
            response = {'id': request['id'], 'result': result}
        except Exception as error:
            response = {'id': request['id'], 'error': type(error).__name__ + ': ' + str(error)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
