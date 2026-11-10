"""Audit SFT and DPO training branches against fixed privacy nonmembers."""
from .core import fingerprint
from .hardening_data import normalize, user_prompt


def training_nonmember_overlap(training, nonmembers, algorithm):
    if algorithm not in ('sft', 'dpo'):
        raise ValueError('Unsupported safety training exposure format')
    exact = {fingerprint(row['messages']) for row in nonmembers}
    prompts = {normalize(user_prompt(row)) for row in nonmembers}
    prompts.discard('')
    result = {'exact_message_overlap': 0, 'user_prompt_overlap': 0}
    if algorithm == 'dpo':
        result['rejected_message_overlap'] = 0
    for row in training:
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'assistant':
            raise ValueError('Safety training messages must contain the chosen assistant response')
        result['exact_message_overlap'] += fingerprint(messages) in exact
        prompt = normalize(user_prompt(row))
        result['user_prompt_overlap'] += bool(prompt and prompt in prompts)
        if algorithm == 'dpo':
            rejected = row.get('rejected_response')
            if not isinstance(rejected, str) or not rejected.strip():
                raise ValueError('DPO privacy audit requires its actual rejected response')
            other = messages[:-1] + [{'role': 'assistant', 'content': rejected}]
            result['rejected_message_overlap'] += fingerprint(other) in exact
    return result
