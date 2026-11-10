"""Literal code exposure checks that preserve raw newlines inside training text."""
import json

from .hardening_data import normalize


def text_values(value, depth=0):
    if depth > 8:
        return
    if isinstance(value, str):
        yield value
        # Audit prompts can contain a JSON object with escaped source-code newlines.
        # Parse data only; never interpret or execute the program under audit.
        stripped = value.strip()
        if stripped.startswith(('{', '[')):
            try:
                embedded = json.loads(stripped)
            except (ValueError, TypeError):
                return
            if isinstance(embedded, (dict, list)):
                yield from text_values(embedded, depth + 1)
    elif isinstance(value, list):
        for element in value:
            yield from text_values(element, depth + 1)
    elif isinstance(value, dict):
        for element in value.values():
            yield from text_values(element, depth + 1)


def full_code_overlap_ids(cases, training):
    documents = []
    for row in training:
        if not isinstance(row.get('messages'), list):
            raise ValueError('Expected actual model training messages')
        parts = list(text_values(row['messages']))
        if row.get('rejected_response') is not None:
            parts.extend(text_values(row['rejected_response']))
        documents.append(normalize('\n'.join(parts)))
    overlaps = []
    for case in cases:
        code = normalize(case['code'])
        if len(code) >= 30 and any(code in document for document in documents):
            overlaps.append(case['id'])
    return overlaps
