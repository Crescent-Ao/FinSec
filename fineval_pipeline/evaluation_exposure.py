"""Exact reference matching that supplements the legacy40-character fuzzy guard."""
from collections import defaultdict
import re

from scripts.validate_pilot_data import normalized


class ExactFinancialReferenceIndex:
    def __init__(self, records):
        patterns = defaultdict(list)
        for row in records:
            key = row['subject'] + ':' + row['id']
            question = row['question']
            forms = {'question': question,
                     'with_options': question + ' ' + ' '.join(row['options'][k] for k in 'ABCD'),
                     'with_lettered_options': question + ' ' + ' '.join(k + ' ' + row['options'][k] for k in 'ABCD'),
                     'with_dotted_options': question + ' ' + ' '.join(k + '. ' + row['options'][k] for k in 'ABCD')}
            for kind, text in forms.items():
                text = normalized(text)
                if len(text) >= 12:
                    patterns[text].append({'reference': key, 'kind': kind, 'normalized_length': len(text)})
        if not patterns:
            raise ValueError('No exact reference patterns')
        self.patterns = patterns
        self.only_cjk_patterns = all(re.search('[\u4e00-\u9fff]', text) for text in patterns)
        self.regex = re.compile('|'.join(re.escape(text) for text in sorted(patterns, key=lambda x: (-len(x), x))))

    def search(self, normalized_text):
        if self.only_cjk_patterns and not re.search('[\u4e00-\u9fff]', normalized_text):
            return []
        matches, seen = [], set()
        for found in self.regex.finditer(normalized_text):
            for item in self.patterns[found.group()]:
                key = (item['reference'], item['kind'])
                if key not in seen:
                    matches.append(item)
                    seen.add(key)
        return matches
