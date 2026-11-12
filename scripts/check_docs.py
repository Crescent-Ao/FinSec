"""Check published links, aggregate arithmetic and demo export boundaries."""
import json
import math
from pathlib import Path
import re
from decimal import Decimal, ROUND_HALF_UP

ROOT = Path(__file__).resolve().parents[1]


def main():
    import subprocess
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT, text=True).split('\0')
    markdown = [ROOT / name for name in names if name.endswith('.md')]
    for path in markdown:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith(('#', 'mailto:')):
                assert (path.parent / target.split('#')[0]).exists(), (path.name, target)
    data = json.loads((ROOT / 'docs/results/metrics.json').read_text())
    english = (ROOT / 'README.en.md').read_text()
    def percent(value):
        return str(Decimal(str(value * 100)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)) + '%'
    for stage in data['stages']:
        finance, agent = stage['finance'], stage['agent']
        assert math.isclose(finance['accuracy'], finance['correct'] / finance['total'])
        for task in ('gcg', 'pair'):
            item = stage[task]
            if item:
                assert math.isclose(item['asr'], item['successes'] / item['behaviors'])
                assert percent(item['asr']) in english
        assert math.isclose(agent['precision'], agent['TP'] / (agent['TP'] + agent['FP']))
        assert math.isclose(agent['recall'], agent['TP'] / (agent['TP'] + agent['FN']))
        assert percent(finance['accuracy']) in english
        for key in ('precision', 'recall', 'f1'):
            assert percent(agent[key]) in english
    cases = json.loads((ROOT / 'demo/cases.json').read_text())
    assert len(cases) == 3
    for case in cases:
        assert 'expected' not in case and 'family' not in case and 'source_record' not in case
        assert case['model'] == 'FinSec' and case['model_inference_performed'] is True
        assert case['result']['external_model_calls'] == 0
        assert any(t['kind'] == 'tool' for t in case['result']['trace'])
    print(f'PASS: {len(markdown)} Markdown files, aggregate arithmetic, English result values and three bounded demo exports.')


if __name__ == '__main__':
    main()
