"""Deterministic rewards for generated financial and crypto-detection tasks."""
import json
from decimal import Decimal, localcontext
from fineval_pipeline.serialization import strip_empty_thinking_prefix


def financial_answer(operation, parameters):
    """Recompute answers from public parameters with explicit conventions."""
    with localcontext() as context:
        context.prec = 40
        values = {key: ([Decimal(str(item)) for item in value] if isinstance(value, list)
                        else Decimal(str(value))) for key, value in parameters.items()}
        if operation == 'simple_interest':
            return values['principal'] * values['rate'] * values['days'] / values['basis']
        if operation == 'compound_future':
            periods = values['periods']
            if periods != int(periods) or not 0 <= periods <= 100:
                raise ValueError('Invalid integer compounding periods')
            return values['principal'] * (1 + values['rate']) ** int(periods)
        if operation == 'npv':
            return sum(amount / (1 + values['rate']) ** index
                       for index, amount in enumerate(values['cashflows']))
        if operation == 'bond_price':
            periods = values['periods']
            if periods != int(periods) or not 1 <= periods <= 100:
                raise ValueError('Invalid integer bond periods')
            coupon = values['face'] * values['coupon_per_period']
            return (sum(coupon / (1 + values['yield_per_period']) ** index
                        for index in range(1, int(periods) + 1))
                    + values['face'] / (1 + values['yield_per_period']) ** int(periods))
        if operation == 'equity':
            return values['assets'] - values['liabilities']
        if operation == 'current_ratio':
            return values['current_assets'] / values['current_liabilities']
        if operation == 'portfolio_variance':
            weight = values['weight_a']
            return (weight ** 2 * values['sigma_a'] ** 2
                    + (1 - weight) ** 2 * values['sigma_b'] ** 2
                    + 2 * weight * (1 - weight) * values['rho'] * values['sigma_a'] * values['sigma_b'])
        if operation == 'call_payoff':
            return max(values['spot'] - values['strike'], Decimal(0)) * values['quantity']
        if operation == 'forward_payoff':
            return (values['spot'] - values['delivery']) * values['quantity']
        if operation == 'mean':
            return sum(values['observations']) / len(values['observations'])
        raise ValueError('Unsupported operation')


def strict_json(text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(strip_empty_thinking_prefix(text), object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def verify(task, candidate, workspace=None):
    """Evaluate a completion against private truth without executing model code."""
    try:
        output = strict_json(candidate['completion'])
        truth = task['private_ground_truth']
        if task['task_type'] == 'finance_numeric':
            expected = financial_answer(truth['operation'], truth['parameters'])
            value = Decimal(str(output['value']))
            tolerance = Decimal(truth['absolute_tolerance'])
            if not value.is_finite():
                raise ValueError('Nonfinite numeric answer')
            correct = abs(value - expected) <= tolerance
            unit = output.get('unit') == truth['unit']
            format_valid = set(output) == {'value', 'unit'}
            reward = .9 * float(correct and unit) + .1 * float(format_valid)
            return {'valid': True, 'reward': reward,
                    'metrics': {'answer_correct': float(correct), 'unit_correct': float(unit),
                                'format_valid': float(format_valid)},
                    'errors': [], 'verifier_version': '1'}
        expected = {(row['file'], row['rule_id'], row['line']) for row in truth['findings']}
        source = {row['path']: row['code'].splitlines() for row in truth['files']}
        findings = output['findings']
        predicted = set()
        for row in findings:
            lines = source[row['file']]
            if type(row['line']) is not int or not 1 <= row['line'] <= len(lines):
                raise ValueError('Invalid source line')
            if row['evidence'].strip() != lines[row['line'] - 1].strip():
                raise ValueError('Source evidence mismatch')
            if not row.get('reason', '').strip() or not row.get('fix', '').strip():
                raise ValueError('Missing finding explanation')
            predicted.add((row['file'], row['rule_id'], row['line']))
        if len(predicted) != len(findings):
            raise ValueError('Duplicate findings')
        denominator = len(expected) + len(predicted)
        f1 = 2 * len(expected & predicted) / denominator if denominator else 1.0
        return {'valid': True, 'reward': .8 * f1 + .2,
                'metrics': {'finding_f1': f1, 'evidence_score': 1.0, 'protocol_score': 1.0},
                'errors': [], 'verifier_version': '1'}
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError) as exc:
        return {'valid': False, 'reward': -1.0, 'metrics': {},
                'errors': [str(exc)[:300]], 'verifier_version': '1'}
