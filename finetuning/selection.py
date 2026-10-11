"""Pure retention-policy checks, shared by initial and resumed selection."""
import math

REQUIRED_FAMILIES = ('massive', 'banking', 'nli', 'trec', 'sst2', 'boolean', 'score')


def validate_thresholds(*values):
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Retention thresholds must be finite and nonnegative')


def validate_metrics(metrics):
    for key in ('typed_acc', 'typed_ce', 'retention_acc'):
        v = metrics.get(key)
        if v is None or not math.isfinite(v):
            raise ValueError(f'Missing or nonfinite selection metric: {key}')
    families = metrics.get('families', {})
    for name in REQUIRED_FAMILIES:
        key = f'retention/{name}'
        metric = 'ordinal_mae' if name == 'score' else 'accuracy'
        value = families.get(key, {}).get(metric)
        if value is None or not math.isfinite(value):
            raise ValueError(f'Missing or nonfinite retention metric: {key}/{metric}')


def retention_violations(base, current, global_drop, family_drop, mae_increase):
    validate_thresholds(global_drop, family_drop, mae_increase)
    validate_metrics(base)
    validate_metrics(current)
    violations = []
    if base['retention_acc'] - current['retention_acc'] > global_drop + 1e-12:
        violations.append('Global retention accuracy drop')
    for name in REQUIRED_FAMILIES:
        key = f'retention/{name}'
        b, c = base['families'][key], current['families'][key]
        if name == 'score':
            if c['ordinal_mae'] - b['ordinal_mae'] > mae_increase + 1e-12:
                violations.append('Ordinal MAE increase')
        elif b['accuracy'] - c['accuracy'] > family_drop + 1e-12:
            violations.append(f'{key} accuracy drop')
    return violations


def improves(acc, ce, best_acc, best_ce):
    return acc > best_acc + 1e-12 or (abs(acc-best_acc) <= 1e-12 and ce < best_ce)
