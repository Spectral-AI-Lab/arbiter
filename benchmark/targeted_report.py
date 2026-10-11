"""Paired six-row report from frozen historical/public prediction files."""
import argparse
import json
from pathlib import Path


def load(paths):
    rows = {}
    checkpoints = set()
    for path in paths:
        meta = json.loads((path / 'metadata.json').read_text())
        if meta.get('status') != 'completed':
            raise ValueError(f'Incomplete run: {path}')
        checkpoints.add(meta.get('checkpoint_sha256'))
        for line in (path / 'predictions.jsonl').open():
            row = json.loads(line)
            key = (row['suite'], row['id'])
            if key in rows:
                raise ValueError(f'Duplicate request: {key}')
            rows[key] = row
    if len(checkpoints) != 1 or None in checkpoints:
        raise ValueError('Each variant must use one recorded checkpoint hash')
    return rows, checkpoints.pop()


def metrics(rows):
    groups = {
        'Typed Decisions': [r for r in rows if r['suite'] == 'typed_decisions'],
        'XNLI · 15 languages': [r for r in rows if r['suite'].startswith('xnli.')],
        'Fast Single': [r for r in rows if r['suite'].startswith('fast_single.')],
        'Fast Multi': [r for r in rows if r['suite'].startswith('fast_multilabel.')],
    }
    def accuracy(rs):
        total = sum(r['expected_answers'] for r in rs)
        return sum(o['correct'] for r in rs for o in r['observations'].values()) / total
    if any(not rs for rs in groups.values()):
        raise ValueError('Missing target benchmark family')
    single = groups['Fast Single']
    suites = sorted({r['suite'] for r in single})
    multi = groups['Fast Multi']
    if any('error' in r or len(r['observations']) != r['expected_answers'] for r in multi):
        raise ValueError('Fast Multi has failed/missing predictions; F1 requires complete outputs')
    tp = fp = fn = 0
    for r in multi:
        for o in r['observations'].values():
            if o['kind'] != 'noul':
                raise ValueError('Fast Multi must use binary noul decisions')
            tp += o['pred'] == 1 and o['gold'] == 1
            fp += o['pred'] == 1 and o['gold'] == 0
            fn += o['pred'] == 0 and o['gold'] == 1
    return {
        'Typed Decisions': accuracy(groups['Typed Decisions']),
        'XNLI · 15 languages': accuracy(groups['XNLI · 15 languages']),
        'Fast Single · micro': accuracy(single),
        'Fast Single · macro': sum(accuracy([r for r in single if r['suite'] == s]) for s in suites) / len(suites),
        'Fast Multi · exact match': sum(all(o['correct'] for o in r['observations'].values()) for r in multi) / len(multi),
        'Fast Multi · micro F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, nargs='+', required=True)
    p.add_argument('--adapted', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    base, bh = load(a.base)
    adapted, ah = load(a.adapted)
    # Require all frozen target requests, not a selectively evaluated subset.
    root = Path(__file__).parent / 'data'
    expected = {}
    from benchmark.run import fingerprint
    for file in [root / 'cases.jsonl', root / 'public_requested_v1/cases.jsonl']:
        for line in file.open():
            r = json.loads(line)
            if r['suite'] == 'typed_decisions' or r['suite'].startswith(('xnli.', 'fast_single.', 'fast_multilabel.')):
                expected[(r['suite'], r['id'])] = r
    selected = []
    for key, ref in expected.items():
        for variant in [base, adapted]:
            row = variant.get(key)
            if row is None or row['request_sha256'] != fingerprint({'state': ref['state'], 'questions': ref['questions']}) or row['gold_sha256'] != fingerprint(ref['gold']) or row['expected_answers'] != len(ref['gold']):
                raise ValueError(f'Missing/mismatched frozen request: {key}')
        selected.append(key)
    b, d = metrics([base[k] for k in selected]), metrics([adapted[k] for k in selected])
    lines = ['# Targeted before/after evaluation', '', '| Metric | Base | Adapted | Delta (pp) |', '|---|---:|---:|---:|']
    for key in b:
        lines.append(f'| {key} | {100*b[key]:.2f}% | {100*d[key]:.2f}% | {100*(d[key]-b[key]):+.2f} |')
    lines += ['', 'Observed differences, not statistical significance. All frozen target requests are required.',
              'Fast Multi errors abort this report rather than silently dropping missing labels.']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n'.join(lines) + '\n')
    a.output.with_suffix('.json').write_text(json.dumps({'base_sha256': bh, 'adapted_sha256': ah, 'base': b, 'adapted': d}, indent=2)+'\n')


if __name__ == '__main__':
    main()
