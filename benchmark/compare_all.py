"""Create one domain/latency table from completed benchmark runs."""
import argparse
import json
from pathlib import Path


def discover(paths):
    runs = {}
    for root in paths:
        candidates = [root] if (root / 'summary.json').exists() else list(root.iterdir())
        for d in candidates:
            summary = d / 'summary.json'
            meta = d / 'metadata.json'
            if not summary.exists() or not meta.exists():
                continue
            m = json.loads(meta.read_text())
            if m.get('status') != 'completed':
                continue
            name = m.get('model', d.name)
            if name in runs:
                raise ValueError(f'Duplicate model name {name}; use targeted_report for base/adapted variants')
            runs[name] = (m, json.loads(summary.read_text()))
    if not runs:
        raise SystemExit('No completed runs found')
    return runs


def weighted(rows, key):
    rows = [r for r in rows if r.get(key) is not None and r.get('expected_answers', 0)]
    n = sum(r['expected_answers'] for r in rows)
    return sum(r[key] * r['expected_answers'] for r in rows) / n if n else None


def main(paths, output):
    runs = discover(paths)
    models = sorted(runs)
    labels = {'embeddinggemma2': 'EmbeddingGemma2', 'laya': 'Laya',
              'laya-multilingual': 'Laya Multilingual',
              'laya-typed-decisions': 'Laya Typed Decisions',
              'gliner25-decide': 'GLiNER2.5-Decide',
              'gliner25-multi-decide': 'GLiNER2.5-multi-Decide'}
    display_models = [labels.get(m, m) for m in models]
    groups = {
        'Typed decisions': lambda s: s == 'typed_decisions',
        'MASSIVE intent': lambda s: s.startswith('massive_intent.'),
        'MASSIVE scenario': lambda s: s.startswith('massive_scenario.'),
        'XNLI': lambda s: s.startswith('xnli.'),
        'English classification': lambda s: s.startswith('en.'),
        'Applications': lambda s: s.startswith(('app.', 'jev.')),
    }
    lines = ['# Aggregated Domain Comparison', '',
             'Values are weighted averages based on expected response counts. Latency columns average suite percentiles; they are not pooled percentiles.', '']
    metric_specs = [
        ('Accuracy (%)', 'accuracy_including_failures', 100, '.1f'),
        ('Latency p50 (ms)', 'latency_p50_ms', 1, '.1f'),
        ('Latency p95 (ms)', 'latency_p95_ms', 1, '.1f'),
        ('ECE', 'ece_15', 1, '.3f'),
        ('Brier', 'brier', 1, '.3f'),
        ('NLL', 'nll', 1, '.3f'),
    ]
    aggregate = {}
    for title, key, scale, fmt in metric_specs:
        lines += [f'## {title}', '', '| Domain | ' + ' | '.join(display_models) + ' |',
                  '|' + '---|' * (len(models) + 1)]
        for group, predicate in groups.items():
            cells = []
            for model in models:
                rows = [r for s, r in runs[model][1].items() if predicate(s)]
                value = weighted(rows, key)
                cells.append('—' if value is None else format(value * scale, fmt))
                aggregate.setdefault(model, {}).setdefault(group, {})[key] = value
            lines.append('| ' + group + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Overall Summary', '',
              '| Model | Answers | Accuracy (%) | ECE | Latency p50 (ms) | Latency p95 (ms) |',
              '|---|---:|---:|---:|---:|---:|']
    for model, (meta, report) in sorted(runs.items()):
        rows = list(report.values())
        n = sum(r.get('expected_answers', 0) for r in rows)
        lines.append(f"| {model} | {n} | {100*weighted(rows, 'accuracy_including_failures'):.3f} | "
                     f"{weighted(rows, 'ece_15'):.3f} | {weighted(rows, 'latency_p50_ms'):.1f} | "
                     f"{weighted(rows, 'latency_p95_ms'):.1f} |")
        aggregate[model]['metadata'] = meta
        aggregate[model]['n'] = n
        aggregate[model]['global'] = {
                            'accuracy': weighted(rows, 'accuracy_including_failures'),
                            'ece': weighted(rows, 'ece_15'),
                            'latency_p50_ms': weighted(rows, 'latency_p50_ms'),
                            'latency_p95_ms': weighted(rows, 'latency_p95_ms')}
    output.write_text('\n'.join(lines) + '\n')
    output.with_suffix('.json').write_text(json.dumps({'models': aggregate}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('results', nargs='+', type=Path, help='Run directories or parent results directories')
    p.add_argument('--output', type=Path, default=Path('results/comparison_all.md'))
    a = p.parse_args()
    main(a.results, a.output)
