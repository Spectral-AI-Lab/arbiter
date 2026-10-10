"""Compare paired predictions, refusing different data or request definitions."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(root):
    reports = {}
    metadata = {}
    records = {}
    for directory in sorted(root.iterdir()):
        if not (directory/'summary.json').exists():
            continue
        meta = json.loads((directory/'metadata.json').read_text())
        if meta['status'] != 'completed':
            raise ValueError(f'Incomplete run: {directory.name}')
        metadata[directory.name] = meta
        reports[directory.name] = json.loads((directory/'summary.json').read_text())
        records[directory.name] = [json.loads(s) for s in (directory/'predictions.jsonl').read_text().splitlines()]
    if 'embeddinggemma2' not in reports or len(reports) < 2:
        raise ValueError('Need completed EmbeddingGemma 2 and Laya runs')
    ref = metadata['embeddinggemma2']
    for meta in metadata.values():
        for key in ['selection_sha256','dataset_sha256','threads','device','batch_size']:
            if meta[key] != ref[key]:
                raise ValueError(f'Incompatible {key}')
    rows = []
    paired = {}
    rng = np.random.default_rng(13)
    reference = records['embeddinggemma2']
    for model, rs in records.items():
        if model == 'embeddinggemma2':
            continue
        if len(rs) != len(reference):
            raise ValueError('Different record counts')
        deltas = {}
        for a,b in zip(reference,rs):
            for key in ['id','request_sha256','gold_sha256','expected_answers']:
                if a[key] != b[key]:
                    raise ValueError(f'Unpaired {key}')
            # Request-cluster bootstrap: retain all questions belonging to a state.
            count = a['expected_answers']
            correct_a = sum(o['correct'] for o in a['observations'].values())
            correct_b = sum(o['correct'] for o in b['observations'].values())
            deltas.setdefault(a['suite'], []).append((correct_a-correct_b, count))
        paired[model] = {}
        for suite, data in deltas.items():
            d = np.asarray(data, float)
            estimates=[]
            for _ in range(2000):
                draw=d[rng.integers(0,len(d),len(d))]
                estimates.append(draw[:,0].sum()/draw[:,1].sum())
            paired[model][suite] = {'eg2_minus_laya_accuracy': float(d[:,0].sum()/d[:,1].sum()),
                                    'paired_bootstrap_95ci': np.quantile(estimates,[.025,.975]).tolist()}
    title = 'PILOT — subsample' if ref['pilot'] else 'Official Laya Benchmark Samplings'
    rows = [f'# {title}', '', f"{ref['device']} ; batch size 1 ; effective precision recorded per model. Failures counted as incorrect.", '',
            '| Suite | Model | N Answers | Accuracy (failures included) | Coverage | ECE¹ | p50 ms/request |',
            '|---|---|---:|---:|---:|---:|---:|']
    for model, suites in reports.items():
        for suite, m in suites.items():
            ece = f"{m['ece_15']:.3f}" if 'ece_15' in m else '—'
            latency = f"{m['latency_p50_ms']:.1f}" if m['latency_p50_ms'] is not None else '—'
            rows.append(f"| {suite} | {model} | {m['expected_answers']} | {m['accuracy_including_failures']:.3f} | {m['coverage']:.3f} | {ece} | {latency} |")
    rows += ['', '¹ ECE on valid responses and max(probabilities), not the confidence field. Compare with coverage.',
             'Differences and paired bootstrap 95% CIs (per request) are in comparison.json. They are exploratory, without multiple comparisons correction.',
             'Laya typed-decisions was trained on the typed-decisions train split; EmbeddingGemma 2 was not. The two conditions are not equivalent.',
             'End-to-end pipeline latencies on recorded hardware: do not conflate with historical Laya reports.']
    (root/'comparison.md').write_text('\n'.join(rows)+'\n')
    (root/'comparison.json').write_text(json.dumps({'models': metadata, 'paired': paired}, indent=2)+'\n')
    print(root/'comparison.md')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    compare(p.parse_args().results)
