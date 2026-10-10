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
    title = 'PILOT — sous-échantillon' if ref['pilot'] else 'Échantillonnages officiels Laya'
    rows = [f'# {title}', '', f"{ref['device']} ; lot de 1 ; précision effective enregistrée par modèle. Échecs comptés comme incorrects.", '',
            '| Suite | Modèle | N réponses | Accuracy (échecs inclus) | Couverture | ECE¹ | p50 ms/requête |',
            '|---|---|---:|---:|---:|---:|---:|']
    for model, suites in reports.items():
        for suite, m in suites.items():
            ece = f"{m['ece_15']:.3f}" if 'ece_15' in m else '—'
            latency = f"{m['latency_p50_ms']:.1f}" if m['latency_p50_ms'] is not None else '—'
            rows.append(f"| {suite} | {model} | {m['expected_answers']} | {m['accuracy_including_failures']:.3f} | {m['coverage']:.3f} | {ece} | {latency} |")
    rows += ['', '¹ ECE sur réponses valides et max(probabilités), pas le champ confidence. Comparer avec la couverture.',
             'Les différences et IC bootstrap appariés (par requête) sont dans comparison.json. Ils sont exploratoires, sans correction pour comparaisons multiples.',
             'Laya typed-decisions a été entraîné sur le train de typed-decisions ; EmbeddingGemma 2 ne l’a pas été par ce projet. Les deux conditions ne sont pas équivalentes.',
             'Latences des pipelines complets sur le matériel enregistré : ne pas les assimiler aux mesures historiques de Laya.']
    (root/'comparison.md').write_text('\n'.join(rows)+'\n')
    (root/'comparison.json').write_text(json.dumps({'models': metadata, 'paired': paired}, indent=2)+'\n')
    print(root/'comparison.md')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    compare(p.parse_args().results)
