"""Execute only the dataset-building code from the frozen official Laya scripts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
COMMIT = '3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63'


def notebook_cells():
    tree = ast.parse((HERE / 'upstream/build_benchmark_nb.py').read_text())
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'CELLS' for t in n.targets))
    return [ast.literal_eval(n.args[0]) for n in assignment.value.elts
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'code']


def build(groups, output, datasets_lock=None):
    provenance = json.loads((HERE/'upstream/SOURCES.json').read_text())
    for name, expected in provenance['files'].items():
        if hashlib.sha256((HERE/'upstream'/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Official source modified: {name}')
    import datasets
    from huggingface_hub import HfApi
    import numpy as np
    revisions = json.loads(datasets_lock.read_text()) if datasets_lock else {}
    failures = []
    loads = []
    original_load = datasets.load_dataset
    from banking77 import load_banking77

    def pinned_load(repo, *args, **kwargs):
        try:
            if repo not in revisions:
                revisions[repo] = HfApi().dataset_info(repo).sha
            kwargs['revision'] = revisions[repo]
            extra = {}
            if repo == 'PolyAI/banking77':
                data, extra = load_banking77(revisions[repo], output/'_sources', kwargs.get('split', 'test'))
            else:
                data = original_load(repo, *args, **kwargs)
            loads.append({'repo': repo, 'config': args[0] if args else None,
                          'split': kwargs.get('split'), 'revision': revisions[repo],
                          'fingerprint': getattr(data, '_fingerprint', None), **extra})
            return data
        except Exception as exc:
            failures.append({'repo': repo, 'error': str(exc)})
            raise

    datasets.load_dataset = pinned_load
    namespace = {'__name__': 'official_dataset_builder', 'np': np}
    try:
        cells = notebook_cells()
        setup = next(c for c in cells if 'def build_choice_questions(' in c)
        exec(compile(setup, 'laya_dataset_setup', 'exec'), namespace)
        signatures = {'typed': 'td_cases, td_golds, td_wf =',
                      'multilingual': 'MASSIVE_LANGS =', 'english': '# SST-5: ordinal score'}
        for group, signature in signatures.items():
            if group in groups:
                source = next(c for c in cells if signature in c)
                exec(compile(source, f'laya_{group}_builder', 'exec'), namespace)
        suites = namespace['SUITES']
        expected = sum({'typed': 1, 'multilingual': 43, 'english': 6}.get(g, 0) for g in groups)
        if len(suites) != expected:
            raise RuntimeError(f'Expected {expected} official notebook suites, got {len(suites)}; dataset failures: {failures}')
        if 'apps' in groups:
            import laya
            source = (HERE / 'upstream/bench_apps.py').read_text()
            tree = ast.parse(source)
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in {'register', 'choice_q', 'build'}]
            app = {'random': random, 'laya': laya, 'SEED': 13, 'N': 400, 'SUITES': {}}
            exec(compile(ast.Module(body=functions, type_ignores=[]), 'laya_apps_builder', 'exec'), app)
            app['build']()
            if len(app['SUITES']) != 10:
                raise RuntimeError(f"Expected 10 application suites, got {len(app['SUITES'])}")
            for name, suite in app['SUITES'].items():
                suite['gold'] = [{next(iter(case[1])): {'idx': int(gold)}}
                                 for case, gold in zip(suite['cases'], suite['gold'])]
                suites[name] = suite
        if failures:
            raise RuntimeError(f'Dataset loads failed: {failures}')
        rows = []
        for name, suite in suites.items():
            if not suite['cases'] or len(suite['cases']) != len(suite['gold']):
                raise ValueError(f'Empty or inconsistent suite: {name}')
            for i, ((state, questions), gold) in enumerate(zip(suite['cases'], suite['gold'])):
                for q in questions.values():
                    if q['type'] == 'choice' and isinstance(q['criteria'], list):
                        raise ValueError('Unexpected choice-list schema; do not silently rewrite official requests')
                rows.append({'id': f'{name}/{i}', 'suite': name, 'state': state,
                             'questions': questions, 'gold': gold})
        content = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'cases.jsonl').write_text(content, encoding='utf-8')
        manifest = {'upstream_commit': COMMIT, 'groups': groups, 'seed': 13,
                    'dataset_revisions': revisions, 'loads': loads,
                    'cases_sha256': hashlib.sha256(content.encode()).hexdigest(),
                    'cases': len(rows), 'questions': sum(len(r['questions']) for r in rows),
                    'suites': {k: {'cases': len(s['cases']), 'meta': s['meta']} for k, s in suites.items()},
                    'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in (HERE / 'upstream').glob('*.py')}}
        (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        (output / 'dataset_revisions.json').write_text(json.dumps(revisions, indent=2)+'\n')
        print(f"Prepared {len(rows)} cases in {output}")
    finally:
        datasets.load_dataset = original_load


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', nargs='+', choices=['typed','multilingual','english','apps'],
                   default=['typed','multilingual','english','apps'])
    p.add_argument('--output', type=Path, default=HERE / 'data')
    p.add_argument('--datasets-lock', type=Path)
    a = p.parse_args()
    build(list(dict.fromkeys(a.groups)), a.output, a.datasets_lock)
