"""Verified preset preparation. Run from the repository root."""
import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetuning.build_80k import SOURCES, TRAIN_QUOTAS, VAL_QUOTAS

REVISION = SOURCES['typed'][1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(output, preset, seed):
    if preset == 'typed':
        manifest = json.loads((output / 'source_manifest.json').read_text())
        if any(manifest.get(k) != v for k, v in
               {'source': 'typed', 'revision': REVISION, 'seed': seed}.items()):
            raise ValueError('Typed source revision/seed mismatch; use a new data directory')
        # Reconstruct the pinned source conversion to verify existing files, not just existence.
        from datasets import load_dataset
        from finetuning.prepare_sources import typed_rows, split_group
        expected = {'train': hashlib.sha256(), 'validation': hashlib.sha256()}
        for row in load_dataset(SOURCES['typed'][0], 'all', split='train', revision=REVISION):
            for converted in typed_rows(row, REVISION):
                split = split_group(converted['group_id'], seed)
                expected[split].update((json.dumps(converted, ensure_ascii=False) + '\n').encode())
        for split, sha in expected.items():
            if digest(output / f'{split}.jsonl') != sha.hexdigest():
                raise ValueError(f'Typed {split} differs from pinned conversion')
    else:
        manifest = json.loads((output / 'manifest.json').read_text())
        if manifest.get('recipe') != 'typed-boost-80k-v1':
            raise ValueError('Recipe version mismatch')
        if manifest.get('counts') != {'train': TRAIN_QUOTAS, 'validation': VAL_QUOTAS}:
            raise ValueError('Recipe quotas mismatch; use a new data directory')
        identity = manifest.get('identity', {})
        if identity.get('seed') != seed:
            raise ValueError('Recipe seed mismatch')
        recorded = identity.get('builder_sha256', {})
        for name in ('build_80k.py', 'prepare_sources.py', 'data.py'):
            if recorded.get(name) != digest(ROOT / 'finetuning' / name):
                raise ValueError(f'Conversion source changed: {name}; build in a new directory')
        hashes = manifest.get('output_sha256', {})
        for split, quotas in [('train', TRAIN_QUOTAS), ('validation', VAL_QUOTAS)]:
            name = f'{split}.jsonl'
            if name not in hashes or digest(output / name) != hashes[name]:
                raise ValueError(f'Missing or invalid {name} hash')
            counts = Counter()
            for line in (output / name).open():
                row = json.loads(line)
                family = row['family'].split('/')[0]
                key = f'{family}/{row["question_type"]}'
                if family == 'nli':
                    key = 'nli/' + max(row['target'], key=row['target'].get)
                counts[key] += 1
            if counts != quotas:
                raise ValueError(f'Actual {split} quotas mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', choices=['typed', 'mixture_80k'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--heldout', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--max-length', type=int, default=1024)
    a = p.parse_args()
    if a.output.exists():
        verify(a.output, a.preset, a.seed)
    if a.preset == 'mixture_80k':
        command = [sys.executable, '-m', 'finetuning.build_80k', '--model', a.model,
                   '--heldout', *a.heldout, '--output', str(a.output), '--seed', str(a.seed),
                   '--max-length', str(a.max_length)]
        subprocess.run(command, check=True, cwd=ROOT)
    elif not a.output.exists():
        subprocess.run([sys.executable, '-m', 'finetuning.prepare_sources', '--source', 'typed',
                        '--revision', REVISION, '--output', str(a.output), '--seed', str(a.seed)],
                       check=True, cwd=ROOT)
    verify(a.output, a.preset, a.seed)
    print('Preset data verified. Run the trainer preflight before training.')


if __name__ == '__main__':
    main()
