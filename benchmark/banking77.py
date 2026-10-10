"""Read the original Banking77 CSV without executing its obsolete dataset script."""
import ast
import csv
import hashlib
from pathlib import Path
import urllib.request

CSV_REVISION = '57ec275d8078af65b7731c2a98be812d844a6d6b'


def label_names(script):
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'ClassLabel':
            names = ast.literal_eval(next(k.value for k in node.keywords if k.arg == 'names'))
            if len(names) != 77 or len(set(names)) != 77:
                raise ValueError('Unexpected Banking77 label schema')
            return names
    raise ValueError('Missing original Banking77 ClassLabel schema')


def read_csv(path, names):
    from datasets import Dataset, Features, ClassLabel, Value
    with Path(path).open(encoding='utf-8', newline='') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(reader)
        rows = list(reader)
    indexes = {name: i for i, name in enumerate(names)}
    features = Features({'text': Value('string'), 'label': ClassLabel(names=names)})
    return Dataset.from_dict({'text': [r[0] for r in rows],
                              'label': [indexes[r[1]] for r in rows]}, features=features)


def load_banking77(revision, cache_dir, split='test'):
    from huggingface_hub import hf_hub_download
    if split not in {'test', 'train'}:
        raise ValueError(f'Unsupported Banking77 split: {split}')
    script = Path(hf_hub_download('PolyAI/banking77', 'banking77.py', repo_type='dataset', revision=revision))
    names = label_names(script.read_text())
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'banking77-{split}.csv'
    url = f'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{CSV_REVISION}/banking_data/{split}.csv'
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    if split == 'test' and hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() != 'e10f6bc95fe4e3eb5e8060f60f09aee8b17d03f23ce0047d9511e8003c26c808':
        raise ValueError('Banking77 CSV does not match the pinned original test split')
    data = read_csv(path, names)
    expected = 3080 if split == 'test' else 10003
    if len(data) != expected:
        raise ValueError(f'Banking77 {split}: expected {expected} rows, got {len(data)}')
    return data, {'loader': 'original_csv', 'csv_url': url, 'csv_revision': CSV_REVISION,
                  'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'label_schema_sha256': hashlib.sha256(script.read_bytes()).hexdigest()}
