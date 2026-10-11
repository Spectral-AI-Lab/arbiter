"""Portable decision JSONL contract and conservative cross-split audit."""
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def context_hash(text):
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    seen = set()
    for r in rows:
        for key in ('id', 'group_id', 'family', 'source', 'source_revision', 'source_split', 'license',
                    'context', 'instruction', 'question_type', 'criteria', 'target'):
            if key not in r:
                raise ValueError(f'Missing {key} in {r.get("id")}')
        if r['id'] in seen:
            raise ValueError(f'Duplicate decision id: {r["id"]}')
        seen.add(r['id'])
        for key in ('id', 'group_id', 'family', 'source', 'source_revision', 'source_split', 'license', 'context', 'instruction'):
            if not isinstance(r[key], str) or not r[key].strip():
                raise ValueError(f'Nonempty string required: {key}')
        if r['question_type'] not in ('choice', 'noul', 'score'):
            raise ValueError('Unsupported question_type')
        keys = list(r['criteria']) if isinstance(r['criteria'], dict) else []
        if len(keys) < 2 or any(not k.strip() or k != k.strip() or '<OPTION>' in k for k in keys):
            raise ValueError('criteria must map >=2 clean keys to descriptions')
        if r['question_type'] == 'noul' and keys != ['false', 'true']:
            raise ValueError('noul keys must be ordered false, true')
        if not isinstance(r['target'], dict) or set(r['target']) != set(keys):
            raise ValueError('target must specify a probability for every criterion')
        values = list(r['target'].values())
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Invalid target probability')
        if abs(sum(values) - 1) > 1e-4:
            raise ValueError('Target probabilities must sum to one')
    return rows


def audit(train, validation, heldout_paths):
    """Reject context/group overlaps. Benchmark case JSONL or canonical rows supported.

    This does not detect paraphrases or unknown exposure in prior training.
    Group IDs must be globally namespaced and shared by translations/variants.
    """
    if not heldout_paths:
        raise ValueError('Provide held-out benchmark case files')
    train_contexts = {context_hash(r['context']) for r in train}
    val_contexts = {context_hash(r['context']) for r in validation}
    train_groups = {r['group_id'] for r in train}
    val_groups = {r['group_id'] for r in validation}
    if train_contexts & val_contexts or train_groups & val_groups:
        raise ValueError('Train/validation overlap by context or group')
    hashes = {}
    for path in heldout_paths:
        raw = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ctx = r.get('context', r.get('state'))
            if ctx is None:
                raise ValueError(f'No state/context in heldout file {path}')
            if not isinstance(ctx, str):
                ctx = json.dumps(ctx, ensure_ascii=False)
            if context_hash(ctx) in train_contexts | val_contexts:
                raise ValueError(f'Heldout context overlap: {path}, {r.get("id")}')
            if r.get('group_id') in train_groups | val_groups:
                raise ValueError(f'Heldout group overlap: {path}, {r.get("id")}')
    return {'heldout_sha256': hashes, 'train_rows': len(train), 'validation_rows': len(validation),
            'train_families': dict(Counter(r['family'] for r in train)),
            'train_types': dict(Counter(r['question_type'] for r in train)),
            'scope': 'normalized exact contexts and supplied group IDs only; no semantic audit'}


def prepare(model, rows, max_length):
    output = []
    for r in rows:
        # The same formatting and tokenizer as inference; no silent truncation.
        p = model._prepare_decision(**{k: r[k] for k in ('context', 'instruction', 'criteria', 'question_type')},
                                    max_length=max_length, allow_truncation=False)
        output.append(dict(p, target=[r['target'][k] for k in p['keys']], family=r['family'], id=r['id']))
    return output
