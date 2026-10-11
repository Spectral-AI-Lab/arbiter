"""LOCAL EXPERIMENTAL UTILITY: requires historical datasets outside this repository.
Provenance mappings below are legacy annotations, not verified upstream licenses.
Build the verified 80% Typed / 20% Replay adaptation dataset with full retention support.

Consensus recipe agreed between GPT and Gemini reviews:
- 5,410 official Typed training decisions repeated 5 passes (27,050 presentations).
- 6,760 replay decisions diversified across output types:
    * 4,900 choice decisions (MASSIVE, Banking77, NLI, TREC, SST-2)
    * 1,000 boolean (noul) decisions (RAG relevance and Application verification)
    * 860 ordinal score decisions (Generalist calibration from HelpSteer/CONDAQA)
- Disjoint, statistically resolved retention validation suite (2,500 rows):
    * 500 MASSIVE
    * 500 Banking77
    * 500 NLI
    * 250 TREC
    * 250 SST-2
    * 250 Boolean (noul)
    * 250 Score (ordinal)
    plus 590 official Typed validation rows = 3,090 total validation rows.
- Strict anti-leak audit against heldout benchmarks.
- Atomic directory writing with cryptographic SHA-256 manifest and source provenance.
- Exact match between train and validation family sets.
"""
import tempfile
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from finetuning.data import audit, read_rows


SOURCE_PROVENANCE = {
    'massive': {'source': 'mteb/amazon_massive_intent', 'license': 'apache-2.0', 'source_split': 'train'},
    'banking': {'source': 'PolyAI/banking77', 'license': 'mit', 'source_split': 'train'},
    'nli': {'source': 'facebook/xnli', 'license': 'cc-by-nc-4.0', 'source_split': 'train'},
    'trec': {'source': 'trec', 'license': 'apache-2.0', 'source_split': 'train'},
    'sst2': {'source': 'sst2', 'license': 'apache-2.0', 'source_split': 'train'},
    'boolean': {'source': 'arbiter-internal-triage', 'license': 'apache-2.0', 'source_split': 'train'},
    'score': {'source': 'generalist-scoring', 'license': 'various-historical-open', 'source_split': 'train'},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def convert_replay_row(d, idx, split='train', domain='general', family_prefix='retention'):
    opts = d['options']
    targets = d['targets']
    criteria = {str(k): str(text) for k, text in opts}
    total = sum(targets)
    if total <= 0 or abs(total - 1.0) > 1e-4:
        target = {str(k): float(p) / total for (k, _), p in zip(opts, targets)}
    else:
        target = {str(k): float(p) for (k, _), p in zip(opts, targets)}
    
    case_id = d.get('metadata', {}).get('case_id', hashlib.sha256(d['context'].encode()).hexdigest()[:12])
    ctx_hash = hashlib.sha256(f"{d['context']}:{d['instruction']}".encode()).hexdigest()[:8]

    qtype = d['question_type']
    if qtype == 'noul':
        norm_criteria = {'false': criteria.get('false', 'False'), 'true': criteria.get('true', 'True')}
        norm_target = {'false': target.get('false', 0.0), 'true': target.get('true', 1.0)}
        criteria, target = norm_criteria, norm_target

    prov = SOURCE_PROVENANCE.get(domain, {})
    source_name = prov.get('source', d.get('metadata', {}).get('source', 'arbiter-historical-socle'))
    license_name = prov.get('license', d.get('metadata', {}).get('license_status', 'apache-2.0'))

    return {
        'id': f"{family_prefix}:{split}:{domain}:{idx}_{ctx_hash}",
        'group_id': f"{family_prefix}:{case_id}",
        'family': f"{family_prefix}/{domain}",
        'source': source_name,
        'source_revision': 'historical_p1',
        'source_split': split,
        'license': license_name,
        'context': d['context'],
        'instruction': d['instruction'],
        'question_type': qtype,
        'criteria': criteria,
        'target': target,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--typed-train', type=Path, default=Path('results/adaptation/typed/train.jsonl'))
    p.add_argument('--typed-val', type=Path, default=Path('results/adaptation/typed/validation.jsonl'))
    p.add_argument('--replay-train-source', type=Path, required=True)
    p.add_argument('--generalist-train', type=Path, required=True)
    p.add_argument('--replay-val-dir', type=Path, required=True)
    p.add_argument('--generalist-val-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('results/adaptation/typed_replay_33k'))
    p.add_argument('--passes', type=int, default=5, help='Number of passes over official Typed')
    p.add_argument('--seed', type=int, default=13)
    a = p.parse_args()

    rng = random.Random(a.seed)
    if a.output_dir.exists():
        p.error('Output already exists; choose a new directory. Existing datasets are never replaced.')
    if a.passes != 5:
        p.error('This fixed 80/20 recipe requires --passes 5.')
    a.output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = Path(tempfile.mkdtemp(prefix=a.output_dir.name + '.', dir=a.output_dir.parent))

    print("=== Step 1: Loading official Typed dataset ===")
    typed_raw = [json.loads(line) for line in a.typed_train.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(typed_raw)} official Typed training rows.")

    typed_train_pool = []
    for pass_idx in range(a.passes):
        for r in typed_raw:
            entry = dict(r)
            if pass_idx > 0:
                entry['id'] = f"{r['id']}:p{pass_idx}"
            typed_train_pool.append(entry)
    print(f"Expanded Typed to {len(typed_train_pool)} presentations ({a.passes} passes).")

    print("\n=== Step 2: Sampling diversified Replay training pool ===")
    # Load replay candidates from g2m_booster_65k/train.jsonl
    g2m_buckets = {
        ('choice', 'massive'): [],
        ('choice', 'banking'): [],
        ('choice', 'nli'): [],
        ('choice', 'question_type'): [],
        ('choice', 'sentiment'): [],
        ('noul', 'rag_relevance'): [],
        ('noul', 'app'): [],
    }

    with open(a.replay_train_source) as f:
        for line in f:
            d = json.loads(line)
            qt = d.get('question_type')
            dom = d.get('metadata', {}).get('domain')
            key = (qt, dom)
            if key in g2m_buckets:
                g2m_buckets[key].append(d)

    sampled_replay_train = []
    
    # 1. Choice replay
    choice_mapping = [
        ('massive', ('choice', 'massive'), 1400),
        ('banking', ('choice', 'banking'), 1000),
        ('nli', ('choice', 'nli'), 1200),
        ('trec', ('choice', 'question_type'), 650),
        ('sst2', ('choice', 'sentiment'), 650),
    ]
    for dom_label, bucket_key, q in choice_mapping:
        pool = g2m_buckets[bucket_key]
        if len(pool) < q:
            raise ValueError(f"Insufficient pool for {bucket_key}: {len(pool)} < {q}")
        rng.shuffle(pool)
        for idx, d in enumerate(pool[:q]):
            sampled_replay_train.append(convert_replay_row(d, idx, split='train', domain=dom_label, family_prefix='retention'))
        print(f"  Choice replay [{dom_label}]: {q} sampled.")

    # 2. Boolean (noul) replay -> mapped to family 'retention/boolean'
    boolean_candidates = []
    rag_pool = list(g2m_buckets[('noul', 'rag_relevance')])
    app_pool = list(g2m_buckets[('noul', 'app')])
    rng.shuffle(rag_pool)
    rng.shuffle(app_pool)
    boolean_candidates.extend(rag_pool[:500])
    boolean_candidates.extend(app_pool[:500])
    assert len(boolean_candidates) == 1000
    for idx, d in enumerate(boolean_candidates):
        sampled_replay_train.append(convert_replay_row(d, idx, split='train', domain='boolean', family_prefix='retention'))
    print(f"  Boolean (noul) replay [boolean]: 1000 sampled (500 rag + 500 app).")

    # 3. Ordinal (score) replay -> mapped to family 'retention/score'
    score_q = 860
    score_pool = []
    with open(a.generalist_train) as f:
        for line in f:
            d = json.loads(line)
            if d.get('question_type') == 'score':
                src = d.get('metadata', {}).get('source', '')
                tok = d.get('metadata', {}).get('tokens', 0)
                if 'procedural' not in src and 0 < tok <= 750:
                    score_pool.append(d)
                    if len(score_pool) >= score_q * 3:
                        break
    if len(score_pool) < score_q:
        raise ValueError(f"Insufficient pool for score: {len(score_pool)} < {score_q}")
    rng.shuffle(score_pool)
    for idx, d in enumerate(score_pool[:score_q]):
        sampled_replay_train.append(convert_replay_row(d, idx, split='train', domain='score', family_prefix='retention'))
    print(f"  Ordinal (score) replay [score]: {score_q} sampled (filtered <= 750 tokens).")

    assert len(sampled_replay_train) == 6760, f"Expected 6,760 replay rows, got {len(sampled_replay_train)}"
    print(f"Total replay train rows: {len(sampled_replay_train)}")

    # Combine and shuffle
    all_train = typed_train_pool + sampled_replay_train
    rng.shuffle(all_train)
    assert len(all_train) == 33810, f"Expected 33,810 total train rows, got {len(all_train)}"
    print(f"Combined train rows: {len(all_train)} ({100*len(typed_train_pool)/len(all_train):.2f}% Typed, {100*len(sampled_replay_train)/len(all_train):.2f}% Replay)")

    print("\n=== Step 3: Building Disjoint Retention Validation Suite ===")
    typed_val_raw = [json.loads(line) for line in a.typed_val.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(typed_val_raw)} official Typed validation rows.")

    val_quotas = {
        'massive': (a.replay_val_dir / 'val_massive.jsonl', 500),
        'banking': (a.replay_val_dir / 'val_banking.jsonl', 500),
        'nli': (a.replay_val_dir / 'val_nli.jsonl', 500),
        'trec': (a.replay_val_dir / 'val_trec.jsonl', 250),
        'sst2': (a.replay_val_dir / 'val_sst2.jsonl', 250),
    }

    sampled_retention_val = []
    for dom, (path, q) in val_quotas.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing validation file: {path}")
        rows = [json.loads(l) for l in path.open() if l.strip()]
        if len(rows) < q:
            raise ValueError(f"Fail-closed: {path} has only {len(rows)} rows, required {q}")
        rng.shuffle(rows)
        for idx, d in enumerate(rows[:q]):
            sampled_retention_val.append(convert_replay_row(d, idx, split='val', domain=dom, family_prefix='retention'))
        print(f"  Retention val [{dom}]: {q} rows sampled.")

    # Boolean (noul) validation (250 rows)
    noul_val_path = a.generalist_val_dir / 'val_generalist_classification.jsonl'
    noul_val_rows = []
    with open(noul_val_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('question_type') == 'noul' and 'procedural' not in d.get('metadata', {}).get('source', '') and 0 < d.get('metadata', {}).get('tokens', 0) <= 750:
                noul_val_rows.append(d)
    if len(noul_val_rows) < 250:
        raise ValueError(f"Fail-closed: insufficient noul validation rows: {len(noul_val_rows)} < 250")
    rng.shuffle(noul_val_rows)
    for idx, d in enumerate(noul_val_rows[:250]):
        sampled_retention_val.append(convert_replay_row(d, idx, split='val', domain='boolean', family_prefix='retention'))
    print(f"  Retention val [boolean]: 250 rows sampled.")

    # Score (ordinal) validation (250 rows)
    score_val_path = a.generalist_val_dir / 'val_generalist_score.jsonl'
    score_val_rows = [
        json.loads(l) for l in score_val_path.open()
        if l.strip() and 0 < json.loads(l).get('metadata', {}).get('tokens', 0) <= 750
    ]
    if len(score_val_rows) < 250:
        raise ValueError(f"Fail-closed: insufficient score validation rows: {len(score_val_rows)} < 250")
    rng.shuffle(score_val_rows)
    for idx, d in enumerate(score_val_rows[:250]):
        sampled_retention_val.append(convert_replay_row(d, idx, split='val', domain='score', family_prefix='retention'))
    print(f"  Retention val [score]: 250 rows sampled (filtered <= 750 tokens).")

    assert len(sampled_retention_val) == 2500, f"Expected 2,500 retention val rows, got {len(sampled_retention_val)}"
    all_val = typed_val_raw + sampled_retention_val
    rng.shuffle(all_val)
    assert len(all_val) == 3090, f"Expected 3,090 total val rows, got {len(all_val)}"
    print(f"Total validation rows: {len(all_val)} (590 official Typed + 2,500 Retention).")

    # Verify family equality
    train_fams = set(r['family'] for r in all_train)
    val_fams = set(r['family'] for r in all_val)
    if train_fams != val_fams:
        raise ValueError(f"Family mismatch:\nTrain only: {train_fams - val_fams}\nVal only: {val_fams - train_fams}")
    print(f"Family alignment VERIFIED: {len(train_fams)} families perfectly identical in train and validation.")

    print("\n=== Step 4: Writing files to temporary location ===")
    train_out = tmp_out / 'train.jsonl'
    val_out = tmp_out / 'validation.jsonl'

    with open(train_out, 'w', encoding='utf-8') as f:
        for r in all_train:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(val_out, 'w', encoding='utf-8') as f:
        for r in all_val:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print("\n=== Step 5: Strict Anti-Leak Audit Against Heldout Benchmarks ===")
    heldout = [
        Path('benchmark/data/cases.jsonl'),
        Path('benchmark/data/public_requested_v1/cases.jsonl')
    ]
    train_rows = read_rows(train_out)
    val_rows = read_rows(val_out)
    audit(train_rows, val_rows, heldout)
    print("Anti-leak audit PASSED with 0 heldout overlap!")

    print("\n=== Step 6: Generating Cryptographic Manifest ===")
    manifest = {
        'recipe': 'typed_replay_80_20_retention',
        'script_sha256': sha256_file(Path(__file__)),
        'inputs': {
            'typed_train': {'path': str(a.typed_train), 'sha256': sha256_file(a.typed_train)},
            'typed_val': {'path': str(a.typed_val), 'sha256': sha256_file(a.typed_val)},
            'replay_train_source': {'path': str(a.replay_train_source), 'sha256': sha256_file(a.replay_train_source)},
            'generalist_train': {'path': str(a.generalist_train), 'sha256': sha256_file(a.generalist_train)},
        },
        'train_counts': {
            'typed_raw': len(typed_raw),
            'passes': a.passes,
            'typed_presentations': len(typed_train_pool),
            'replay_choice': 4900,
            'replay_noul': 1000,
            'replay_score': 860,
            'total_replay': len(sampled_replay_train),
            'total_train': len(all_train),
            'typed_share_pct': 100 * len(typed_train_pool) / len(all_train),
            'replay_share_pct': 100 * len(sampled_replay_train) / len(all_train),
        },
        'validation_counts': {
            'typed_val': len(typed_val_raw),
            'retention_massive': 500,
            'retention_banking': 500,
            'retention_nli': 500,
            'retention_trec': 250,
            'retention_sst2': 250,
            'retention_boolean': 250,
            'retention_score': 250,
            'total_retention': len(sampled_retention_val),
            'total_val': len(all_val),
        },
        'outputs': {
            'train_sha256': sha256_file(train_out),
            'validation_sha256': sha256_file(val_out),
        },
        'seed': a.seed,
        'heldout_decontaminated': True,
        'status': 'local_experimental',
        'limitations': ['Legacy provenance mappings require upstream verification before redistribution.'],
    }

    manifest_path = tmp_out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    print("\n=== Step 7: Atomic Directory Replacement ===")
    if a.output_dir.exists():
        raise FileExistsError(f"Output appeared during build: {a.output_dir}")
    tmp_out.replace(a.output_dir)
    print(f"Dataset successfully and atomically published to {a.output_dir}!")


if __name__ == '__main__':
    main()
