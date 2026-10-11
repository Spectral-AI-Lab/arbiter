"""Convert pinned TRAIN sources; group-held-out validation before flattening.

Produces one source at a time. Mixing/replay and semantic decontamination are
explicit dataset-curation steps, not silently performed here.
"""
import argparse
import hashlib
import json
from pathlib import Path
from datasets import load_dataset

LANGUAGES = 'ar bg de el en es fr hi ru sw th tr ur vi zh'.split()


def split_group(group, seed):
    value = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:8], 16)
    return 'validation' if value % 10 == 0 else 'train'


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def typed_rows(row, revision):
    questions, gold = decode(row['questions']), decode(row['gold'])
    # Match the inference API's structured-state serialization.
    try:
        context = json.dumps(decode(row['state']), ensure_ascii=False)
    except (ValueError, TypeError):
        context = row['state']
    for qid, question in questions.items():
        kind = question['type']
        criteria = question.get('criteria')
        if kind == 'noul':
            criteria = {'false': 'The statement is false.', 'true': 'The statement is true.'}
        elif isinstance(criteria, list):
            criteria = {str(i): str(v) for i,v in enumerate(criteria)}
        probabilities = gold[qid].get('probabilities')
        if probabilities is None:
            raise ValueError('Typed source must include native soft targets')
        target = {key: float(probabilities[key]) for key in criteria}
        total = sum(target.values())
        if total <= 0 or abs(total-1) > 1e-4:
            raise ValueError('Invalid source distribution')
        target = {key: value/total for key,value in target.items()}
        yield dict(id=f'typed:{row["id"]}:{qid}', group_id=f'typed:{row["id"]}',
                   family=f'typed/{row["workflow"]}/{kind}', context=context,
                   instruction=question['instructions'], question_type=kind, criteria=criteria, target=target,
                   source='LocalLLaMA/typed-decisions', source_revision=revision, source_split='train', license='apache-2.0')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, choices=['typed', 'xnli'])
    p.add_argument('--revision', required=True, help='Full upstream commit SHA')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--license', help='Verified source license identifier (required for XNLI)')
    p.add_argument('--groups', type=int, default=2000, help='XNLI underlying premise/hypothesis groups, not translated decisions')
    a = p.parse_args()
    if len(a.revision) != 40 or any(c not in '0123456789abcdef' for c in a.revision) or a.groups <= 0:
        p.error('Require full revision SHA and positive group budget')
    if a.source == 'xnli' and not a.license:
        p.error('Supply --license after checking terms for the pinned XNLI/MultiNLI source')
    if a.output.exists():
        p.error('Output already exists')
    a.output.mkdir(parents=True)
    files = {s: (a.output/f'{s}.jsonl').open('w') for s in ('train','validation')}
    counts = {s: 0 for s in files}
    def write(r):
        split = split_group(r['group_id'], a.seed)
        files[split].write(json.dumps(r, ensure_ascii=False)+'\n')
        counts[split] += 1
    try:
        if a.source == 'typed':
            ds = load_dataset('LocalLLaMA/typed-decisions', 'all', split='train', revision=a.revision)
            for row in ds:
                for r in typed_rows(row, a.revision):
                    write(r)
        else:
            # One multilingual row is one semantic group. All translations stay together.
            ds = load_dataset('facebook/xnli', 'all_languages', split='train', revision=a.revision)
            ds = ds.shuffle(seed=a.seed).select(range(min(a.groups, len(ds))))
            names = ds.features['label'].names
            criteria = {'entailment': 'The hypothesis follows from the premise.',
                        'neutral': 'The premise does not determine whether the hypothesis is true.',
                        'contradiction': 'The premise contradicts the hypothesis.'}
            for row in ds:
                hypotheses = dict(zip(row['hypothesis']['language'], row['hypothesis']['translation']))
                group = hashlib.sha256(json.dumps([row['premise']['en'], hypotheses['en']], ensure_ascii=False).encode()).hexdigest()
                label = names[row['label']]
                for lang in LANGUAGES:
                    write(dict(id=f'xnli:{group}:{lang}', group_id=f'xnli:{group}', family=f'nli/{lang}',
                               source='facebook/xnli', source_revision=a.revision, source_split='train', license=a.license,
                               context=row['premise'][lang], instruction='Classify the relationship to this hypothesis: '+hypotheses[lang],
                               criteria=criteria, question_type='choice', target={k: float(k==label) for k in criteria}))
    finally:
        for f in files.values():
            f.close()
    (a.output/'source_manifest.json').write_text(json.dumps(dict(source=a.source, revision=a.revision,
        seed=a.seed, counts=counts, status='Converted only; requires license review and heldout audit before training'), indent=2)+'\n')
    print(counts)


if __name__ == '__main__':
    main()
