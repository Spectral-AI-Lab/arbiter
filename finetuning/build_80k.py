"""Build the downloadable 80k adaptation recipe from pinned TRAIN sources.

No model-generated labels, test-set fitting, oversampling or private local datasets.
This recipe uses published procedural examples, not newly human-annotated policies.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from arbiter import ArbiterModel
from .data import audit, context_hash, read_rows
from .prepare_sources import LANGUAGES, split_group, typed_rows

SOURCES = {
    'typed': ('LocalLLaMA/typed-decisions', 'd0e2f0c42fef86cc15d1688d25a19f5ba7c85b18', 'apache-2.0'),
    'procedural': ('tasksource/procedural-typed-decisions', '609513a3faddf729123266efe9861456fe748f95', 'apache-2.0'),
    'nli': ('facebook/xnli', 'b8dd5d7af51114dbda02c0e3f6133f332186418e', 'unspecified-see-upstream-terms'),
    'massive': ('mteb/amazon_massive_intent', '940fd47a81eaa7f2cc7b129674d945d618ac38c2', 'apache-2.0'),
    'banking': ('mteb/banking77', '18072d2685ea682290f7b8924d94c62acc19c0b2', 'mit'),
}
TRAIN_QUOTAS = {'typed/choice':6400, 'typed/noul':4800, 'typed/score':4800,
                'policy/choice':12000, 'policy/noul':12000,
                'nli/entailment':8010, 'nli/neutral':7995, 'nli/contradiction':7995,
                'replay/banking':8000, 'replay/massive':8000}
VAL_QUOTAS = {'typed/choice':640, 'typed/noul':480, 'typed/score':480,
              'policy/choice':1200, 'policy/noul':1200,
              'nli/entailment':810, 'nli/neutral':795, 'nli/contradiction':795,
              'replay/banking':800, 'replay/massive':800}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def record(source, identity, group, family, context, instruction, kind, criteria, target, **extra):
    repo, rev, license_id = SOURCES[source]
    return dict(id=f'{source}:{identity}', group_id=f'{source}:{group}', family=family,
                source=repo, source_revision=rev, source_split='train', license=license_id,
                context=context, instruction=instruction, question_type=kind,
                criteria=criteria, target=target, **extra)


def procedural_rows(row, family):
    questions, answers = json.loads(row['questions']), json.loads(row['answers'])
    for qid, q in questions.items():
        kind, a = q['type'], answers[qid]
        if kind == 'noul':
            criteria = {'false':'The statement is false.', 'true':'The statement is true.'}
            prob = float(a['noul'])
            target = {'false':1-prob, 'true':prob}
        else:
            criteria = q['criteria']
            if isinstance(criteria,list):
                criteria = {str(i):str(v) for i,v in enumerate(criteria)}
            target = {k:float(a['probabilities'][k]) for k in criteria}
            total = sum(target.values())
            if total <= 0 or abs(total-1)>1e-4:
                raise ValueError(f'Invalid source target: {row["id"]}/{qid}')
            target = {k:v/total for k,v in target.items()}
        yield record('procedural', f'{row["id"]}:{qid}', row['id'],
                     f'{family}/{row["id"].split(":")[0]}/{kind}', row['state'],
                     q['instructions'], kind, criteria, target, synthetic=True)


class Selector:
    """Budgeted group assignment with conservative exact context exclusion."""
    def __init__(self, quotas, heldout, proxy, max_length, seed):
        self.quotas, self.proxy, self.max_length, self.seed = quotas, proxy, max_length, seed
        self.rows = {s:[] for s in quotas}
        self.counts = {s:Counter() for s in quotas}
        self.rejected = Counter()
        self.blocked = set()
        self.owners = {}
        self.seen = set()
        for path in heldout:
            for line in Path(path).read_text().splitlines():
                r=json.loads(line)
                ctx=r.get('context',r.get('state'))
                if ctx is None:
                    raise ValueError('Heldout row has no context/state')
                if not isinstance(ctx,str): ctx=json.dumps(ctx,ensure_ascii=False)
                self.blocked.add(context_hash(ctx))

    def complete(self, keys):
        return all(self.counts[s][k] == self.quotas[s][k] for s in self.quotas for k in keys)

    def add(self, rows, keys, whole=False):
        # Every row supplied here belongs to one semantic group.
        group=rows[0]['group_id']
        assert all(r['group_id']==group for r in rows)
        split=split_group(group,self.seed)
        hashes={context_hash(r['context']) for r in rows}
        if hashes & self.blocked:
            self.rejected['heldout_context_groups']+=1; return
        if any(h in self.owners and self.owners[h]!=split for h in hashes):
            self.rejected['cross_split_context_groups']+=1; return
        candidates=[]
        used=Counter()
        for r,key in zip(rows,keys):
            if key not in self.quotas[split] or self.counts[split][key]+used[key]>=self.quotas[split][key]:
                if whole:return
                continue
            # Semantic request hash ignores source ID and criterion insertion order.
            fp=hashlib.sha256(json.dumps([context_hash(r['context']),r['instruction'].strip(),
                r['question_type'],r['criteria']],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            if fp in self.seen:
                self.rejected['duplicate_request']+=1
                if whole:return
                continue
            try:
                ArbiterModel._prepare_decision(self.proxy, **{k:r[k] for k in
                    ('context','instruction','question_type','criteria')},
                    max_length=self.max_length,allow_truncation=False)
            except ValueError as e:
                # Fail on unexpected schema errors; only drop explicit length overflow.
                if 'token' not in str(e).lower() and 'length' not in str(e).lower():raise
                self.rejected['length']+=1
                if whole:return
                continue
            candidates.append((r,key,fp));used[key]+=1
        if whole and len(candidates)!=len(rows):return
        for r,key,fp in candidates:
            self.rows[split].append(r);self.counts[split][key]+=1;self.seen.add(fp)
        if candidates:
            for h in hashes:self.owners[h]=split


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True,help='Local tokenizer/config package; weights are not loaded')
    p.add_argument('--heldout',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--max-length',type=int,default=1024)
    a=p.parse_args()
    config=json.loads((a.model/'config.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    proxy=SimpleNamespace(tokenizer=tokenizer,option_token=config.get('option_token','<OPTION>'),
                          option_token_id=config['option_token_id'],
                          config=SimpleNamespace(max_position_embeddings=config.get('max_position_embeddings',4096)))
    if a.max_length<=0 or a.max_length>proxy.config.max_position_embeddings:p.error('Invalid max length')
    identity={'seed':a.seed,'max_length':a.max_length,'sources':SOURCES,
              'heldout_sha256':{str(x):digest(x) for x in a.heldout},
              'builder_sha256':{x.name:digest(x) for x in Path(__file__).parent.glob('*.py')},
              'tokenizer_sha256':{x.name:digest(x) for x in a.model.iterdir() if x.is_file() and
                                    ('token' in x.name or x.name=='config.json')}}
    # JSON roundtrip makes tuple/list representations stable for reuse.
    identity=json.loads(json.dumps(identity))
    manifest_path=a.output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['identity']==identity and all(digest(a.output/name)==sha for name,sha in old['output_sha256'].items()):
            print('Existing verified 80k dataset reused.',flush=True);return
        p.error('Existing dataset/config/hash mismatch; choose a new output directory')
    if a.output.exists():p.error('Incomplete output directory exists; choose a new output directory')
    rng=random.Random(a.seed)
    selector=Selector({'train':TRAIN_QUOTAS,'validation':VAL_QUOTAS},a.heldout,proxy,a.max_length,a.seed)
    source_files={}
    def fetch(source,filename):
        repo,rev,_=SOURCES[source]
        path=Path(hf_hub_download(repo,filename,revision=rev,repo_type='dataset'))
        source_files[f'{repo}/{filename}']=digest(path)
        return path
    def parquet(source,filename):
        rows=pq.read_table(fetch(source,filename)).to_pylist()
        rng.shuffle(rows)
        return rows
    def progress(name):
        print(name, {s:dict(c) for s,c in selector.counts.items()},flush=True)

    # Official Typed data first: no duplicated rows to inflate its limited volume.
    from datasets import load_dataset
    typed=load_dataset(SOURCES['typed'][0],'all',split='train',revision=SOURCES['typed'][1])
    indices=list(range(len(typed)));rng.shuffle(indices)
    for i in indices:
        rows=list(typed_rows(typed[i],SOURCES['typed'][1]))
        selector.add(rows,[f'typed/{r["question_type"]}' for r in rows])
    progress('Official Typed selected')

    # Separate published procedural configurations for workflow and policy groups.
    for family,configs in [('typed',['event_state_reconstruction','evidence_sufficiency','state_perturbation']),
                           ('policy',['policy_applicability','policy_under_uncertainty','taxonomy_routing'])]:
        streams=[iter(parquet('procedural',f'{c}/train-00000-of-00001.parquet')) for c in configs]
        active=list(streams)
        keys=[k for k in TRAIN_QUOTAS if k.startswith(family+'/')]
        while active and not selector.complete(keys):
            for stream in list(active):
                row=next(stream,None)
                if row is None:active.remove(stream);continue
                rows=list(procedural_rows(row,family))
                selector.add(rows,[f'{family}/{r["question_type"]}' for r in rows])
        progress(f'Procedural {family} selected')

    # NLI: one English premise defines a group; all 15 translations selected together.
    nli_keys=[k for k in TRAIN_QUOTAS if k.startswith('nli/')]
    seen_premises=set()
    criteria={'entailment':'The hypothesis follows from the premise.',
              'neutral':'The premise does not determine whether the hypothesis is true.',
              'contradiction':'The premise contradicts the hypothesis.'}
    for shard in range(4):
        filename=f'all_languages/train-{shard:05d}-of-00004.parquet'
        path=fetch('nli',filename)
        meta=json.loads(pq.ParquetFile(path).schema_arrow.metadata[b'huggingface'])
        labels=meta['info']['features']['label']['names']
        raw_rows=pq.read_table(path).to_pylist();rng.shuffle(raw_rows)
        for row in raw_rows:
            if selector.complete(nli_keys):break
            group=context_hash(row['premise']['en'])
            if group in seen_premises:continue
            seen_premises.add(group)
            hypotheses=dict(zip(row['hypothesis']['language'],row['hypothesis']['translation']))
            label=labels[row['label']]
            rows=[record('nli',f'{group}:{lang}',group,f'nli/{lang}',row['premise'][lang],
                         'Classify the relationship to this hypothesis: '+hypotheses[lang],
                         'choice',criteria,{k:float(k==label) for k in criteria},language=lang)
                  for lang in LANGUAGES]
            selector.add(rows,[f'nli/{label}']*len(rows),whole=True)
        del raw_rows
        if selector.complete(nli_keys):break
    progress('NLI selected')

    # Banking: cycle through labels for broad coverage without inventing repeated data.
    bank=parquet('banking','data/train-00000-of-00001.parquet')
    labels=sorted({r['label_text'] for r in bank})
    options={k:k.replace('_',' ') for k in labels}
    by_label=defaultdict(list)
    for r in bank:by_label[r['label_text']].append(r)
    while any(by_label.values()) and not selector.complete(['replay/banking']):
        for label in labels:
            if not by_label[label]:continue
            r=by_label[label].pop();group=context_hash(r['text'])
            row=record('banking',group,group,'replay/banking',r['text'],
                       'Select the banking intent that best describes the request.', 'choice',options,
                       {k:float(k==label) for k in labels})
            selector.add([row],['replay/banking'])
    progress('Banking replay selected')

    # MASSIVE: aligned IDs share a split across languages. Cycle languages and labels.
    mass=defaultdict(list)
    for lang in [lang for lang in LANGUAGES if lang != 'bg']:
        source_lang = 'zh-CN' if lang == 'zh' else lang
        with gzip.open(fetch('massive',f'train/{source_lang}.json.gz'),'rt') as f:
            rows=[json.loads(line) for line in f if line.strip()]
        rng.shuffle(rows)
        for row in rows:mass[(lang,row['label_text'])].append(row)
    labels=sorted({label for lang,label in mass})
    options={label:label.replace('_',' ') for label in labels}
    slots=sorted(mass)
    while any(mass.values()) and not selector.complete(['replay/massive']):
        for slot in slots:
            if not mass[slot]:continue
            r=mass[slot].pop();lang,label=slot
            row=record('massive',f'{r["id"]}:{lang}',str(r['id']),f'replay/massive/{lang}',r['text'],
                       'Select the intent expressed by the user.', 'choice',options,
                       {k:float(k==label) for k in labels},language=lang)
            selector.add([row],['replay/massive'])
    progress('MASSIVE replay selected')
    deficits={s:{k:q-selector.counts[s][k] for k,q in quotas.items() if selector.counts[s][k]!=q}
              for s,quotas in selector.quotas.items()}
    if any(deficits.values()):
        raise RuntimeError(f'Insufficient eligible unique data; no oversampling or silent substitution: {deficits}')
    report=audit(selector.rows['train'],selector.rows['validation'],a.heldout)
    # Write only after all quotas and audits pass; manifest marks a completed build.
    a.output.mkdir(parents=True)
    for split,rows in selector.rows.items():
        rng.shuffle(rows)
        (a.output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        read_rows(a.output/f'{split}.jsonl')
    manifest={'recipe':'public-80k-v1','identity':identity,'source_files_sha256':source_files,
              'counts':{s:dict(c) for s,c in selector.counts.items()},'rejected':dict(selector.rejected),
              'groups':{s:len({r['group_id'] for r in rows}) for s,rows in selector.rows.items()},
              'source_counts':{s:dict(Counter(r['source'] for r in rows)) for s,rows in selector.rows.items()},
              'type_counts':{s:dict(Counter(r['question_type'] for r in rows)) for s,rows in selector.rows.items()},
              'audit':report,'output_sha256':{f'{s}.jsonl':digest(a.output/f'{s}.jsonl') for s in selector.rows},
              'limitations':['Published procedural data, not new human annotations.',
                             'Exact audit only; previous base exposure and semantic overlap are not ruled out.',
                             'XNLI train translations have variable quality; source license metadata is unspecified.',
                             'Replay focuses on Banking77/MASSIVE; retention of other domains must be evaluated.',
                             'NLI uses the first sufficient shuffled shard(s), not a full-corpus uniform sample.',
                             'Grouping by source IDs does not ensure unseen templates or rules.']}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built 80,000 train + 8,000 validation decisions in {a.output}',flush=True)


if __name__=='__main__':main()
