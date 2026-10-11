"""Paired CPU/CUDA evaluation on frozen official Laya requests; one model per process."""
import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

os.environ.setdefault('USE_TF', '0')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from .metrics import observation, summarize
except ImportError:
    try:
        from metrics import observation, summarize
    except ImportError:
        from benchmark.metrics import observation, summarize

os.environ.setdefault('HF_MODULES_CACHE', str(ROOT/'results/hf_modules'))
MODELS = {
    'laya': ('convaiinnovations/laya', '55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851'),
    'laya-multilingual': ('convaiinnovations/laya-multilingual', 'e4e9ddf21a7b1903b7acffd8814ad4307bf63a67'),
    'laya-typed-decisions': ('convaiinnovations/laya-typed-decisions', '1a793eb568e6718f15941d08f85432581df534e3'),
    'gliner25-decide': ('fastino/GLiNER2.5-Decide', None),
    'gliner25-multi-decide': ('helmo/GLiNER2.5-multi-Decide', None),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def evaluate(args):
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13)
    device = torch.device(args.device)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Use cpu, cuda or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: choose --device cpu or install a CUDA-enabled PyTorch')
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
        device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    raw = args.data.read_bytes()
    manifest = json.loads(args.data.with_name('manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest() != manifest['cases_sha256']:
        raise ValueError('Dataset differs from preparation manifest')
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    selected, counts = [], defaultdict(int)
    for r in rows:
        if args.suites and r['suite'] not in args.suites:
            continue
        if args.limit and counts[r['suite']] >= args.limit:
            continue
        counts[r['suite']] += 1
        selected.append(r)
    if not selected:
        raise ValueError('No matching cases')
    if args.suites and set(args.suites)-set(counts):
        raise ValueError(f'Unknown/empty suites: {set(args.suites)-set(counts)}')
    destination = args.output / args.model
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'predictions.jsonl').exists():
        raise FileExistsError(f'{destination} already has predictions; use a new --output')
    metadata = {'model': args.model, 'device': str(device), 'threads': args.threads,
                'batch_size': 1, 'warmup_requests': 3, 'pilot': bool(args.limit),
                'limit_per_suite': args.limit, 'dataset_sha256': manifest['cases_sha256'],
                'selection_sha256': fingerprint(selected), 'cases': len(selected),
                'upstream_commit': manifest['upstream_commit'],
                'platform': platform.platform(), 'cpu': platform.processor(),
                'logical_cpus': os.cpu_count(), 'packages': {},
                'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for d in [HERE, ROOT/'hf_pipeline'] for p in d.glob('*.py')},
                'status': 'loading'}
    if device.type == 'cuda':
        metadata.update(gpu_name=torch.cuda.get_device_name(device),
                        gpu_total_memory_mib=torch.cuda.get_device_properties(device).total_memory / 1024**2,
                        cuda_runtime=torch.version.cuda)
    for package in ['torch','transformers','sentence-transformers','numpy','laya','datasets']:
        try:
            metadata['packages'][package] = version(package)
        except PackageNotFoundError:
            pass
    meta_file = destination/'metadata.json'
    def save_meta():
        meta_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n')
    save_meta()
    start = time.perf_counter()
    try:
        if args.model == 'trained-eg2':
            sys.path.insert(0, str(ROOT))
            from training.export_pipeline import DecisionPredictor
            if not args.checkpoint:
                raise ValueError('--checkpoint is required for trained-eg2')
            agent = DecisionPredictor(str(args.checkpoint), device=str(device))
            def predict(row):
                state = row['state']
                context = json.dumps(state, ensure_ascii=False) if isinstance(state, dict) else str(state)
                return {'answers': {qid: agent.predict_single(
                    context, q.get('instructions', ''), q.get('criteria'), q.get('type', 'choice'))
                    for qid, q in row['questions'].items()}}
            with args.checkpoint.open('rb') as checkpoint_handle:
                checkpoint_sha = hashlib.file_digest(checkpoint_handle, 'sha256').hexdigest()
            metadata.update(checkpoint=str(args.checkpoint), dtype=str(next(agent.model.parameters()).dtype),
                            checkpoint_sha256=checkpoint_sha,
                            token_budget='untruncated, matching training/eval_test.py')
        elif args.model == 'embeddinggemma2':
            from transformers import pipeline
            pipe = pipeline('decision-making', model=str(ROOT/'hf_pipeline'), trust_remote_code=True,
                            device=str(device), dtype='auto')
            predict = lambda row: pipe({'state': row['state'], 'questions': row['questions']})
            metadata.update(pipe.model.engine.metadata)
            metadata['model'] = args.model
            metadata['config_sha256'] = hashlib.sha256((ROOT/'hf_pipeline/config.json').read_bytes()).hexdigest()
        elif args.model in ('arbiter', 'arbiter-272m'):
            import arbiter
            from arbiter import ArbiterModel
            candidate_paths = [
                args.checkpoint,
                ROOT / "checkpoints/arbiter-272m",
                ROOT.parent / "arbiter/checkpoints/arbiter-272m",
            ]
            model_path = next((p for p in candidate_paths if p and Path(p).exists()), None)
            if model_path is None:
                raise FileNotFoundError("Could not find arbiter-272m checkpoint directory. Specify --checkpoint.")
            agent = ArbiterModel.from_pretrained(str(model_path), device=str(device), local_files_only=True)
            predict = lambda row: agent.predict(row['state'], row['questions'])
            with (Path(model_path) / "model.safetensors").open('rb') as f:
                safetensors_sha = hashlib.file_digest(f, 'sha256').hexdigest()
            metadata.update(
                arbiter_package_file=arbiter.__file__,
                arbiter_version=getattr(arbiter, '__version__', 'unknown'),
                checkpoint=str(model_path),
                dtype=str(next(agent.parameters()).dtype),
                checkpoint_sha256=safetensors_sha,
                token_budget='4096 tokens max, strict truncation guard',
            )
        elif args.model.startswith('gliner25-'):
            from gliner_adapter import GLiNERDecisionAgent
            repo, _ = MODELS[args.model]
            agent = GLiNERDecisionAgent(repo, str(device))
            predict = lambda row: agent.predict(row['state'], row['questions'])
            metadata.update(repo=repo, dtype='model-default')
        else:
            import laya
            repo, revision = MODELS[args.model]
            agent = laya.load(repo, device=str(device), revision=revision)
            predict = lambda row: agent.predict(row['state'], row['questions'])
            metadata.update(repo=repo, revision=revision, dtype=str(next(agent.model.parameters()).dtype),
                            token_budget={k: agent.cfg.get(k) for k in ['max_len', 'head_max_len']})
        if args.model == 'embeddinggemma2':
            actual_device = pipe.model.device
        elif args.model in ('trained-eg2',):
            actual_device = next(agent.model.parameters()).device
        elif args.model in ('arbiter', 'arbiter-272m'):
            actual_device = next(agent.parameters()).device
        elif args.model.startswith('gliner25-'):
            actual_device = device
        else:
            actual_device = next(agent.model.parameters()).device
        if actual_device != device:
            raise RuntimeError(f'Requested {device} but model loaded on {actual_device}')
        metadata['device'] = str(actual_device)
        synchronize()
        metadata['load_seconds'] = time.perf_counter()-start
    except Exception as exc:
        metadata.update(status='load_failed', error=f'{type(exc).__name__}: {exc}')
        save_meta()
        raise
    for row in selected[:3]:
        try:
            predict(row)
        except Exception:
            pass  # The measured call still records failures; no example is discarded.
    records = []
    metadata['status'] = 'running'
    save_meta()
    with (destination/'predictions.jsonl').open('x') as handle:
        for i, row in enumerate(selected):
            record = {'id': row['id'], 'suite': row['suite'], 'request_sha256': fingerprint(
                {'state': row['state'], 'questions': row['questions']}),
                'gold_sha256': fingerprint(row['gold']), 'expected_answers': len(row['gold'])}
            synchronize()
            start = time.perf_counter()
            try:
                response = predict(row)
                synchronize()
                record['latency_ms'] = (time.perf_counter()-start)*1000
                record['response'] = response
                record['observations'] = {qid: observation(row['questions'][qid], response['answers'][qid], gold)
                                          for qid, gold in row['gold'].items()}
            except Exception as exc:
                record.update(latency_ms=(time.perf_counter()-start)*1000,
                              error=f'{type(exc).__name__}: {exc}', observations={})
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            handle.flush()
            records.append(record)
            if (i+1) % 25 == 0 or i+1 == len(selected):
                print(f'{args.model}: {i+1}/{len(selected)}', flush=True)
    summaries = {}
    for suite in counts:
        rs = [r for r in records if r['suite'] == suite]
        obs = [o for r in rs for o in r['observations'].values()]
        total = sum(r['expected_answers'] for r in rs)
        successes = [r['latency_ms'] for r in rs if 'error' not in r]
        summaries[suite] = {**summarize(obs), 'expected_answers': total,
                            'failed_requests': sum('error' in r for r in rs),
                            'accuracy_including_failures': sum(o['correct'] for o in obs)/total,
                            'coverage': len(obs)/total,
                            'latency_p50_ms': float(np.median(successes)) if successes else None,
                            'latency_p95_ms': float(np.percentile(successes,95)) if successes else None,
                            'by_type': {kind: summarize([o for o in obs if o['kind']==kind])
                                        for kind in sorted({o['kind'] for o in obs})}}
    (destination/'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False)+'\n')
    # Linux ru_maxrss is KiB, macOS bytes; includes loading and inference in this fresh process.
    if device.type == 'cuda':
        synchronize()
        metadata.update(peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
                        peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2)
    metadata.update(status='completed', failed_requests=sum('error' in r for r in records),
                    peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                    (1024**2 if sys.platform == 'darwin' else 1024))
    save_meta()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['all','embeddinggemma2','trained-eg2','arbiter','arbiter-272m',*MODELS], default='all')
    p.add_argument('--checkpoint', type=Path, help='EG2 checkpoint file or ARBITER inference-package directory')
    p.add_argument('--data', type=Path, default=HERE/'data/cases.jsonl')
    p.add_argument('--output', type=Path, default=ROOT/'results/laya_comparison')
    p.add_argument('--suites', nargs='+')
    p.add_argument('--limit', type=int, default=0, help='First N cases per suite; 0 = official full sample')
    p.add_argument('--device', default='cpu', help='cpu, cuda or cuda:N')
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    if args.limit < 0 or args.threads < 1:
        p.error('limit must be nonnegative and threads positive')
    if args.model == 'all':
        failures = []
        for model in ['embeddinggemma2', *MODELS]:
            cmd = [sys.executable, __file__, '--model',model,'--data',str(args.data),
                   '--output',str(args.output),'--threads',str(args.threads),'--limit',str(args.limit),'--device',args.device]
            if args.suites:
                cmd += ['--suites',*args.suites]
            if subprocess.run(cmd).returncode:
                failures.append(model)
        if failures:
            raise SystemExit(f'Failed runs: {failures}')
    else:
        evaluate(args)
