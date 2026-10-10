"""Run the frozen epoch-2 checkpoint on CPU and write its comparison report."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
output = ROOT / 'results/epoch2_cpu_full'
output.mkdir(exist_ok=True)
status = output / 'status.json'
status.write_text(json.dumps({'status': 'running', 'device': 'cpu', 'epoch': 2}))
command = [sys.executable, '-B', 'benchmark/run.py', '--model', 'trained-eg2',
           '--checkpoint', 'checkpoints/multitask_epoch2_cpu/source_checkpoint.pt',
           '--device', 'cpu', '--threads', '4', '--output', str(output)]
try:
    subprocess.run(command, check=True)
    report = json.loads((output / 'trained-eg2/summary.json').read_text())
    groups = {
        'Typed': lambda s: s == 'typed_decisions',
        'MASSIVE intent': lambda s: s.startswith('massive_intent.'),
        'XNLI': lambda s: s.startswith('xnli.'),
        'Classification EN': lambda s: s.startswith('en.'),
        'Applications': lambda s: s.startswith(('app.', 'jev.')),
    }
    def accuracy(predicate):
        rows = [r for s, r in report.items() if predicate(s)]
        n = sum(r['expected_answers'] for r in rows)
        return sum(r['accuracy_including_failures'] * r['expected_answers'] for r in rows) / n
    scores = {name: accuracy(pred) for name, pred in groups.items()}
    baseline = json.loads(Path('results/comparison_all_v2.json').read_text())['models']
    mapping = ['Typed decisions', 'MASSIVE intent', 'XNLI', 'English classification', 'Applications']
    lines = ['# Exploratory CPU Evaluation — EG2 Epoch 2', '',
             'Checkpoint frozen at epoch 2, step 6188. CPU FP32, 4 threads, CUDA masked.',
             'Identical complete dataset as references; CPU latencies not directly comparable to GPU measurements.',
             'Exploratory test requested for diagnostics; not used for final checkpoint selection.', '',
             '| Model | ' + ' | '.join(groups) + ' |',
             '|---|' + '---:|' * len(groups)]
    for name, model in baseline.items():
        cells = [f"{model[g]['accuracy_including_failures'] * 100:.2f} %" for g in mapping]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines.append('| **EG2 Multi-Task Epoch 2** | ' + ' | '.join(f'**{s*100:.2f} %**' for s in scores.values()) + ' |')
    failures = sum(r['failed_requests'] for r in report.values())
    lines += ['', f'Failed requests: {failures}. Counted as incorrect in the summary table.',
              'MASSIVE scenario is also available in summary.json.',
              'Predictor uses untruncated prompts matching training/eval_test.py; context budgets differ across external models.',
              'Sources: results/epoch2_cpu_full/trained-eg2/{metadata,summary}.json and predictions.jsonl.', '']
    text = '\n'.join(lines)
    (output / 'comparison.md').write_text(text)
    Path('docs/epoch2_cpu_benchmark.md').write_text(text)
    status.write_text(json.dumps({'status': 'completed', 'epoch': 2, 'device': 'cpu',
                                  'accuracy': scores, 'failed_requests': failures}, indent=2))
except Exception as exc:
    status.write_text(json.dumps({'status': 'failed', 'error': str(exc)}, indent=2))
    raise
