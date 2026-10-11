"""Evaluate a local package on the fixed adaptation development set, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from arbiter import ArbiterModel
from finetuning.data import read_rows, prepare
from finetuning.train import evaluate, collate, amp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=4)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new file.')
    torch.set_num_threads(4)
    device = torch.device(a.device)
    model = ArbiterModel.from_pretrained(a.model, device=device, dtype=torch.float32)
    rows = prepare(model, read_rows(a.validation), a.max_length)
    result = evaluate(model, rows, a.batch_size, device)
    # Diagnose ordinal argmax errors separately from expected-score error.
    model.eval()
    ordinal = {}
    with torch.inference_mode():
        for row in rows:
            if row['question_type'] != 'score':
                continue
            inputs, target = collate([row], model.tokenizer.pad_token_id or 0, device)
            with amp(device):
                out = model(**inputs)
            logits = out['logits'].float().masked_fill(~out['mask'], -1e4)
            probs = logits.softmax(-1)[0]
            gold, pred = int(target[0].argmax()), int(probs.argmax())
            k = row['family']
            d = ordinal.setdefault(k, {'n': 0, 'correct': 0, 'confusion': {}, 'target_entropy_sum': 0.0})
            d['n'] += 1
            d['correct'] += int(gold == pred)
            pair = f'{gold}->{pred}'
            d['confusion'][pair] = d['confusion'].get(pair, 0) + 1
            t = target[0].float()
            d['target_entropy_sum'] += float(-(t * t.clamp_min(1e-12).log()).sum())
    result['ordinal_diagnostics'] = ordinal
    result['provenance'] = {
        'model': str(a.model),
        'weights_sha256': hashlib.sha256((a.model/'model.safetensors').read_bytes()).hexdigest(),
        'validation_sha256': hashlib.sha256(a.validation.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'device': str(device), 'max_length': a.max_length,
        'precision': 'FP32 weights, BF16 autocast on CUDA (trainer evaluation protocol)',
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(a.output)


if __name__ == '__main__':
    main()
