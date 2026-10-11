"""Single-device supervised adaptation. Run from repository root with -m.

No test-based checkpoint selection; requires a separate grouped validation set.
Stores latest resumable state and best validation weights, never base weights.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

import torch
from safetensors.torch import save_file
from arbiter import ArbiterModel
from .data import read_rows, audit, prepare


def collate(rows, pad_id, device):
    length = max(len(r['ids']) for r in rows)
    k = max(len(r['target']) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    target = torch.zeros(len(rows), k)
    for i, r in enumerate(rows):
        ids[i, :len(r['ids'])] = torch.tensor(r['ids'])
        attention[i, :len(r['ids'])] = 1
        target[i, :len(r['target'])] = torch.tensor(r['target'])
    return dict(input_ids=ids.to(device), attention_mask=attention.to(device),
                option_indices=[r['positions'] for r in rows],
                question_types=[r['question_type'] for r in rows]), target.to(device)


def loss_terms(logits, targets, mask, qtypes):
    logits = logits.float().masked_fill(~mask, -1e4)
    ce = -(targets * logits.log_softmax(-1)).sum(-1)
    levels = torch.arange(logits.shape[-1], device=logits.device, dtype=torch.float32)
    mae = ((logits.softmax(-1) - targets) * levels).sum(-1).abs()
    score = torch.tensor([q == 'score' for q in qtypes], device=logits.device)
    return ce, mae, score


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


@torch.inference_mode()
def evaluate(model, rows, batch_size, device):
    model.eval()
    totals = defaultdict(lambda: [0., 0., 0., 0., 0.])
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        inputs, targets = collate(batch, model.tokenizer.pad_token_id or 0, device)
        with amp(device):
            out = model(**inputs)
        ce, mae, score = loss_terms(out['logits'], targets, out['mask'], inputs['question_types'])
        correct = out['logits'].argmax(-1).eq(targets.argmax(-1))
        values = torch.stack([ce, correct.float(), mae, score.float()], dim=1).cpu().tolist()
        for row, (c, acc, m, s) in zip(batch, values):
            t = totals[row['family']]
            for i, v in enumerate((1, c, acc, m*s, s)):
                t[i] += v
    result = {f: {'n': t[0], 'ce': t[1]/t[0], 'accuracy': t[2]/t[0],
                  'ordinal_mae': t[3]/t[4] if t[4] else None} for f, t in totals.items()}
    model.train()
    return {'families': result, 'macro_ce': sum(v['ce'] for v in result.values())/len(result)}


def atomic_checkpoint(path, payload):
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def export(model, path):
    path.mkdir(exist_ok=True)
    # Clone shared storages and explicitly export a BF16 inference package.
    state = {k: v.detach().cpu().to(torch.bfloat16).contiguous().clone() for k, v in model.state_dict().items()}
    temp = path / 'model.safetensors.tmp'
    save_file(state, str(temp))
    temp.replace(path / 'model.safetensors')
    config = model.config.to_dict()
    config['dtype'] = 'bfloat16'
    config['torch_dtype'] = 'bfloat16'
    (path / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    model.tokenizer.save_pretrained(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path, help='Frozen LOCAL package with tokenizer/config/weights')
    p.add_argument('--train', required=True, type=Path)
    p.add_argument('--validation', required=True, type=Path)
    p.add_argument('--heldout', required=True, nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--backbone-lr', type=float, default=5e-6)
    p.add_argument('--head-lr', type=float, default=2e-5)
    p.add_argument('--clip', type=float, default=3.)
    p.add_argument('--ordinal-lambda', type=float, default=.5)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', action='store_true', help='Resume this output/latest.pt (trusted local checkpoint only)')
    p.add_argument('--validate-only', action='store_true', help='Audit and tokenize, no forward/backward')
    a = p.parse_args()
    if any(v <= 0 for v in [a.epochs, a.batch_size, a.accumulation, a.max_length, a.eval_every, a.threads,
                            a.backbone_lr, a.head_lr, a.clip]) or a.ordinal_lambda < 0:
        p.error('Budgets/LRs/clip must be positive; ordinal-lambda nonnegative')
    if not a.model.is_dir() or a.output.resolve() == a.model.resolve() or a.model.resolve() in a.output.resolve().parents:
        p.error('Use an existing local base and a separate output directory')
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        p.error('Output is nonempty; choose another directory or --resume')
    train, val = read_rows(a.train), read_rows(a.validation)
    report = audit(train, val, a.heldout)
    if set(r['family'] for r in train) != set(r['family'] for r in val):
        p.error('Train and validation must cover identical families')
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items() if k not in ('resume', 'validate_only', 'heldout')}
    report['config'] = config
    report['inputs'] = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.train, a.validation]}
    report['base_files'] = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in a.model.iterdir() if x.is_file()}
    report['code'] = {str(x.relative_to(Path(__file__).parents[1])): hashlib.sha256(x.read_bytes()).hexdigest()
                      for folder in (Path(__file__).parent, Path(__file__).parents[1]/'arbiter') for x in folder.glob('*.py')}
    report['versions'] = {'torch': torch.__version__, 'python': sys.version}
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = a.output/'manifest.json'
    if a.resume and (not manifest.exists() or json.loads(manifest.read_text()) != report):
        p.error('Resume manifest mismatch: data, base, code or configuration changed')
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device('cpu' if a.validate_only else a.device)
    model = ArbiterModel.from_pretrained(a.model, device=device, dtype=torch.float32)
    train, val = prepare(model, train, a.max_length), prepare(model, val, a.max_length)
    if a.validate_only:
        print(json.dumps(report, indent=2))
        return
    from torch.utils.tensorboard import SummaryWriter
    model.train()
    optimizer = torch.optim.AdamW([{'params': model.language_model.parameters(), 'lr': a.backbone_lr},
                                  {'params': model.decision_head.parameters(), 'lr': a.head_lr}], weight_decay=.01)
    window_size = a.batch_size*a.accumulation
    total_steps = math.ceil(len(train)/window_size)*a.epochs
    warmup = max(1, math.ceil(total_steps*.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min((s+1)/warmup, 1.) *
                                                max(0., (total_steps-s)/max(1, total_steps-warmup)) if s >= warmup
                                                else (s+1)/warmup)
    step, start_epoch, next_window, best = 0, 0, 0, float('inf')
    if a.resume:
        saved = torch.load(a.output/'latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step, start_epoch, next_window, best = (saved[k] for k in ('step','epoch','next_window','best'))
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    writer = SummaryWriter(str(a.output/'tensorboard'), purge_step=step+1 if a.resume else None)
    def validate():
        nonlocal best
        result = evaluate(model, val, a.batch_size, device)
        for family, values in result['families'].items():
            for metric, value in values.items():
                if value is not None:
                    writer.add_scalar(f'Validation/{family}/{metric}', value, step)
        writer.add_scalar('Validation/macro_ce', result['macro_ce'], step)
        with (a.output/'validation.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **result})+'\n')
        if result['macro_ce'] < best:
            best = result['macro_ce']
            export(model, a.output/'best')
    if not a.resume:
        validate()  # Includes step-zero baseline, which may remain best.
    try:
        for epoch in range(start_epoch, a.epochs):
            order = list(range(len(train)))
            random.Random(a.seed+epoch).shuffle(order)
            for offset in range(next_window if epoch == start_epoch else 0, len(order), window_size):
                rows = [train[i] for i in order[offset:offset+window_size]]
                n_score = sum(r['question_type'] == 'score' for r in rows)
                optimizer.zero_grad(set_to_none=True)
                ce_sum = torch.zeros((), device=device)
                ord_sum = torch.zeros((), device=device)
                for start in range(0, len(rows), a.batch_size):
                    inputs, target = collate(rows[start:start+a.batch_size], model.tokenizer.pad_token_id or 0, device)
                    with amp(device):
                        out = model(**inputs)
                        ce, mae, is_score = loss_terms(out['logits'], target, out['mask'], inputs['question_types'])
                        loss = ce.sum()/len(rows) + a.ordinal_lambda*mae[is_score].sum()/max(1,n_score)
                    loss.backward()
                    ce_sum += ce.detach().sum()
                    ord_sum += mae.detach()[is_score].sum()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0:
                    vals = torch.stack((ce_sum/len(rows), ord_sum/max(1,n_score), norm)).cpu().tolist()
                    for key, value in zip(('Loss/CE','Loss/Ordinal_MAE','Gradient/pre_clip_norm'), vals):
                        writer.add_scalar(key, value, step)
                    writer.add_scalar('Gradient/clip_scale', min(1., a.clip/(vals[2]+1e-6)), step)
                    writer.add_scalar('LR/backbone', optimizer.param_groups[0]['lr'], step)
                if step % a.eval_every == 0 or offset+window_size >= len(order):
                    validate()
                    atomic_checkpoint(a.output/'latest.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), epoch=epoch, next_window=offset+window_size, step=step, best=best,
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
                    writer.flush()
    finally:
        writer.close()
    print(f'Finished {step} optimizer steps. Selected package: {a.output / "best"}')


if __name__ == '__main__':
    main()
