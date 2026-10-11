"""Verify two local resume checkpoints and a trained BF16 export on real inputs.

Usage: python -m finetuning.verify_run --base checkpoints/arbiter-272m
 --continuous results/run-a --resumed results/run-b --validation data/validation.jsonl
 --output results/verification --device cuda
Only load checkpoints you created and trust.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import torch
from arbiter import ArbiterModel
from .data import read_rows
from .train import export


def compare(a, b, path='state'):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError(f'Tensor differs: {path}; max abs delta {(a.float()-b.float()).abs().max().item()}')
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x,y) in enumerate(zip(a,b)):
            compare(x,y,f'{path}/{i}')
    else:
        assert a == b, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','continuous','resumed','validation','output'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new verification output directory')
    torch.set_num_threads(4)
    first = torch.load(a.continuous/'latest.pt',map_location='cpu',weights_only=False)
    second = torch.load(a.resumed/'latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','rng','cuda_rng','step','epoch','next_window','best'):
        compare(first[key],second[key],key)
    report = {'resume_exact_equal':True,'step':first['step']}
    del second
    model = ArbiterModel.from_pretrained(a.base,device='cpu',dtype=torch.float32)
    report['parameters_changed'] = sum(not torch.equal(v,first['model'][k]) for k,v in model.state_dict().items())
    assert report['parameters_changed'] > 0, 'No training update detected'
    model.load_state_dict(first['model'])
    del first
    a.output.mkdir(parents=True)
    export(model,a.output/'trained-export')
    requests = [{k:r[k] for k in ('context','instruction','criteria','question_type')} for r in read_rows(a.validation)]
    model.to(device=a.device,dtype=torch.bfloat16).eval()
    reference = model.decide_batch(requests,batch_size=4,max_length=1024)
    del model
    gc.collect()
    if a.device.startswith('cuda'):
        torch.cuda.empty_cache()
    restored = ArbiterModel.from_pretrained(a.output/'trained-export',device=a.device)
    actual = restored.decide_batch(requests,batch_size=4,max_length=1024)
    report['export_prediction_exact_equal'] = reference == actual
    assert reference == actual, 'Export/reload predictions differ'
    report['export_test_decisions'] = len(requests)
    report['export_question_types'] = sorted({r['question_type'] for r in requests})
    report['export_sha256'] = hashlib.sha256((a.output/'trained-export/model.safetensors').read_bytes()).hexdigest()
    manifest = json.loads((a.continuous/'manifest.json').read_text())
    report['base_unchanged'] = all(hashlib.sha256((a.base/k).read_bytes()).hexdigest()==v for k,v in manifest['base_files'].items())
    assert report['base_unchanged'], 'Base package changed'
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
