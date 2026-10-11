"""Bounded offline load test; run from an installed ARBITER environment.
python examples/benchmark_batch.py --model checkpoints/arbiter-272m --device cpu
Use a free GPU for deployment measurements. No HTTP concurrency is simulated.
"""
import argparse
import json
import time
import statistics
import torch
from arbiter import ArbiterModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--max-batch-tokens', type=int, default=8192)
    parser.add_argument('--max-attention-elements', type=int, default=16777216)
    args = parser.parse_args()
    if args.requests < 1 or args.repeats < 1:
        parser.error('requests and repeats must be positive')
    model = ArbiterModel.from_pretrained(args.model, device=args.device)
    device = next(model.parameters()).device
    data = [dict(context='A payment was charged twice. '*(1+i%8), instruction='Classify the issue.',
                 criteria=['duplicate payment', 'account access', 'delivery']) for i in range(args.requests)]
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    report = []
    reference = None
    for size in args.batch_sizes:
        options = dict(batch_size=size, max_batch_tokens=args.max_batch_tokens,
                       max_attention_elements=args.max_attention_elements)
        model.decide_batch(data[:min(size, len(data))], **options)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            outputs = model.decide_batch(data, **options)
            sync()
            times.append(time.perf_counter()-start)
        if reference is None:
            reference = outputs
        report.append(dict(batch_size=size, requests=len(data), seconds=times,
            median_requests_per_second=len(data)/statistics.median(times),
            label_changes_vs_first_configuration=sum(a['selected_key'] != b['selected_key'] for a,b in zip(reference,outputs)),
            max_probability_delta_vs_first_configuration=max(abs(a['probabilities'][k]-b['probabilities'][k]) for a,b in zip(reference,outputs) for k in a['probabilities']),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
    print(json.dumps(dict(device=str(device), dtype=str(next(model.parameters()).dtype),
                          scope='offline end-to-end batch calls, synthetic short inputs', results=report), indent=2))

if __name__ == '__main__':
    main()
