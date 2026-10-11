# Batched inference

`decide_batch` combines independent decisions in one backbone forward per scheduled
batch. `predict` now batches its questions; `predict_batch` also combines questions
across states. Results retain input order, question IDs and the existing answer schema.
Use `batch_size=1` to retain single-request execution behavior.

```python
from arbiter import ArbiterModel

model = ArbiterModel.from_pretrained("checkpoints/arbiter-272m", device="cpu")
results = model.decide_batch([
    {"context": "My parcel is late.", "instruction": "Classify the issue.",
     "criteria": ["delivery", "payment"]},
    {"context": "Earth orbits the sun.", "instruction": "Is this true?",
     "criteria": None, "question_type": "noul"},
], batch_size=8, max_batch_tokens=8192, max_attention_elements=16777216)

answers = model.predict_batch([
    {"state": "My parcel is late.", "questions": {
        "category": {"instructions": "Classify the issue.", "type": "choice",
                     "criteria": ["delivery", "payment"]},
        "urgent": {"instructions": "Is immediate action needed?", "type": "noul"},
    }},
], batch_size=8)
```

Inputs are prepared on CPU, sorted by token length and right-padded. Option positions
are extracted on CPU. Each batch transfers its probabilities back together. Candidate
heads still execute separately inside the shared backbone forward.

All requests are validated before any forward. Invalid requests raise with their index;
there are no partial results. Empty lists return empty lists. `max_length` (default
4096) and `allow_truncation` (default false) can be set globally or per `decide_batch`
request. Truncation only removes context. Budget overflow for a singleton raises;
it never silently truncates or exceeds the scheduling budget.

The limits bound padded `B*L` and `B*L*L`; they are **not byte-level VRAM guarantees**.
Tune them on the target device with representative lengths and option counts. CPU
memory holds the prepared request list: callers must bound its size. The model must
be in evaluation mode (the loader already sets it). BF16 batching can change rounding,
probabilities and close decisions; exact bitwise parity is not promised.

## Measure before deployment

```bash
python examples/benchmark_batch.py --model checkpoints/arbiter-272m \
  --device cpu --batch-sizes 1 4 8 --requests 16 --repeats 3
# On a free GPU, repeat with --device cuda:0 and representative production inputs.
```

This offline harness measures end-to-end batch throughput, label/probability changes,
and CUDA peak allocated memory. It is not a concurrent HTTP load test. It includes
warmup and CUDA synchronization when applicable. Its default synthetic short examples
cannot establish long-context production performance.

Production services still need a bounded queue, admission control, timeouts,
backpressure, readiness checks and concurrent load testing. These APIs dynamically
size batches within a supplied list; they do not combine requests arriving over time.
Use a dedicated model worker and profile the target workload before promising latency
or concurrency. No GPU acceleration claim follows from the CPU integration checks.

## Input and output contracts

- Criterion keys containing the reserved `<OPTION>` marker are rejected before
  inference. Other keys are returned unchanged; literal marker text in descriptions
  is sanitized. Rejecting keys avoids merging `<OPTION>` and `[OPTION]` labels.
- `predict` and `predict_batch` serialize dictionary **and list** states with
  `json.dumps(..., ensure_ascii=False)`. This changes the prompt for list states
  compared with the previous Python-repr serialization; revalidate those workflows.
- Questions accept `instructions` or its alias `instruction`. If both are present,
  their values must agree; otherwise the request is rejected. Empty instructions
  remain invalid.
- A `score` result is the expected **zero-based position**, `sum(i * p[i])`, in
  `[0, K-1]`. Numeric-looking keys do not change that scale. For ordered levels
  1–5, add 1 to the result; for arbitrary values, compute `sum(value[i] * p[i])`
  explicitly. Options must be supplied in their intended ordinal order.
- `from_pretrained` supports a single `device` and `dtype`/`torch_dtype`.
  Remote-only download keywords are `revision`, `cache_dir`, `token`, and
  `force_download`. Unsupported keywords (including `device_map`) are rejected;
  download keywords are also rejected for existing local paths. No automatic
  multi-device dispatch is implemented.

The low-level `forward()` retains normal PyTorch training behavior. Only the public
inference APIs require evaluation mode; this preserves the ability to fine-tune.
