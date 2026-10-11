# Development and release checks

Use Python 3.11 or newer and run commands from the repository root.
The wheel provides `arbiter` inference; data preparation, fine-tuning and benchmark
commands are source-checkout workflows. Their extras install dependencies only.

## Install and test

```bash
python -m pip install -e '.[dev,finetuning,benchmark]'
CUDA_VISIBLE_DEVICES='' python -m pytest tests benchmark/tests -m 'not integration' -q
python scripts/check_docs.py
python -m build
```

Unit tests require no downloaded weights or GPU. Real-weight tests are marked
`integration`; after obtaining the complete model package, run:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests benchmark/tests -q
```

The private-reference parity script is optional and is not collected by pytest.
The notebook must remain in English with no saved outputs. Run it cell by cell;
Run All starts training. Do not execute it in automated documentation checks.

## Compatibility

| Component | Declared support | Locally tested during review |
|---|---|---|
| Python | >=3.11 | 3.12.3 |
| PyTorch | >=2.0 | 2.14.1+cu130 |
| Transformers | >=5.19 | 5.19.0 |
| safetensors | >=0.4 | 0.8.0 |
| Device | CPU; CUDA with suitable PyTorch installation | CPU tests; earlier RTX 5090 training smoke |

Lower dependency bounds are not a claim that every combination has been tested.
CI checks Python 3.11 and 3.12 with resolved dependencies. Record exact installed
versions for every release; CUDA wheels depend on the target platform.

## Changes

Keep input/output contracts stable, add focused tests for behavior changes, and use
clear functions and conventional Python formatting. Avoid unrelated mass formatting
while an experiment is running: the trainer hashes its source and the model code for
resume. Do not edit these files or base artifacts during a protected run.

## Release checklist

1. Run unit and real-weight tests, documentation checks and wheel build.
2. Install the wheel in a separate environment and import `arbiter` outside the checkout.
3. Check one choice, noul, score and mixed-batch request with the exact release weights.
4. For trainer changes, run a short continuous/interrupted-resumed comparison and
   export/reload verification as described in `docs/finetuning_validation.md`.
5. Record commit, dependency versions, model/config/tokenizer hashes and HF revision.
6. Replace the download placeholder with the accessible release SHA in user examples.
7. Update the changelog. Keep code, weight and dataset licensing distinct.

CI does not run full training or publish weights. Publication is a separate action.
