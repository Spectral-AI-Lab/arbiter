# Reproduce ARBITER-272M-Typed

This is the **Typed-only, three-epoch recipe**, not the 80k or 80/20 replay run.
Observed Typed accuracy: 76.45%; base: 65.50%; Laya Typed: 76.75%.
The selected checkpoint is step 510, chosen by macro validation CE.

Run commands from the repository root with the documented dependencies installed.
Obtain the original base package in `checkpoints/arbiter-272m`. Its weights SHA-256 must be:
`f095a52596496660f2a6af042763052f22c9824e08f29d506d6d7128b54f4f77`.
Use a fresh output directory; do not overwrite the published specialist.

## Prepare the dataset

```bash
python scripts/prepare_adaptation.py --preset typed \
  --model checkpoints/arbiter-272m \
  --heldout benchmark/data/cases.jsonl benchmark/data/public_requested_v1/cases.jsonl \
  --output results/adaptation/typed --seed 13 --max-length 1024
```

Expected counts: 5,410 train / 590 validation. Expected SHA-256:

- Train: `79ff2ffe7a0369ab3559a3a226e788333ac53822bbbb0a2821d4ad084a7e48bd`
- Validation: `b0a0163878b38a9496f2ac560fe1439cfd5cc743f8723cbd48386554c9d16727`

## Train and monitor

```bash
python -m finetuning.train \
  --model checkpoints/arbiter-272m \
  --train results/adaptation/typed/train.jsonl \
  --validation results/adaptation/typed/validation.jsonl \
  --heldout benchmark/data/cases.jsonl benchmark/data/public_requested_v1/cases.jsonl \
  --output results/typed-specialist-reproduction \
  --device cuda:0 --epochs 3 --batch-size 4 --accumulation 8 \
  --max-length 1024 --backbone-lr 1e-5 --head-lr 4e-5 \
  --clip 3 --ordinal-lambda 0.5 --eval-every 50 --log-every 25 \
  --threads 4 --seed 13 --deterministic
```

The default selection metric is `macro_ce`, matching the original run. Do not use
`constrained_typed` for this reproduction. There are 170 optimizer windows per epoch,
including the partial last window: **510 steps** in total.

In another terminal:

```bash
tensorboard --logdir results/typed-specialist-reproduction/tensorboard --host 0.0.0.0 --port 6006
```

In `notebooks/finetuning_arbiter.ipynb`, select `CONFIG_NAME = "typed"`, set a new
`RUN_NAME`, then execute preparation, preflight and training in order. The notebook
uses the same three-epoch preset. The original environment and file hashes are
in the specialist's HF `reproduction.json`. Original trainer code matches Git commit
`82b6f61b02873b103f82be91949106b853a4da68`; matching one file is not a claim that every
file in that commit matches the run. Compare the full recorded code hashes for strict
reproduction. Deterministic mode alone does not guarantee cross-hardware identity.

## Benchmark the selected package

```bash
python -m benchmark.run --model arbiter \
  --checkpoint results/typed-specialist-reproduction/best \
  --data benchmark/data/cases.jsonl --device cuda:0 --threads 8 \
  --output results/typed-specialist-reproduction-eval/historical
python -m benchmark.run --model arbiter \
  --checkpoint results/typed-specialist-reproduction/best \
  --data benchmark/data/public_requested_v1/cases.jsonl --device cuda:0 --threads 8 \
  --output results/typed-specialist-reproduction-eval/public
```

Preserve metadata, predictions and summaries. Evaluate the base on these same files
when comparing retention. Select checkpoints on validation only, not test results.
The published single-seed result is an observation, not a guaranteed score for every run.
