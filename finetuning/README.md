# Fine-tune ARBITER from the command line

You do not need a notebook. Run the commands below **from the ARBITER repository root**,
not from this `finetuning/` directory.

The workflow is: **build data → check data → train → evaluate**.
The original model stays unchanged. Your adapted model is saved in a separate folder.

## 1. Install

Activate your Python environment, then install the project:

```bash
python -m pip install -e '.[finetuning,benchmark]'
```

You need PyTorch with CUDA support and a BF16-capable GPU for the training command below.
The workflow has been tested on an RTX 5090. Check your environment:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('BF16 supported:', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
```

The base model package must be in `checkpoints/arbiter-272m/` and include the weights,
configuration and tokenizer. If you have not downloaded it yet:

```bash
hf download spectral-ai-lab/arbiter-272m --revision <release-commit> --local-dir checkpoints/arbiter-272m
```

Replace `<release-commit>` with the release SHA when the weights are accessible.
See [weight availability](../README.md#obtain-the-weights); access is required for a private release.
Do not download over the base directory of an active training run.

## 2. Configuration presets

ARBITER provides two fine-tuning presets in [`finetuning.configs`](configs.py):
1. **`typed`**: Mono-task specialist on Typed Decisions (5,410 train, 590 val, 3 epochs, reaches **76.45%** on the frozen benchmark to rival Laya Typed).
2. **`mixture_80k`**: Balanced 80,000 decisions mixture boosted with typed workflows (32k typed, 24k policy, 24k NLI in 15 languages, 0 historical replay).

### Build the 80k mixture dataset (`mixture_80k`)

```bash
python -m finetuning.build_80k \
  --model checkpoints/arbiter-272m \
  --heldout benchmark/data/cases.jsonl benchmark/data/public_requested_v1/cases.jsonl \
  --output results/adaptation/80k_typed \
  --max-length 1024
```

This downloads pinned public **training** sources and creates:

```text
results/adaptation/80k_typed/
├── train.jsonl         # 80,000 decisions
├── validation.jsonl    # 8,000 decisions
└── manifest.json      # Sources, counts, exclusions and file hashes
```

| Training data | Decisions |
|---|---:|
| Typed decisions and procedural workflows | 32,000 |
| Procedural policy decisions: choice and boolean | 24,000 |
| NLI in 15 languages | 24,000 |
| Historical benchmark replay | 0 (removed) |
| **Total** | **80,000** |

The builder runs on CPU. It does not train the model or call a paid API. It reuses
matching completed outputs; it never duplicates rows to fill a missing quota.

## 3. Check the data before training

```bash
python -m finetuning.train \
  --model checkpoints/arbiter-272m \
  --train results/adaptation/80k_typed/train.jsonl \
  --validation results/adaptation/80k_typed/validation.jsonl \
  --heldout benchmark/data/cases.jsonl benchmark/data/public_requested_v1/cases.jsonl \
  --output results/arbiter-80k-preflight \
  --max-length 1024 \
  --validate-only
```

This audits and tokenizes all examples on CPU. **No training happens.** Wait for the
command to finish successfully before continuing. If you repeat this check, choose a
new preflight output name.

## 4. Start training

### Option A: Train the 80k Mixture (`mixture_80k`)

```bash
python -m finetuning.train \
  --model checkpoints/arbiter-272m \
  --train results/adaptation/80k_typed/train.jsonl \
  --validation results/adaptation/80k_typed/validation.jsonl \
  --heldout benchmark/data/cases.jsonl benchmark/data/public_requested_v1/cases.jsonl \
  --output results/arbiter-targeted-80k-v2 \
  --device cuda \
  --epochs 1 --batch-size 4 --accumulation 8 \
  --max-length 1024 \
  --backbone-lr 1e-5 --head-lr 4e-5 \
  --clip 3 --eval-every 250 --seed 13 \
  --deterministic
```

### Option B: Train the Typed Specialist (`typed`)

```bash
python -m finetuning.train \
  --model checkpoints/arbiter-272m \
  --train results/adaptation/typed/train.jsonl \
  --validation results/adaptation/typed/validation.jsonl \
  --heldout benchmark/data/cases.jsonl benchmark/data/public_requested_v1/cases.jsonl \
  --output results/arbiter-targeted-typed-v1 \
  --device cuda \
  --epochs 3 --batch-size 4 --accumulation 8 \
  --max-length 1024 \
  --backbone-lr 1e-5 --head-lr 4e-5 \
  --clip 3 --eval-every 50 --seed 13 \
  --deterministic
```

| Setting | What it means |
|---|---|
| `--epochs 1` | One pass through the 80,000 training decisions |
| `--batch-size 4 --accumulation 8` | Effective batch size of 32; **2,500 optimizer steps** |
| `--max-length 1024` | Total token budget, including instruction and options |
| `--clip 3` | Clip the global gradient norm at 3 |
| `--eval-every 250` | Validate and save every 250 updates, plus the end of the epoch |
| `--deterministic` | Enable deterministic kernels for repeatability in the same environment |

Training also evaluates the base at step zero. If adaptation does not improve the
validation selection metric, the base may remain the best model. No fixed runtime or
benchmark improvement is promised.

**Keep the terminal running.** Stop with `Ctrl+C` if needed. Unsaved steps will be
recomputed when you resume. For a planned stop after a completed save, add
`--stop-after-step 250`; remove this option when resuming.

### Watch TensorBoard

In a **second terminal**, using the same Python environment:

```bash
tensorboard --logdir results/arbiter-targeted-80k-v2/tensorboard --host 0.0.0.0 --port 6006
```

Open `http://localhost:6006` locally, or `http://YOUR_SERVER_IP:6006` remotely.
Look at validation by family, training CE, ordinal MAE, gradient norm and clipping scale.

## 5. Resume an interrupted run

Use the **same command and output directory**, adding `--resume`:

```bash
python -m finetuning.train \
  --model checkpoints/arbiter-272m \
  --train results/adaptation/80k_typed/train.jsonl \
  --validation results/adaptation/80k_typed/validation.jsonl \
  --heldout benchmark/data/cases.jsonl benchmark/data/public_requested_v1/cases.jsonl \
  --output results/arbiter-targeted-80k-v2 \
  --device cuda \
  --epochs 1 --batch-size 4 --accumulation 8 \
  --max-length 1024 \
  --backbone-lr 1e-5 --head-lr 4e-5 \
  --clip 3 --eval-every 250 --seed 13 \
  --deterministic --resume
```

Do not change the dataset, base package, code or training settings between runs.
Resume restores the latest saved weights, optimizer, scheduler, RNG state and position.
Use a **new output directory** for a different experiment.

## 6. Find and use your model

```text
results/arbiter-targeted-80k-v2/   (or results/arbiter-targeted-typed-v1/)
├── best/               # Best validation-selected inference package
├── latest.pt           # Latest resumable training checkpoint
├── tensorboard/        # Learning curves
├── validation.jsonl    # Validation history
├── runtime.jsonl       # Runtime and peak CUDA memory
└── manifest.json       # Run configuration and hashes
```

`latest.pt` is replaced at each save; `best/` is replaced when validation improves.
The trainer does not keep a separate checkpoint for every validation.

```python
from arbiter import ArbiterModel

# Load the adapted model (mixture or typed specialist)
model = ArbiterModel.from_pretrained(
    "results/arbiter-targeted-80k-v2/best",  # or "results/arbiter-targeted-typed-v1/best"
    device="cuda",
)
result = model.decide(
    context="The customer was charged twice for one order.",
    instruction="Which team should handle this issue?",
    criteria={"payments": "Incorrect or duplicate payments", "access": "Account access problems"},
    question_type="choice",
)
print(result)
```

## 7. Evaluate the adapted model

Run both frozen benchmark families:

```bash
python -m benchmark.run --model arbiter \
  --checkpoint results/arbiter-targeted-80k-v2/best \
  --data benchmark/data/cases.jsonl \
  --output results/arbiter-80k-evaluation/historical --device cuda

python -m benchmark.run --model arbiter \
  --checkpoint results/arbiter-targeted-80k-v2/best \
  --data benchmark/data/public_requested_v1/cases.jsonl \
  --output results/arbiter-80k-evaluation/public --device cuda
```

For a before/after comparison, run the same commands with
`--checkpoint checkpoints/arbiter-272m` and different output directories.
A positive `--limit` is for smoke tests only, not publishable benchmark scores.

## Common problems

| Problem | Fix |
|---|---|
| CUDA is unavailable | Activate your CUDA-enabled PyTorch environment, or use CPU for a small technical test |
| Out of GPU memory | For a new run, try batch size 2 and accumulation 16 to keep the effective batch at 32 |
| Output directory already exists | Resume an existing run, or choose a new output name |
| Dataset manifest mismatch | Restore the original configuration/files or build into a new directory |
| Input exceeds the token budget | Check that builder and trainer use the same maximum length |
| Train/test overlap | Fix the new training split; do not delete evaluation examples |

Prefer a notebook? Open [the guided notebook](../finetuning/finetunning_arbiter.ipynb).
For what has actually been tested, read the [validation report](../docs/finetuning_validation.md).

## Interruptions and exports

An interruption resumes from the last completed `latest.pt`, not the interrupted
microbatch. Saves occur after validation at the configured cadence and at epoch end.
Use `--stop-after-step` for a planned stop at a saved optimizer boundary.
The trainer hashes code, base files, paths and settings: keep them unchanged for resume.
Load or copy `best/` after training/export has completed; its files are not promoted
as one atomic directory transaction.
