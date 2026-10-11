---
license: gemma
base_model: spectral-ai-lab/arbiter-272m
base_model_relation: finetune
tags:
- arbiter
- decision-making
- classification
- ordinal-regression
- safetensors
datasets:
- LocalLLaMA/typed-decisions
---
# ARBITER-272M-Typed

**A lightweight Typed Decisions specialist: 76.45% accuracy, up from 65.50% for ARBITER Base, with broadly retained routing and classification performance.**

This model fine-tunes [ARBITER-272M](https://huggingface.co/spectral-ai-lab/arbiter-272m) on Typed Decisions. It supports categorical `choice`, boolean `noul` and ordinal `score` requests using the same ARBITER API and architecture. It jointly encodes context, instruction and options; it is not a generative chat model.

## Benchmark results

Identical frozen requests, prompts and adapters were used for the compared models. Typed includes 2,000 decisions grouped into 400 cases. Results are from our local benchmark protocol, not a universal leaderboard.

| Typed Decisions | Accuracy |
|---|---:|
| Laya Typed | **76.75%** |
| **ARBITER-272M-Typed** | **76.45%** |
| ARBITER Base | 65.50% |
| GLiNER2.5 Decide | 50.35% |
| GLiNER2.5 Multi | 45.95% |
| EmbeddingGemma baseline | 40.85% |
| Laya | 36.20% |
| Laya Multilingual | 35.25% |

The specialist gains **10.95 percentage points** over its base and remains **0.30 point (six correct decisions) below Laya Typed**. This is a single-seed observation, not statistical equivalence or superiority.

| Typed output | Decisions | Accuracy |
|---|---:|---:|
| Choice | 600 | 75.17% |
| Boolean | 600 | 80.50% |
| Ordinal score | 800 | 74.38% |

## Retention after specialization

| Benchmark / metric | ARBITER Base | ARBITER Typed |
|---|---:|---:|
| MASSIVE intent, 14 languages | 91.33% | 91.62% |
| MASSIVE scenario, 14 languages | 89.40% | 89.52% |
| XNLI, 15 languages | 68.87% | 68.47% |
| SST-2 | 86.93% | 86.47% |
| TREC | 94.60% | 94.20% |
| Banking77 | 87.34% | 87.11% |
| Fast Single micro | 43.35% | 43.65% |
| Fast Single macro | 43.97% | 44.13% |
| Fast Multi exact match | 4.67% | 5.00% |
| Fast Multi micro F1 | 51.37% | 51.34% |

These measurements show no broad collapse on the evaluated tasks, but do include small regressions. They do not guarantee absence of forgetting on other distributions. MASSIVE uses 20 candidates (target plus 19 distractors); Banking77 uses its full candidate inventory. Published benchmark splits and subsets are documented in the repository. Some tasks were represented in the base's supervised training; these are not all unseen-task evaluations.

A subsequent check on 2,500 retention development decisions passed the proposed observed-score gates (global classification drop <=0.5 pp, per-family <=1 pp, ordinal MAE increase <=0.05). This was a **post-training check**, not the original checkpoint selection rule or a statistical non-inferiority proof. The original checkpoint was selected by validation loss.

## Training and reproduction

- Initialization: ARBITER Base; unchanged architecture and losses, no replay in this specialist run.
- Data: 5,410 Typed training decisions and 590 grouped validation decisions; source revision and data hashes are recorded in `reproduction.json`.
- Three epochs; micro-batch 4, accumulation 8; maximum sequence length 1,024.
- Backbone LR 1e-5; head LR 4e-5; gradient clipping 3; ordinal coefficient 0.5.
- Seed 13, deterministic mode; validation every 50 steps, best macro validation CE.
- Selected step: 510. Recorded runtime: 455.37 seconds on RTX 5090 for this run.
- Inference weights: BF16 safetensors; training used FP32 master weights with BF16 autocast.

See the GitHub repository's `finetuning/TYPED_SPECIALIST.md` for dataset preparation, training, TensorBoard and evaluation commands. `reproduction.json` records the original environment, configuration and hashes. Re-running on another environment need not reproduce identical floating-point outputs or exactly 76.45%.

## Usage

Install the ARBITER package from https://github.com/seif-attoui/arbiter, then download this repository to a local directory. The package provides the custom loader; this release is not a standard Transformers text-classification pipeline.

```python
from huggingface_hub import snapshot_download
from arbiter import ArbiterModel

path = snapshot_download("spectral-ai-lab/arbiter-272m-typed")
model = ArbiterModel.from_pretrained(path, device="cpu", local_files_only=True)
result = model.decide(
    context="The customer was charged twice for the same order.",
    instruction="Which team should handle this issue?",
    criteria={"payments": "Incorrect or duplicate payments", "access": "Account access problems"},
    question_type="choice",
)
print(result)
```

Expected semantic answer for this illustration: `payments`; it is not an additional benchmark result. Native batching uses the same API as ARBITER Base.

## Scope and licensing

Use the base for the general-purpose reference and this variant when Typed Decisions performance matters. Candidate-order robustness is not absolute permutation invariance. The benchmark has informed development; this is not a fresh blind evaluation. Exact/group audits do not rule out every semantic overlap or upstream pretraining exposure.

Weights remain subject to upstream Gemma terms; the inference code's Apache-2.0 license does not replace those terms. Dataset provenance and licenses must be considered separately. Authors: Seif Attoui and Ichrak Mokhtari, Spectral AI Lab.
