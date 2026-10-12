# ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Weights License](https://img.shields.io/badge/Weights_License-Gemma_Terms_of_Use-green.svg)](https://ai.google.dev/gemma/terms)
[![Model Size](https://img.shields.io/badge/Model_Size-272M_Parameters-orange.svg)](#)
[![Format](https://img.shields.io/badge/Format-Safetensors_BF16_(544_MB)-purple.svg)](#)

**ARBITER** is a compact encoder by **Spectral AI Lab** (*Where intelligence emerges*), designed for **decision-making, routing, and relational structured reasoning** in autonomous AI agents and enterprise production systems.

Built upon the **ARBITER** relational architecture (Inter-Candidate Set-Attention & Dynamic HyperFiLM Modulation), ARBITER jointly encodes context, instructions, and candidate options, allowing competing choices to attend to one another rather than being evaluated in isolation.

---

## Key Features

* **Set-Attention Over Candidates**: Candidates attend to each other via a multi-head self-attention mechanism before decision projection, enabling relational ranking and order robustness.
* **Dynamic Context & Task HyperFiLM**: Affine modulation conditioned on context embeddings and task type embeddings (`choice`, `noul`, `score`).
* **Multi-Format Native Inference**:
  * **`choice`**: Polytomous classification and categorical routing.
  * **`noul`**: Strict boolean factuality and claim verification.
  * **`score`**: Ordinal regression grading and severity scoring.
* **Order & Permutation Robustness**: 82.8% decision consistency across candidate permutations, reducing position-0 bias to 22.6% (compared to 56.6% in standard pointwise classifiers).
* **High Efficiency & Compactness**: Native **BF16 export of 544 MB** (`model.safetensors`), peak allocated VRAM of **759 MiB** (1,562 MiB reserved), median GPU latency of **15.36 ms** (RTX 5090).
* **Autonomous Loading**: Self-contained weights and tokenizer; no external Hugging Face downloads required at runtime.

---

## Installation

From PyPI:

```bash
pip install spectral-arbiter
```

Or from source for development:

```bash
git clone https://github.com/spectral-ai-lab/arbiter.git
cd arbiter
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Requires Python >=3.11, PyTorch >=2.0, Transformers >=5.19 and safetensors >=0.4.
The tested environment is recorded in [development instructions](CONTRIBUTING.md).
Install a CUDA-enabled PyTorch build to use a GPU; the first example also runs on CPU.

### Obtain the weights

Git contains the code, configuration and tokenizer, **not the model weights**.
Place the complete release package in `checkpoints/arbiter-272m/` before running
local examples. It must contain `model.safetensors`, `config.json`, `tokenizer.json`
and `tokenizer_config.json` from the same release.

Once you have access to the HF release, download it using its exact commit:

```bash
hf download spectral-ai-lab/arbiter-272m --revision <release-commit> --local-dir checkpoints/arbiter-272m
```

Replace `<release-commit>` with the published release SHA. This repository does not
yet record an accessible public release revision; a private repository requires
`hf auth login` and access. A 404 can mean unavailable or unauthorized weights.
After downloading the complete package, local inference requires no network access.
The library wheel contains inference only; fine-tuning and benchmark commands below
run from this Git checkout.

---

## Quickstart

### 1. High-Level Decision (`model.decide`)

```python
from arbiter import ArbiterModel

# Load the official 272M model in native BF16 from the local checkpoint
model = ArbiterModel.from_pretrained("checkpoints/arbiter-272m", device="cpu", local_files_only=True)

# Route a customer support ticket to the appropriate service
result = model.decide(
    context="My package was marked as delivered yesterday, but I never received it. The courier is unresponsive.",
    instruction="Which customer support team must handle this claim?",
    criteria={
        "delivery": "Lost parcel investigation and courier disputes",
        "billing": "Payment errors, refunds, and invoice questions",
        "technical": "Mobile application login or website bugs"
    },
    question_type="choice"
)

print("Selected Department :", result["selected_key"])
print("Confidence          :", f"{result['confidence']:.2%}")
print("Class Probabilities :", result["probabilities"])
```

### 2. Multi-Question Dictionary Interface (`model.predict`)

ARBITER natively integrates into agent loops and multi-step evaluation pipelines:

```python
results = model.predict(
    state="Customer has been waiting 14 days for a replacement item under warranty.",
    questions={
        "intent": {
            "instructions": "Identify the primary customer intent",
            "type": "choice",
            "criteria": {
                "warranty_claim": "Warranty replacement request",
                "general_inquiry": "General question"
            }
        },
        "dissatisfied": {
            "instructions": "Is the customer dissatisfied with the resolution time?",
            "type": "noul"
        },
        "urgency": {
            "instructions": "Rate urgency level on our incident scale",
            "type": "score",
            "criteria": ["P4 Low", "P3 Medium", "P2 High", "P1 Critical Outage"]
        }
    }
)

print(results["answers"])
```

---

## Workflows

- [Native batch inference](docs/batching.md): `decide_batch()` and `predict_batch()`.
- [Fine-tuning commands](finetuning/README.md): build data, validate, train, resume and evaluate.
- [Guided notebook](notebooks/finetuning_arbiter.ipynb): the same workflow, cell by cell.
- [Benchmark reproduction](benchmark/README.md): ARBITER and baseline comparisons.
- [Development and tests](CONTRIBUTING.md).

## Repository Structure

```text
arbiter/       # Installable inference library
finetuning/    # Dataset builder, trainer, verifier and guided notebook
benchmark/     # Frozen requests, metrics and comparator adapters
hf_pipeline/   # Untrained EmbeddingGemma baseline, not the ARBITER loader
checkpoints/   # Local downloaded model packages (Git-ignored)
tests/         # Unit and real-weight integration tests
docs/          # Batching, data recipe and validation reports
examples/      # Runnable inference measurements
```

## Benchmark Results

ARBITER achieves the highest micro accuracy on both evaluated protocol aggregates,
with particularly strong intent routing and probability quality. Tables use a fixed
set of model variants throughout; **bold marks the best reported result in each row**
(including ties). Accuracy is in percent; ↓ means lower is better. These are local
protocol comparisons, not a claim of leadership on every task.

### Intent Routing & Classification

| Benchmark / metric | ARBITER | Laya | Laya Multilingual | Laya Typed | GLiNER Decide | GLiNER Multi | EmbeddingGemma |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MASSIVE Intent · 14 languages | **91.33%** | 33.95% | 46.69% | 34.43% | 42.29% | 49.52% | 73.79% |
| MASSIVE Scenario · 14 languages | **89.40%** | 30.10% | 44.74% | 28.43% | 36.98% | 45.26% | 58.07% |
| Banking77 · 77 intents | **87.34%** | 37.44% | 35.94% | 37.31% | 65.29% | 58.90% | 85.55% |
| TREC · question classification | **94.60%** | 88.20% | 90.40% | 85.40% | 32.60% | 36.20% | 47.40% |
| SST-2 · sentiment | 86.93% | **90.02%** | 84.98% | 89.79% | 87.96% | 78.78% | 83.26% |

### Typed Decisions & Natural Language Inference

| Benchmark / metric | ARBITER | Laya | Laya Multilingual | Laya Typed | GLiNER Decide | GLiNER Multi | EmbeddingGemma |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Typed Decisions | 65.50% | 36.20% | 35.25% | **76.75%** | 50.35% | 45.95% | 40.85% |
| XNLI · 15 languages | 68.87% | 54.38% | **73.82%** | 55.64% | 38.24% | 38.38% | 33.56% |
| Fast Decisions Single · micro | 43.35% | 52.96% | 46.77% | 54.92% | **64.42%** | 59.38% | 41.73% |
| Fast Decisions Single · macro | 43.97% | 53.80% | 46.66% | 55.66% | **66.30%** | 60.63% | 42.60% |
| Fast Decisions Multi · exact match | **4.67% (14/300)** | 2.33% (7/300) | 1.00% (3/300) | 3.67% (11/300) | 1.33% (4/300) | 4.00% (12/300) | 0.33% (1/300) |
| Fast Decisions Multi · micro F1 | 51.37% | 51.72% | 45.50% | **56.41%** | 51.74% | 48.41% | 31.22% |

### Overall Protocol Results

| Benchmark / metric | ARBITER | Laya | Laya Multilingual | Laya Typed | GLiNER Decide | GLiNER Multi | EmbeddingGemma |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical · micro accuracy | **74.47%** | 48.30% | 56.86% | 52.66% | 46.58% | 47.84% | 51.25% |
| Historical · suite macro accuracy | **77.03%** | 47.18% | 57.89% | 48.33% | 44.61% | 47.33% | 52.45% |
| Public · micro accuracy | **69.68%** | 53.92% | 51.18% | 52.25% | 62.86% | 57.85% | 59.87% |

Historical: 21,915 decisions across 60 suites. Public: 9,252 decisions across 23 suites.
Micro averages weight decisions equally; suite macro averages weight suites equally.
Language variants count as separate suites. These aggregates reflect this task mixture.
Column names are abbreviated: GLiNER refers to GLiNER2.5, Laya to ConvAI base,
and EmbeddingGemma to the EmbeddingGemma-2 baseline pipeline.

### Probability Quality

Measured on the same 21,915 historical decisions. GLiNER's reconstructed adapter
probabilities are reported separately below and do not enter this comparison.

| Benchmark / metric | ARBITER | Laya | Laya Multilingual | Laya Typed | GLiNER Decide | GLiNER Multi | EmbeddingGemma |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Brier score ↓ | **0.3395** | 0.7974 | 0.6788 | 0.6527 | — | — | 0.7106 |
| NLL ↓ | **0.6761** | 4.3655 | 2.3747 | 1.8325 | — | — | 1.5450 |
| Suite-weighted ECE · 15 bins ↓ | **0.0674** | 0.3214 | 0.2769 | 0.2166 | — | — | 0.2947 |

Brier is the mean squared error of probability vectors against one-hot targets;
NLL is mean negative log probability of the target. ECE is computed per suite with
15 bins, then weighted by suite size (it differs from pooled ECE). ARBITER has
**48% lower Brier and 63% lower NLL than Laya Typed** in this protocol.
ARBITER and Laya use their native decision probabilities; EmbeddingGemma uses
cosine-similarity softmax. Application thresholds require separate validation.

<details>
<summary>GLiNER adapter probability metrics</summary>

| Metric | GLiNER Decide | GLiNER Multi |
| :--- | ---: | ---: |
| Brier ↓ | 0.6611 | 0.6664 |
| NLL ↓ | 1.5762 | 1.5555 |
| Suite-weighted ECE ↓ | 0.1098 | 0.1379 |

These distributions are reconstructed by the benchmark adapter, not native
full-candidate GLiNER distributions. A dash above means excluded from the comparable
probability table, not that the model cannot produce confidence scores.

</details>

### Inference Efficiency

**Accuracy–memory trade-off:** ARBITER combines **74.47% historical micro accuracy**
with **759 MiB peak allocated GPU memory**, the lowest measured footprint among
these pipeline configurations. Its **15.36 ms median latency is 28% lower than the
EmbeddingGemma baseline** (21.36 ms). Compared with Laya Multilingual, it delivers
**+17.61 accuracy points with 53% less allocated GPU memory**, at higher median
latency (15.36 vs 9.44 ms). These comparisons apply to the evaluated task mixture
and loading formats below.

| Benchmark / metric | ARBITER | Laya | Laya Multilingual | Laya Typed | GLiNER Decide | GLiNER Multi | EmbeddingGemma |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Median latency · ms ↓ | 15.36 ms | 12.07 ms | 9.44 ms | 11.91 ms | 15.32 ms | **7.97 ms** | 21.36 ms |
| Peak allocated VRAM · MiB ↓ | **759.0 MiB** | 2,415.8 MiB | 1,600.4 MiB | 2,438.6 MiB | 3,237.4 MiB | 3,823.4 MiB | 862.6 MiB |

Historical protocol, NVIDIA GeForce RTX 5090, single-request inference. ARBITER
and EmbeddingGemma use BF16; Laya variants use FP32; GLiNER uses its default loading
configuration. These are measured pipeline configurations, not equal-dtype kernel
comparisons. Latency includes each adapter's execution path. These measurements
predate the native batch API and do not measure batched throughput.

### Protocol Notes

- **MASSIVE Intent:** 20 candidates (correct intent plus 19 distractors), 4,200 test
  decisions across 14 languages; this is not full 60-intent classification.
  Scenario uses 18 candidates and 4,200 decisions across the same languages.
- **Banking77:** 3,080 test examples with all 77 intents. **TREC:** 500 test examples,
  six coarse classes. **SST-2:** 872 validation examples.
- **XNLI:** 4,500 decisions across 15 languages. **Typed Decisions:** 2,000 decisions.
  **Fast Single:** 2,600 decisions. **Fast Multi:** 300 multi-question cases;
  exact match requires every decision in a case to be correct, whereas micro F1
  measures the pooled binary decisions. Public micro accuracy counts those binary
  decisions individually, not the 300 exact-match outcomes.
- **Supervision:** Banking77, TREC, SST-2 and MASSIVE training splits were included
  in ARBITER's adaptation mixture. These results demonstrate supervised capability,
  not zero-shot evaluation on unseen task families.
- **Adapters:** comparisons use the repository's model-specific adapters. GLiNER's
  adapter classifies the state against labels without the separate instruction;
  results characterize these evaluated pipelines. Failed cases remain in denominators.
- **Statistics:** bold denotes the highest observed score, not statistical significance.
  Exploratory paired McNemar tests: TREC vs Laya Multilingual p=0.0019;
  Banking77 vs EmbeddingGemma p=0.0068. These assume independent examples and are
  unadjusted for multiple comparisons.

<details>
<summary>Additional reasoning and long-context evaluations</summary>

These auxiliary evaluations have a narrower comparator set than the main tables.
Tasksource Long is an existing project validation set, not a certified unseen-domain test.

| Evaluation Benchmark | Domain | Items | ARBITER-272M | Baseline Comparison | Note |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tasksource Long** | Long context (up to 4,096 tokens) | 2,373 | **58.45%** | Laya-Typed 32.91% | Observed advantage on long-context criteria matching (+25.5 pts). |
| **ARC-Easy** | Elementary science reasoning | 2,376 | **48.48%** | Laya-Typed 48.48% | Tied with Laya-Typed (48.48%), outperforms Laya-Base (47.26%). |
| **ARC-Challenge** | Hard science reasoning | 1,172 | **33.79%** | Laya-Typed 31.57% | McNemar test p = 0.2180 (observed gain; not statistically significant). |
| **HellaSwag** | Common-sense inference | 1,000 | **41.80%** | Laya-Base 35.90% | Outperforms Laya-Base (+5.9 pts) and GLiNER (29.80%, +12.0 pts). |

</details>

---

## Reproducing Empirical Benchmarks

The frozen evaluation harness and evaluation request suites are packaged within [`benchmark/`](benchmark/):

```bash
# Install benchmark dependencies
pip install -e .[benchmark]

# Run the 23 public requested suites (9,252 decisions: SST-2, TREC, Banking77, Fast Decisions)
python benchmark/run.py --model arbiter-272m \
  --data benchmark/data/public_requested_v1/cases.jsonl \
  --device cuda:0 --output results/public_benchmark

# Run the 60 historical suites (21,915 decisions: MASSIVE, Typed Decisions, etc.)
python benchmark/run.py --model arbiter-272m \
  --data benchmark/data/cases.jsonl \
  --device cuda:0 --output results/historical_benchmark
```

---

## Testing & Numerical Parity

```bash
python -m pip install -e '.[dev,finetuning,benchmark]'
# No weights or GPU required:
CUDA_VISIBLE_DEVICES='' python -m pytest tests benchmark/tests -m 'not integration' -q
# After obtaining the complete release package:
CUDA_VISIBLE_DEVICES='' python -m pytest tests benchmark/tests -q
```

The optional `tests/test_arbiter_parity.py` script compares an internal reference
checkpoint with the export. It additionally requires the original training repository
and reference weights; it is not the public installation check. See
[development instructions](CONTRIBUTING.md) for release validation.

---

## Licensing & Provenance

* **Source Code (`arbiter/`)**: Released under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.
* **Model Weights (`checkpoints/arbiter-272m/model.safetensors`)**: Derived from `google/embeddinggemma-2` and fine-tuned on relational decision data. Weights are subject to the **Gemma Terms of Use**.

---

## Authors & Contributors

* **Ichrak Mokhtari** (Equal contribution)
* **Seif Attoui** (Equal contribution)

---

## Citation

```bibtex
@misc{arbiter2026,
  title={ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning},
  author={Mokhtari, Ichrak and Attoui, Seif},
  year={2026},
  note={Equal contribution},
  howpublished={\url{https://huggingface.co/spectral-ai-lab/arbiter-272m}},
}
```

### Native batching

`decide_batch()` and `predict_batch()` support length-based batching with token and attention budgets. `predict()` batches its questions automatically. See [batching, examples and deployment limits](docs/batching.md).

## Targeted fine-tuning

Start with the [command-line fine-tuning guide](finetuning/README.md) or the [guided notebook](notebooks/finetuning_arbiter.ipynb). Both support building an 80k training / 8k validation mixture from pinned public sources, data checks, training, TensorBoard and resume. Data is generated locally rather than bundled in Git. See the [dataset recipe](docs/build_80k.md) for scope and limitations; no benchmark improvement is claimed before evaluation.

### Adapt on your own data in 5 lines

You can easily adapt ARBITER to your custom enterprise data (support tickets, intent routing, policy triage) by converting any CSV or DataFrame into the ARBITER JSONL format:

```python
import pandas as pd, json

# 1. Your tabular data
df = pd.DataFrame([
    {"context": "Cannot reset my 2FA password.", "label": "auth"},
    {"context": "Charged twice for subscription.", "label": "billing"}
])
criteria = {"auth": "Login and 2FA issues", "billing": "Invoice and charge errors"}

# 2. Format into ARBITER decision contract
rows = [{
    "id": f"row_{i}", "group_id": f"grp_{i}", "family": "custom/support",
    "source": "internal", "source_revision": "v1", "source_split": "train",
    "license": "proprietary", "context": r["context"],
    "instruction": "Classify the ticket category",
    "question_type": "choice", "criteria": criteria,
    "target": {c: 1.0 if c == r["label"] else 0.0 for c in criteria}
} for i, r in df.iterrows()]

# 3. Save to JSONL
with open("train.jsonl", "w") as f:
    for row in rows: f.write(json.dumps(row) + "\n")
```

Then train with:

```bash
python -m finetuning.train \
  --model checkpoints/arbiter-272m \
  --train train.jsonl --validation val.jsonl \
  --heldout benchmark/data/cases.jsonl \
  --output results/my-custom-model \
  --epochs 3 --batch-size 4 --accumulation 8
```

## ARBITER Typed specialist

The Typed specialist reaches 76.45% on Typed Decisions versus 65.50% for the base. See the [reproduction instructions](finetuning/TYPED_SPECIALIST.md) and [specialist model card](docs/model_card_typed.md).
