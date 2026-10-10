# ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Weights License](https://img.shields.io/badge/Weights_License-Gemma_Terms_of_Use-green.svg)](https://ai.google.dev/gemma/terms)
[![Model Size](https://img.shields.io/badge/Model_Size-272M_Parameters-orange.svg)](#)
[![Format](https://img.shields.io/badge/Format-Safetensors_BF16_(544_MB)-purple.svg)](#)

**ARBITER** is an industrial-grade, ultra-compact foundation encoder designed for **decision-making, routing, and relational structured reasoning** in autonomous AI agents and enterprise production systems.

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

```bash
# Clone the repository
git clone https://github.com/seif-attoui/arbiter.git
cd arbiter

# Install in editable mode
pip install -e .
```

### Dependencies
- Python >= 3.10
- PyTorch >= 2.0.0
- Transformers >= 4.45.0
- Safetensors >= 0.4.0

---

## Quickstart

### 1. High-Level Decision (`model.decide`)

```python
from arbiter import ArbiterModel

# Load the official 272M model in native BF16 from the local checkpoint
model = ArbiterModel.from_pretrained("checkpoints/arbiter-272m", device="cuda:0")

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

## Repository Structure

```
arbiter/
├── arbiter/                         # Core Python package
│   ├── __init__.py                  # Package exports (ArbiterModel, ArbiterConfig)
│   ├── configuration_arbiter.py     # Configuration class
│   └── modeling_arbiter.py          # PyTorch model architecture & inference logic
├── checkpoints/
│   └── arbiter-272m/                # Release model package
│       ├── model.safetensors        # 544 MB BF16 weights (429 tensors)
│       ├── config.json              # Model configuration
│       ├── tokenizer.json           # Fast tokenizer with <OPTION> special token
│       ├── tokenizer_config.json    # Tokenizer settings
│       └── README.md                # Official Model Card
├── tests/
│   └── test_arbiter_parity.py       # Strict numerical parity and contract validation
├── pyproject.toml                   # Standard PEP 517/621 packaging metadata
├── LICENSE                          # Apache-2.0 License for source code
└── README.md                        # Documentation
```

## Empirical Benchmark Results

> **ARBITER-272M is an ultra-compact 272M parameter decision model that consistently outperforms evaluated open-weights baselines (Laya variants and GLiNER) across core routing and intent classification benchmarks.**

| Benchmark | ARBITER-272M (BF16) | Laya (ConvAI) | GLiNER2.5-Decide | EmbeddingGemma-2 (Base) | Scope & Task Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **TREC (Coarse)** | **94.60%** | 88.20% | 32.60% | 47.40% | Question classification (6 coarse classes, 500 test questions) |
| **MASSIVE Intent** | **91.33%** | 33.95% | 42.29% | 73.79% | Multilingual intent routing (14 languages, 4,200 test decisions) |
| **Banking77** | **87.34%** | 37.44% | 65.29% | 85.55% | Fine-grained customer support intent (77 real classes, 3,080 test cases) |
| **MASSIVE Scenario** | **89.40%** | 30.10% | 36.98% | 58.07% | Multilingual scenario routing (14 languages, 4,200 decisions) |
| **SST-2 (GLUE)** | **86.93%** | **90.02%** | 87.96% | 83.26% | Sentiment classification (validation split, 872 sentences) |

### Summary Protocol Aggregates

| Benchmark Suite | ARBITER-272M (BF16) | Laya (Best Variant) | GLiNER (Best Variant) | EmbeddingGemma-2 (Base) | Decisions Evaluated |
| :--- | :---: | :---: | :---: | :---: | ---: |
| **Grand Benchmark (60 historical suites)** | **74.47%** | 56.86% | 47.84% | 51.25% | 21,915 decisions |
| **Public Benchmark (23 requested suites)** | **69.68%** | 53.92% | 62.86% | 59.87% | 9,252 decisions |

### Probabilistic Calibration & Uncertainty Quality (21,915 Decisions)

| Model | Historical Micro Acc | Brier Score (↓ lower is better) | NLL (↓ lower is better) | ECE (15 bins) (↓ lower is better) | Calibration Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ARBITER-272M (BF16)** | **74.47%** | **0.3395** | **0.6761** | **0.0674** | **Superior calibration** (lowest probability error & minimal overconfidence) |
| `EmbeddingGemma-2` (Base Pipeline) | 51.25% | 0.7106 | 1.5450 | 0.2947 | High entropy, uncalibrated pooled representations |
| `Laya` (ConvAI base) | 48.30% | 0.7974 | 4.3655 | 0.3214 | Heavy overconfidence penalty (high NLL) |
| `Laya-Multilingual` | 56.86% | 0.6788 | 2.3747 | 0.2769 | Moderate calibration |
| `Laya-Typed-Decisions` | 52.66% | 0.6527 | 1.8325 | 0.2166 | Moderate calibration |
| `GLiNER2.5-Decide` | 46.58% | 0.6611* | 1.5762* | 0.1098* | *Probabilities reconstructed via adapter (not native softmax) |
| `GLiNER2.5-Multi-Decide` | 47.84% | 0.6664* | 1.5555* | 0.1379* | *Probabilities reconstructed via adapter (not native softmax) |

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

To verify strict numerical parity against the original reference checkpoint and validate input contracts:

```bash
python tests/test_arbiter_parity.py
```

Expected output:
- **Decision Agreement Rate**: 100.0% (8/8)
- **Maximum Probability Delta**: `0.00e+00` (exact bit-level identity in native BF16)
- **Contract Validations**: 4/4 passed (rejection of invalid question types, empty criteria, adversarial `<OPTION>` token sanitization, multi-question dictionary API).

---

## Licensing & Provenance

* **Source Code (`arbiter/`)**: Released under the **Apache-2.0 License**. See [LICENSE](LICENSE) for details.
* **Model Weights (`checkpoints/arbiter-272m/model.safetensors`)**: Derived from `google/embeddinggemma-2` and fine-tuned on relational decision data. Weights are subject to the **Gemma Terms of Use**.

---

## Authors & Contributors

* **Seif Attoui** (Equal contribution)
* **Ichrak Mokhtari** (Equal contribution)

---

## Citation

```bibtex
@misc{arbiter2026,
  title={ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning},
  author={Attoui, Seif and Mokhtari, Ichrak},
  year={2026},
  note={Equal contribution},
  howpublished={\url{https://huggingface.co/spectral-ai-lab/arbiter-272m}},
}
```
