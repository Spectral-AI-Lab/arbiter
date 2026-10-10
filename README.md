# ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Weights License](https://img.shields.io/badge/Weights_License-Gemma_Terms_of_Use-green.svg)](https://ai.google.dev/gemma/terms)
[![Model Size](https://img.shields.io/badge/Model_Size-272M_Parameters-orange.svg)](#)
[![Format](https://img.shields.io/badge/Format-Safetensors_BF16_(544_MB)-purple.svg)](#)

**ARBITER** is an industrial-grade, ultra-compact foundation encoder designed for **decision-making, routing, and relational structured reasoning** in autonomous AI agents and enterprise production systems.

Built upon **Architecture G** (Inter-Candidate Set-Attention & Dynamic HyperFiLM Modulation), ARBITER jointly encodes context, instructions, and candidate options, allowing competing choices to attend to one another rather than being evaluated in isolation.

---

## Key Features

* **Set-Attention Over Candidates (Block E)**: Candidates attend to each other via a multi-head self-attention mechanism before decision projection, enabling relational ranking and order robustness.
* **Dynamic Context & Task HyperFiLM (Block F)**: Affine modulation conditioned on context embeddings and task type embeddings (`choice`, `noul`, `score`).
* **Multi-Format Native Inference**:
  * **`choice`**: Polytomous classification and categorical routing.
  * **`noul`**: Strict boolean factuality and claim verification.
  * **`score`**: Ordinal regression grading and severity scoring.
* **Order & Permutation Robustness**: 82.8% decision consistency across candidate permutations, reducing position-0 bias to 22.6% (compared to 56.6% in standard pointwise classifiers).
* **High Efficiency & Compactness**: Native **BF16 export of 544 MB** (`model.safetensors`), allocated VRAM of **1,294 MiB**, median GPU latency of **15.36 ms** (RTX 5090, batch 1).
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
