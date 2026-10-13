---
base_model: google/embeddinggemma-2
library_name: transformers
pipeline_tag: text-classification
tags:
- custom-code
- decision-making
---
# EmbeddingGemma 2 — Decision Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spectral-ai-lab/arbiter/blob/main/notebooks/getting_started.ipynb)

Try ARBITER in your browser with installation, ready-to-run examples, and expected answers. The notebook loads ARBITER Base by default and also lets you try Typed.

Local Hugging Face pipeline utilizing official Google base weights, without quantization. Uses CPU and `dtype="auto"` by default. The checkpoint revision is anchored in `config.json`. No external hub repo published.

```bash
python -m pip install -r requirements.txt
```

From this directory:

```python
from transformers import pipeline

pipe = pipeline(
    "decision-making", model=".",
    trust_remote_code=True, device="cpu", dtype="auto",
)
result = pipe({
    "state": "My payment was debited twice. I would like a refund.",
    "questions": {
        "service": {
            "type": "choice", "instructions": "Which department?",
            "criteria": {"billing": "Payments and invoices", "shipping": "Parcels and delivery"},
        },
        "urgency": {
            "type": "score", "instructions": "What urgency level?",
            "criteria": ["Low", "Medium", "High"],
        },
        "refund": {
            "type": "noul", "instructions": "The customer is requesting a refund.",
        },
    },
})
print(result)
```

From another directory, replace `model="."` with the absolute path to this project. Weights are downloaded from `google/embeddinggemma-2` or read from the local HF cache; this directory contains only code and configuration. Install dependencies before loading. Checkpoint access and Google terms remain applicable.

## Files

- `config.json`: Google revision, decision settings, and custom pipeline declaration.
- `modeling_decision.py`: Model loader and `AutoModel` integration.
- `decision_pipeline.py`: `transformers.pipeline` interface.
- `engine.py`: Embeddings, mean centering, cosine similarities, and LogSumExp window aggregation.
- `compat.py`: Request and response parsing matching the common Jev/Laya contract.
- `requirements.txt`: Dependencies.

## Scope & Capabilities

`decision-making` is a custom task, not a native Transformers task. Processes one request dictionary per call. Supports `choice`, `score`, and `noul` question types without HTTP server overhead or Laya router extensions.

Preprocessing inspired by MediaPipe: classification prefixes, candidate mean-centering, L2 normalization, cosine similarity, and LogSumExp window aggregation. This is not an exact replica of the native MediaPipe runtime. No fine-tuning applied; probabilities are uncalibrated. Confidence follows the Jev formula, distinct from Laya's. `usage.input_tokens` counts logical inputs, including cached candidate descriptions.
