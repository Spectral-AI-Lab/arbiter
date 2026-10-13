# ARBITER Online Serving & vLLM Structured Decisions (/v1/systemone)

High-performance containerized inference micro-service for **ARBITER-272M** (Base and Typed models).

The service exposes two interoperable decision interfaces:
1. **`POST /v1/systemone`**: Text-only implementation of the official [vLLM Structured Decisions specification](https://docs.vllm.ai/en/latest/serving/online_serving/structured_decisions/) for non-causal System 1 routing, choice, and scoring.
2. **`POST /decide`**: ARBITER's lightweight native flat decision endpoint.

It features an asynchronous **Dynamic Micro-Batching Scheduler** that automatically groups concurrent client requests into length-bucketed batches to maximize GPU saturation while enforcing strict memory budgets and bounded queue delays.

---

## Architecture & Endpoints

```
                      ┌────────────────────────────────────────┐
                      │            FastAPI Server              │
                      │                                        │
Client Requests ─────►│  POST /v1/systemone  (vLLM Standard)   │
                      │  POST /predict       (Laya / Jev Alias)│
                      │  POST /decide        (ARBITER Flat)    │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │    Dynamic Micro-Batch Scheduler       │
                      │                                        │
                      │  • Async non-blocking queue            │
                      │  • Configurable batch window (0–8 ms)  │
                      │  • Token & attention element budgets   │
                      │  • Length-bucketing regrouping         │
                      └──────────────────┬─────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────┐
                      │          ArbiterModel Engine           │
                      │                                        │
                      │  • Joint bidirectional text encoder    │
                      │  • Relational Set-Attention head       │
                      │  • HyperFiLM conditioning              │
                      │  • Single forward pass (0 out tokens)  │
                      └────────────────────────────────────────┘
```

### Endpoints Overview

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/systemone` | Official vLLM Structured Decisions endpoint. Accepts `state` + `questions` (`choice`, `noul`, `score`). |
| `POST` | `/predict` | Legacy Jev / Laya response contract (`{"answers": {...}}`) with native boolean identifiers (`"false"`/`"true"`). Accepts `state` + `questions`. |
| `POST` | `/decide` | Native single-query ARBITER endpoint (`context`, `instruction`, `criteria`). |
| `GET` | `/health` | Liveness check, loaded model ID, and device status. |
| `GET` | `/metrics` | Real-time p50/p95 latencies, throughput, avg batch size, and queue depth. |

---

## Measured Performance & Empirical Benchmarks

Evaluated on **NVIDIA GeForce RTX 5090** (32 GB VRAM, PyTorch 2.14, BF16 weights) across 1,000 mixed client requests (`/v1/systemone` with 3 questions + `/decide`):

> **Benchmark Methodology & Scope**:
> Measurements were conducted using `benchmark/benchmark_serving.py` via in-process `httpx.ASGITransport` to isolate scheduler queuing, micro-batching, and PyTorch tensor execution from network interface latency. Throughput reflects strictly successful (HTTP 200) completed decisions. Window 0 ms evaluates immediate dispatch (with opportunistic micro-batching when concurrent queue depth > 1).

| Batch Window | Concurrency | Decisions / sec | Requests / sec | Latency p50 | Latency p95 | Avg Batch Size | Errors |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0 ms** (Immediate) | 1 | 102.0 dec/s | 58.3 req/s | 17.3 ms | 19.4 ms | 1.8 | 0 |
| **0 ms** (Immediate) | 4 | 192.3 dec/s | 109.9 req/s | 36.6 ms | 40.4 ms | 3.6 | 0 |
| **0 ms** (Immediate) | 8 | 339.6 dec/s | 194.0 req/s | 41.6 ms | 44.3 ms | 7.3 | 0 |
| **0 ms** (Immediate) | 16 | 534.0 dec/s | 305.2 req/s | 50.2 ms | 53.8 ms | 13.5 | 0 |
| **0 ms** (Immediate) | 32 | **585.8 dec/s** | **334.7 req/s** | 82.9 ms | 108.2 ms | 15.9 | 0 |
| **8 ms** (Micro-batch) | 1 | 67.6 dec/s | 38.6 req/s | 25.7 ms | 28.2 ms | 1.8 | 0 |
| **8 ms** (Micro-batch) | 4 | **230.7 dec/s** | **131.8 req/s** | **30.3 ms** | **33.6 ms** | 7.0 | 0 |
| **8 ms** (Micro-batch) | 8 | **411.8 dec/s** | **235.3 req/s** | **32.7 ms** | **36.0 ms** | 13.5 | 0 |
| **8 ms** (Micro-batch) | 16 | **559.7 dec/s** | **319.8 req/s** | **53.5 ms** | **59.9 ms** | 15.9 | 0 |
| **8 ms** (Micro-batch) | 32 | 581.7 dec/s | 332.4 req/s | 81.8 ms | 105.1 ms | 15.9 | 0 |

### Key Empirical Findings:
1. **Low Concurrency ($C=1$)**: Immediate dispatch (`batch_window = 0 ms`) delivers minimal latency (**17.3 ms** p50) without queue delay.
2. **Medium Concurrency ($C=4$ to $8$)**: Micro-batching (`batch_window = 8 ms`) delivers **+21% higher throughput** (411.8 vs 339.6 dec/s) and **lower latency** (32.7 ms vs 41.6 ms) due to optimal batch bundling.
3. **High Concurrency ($C \ge 16$)**: The engine saturates GPU parallelism at **~585 decisions/sec** (~335 HTTP requests/sec).
4. **Memory Footprint**: Peak allocated GPU VRAM during inference under load is **662.2 MiB** (well within standard cloud GPU bounds).

---

## Quickstart

### 1. Installation

Install `spectral-arbiter` with serving dependencies:

```bash
pip install "spectral-arbiter[serve]"
```

### 2. Launch Local Server

You can launch the server using either the `arbiter` CLI command or the Python module:

```bash
# Launch with GPU (CLI command)
arbiter serve --port 8000 --device cuda:0 --batch-window 8.0

# Launch with GPU (Python module)
python -m arbiter.serve --port 8000 --device cuda:0 --batch-window 8.0

# Launch with CPU
arbiter serve --port 8000 --device cpu --batch-window 0.0
# (or: python -m arbiter.serve --port 8000 --device cpu --batch-window 0.0)
```

CLI options:
- `--host`: Network interface (default: `0.0.0.0`).
- `--port`: Port to listen on (default: `8000`).
- `--model`: Hugging Face Hub ID (e.g. `spectral-ai-lab/arbiter-272m` or `spectral-ai-lab/arbiter-272m-typed`) or local checkpoint directory.
- `--served-model-name`: Explicit canonical model identity to expose in responses and validate against in incoming requests (overrides directory heuristics; env var `ARBITER_SERVED_MODEL_NAME`).
- `--batch-window`: Micro-batch window in milliseconds (default: `8.0`).
- `--max-batch-size`: Max batch size for grouped execution (default: `16`).
- `--max-batch-tokens`: Maximum total tokens per batch (default: `8192`).
- `--max-attention-elements`: Maximum attention quadratic elements per batch (default: `16777216`, supporting full 4096-token sequence contexts).
- `--max-queue-size`: Maximum pending queue depth before HTTP 503 backpressure (default: `1024`).

---

## API Usage Examples

### Example 1: `POST /v1/systemone` (vLLM Structured Decisions)

Send a multi-question structured decision request evaluating choice, boolean, and ordinal score in a single forward pass:

```bash
curl -s http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "ticket_id": "TCK-8492",
      "text": "My credit card was charged twice for order #9921."
    },
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {
          "billing": "Refunds and payment disputes",
          "shipping": "Carrier tracking and package delivery",
          "security": "Unauthorized account login"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does the customer require immediate financial intervention?",
        "criteria": {"true": "Card charged erroneously or account locked"}
      },
      "sentiment": {
        "type": "score",
        "instructions": "Rate customer frustration level",
        "criteria": ["calm", "annoyed", "furious"]
      }
    }
  }'
```

**Response**:
```json
{
  "id": "decision-7b98d234f9a1c80e1234abcd",
  "object": "structured_decision",
  "created": 1790000000,
  "model": "spectral-ai-lab/arbiter-272m",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.965412,
        "shipping": 0.012105,
        "security": 0.022483
      },
      "confidence": 0.965412
    },
    "urgent": {
      "type": "noul",
      "choice": "yes",
      "noul": 0.941208,
      "probabilities": {
        "yes": 0.941208,
        "no": 0.058792
      },
      "confidence": 0.941208
    },
    "sentiment": {
      "type": "score",
      "choice": "1",
      "score": 1.1205,
      "legend": {
        "0": "calm",
        "1": "annoyed",
        "2": "furious"
      },
      "probabilities": {
        "0": 0.124501,
        "1": 0.630498,
        "2": 0.245001
      },
      "confidence": 0.630498
    }
  },
  "usage": {
    "input_tokens": 142,
    "output_tokens": 0
  },
  "diagnostics": {
    "team": {
      "label_mass": 1.0,
      "argmax_is_label": true
    },
    "urgent": {
      "label_mass": 1.0,
      "argmax_is_label": true
    },
    "sentiment": {
      "label_mass": 1.0,
      "argmax_is_label": true
    }
  }
}
```

---

### Example 2: `POST /decide` (Native Flat Decision)

```bash
curl -s http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "context": "Customer wants to dispute an international wire transfer fee.",
    "instruction": "Route query to resolution desk",
    "criteria": {
      "disputes": "Chargebacks and transaction disputes",
      "fx_desk": "Foreign exchange and international rates",
      "support": "General information"
    },
    "question_type": "choice"
  }'
```

**Response**:
```json
{
  "selected_key": "disputes",
  "probabilities": {
    "disputes": 0.884120,
    "fx_desk": 0.098450,
    "support": 0.017430
  },
  "confidence": 0.884120,
  "input_tokens": 48,
  "truncated": false
}
```

---

## Docker & Container Deployment

### 1. Pre-Built Images from GitHub Container Registry (Recommended)

Pre-built, certified container images are published on **GitHub Container Registry (`ghcr.io`)**:

#### Run on GPU (NVIDIA CUDA 12, Blackwell / RTX 5090 native support)
```bash
docker run --gpus all -d -p 8000:8000 \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  --name arbiter-gpu \
  ghcr.io/spectral-ai-lab/arbiter-serve:latest
```

#### Run on CPU
```bash
docker run -d -p 8000:8000 \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  --name arbiter-cpu \
  ghcr.io/spectral-ai-lab/arbiter-serve:cpu
```

#### Serve the Typed Specialist
```bash
docker run --gpus all -d -p 8000:8000 \
  -v ${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface \
  --name arbiter-gpu-typed \
  ghcr.io/spectral-ai-lab/arbiter-serve:latest \
  --model spectral-ai-lab/arbiter-272m-typed
```

---

### 2. Building from Source (Local Docker / Podman)

If you prefer building the images locally:

#### GPU Image:
```bash
docker build -t arbiter-serve:gpu -f docker/Dockerfile.gpu .
```

#### CPU Image:
```bash
docker build -t arbiter-serve:cpu -f docker/Dockerfile.cpu .
```
