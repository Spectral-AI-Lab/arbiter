# ARBITER Benchmark Suite

This module provides the standalone evaluation harness and frozen datasets required to reproduce empirical benchmarks for **ARBITER-272M**, **EmbeddingGemma-2 base** (`hf_pipeline/`), **Laya** (ConvAI), and **GLiNER**.

## Quick Reproduction

### 1. Public Suites (23 suites, 9,252 decisions: SST-2, TREC, Banking77, Fast Decisions)

```bash
# Evaluate ARBITER-272M on GPU:
python benchmark/run.py --model arbiter-272m \
  --data benchmark/data/public_requested_v1/cases.jsonl \
  --device cuda:0 --threads 16 --output results/public_benchmark

# Evaluate EmbeddingGemma-2 base pipeline on GPU:
python benchmark/run.py --model embeddinggemma2 \
  --data benchmark/data/public_requested_v1/cases.jsonl \
  --device cuda:0 --threads 16 --output results/public_benchmark
```

### 2. Historical Suites (60 suites, 21,915 decisions: MASSIVE, Typed Decisions, etc.)

```bash
python benchmark/run.py --model arbiter-272m \
  --data benchmark/data/cases.jsonl \
  --device cuda:0 --threads 16 --output results/historical_benchmark
```

## Baseline Integration & Upstream Origin

The original reference scripts are preserved intact in `upstream/`, anchored at commit `3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63`. Benchmark preparation executes only their dataset-building routines, without their model loading or GPU execution cells. The pipeline strictly fails if any suite is missing; no prompt or dataset substitution is permitted. For `PolyAI/banking77`, the deprecated legacy script is replaced with direct reading of the official PolyAI CSV at commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. Row ordering and `ClassLabel` mappings from the Hugging Face script are preserved; cryptographic source hashes are recorded.

| Group | Official Suites & Sampling |
|---|---|
| `typed` | LocalLLaMA/typed-decisions, full test split (400 cases, 2,000 decisions) |
| `multilingual` | MASSIVE intent and scenario, 14 languages including French; XNLI, 15 languages; first 300 cases per suite |
| `english` | SST-5 (600), Emotion (600), prompt-injections (full test), Banking77 (500), AG News (600), BoolQ (600) |
| `apps` | 10 suites: AG News, Emotion, Banking77, support, spam, phishing, jailbreak, toxicity, RAG relevance, domain routing; N=400 per official filters |

MASSIVE uses 20 candidate options: the ground truth label + 19 distractors (seed 13). This evaluates candidate selection rather than classification across the full intent inventory. The application suites also include training splits and tasks known to Laya: their scores do not all represent out-of-distribution generalization. Head subsets are not necessarily representative across entire distributions. English and application groups overlap: they must not be summed into an unweighted global score.

## Comparison Protocol

- Identical inputs and questions across all models, constructed once into JSONL format and verified via SHA-256.
- Distinct separate runs: ARBITER package, HF base pipeline, Laya English, Laya Multilingual, Laya Typed Decisions, GLiNER2.5-Decide, and GLiNER2.5-Multi-Decide. No retrospective checkpoint selection.
- Laya checkpoints locked to the official revisions recommended by `laya/revisions.py`. Laya upstream code locked to the commit above. Google revision locked in `hf_pipeline/config.json`.
- Unified target device: CPU or CUDA (`--device cpu`, `cuda`, `cuda:0`), default 8 CPU threads, batch size 1, 3 warm-up requests. ARBITER runs in native BF16 (`dtype="auto"`), Laya applies its native precision policy on the selected device. Actual precision is recorded.
- No test-set temperature tuning: models retain their delivered default inference temperatures.
- All predictions, errors, truncations, versions, parameters, and execution times are persisted. No silent replacement of failed model invocations.

## Metrics & Statistical Rigor

- Primary metric: Argmax Accuracy (`accuracy_including_failures`), ECE (15 bins), multiclass Brier score, NLL, score MAE, soft-target agreement for typed-decisions.
- Calibration uses `max(probabilities)` for all models, never the raw `confidence` field (due to disparate formulas between architectures). Minor floating-point rounding discrepancies are renormalized.
- Failures remain in the denominator of `accuracy_including_failures`; secondary metrics are computed on valid responses with reported coverage.
- Paired request-cluster bootstrap: all questions tied to a single state remain clustered together (2,000 resamples, seed 13). Confidence intervals are exploratory without multiple-testing adjustment.
- Latencies (p50 / p95) measure the entire end-to-end call, including tokenization and post-processing, after warm-up. Peak RSS includes Python imports, weight loading, and inference.
- Known constraints: context budgets are model-specific; Banking77 (77 options) may exceed Laya's option capacity, and these failures are preserved in the reported metrics.

## Local Test Verification

```bash
python -m unittest discover -s benchmark/tests -v
```

These unit tests verify metric calculations and pipeline coupling. Full benchmark runs require dependencies, datasets, and model checkpoints.

Upstream sources: [official scripts & protocols](https://github.com/NandhaKishorM/laya/tree/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/research), [published limits](https://github.com/NandhaKishorM/laya/blob/3cf26cbcb18725dbc2d127bb8bb2c4c43243ae63/docs/benchmarks.md). Upstream code Apache-2.0; license included in `upstream/LICENSE`.

## GPU Execution from Notebook or CLI

In `notebook/benchmark_laya.ipynb`, configure `DEVICE = "cuda:0"` (GPU) or `DEVICE = "cpu"`.

```bash
python benchmark/run.py --model all --device cuda:0 --threads 8 --output results/laya_comparison_cuda_0
python benchmark/compare.py results/laya_comparison_cuda_0
```

Each model runs in a separate process on the requested device without silent CPU fallback. CUDA timings are synchronized before and after each successful call. Metadata records GPU device, CUDA version, and PyTorch peak allocated/reserved memory stats.
