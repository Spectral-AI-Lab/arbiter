# Changelog

## [1.1.1] - 2026-10-13

- Declare `arbiter` console script CLI entrypoint in `pyproject.toml` (`arbiter serve`).
- Add `arbiter/cli.py` with serving delegation and graceful error messages for missing dependencies.
- Add defensive imports in `configuration_arbiter.py` and `modeling_arbiter.py` for older `transformers` runtimes.
- Support official custom `notebook.ipynb` directly in Hugging Face repository for 1-click Colab execution.
- Update Docker commands across documentation to explicitly specify `--model`.

## [1.1.0] - 2026-10-13

- Add online serving layer `arbiter.serve` with FastAPI and vLLM-compatible `/v1/systemone` adapter.
- Implement dynamic micro-batching scheduler and real-time metrics collector.
- Add CPU and GPU container definitions for GHCR deployment.
- Publish `spectral-arbiter` v1.1.0 to PyPI with `[serve]` dependencies extra.
- Add standalone Hugging Face `AutoModel` remote code execution support.
- Streamline repository header badges (Colab, Hugging Face, PyPI, Docker).

## [1.0.0] - 2026-10-12

- Initial official release of ARBITER-272M (BF16 safetensors).
- Clarify weight availability, checkout-only workflows and dependency requirements.
- Add separate unit/integration test selection, documentation checks and CI.
