"""Full-weight HF embeddings with an explicit, MediaPipe-inspired decision layer.

This implements our documented formulas, not Google's unavailable native code.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np


@dataclass(frozen=True)
class DecisionConfig:
    temperature: float = 0.1
    center_candidates: bool = True
    normalize_prior: bool = False
    chunk_tokens: int = 512
    chunk_overlap: int = 128
    pooling_temperature: float = 0.1
    head_weight: float = 0.35

    def __post_init__(self):
        for name in ("temperature", "pooling_temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.chunk_overlap < self.chunk_tokens:
            raise ValueError("Require 0 <= chunk_overlap < chunk_tokens")
        if not math.isfinite(self.head_weight) or not 0 < self.head_weight < 1:
            raise ValueError("head_weight must be between 0 and 1")


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned nonfinite embeddings")
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero/degenerate embedding; revise candidate descriptions")
    return vectors / norms


def probabilities(scores, temperature):
    logits = np.asarray(scores, dtype=np.float64) / temperature
    logits -= logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def pool_scores(scores, temperature, head_weight):
    """Weighted log-mean-exp over spans, independently for each candidate."""
    if len(scores) == 1:
        return scores[0]
    weights = np.full(len(scores), (1 - head_weight) / (len(scores) - 1))
    weights[0] = head_weight
    scaled = np.asarray(scores, dtype=np.float64) / temperature
    maximum = scaled.max(axis=0)
    return temperature * (maximum + np.log((weights[:, None] * np.exp(scaled - maximum)).sum(axis=0)))


class HFEncoder:
    """Original google/embeddinggemma-2 checkpoint, never a quantized export."""
    def __init__(self, device="cpu", dtype="auto", revision=None, *, token=None,
                 cache_folder=None, local_files_only=False):
        import torch
        from sentence_transformers import SentenceTransformer

        if dtype not in ("auto", "float32", "bfloat16"):
            raise ValueError("Use auto, float32 or bfloat16; FP16 is not supported")
        self.model = SentenceTransformer(
            "google/embeddinggemma-2", revision=revision, device=device,
            token=token, cache_folder=cache_folder, local_files_only=local_files_only,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": "auto" if dtype == "auto" else getattr(torch, dtype)},
        )
        self.model.eval()
        # Full 768-dimensional representations, no MRL truncation or quantizer.
        self.model.max_seq_length = 8192
        self.tokenizer = self.model.tokenizer
        self.revision = getattr(self.model[0].auto_model.config, "_commit_hash", revision)
        self.device = str(self.model.device)
        self.dtype = str(self.model[0].auto_model.dtype).removeprefix("torch.")

    @staticmethod
    def format_text(text, instructions):
        content = f"{instructions}\n{text}" if instructions else text
        return f"task: classification | query: {content}"

    def encode(self, texts, instructions=""):
        inputs = [self.format_text(text, instructions) for text in texts]
        # Refuse silent truncation, including prefixes and question instructions.
        lengths = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in inputs]
        if max(lengths, default=0) > self.model.max_seq_length:
            raise ValueError("Input including instructions exceeds the model token limit")
        return self.model.encode(inputs, prompt="", normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False,
                                 precision="float32")

    def chunks(self, text, size, overlap):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= size:
            return [text]
        result = []
        for start in range(0, len(ids), size - overlap):
            result.append(self.tokenizer.decode(ids[start:start + size], skip_special_tokens=True))
            if start + size >= len(ids):
                break
        return result


class DecisionEngine:
    def __init__(self, *, device="cpu", dtype="auto", revision=None, config=None, encoder=None):
        self.config = config or DecisionConfig()
        self.encoder = encoder if encoder is not None else HFEncoder(device, dtype, revision)
        # Per-instance bounded cache: descriptions, order and instructions are keys.
        self._prepare = lru_cache(maxsize=128)(self._prepare_uncached)

    @property
    def metadata(self):
        return {"model": "google/embeddinggemma-2", "quantization": None,
                "dtype": getattr(self.encoder, "dtype", "test"),
                "revision": getattr(self.encoder, "revision", None),
                "device": getattr(self.encoder, "device", "test"),
                "decision_config": asdict(self.config), "mediapipe_exact_replica": False}

    def _prepare_uncached(self, items, instructions):
        descriptions = [description or label for label, description in items]
        raw = unit(self.encoder.encode(descriptions, instructions))
        if raw.ndim != 2 or len(raw) != len(items):
            raise ValueError("Unexpected candidate embedding shape")
        center = raw.mean(axis=0) if self.config.center_candidates else np.zeros(raw.shape[1], dtype=np.float32)
        candidates = unit(raw - center)
        prior = None
        if self.config.normalize_prior:
            baseline = unit(self.encoder.encode([""], instructions))
            prior = (unit(baseline - center) @ candidates.T)[0]
        return center, candidates, prior

    def choice(self, text, criteria, *, instructions="", temperature=None):
        if not isinstance(text, str) or not isinstance(instructions, str):
            raise ValueError("text and instructions must be strings")
        if len(criteria) < 2 or any(not isinstance(k, str) or not isinstance(v, str)
                                   for k, v in criteria.items()):
            raise ValueError("Provide at least two labels with string descriptions")
        temp = self.config.temperature if temperature is None else temperature
        if not math.isfinite(temp) or temp <= 0:
            raise ValueError("temperature must be finite and positive")
        items = tuple(criteria.items())
        center, candidates, prior = self._prepare(items, instructions)
        chunks = self.encoder.chunks(text, self.config.chunk_tokens, self.config.chunk_overlap)
        queries = unit(self.encoder.encode(chunks, instructions))
        scores = unit(queries - center) @ candidates.T
        pooled = pool_scores(scores, self.config.pooling_temperature, self.config.head_weight)
        if prior is not None:
            pooled = pooled - prior
        probs = probabilities(pooled, temp)
        labels = list(criteria)
        best = int(probs.argmax())
        return {"selected_key": labels[best],
                "probabilities": dict(zip(labels, map(float, probs))),
                "scores": dict(zip(labels, map(float, pooled))),
                "confidence": float(probs[best]), "chunks": len(chunks)}

    def predict(self, state, questions):
        """Accept the common Jev/Laya state/questions JSON contract."""
        from .compat import predict
        return predict(self, state, questions)
