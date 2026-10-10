"""Configuration class for ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning."""
from __future__ import annotations

from typing import Any
from transformers import PretrainedConfig
from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import EmbeddingGemma2TextConfig


class ArbiterConfig(PretrainedConfig):
    model_type = "arbiter"

    def __init__(
        self,
        base_model_name: str = "google/embeddinggemma-2",
        hidden_size: int = 512,
        vocab_size: int = 262145,
        max_position_embeddings: int = 4096,
        torch_dtype: str = "bfloat16",
        dtype: str | None = None,
        head_arch: str = "hybrid_ef",
        num_heads: int = 4,
        type_embed_dim: int = 64,
        hypernet_hidden_dim: int = 128,
        option_token: str = "<OPTION>",
        option_token_id: int = 262144,
        text_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        actual_dtype = dtype or torch_dtype or "bfloat16"
        kwargs["torch_dtype"] = actual_dtype
        kwargs["dtype"] = actual_dtype

        self.base_model_name = base_model_name
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.torch_dtype = actual_dtype
        self.dtype = actual_dtype
        self.head_arch = head_arch
        self.num_heads = num_heads
        self.type_embed_dim = type_embed_dim
        self.hypernet_hidden_dim = hypernet_hidden_dim
        self.option_token = option_token
        self.option_token_id = option_token_id
        self.text_config = text_config or {}

        super().__init__(**kwargs)
        self.torch_dtype = actual_dtype
        self.dtype = actual_dtype

    def get_text_config(self, decoder: bool = False, **kwargs) -> EmbeddingGemma2TextConfig:
        if self.text_config:
            return EmbeddingGemma2TextConfig(**self.text_config)
        raise ValueError(
            "text_config dictionary is missing from ArbiterConfig. "
            "ARBITER requires self-contained offline text_config for fully autonomous initialization."
        )
