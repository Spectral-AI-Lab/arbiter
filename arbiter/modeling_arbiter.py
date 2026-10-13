"""ARBITER: Attentive Relational Bidirectional Impartial Typed Encoder for Reasoning.

Official inference and decision model with Set-Attention (Inter-Candidate Multi-Head Self-Attention)
and Dynamic HyperFiLM modulation. Fully autonomous loading from local safetensors without external downloads.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file
from transformers import PreTrainedModel, AutoTokenizer

try:
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import EmbeddingGemma2TextModel
except ImportError:
    try:
        from transformers import EmbeddingGemma2TextModel
    except ImportError:
        EmbeddingGemma2TextModel = None

try:
    from .configuration_arbiter import ArbiterConfig
except (ImportError, ValueError):
    from configuration_arbiter import ArbiterConfig

logger = logging.getLogger(__name__)


def _build_mlp(in_dim: int = 512, hidden_dim: int = 256, dropout_p: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim, dtype=torch.float32),
        nn.GELU(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_dim, 1, dtype=torch.float32),
    )


class SetTransformerHyperFiLMHead(nn.Module):
    """
    ARBITER Relational Decision Head: Inter-Candidate Multihead Attention + Dynamic HyperFiLM Modulation.
    Pipeline:
    1. Inter-Candidate Multihead Self-Attention:
       H_attn = H + alpha_attn * MHA(LN(H), LN(H), LN(H))
    2. Dynamic Context & Task Modulation:
       c = [h_context || e_t]
       [gamma_raw, beta_raw] = HyperNet(c)
       gamma = 1.0 + alpha_film * gamma_raw
       beta = alpha_film * beta_raw
       H' = gamma * H_attn + beta
    3. Final Scoring:
       z_i = MLP(h'_i)
    """

    def __init__(
        self,
        dropout_p: float = 0.1,
        num_heads: int = 4,
        cond_dim: int = 512,
        type_embed_dim: int = 64,
        hypernet_hidden_dim: int = 128,
    ):
        super().__init__()
        self.head = _build_mlp(512, 256, dropout_p)

        self.ln = nn.LayerNorm(512, dtype=torch.float32)
        self.mha = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=num_heads,
            batch_first=True,
            dtype=torch.float32,
        )
        self.alpha_attn = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.type_to_id = {"choice": 0, "noul": 1, "score": 2}
        self.type_embed = nn.Embedding(3, type_embed_dim, dtype=torch.float32)
        self.hypernet = nn.Sequential(
            nn.Linear(cond_dim + type_embed_dim, hypernet_hidden_dim, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(hypernet_hidden_dim, 512 * 2, dtype=torch.float32),
        )
        self.alpha_film = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(
        self,
        hidden: torch.Tensor,
        qtype: str = "choice",
        cond_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden: [K, 512]
        h_norm = self.ln(hidden)
        attn_out, _ = self.mha(
            h_norm.unsqueeze(0),
            h_norm.unsqueeze(0),
            h_norm.unsqueeze(0),
            need_weights=False,
        )
        attn_out = attn_out.squeeze(0)
        h_attn = hidden + self.alpha_attn * attn_out

        t_id = torch.tensor(self.type_to_id.get(qtype, 0), device=hidden.device)
        e_t = self.type_embed(t_id)
        if cond_vector is None:
            cond_vector = hidden.mean(dim=0)
        c = torch.cat([cond_vector, e_t], dim=-1).unsqueeze(0)
        film_params = self.hypernet(c)
        gamma_raw, beta_raw = film_params.chunk(2, dim=-1)

        gamma = 1.0 + self.alpha_film * gamma_raw
        beta = self.alpha_film * beta_raw
        modulated = gamma * h_attn + beta

        return self.head(modulated).squeeze(-1).float()


class ArbiterModel(PreTrainedModel):
    """
    ARBITER Decision Model:
    - Backbone: EmbeddingGemma-2 Text Encoder (512-dim native hidden states)
    - Decision Head: ARBITER (Set-Attention + HyperFiLM)
    - Full unified multi-task support: choice, noul (boolean), score (ordinal)
    """
    config_class = ArbiterConfig
    base_model_prefix = "arbiter"

    def __init__(self, config: ArbiterConfig):
        super().__init__(config)
        self.config = config

        if EmbeddingGemma2TextModel is None:
            raise ImportError(
                "ARBITER requires transformers>=5.19.0 (with EmbeddingGemma2 support).\n"
                "Your current environment has an older version of transformers without EmbeddingGemma2.\n"
                "To resolve this, please upgrade transformers:\n"
                "    pip install --upgrade 'transformers>=5.19.0'\n"
                "or install the official spectral-arbiter package:\n"
                "    pip install --upgrade spectral-arbiter"
            )

        # Instantiate language model directly from config without downloading Google weights
        text_config = config.get_text_config()
        text_config.vocab_size = config.vocab_size
        self.language_model = EmbeddingGemma2TextModel(text_config)
        self.language_model.embedding_projection = nn.Identity()

        # Decision head
        self.decision_head = SetTransformerHyperFiLMHead(
            dropout_p=0.1,
            num_heads=config.num_heads,
            cond_dim=config.hidden_size,
            type_embed_dim=config.type_embed_dim,
            hypernet_hidden_dim=config.hypernet_hidden_dim,
        )

        self.tokenizer = None
        self.option_token = config.option_token
        self.option_token_id = config.option_token_id

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        device: str | torch.device | None = None,
        dtype: torch.dtype | str | None = None,
        torch_dtype: torch.dtype | str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> ArbiterModel:
        """
        Load ARBITER from local directory or Hugging Face Hub repo.
        """
        # Handle device_map gracefully for Hugging Face ergonomics
        device_map = kwargs.pop("device_map", None)
        if device_map is not None:
            logger.info(
                "ARBITER is a compact 272M parameter single-device model. "
                "Interpreting device_map=%r as single-device placement.",
                device_map,
            )
            if device is None:
                if str(device_map).lower() in ("auto", "balanced", "cuda"):
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                elif str(device_map).lower() == "cpu":
                    device = "cpu"
                elif isinstance(device_map, str) and device_map.startswith("cuda"):
                    device = device_map
                elif isinstance(device_map, dict):
                    val = next(iter(device_map.values()), "cpu")
                    device = f"cuda:{val}" if isinstance(val, int) else str(val)
                else:
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"

        hub_options = {"revision", "cache_dir", "token", "force_download", "local_files_only"}
        hf_internal_kwargs = {"trust_remote_code", "config", "adapter_kwargs", "code_revision", "proxies"}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        for k in hf_internal_kwargs:
            kwargs.pop(k, None)
        unsupported = set(kwargs) - hub_options
        if unsupported:
            raise TypeError(f"Unsupported loading arguments: {sorted(unsupported)}. "
                            "Use device or device_map for device placement.")
        model_path_str = str(pretrained_model_name_or_path)
        model_path = Path(model_path_str)

        if model_path.exists():
            for k in list(kwargs):
                if k in hub_options:
                    kwargs.pop(k)

        # Resolve remote repo if path does not exist locally
        if not model_path.exists():
            try:
                from huggingface_hub import snapshot_download
                model_path_str = snapshot_download(
                    repo_id=model_path_str,
                    local_files_only=local_files_only,
                    **kwargs,
                )
                model_path = Path(model_path_str)
            except Exception as e:
                if local_files_only:
                    raise FileNotFoundError(f"Local files not found for {model_path_str} and local_files_only=True") from e
                raise

        # 1. Load config
        config_file = model_path / "config.json"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config = ArbiterConfig(**config_dict)
        else:
            config = ArbiterConfig()

        # 2. Determine target dtype safely
        dtype_arg = dtype if dtype is not None else torch_dtype
        if dtype_arg is None or dtype_arg == "auto":
            cfg_dt = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None) or "bfloat16"
            if isinstance(cfg_dt, torch.dtype):
                target_dtype = cfg_dt
            elif str(cfg_dt).replace("torch.", "").lower() in ("bfloat16", "bf16"):
                target_dtype = torch.bfloat16
            elif str(cfg_dt).replace("torch.", "").lower() in ("float16", "fp16"):
                target_dtype = torch.float16
            elif str(cfg_dt).replace("torch.", "").lower() in ("float32", "fp32"):
                target_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported dtype in config: {cfg_dt!r}")
        elif isinstance(dtype_arg, torch.dtype):
            target_dtype = dtype_arg
        elif isinstance(dtype_arg, str):
            clean_dt = dtype_arg.replace("torch.", "").lower()
            if clean_dt in ("bfloat16", "bf16"):
                target_dtype = torch.bfloat16
            elif clean_dt in ("float16", "fp16"):
                target_dtype = torch.float16
            elif clean_dt in ("float32", "fp32"):
                target_dtype = torch.float32
            else:
                raise ValueError(
                    f"Unsupported dtype string '{dtype_arg}'. Expected one of 'bfloat16', 'float16', 'float32', or 'auto'."
                )
        else:
            raise ValueError(f"Unsupported dtype type: {type(dtype_arg)}. Expected torch.dtype or str.")

        # 3. Instantiate model
        model = cls(config)

        # 4. Load weights directly from safetensors or pt
        safetensors_path = model_path / "model.safetensors"
        pt_path = model_path / "best_model.pt"

        if safetensors_path.exists():
            state_dict = load_safetensors_file(str(safetensors_path))
        elif pt_path.exists():
            ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=True)
            state_dict = ckpt.get("model_state_dict", ckpt)
        else:
            # Fallback search for any .safetensors in directory
            st_files = list(model_path.glob("*.safetensors"))
            if len(st_files) > 1:
                raise ValueError("Ambiguous weights: provide a single model.safetensors file; sharded packages are not supported.")
            if st_files:
                state_dict = load_safetensors_file(str(st_files[0]))
            else:
                raise FileNotFoundError(f"No safetensors or pt weights found in {model_path}")

        # Cast state_dict to target dtype and load strictly
        state_dict_converted = {k: v.to(dtype=target_dtype) for k, v in state_dict.items()}
        model.to(dtype=target_dtype)
        model.load_state_dict(state_dict_converted, strict=True)

        # 5. Load tokenizer locally from the directory if present (fully autonomous)
        has_local_tok = (model_path / "tokenizer.json").exists() or (model_path / "tokenizer_config.json").exists()
        if has_local_tok:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        elif not local_files_only:
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except Exception:
                raise FileNotFoundError(
                    f"Tokenizer files not found in {model_path}. Autonomous loading requires tokenizer.json in release directory."
                )
        else:
            raise FileNotFoundError(
                f"Tokenizer files (tokenizer.json) not found in {model_path} with local_files_only=True"
            )

        # Strict validation of option marker, token ID and embedding table size
        if config.option_token not in tokenizer.get_vocab():
            raise ValueError(
                f"Special marker '{config.option_token}' is missing from tokenizer vocabulary. Corrupted or mismatched tokenizer."
            )

        tok_opt_id = tokenizer.convert_tokens_to_ids(config.option_token)
        if tok_opt_id != config.option_token_id:
            raise ValueError(
                f"Option token ID mismatch: tokenizer assigns {tok_opt_id} to '{config.option_token}', "
                f"but config specifies {config.option_token_id}."
            )

        embed_vocab_size = model.language_model.embed_tokens.weight.shape[0]
        if embed_vocab_size < len(tokenizer):
            raise ValueError(
                f"Embedding table size ({embed_vocab_size}) is smaller than tokenizer vocabulary ({len(tokenizer)})."
            )

        model.tokenizer = tokenizer
        model.option_token_id = tok_opt_id

        # Move to target device
        target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device=target_device)
        model.eval()

        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_indices: list[list[int]],
        question_types: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = input_ids.shape[0]
        max_options = max((len(opts) for opts in option_indices), default=1)
        qtypes = question_types if question_types is not None else ["choice"] * batch_size

        encoder_out = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_out.last_hidden_state  # [B, seq_len, 512]

        dev = input_ids.device
        padded_logits = torch.full((batch_size, max_options), -1e4, device=dev, dtype=torch.float32)
        mask = torch.zeros((batch_size, max_options), device=dev, dtype=torch.bool)

        for b in range(batch_size):
            opts = option_indices[b]
            k = len(opts)
            if k == 0:
                continue
            opts_tensor = torch.tensor(opts, device=dev, dtype=torch.long)
            opt_hidden = hidden_states[b, opts_tensor, :]  # [k, 512]
            qtype = qtypes[b] if b < len(qtypes) else "choice"
            cond_vec = hidden_states[b, 0, :]  # [512] at <bos>

            scores = self.decision_head(opt_hidden, qtype, cond_vector=cond_vec)
            padded_logits[b, :k] = scores
            mask[b, :k] = True

        probs = torch.softmax(padded_logits.masked_fill(~mask, -1e4), dim=-1)

        return {
            "logits": padded_logits,
            "probabilities": probs,
            "mask": mask,
        }

    def _prepare_decision(
        self,
        context: str,
        instruction: str,
        criteria: dict[str, str] | list[str] | None,
        question_type: str = "choice",
        max_length: int = 4096,
        allow_truncation: bool = False,
    ) -> dict[str, Any]:
        """
        High-level single-query decision interface.
        Supports choice, noul, and score question types with contract validation.
        """
        if self.tokenizer is None:
            raise RuntimeError("Model tokenizer is not loaded. Ensure the model was loaded via from_pretrained.")

        # Input contract validations
        if not isinstance(context, str) or not isinstance(instruction, str):
            raise TypeError("Both 'context' and 'instruction' must be strings.")
        if not context.strip():
            raise ValueError("Context cannot be empty.")
        if not instruction.strip():
            raise ValueError("Instruction cannot be empty.")

        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if max_length > self.config.max_position_embeddings:
            raise ValueError(
                f"Requested max_length ({max_length}) exceeds model max_position_embeddings "
                f"({self.config.max_position_embeddings})."
            )

        if question_type not in ("choice", "noul", "score"):
            raise ValueError(f"Invalid question_type '{question_type}'. Must be 'choice', 'noul', or 'score'.")

        # Canonicalize and validate criteria per question type
        if question_type == "noul":
            if criteria is None or (isinstance(criteria, (dict, list)) and len(criteria) == 0):
                criteria = {"false": "The statement is false.", "true": "The statement is true."}
            elif isinstance(criteria, dict):
                raw_keys = list(criteria.keys())
                if len(raw_keys) != 2:
                    raise ValueError(f"For question_type='noul', criteria must contain exactly 2 options (got {len(raw_keys)}).")
                norm_keys = [str(k).lower().strip() for k in raw_keys]
                if len(set(norm_keys)) != len(norm_keys):
                    raise ValueError(
                        f"Criteria contains conflicting or colliding boolean keys: {raw_keys}. "
                        "Cannot resolve ambiguous duplicate boolean representations."
                    )
                if set(norm_keys) != {"false", "true"}:
                    raise ValueError(
                        f"For question_type='noul', criteria dictionary keys must be canonical ('false' and 'true'). "
                        f"Got: {raw_keys}. Arbitrary keys are rejected to prevent semantic inversion."
                    )
                clean_dict = {norm_keys[i]: str(criteria[raw_keys[i]]) for i in range(len(raw_keys))}
                criteria = {"false": clean_dict["false"], "true": clean_dict["true"]}
            elif isinstance(criteria, (list, tuple)):
                if len(criteria) != 2:
                    raise ValueError(f"For question_type='noul', criteria must contain exactly 2 options (got {len(criteria)}).")
                c0 = str(criteria[0]).lower().strip()
                c1 = str(criteria[1]).lower().strip()
                if {c0, c1} == {"false", "true"}:
                    desc_f = str(criteria[0]) if c0 == "false" else str(criteria[1])
                    desc_t = str(criteria[1]) if c1 == "true" else str(criteria[0])
                    criteria = {"false": desc_f, "true": desc_t}
                else:
                    raise ValueError(
                        f"For question_type='noul', criteria list items must be canonical ('false', 'true'). "
                        f"Got: {list(criteria)}."
                    )
            else:
                raise TypeError(f"Invalid type for criteria: {type(criteria)}. Expected dict, list, or None.")
        elif question_type in ("choice", "score"):
            if criteria is None or len(criteria) < 2:
                raise ValueError(f"For question_type='{question_type}', criteria must contain at least 2 options.")

        if isinstance(criteria, dict):
            raw_keys = list(criteria.keys())
            norm_keys = [str(k).strip() for k in raw_keys]
            if len(set(norm_keys)) != len(norm_keys):
                raise ValueError(f"Criteria contains duplicate or colliding keys: {raw_keys}")
            options = [(norm_keys[i], str(criteria[raw_keys[i]])) for i in range(len(raw_keys))]
            keys = norm_keys
        elif isinstance(criteria, (list, tuple)):
            options = [(str(i), str(desc)) for i, desc in enumerate(criteria)]
            keys = [str(i) for i in range(len(criteria))]
        else:
            raise TypeError(f"Invalid type for criteria: {type(criteria)}. Expected dict or list.")

        if any(self.option_token in key for key in keys):
            raise ValueError("Criterion keys must not contain the reserved <OPTION> marker. "
                             "Rename the key; descriptions may contain literal marker text.")

        # Guard: sanitize raw context/instruction to prevent collision with the internal <OPTION> marker
        safe_context = re.sub(r"<OPTION>", "[OPTION]", str(context).strip())
        safe_instruction = re.sub(r"<OPTION>", "[OPTION]", str(instruction).strip())

        opt_lines = [
            f"{self.option_token} {label}: {re.sub(r'<OPTION>', '[OPTION]', str(desc).strip())}" if desc else f"{self.option_token} {label}"
            for label, desc in options
        ]
        options_text = "\n".join(opt_lines)
        header = f"Contexte : {safe_context}\nInstruction : {safe_instruction}\n\n"
        prompt = header + options_text

        # Length and Truncation Contract Check
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        raw_token_count = len(ids)
        is_truncated = False

        if raw_token_count > max_length:
            if not allow_truncation:
                raise ValueError(
                    f"Input sequence length ({raw_token_count} tokens) exceeds max_length ({max_length}). "
                    "By default, ARBITER rejects truncated inputs to prevent silent loss of candidate descriptions. "
                    "Set allow_truncation=True to safely truncate context without clipping candidate options."
                )
            # Safe truncation: ONLY truncate context; never cut instruction or candidate options
            overhead_tokens = len(self.tokenizer.encode(f"Contexte : \nInstruction : {safe_instruction}\n\n{options_text}", add_special_tokens=True))
            if overhead_tokens >= max_length:
                raise ValueError(
                    f"Instruction and candidate options alone require {overhead_tokens} tokens, exceeding max_length ({max_length}). "
                    "Cannot safely fit candidate options within budget."
                )
            avail_ctx_tokens = max(0, max_length - overhead_tokens)
            ctx_ids = self.tokenizer.encode(safe_context, add_special_tokens=False)[:avail_ctx_tokens]

            # Iterative reduction loop to strictly guarantee token budget
            while True:
                truncated_context = self.tokenizer.decode(ctx_ids, skip_special_tokens=True)
                header = f"Contexte : {truncated_context}\nInstruction : {safe_instruction}\n\n"
                prompt = header + options_text
                ids = self.tokenizer.encode(prompt, add_special_tokens=True)
                current_tokens = len(ids)
                if current_tokens <= max_length or len(ctx_ids) == 0:
                    break
                overshoot = max(1, current_tokens - max_length)
                ctx_ids = ctx_ids[:-overshoot]

            is_truncated = True

        opt_pos = [i for i, token in enumerate(ids) if token == self.option_token_id]
        if len(opt_pos) != len(options):
            raise ValueError("Option count mismatch: candidate markers are damaged or desynchronized.")
        return dict(ids=ids, positions=opt_pos, keys=keys, question_type=question_type,
                    raw_token_count=raw_token_count, is_truncated=is_truncated)

    @staticmethod
    def _decision_result(prepared, probs):
        keys = prepared["keys"]
        question_type = prepared["question_type"]
        raw_token_count = prepared["raw_token_count"]
        is_truncated = prepared["is_truncated"]
        prob_dict = dict(zip(keys, probs))
        best_idx = max(range(len(probs)), key=probs.__getitem__)
        best_key = keys[best_idx]

        res = {
            "selected_key": best_key,
            "probabilities": prob_dict,
            "confidence": probs[best_idx],
            "input_tokens": raw_token_count,
            "truncated": is_truncated,
        }

        if question_type == "score":
            res["score"] = sum(i * p for i, p in enumerate(probs))
        elif question_type == "noul":
            res["noul"] = prob_dict["true"]

        return res

    def decide(
        self, context: str, instruction: str,
        criteria: dict[str, str] | list[str] | None,
        question_type: str = "choice", max_length: int = 4096,
        allow_truncation: bool = False,
    ) -> dict[str, Any]:
        """Run one decision using the same preparation and inference path as batching."""
        return self.decide_batch([dict(context=context, instruction=instruction,
            criteria=criteria, question_type=question_type, max_length=max_length,
            allow_truncation=allow_truncation)], batch_size=1)[0]

    @torch.inference_mode()
    def decide_batch(
        self, requests: list[dict[str, Any]], *, batch_size: int = 8,
        max_batch_tokens: int = 8192,
        max_attention_elements: int = 16777216,
        max_length: int = 4096, allow_truncation: bool = False,
    ) -> list[dict[str, Any]]:
        """Length-bucketed inference, returning results in original request order.

        Budgets bound padded B*L and B*L², not actual device memory in bytes.
        A request that cannot fit alone is rejected before any forward pass.
        Per-request max_length/allow_truncation override batch defaults.
        All inputs are validated before execution. Use bounded request lists in services.
        """
        for name, value in (("batch_size", batch_size),
                            ("max_batch_tokens", max_batch_tokens),
                            ("max_attention_elements", max_attention_elements)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.training:
            raise RuntimeError("Call model.eval() before inference.")
        prepared = []
        for i, request in enumerate(requests):
            try:
                if "_prepared" in request:
                    item = request["_prepared"]
                else:
                    item = self._prepare_decision(**dict(
                        dict(max_length=max_length, allow_truncation=allow_truncation), **request))
                length = len(item["ids"])
                if length > max_batch_tokens or length * length > max_attention_elements:
                    raise ValueError("Request exceeds batch budget even as a singleton")
                prepared.append(item)
            except (ValueError, TypeError) as exc:
                raise type(exc)(f"Request {i}: {exc}") from exc
        order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]["ids"]))
        batches, current = [], []
        for index in order:
            length = len(prepared[index]["ids"])
            size = len(current) + 1
            if current and (size > batch_size or size * length > max_batch_tokens
                            or size * length * length > max_attention_elements):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        results = [None] * len(prepared)
        device = next(self.parameters()).device
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        for batch in batches:
            length = max(len(prepared[i]["ids"]) for i in batch)
            ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
            attention = torch.zeros_like(ids)
            for row, index in enumerate(batch):
                tokens = prepared[index]["ids"]
                ids[row, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                attention[row, :len(tokens)] = 1
            out = self.forward(input_ids=ids.to(device), attention_mask=attention.to(device),
                option_indices=[prepared[i]["positions"] for i in batch],
                question_types=[prepared[i]["question_type"] for i in batch])
            probabilities = out["probabilities"].float().cpu().tolist()
            for row, index in enumerate(batch):
                item = prepared[index]
                results[index] = self._decision_result(item, probabilities[row][:len(item["keys"])])
        return results

    def predict(self, state: Any, questions: dict[str, Any], **batch_options) -> dict[str, Any]:
        """Batch the questions for one state; preserve the existing answer schema."""
        return self.predict_batch([dict(state=state, questions=questions)], **batch_options)[0]

    def predict_batch(self, requests: list[dict[str, Any]], **batch_options) -> list[dict[str, Any]]:
        """Batch questions across states, then restore state and question order."""
        flat, locations = [], []
        outputs = [{"answers": {}} for _ in requests]
        for index, request in enumerate(requests):
            state, questions = request["state"], request["questions"]
            context = json.dumps(state, ensure_ascii=False) if isinstance(state, (dict, list)) else str(state)
            for qid, question in questions.items():
                qtype = question.get("type", "choice")
                if ("instructions" in question and "instruction" in question
                        and question["instructions"] != question["instruction"]):
                    raise ValueError(f"Question {qid!r}: conflicting instruction and instructions values")
                instruction = question.get("instructions", question.get("instruction", ""))
                flat.append(dict(context=context, instruction=instruction,
                    criteria=question.get("criteria"), question_type=qtype))
                locations.append((index, qid, qtype))
        for (index, qid, qtype), result in zip(locations, self.decide_batch(flat, **batch_options)):
            answer = {"probabilities": result["probabilities"]}
            if qtype in ("choice", "score"):
                answer["choice"] = result["selected_key"]
            if qtype in ("score", "noul"):
                answer[qtype] = result[qtype]
            outputs[index]["answers"][qid] = answer
        return outputs
