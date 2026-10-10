"""HF wrapper: configuration and code only; weights come from Google's pinned repo."""
from transformers import PretrainedConfig, PreTrainedModel
from .engine import DecisionConfig, DecisionEngine, HFEncoder


class DecisionHFConfig(PretrainedConfig):
    model_type = "embeddinggemma2_decision"

    def __init__(self, base_revision=None, decision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.base_revision = base_revision
        self.decision_config = decision_config or {}


class DecisionHFModel(PreTrainedModel):
    config_class = DecisionHFConfig

    def __init__(self, config, engine):
        super().__init__(config)
        self.engine = engine
        self.encoder = engine.encoder.model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, config=None, **kwargs):
        if config is None:
            config = DecisionHFConfig.from_pretrained(pretrained_model_name_or_path)
        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", "auto"))
        dtype = str(dtype).removeprefix("torch.")
        if kwargs.get("device_map") is not None:
            raise ValueError("Use the pipeline device argument; device_map is not supported by this wrapper")
        encoder = HFEncoder(device="cpu", dtype=dtype, revision=config.base_revision,
                            token=kwargs.get("token"), cache_folder=kwargs.get("cache_dir"),
                            local_files_only=kwargs.get("local_files_only", False))
        engine = DecisionEngine(encoder=encoder, config=DecisionConfig(**config.decision_config))
        return cls(config, engine).eval()

    def forward(self, state, questions):
        return self.engine.predict(state, questions)
