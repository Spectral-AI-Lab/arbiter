from transformers import Pipeline


class DecisionPipeline(Pipeline):
    """State/questions in, Jev-compatible typed answers out."""
    _load_tokenizer = False
    _load_feature_extractor = False
    _load_image_processor = False
    _load_processor = False

    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, device=device, **kwargs)
        # Pipeline moves the registered SentenceTransformer to the requested device.
        self.model.engine.encoder.device = str(self.model.device)
        self.model.engine.encoder.dtype = str(self.model.dtype).removeprefix("torch.")

    def _sanitize_parameters(self, **kwargs):
        if kwargs:
            raise ValueError(f"Unsupported pipeline arguments: {sorted(kwargs)}")
        return {}, {}, {}

    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or set(inputs) != {"state", "questions"}:
            raise ValueError("Expected an object containing exactly state and questions")
        return inputs

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        return model_outputs
