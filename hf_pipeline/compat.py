"""Common Jev/Laya JSON decision contract (no HTTP server or router extensions)."""
import json


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)


def predict(engine, state, questions):
    if state is None or not isinstance(questions, dict) or not questions:
        raise ValueError("Provide a non-null state and a nonempty questions object")
    state = text(state)
    prepared = []
    for key, question in questions.items():
        if not isinstance(key, str) or not isinstance(question, dict):
            raise ValueError("Questions must map string IDs to objects")
        unknown = set(question) - {"type", "instructions", "criteria"}
        if unknown:
            raise ValueError(f"Unsupported question fields: {sorted(unknown)}")
        kind = question.get("type")
        instructions = text(question.get("instructions", ""))
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError("choice criteria must contain 2 to 255 options")
            if not all(isinstance(label, str) for label in criteria):
                raise ValueError("Option labels must be strings")
            options = {label: label if desc is None else text(desc) for label, desc in criteria.items()}
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("score criteria must contain 2 to 10 levels")
            options = {str(i): text(desc) for i, desc in enumerate(criteria)}
        elif kind == "noul":
            if criteria is None:
                criteria = {}
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise ValueError("noul criteria accepts only true and false")
            if not instructions.strip():
                raise ValueError("noul requires instructions")
            options = {
                "false": text(criteria["false"]) if criteria.get("false") is not None else f"The following condition does not hold: {instructions}",
                "true": text(criteria["true"]) if criteria.get("true") is not None else f"The following condition holds: {instructions}",
            }
        else:
            raise ValueError(f"Unknown question type: {kind}")
        prepared.append((key, kind, instructions, options))

    answers = {}
    input_tokens = 0
    for key, kind, instructions, options in prepared:
        result = engine.choice(state, options, instructions=instructions)
        probs = result["probabilities"]
        if kind == "noul":
            answer = {"type": "noul", "noul": probs["true"]}
        else:
            n = len(probs)
            # Jev's normalized maximum, not Laya's entropy confidence.
            confidence = max(0.0, min(1.0, (n * max(probs.values()) - 1) / (n - 1)))
            answer = {"type": kind, "probabilities": probs, "confidence": confidence}
            if kind == "choice":
                answer["choice"] = result["selected_key"]
            else:
                answer["score"] = sum(int(i) * p for i, p in probs.items())
                answer["legend"] = options
        answers[key] = answer
        # Logical embedding input volume, including candidates even when cached.
        inputs = list(options.values()) + engine.encoder.chunks(
            state, engine.config.chunk_tokens, engine.config.chunk_overlap)
        if engine.config.normalize_prior:
            inputs.append("")
        input_tokens += sum(len(engine.encoder.tokenizer.encode(
            engine.encoder.format_text(s, instructions), add_special_tokens=True)) for s in inputs)
    return {"model": "google/embeddinggemma-2", "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}
