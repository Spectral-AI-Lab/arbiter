"""Adapter layer translating between vLLM Structured Decisions specification
and ARBITER internal decision execution format.
"""

import json
import time
import uuid
from typing import Any, Optional, Union

from .schemas import (
    SystemOneAnswer,
    SystemOneDiagnostic,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
    SystemOneUsage,
)


def serialize_state(state: Union[str, dict[str, Any], list[Any]]) -> str:
    """Serialize a state context into a deterministic string representation."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def canonicalize_noul_criteria(
    criteria: Optional[Union[dict[str, Optional[str]], list[str], None]]
) -> dict[str, str]:
    """Convert flexible noul criteria into ARBITER canonical {"false": ..., "true": ...}."""
    if criteria is None:
        return {"false": "The statement is false.", "true": "The statement is true."}

    if isinstance(criteria, dict):
        lower_keys = {str(k).strip().lower(): v for k, v in criteria.items()}
        valid_bool_keys = {"true", "false", "yes", "no", "1", "0"}
        invalid_keys = set(lower_keys.keys()) - valid_bool_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid criteria keys for noul question: {list(criteria.keys())}. "
                "Keys must be boolean identifiers ('true'/'false', 'yes'/'no', or partial 'true'/'false')."
            )

        # Case 1: already {"false": ..., "true": ...}
        if "false" in lower_keys and "true" in lower_keys:
            return {"false": str(lower_keys["false"] or "False"), "true": str(lower_keys["true"] or "True")}
        # Case 2: {"no": ..., "yes": ...}
        if "no" in lower_keys and "yes" in lower_keys:
            return {"false": str(lower_keys["no"] or "No"), "true": str(lower_keys["yes"] or "Yes")}
        # Case 3: partial criteria with only {"true": "..."}
        if "true" in lower_keys:
            t_val = lower_keys["true"] or "True condition met"
            return {"false": f"Not ({t_val})", "true": str(t_val)}
        # Case 4: partial criteria with only {"false": "..."}
        if "false" in lower_keys:
            f_val = lower_keys["false"] or "False condition"
            return {"false": str(f_val), "true": f"Not ({f_val})"}
        # Case 5: partial {"yes": "..."}
        if "yes" in lower_keys:
            t_val = lower_keys["yes"] or "Yes"
            return {"false": f"Not ({t_val})", "true": str(t_val)}
        # Case 6: partial {"no": "..."}
        if "no" in lower_keys:
            f_val = lower_keys["no"] or "No"
            return {"false": str(f_val), "true": f"Not ({f_val})"}

    if isinstance(criteria, list):
        if len(criteria) == 2:
            return {"false": str(criteria[0]), "true": str(criteria[1])}
        raise ValueError(f"List criteria for noul must have exactly 2 options, got {len(criteria)}.")

    raise ValueError(f"Invalid criteria format for noul: {type(criteria)}")


def prepare_question_task(
    state_str: str,
    qid: str,
    question: SystemOneQuestion,
    max_length: int = 4096,
    allow_truncation: bool = False,
) -> dict[str, Any]:
    """Transform one SystemOneQuestion into an ARBITER internal decision task dict."""
    instruction = question.prompt_instruction
    qtype = question.type
    criteria = question.criteria

    if qtype == "noul":
        norm_criteria = canonicalize_noul_criteria(criteria)
        return dict(
            context=state_str,
            instruction=instruction,
            criteria=norm_criteria,
            question_type="noul",
            max_length=max_length,
            allow_truncation=allow_truncation,
            _qid=qid,
            _is_singleton_choice=False,
        )

    if qtype == "choice":
        # Normalize null descriptions in criteria dict
        if isinstance(criteria, dict):
            criteria = {str(k): (str(v) if v is not None else str(k)) for k, v in criteria.items()}

        # Check for singleton option (allowed in vLLM spec)
        if isinstance(criteria, dict) and len(criteria) == 1:
            key = next(iter(criteria.keys()))
            return dict(
                context=state_str,
                instruction=instruction,
                criteria=criteria,
                question_type="choice",
                max_length=max_length,
                allow_truncation=allow_truncation,
                _qid=qid,
                _is_singleton_choice=True,
                _singleton_key=key,
            )
        if isinstance(criteria, list) and len(criteria) == 1:
            return dict(
                context=state_str,
                instruction=instruction,
                criteria=criteria,
                question_type="choice",
                max_length=max_length,
                allow_truncation=allow_truncation,
                _qid=qid,
                _is_singleton_choice=True,
                _singleton_key=str(criteria[0]),
            )
        return dict(
            context=state_str,
            instruction=instruction,
            criteria=criteria,
            question_type="choice",
            max_length=max_length,
            allow_truncation=allow_truncation,
            _qid=qid,
            _is_singleton_choice=False,
        )

    if qtype == "score":
        return dict(
            context=state_str,
            instruction=instruction,
            criteria=criteria,
            question_type="score",
            max_length=max_length,
            allow_truncation=allow_truncation,
            _qid=qid,
            _is_singleton_choice=False,
        )

    raise ValueError(f"Unsupported question type: {qtype}")


def build_systemone_answer(
    question: SystemOneQuestion,
    raw_result: dict[str, Any],
) -> SystemOneAnswer:
    """Format an ARBITER raw decision result into a standardized SystemOneAnswer."""
    qtype = question.type

    if qtype == "choice":
        # Check if singleton bypass
        if raw_result.get("_is_singleton_choice"):
            singleton_key = raw_result.get("_singleton_key", "A")
            return SystemOneAnswer(
                type="choice",
                choice=singleton_key,
                probabilities={singleton_key: 1.0},
                confidence=1.0,
            )
        selected_key = raw_result["selected_key"]
        probs = {str(k): round(float(v), 6) for k, v in raw_result["probabilities"].items()}
        conf = round(float(raw_result["confidence"]), 6)
        return SystemOneAnswer(
            type="choice",
            choice=selected_key,
            probabilities=probs,
            confidence=conf,
        )

    if qtype == "noul":
        probs = raw_result["probabilities"]
        p_true = float(probs.get("true", 0.0))
        p_false = float(probs.get("false", 1.0 - p_true))
        # Normalize in case of rounding
        total = p_true + p_false
        if total > 0:
            p_true /= total
            p_false /= total

        choice_str = "yes" if p_true >= p_false else "no"
        conf = max(p_true, p_false)
        return SystemOneAnswer(
            type="noul",
            choice=choice_str,
            noul=round(p_true, 6),
            probabilities={"yes": round(p_true, 6), "no": round(p_false, 6)},
            confidence=round(conf, 6),
        )

    if qtype == "score":
        # Build legend
        criteria = question.criteria
        raw_probs = raw_result["probabilities"]
        if isinstance(criteria, list):
            legend = {str(i): str(desc) for i, desc in enumerate(criteria)}
        elif isinstance(criteria, dict):
            legend = {str(k): str(v) for k, v in criteria.items()}
        else:
            legend = {str(k): str(k) for k in raw_probs.keys()}

        probs = {str(k): round(float(v), 6) for k, v in raw_probs.items()}
        conf = round(float(raw_result["confidence"]), 6)
        score_val = round(float(raw_result.get("score", 0.0)), 4)
        choice_key = str(raw_result["selected_key"])
        return SystemOneAnswer(
            type="score",
            choice=choice_key,
            score=score_val,
            legend=legend,
            probabilities=probs,
            confidence=conf,
        )

    raise ValueError(f"Unknown question type: {qtype}")


def assemble_systemone_response(
    request: SystemOneRequest,
    model_name: str,
    answers_by_qid: dict[str, SystemOneAnswer],
    token_counts_by_qid: dict[str, int],
) -> SystemOneResponse:
    """Assemble final SystemOneResponse matching vLLM format exactly."""
    total_input_tokens = sum(token_counts_by_qid.values())
    diagnostics = {
        qid: SystemOneDiagnostic(label_mass=1.0, argmax_is_label=True)
        for qid in answers_by_qid
    }
    return SystemOneResponse(
        id=f"decision-{uuid.uuid4().hex[:24]}",
        object="structured_decision",
        created=int(time.time()),
        model=model_name,
        answers=answers_by_qid,
        usage=SystemOneUsage(input_tokens=total_input_tokens, output_tokens=0),
        diagnostics=diagnostics,
    )


def build_legacy_predict_answer(
    question: SystemOneQuestion,
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """Format decision result for native model.predict / Jev / Laya legacy contract.
    
    Preserves native boolean identifiers ('false' and 'true') instead of 'yes'/'no'.
    """
    sel_key = str(raw_result.get("selected_key", ""))
    raw_probs = {str(k): round(float(v), 6) for k, v in raw_result.get("probabilities", {}).items()}
    conf = round(float(raw_result.get("confidence", 1.0)), 6)

    ans: dict[str, Any] = {
        "choice": sel_key,
        "selected_key": sel_key,
        "probabilities": raw_probs,
        "confidence": conf,
        "input_tokens": raw_result.get("input_tokens", 0),
        "truncated": raw_result.get("truncated", False),
    }

    if question.type == "noul":
        p_true = float(raw_result.get("noul", raw_probs.get("true", 0.5)))
        ans["noul"] = round(p_true, 6)
    elif question.type == "score":
        ans["score"] = round(float(raw_result.get("score", 0.0)), 4)

    return ans
