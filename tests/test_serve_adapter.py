import pytest

pydantic = pytest.importorskip("pydantic")

from arbiter.serve.adapter import (
    assemble_systemone_response,
    build_systemone_answer,
    canonicalize_noul_criteria,
    prepare_question_task,
    serialize_state,
)
from arbiter.serve.schemas import (
    SystemOneQuestion,
    SystemOneRequest,
)


def test_serialize_state():
    assert serialize_state("plain text") == "plain text"
    # Dict serialized deterministically with sorted keys
    dict_state = {"b": 2, "a": 1}
    assert serialize_state(dict_state) == '{"a": 1, "b": 2}'
    assert serialize_state([1, 2, 3]) == "[1, 2, 3]"


def test_canonicalize_noul_criteria():
    # None
    c_none = canonicalize_noul_criteria(None)
    assert "false" in c_none and "true" in c_none

    # Already canonical
    c_canon = canonicalize_noul_criteria({"false": "no way", "true": "absolutely"})
    assert c_canon == {"false": "no way", "true": "absolutely"}

    # yes/no keys
    c_yn = canonicalize_noul_criteria({"no": "refused", "yes": "accepted"})
    assert c_yn == {"false": "refused", "true": "accepted"}

    # partial true only
    c_part = canonicalize_noul_criteria({"true": "money moving"})
    assert c_part["true"] == "money moving"
    assert "false" in c_part


def test_build_systemone_answer_choice():
    q = SystemOneQuestion(
        type="choice",
        instructions="Pick category",
        criteria={"billing": "Payment", "shipping": "Delivery"},
    )
    raw = {
        "selected_key": "billing",
        "probabilities": {"billing": 0.95, "shipping": 0.05},
        "confidence": 0.95,
    }
    ans = build_systemone_answer(q, raw)
    assert ans.type == "choice"
    assert ans.choice == "billing"
    assert ans.confidence == 0.95
    assert ans.probabilities["billing"] == 0.95


def test_build_systemone_answer_singleton_choice():
    q = SystemOneQuestion(
        type="choice",
        instructions="Pick category",
        criteria={"single_option": "The only choice"},
    )
    raw = {
        "_is_singleton_choice": True,
        "_singleton_key": "single_option",
    }
    ans = build_systemone_answer(q, raw)
    assert ans.type == "choice"
    assert ans.choice == "single_option"
    assert ans.confidence == 1.0
    assert ans.probabilities["single_option"] == 1.0


def test_build_systemone_answer_noul():
    q = SystemOneQuestion(
        type="noul",
        instructions="Is ticket urgent?",
    )
    raw = {
        "selected_key": "true",
        "probabilities": {"false": 0.15, "true": 0.85},
        "confidence": 0.85,
        "noul": 0.85,
    }
    ans = build_systemone_answer(q, raw)
    assert ans.type == "noul"
    assert ans.choice == "yes"
    assert ans.noul == 0.85
    assert ans.confidence == 0.85
    assert ans.probabilities["yes"] == 0.85
    assert ans.probabilities["no"] == 0.15


def test_build_systemone_answer_score():
    q = SystemOneQuestion(
        type="score",
        instructions="Rate frustration",
        criteria=["calm", "annoyed", "furious"],
    )
    raw = {
        "selected_key": "1",
        "score": 1.25,
        "probabilities": {"0": 0.1, "1": 0.55, "2": 0.35},
        "confidence": 0.55,
    }
    ans = build_systemone_answer(q, raw)
    assert ans.type == "score"
    assert ans.choice == "1"
    assert ans.score == 1.25
    assert ans.confidence == 0.55
    assert ans.legend == {"0": "calm", "1": "annoyed", "2": "furious"}
    assert ans.probabilities["1"] == 0.55


def test_assemble_systemone_response():
    req = SystemOneRequest(
        state="Test state",
        questions={
            "q1": SystemOneQuestion(type="choice", instructions="Pick", criteria=["A", "B"])
        },
    )
    q = req.questions["q1"]
    raw = {
        "selected_key": "A",
        "probabilities": {"A": 0.8, "B": 0.2},
        "confidence": 0.8,
    }
    ans = build_systemone_answer(q, raw)
    resp = assemble_systemone_response(
        req,
        model_name="spectral-ai-lab/arbiter-272m",
        answers_by_qid={"q1": ans},
        token_counts_by_qid={"q1": 42},
    )
    assert resp.id.startswith("decision-")
    assert resp.object == "structured_decision"
    assert resp.model == "spectral-ai-lab/arbiter-272m"
    assert resp.usage.input_tokens == 42
    assert resp.usage.output_tokens == 0
    assert resp.diagnostics["q1"].label_mass == 1.0
    assert resp.diagnostics["q1"].argmax_is_label is True
