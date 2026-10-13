import pytest

pydantic = pytest.importorskip("pydantic")
from pydantic import ValidationError

from arbiter.serve.schemas import (
    DecideRequest,
    DecideResponse,
    HealthResponse,
    MetricsResponse,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
)


def test_question_valid_choice():
    q = SystemOneQuestion(
        type="choice",
        instructions="Pick team",
        criteria={"billing": "Payment issues", "tech": "Bugs"},
    )
    assert q.type == "choice"
    assert q.prompt_instruction == "Pick team"


def test_question_instruction_alias():
    q = SystemOneQuestion(
        type="choice",
        instruction="Pick team",
        criteria={"billing": "Payment issues", "tech": "Bugs"},
    )
    assert q.prompt_instruction == "Pick team"


def test_question_missing_instructions():
    with pytest.raises(ValidationError):
        SystemOneQuestion(
            type="choice",
            criteria={"billing": "Payment issues", "tech": "Bugs"},
        )


def test_question_conflicting_instructions():
    with pytest.raises(ValidationError):
        SystemOneQuestion(
            type="choice",
            instructions="Pick team A",
            instruction="Pick team B",
            criteria={"billing": "Payment issues", "tech": "Bugs"},
        )


def test_question_reject_extra_fields():
    """vLLM rule: Any question with any field other than type, instructions and criteria is rejected."""
    with pytest.raises(ValidationError):
        SystemOneQuestion(
            type="choice",
            instructions="Pick team",
            criteria={"billing": "Payment issues", "tech": "Bugs"},
            invalid_field="not_allowed",  # Extra field forbidden!
        )


def test_systemone_request_valid():
    req = SystemOneRequest(
        state={"ticket": "Charged twice"},
        questions={
            "team": SystemOneQuestion(
                type="choice",
                instructions="Which team?",
                criteria={"billing": "Payments", "tech": "Bugs"},
            ),
            "urgent": SystemOneQuestion(
                type="noul",
                instructions="Is urgent?",
            ),
        },
    )
    assert len(req.questions) == 2


def test_systemone_request_reject_multimodal():
    """Text-only model must reject images, videos, audio with 400 validation error."""
    with pytest.raises(ValidationError, match="Multimodal inputs"):
        SystemOneRequest(
            state="Some text",
            questions={
                "q1": SystemOneQuestion(type="choice", instructions="Pick", criteria=["A", "B"])
            },
            images=["base64_image_data"],
        )

    with pytest.raises(ValidationError, match="Multimodal inputs"):
        SystemOneRequest(
            state="Some text",
            questions={
                "q1": SystemOneQuestion(type="choice", instructions="Pick", criteria=["A", "B"])
            },
            videos=["video.mp4"],
        )

    with pytest.raises(ValidationError, match="Multimodal inputs"):
        SystemOneRequest(
            state="Some text",
            questions={
                "q1": SystemOneQuestion(type="choice", instructions="Pick", criteria=["A", "B"])
            },
            audio=["audio.wav"],
        )


def test_systemone_request_reject_empty_questions():
    with pytest.raises(ValidationError, match="questions.*must not be empty"):
        SystemOneRequest(state="Some text", questions={})


def test_systemone_request_reject_empty_question_id():
    with pytest.raises(ValidationError, match="Question IDs must be non-empty"):
        SystemOneRequest(
            state="Some text",
            questions={
                "": SystemOneQuestion(type="choice", instructions="Pick", criteria=["A", "B"])
            },
        )


def test_decide_request_valid():
    req = DecideRequest(
        context="User logged out unexpectedly",
        instruction="Identify issue type",
        criteria={"auth": "Authentication", "network": "Connection"},
        question_type="choice",
    )
    assert req.question_type == "choice"
    assert req.max_length == 4096


def test_decide_request_invalid_type():
    with pytest.raises(ValidationError):
        DecideRequest(
            context="Some context",
            instruction="Some instruction",
            criteria=["A", "B"],
            question_type="unsupported",
        )
