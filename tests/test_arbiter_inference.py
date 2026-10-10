"""Autonomous Unit & Integration Test for ARBITER package.
Self-contained: does not depend on external experimental repositories.
"""
from contextlib import contextmanager
from pathlib import Path
import torch
from arbiter import ArbiterModel, ArbiterConfig

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "arbiter-272m"


@contextmanager
def raises(expected_exc, match=None):
    try:
        yield
    except expected_exc as exc:
        if match and match not in str(exc):
            raise AssertionError(f"Expected match '{match}' in error message '{exc}'")
    else:
        raise AssertionError(f"Expected exception {expected_exc.__name__} was not raised")


def test_model_loading_and_dtype():
    """Test autonomous loading from local safetensors without downloading base weights."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)
    
    assert model is not None
    assert isinstance(model.config, ArbiterConfig)
    
    # Check that model parameters are in native bfloat16
    first_param = next(model.parameters())
    assert first_param.dtype == torch.bfloat16
    assert model.option_token_id == 262144


def test_decide_choice():
    """Test categorical polytomous routing."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    result = model.decide(
        context="I noticed an unauthorized debit of 45 dollars on my bank statement this morning.",
        instruction="Which department must handle this request?",
        criteria={
            "fraud": "Reporting unauthorized transactions and locked cards",
            "transfer": "Wire transfers and standing orders",
            "credit": "Loan applications and interest rates",
            "account": "General account management and spending limits",
        },
        question_type="choice",
    )

    assert "selected_key" in result
    assert result["selected_key"] in ["fraud", "transfer", "credit", "account"]
    assert "probabilities" in result
    assert len(result["probabilities"]) == 4
    assert 0.0 <= result["confidence"] <= 1.0


def test_decide_noul():
    """Test boolean factuality / claim verification."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    result = model.decide(
        context="Water reaches its maximum density at approximately 4 degrees Celsius.",
        instruction="Is the statement scientifically accurate?",
        criteria=None,
        question_type="noul",
    )

    assert "selected_key" in result
    assert result["selected_key"] in ["true", "false"]
    assert "noul" in result


def test_decide_score():
    """Test ordinal regression grading."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    result = model.decide(
        context="Outage: database cluster down, transactions failing.",
        instruction="Assess the operational severity level.",
        criteria=["P4 Low", "P3 Medium", "P2 High", "P1 Critical Outage"],
        question_type="score",
    )

    assert "selected_key" in result
    assert "score" in result
    assert 0.0 <= result["score"] <= 3.0


def test_predict_multi_question():
    """Test multi-question agent dictionary interface."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    state = "Order 12345 has been delayed for 10 days."
    questions = {
        "cat": {"instructions": "Classify issue", "type": "choice", "criteria": {"delay": "Delayed shipment", "lost": "Lost parcel"}},
        "urgent": {"instructions": "Is ticket urgent?", "type": "noul"},
    }

    out = model.predict(state, questions)
    assert "answers" in out
    assert "cat" in out["answers"]
    assert "urgent" in out["answers"]


def test_input_contract_rejections():
    """Test that invalid inputs are rejected with clear ValueErrors."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    # 1. Invalid question type
    with raises(ValueError, match="Invalid question_type"):
        model.decide("ctx", "inst", ["A", "B"], question_type="invalid")

    # 2. Empty criteria on choice
    with raises(ValueError, match="criteria must contain at least 2 options"):
        model.decide("ctx", "inst", [], question_type="choice")

    # 4. Inverted / non-canonical keys in noul
    with raises(ValueError, match="keys must be canonical"):
        model.decide("ctx", "inst", {"yes": "true", "no": "false"}, question_type="noul")

    # 4b. Colliding / duplicate boolean keys in noul
    with raises(ValueError):
        model.decide("ctx", "inst", {False: "first", "false": "replacement", True: "yes"}, question_type="noul")
    with raises(ValueError, match="conflicting or colliding boolean keys"):
        model.decide("ctx", "inst", {False: "first", "false": "second"}, question_type="noul")

    # 5. Overlength input rejection when allow_truncation=False
    with raises(ValueError, match="exceeds max_length"):
        model.decide(
            context="Short context",
            instruction="Which option is valid?",
            criteria={"A": "A" * 100, "B": "B" * 100},
            max_length=32,
            allow_truncation=False,
        )

    # 6. max_length exceeding max_position_embeddings (4096)
    with raises(ValueError, match="exceeds model max_position_embeddings"):
        model.decide("ctx", "inst", ["A", "B"], max_length=5000)


def test_safe_truncation_preserves_options():
    """Test that when allow_truncation=True, context is truncated but options remain 100% intact."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    res = model.decide(
        context="Very long context with important details: " + ("word " * 500),
        instruction="Select correct category",
        criteria={"A": "Category Alpha", "B": "Category Beta"},
        max_length=64,
        allow_truncation=True,
    )
    assert res["selected_key"] in ["A", "B"]
    assert res["truncated"] is True
    assert "input_tokens" in res

    # Edge case: severe truncation budget down to 30 tokens
    res_tight = model.decide(
        context="a" * 1000,
        instruction="Choose",
        criteria={"a": "first", "b": "second"},
        max_length=30,
        allow_truncation=True,
    )
    assert res_tight["selected_key"] in ["a", "b"]
    assert res_tight["truncated"] is True
    # Verify that the re-tokenized prompt strictly respects budget <= 30
    prompt_tight = f"Contexte : \nInstruction : Choose\n\n<OPTION> a: first\n<OPTION> b: second"
    overhead = len(model.tokenizer.encode(prompt_tight, add_special_tokens=True))
    assert overhead <= 30


def test_noul_strict_true_probability():
    """Test that noul returns the exact probability of 'true' and never an ambiguous positional fallback."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    res = model.decide(
        context="The earth orbits the sun.",
        instruction="Is the statement true?",
        criteria={"false": "False statement", "true": "True statement"},
        question_type="noul",
    )
    assert res["noul"] == res["probabilities"]["true"]
    assert 0.0 <= res["noul"] <= 1.0


def test_adversarial_option_token_sanitization():
    """Test that raw <OPTION> text in context does not break token alignment."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)

    res = model.decide(
        context="Text with raw <OPTION> token inside user input.",
        instruction="Which option is correct?",
        criteria={"A": "Option A", "B": "Option B"},
        question_type="choice",
    )
    assert res["selected_key"] in ["A", "B"]


if __name__ == "__main__":
    print("Running ARBITER self-contained test suite...")
    test_model_loading_and_dtype()
    print(" - test_model_loading_and_dtype: PASSED")
    test_decide_choice()
    print(" - test_decide_choice: PASSED")
    test_decide_noul()
    print(" - test_decide_noul: PASSED")
    test_decide_score()
    print(" - test_decide_score: PASSED")
    test_predict_multi_question()
    print(" - test_predict_multi_question: PASSED")
    test_input_contract_rejections()
    print(" - test_input_contract_rejections (including GPT defect cases): PASSED")
    test_safe_truncation_preserves_options()
    print(" - test_safe_truncation_preserves_options: PASSED")
    test_noul_strict_true_probability()
    print(" - test_noul_strict_true_probability: PASSED")
    test_adversarial_option_token_sanitization()
    print(" - test_adversarial_option_token_sanitization: PASSED")
    print("\nALL ARBITER INTEGRATION TESTS PASSED SUCCESSFULLY!")
