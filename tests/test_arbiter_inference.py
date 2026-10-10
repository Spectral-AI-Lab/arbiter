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
        context="J'ai un prélèvement suspect de 45 euros sur mon relevé bancaire ce matin.",
        instruction="Quel département doit traiter cette demande ?",
        criteria={
            "fraude": "Signalement d'opérations non autorisées et cartes bloquées",
            "virement": "Ordres de virement et virements permanents",
            "credit": "Demandes d'emprunt et taux d'intérêt",
            "compte": "Gestion générale du compte et plafonds",
        },
        question_type="choice",
    )

    assert "selected_key" in result
    assert result["selected_key"] in ["fraude", "virement", "credit", "compte"]
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

    # 3. Empty context
    with raises(ValueError, match="Context cannot be empty"):
        model.decide("  ", "inst", ["A", "B"], question_type="choice")


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
    print(" - test_input_contract_rejections: PASSED")
    test_adversarial_option_token_sanitization()
    print(" - test_adversarial_option_token_sanitization: PASSED")
    print("\nALL ARBITER INTEGRATION TESTS PASSED SUCCESSFULLY!")
