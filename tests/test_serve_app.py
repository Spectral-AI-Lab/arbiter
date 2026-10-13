"""Inference serving tests for ARBITER.

Includes:
1. Fast dummy-model contract tests (no weights required, run on standard CPU CI)
2. Real-weight integration tests (marked with @pytest.mark.integration)
"""

from pathlib import Path
from typing import Any

import pytest
import torch

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from arbiter.modeling_arbiter import ArbiterModel
from arbiter.serve.app import create_app

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints" / "arbiter-272m"


# =========================================================================
# Part 1: Fast Dummy Model Contract Tests (CI compatible, zero GPU/weights)
# =========================================================================

class DummyTokenizer:
    pad_token_id = 0
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [1, 2, 3, 4]


class DummyArbiterModel:
    """Mock model implementing required ArbiterModel interface for serving tests."""
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def _prepare_decision(self, **kwargs) -> dict[str, Any]:
        ctx = str(kwargs.get("context", ""))
        if "2049_tokens" in ctx:
            token_count = 2049
        elif "over_budget" in ctx:
            token_count = 20000
        else:
            token_count = 4
        return {
            "ids": list(range(token_count)),
            "positions": [1, 2],
            "keys": ["a", "b"],
            "question_type": kwargs.get("question_type", "choice"),
            "raw_token_count": token_count,
            "is_truncated": False,
        }

    def decide_batch(self, requests: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        results = []
        for r in requests:
            crit = r.get("criteria")
            if isinstance(crit, dict):
                keys = list(crit.keys())
            elif isinstance(crit, list):
                keys = [str(k) for k in crit]
            else:
                keys = ["false", "true"]
            probs = {k: 1.0 / len(keys) for k in keys}
            token_count = r.get("_prepared", {}).get("raw_token_count", 12)
            res = {
                "selected_key": keys[0],
                "probabilities": probs,
                "confidence": round(1.0 / len(keys), 4),
                "input_tokens": token_count,
                "truncated": False,
            }
            if r.get("question_type") == "score":
                res["score"] = 0.5
            elif r.get("question_type") == "noul":
                res["noul"] = probs.get("true", 0.5)
            results.append(res)
        return results

    def parameters(self):
        # Fake parameter to return cpu device
        yield torch.empty(0, device="cpu")


@pytest.fixture
def dummy_client():
    """Client backed by a lightweight dummy model for contract and edge-case verification."""
    model = DummyArbiterModel()
    app = create_app(
        model=model,
        model_id="spectral-ai-lab/arbiter-272m",
        batch_window_ms=0.0,
        max_batch_size=8,
        max_queue_size=10,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_contract_arbitrary_model_id_rejected(dummy_client):
    """GPT Probe: Arbitrary requested model name must return 400."""
    payload = {
        "model": "gpt-4-turbo",  # Not served by this model
        "state": "Ticket state",
        "questions": {
            "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]}
        },
    }
    resp = dummy_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400
    assert "does not match served model" in resp.json()["detail"]


def test_contract_top_level_instructions_rejected(dummy_client):
    """vLLM rule: top-level instructions returns 400."""
    payload = {
        "instructions": "Top-level instructions forbidden",
        "state": "Ticket state",
        "questions": {
            "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]}
        },
    }
    resp = dummy_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400


def test_contract_invalid_noul_keys_rejected(dummy_client):
    """GPT Probe: invalid noul keys like apple/banana must return 400."""
    payload = {
        "state": "Ticket state",
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": "Is urgent?",
                "criteria": {"apple": "invalid", "banana": "invalid"},
            }
        },
    }
    resp = dummy_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400
    assert "Keys must be boolean identifiers" in resp.json()["detail"]


def test_contract_null_choice_descriptions_accepted(dummy_client):
    """GPT Probe: valid null choice descriptions must return 200."""
    payload = {
        "state": "Ticket state",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Pick category",
                "criteria": {"billing": None, "shipping": None},
            }
        },
    }
    resp = dummy_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert resp.json()["answers"]["q1"]["choice"] in ("billing", "shipping")


def test_contract_queue_saturation_returns_503(dummy_client):
    """GPT Probe: queue saturation must return 503 with Retry-After header."""
    # Create client with max_queue_size = 1
    tiny_model = DummyArbiterModel()
    app = create_app(
        model=tiny_model,
        model_id="spectral-ai-lab/arbiter-272m",
        batch_window_ms=10.0,
        max_queue_size=1,
    )
    with TestClient(app) as client:
        # Send 3 questions in a single request -> exceeds capacity of 1!
        payload = {
            "state": "State",
            "questions": {
                "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]},
                "q2": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]},
                "q3": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]},
            },
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 503
        assert resp.headers.get("Retry-After") == "1"


def test_contract_predict_legacy_answers(dummy_client):
    """Verify /predict returns the exact Jev / Laya legacy answers format."""
    payload = {
        "state": "Account locked",
        "questions": {
            "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]}
        },
    }
    resp = dummy_client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answers" in data
    assert "object" not in data  # Legacy envelope contains only answers
    assert "q1" in data["answers"]
    assert "choice" in data["answers"]["q1"]


def test_contract_typed_requested_on_base_rejected(dummy_client):
    """GPT Probe: When Base is served, requesting Typed must return 400."""
    payload = {
        "model": "spectral-ai-lab/arbiter-272m-typed",
        "state": "Ticket state",
        "questions": {
            "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]}
        },
    }
    resp = dummy_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400
    assert "does not match served model" in resp.json()["detail"]


def test_contract_canonical_identity_returned(dummy_client):
    """When an alias (e.g. 'arbiter') is requested, canonical model ID is returned."""
    payload = {
        "model": "arbiter",
        "state": "Ticket state",
        "questions": {
            "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]}
        },
    }
    resp = dummy_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 200
    assert resp.json()["model"] == "spectral-ai-lab/arbiter-272m"


def test_contract_predict_legacy_noul_false_true(dummy_client):
    """Verify /predict returns native false/true keys for noul questions."""
    payload = {
        "state": "Ticket state",
        "questions": {
            "q1": {
                "type": "noul",
                "instructions": "Is urgent?",
            }
        },
    }
    resp = dummy_client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answers" in data
    assert "q1" in data["answers"]
    q1 = data["answers"]["q1"]
    assert "false" in q1["probabilities"]
    assert "true" in q1["probabilities"]
    assert "yes" not in q1["probabilities"]
    assert "no" not in q1["probabilities"]
    assert q1["choice"] in ("false", "true")
    assert "noul" in q1


def test_contract_2049_tokens_accepted():
    """GPT Probe: 2049 tokens fits within max_batch_tokens (8192) and attention budget (16777216)."""
    model = DummyArbiterModel()
    app = create_app(
        model=model,
        model_id="spectral-ai-lab/arbiter-272m",
        batch_window_ms=0.0,
        max_batch_tokens=8192,
        max_attention_elements=16777216,
    )
    with TestClient(app) as client:
        payload = {
            "state": "2049_tokens context",
            "questions": {
                "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]}
            },
        }
        resp = client.post("/v1/systemone", json=payload)
        assert resp.status_code == 200
        assert resp.json()["usage"]["input_tokens"] == 2049


def test_contract_question_validation_failure_cancels_siblings(dummy_client):
    """If a question in a request fails admission, siblings are cancelled and 400 is returned."""
    payload = {
        "state": "over_budget state",
        "questions": {
            "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]},
        },
    }
    resp = dummy_client.post("/v1/systemone", json=payload)
    assert resp.status_code == 400
    assert "exceeds maximum batch tokens" in resp.json()["detail"]


def test_contract_scheduler_stop_settles_inflight_batch():
    """GPT Probe: Calling stop() settles active batches cleanly without pending futures."""
    import asyncio

    async def _test():
        model = DummyArbiterModel()
        app = create_app(model=model, batch_window_ms=5.0)
        scheduler = app.state.scheduler
        await scheduler.start()

        fut = scheduler.submit({
            "context": "Context",
            "instruction": "Inst",
            "criteria": ["A", "B"],
            "question_type": "choice",
        })

        # Stop scheduler immediately while task is queued or in flight
        await scheduler.stop(drain_timeout=2.0)

        # Future must be done (either settled with result or cancelled), NEVER pending
        assert fut.done()

    asyncio.run(_test())


# =========================================================================
# Part 2: Real-Weight Integration Tests (@pytest.mark.integration)
# =========================================================================

@pytest.fixture(scope="module")
def real_client():
    """Load real model checkpoint once for real-weight integration tests."""
    if not CHECKPOINT_DIR.exists():
        pytest.skip(f"Checkpoint directory {CHECKPOINT_DIR} not found.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ArbiterModel.from_pretrained(CHECKPOINT_DIR, device=device, local_files_only=True)
    model.eval()

    app = create_app(
        model=model,
        model_id="spectral-ai-lab/arbiter-272m",
        batch_window_ms=5.0,
        max_batch_size=8,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.integration
def test_real_health_endpoint(real_client):
    response = real_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_id"] == "spectral-ai-lab/arbiter-272m"
    assert data["loaded"] is True
    assert "version" in data


@pytest.mark.integration
def test_real_metrics_endpoint(real_client):
    response = real_client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "avg_batch_size" in data
    assert "p50_latency_ms" in data


@pytest.mark.integration
def test_real_decide_endpoint_choice(real_client):
    payload = {
        "context": "Customer complains about unexpected charges on monthly bill.",
        "instruction": "Route to the appropriate support team",
        "criteria": {
            "billing": "Invoice, pricing and payment issues",
            "technical": "Software bugs and service errors",
            "sales": "Plan upgrades and quotes",
        },
        "question_type": "choice",
    }
    response = real_client.post("/decide", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["selected_key"] == "billing"
    assert data["confidence"] > 0.5
    assert "billing" in data["probabilities"]
    assert data["input_tokens"] > 0
    assert data["truncated"] is False


@pytest.mark.integration
def test_real_decide_endpoint_noul(real_client):
    payload = {
        "context": "The package arrived 5 days ago and is completely undamaged.",
        "instruction": "Is the parcel delayed or damaged?",
        "question_type": "noul",
    }
    response = real_client.post("/decide", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["selected_key"] in ("false", "true")
    assert "noul" in data
    assert 0.0 <= data["noul"] <= 1.0


@pytest.mark.integration
def test_real_decide_endpoint_score(real_client):
    payload = {
        "context": "The product works as described, nothing special but acceptable.",
        "instruction": "Rate customer satisfaction on a 3-level scale",
        "criteria": ["dissatisfied", "neutral", "satisfied"],
        "question_type": "score",
    }
    response = real_client.post("/decide", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert 0.0 <= data["score"] <= 2.0


@pytest.mark.integration
def test_real_systemone_multi_question(real_client):
    payload = {
        "state": {
            "ticket_id": "TCK-8492",
            "body": "My card was charged twice for order #9921.",
        },
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department handles this?",
                "criteria": {
                    "billing": "Refunds and card charges",
                    "logistics": "Tracking shipments",
                    "account": "Password reset",
                },
            },
            "urgent": {
                "type": "noul",
                "instructions": "Does this require urgent financial resolution?",
                "criteria": {"true": "Money moved erroneously or unauthorized access"},
            },
            "sentiment": {
                "type": "score",
                "instructions": "Estimate customer anger level",
                "criteria": ["calm", "annoyed", "furious"],
            },
        },
    }
    response = real_client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify OpenAI-style envelope
    assert data["id"].startswith("decision-")
    assert data["object"] == "structured_decision"
    assert data["model"] == "spectral-ai-lab/arbiter-272m"
    assert data["usage"]["output_tokens"] == 0
    assert data["usage"]["input_tokens"] > 0

    answers = data["answers"]
    assert len(answers) == 3

    # Department choice
    dept = answers["department"]
    assert dept["type"] == "choice"
    assert dept["choice"] == "billing"
    assert dept["confidence"] > 0.5
    assert "billing" in dept["probabilities"]

    # Urgent noul
    urg = answers["urgent"]
    assert urg["type"] == "noul"
    assert urg["choice"] in ("yes", "no")
    assert "yes" in urg["probabilities"]
    assert "no" in urg["probabilities"]
    assert 0.0 <= urg["noul"] <= 1.0

    # Sentiment score
    sent = answers["sentiment"]
    assert sent["type"] == "score"
    assert sent["score"] is not None
    assert sent["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}

    # Diagnostics
    diag = data["diagnostics"]
    for qid in ("department", "urgent", "sentiment"):
        assert diag[qid]["label_mass"] == 1.0
        assert diag[qid]["argmax_is_label"] is True


@pytest.mark.integration
def test_real_rejections_and_error_codes(real_client):
    # 1. Reject extra fields in question with 400
    bad_q = {
        "state": "Some context",
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": ["A", "B"],
                "unsupported_field": "error",
            }
        },
    }
    assert real_client.post("/v1/systemone", json=bad_q).status_code == 400

    # 2. Reject multimodal inputs with 400
    multimodal_q = {
        "state": "Some context",
        "questions": {
            "q1": {"type": "choice", "instructions": "Pick", "criteria": ["A", "B"]}
        },
        "images": ["image_bytes"],
    }
    assert real_client.post("/v1/systemone", json=multimodal_q).status_code == 400

    # 3. Reject empty questions with 400
    empty_q = {"state": "Some context", "questions": {}}
    assert real_client.post("/v1/systemone", json=empty_q).status_code == 400

    # 4. Reject invalid question type in /decide with 400
    bad_decide = {
        "context": "context",
        "instruction": "inst",
        "criteria": ["A", "B"],
        "question_type": "invalid",
    }
    assert real_client.post("/decide", json=bad_decide).status_code == 400
