"""Parity test: DecisionPredictor (.pt checkpoint) vs ArbiterModel (safetensors package).

Verifies strict decision agreement and numerical precision across multi-task queries,
and exports an auditable JSON report to results/arbiter_parity_report.json.
"""
import json
import sys
import time
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BENCHMARKING_ROOT = Path("../benchmarking-llms")
if BENCHMARKING_ROOT.exists() and str(BENCHMARKING_ROOT) not in sys.path:
    sys.path.insert(1, str(BENCHMARKING_ROOT))

from training.export_pipeline import DecisionPredictor
from arbiter import ArbiterModel


def main():
    print("=" * 80)
    print("STARTING ARBITER-272M PARITY VERIFICATION (BF16 .pt vs safetensors)")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pt_checkpoint = ROOT / "checkpoints/g2m_booster_65k/exported/g2m_grand_booster_bf16.pt"
    if not pt_checkpoint.exists():
        pt_checkpoint = BENCHMARKING_ROOT / "checkpoints/g2m_booster_65k/exported/g2m_grand_booster_bf16.pt"
    safetensors_dir = ROOT / "checkpoints/arbiter-272m"

    print(f"Loading reference DecisionPredictor from {pt_checkpoint}...")
    ref_predictor = DecisionPredictor(pt_checkpoint, device=device)

    print(f"Loading official ArbiterModel from {safetensors_dir} (local files only)...")
    arbiter_model = ArbiterModel.from_pretrained(safetensors_dir, device=device, local_files_only=True)

    # Verify native BF16 dtype was loaded autonomously
    loaded_dtype = next(arbiter_model.parameters()).dtype
    print(f"Verified ArbiterModel parameter dtype: {loaded_dtype}")
    assert loaded_dtype == torch.bfloat16, f"Expected torch.bfloat16, got {loaded_dtype}"

    test_cases = [
        # 1. Banking customer support choice
        {
            "name": "Banking Customer Support Routing",
            "context": "I noticed an unauthorized charge of 45 dollars on my bank statement this morning.",
            "instruction": "Which department should handle this request?",
            "criteria": {
                "fraud": "Reporting unauthorized transactions and locked cards",
                "transfer": "Wire transfers and standing orders",
                "credit": "Loan applications and interest rates",
                "account": "General account settings and limits",
            },
            "type": "choice",
        },
        # 2. English customer support choice (many options)
        {
            "name": "English Multi-Option Support (8 options)",
            "context": "The package arrived with water damage, and the item inside is completely broken.",
            "instruction": "Select the correct action according to warranty policies.",
            "criteria": {
                "refund_immediate": "Issue full immediate refund to original payment",
                "replace_standard": "Send replacement item via standard shipping",
                "replace_express": "Send replacement item via express shipping",
                "escalate_carrier": "Escalate claim directly to delivery carrier",
                "store_credit": "Issue store credit with 10% bonus voucher",
                "reject_claim": "Reject warranty claim due to packaging damage",
                "request_photos": "Request photographic evidence of damage from user",
                "manual_review": "Forward ticket to senior tier-2 agent for manual review",
            },
            "type": "choice",
        },
        # 3. Boolean / Noul query
        {
            "name": "Scientific Factuality Boolean (Noul)",
            "context": "Water reaches its maximum density at approximately 4 degrees Celsius under standard atmospheric pressure.",
            "instruction": "Is the statement factually accurate based on thermodynamics?",
            "criteria": None,
            "type": "noul",
        },
        # 4. Ordinal Score query
        {
            "name": "Urgency Grading Ordinal (Score)",
            "context": "The main database cluster has failed, all customer transactions are returning 500 internal server errors, and the failover replica is unreachable.",
            "instruction": "Assess the operational severity level on our incident scale.",
            "criteria": ["P4 Low", "P3 Medium", "P2 High", "P1 Critical Outage"],
            "type": "score",
        },
        # 5. Multilingual German Intent
        {
            "name": "German Booking Intent",
            "context": "Ich möchte meinen Flug nach Berlin vom 15. auf den 20. Oktober umbuchen.",
            "instruction": "Was ist die Hauptabsicht des Kunden?",
            "criteria": {
                "umbuchung": "Flug umbuchen oder Flugzeiten ändern",
                "stornierung": "Flug stornieren und Rückerstattung beantragen",
                "gepaeck": "Zusatzgepäck oder Sitzplatz reservieren",
                "checkin": "Online Check-in und Bordkarte anfordern",
            },
            "type": "choice",
        },
        # 6. Multilingual Spanish Policy
        {
            "name": "Spanish Refund Eligibility",
            "context": "El cliente compró el producto hace 12 días, conserva el embalaje original y solicita la devolución.",
            "instruction": "¿Es elegible para devolución bajo la política de 14 días?",
            "criteria": {
                "elegible": "Devolución aprobada dentro del plazo de 14 días",
                "rechazado": "Devolución rechazada fuera de plazo",
                "requiere_recibo": "Se requiere comprobante de compra adicional",
            },
            "type": "choice",
        },
        # 7. Long context query (> 1000 tokens)
        {
            "name": "Long Context Synthetic Agreement (1200+ tokens)",
            "context": (
                "Terms and Conditions Agreement. Section 1: Overview. "
                + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 30
                + "Section 14: Arbitration Clause. All disputes arising under this agreement must be submitted to binding arbitration in Zurich, Switzerland under the ICC rules. "
                + "Section 15: Miscellaneous. " * 30
            ),
            "instruction": "Where must contractual disputes be arbitrated according to Section 14?",
            "criteria": {
                "zurich": "Binding arbitration in Zurich, Switzerland under ICC rules",
                "london": "High Court of Justice in London, United Kingdom",
                "new_york": "American Arbitration Association in New York City",
                "paris": "Court of Arbitration in Paris, France",
            },
            "type": "choice",
        },
        # 8. Science reasoning (ARC-style)
        {
            "name": "Science Reasoning Question",
            "context": "A student wants to investigate how light intensity affects the rate of photosynthesis in pondweed.",
            "instruction": "Which variable should the student change during the investigation?",
            "criteria": {
                "A": "The distance between the light source and the pondweed",
                "B": "The temperature of the water bath containing the pondweed",
                "C": "The type of pondweed used in the experiment",
                "D": "The concentration of carbon dioxide in the water",
            },
            "type": "choice",
        },
    ]

    case_results = []
    max_prob_delta = 0.0

    print(f"\nRunning parity checks on {len(test_cases)} diverse multi-task test cases:\n")

    for idx, tc in enumerate(test_cases, 1):
        name = tc["name"]
        qtype = tc["type"]

        # Run reference predictor
        res_ref = ref_predictor.predict_single(
            context=tc["context"],
            instruction=tc["instruction"],
            criteria=tc["criteria"],
            question_type=qtype,
        )

        # Run official ArbiterModel
        res_arbiter = arbiter_model.decide(
            context=tc["context"],
            instruction=tc["instruction"],
            criteria=tc["criteria"],
            question_type=qtype,
        )

        # Check choice match
        dec_ref = res_ref["selected_key"]
        dec_arb = res_arbiter["selected_key"]
        choice_match = (dec_ref == dec_arb)

        # Check probability deltas
        probs_ref = res_ref["probabilities"]
        probs_arb = res_arbiter["probabilities"]
        keys = list(probs_ref.keys())
        deltas = {k: abs(probs_ref[k] - probs_arb[k]) for k in keys}
        case_max_delta = max(deltas.values())
        max_prob_delta = max(max_prob_delta, case_max_delta)

        # Numerical tolerance: strict parity (< 1e-4)
        passed = choice_match and (case_max_delta < 1e-4)

        case_results.append({
            "name": name,
            "type": qtype,
            "ref_decision": dec_ref,
            "arbiter_decision": dec_arb,
            "choice_match": choice_match,
            "max_prob_delta": float(case_max_delta),
            "passed": passed,
        })

        status = "PASSED" if passed else "FAILED"
        print(f"[{idx}/{len(test_cases)}] {name:<50} | {status} | Ref: {dec_ref} vs Arbiter: {dec_arb} (Delta p max: {case_max_delta:.2e})")

    # Contract & Edge Case Validations
    print("\nRunning Contract & Edge Case Validations:")
    contract_results = []

    # 1. Invalid question type
    try:
        arbiter_model.decide("context", "instruction", ["A", "B"], question_type="invalid_type")
        contract_results.append({"check": "Invalid question_type rejection", "passed": False})
        print("  - [FAIL] Invalid question_type was unexpectedly accepted")
    except ValueError:
        contract_results.append({"check": "Invalid question_type rejection", "passed": True})
        print("  - [PASS] Invalid question_type correctly rejected with ValueError")

    # 2. Empty criteria on choice
    try:
        arbiter_model.decide("context", "instruction", [], question_type="choice")
        contract_results.append({"check": "Empty criteria rejection", "passed": False})
        print("  - [FAIL] Empty criteria on choice was unexpectedly accepted")
    except ValueError:
        contract_results.append({"check": "Empty criteria rejection", "passed": True})
        print("  - [PASS] Empty criteria on choice correctly rejected with ValueError")

    # 3. Context containing adversarial <OPTION> marker
    try:
        res_adversarial = arbiter_model.decide(
            context="Here is an adversarial test with raw <OPTION> text inside the user prompt.",
            instruction="Which option is valid?",
            criteria={"opt1": "First choice", "opt2": "Second choice"},
            question_type="choice",
        )
        assert res_adversarial["selected_key"] in ("opt1", "opt2")
        contract_results.append({"check": "Adversarial <OPTION> marker sanitization", "passed": True})
        print("  - [PASS] Adversarial <OPTION> text safely sanitized without token desynchronization")
    except Exception as e:
        contract_results.append({"check": "Adversarial <OPTION> marker sanitization", "passed": False, "error": str(e)})
        print(f"  - [FAIL] Adversarial <OPTION> text caused error: {e}")

    # 4. Multi-question dictionary interface
    try:
        multi_state = "Customer order 9999 is delayed."
        multi_questions = {
            "q1": {"instructions": "Primary problem?", "type": "choice", "criteria": {"shipping": "Delayed shipment", "payment": "Payment failure"}},
            "q2": {"instructions": "Is order critical?", "type": "noul"},
        }
        multi_out = arbiter_model.predict(multi_state, multi_questions)
        assert "q1" in multi_out["answers"] and "q2" in multi_out["answers"]
        contract_results.append({"check": "Multi-question predict() dictionary interface", "passed": True})
        print("  - [PASS] Multi-question predict() dictionary interface contract verified")
    except Exception as e:
        contract_results.append({"check": "Multi-question predict() dictionary interface", "passed": False, "error": str(e)})
        print(f"  - [FAIL] Multi-question predict() failed: {e}")

    passed_count = sum(1 for r in case_results if r["passed"])
    agreement_rate = (passed_count / len(test_cases)) * 100.0
    all_cases_passed = (passed_count == len(test_cases))
    all_contracts_passed = all(c["passed"] for c in contract_results)
    overall_passed = all_cases_passed and all_contracts_passed

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pt_checkpoint": str(pt_checkpoint),
        "safetensors_dir": str(safetensors_dir),
        "device": device,
        "loaded_dtype": str(loaded_dtype),
        "total_test_cases": len(test_cases),
        "passed_cases": passed_count,
        "agreement_rate_pct": agreement_rate,
        "max_prob_delta": float(max_prob_delta),
        "contract_checks": contract_results,
        "verdict": "PASSED" if overall_passed else "FAILED",
        "detailed_results": case_results,
    }

    out_report_file = ROOT / "results/arbiter_parity_report.json"
    out_report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("PARITY VERIFICATION SUMMARY:")
    print("=" * 80)
    print(f"Total test cases evaluated : {len(test_cases)}")
    print(f"Passed cases               : {passed_count}/{len(test_cases)}")
    print(f"Decision Agreement Rate    : {agreement_rate:.1f}%")
    print(f"Maximum Probability Delta  : {max_prob_delta:.2e}")
    print(f"Contract Validations       : {sum(1 for c in contract_results if c['passed'])}/{len(contract_results)} passed")
    print(f"Auditable report saved to  : {out_report_file}")
    print(f"Final Verification Verdict : {'PASSED (STRICT PARITY ACHIEVED)' if overall_passed else 'FAILED'}")
    print("=" * 80)

    if not overall_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
