"""
Unit tests for Rubric v0 metric calculations, tie-break rules, and schema conformance.
"""

import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator

from eval.score_reports import (
    AtomicClaim,
    MetricScores,
    calculate_metrics,
    apply_tie_break,
    evaluate_report,
)

def test_metric_calculations_standard():
    """
    Test a known 10-claim distribution:
      4 FULL, 2 PARTIAL, 2 UNSUPPORTED, 1 CONTRADICTED, 1 FABRICATED
    Expected metrics:
      Groundedness = (4 + 2) / 10 = 0.60
      Faithfulness = 4 / 10 = 0.40
      Fabrication Rate = 1 / 10 = 0.10
      Hallucination Rate = (2 + 1) / 10 = 0.30
    """
    claims = [
        AtomicClaim(claim_id=1, text="c1", label="FULL", justification=""),
        AtomicClaim(claim_id=2, text="c2", label="FULL", justification=""),
        AtomicClaim(claim_id=3, text="c3", label="FULL", justification=""),
        AtomicClaim(claim_id=4, text="c4", label="FULL", justification=""),
        AtomicClaim(claim_id=5, text="c5", label="PARTIAL", justification=""),
        AtomicClaim(claim_id=6, text="c6", label="PARTIAL", justification=""),
        AtomicClaim(claim_id=7, text="c7", label="UNSUPPORTED", justification=""),
        AtomicClaim(claim_id=8, text="c8", label="UNSUPPORTED", justification=""),
        AtomicClaim(claim_id=9, text="c9", label="CONTRADICTED", justification=""),
        AtomicClaim(claim_id=10, text="c10", label="FABRICATED", justification=""),
    ]
    metrics = calculate_metrics(claims)

    assert metrics.total_claims == 10
    assert metrics.full_count == 4
    assert metrics.partial_count == 2
    assert metrics.unsupported_count == 2
    assert metrics.contradicted_count == 1
    assert metrics.fabricated_count == 1

    assert pytest.approx(metrics.groundedness, 1e-4) == 0.60
    assert pytest.approx(metrics.faithfulness, 1e-4) == 0.40
    assert pytest.approx(metrics.fabrication_rate, 1e-4) == 0.10
    assert pytest.approx(metrics.hallucination_rate, 1e-4) == 0.30

def test_metric_calculations_zero_claims():
    """
    Verify zero-division safety returns 0.0 across all metrics when claims list is empty.
    """
    metrics = calculate_metrics([])
    assert metrics.total_claims == 0
    assert metrics.groundedness == 0.0
    assert metrics.faithfulness == 0.0
    assert metrics.fabrication_rate == 0.0
    assert metrics.hallucination_rate == 0.0

def test_completeness_identity():
    """
    Verify that Faithfulness + (PARTIAL/N) + Fabrication + Hallucination sums to 1.0.
    """
    claims = [
        AtomicClaim(claim_id=1, text="c1", label="FULL", justification=""),
        AtomicClaim(claim_id=2, text="c2", label="PARTIAL", justification=""),
        AtomicClaim(claim_id=3, text="c3", label="UNSUPPORTED", justification=""),
        AtomicClaim(claim_id=4, text="c4", label="FABRICATED", justification=""),
        AtomicClaim(claim_id=5, text="c5", label="CONTRADICTED", justification=""),
    ]
    metrics = calculate_metrics(claims)
    sum_identity = (
        metrics.faithfulness
        + (metrics.partial_count / metrics.total_claims)
        + metrics.fabrication_rate
        + metrics.hallucination_rate
    )
    assert pytest.approx(sum_identity, 1e-4) == 1.0

def test_tie_break_order():
    """
    Test edge cases in apply_tie_break to ensure priority rule enforcement:
    FABRICATED > CONTRADICTED > UNSUPPORTED > FULL / PARTIAL
    """
    # Case 1: Both fabricated and contradicted -> FABRICATED takes precedence
    assert apply_tie_break(is_fabricated=True, is_contradicted=True, is_unsupported=True, is_full=True) == "FABRICATED"
    
    # Case 2: Contradicted and unsupported -> CONTRADICTED takes precedence
    assert apply_tie_break(is_fabricated=False, is_contradicted=True, is_unsupported=True, is_full=True) == "CONTRADICTED"
    
    # Case 3: Unsupported and full -> UNSUPPORTED takes precedence
    assert apply_tie_break(is_fabricated=False, is_contradicted=False, is_unsupported=True, is_full=True) == "UNSUPPORTED"
    
    # Case 4: Not unsupported, is_full True -> FULL
    assert apply_tie_break(is_fabricated=False, is_contradicted=False, is_unsupported=False, is_full=True) == "FULL"
    
    # Case 5: Not unsupported, is_full False -> PARTIAL
    assert apply_tie_break(is_fabricated=False, is_contradicted=False, is_unsupported=False, is_full=False) == "PARTIAL"

def test_schema_validation():
    """
    Validate all JSON files in schema/examples/ against schema/alert_schema_v1.json.
    """
    base_dir = Path(__file__).resolve().parent.parent
    schema_path = base_dir / "schema" / "alert_schema_v1.json"
    examples_dir = base_dir / "schema" / "examples"

    assert schema_path.exists(), f"Missing schema at {schema_path}"
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    example_files = list(examples_dir.glob("*.json"))
    assert len(example_files) >= 2, "Expected at least 2 example JSON files"

    for example_file in example_files:
        with open(example_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        errors = list(validator.iter_errors(data))
        assert len(errors) == 0, f"Validation failed for {example_file.name}: {errors}"

def test_evaluate_report_mock():
    """
    Verify that evaluate_report operates in mock mode without API client.
    """
    evaluation = evaluate_report("Sample evidence", "Sample report")
    assert len(evaluation.claims) == 6
    assert evaluation.metrics.total_claims == 6
    assert evaluation.metrics.groundedness > 0.0
