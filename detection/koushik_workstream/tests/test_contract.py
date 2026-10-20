"""Tests for validating generated alert payloads against contracts/alert_schema_v1.json.
"""

from __future__ import annotations

import copy
import pytest

from src.validate_alerts import load_validator, validate_alert_payload, find_schema_path


@pytest.fixture
def valid_alert_payload():
    """Canonical valid alert payload conforming strictly to AlertSchemaV1."""
    return {
        "identity": {
            "alert_id": "ALT-20261020-0001",
            "flow_id": "FLW-20261020-000001",
            "detector_family": "traditional_ml",
            "timestamp": "2026-10-20T12:00:00Z",
            "dataset": "Reference-Synthetic-Network",
            "split": "pilot",
        },
        "prediction": {
            "predicted_label": "DDoS_UDP",
            "confidence": 0.9421,
            "base_learner_votes": [
                {"learner_name": "RandomForest", "voted_label": "DDoS_UDP", "learner_confidence": 0.9200},
                {"learner_name": "XGBoost", "voted_label": "DDoS_UDP", "learner_confidence": 0.9650},
                {"learner_name": "MLP", "voted_label": "Benign", "learner_confidence": 0.5100},
            ],
        },
        "shap_attribution": {
            "explainer_type": "TreeExplainer",
            "explained_class": "DDoS_UDP",
            "base_value": 0.1250,
            "top_features": [
                {"rank": i + 1, "feature_name": f"feature_{i}", "feature_value": float(i * 1.5), "shap_value": float(1.0 / (i + 1))}
                for i in range(10)
            ],
        },
        "evaluation_metadata": {
            "true_label": "DDoS_UDP",
            "in_train_leakage_flag": False,
            "detector_correct": True,
        },
    }


def test_valid_alert_passes_schema(valid_alert_payload):
    validator = load_validator()
    is_valid, errors = validate_alert_payload(valid_alert_payload, validator)
    assert is_valid is True, f"Validation failed: {errors}"
    assert len(errors) == 0


def test_invalid_alert_id_format(valid_alert_payload):
    validator = load_validator()
    bad_alert = copy.deepcopy(valid_alert_payload)
    bad_alert["identity"]["alert_id"] = "KOU-PILOT-001"  # Invalid format

    is_valid, errors = validate_alert_payload(bad_alert, validator)
    assert is_valid is False
    assert any("does not match '^ALT-[0-9]{8}-[0-9]{4}$'" in e for e in errors)


def test_invalid_detector_family_enum(valid_alert_payload):
    validator = load_validator()
    bad_alert = copy.deepcopy(valid_alert_payload)
    bad_alert["identity"]["detector_family"] = "deep_learning"  # Invalid enum

    is_valid, errors = validate_alert_payload(bad_alert, validator)
    assert is_valid is False
    assert any("is not one of ['traditional_ml', 'dl_ensemble']" in e for e in errors)


def test_base_learner_votes_must_be_exactly_3(valid_alert_payload):
    validator = load_validator()
    bad_alert = copy.deepcopy(valid_alert_payload)
    bad_alert["prediction"]["base_learner_votes"].pop()  # Only 2 votes

    is_valid, errors = validate_alert_payload(bad_alert, validator)
    assert is_valid is False
    assert any("too short" in e or "minItems" in e or "fewer than" in e for e in errors)


def test_top_features_must_be_exactly_10(valid_alert_payload):
    validator = load_validator()
    bad_alert = copy.deepcopy(valid_alert_payload)
    bad_alert["shap_attribution"]["top_features"].pop()  # Only 9 features

    is_valid, errors = validate_alert_payload(bad_alert, validator)
    assert is_valid is False
    assert any("too short" in e or "minItems" in e or "fewer than" in e for e in errors)


def test_explainer_type_must_be_valid_enum(valid_alert_payload):
    validator = load_validator()
    bad_alert = copy.deepcopy(valid_alert_payload)
    bad_alert["shap_attribution"]["explainer_type"] = "LinearExplainer"  # Invalid enum

    is_valid, errors = validate_alert_payload(bad_alert, validator)
    assert is_valid is False
    assert any("is not one of ['TreeExplainer', 'DeepExplainer', 'GradientExplainer']" in e for e in errors)
