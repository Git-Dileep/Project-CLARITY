"""Automated test suite verifying the reproducible 10-flow sample pilot run.

Validates:
1. Presence of all required pilot artifacts.
2. Zero Draft 2020-12 schema errors and semantic compliance.
3. Zero data leakage between training split and pilot evaluation flows.
4. 1:1 flow ID identity across Traditional ML and Deep Learning ensembles.
5. Representation of at least 5 unique attack/benign classes.
6. SHAP top-10 attribution ranking integrity and monotonicity.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.leakage_audit import run_leakage_audit
import sys

# Add repo root to sys.path to import validate_schemas
repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.validate_schemas import run_full_validation


@pytest.fixture(scope="module")
def sample_reports_dir():
    base_dir = Path(__file__).resolve().parent.parent / "reports" / "sample_pilot10"
    assert base_dir.exists(), f"Sample pilot reports directory does not exist: {base_dir}"
    return base_dir


def test_pilot10_artifacts_exist(sample_reports_dir):
    required_files = [
        "pilot_traditional_alerts_10.json",
        "pilot_dl_alerts_10.json",
        "pilot_manifest.json",
        "pilot_manifest.csv",
        "split_manifest.json",
        "leakage_audit_report.json",
        "leakage_audit_report.md",
        "schema_and_semantic_validation_report.json",
    ]
    for fname in required_files:
        fpath = sample_reports_dir / fname
        assert fpath.exists(), f"Required artifact missing: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"


def test_pilot10_schema_and_semantics(sample_reports_dir):
    schema_path = repo_root / "contracts" / "alert_schema_v1.json"
    trad_file = sample_reports_dir / "pilot_traditional_alerts_10.json"
    dl_file = sample_reports_dir / "pilot_dl_alerts_10.json"
    report_file = sample_reports_dir / "schema_and_semantic_validation_report.json"

    res = run_full_validation(
        schema_path=schema_path,
        trad_path=trad_file,
        dl_path=dl_file,
        output_report_path=report_file,
        expected_count=10,
    )
    assert res["status"] == "PASSED", f"Schema validation failed: {res['errors']}"
    assert res["total_errors"] == 0
    assert res["alerts_count"]["traditional"] == 10
    assert res["alerts_count"]["dl"] == 10
    assert len(res["classes_represented"]) >= 5


def test_pilot10_zero_data_leakage(sample_reports_dir):
    audit = run_leakage_audit(sample_reports_dir)
    assert audit["status"] == "PASSED — ZERO LEAKAGE VERIFIED"
    assert not audit["leakage_detected"]
    assert audit["leakage_metrics"]["train_pilot_overlap_count"] == 0
    assert audit["leakage_metrics"]["train_traditional_overlap_count"] == 0
    assert audit["leakage_metrics"]["train_dl_overlap_count"] == 0
    assert audit["leakage_metrics"]["pilot_subset_of_test"] is True
    assert audit["leakage_metrics"]["cross_detector_flows_identical"] is True


def test_pilot10_identical_flow_ids(sample_reports_dir):
    with open(sample_reports_dir / "pilot_traditional_alerts_10.json", "r", encoding="utf-8") as f:
        trad_alerts = json.load(f)
    with open(sample_reports_dir / "pilot_dl_alerts_10.json", "r", encoding="utf-8") as f:
        dl_alerts = json.load(f)
    with open(sample_reports_dir / "pilot_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    trad_flows = [a["identity"]["flow_id"] for a in trad_alerts]
    dl_flows = [a["identity"]["flow_id"] for a in dl_alerts]
    manifest_flows = manifest["flow_ids"]

    assert len(trad_flows) == 10
    assert len(dl_flows) == 10
    assert trad_flows == dl_flows, "Traditional and DL alerts must have identical flow IDs"
    assert trad_flows == manifest_flows, "Alert flow IDs must match pilot manifest flow IDs"


def test_pilot10_class_diversity(sample_reports_dir):
    with open(sample_reports_dir / "pilot_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["represented_classes_count"] >= 5
    assert len(manifest["represented_classes"]) >= 5
    assert "Benign" in manifest["represented_classes"]
    # Check that there are attack classes too
    attack_classes = [c for c in manifest["represented_classes"] if c != "Benign"]
    assert len(attack_classes) >= 4


def test_pilot10_shap_structure_and_monotonicity(sample_reports_dir):
    for fname in ["pilot_traditional_alerts_10.json", "pilot_dl_alerts_10.json"]:
        with open(sample_reports_dir / fname, "r", encoding="utf-8") as f:
            alerts = json.load(f)

        for alert in alerts:
            top_features = alert["shap_attribution"]["top_features"]
            assert len(top_features) == 10
            ranks = [feat["rank"] for feat in top_features]
            assert ranks == list(range(1, 11))

            abs_shap = [abs(feat["shap_value"]) for feat in top_features]
            for i in range(len(abs_shap) - 1):
                assert abs_shap[i] >= abs_shap[i + 1] - 1e-6, (
                    f"SHAP magnitude non-monotonic in {fname}: {abs_shap[i]} < {abs_shap[i+1]}"
                )
