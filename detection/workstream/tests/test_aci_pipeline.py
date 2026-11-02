"""Test suite verifying ACI IoT 2023 configuration, streaming loader, preprocessor, and alert validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.aci_data_pipeline import (
    clean_aci_chunk,
    split_and_scale_aci,
    generate_synthetic_aci_sample,
    ACI_IDENTIFIERS,
)
from src.config import WorkstreamConfig

workstream_dir = Path(__file__).resolve().parent.parent
repo_root = workstream_dir.parent.parent


def test_aci_feature_schema_valid():
    schema_path = workstream_dir / "config" / "feature_schema_aci.json"
    assert schema_path.exists(), f"Missing schema file: {schema_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["dataset_name"] == "ACI-IoT-Network-Dataset-2023"
    assert meta["target_column"] == "Label"
    assert meta["flow_id_column"] == "Flow ID"
    assert len(meta["features"]) >= 70
    assert len(meta["classes"]) == 12
    assert len(meta["macro_categories"]) == 5

    # Check that identifiers are explicitly documented
    for ident in ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"]:
        assert ident in meta["identifier_columns"]


def test_aci_config_loading():
    config_path = workstream_dir / "config" / "aci_iot_2023.json"
    assert config_path.exists(), f"Missing config file: {config_path}"

    cfg = WorkstreamConfig.load(config_path)
    assert cfg.dataset_name == "ACI-IoT-Network-Dataset-2023"
    assert cfg.target_column == "Label"
    assert cfg.traditional.rf_n_estimators == 100
    assert cfg.traditional.xgb_n_estimators == 100
    assert cfg.traditional.mlp_hidden_layer_sizes == [128, 64]
    assert cfg.dl.enabled is True


def test_aci_clean_chunk_imputation():
    # Construct a chunk with missing and infinite values in rate columns
    df = pd.DataFrame({
        "Flow ID": ["F1", "F2", "F3"],
        "Flow Byts/s": [100.5, np.inf, np.nan],
        "Flow Pkts/s": [-np.inf, 50.0, np.nan],
        "Tot Fwd Pkts": [5, 10, 15],
        "Label": ["Benign", "Port Scan", "SYN Flood"],
    })

    cleaned = clean_aci_chunk(df, target_col="Label")
    assert len(cleaned) == 3
    assert not np.isinf(cleaned["Flow Byts/s"]).any()
    assert not np.isinf(cleaned["Flow Pkts/s"]).any()
    assert not cleaned["Flow Byts/s"].isna().any()
    assert not cleaned["Flow Pkts/s"].isna().any()
    assert cleaned["Flow Byts/s"].iloc[1] == 0.0  # replaced inf
    assert cleaned["Flow Pkts/s"].iloc[0] == 0.0  # replaced -inf


def test_aci_split_determinism_and_zero_leakage():
    sample_df = generate_synthetic_aci_sample(n_samples=500, seed=42)
    res = split_and_scale_aci(sample_df, test_size=0.2, seed=42)

    assert len(res["X_train"]) == 400
    assert len(res["X_test"]) == 100
    assert res["manifest"]["overlap_count"] == 0

    # Assert scaler was fitted on train (mean should be approx 0 on train)
    assert np.allclose(res["X_train"].mean(axis=0), 0.0, atol=1e-5)

    # Re-run to verify determinism
    res2 = split_and_scale_aci(sample_df, test_size=0.2, seed=42)
    assert np.array_equal(res["X_train"], res2["X_train"])
    assert np.array_equal(res["y_train"], res2["y_train"])


def test_aci_pilot_alerts_schema_conformance():
    trad_file = workstream_dir / "reports" / "aci_experiment" / "pilot_traditional_alerts_10.json"
    dl_file = workstream_dir / "reports" / "aci_experiment" / "pilot_dl_alerts_10.json"

    assert trad_file.exists(), f"Traditional alerts missing: {trad_file}"
    assert dl_file.exists(), f"DL alerts missing: {dl_file}"

    with open(trad_file, "r", encoding="utf-8") as f:
        trad_alerts = json.load(f)
    with open(dl_file, "r", encoding="utf-8") as f:
        dl_alerts = json.load(f)

    assert len(trad_alerts) == 10
    assert len(dl_alerts) == 10

    # Check 1:1 flow ID mapping
    trad_flows = [a["identity"]["flow_id"] for a in trad_alerts]
    dl_flows = [a["identity"]["flow_id"] for a in dl_alerts]
    assert trad_flows == dl_flows

    # Check top-10 SHAP feature monotonicity and ranking
    for a in trad_alerts + dl_alerts:
        top_f = a["shap_attribution"]["top_features"]
        assert len(top_f) == 10
        ranks = [item["rank"] for item in top_f]
        assert ranks == list(range(1, 11))
        # Descending absolute SHAP
        vals = [abs(item["shap_value"]) for item in top_f]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] - 1e-6


def test_aci_leakage_audit_zero_overlap():
    audit_file = workstream_dir / "reports" / "aci_experiment" / "leakage_audit_report.json"
    assert audit_file.exists(), f"Leakage audit report missing: {audit_file}"

    with open(audit_file, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["status"] == "PASSED — ZERO LEAKAGE VERIFIED"
    assert audit["leakage_detected"] is False
    assert audit["leakage_metrics"]["train_pilot_overlap_count"] == 0
    assert audit["leakage_metrics"]["train_traditional_overlap_count"] == 0
    assert audit["leakage_metrics"]["train_dl_overlap_count"] == 0
    assert audit["leakage_metrics"]["cross_detector_flows_identical"] is True


def test_aci_dl_architecture_and_learner_votes():
    from src.train_dl import LSTMModel
    lstm = LSTMModel(n_features=75, n_classes=12, hidden_size=32)
    assert lstm.bidirectional is False
    assert lstm.lstm.bidirectional is False

    dl_file = workstream_dir / "reports" / "aci_experiment" / "pilot_dl_alerts_10.json"
    with open(dl_file, "r", encoding="utf-8") as f:
        dl_alerts = json.load(f)

    for alert in dl_alerts:
        votes = alert["prediction"]["base_learner_votes"]
        assert len(votes) == 3
        learner_names = [v["learner_name"] for v in votes]
        assert learner_names == ["1D-CNN", "LSTM", "Transformer"]
