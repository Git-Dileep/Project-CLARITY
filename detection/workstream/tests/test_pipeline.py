"""End-to-end smoke test and reproducibility tests.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from src.synthetic_data import generate_synthetic_network_data, CLASSES
from src.export_pilot import select_stratified_pilot_indices
from src.run_all import run_pipeline


def test_synthetic_data_generation_contract():
    df1 = generate_synthetic_network_data(n_rows=500, seed=42, target_col="label")
    df2 = generate_synthetic_network_data(n_rows=500, seed=42, target_col="label")

    # Reproducibility
    pd.testing.assert_frame_equal(df1, df2)

    # Classes: at least 5
    unique_classes = df1["label"].unique()
    assert len(unique_classes) >= 5

    # Features: at least 15 numeric/categorical features
    non_feature_cols = ["flow_id", "timestamp", "device_id", "src_ip", "dst_ip", "label"]
    feature_cols = [c for c in df1.columns if c not in non_feature_cols]
    assert len(feature_cols) >= 15


def test_stratified_pilot_indices_reproducibility():
    df = generate_synthetic_network_data(n_rows=300, seed=42, target_col="label")
    idx1 = select_stratified_pilot_indices(df, target_col="label", pilot_size=20, seed=42)
    idx2 = select_stratified_pilot_indices(df, target_col="label", pilot_size=20, seed=42)

    assert idx1 == idx2
    assert len(idx1) == 20
    # Must span at least 5 classes
    sampled_classes = df.loc[idx1, "label"].unique()
    assert len(sampled_classes) >= 5


def test_end_to_end_pipeline_smoke_run(temp_workstream_env):
    cfg_file = temp_workstream_env["config_file"]
    exit_code = run_pipeline(str(cfg_file))
    assert exit_code == 0

    reports_dir = temp_workstream_env["reports_dir"]
    assert (reports_dir / "data_audit.json").exists()
    assert (reports_dir / "traditional_metrics.json").exists()
    assert (reports_dir / "traditional_metrics.csv").exists()
    assert (reports_dir / "traditional_test_predictions.csv").exists()
    assert (reports_dir / "dl" / "dl_metrics.json").exists()
    assert (reports_dir / "dl" / "dl_test_predictions.csv").exists()
    assert (reports_dir / "pilot_traditional_alerts.json").exists()
    assert (reports_dir / "pilot_dl_alerts.json").exists()
    assert (reports_dir / "pilot_20_alerts.csv").exists()
    assert (reports_dir / "alert_validation_report.json").exists()
