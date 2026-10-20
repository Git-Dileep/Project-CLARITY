"""Pytest fixtures for koushik_workstream testing suite.
"""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from src.synthetic_data import generate_synthetic_network_data
from src.config import WorkstreamConfig


@pytest.fixture
def sample_synthetic_df():
    """Deterministic synthetic DataFrame with 6 classes and 17 features."""
    return generate_synthetic_network_data(n_rows=200, seed=42, target_col="label")


@pytest.fixture
def temp_workstream_env(tmp_path: Path):
    """Temporary workspace environment with synthetic data and configuration."""
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    config_dir = tmp_path / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "synthetic_sample.csv"
    df = generate_synthetic_network_data(n_rows=300, seed=42, target_col="label")
    df.to_csv(csv_path, index=False)

    config = WorkstreamConfig(
        dataset_name="Test-Synthetic-Dataset",
        is_synthetic=True,
        input_path=str(csv_path),
        target_column="label",
        flow_id_column="flow_id",
        timestamp_column="timestamp",
        device_id_column="device_id",
        metadata_columns=["device_id", "source_file", "sample_id"],
        drop_columns=["src_ip", "dst_ip", "flow_id", "timestamp"],
        test_size=0.25,
        random_seed=42,
        reports_dir=str(reports_dir),
        models_dir=str(models_dir),
        data_dir=str(data_dir),
    )
    config.traditional.rf_n_estimators = 10
    config.traditional.xgb_n_estimators = 10
    config.traditional.mlp_max_iter = 15
    config.dl.epochs = 2
    config.dl.batch_size = 32
    config.explainability.background_samples = 15
    config.pilot.pilot_size = 10
    config.pilot.min_classes = 4

    config_file = config_dir / "test_config.json"
    config.save(config_file)

    return {
        "tmp_path": tmp_path,
        "config": config,
        "config_file": config_file,
        "csv_path": csv_path,
        "df": df,
        "reports_dir": reports_dir,
        "models_dir": models_dir,
    }
