"""Tests for leakage-free preprocessing, train/test splitting, and transformer behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    FeaturePreprocessor,
    prepare_data_and_splits,
    identify_columns,
)
from src.config import WorkstreamConfig


def test_feature_preprocessor_fit_transform():
    train_df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0],
        "num2": [10.0, 20.0, 30.0, 40.0],
        "cat1": ["tcp", "udp", "tcp", np.nan],
    })
    test_df = pd.DataFrame({
        "num1": [5.0, np.nan],
        "num2": [50.0, 60.0],
        "cat1": ["tcp", "unseen_proto"],  # Unknown category at inference
    })

    prep = FeaturePreprocessor(numeric_cols=["num1", "num2"], categorical_cols=["cat1"])
    prep.fit(train_df)

    X_train = prep.transform(train_df)
    X_test = prep.transform(test_df)

    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 2
    # Verify no NaN values remained
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    # Check feature names
    names = prep.feature_names_
    assert "num1" in names
    assert "num2" in names
    assert any("cat1" in name for name in names)


def test_prepare_splits_leakage_and_manifest(temp_workstream_env):
    cfg: WorkstreamConfig = temp_workstream_env["config"]
    df: pd.DataFrame = temp_workstream_env["df"]
    reports_dir = Path(cfg.reports_dir)

    splits = prepare_data_and_splits(df, cfg)

    assert splits.X_train.shape[0] > 0
    assert splits.X_test.shape[0] > 0
    assert len(splits.y_train) == splits.X_train.shape[0]
    assert len(splits.y_test) == splits.X_test.shape[0]

    # Verify split manifest was written
    manifest_file = reports_dir / "split_manifest.json"
    assert manifest_file.exists()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["random_seed"] == cfg.random_seed
    assert manifest["train_rows"] == len(splits.train_df)
    assert manifest["test_rows"] == len(splits.test_df)

    # Verify that metadata columns (flow_id, timestamp, src_ip, dst_ip) were NOT included as features
    for col in cfg.drop_columns:
        assert col not in splits.feature_names


def test_unknown_category_in_test_does_not_crash():
    train_df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0],
        "cat": ["A", "B", "A"],
    })
    test_df = pd.DataFrame({
        "num": [4.0],
        "cat": ["NEW_CATEGORY"],
    })

    prep = FeaturePreprocessor(numeric_cols=["num"], categorical_cols=["cat"])
    prep.fit(train_df)
    res = prep.transform(test_df)
    assert res.shape[0] == 1
    assert not np.isnan(res).any()
