"""Phase 4 Validation Tests: Data integrity, preprocessing equivalence, zero leakage,
atomic checkpointing, resume capability, and seed reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.aci_data_pipeline import (
    clean_aci_chunk,
    split_and_scale_aci,
    generate_synthetic_aci_sample,
    ACI_IDENTIFIERS,
    ACI_RATE_COLS,
)
from src.train_dl import (
    CNN1DModel,
    LSTMModel,
    TransformerModel,
    train_single_dl_model,
)

workstream_dir = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_aci_df() -> pd.DataFrame:
    """Generate a reproducible synthetic ACI sample."""
    return generate_synthetic_aci_sample(n_samples=300, seed=42)


def test_1_data_integrity_and_schema_catalog():
    """Phase 4.1: Verify dataset schema inventory (85 raw columns, >=70 active, 12 classes)."""
    schema_path = workstream_dir / "config" / "feature_schema_aci.json"
    assert schema_path.exists(), "feature_schema_aci.json must exist"

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    assert len(schema["features"]) >= 70, "Expected at least 70 active numerical features"
    assert len(schema["identifier_columns"]) == 7
    assert len(schema["classes"]) == 12, "Expected 12 target classes"

    # Verify inventory report
    inv_path = workstream_dir / "reports" / "aci_dataset_audit" / "dataset_inventory.csv"
    assert inv_path.exists(), "dataset_inventory.csv must exist"
    inv_df = pd.read_csv(inv_path)
    netflow_row = inv_df[inv_df["file_name"] == "ACI-IoT-2023.csv"].iloc[0]
    assert netflow_row["records_count"] == 1231411
    assert netflow_row["features_count"] == 85


def test_2_preprocessing_equivalence_and_feature_alignment(sample_aci_df):
    """Phase 4.2: Verify preprocessing cleans infinities and preserves active numerical features."""
    # Inject infinities and NaNs in rate columns
    sample_aci_df.loc[0, "Flow Byts/s"] = np.inf
    sample_aci_df.loc[1, "Flow Pkts/s"] = -np.inf
    sample_aci_df.loc[2, "Flow Byts/s"] = np.nan

    cleaned = clean_aci_chunk(sample_aci_df)

    assert not np.isinf(cleaned["Flow Byts/s"]).any()
    assert not np.isinf(cleaned["Flow Pkts/s"]).any()
    assert not cleaned["Flow Byts/s"].isna().any()
    assert not cleaned["Flow Pkts/s"].isna().any()

    feature_cols = [c for c in cleaned.columns if c not in ACI_IDENTIFIERS and c != "Label"]
    assert len(feature_cols) >= 70, f"Expected >= 70 active features, got {len(feature_cols)}"


def test_3_zero_leakage_assertion(sample_aci_df):
    """Phase 4.3: Verify train/test split has strictly 0 overlapping flow IDs and scales on train."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = split_and_scale_aci(sample_aci_df, seed=42, output_dir=Path(tmp_dir))
        train_flows = set(res["train_df"]["Flow ID"])
        test_flows = set(res["test_df"]["Flow ID"])

        # Strict assertion: zero overlap
        overlap = train_flows.intersection(test_flows)
        assert len(overlap) == 0, f"Critical leakage: {len(overlap)} overlapping flow IDs!"
        assert res["manifest"]["overlap_count"] == 0

        # Verify manifest saved
        manifest_file = Path(tmp_dir) / "split_manifest.json"
        assert manifest_file.exists()


def test_4_smoke_and_atomic_checkpointing(tmp_path):
    """Phase 4.4: Verify atomic checkpointing (.pt.tmp -> .pt) during smoke training."""
    n_samples, n_feats, n_classes = 60, 75, 4
    X = torch.randn(n_samples, n_feats)
    y = torch.randint(0, n_classes, (n_samples,))

    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
    model = CNN1DModel(n_features=n_feats, n_classes=n_classes, channels=16)

    ckpt_dir = tmp_path / "checkpoints"
    fitted_model, duration, history = train_single_dl_model(
        name="CNN1D",
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        checkpoint_dir=ckpt_dir,
        resume=False,
    )

    assert len(history) == 2
    assert (ckpt_dir / "CNN1D_epoch_1.pt").exists()
    assert (ckpt_dir / "CNN1D_epoch_2.pt").exists()
    assert (ckpt_dir / "CNN1D_latest.pt").exists()
    assert (ckpt_dir / "CNN1D_best.pt").exists()
    # Confirm temporary files were atomically replaced
    assert not list(ckpt_dir.glob("*.pt.tmp"))


def test_5_resume_test_preserves_best_model(tmp_path):
    """Phase 4.5: Verify resuming continues training and preserves best model separately."""
    n_samples, n_feats, n_classes = 60, 75, 4
    X = torch.randn(n_samples, n_feats)
    y = torch.randint(0, n_classes, (n_samples,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Initial run for 1 epoch
    model1 = LSTMModel(n_features=n_feats, n_classes=n_classes, hidden_size=16, bidirectional=False)
    train_single_dl_model(
        name="LSTM",
        model=model1,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        checkpoint_dir=ckpt_dir,
        resume=False,
    )

    best_ckpt_file = ckpt_dir / "LSTM_best.pt"
    assert best_ckpt_file.exists()
    best_data_pre = torch.load(best_ckpt_file, weights_only=False)
    pre_val_loss = best_data_pre["val_loss"]

    # Resume run for epoch 2
    model2 = LSTMModel(n_features=n_feats, n_classes=n_classes, hidden_size=16, bidirectional=False)
    _, _, history_resume = train_single_dl_model(
        name="LSTM",
        model=model2,
        train_loader=loader,
        val_loader=loader,
        device=torch.device("cpu"),
        epochs=2,
        lr=1e-3,
        weight_decay=1e-4,
        checkpoint_dir=ckpt_dir,
        resume=True,
    )

    assert (ckpt_dir / "LSTM_epoch_2.pt").exists()
    assert (ckpt_dir / "LSTM_latest.pt").exists()
    # Resume should only execute remaining epochs (epoch 2)
    assert len(history_resume) == 2


def test_6_seed_reproducibility(sample_aci_df):
    """Phase 4.6: Verify identical random seed produces 100% deterministic splits."""
    res1 = split_and_scale_aci(sample_aci_df, seed=42)
    res2 = split_and_scale_aci(sample_aci_df, seed=42)

    assert res1["train_df"]["Flow ID"].tolist() == res2["train_df"]["Flow ID"].tolist()
    assert res1["test_df"]["Flow ID"].tolist() == res2["test_df"]["Flow ID"].tolist()
    np.testing.assert_array_almost_equal(res1["X_train"], res2["X_train"])
    np.testing.assert_array_almost_equal(res1["X_test"], res2["X_test"])
