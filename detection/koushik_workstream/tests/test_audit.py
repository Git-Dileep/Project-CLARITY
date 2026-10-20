"""Tests for dataset audit reporting, statistics aggregation, and error states.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd

from src.audit_datasets import run_audit
from src.config import WorkstreamConfig


def test_audit_runs_successfully(temp_workstream_env):
    cfg: WorkstreamConfig = temp_workstream_env["config"]
    reports_dir = Path(cfg.reports_dir)

    result = run_audit(cfg)
    assert result["status"] == "COMPLETE_SYNTHETIC_REFERENCE"
    assert result["total_rows"] == 300
    assert result["unique_classes_count"] >= 4

    # Verify report files were generated
    assert (reports_dir / "data_audit.json").exists()
    assert (reports_dir / "data_audit.csv").exists()
    assert (reports_dir / "label_counts.csv").exists()
    assert (reports_dir / "audit_summary.md").exists()

    # Verify audit summary md content
    md_text = (reports_dir / "audit_summary.md").read_text(encoding="utf-8")
    assert "Dataset Audit Summary" in md_text
    assert "Total Rows" in md_text


def test_audit_marks_blocked_when_dataset_missing(tmp_path: Path):
    non_existent = tmp_path / "missing_data" / "*.csv"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = WorkstreamConfig(
        dataset_name="ACI-IoT-Network-2023",
        is_synthetic=False,
        input_path=str(non_existent),
        target_column="Label",
        reports_dir=str(reports_dir),
    )

    result = run_audit(cfg)
    assert result["status"] == "BLOCKED — DATASET NOT AVAILABLE"
    assert (reports_dir / "data_audit.json").exists()
    assert (reports_dir / "audit_summary.md").exists()

    audit_json = json.loads((reports_dir / "data_audit.json").read_text(encoding="utf-8"))
    assert audit_json["status"] == "BLOCKED — DATASET NOT AVAILABLE"


def test_audit_fails_when_target_column_missing(tmp_path: Path):
    f = tmp_path / "invalid.csv"
    f.write_text("feature1,feature2\n1.0,2.0\n3.0,4.0\n", encoding="utf-8")
    reports_dir = tmp_path / "reports"

    cfg = WorkstreamConfig(
        dataset_name="Test",
        is_synthetic=True,
        input_path=str(f),
        target_column="missing_target",
        reports_dir=str(reports_dir),
    )

    with pytest.raises(ValueError, match="is missing or contains no valid values"):
        run_audit(cfg)
