"""Tests for data loading, deduplication, schema compatibility, and error handling.
"""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import (
    find_files,
    unique_files_by_content,
    check_schema_compatibility,
    load_dataset,
    normalize_label,
)
from src.config import WorkstreamConfig


def test_find_files_single_and_glob(tmp_path: Path):
    f1 = tmp_path / "data1.csv"
    f2 = tmp_path / "data2.csv"
    txt = tmp_path / "other.txt"
    f1.write_text("a,b\n1,2", encoding="utf-8")
    f2.write_text("a,b\n3,4", encoding="utf-8")
    txt.write_text("text", encoding="utf-8")

    # Match glob
    found = find_files(tmp_path / "*.csv")
    assert len(found) == 2
    assert f1 in found and f2 in found

    # Match directory
    found_dir = find_files(tmp_path)
    assert len(found_dir) == 2


def test_unique_files_by_content(tmp_path: Path):
    f1 = tmp_path / "file1.csv"
    f2 = tmp_path / "file2.csv"  # exact duplicate of file1
    f3 = tmp_path / "file3.csv"  # different content

    content = "col1,col2\n10,20\n"
    f1.write_text(content, encoding="utf-8")
    f2.write_text(content, encoding="utf-8")
    f3.write_text("col1,col2\n99,88\n", encoding="utf-8")

    kept, dups = unique_files_by_content([f1, f2, f3])
    assert len(kept) == 2
    assert len(dups) == 1
    assert dups[0]["duplicate_file"] == str(f2) or dups[0]["duplicate_file"] == str(f1)


def test_check_schema_compatibility_success(tmp_path: Path):
    f1 = tmp_path / "p1.csv"
    f2 = tmp_path / "p2.csv"
    f1.write_text("ts,feature1,label\n1,10,benign\n", encoding="utf-8")
    f2.write_text("ts,feature1,label\n2,20,ddos\n", encoding="utf-8")

    compatible, cols, report = check_schema_compatibility([f1, f2], target_col="label")
    assert compatible is True
    assert "label" in cols
    assert "feature1" in cols


def test_check_schema_compatibility_mismatch(tmp_path: Path):
    f1 = tmp_path / "p1.csv"
    f2 = tmp_path / "p2.csv"
    f1.write_text("ts,feature1,label\n1,10,benign\n", encoding="utf-8")
    f2.write_text("ts,feature2,label\n2,20,ddos\n", encoding="utf-8")

    compatible, _, report = check_schema_compatibility([f1, f2], target_col="label")
    assert compatible is False
    assert len(report["mismatched_files"]) > 0


def test_check_schema_missing_target(tmp_path: Path):
    f1 = tmp_path / "p1.csv"
    f1.write_text("ts,feature1\n1,10\n", encoding="utf-8")

    compatible, _, report = check_schema_compatibility([f1], target_col="label")
    assert compatible is False
    assert report["mismatched_files"][str(f1)]["has_target"] is False


def test_load_dataset_missing_target_raises_error(tmp_path: Path):
    f1 = tmp_path / "data.csv"
    f1.write_text("colA,colB\n1,2\n", encoding="utf-8")

    cfg = WorkstreamConfig(
        dataset_name="Test",
        is_synthetic=False,
        input_path=str(f1),
        target_column="non_existent_label",
    )

    with pytest.raises(ValueError, match="not found in dataset columns|compatibility error"):
        load_dataset(cfg)


def test_normalize_label():
    series = pd.Series([" benign ", "", "nan", "None", "ddos", None])
    norm = normalize_label(series)
    assert norm.iloc[0] == "benign"
    assert norm.iloc[4] == "ddos"
    assert pd.isna(norm.iloc[1])
    assert pd.isna(norm.iloc[2])
    assert pd.isna(norm.iloc[3])
    assert pd.isna(norm.iloc[5])
