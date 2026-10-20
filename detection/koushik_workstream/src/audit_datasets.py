"""Audit datasets and produce detailed machine-readable and human-readable reports.

Generates:
- data_audit.json
- data_audit.csv
- label_counts.csv
- audit_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from .config import WorkstreamConfig
from .data_loader import find_files, unique_files_by_content, check_schema_compatibility, normalize_label
from .synthetic_data import save_synthetic_data


def inspect_single_file(path: Path, target_col: str, chunksize: int = 100000) -> Dict[str, Any]:
    """Inspect row/column counts, types, nulls, infs, and target distributions for a single file."""
    rows = 0
    missing_per_col: Dict[str, int] = {}
    inf_per_col: Dict[str, int] = {}
    dtypes: Dict[str, str] = {}
    columns: List[str] = []
    label_counts: Dict[str, int] = {}
    invalid_target_count = 0
    total_duplicates = 0

    seen_hashes = set()

    try:
        for i, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False)):
            if i == 0:
                columns = list(chunk.columns)
                dtypes = {col: str(chunk[col].dtype) for col in columns}

            rows += len(chunk)

            # Check missing
            for col in chunk.columns:
                n_miss = int(chunk[col].isna().sum())
                missing_per_col[col] = missing_per_col.get(col, 0) + n_miss

                # Check inf in numeric
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    num_inf = int(np.isinf(pd.to_numeric(chunk[col], errors="coerce")).sum())
                    inf_per_col[col] = inf_per_col.get(col, 0) + num_inf

            # Target validation
            if target_col in chunk.columns:
                raw_target = chunk[target_col]
                invalid_target_count += int(raw_target.isna().sum())
                norm_target = normalize_label(raw_target).dropna()
                for lbl, cnt in norm_target.value_counts().items():
                    label_counts[str(lbl)] = label_counts.get(str(lbl), 0) + int(cnt)

    except Exception as exc:
        return {
            "file": str(path.name),
            "full_path": str(path),
            "status": "ERROR",
            "error": str(exc),
            "rows": rows,
            "columns": columns,
            "dtypes": dtypes,
            "missing_per_col": missing_per_col,
            "total_missing": sum(missing_per_col.values()),
            "inf_per_col": inf_per_col,
            "total_inf": sum(inf_per_col.values()),
            "label_counts": label_counts,
            "invalid_target_count": invalid_target_count,
        }

    total_missing = sum(missing_per_col.values())
    total_inf = sum(inf_per_col.values())

    return {
        "file": str(path.name),
        "full_path": str(path),
        "size_bytes": path.stat().st_size,
        "status": "OK",
        "rows": rows,
        "columns": columns,
        "dtypes": dtypes,
        "missing_per_col": missing_per_col,
        "total_missing": total_missing,
        "inf_per_col": inf_per_col,
        "total_inf": total_inf,
        "label_counts": label_counts,
        "invalid_target_count": invalid_target_count,
    }


def run_audit(config: WorkstreamConfig, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute complete dataset audit according to KOU-01 criteria."""
    work_dir = base_dir or Path.cwd()
    reports_dir = config.resolve_path(work_dir, config.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    resolved_input = config.resolve_path(work_dir, config.input_path)
    all_files = find_files(resolved_input, work_dir)

    # Handle synthetic data generation if in synthetic mode
    if config.is_synthetic and not all_files:
        print("[AUDIT] Generating synthetic reference data for audit...")
        save_synthetic_data(resolved_input, n_rows=5000, seed=config.random_seed, target_col=config.target_column)
        all_files = find_files(resolved_input, work_dir)

    # Check if real dataset is missing
    if not all_files:
        msg = f"DATASET NOT AVAILABLE: No CSV files found matching {config.input_path}."
        if not config.is_synthetic:
            blocked_summary = {
                "dataset_name": config.dataset_name,
                "status": "BLOCKED — DATASET NOT AVAILABLE",
                "message": msg,
                "input_path": config.input_path,
                "instructions": "Acquire and place the ACI IoT Network Dataset 2023 CSV files into the input directory specified in configuration."
            }
            with (reports_dir / "data_audit.json").open("w", encoding="utf-8") as f:
                json.dump(blocked_summary, f, indent=2)
            md_path = reports_dir / "audit_summary.md"
            md_path.write_text(
                f"# Dataset Audit Summary: {config.dataset_name}\n\n"
                f"**Status: BLOCKED — DATASET NOT AVAILABLE**\n\n"
                f"> [!WARNING]\n"
                f"> The real dataset for `{config.dataset_name}` could not be located at `{config.input_path}`.\n"
                f"> Model training and pilot verification on this dataset remain blocked until files are present.\n",
                encoding="utf-8"
            )
            print(f"[AUDIT] {msg}")
            return blocked_summary
        raise FileNotFoundError(msg)

    unique_files, duplicate_files = unique_files_by_content(all_files)
    is_compatible, common_cols, schema_report = check_schema_compatibility(unique_files, config.target_column)

    file_reports: List[Dict[str, Any]] = []
    total_rows = 0
    aggregated_labels: Dict[str, int] = {}
    aggregated_missing: Dict[str, int] = {}
    aggregated_inf: Dict[str, int] = {}
    total_invalid_targets = 0

    for f in unique_files:
        print(f"[AUDIT] Inspecting {f.name}...")
        report = inspect_single_file(f, config.target_column, config.chunksize)
        file_reports.append(report)
        if report["status"] == "OK":
            total_rows += report["rows"]
            total_invalid_targets += report["invalid_target_count"]
            for l, c in report["label_counts"].items():
                aggregated_labels[l] = aggregated_labels.get(l, 0) + c
            for col, c in report["missing_per_col"].items():
                aggregated_missing[col] = aggregated_missing.get(col, 0) + c
            for col, c in report["inf_per_col"].items():
                aggregated_inf[col] = aggregated_inf.get(col, 0) + c

    # Validate target column presence
    if config.target_column not in common_cols or not aggregated_labels:
        raise ValueError(
            f"Configured target column '{config.target_column}' is missing or contains no valid values across inspected files."
        )

    # Class distribution and imbalance
    total_valid_labels = sum(aggregated_labels.values())
    class_stats = []
    for lbl, count in sorted(aggregated_labels.items(), key=lambda x: x[1], reverse=True):
        class_stats.append({
            "class": lbl,
            "count": count,
            "percentage": round((count / total_valid_labels) * 100, 4) if total_valid_labels > 0 else 0.0,
        })
    df_classes = pd.DataFrame(class_stats)
    df_classes.to_csv(reports_dir / "label_counts.csv", index=False)

    max_c = max(aggregated_labels.values()) if aggregated_labels else 1
    min_c = min(aggregated_labels.values()) if aggregated_labels else 1
    imbalance_ratio = round(max_c / max(1, min_c), 2)

    # Identify suspicious ID/metadata columns
    suspicious_cols = [
        c for c in common_cols
        if any(term in c.lower() for term in ["id", "ip", "time", "ts", "date", "mac", "port"])
    ]

    # Tabular audit CSV
    audit_rows = []
    for r in file_reports:
        audit_rows.append({
            "file": r["file"],
            "status": r["status"],
            "rows": r.get("rows", 0),
            "columns_count": len(r.get("columns", [])),
            "total_missing_cells": r.get("total_missing", 0),
            "total_inf_cells": r.get("total_inf", 0),
            "invalid_targets": r.get("invalid_target_count", 0),
            "unique_labels": len(r.get("label_counts", {})),
        })
    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(reports_dir / "data_audit.csv", index=False)

    audit_summary = {
        "dataset_name": config.dataset_name,
        "is_synthetic": config.is_synthetic,
        "status": "COMPLETE" if not config.is_synthetic else "COMPLETE_SYNTHETIC_REFERENCE",
        "total_files_inspected": len(all_files),
        "unique_files_count": len(unique_files),
        "duplicate_files_count": len(duplicate_files),
        "schema_compatible_across_files": is_compatible,
        "target_column": config.target_column,
        "total_rows": total_rows,
        "total_columns": len(common_cols),
        "columns": common_cols,
        "suspicious_metadata_columns": suspicious_cols,
        "total_missing_cells": sum(aggregated_missing.values()),
        "total_infinite_cells": sum(aggregated_inf.values()),
        "invalid_target_cells": total_invalid_targets,
        "unique_classes_count": len(aggregated_labels),
        "class_imbalance_ratio": imbalance_ratio,
        "class_distribution": class_stats,
        "exclusions": config.drop_columns,
    }

    with (reports_dir / "data_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    # Markdown Summary
    md_lines = [
        f"# Dataset Audit Summary: {config.dataset_name}",
        "",
        f"> [!NOTE]",
        f"> Audit completed for dataset `{config.dataset_name}`. Synthetic Mode: `{config.is_synthetic}`.",
        "",
        "## Overview",
        f"- **Total Rows:** {total_rows:,}",
        f"- **Total Columns:** {len(common_cols)}",
        f"- **Unique Attack/Traffic Classes:** {len(aggregated_labels)}",
        f"- **Class Imbalance Ratio (Max:Min):** {imbalance_ratio}:1",
        f"- **Total Missing Values:** {sum(aggregated_missing.values()):,}",
        f"- **Total Infinite Values:** {sum(aggregated_inf.values()):,}",
        f"- **Duplicate Files Found:** {len(duplicate_files)}",
        "",
        "## Per-Class Distribution",
        "| Class | Record Count | Percentage |",
        "| :--- | :--- | :--- |",
    ]
    for cs in class_stats:
        md_lines.append(f"| {cs['class']} | {cs['count']:,} | {cs['percentage']}% |")

    md_lines.extend([
        "",
        "## Metadata & Excluded Columns",
        f"- **Suspicious ID/Time/IP Columns:** `{', '.join(suspicious_cols)}`",
        f"- **Configured Preprocessing Exclusions:** `{', '.join(config.drop_columns)}`",
    ])

    (reports_dir / "audit_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[AUDIT] Audit finished successfully. Reports generated in {reports_dir}")
    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset audit entry point.")
    parser.add_argument("--config", default="config/reference.json", help="Path to config file")
    args = parser.parse_args()

    cfg = WorkstreamConfig.load(args.config)
    run_audit(cfg)
