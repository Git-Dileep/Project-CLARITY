"""Data Leakage Audit Module.

Verifies strict isolation between training set and evaluation/pilot set:
1. Loads train/test flow IDs from split_manifest.json.
2. Loads pilot flow IDs from pilot_manifest.json.
3. Loads alert flow IDs from both Traditional ML and DL export JSONs.
4. Asserts that the intersection between training flow IDs and pilot/alert flow IDs is strictly EMPTY (overlap == 0).
5. Confirms that pilot flows are a proper subset of held-out test flows.
6. Confirms that Traditional and DL alerts share the exact same flow IDs.
7. Generates JSON and Markdown audit reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from .config import WorkstreamConfig


def run_leakage_audit(reports_dir: str | Path) -> Dict[str, Any]:
    """Execute leakage audit on artifacts in reports_dir."""
    reports_path = Path(reports_dir)
    if not reports_path.is_absolute():
        reports_path = Path.cwd() / reports_path

    split_manifest_path = reports_path / "split_manifest.json"
    pilot_manifest_path = reports_path / "pilot_manifest.json"

    if not split_manifest_path.exists():
        raise FileNotFoundError(f"split_manifest.json not found in {reports_path}")
    if not pilot_manifest_path.exists():
        raise FileNotFoundError(f"pilot_manifest.json not found in {reports_path}")

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    with open(pilot_manifest_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    train_flow_ids: List[str] = split_data.get("train_flow_ids", [])
    test_flow_ids: List[str] = split_data.get("test_flow_ids", [])
    pilot_flow_ids: List[str] = pilot_data.get("pilot_flow_ids", [])

    train_set: Set[str] = set(train_flow_ids)
    test_set: Set[str] = set(test_flow_ids)
    pilot_set: Set[str] = set(pilot_flow_ids)

    # Check alert JSONs if present
    trad_alert_files = [
        reports_path / "pilot_traditional_alerts_10.json",
        reports_path / "pilot_traditional_alerts.json",
    ]
    dl_alert_files = [
        reports_path / "pilot_dl_alerts_10.json",
        reports_path / "pilot_dl_alerts.json",
    ]

    trad_flow_ids: List[str] = []
    trad_file_found: Path | None = None
    for p in trad_alert_files:
        if p.exists():
            trad_file_found = p
            with open(p, "r", encoding="utf-8") as f:
                alerts = json.load(f)
                trad_flow_ids = [a.get("flow_id", "") for a in alerts]
            break

    dl_flow_ids: List[str] = []
    dl_file_found: Path | None = None
    for p in dl_alert_files:
        if p.exists():
            dl_file_found = p
            with open(p, "r", encoding="utf-8") as f:
                alerts = json.load(f)
                dl_flow_ids = [a.get("flow_id", "") for a in alerts]
            break

    trad_set: Set[str] = set(trad_flow_ids)
    dl_set: Set[str] = set(dl_flow_ids)

    # Overlap computations
    train_pilot_overlap = sorted(list(train_set.intersection(pilot_set)))
    train_trad_overlap = sorted(list(train_set.intersection(trad_set))) if trad_set else []
    train_dl_overlap = sorted(list(train_set.intersection(dl_set))) if dl_set else []

    # Test containment
    pilot_in_test = pilot_set.issubset(test_set)
    pilot_missing_from_test = sorted(list(pilot_set - test_set))

    # Cross-detector flow equality
    detectors_identical = (trad_flow_ids == dl_flow_ids) if (trad_flow_ids and dl_flow_ids) else None
    detector_mismatches = sorted(list(trad_set ^ dl_set)) if (trad_set and dl_set) else []

    # Overall audit verdict
    leakage_detected = (
        len(train_pilot_overlap) > 0
        or len(train_trad_overlap) > 0
        or len(train_dl_overlap) > 0
        or not pilot_in_test
        or (detectors_identical is False)
    )

    audit_result = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "FAILED — LEAKAGE DETECTED" if leakage_detected else "PASSED — ZERO LEAKAGE VERIFIED",
        "leakage_detected": leakage_detected,
        "run_id": split_data.get("run_id", "UNKNOWN"),
        "split_id": split_data.get("split_id", "UNKNOWN"),
        "source_sha256": split_data.get("source_sha256", "UNKNOWN"),
        "counts": {
            "train_flows_count": len(train_flow_ids),
            "train_unique_flows_count": len(train_set),
            "test_flows_count": len(test_flow_ids),
            "test_unique_flows_count": len(test_set),
            "pilot_flows_count": len(pilot_flow_ids),
            "pilot_unique_flows_count": len(pilot_set),
            "traditional_alerts_count": len(trad_flow_ids),
            "dl_alerts_count": len(dl_flow_ids),
        },
        "leakage_metrics": {
            "train_pilot_overlap_count": len(train_pilot_overlap),
            "train_pilot_overlap_flow_ids": train_pilot_overlap,
            "train_traditional_overlap_count": len(train_trad_overlap),
            "train_traditional_overlap_flow_ids": train_trad_overlap,
            "train_dl_overlap_count": len(train_dl_overlap),
            "train_dl_overlap_flow_ids": train_dl_overlap,
            "pilot_subset_of_test": pilot_in_test,
            "pilot_flows_missing_from_test": pilot_missing_from_test,
            "cross_detector_flows_identical": detectors_identical,
            "cross_detector_flow_mismatches": detector_mismatches,
        },
        "inspected_files": {
            "split_manifest": str(split_manifest_path),
            "pilot_manifest": str(pilot_manifest_path),
            "traditional_alerts": str(trad_file_found) if trad_file_found else None,
            "dl_alerts": str(dl_file_found) if dl_file_found else None,
        },
    }

    # Save JSON report
    json_out = reports_path / "leakage_audit_report.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)

    # Save Markdown report
    md_out = reports_path / "leakage_audit_report.md"
    md_content = f"""# Project CLARITY — Data Leakage Audit Report

**Status:** `{audit_result["status"]}`
**Audit Timestamp:** `{audit_result["audit_timestamp"]}`
**Run ID:** `{audit_result["run_id"]}`
**Split ID:** `{audit_result["split_id"]}`
**Dataset SHA-256:** `{audit_result["source_sha256"]}`

---

## 1. Executive Summary

| Metric | Target | Observed | Status |
|---|---|---|---|
| Train ∩ Pilot Overlap | 0 | {audit_result["leakage_metrics"]["train_pilot_overlap_count"]} | {'PASS' if audit_result["leakage_metrics"]["train_pilot_overlap_count"] == 0 else 'FAIL'} |
| Train ∩ Traditional Alerts Overlap | 0 | {audit_result["leakage_metrics"]["train_traditional_overlap_count"]} | {'PASS' if audit_result["leakage_metrics"]["train_traditional_overlap_count"] == 0 else 'FAIL'} |
| Train ∩ DL Alerts Overlap | 0 | {audit_result["leakage_metrics"]["train_dl_overlap_count"]} | {'PASS' if audit_result["leakage_metrics"]["train_dl_overlap_count"] == 0 else 'FAIL'} |
| Pilot Held-Out Isolation (Pilot ⊆ Test) | True | {audit_result["leakage_metrics"]["pilot_subset_of_test"]} | {'PASS' if audit_result["leakage_metrics"]["pilot_subset_of_test"] else 'FAIL'} |
| Cross-Detector Flow ID Identity | True | {audit_result["leakage_metrics"]["cross_detector_flows_identical"]} | {'PASS' if audit_result["leakage_metrics"]["cross_detector_flows_identical"] else 'FAIL'} |

---

## 2. Partition & Alert Inventory

- **Training Flows:** {audit_result["counts"]["train_flows_count"]} total ({audit_result["counts"]["train_unique_flows_count"]} unique)
- **Held-Out Test Flows:** {audit_result["counts"]["test_flows_count"]} total ({audit_result["counts"]["test_unique_flows_count"]} unique)
- **Pilot Manifest Flows:** {audit_result["counts"]["pilot_flows_count"]} flows
- **Traditional Alerts Exported:** {audit_result["counts"]["traditional_alerts_count"]} alerts
- **Deep Learning Alerts Exported:** {audit_result["counts"]["dl_alerts_count"]} alerts

---

## 3. Verification Details

- **Split Manifest:** `{split_manifest_path.name}`
- **Pilot Manifest:** `{pilot_manifest_path.name}`
- **Traditional Alert File:** `{trad_file_found.name if trad_file_found else 'N/A'}`
- **DL Alert File:** `{dl_file_found.name if dl_file_found else 'N/A'}`

"""
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[LEAKAGE AUDIT] Status: {audit_result['status']}")
    print(f"  Train ∩ Pilot Overlap: {len(train_pilot_overlap)}")
    print(f"  Train ∩ Traditional Overlap: {len(train_trad_overlap)}")
    print(f"  Train ∩ DL Overlap: {len(train_dl_overlap)}")
    print(f"  Pilot Subset of Test: {pilot_in_test}")
    print(f"  Cross-Detector Flow Identity: {detectors_identical}")
    print(f"  Report written to: {json_out} and {md_out}")

    return audit_result


def main():
    parser = argparse.ArgumentParser(description="Audit train vs pilot data leakage.")
    parser.add_argument("--config", default="config/run_sample_pilot10.json", help="Path to config file")
    parser.add_argument("--reports-dir", default=None, help="Explicit reports directory to audit")
    args = parser.parse_args()

    if args.reports_dir:
        target_dir = args.reports_dir
    else:
        cfg = WorkstreamConfig.load(args.config)
        target_dir = cfg.reports_dir

    result = run_leakage_audit(target_dir)
    sys.exit(1 if result["leakage_detected"] else 0)


if __name__ == "__main__":
    main()
