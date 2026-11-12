"""
Regression test suite to ensure existing Hard Voting baseline outputs
remain byte-identical before and after Soft Voting implementation.
"""

import hashlib
import json
from pathlib import Path
import pytest

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots" / "baseline_hard_voting"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "detection" / "workstream" / "reports" / "aci_iot_2023"


def test_baseline_snapshot_exists():
    assert SNAPSHOT_DIR.exists(), "Baseline snapshot directory missing!"
    manifest_file = SNAPSHOT_DIR / "manifest.json"
    assert manifest_file.exists(), "Snapshot manifest.json missing!"


def test_pilot_alerts_byte_identity():
    """Verify that pilot alerts remain 100% byte-identical to the baseline Hard Voting snapshot."""
    manifest_file = SNAPSHOT_DIR / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for rel_path in ["pilot_traditional_alerts.json", "pilot_dl_alerts.json"]:
        live_file = REPORTS_DIR / rel_path
        assert live_file.exists(), f"Live file {live_file} missing!"
        
        live_bytes = live_file.read_bytes()
        live_sha = hashlib.sha256(live_bytes).hexdigest()
        expected_sha = manifest[rel_path]["sha256"]
        
        assert live_sha == expected_sha, (
            f"Regression violation: {rel_path} has diverged from baseline!\n"
            f"Expected SHA256: {expected_sha}\n"
            f"Live SHA256:     {live_sha}"
        )

