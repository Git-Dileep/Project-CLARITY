"""Validator for alert payloads against the authoritative shared schema at contracts/alert_schema_v1.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Union
import jsonschema
from jsonschema import Draft202012Validator


def find_schema_path(base_dir: Path | None = None) -> Path:
    """Locate contracts/alert_schema_v1.json by walking up directories if needed."""
    start = base_dir or Path.cwd()
    candidates = [
        start / "contracts" / "alert_schema_v1.json",
        start.parent / "contracts" / "alert_schema_v1.json",
        start.parent.parent / "contracts" / "alert_schema_v1.json",
        start.parent.parent.parent / "contracts" / "alert_schema_v1.json",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"Authoritative schema contracts/alert_schema_v1.json could not be found from {start}."
    )


def load_validator(schema_path: Path | None = None) -> Draft202012Validator:
    """Load and instantiate the Draft 2020-12 validator."""
    s_path = schema_path or find_schema_path()
    with s_path.open("r", encoding="utf-8") as f:
        schema_data = json.load(f)
    Draft202012Validator.check_schema(schema_data)
    return Draft202012Validator(schema_data)


def validate_alert_payload(
    alert: Dict[str, Any],
    validator: Draft202012Validator,
) -> Tuple[bool, List[str]]:
    """Validate a single alert object against schema, returning pass/fail and errors."""
    errors = []
    for err in sorted(validator.iter_errors(alert), key=lambda e: e.path):
        loc = ".".join(str(p) for p in err.path) if err.path else "root"
        errors.append(f"[{loc}]: {err.message}")
    return len(errors) == 0, errors


def validate_alert_file(
    file_path: str | Path,
    schema_path: Path | None = None,
) -> Dict[str, Any]:
    """Validate an alert JSON file (either a single alert object or an array of alert objects)."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Alert file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    validator = load_validator(schema_path)
    alerts = data if isinstance(data, list) else [data]

    total = len(alerts)
    passed = 0
    failed = 0
    failure_details = []

    for idx, alert in enumerate(alerts):
        is_valid, errors = validate_alert_payload(alert, validator)
        alert_id = alert.get("identity", {}).get("alert_id", f"INDEX-{idx}")
        if is_valid:
            passed += 1
        else:
            failed += 1
            failure_details.append({
                "alert_index": idx,
                "alert_id": alert_id,
                "errors": errors,
            })

    result = {
        "file": str(p),
        "total_alerts": total,
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
        "failures": failure_details,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate alert payloads against contracts/alert_schema_v1.json")
    parser.add_argument("--file", required=True, help="Path to alert JSON file")
    parser.add_argument("--schema", default=None, help="Path to alert_schema_v1.json")
    args = parser.parse_args()

    schema_p = Path(args.schema) if args.schema else None
    res = validate_alert_file(args.file, schema_p)
    print(json.dumps(res, indent=2))
    if res["status"] != "PASS":
        exit(1)


if __name__ == "__main__":
    main()
