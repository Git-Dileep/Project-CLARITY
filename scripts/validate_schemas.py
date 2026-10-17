#!/usr/bin/env python3
"""
Validation script for incident alert payloads against alert_schema_v1.json.
"""

import json
import sys
from pathlib import Path
import jsonschema
from jsonschema import validate, Draft202012Validator

def main():
    base_dir = Path(__file__).resolve().parent.parent
    schema_path = base_dir / "schema" / "alert_schema_v1.json"
    examples_dir = base_dir / "schema" / "examples"

    if not schema_path.exists():
        print(f"[ERROR] Schema file not found at: {schema_path}")
        sys.exit(1)

    print(f"[INFO] Loading schema: {schema_path.name}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # Validate the schema itself against Draft 2020-12
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    print("[SUCCESS] Schema definition is valid Draft 2020-12.")

    example_files = list(examples_dir.glob("*.json"))
    if not example_files:
        print(f"[WARN] No example files found in {examples_dir}")
        sys.exit(1)

    all_passed = True
    for example_file in sorted(example_files):
        print(f"\n[VALIDATING] {example_file.name} ...")
        with open(example_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  [FAIL] JSON syntax error: {e}")
                all_passed = False
                continue

        errors = list(validator.iter_errors(data))
        if errors:
            all_passed = False
            print(f"  [FAIL] Validation failed with {len(errors)} error(s):")
            for idx, err in enumerate(errors, 1):
                loc = " -> ".join([str(p) for p in err.path]) or "root"
                print(f"    {idx}. Location: '{loc}' | Error: {err.message}")
        else:
            print(f"  [PASS] Successfully validated against {schema_path.name}")

    print("\n" + "=" * 50)
    if all_passed:
        print("[SUMMARY] ALL EXAMPLE ALERTS CONFORM TO ALERT_SCHEMA_V1.")
        sys.exit(0)
    else:
        print("[SUMMARY] SCHEMA VALIDATION ERRORS DETECTED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
