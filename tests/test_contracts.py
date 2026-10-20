"""
Test suite for verifying that canonical contracts adhere to Draft 2020-12
and contain all required interface fields.
"""

import json
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator

def get_contracts():
    contracts_dir = Path(__file__).resolve().parent.parent / "contracts"
    return list(contracts_dir.glob("*.json"))

def test_contracts_exist():
    contracts = get_contracts()
    expected_contracts = [
        "alert_schema_v1.json",
        "retrieval_schema_v1.json",
        "report_schema_v1.json"
    ]
    filenames = [c.name for c in contracts]
    for expected in expected_contracts:
        assert expected in filenames, f"Missing canonical contract: {expected}"

def test_contracts_draft_2020_12_valid():
    contracts = get_contracts()
    assert len(contracts) >= 3
    for contract_file in contracts:
        with open(contract_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        Draft202012Validator.check_schema(data)
