"""Schema and Semantic Validation Script for Project CLARITY Alert Artifacts.

Validates alert JSON files against contracts/alert_schema_v1.json using
jsonschema.Draft202012Validator with format checking, followed by strict semantic
validation checks:
1. Exact alert count (10 alerts per detector family).
2. Unique ALT-YYYYMMDD-XXXX alert IDs.
3. Matching flow IDs between Traditional ML and DL ensembles.
4. Class representation diversity (>= 5 unique classes in 10-flow pilot).
5. Exactly 10 canonical features ranked 1..10 in descending absolute SHAP magnitude.
6. Absolute monotonic ranking order for SHAP attributions.
7. Absence of label/identity leakage in SHAP feature names.
8. Zero train leakage flag (in_train_leakage_flag == False).
9. Correctness boolean consistency: detector_correct == (predicted_label == true_label).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema import Draft202012Validator, FormatChecker


ALERT_ID_REGEX = re.compile(r"^ALT-[0-9]{8}-[0-9]{4}$")
FORBIDDEN_FEATURE_NAMES = {
    "true_label",
    "label",
    "binary_label",
    "attack_cat",
    "flow_id",
    "alert_id",
    "in_train_leakage_flag",
    "detector_correct",
}


def load_schema(schema_path: Path) -> Dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Alert schema contract not found: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_file_against_schema(
    file_path: Path, validator: Draft202012Validator
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Validate JSON alert array against Draft202012Validator."""
    errors: List[str] = []
    if not file_path.exists():
        return [f"File not found: {file_path}"], []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return [f"JSON parse error in {file_path}: {exc}"], []

    if not isinstance(data, list):
        return [f"Root of {file_path} must be a JSON array, got {type(data).__name__}"], []

    for idx, alert in enumerate(data):
        schema_errors = list(validator.iter_errors(alert))
        for err in schema_errors:
            errors.append(f"Alert [{idx}] ({alert.get('identity', {}).get('alert_id', 'UNKNOWN')}): {err.message} at path '{err.json_path}'")

    return errors, data


def validate_semantics(
    alerts: List[Dict[str, Any]],
    expected_family: Optional[str] = None,
    expected_count: int = 10,
) -> List[str]:
    """Execute deep semantic integrity assertions on parsed alert objects."""
    errors: List[str] = []

    if len(alerts) != expected_count:
        errors.append(f"Expected exactly {expected_count} alerts, but found {len(alerts)}")

    seen_alert_ids = set()
    seen_flow_ids = set()

    for idx, alert in enumerate(alerts):
        identity = alert.get("identity", {})
        prediction = alert.get("prediction", {})
        attribution = alert.get("shap_attribution", {})
        eval_meta = alert.get("evaluation_metadata", {})

        # 1. Identity checks
        alert_id = identity.get("alert_id", "")
        flow_id = identity.get("flow_id", "")
        detector_family = identity.get("detector_family", "")

        if not ALERT_ID_REGEX.match(alert_id):
            errors.append(f"Alert [{idx}]: alert_id '{alert_id}' does not match pattern ALT-YYYYMMDD-XXXX")
        if alert_id in seen_alert_ids:
            errors.append(f"Alert [{idx}]: duplicate alert_id '{alert_id}'")
        seen_alert_ids.add(alert_id)

        if not flow_id:
            errors.append(f"Alert [{idx}]: empty flow_id")
        if flow_id in seen_flow_ids:
            errors.append(f"Alert [{idx}]: duplicate flow_id '{flow_id}' within same export")
        seen_flow_ids.add(flow_id)

        if expected_family and detector_family != expected_family:
            errors.append(f"Alert [{idx}]: expected detector_family '{expected_family}', got '{detector_family}'")

        # 2. Prediction checks
        pred_label = prediction.get("predicted_label", "")
        confidence = prediction.get("confidence", -1.0)
        votes = prediction.get("base_learner_votes", [])

        if not pred_label:
            errors.append(f"Alert [{idx}]: empty predicted_label")
        if not (0.0 <= confidence <= 1.0):
            errors.append(f"Alert [{idx}]: confidence {confidence} outside [0.0, 1.0]")
        if len(votes) != 3:
            errors.append(f"Alert [{idx}]: expected 3 base_learner_votes, got {len(votes)}")

        # 3. SHAP attribution checks
        top_features = attribution.get("top_features", [])
        if len(top_features) != 10:
            errors.append(f"Alert [{idx}]: expected exactly 10 top_features, got {len(top_features)}")

        prev_abs_shap = float("inf")
        for f_idx, feat in enumerate(top_features):
            rank = feat.get("rank")
            fname = feat.get("feature_name", "")
            shap_val = feat.get("shap_value", 0.0)

            if rank != f_idx + 1:
                errors.append(f"Alert [{idx}] Feature [{f_idx}]: expected rank {f_idx + 1}, got {rank}")

            if fname.lower() in FORBIDDEN_FEATURE_NAMES:
                errors.append(f"Alert [{idx}] Feature [{f_idx}]: forbidden feature name leakage '{fname}'")

            curr_abs_shap = abs(shap_val)
            # Allow minor numerical precision epsilon of 1e-6
            if curr_abs_shap > prev_abs_shap + 1e-6:
                errors.append(
                    f"Alert [{idx}] Feature [{f_idx}] '{fname}': non-monotonic SHAP magnitude "
                    f"|{shap_val}| > previous |{prev_abs_shap}|"
                )
            prev_abs_shap = curr_abs_shap

        # 4. Evaluation metadata checks
        true_label = eval_meta.get("true_label", "")
        in_train_leakage = eval_meta.get("in_train_leakage_flag", True)
        detector_correct = eval_meta.get("detector_correct", None)

        if not true_label:
            errors.append(f"Alert [{idx}]: empty true_label in evaluation_metadata")
        if in_train_leakage is not False:
            errors.append(f"Alert [{idx}]: in_train_leakage_flag must be False for held-out evaluation, got {in_train_leakage}")

        expected_correct = (pred_label == true_label)
        if detector_correct != expected_correct:
            errors.append(
                f"Alert [{idx}]: detector_correct ({detector_correct}) != (pred '{pred_label}' == true '{true_label}')"
            )

    return errors


def validate_cross_detector_consistency(
    trad_alerts: List[Dict[str, Any]], dl_alerts: List[Dict[str, Any]]
) -> List[str]:
    """Validate that Traditional ML and DL ensembles evaluated the exact same flows."""
    errors: List[str] = []

    if len(trad_alerts) != len(dl_alerts):
        errors.append(f"Cross-detector count mismatch: Traditional has {len(trad_alerts)}, DL has {len(dl_alerts)}")
        return errors

    trad_flow_ids = [a["identity"]["flow_id"] for a in trad_alerts]
    dl_flow_ids = [a["identity"]["flow_id"] for a in dl_alerts]

    if trad_flow_ids != dl_flow_ids:
        errors.append("Traditional and DL alerts do not have matching flow_ids in the same order")
        diff = set(trad_flow_ids) ^ set(dl_flow_ids)
        if diff:
            errors.append(f"Flow ID symmetric difference: {diff}")

    # Check true_label consistency across detectors
    for i in range(len(trad_alerts)):
        t_flow = trad_alerts[i]["identity"]["flow_id"]
        d_flow = dl_alerts[i]["identity"]["flow_id"]
        t_label = trad_alerts[i]["evaluation_metadata"]["true_label"]
        d_label = dl_alerts[i]["evaluation_metadata"]["true_label"]
        if t_flow == d_flow and t_label != d_label:
            errors.append(f"True label mismatch for flow '{t_flow}': Traditional='{t_label}', DL='{d_label}'")

    # Class diversity check: >= 5 unique classes
    trad_classes = set(a["evaluation_metadata"]["true_label"] for a in trad_alerts)
    if len(trad_classes) < 5:
        errors.append(f"Pilot class diversity insufficient: expected >= 5 unique classes, got {len(trad_classes)}: {trad_classes}")

    return errors


def run_full_validation(
    schema_path: Path,
    trad_path: Optional[Path] = None,
    dl_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
    expected_count: int = 10,
) -> Dict[str, Any]:
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    all_errors: Dict[str, List[str]] = {
        "traditional_schema_errors": [],
        "traditional_semantic_errors": [],
        "dl_schema_errors": [],
        "dl_semantic_errors": [],
        "cross_detector_errors": [],
    }

    trad_alerts: List[Dict[str, Any]] = []
    dl_alerts: List[Dict[str, Any]] = []

    if trad_path and trad_path.exists():
        s_errs, trad_alerts = validate_file_against_schema(trad_path, validator)
        all_errors["traditional_schema_errors"].extend(s_errs)
        sem_errs = validate_semantics(trad_alerts, expected_family="traditional_ml", expected_count=expected_count)
        all_errors["traditional_semantic_errors"].extend(sem_errs)
    elif trad_path:
        all_errors["traditional_schema_errors"].append(f"Traditional alert file not found: {trad_path}")

    if dl_path and dl_path.exists():
        s_errs, dl_alerts = validate_file_against_schema(dl_path, validator)
        all_errors["dl_schema_errors"].extend(s_errs)
        sem_errs = validate_semantics(dl_alerts, expected_family="dl_ensemble", expected_count=expected_count)
        all_errors["dl_semantic_errors"].extend(sem_errs)
    elif dl_path:
        all_errors["dl_schema_errors"].append(f"DL alert file not found: {dl_path}")

    if trad_alerts and dl_alerts:
        cross_errs = validate_cross_detector_consistency(trad_alerts, dl_alerts)
        all_errors["cross_detector_errors"].extend(cross_errs)

    total_error_count = sum(len(err_list) for err_list in all_errors.values())
    status = "PASSED" if total_error_count == 0 else "FAILED"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "total_errors": total_error_count,
        "files_validated": {
            "schema": str(schema_path),
            "traditional_alerts": str(trad_path) if trad_path else None,
            "dl_alerts": str(dl_path) if dl_path else None,
        },
        "alerts_count": {
            "traditional": len(trad_alerts),
            "dl": len(dl_alerts),
        },
        "classes_represented": sorted(list(set(a["evaluation_metadata"]["true_label"] for a in trad_alerts))) if trad_alerts else [],
        "errors": all_errors,
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Validate alert JSON artifacts against Draft 2020-12 schema.")
    parser.add_argument(
        "--schema",
        default=str(repo_root / "contracts" / "alert_schema_v1.json"),
        help="Path to alert_schema_v1.json",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to run config file to infer reports directory",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Directory containing exported alert JSON files",
    )
    parser.add_argument(
        "--trad-file",
        default=None,
        help="Path to Traditional ML alert JSON",
    )
    parser.add_argument(
        "--dl-file",
        default=None,
        help="Path to Deep Learning alert JSON",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=10,
        help="Expected number of alerts per file (default: 10)",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)

    # Determine reports directory
    reports_dir: Optional[Path] = None
    if args.reports_dir:
        reports_dir = Path(args.reports_dir).resolve()
    elif args.config:
        cfg_path = Path(args.config).resolve()
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg_rdir = cfg_dict.get("reports_dir", "reports")
        cand1 = cfg_path.parent.parent / cfg_rdir
        cand2 = Path.cwd() / cfg_rdir
        if cand1.exists():
            reports_dir = cand1
        elif cand2.exists():
            reports_dir = cand2
        else:
            reports_dir = cand1

    trad_path = Path(args.trad_file) if args.trad_file else None
    dl_path = Path(args.dl_file) if args.dl_file else None

    if reports_dir and not trad_path:
        for fname in ["pilot_traditional_alerts_10.json", "pilot_traditional_alerts.json"]:
            p = reports_dir / fname
            if p.exists():
                trad_path = p
                break

    if reports_dir and not dl_path:
        for fname in ["pilot_dl_alerts_10.json", "pilot_dl_alerts.json"]:
            p = reports_dir / fname
            if p.exists():
                dl_path = p
                break

    output_report_path = (reports_dir / "schema_and_semantic_validation_report.json") if reports_dir else None

    print(f"Validating alert artifacts against: {schema_path}")
    print(f"Traditional Alerts: {trad_path}")
    print(f"DL Alerts:          {dl_path}")

    report = run_full_validation(
        schema_path=schema_path,
        trad_path=trad_path,
        dl_path=dl_path,
        output_report_path=output_report_path,
        expected_count=args.expected_count,
    )

    print(f"\n[VALIDATION RESULT] Status: {report['status']}")
    print(f"  Traditional Schema Errors: {len(report['errors']['traditional_schema_errors'])}")
    print(f"  Traditional Semantic Errors: {len(report['errors']['traditional_semantic_errors'])}")
    print(f"  DL Schema Errors: {len(report['errors']['dl_schema_errors'])}")
    print(f"  DL Semantic Errors: {len(report['errors']['dl_semantic_errors'])}")
    print(f"  Cross-Detector Errors: {len(report['errors']['cross_detector_errors'])}")
    print(f"  Total Errors: {report['total_errors']}")
    print(f"  Classes Represented ({len(report['classes_represented'])}): {report['classes_represented']}")

    if output_report_path:
        print(f"  Validation Report written to: {output_report_path}")

    if report["total_errors"] > 0:
        for category, errs in report["errors"].items():
            if errs:
                print(f"\n--- {category} ---")
                for e in errs[:10]:
                    print(f"  * {e}")
                if len(errs) > 10:
                    print(f"  ... and {len(errs) - 10} more")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
