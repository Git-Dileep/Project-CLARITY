"""Stratified pilot export for both Traditional and Deep Learning detector ensembles.

Exports:
- reports/pilot_traditional_alerts.json
- reports/pilot_dl_alerts.json
- reports/pilot_20_alerts.csv
- reports/alert_validation_report.json

Validates every alert payload against contracts/alert_schema_v1.json.
Exits with nonzero code if validation fails.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
import torch

from .config import WorkstreamConfig
from .data_loader import load_dataset
from .preprocessing import prepare_data_and_splits, SplitData
from .train_traditional import train_traditional_models
from .train_dl import train_dl_models, CNN1DModel, LSTMModel, TransformerModel, TORCH_AVAILABLE
from .explainability import explain_traditional_tree, explain_dl_gradient
from .validate_alerts import load_validator, validate_alert_payload, find_schema_path


def select_stratified_pilot_indices(
    test_df: pd.DataFrame,
    target_col: str,
    pilot_size: int = 20,
    seed: int = 42,
) -> List[int]:
    """Deterministically select stratified sample indices from the test split."""
    rng = np.random.default_rng(seed)
    classes = test_df[target_col].unique()
    n_classes = len(classes)

    chosen_indices: List[int] = []

    # Ensure at least 1 sample per class if possible
    per_class_quota = max(1, pilot_size // n_classes)
    for c in classes:
        c_indices = test_df[test_df[target_col] == c].index.to_numpy()
        if len(c_indices) > 0:
            take = min(per_class_quota, len(c_indices))
            selected = rng.choice(c_indices, size=take, replace=False)
            chosen_indices.extend(selected.tolist())

    # Fill remainder proportionally or randomly from remaining test samples
    if len(chosen_indices) < pilot_size:
        remaining = [i for i in test_df.index if i not in chosen_indices]
        if remaining:
            needed = min(pilot_size - len(chosen_indices), len(remaining))
            extra = rng.choice(remaining, size=needed, replace=False)
            chosen_indices.extend(extra.tolist())

    # Ensure exact pilot_size and sort
    final_indices = sorted(chosen_indices[:pilot_size])
    return final_indices


def format_alert_id(idx: int, dt: datetime) -> str:
    """Format alert ID strictly adhering to ^ALT-[0-9]{8}-[0-9]{4}$."""
    date_str = dt.strftime("%Y%m%d")
    return f"ALT-{date_str}-{idx+1:04d}"


def build_alert_payload(
    alert_id: str,
    flow_id: str,
    detector_family: str,
    timestamp: str,
    dataset: str,
    predicted_label: str,
    confidence: float,
    base_learner_votes: List[Dict[str, Any]],
    shap_attribution: Dict[str, Any],
    true_label: str,
    in_train_leakage_flag: bool = False,
    voting_strategy: Optional[str] = None,
    hard_pred: Optional[str] = None,
    soft_pred: Optional[str] = None,
    agreement: Optional[bool] = None,
    disagreement: Optional[bool] = None,
    model_predictions: Optional[Dict[str, str]] = None,
    model_confidences: Optional[Dict[str, float]] = None,
    soft_probabilities: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Assemble single alert conforming strictly to AlertSchemaV1 / AlertSchemaV1_1."""
    pred_dict: Dict[str, Any] = {
        "predicted_label": str(predicted_label),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "base_learner_votes": base_learner_votes,
    }
    if voting_strategy is not None:
        pred_dict["voting_strategy"] = str(voting_strategy)
    if hard_pred is not None:
        pred_dict["hard_pred"] = str(hard_pred)
    if soft_pred is not None:
        pred_dict["soft_pred"] = str(soft_pred)
    if agreement is not None:
        pred_dict["agreement"] = bool(agreement)
    if disagreement is not None:
        pred_dict["disagreement"] = bool(disagreement)
    if model_predictions is not None:
        pred_dict["model_predictions"] = model_predictions
    if model_confidences is not None:
        pred_dict["model_confidences"] = model_confidences
    if soft_probabilities is not None:
        pred_dict["soft_probabilities"] = soft_probabilities

    return {
        "identity": {
            "alert_id": alert_id,
            "flow_id": flow_id,
            "detector_family": detector_family,
            "timestamp": timestamp,
            "dataset": dataset,
            "split": "pilot",
        },
        "prediction": pred_dict,
        "shap_attribution": shap_attribution,
        "evaluation_metadata": {
            "true_label": str(true_label),
            "in_train_leakage_flag": bool(in_train_leakage_flag),
            "detector_correct": bool(str(predicted_label) == str(true_label)),
        },
    }


def export_pilot_alerts(
    config: WorkstreamConfig,
    base_dir: Optional[Path] = None,
    trad_result: Optional[Dict[str, Any]] = None,
    dl_result: Optional[Dict[str, Any]] = None,
    splits: Optional[SplitData] = None,
) -> Dict[str, Any]:
    """Execute stratified pilot export and strict validation for both detector families."""
    work_dir = base_dir or Path.cwd()
    reports_dir = config.resolve_path(work_dir, config.reports_dir)
    reports_dl_dir = reports_dir / "dl"
    models_dir = config.resolve_path(work_dir, config.models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports_dl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data & splits if not provided
    if splits is None:
        df, _ = load_dataset(config, work_dir)
        splits = prepare_data_and_splits(df, config, work_dir)

    # 2. Train or load traditional models if not provided
    trad_model_path = models_dir / "traditional_ensemble.joblib"
    if trad_result is None:
        if not trad_model_path.exists():
            print("[PILOT] Training traditional ensemble for pilot export...")
            trad_result = train_traditional_models(splits, config, work_dir)
        else:
            print("[PILOT] Loading existing traditional ensemble...")
            trad_bundle = joblib.load(trad_model_path)
            trad_result = train_traditional_models(splits, config, work_dir)

    # 3. Train or load DL models if not provided
    dl_model_path = models_dir / "dl_ensemble.pt"
    if dl_result is None:
        if TORCH_AVAILABLE and config.dl.enabled:
            print("[PILOT] Training DL ensemble for pilot export...")
            dl_result = train_dl_models(splits, config, work_dir)

    # 4. Select 20 stratified pilot samples
    pilot_indices = select_stratified_pilot_indices(
        test_df=splits.test_df,
        target_col=config.target_column,
        pilot_size=config.pilot.pilot_size,
        seed=config.random_seed,
    )
    pilot_test_df = splits.test_df.loc[pilot_indices].copy().reset_index(drop=True)
    X_pilot = splits.X_test[pilot_indices]
    y_pilot_true = [splits.class_names[i] for i in splits.y_test[pilot_indices]]

    now_dt = datetime.now(timezone.utc)
    base_timestamp = now_dt.isoformat()

    # Flow IDs
    flow_ids = []
    if config.flow_id_column and config.flow_id_column in pilot_test_df.columns:
        flow_ids = pilot_test_df[config.flow_id_column].astype(str).tolist()
    else:
        flow_ids = [f"FLW-20261020-{i+1:04d}" for i in range(len(pilot_test_df))]

    timestamps = []
    if config.timestamp_column and config.timestamp_column in pilot_test_df.columns:
        timestamps = pilot_test_df[config.timestamp_column].astype(str).tolist()
    else:
        timestamps = [base_timestamp] * len(pilot_test_df)

    # Validator
    schema_path = find_schema_path(work_dir)
    validator = load_validator(schema_path)

    validation_results = {
        "timestamp": now_dt.isoformat(),
        "schema_path": str(schema_path),
        "traditional_alerts_count": 0,
        "traditional_valid": False,
        "dl_alerts_count": 0,
        "dl_valid": False,
        "errors": [],
    }

    # ==========================
    # TRADITIONAL ALERTS EXPORT
    # ==========================
    print("\n[PILOT] Exporting Traditional ML Alerts...")
    trad_votes = [trad_result["votes"][i] for i in pilot_indices]
    trad_preds = [trad_result["test_df"].iloc[i]["ensemble_prediction"] for i in pilot_indices]
    trad_confs = [trad_result["test_df"].iloc[i]["ensemble_confidence"] for i in pilot_indices]

    # Generate TreeExplainer SHAP
    trad_model_bundle = trad_result["models"] if (trad_result and "models" in trad_result) else joblib.load(trad_model_path)
    xgb_learner = trad_model_bundle["xgb"]
    trad_shaps = explain_traditional_tree(
        model=xgb_learner,
        X_samples=X_pilot,
        feature_names=splits.feature_names,
        class_names=splits.class_names,
        predicted_classes=trad_preds,
    )

    traditional_alerts = []
    for i in range(len(pilot_test_df)):
        alt_id = format_alert_id(i, now_dt)
        learner_vote_dicts = [v.to_dict() for v in trad_votes[i]]

        payload = build_alert_payload(
            alert_id=alt_id,
            flow_id=flow_ids[i],
            detector_family="traditional_ml",
            timestamp=timestamps[i],
            dataset=config.dataset_name,
            predicted_label=trad_preds[i],
            confidence=trad_confs[i],
            base_learner_votes=learner_vote_dicts,
            shap_attribution=trad_shaps[i],
            true_label=y_pilot_true[i],
            in_train_leakage_flag=False,
        )

        is_valid, errs = validate_alert_payload(payload, validator)
        if not is_valid:
            validation_results["errors"].append({"detector": "traditional_ml", "alert_id": alt_id, "errors": errs})
        traditional_alerts.append(payload)

    trad_alerts_path = reports_dir / "pilot_traditional_alerts.json"
    with trad_alerts_path.open("w", encoding="utf-8") as f:
        json.dump(traditional_alerts, f, indent=2)

    trad_valid_count = sum(1 for a in traditional_alerts if validate_alert_payload(a, validator)[0])
    validation_results["traditional_alerts_count"] = len(traditional_alerts)
    validation_results["traditional_valid"] = trad_valid_count == len(traditional_alerts)

    # ==========================
    # DEEP LEARNING ALERTS EXPORT
    # ==========================
    dl_alerts = []
    if dl_result is not None:
        print("[PILOT] Exporting Deep Learning Ensemble Alerts...")
        dl_votes = [dl_result["votes"][i] for i in pilot_indices]
        dl_preds = [dl_result["test_df"].iloc[i]["ensemble_prediction"] for i in pilot_indices]
        dl_confs = [dl_result["test_df"].iloc[i]["ensemble_confidence"] for i in pilot_indices]

        # Use Transformer model for DL GradientExplainer
        transformer_model = dl_result["models"]["Transformer"]
        bg_samples = splits.X_train[:min(config.explainability.background_samples, len(splits.X_train))]

        dl_shaps = explain_dl_gradient(
            model=transformer_model,
            background_samples=bg_samples,
            X_samples=X_pilot,
            feature_names=splits.feature_names,
            class_names=splits.class_names,
            predicted_classes=dl_preds,
        )

        for i in range(len(pilot_test_df)):
            alt_id = format_alert_id(i + 100, now_dt)
            learner_vote_dicts = [v.to_dict() for v in dl_votes[i]]

            payload = build_alert_payload(
                alert_id=alt_id,
                flow_id=flow_ids[i],
                detector_family="dl_ensemble",
                timestamp=timestamps[i],
                dataset=config.dataset_name,
                predicted_label=dl_preds[i],
                confidence=dl_confs[i],
                base_learner_votes=learner_vote_dicts,
                shap_attribution=dl_shaps[i],
                true_label=y_pilot_true[i],
                in_train_leakage_flag=False,
            )

            is_valid, errs = validate_alert_payload(payload, validator)
            if not is_valid:
                validation_results["errors"].append({"detector": "dl_ensemble", "alert_id": alt_id, "errors": errs})
            dl_alerts.append(payload)

    # Programmatic assertion: verify identical flow IDs across detectors
    trad_flow_ids = [a["identity"]["flow_id"] for a in traditional_alerts]
    if dl_alerts:
        dl_flow_ids = [a["identity"]["flow_id"] for a in dl_alerts]
        if trad_flow_ids != dl_flow_ids:
            raise ValueError(
                f"Mismatch in pilot flow IDs! Traditional: {trad_flow_ids} vs DL: {dl_flow_ids}"
            )
        if len(set(trad_flow_ids)) != len(trad_flow_ids):
            raise ValueError(f"Duplicate flow IDs found in pilot: {trad_flow_ids}")

    # Verify at least min_classes represented
    rep_classes = sorted(list(set(y_pilot_true)))
    print(f"[PILOT] Represented classes in pilot ({len(rep_classes)}): {rep_classes}")
    if len(rep_classes) < config.pilot.min_classes:
        raise ValueError(
            f"Pilot contains only {len(rep_classes)} classes ({rep_classes}), but required at least {config.pilot.min_classes}."
        )

    # Save pilot manifest
    pilot_manifest = {
        "run_id": config.run_id or f"RUN-{config.dataset_name}",
        "split_id": config.split_id or f"SPLIT-{config.random_seed}",
        "dataset_name": config.dataset_name,
        "source_sha256": config.source_sha256,
        "selection_seed": config.random_seed,
        "pilot_size": len(pilot_indices),
        "min_classes_required": config.pilot.min_classes,
        "represented_classes_count": len(rep_classes),
        "represented_classes": rep_classes,
        "flow_ids": flow_ids,
        "sample_ids": pilot_test_df["sample_id"].tolist() if "sample_id" in pilot_test_df else [],
        "true_labels": y_pilot_true,
        "class_counts": pd.Series(y_pilot_true).value_counts().to_dict(),
        "pilot_indices_in_test": [int(i) for i in pilot_indices],
    }
    with (reports_dir / "pilot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(pilot_manifest, f, indent=2)

    df_manifest = pd.DataFrame({
        "flow_id": flow_ids,
        "sample_id": pilot_test_df["sample_id"].tolist() if "sample_id" in pilot_test_df else list(range(len(pilot_test_df))),
        "true_label": y_pilot_true,
        "timestamp": timestamps,
    })
    df_manifest.to_csv(reports_dir / "pilot_manifest.csv", index=False)

    # Save traditional alerts JSON (both standard and pilot10 named)
    trad_alerts_path = reports_dir / "pilot_traditional_alerts.json"
    trad_alerts_10_path = reports_dir / f"pilot_traditional_alerts_{len(traditional_alerts)}.json"
    with trad_alerts_path.open("w", encoding="utf-8") as f:
        json.dump(traditional_alerts, f, indent=2)
    with trad_alerts_10_path.open("w", encoding="utf-8") as f:
        json.dump(traditional_alerts, f, indent=2)

    # Save dual strategy traditional alerts
    trad_soft_preds = trad_result.get("soft_preds", trad_preds) if isinstance(trad_result, dict) else trad_preds
    trad_soft_confs = trad_result.get("soft_confs", trad_confs) if isinstance(trad_result, dict) else trad_confs
    
    trad_hard_alerts = []
    trad_soft_alerts = []
    for i in range(len(traditional_alerts)):
        base_a = traditional_alerts[i]
        hp = trad_preds[i]
        sp = trad_soft_preds[i] if i < len(trad_soft_preds) else hp
        ag = (hp == sp)
        
        # Hard alert with v1.1 fields
        h_payload = json.loads(json.dumps(base_a))
        h_payload["prediction"]["voting_strategy"] = "hard_voting"
        h_payload["prediction"]["hard_pred"] = hp
        h_payload["prediction"]["soft_pred"] = sp
        h_payload["prediction"]["agreement"] = ag
        h_payload["prediction"]["disagreement"] = not ag
        trad_hard_alerts.append(h_payload)

        # Soft alert with v1.1 fields
        s_payload = json.loads(json.dumps(base_a))
        s_payload["prediction"]["voting_strategy"] = "soft_voting"
        s_payload["prediction"]["predicted_label"] = sp
        s_payload["prediction"]["confidence"] = float(trad_soft_confs[i]) if i < len(trad_soft_confs) else float(base_a["prediction"]["confidence"])
        s_payload["prediction"]["hard_pred"] = hp
        s_payload["prediction"]["soft_pred"] = sp
        s_payload["prediction"]["agreement"] = ag
        s_payload["prediction"]["disagreement"] = not ag
        s_payload["evaluation_metadata"]["detector_correct"] = bool(sp == y_pilot_true[i])
        trad_soft_alerts.append(s_payload)

    with (reports_dir / "pilot_traditional_alerts_hard.json").open("w", encoding="utf-8") as f:
        json.dump(trad_hard_alerts, f, indent=2)
    with (reports_dir / "pilot_traditional_alerts_soft.json").open("w", encoding="utf-8") as f:
        json.dump(trad_soft_alerts, f, indent=2)

    trad_valid_count = sum(1 for a in traditional_alerts if validate_alert_payload(a, validator)[0])
    validation_results["traditional_alerts_count"] = len(traditional_alerts)
    validation_results["traditional_valid"] = trad_valid_count == len(traditional_alerts)

    # Save DL alerts JSON (both standard and pilot10 named)
    if dl_alerts:
        dl_alerts_path = reports_dir / "pilot_dl_alerts.json"
        dl_alerts_10_path = reports_dir / f"pilot_dl_alerts_{len(dl_alerts)}.json"
        with dl_alerts_path.open("w", encoding="utf-8") as f:
            json.dump(dl_alerts, f, indent=2)
        with dl_alerts_10_path.open("w", encoding="utf-8") as f:
            json.dump(dl_alerts, f, indent=2)

        dl_soft_preds = dl_result.get("soft_preds", dl_preds) if isinstance(dl_result, dict) else dl_preds
        dl_soft_confs = dl_result.get("soft_confs", dl_confs) if isinstance(dl_result, dict) else dl_confs

        dl_hard_alerts = []
        dl_soft_alerts = []
        for i in range(len(dl_alerts)):
            base_a = dl_alerts[i]
            hp = dl_preds[i]
            sp = dl_soft_preds[i] if i < len(dl_soft_preds) else hp
            ag = (hp == sp)

            h_payload = json.loads(json.dumps(base_a))
            h_payload["prediction"]["voting_strategy"] = "hard_voting"
            h_payload["prediction"]["hard_pred"] = hp
            h_payload["prediction"]["soft_pred"] = sp
            h_payload["prediction"]["agreement"] = ag
            h_payload["prediction"]["disagreement"] = not ag
            dl_hard_alerts.append(h_payload)

            s_payload = json.loads(json.dumps(base_a))
            s_payload["prediction"]["voting_strategy"] = "soft_voting"
            s_payload["prediction"]["predicted_label"] = sp
            s_payload["prediction"]["confidence"] = float(dl_soft_confs[i]) if i < len(dl_soft_confs) else float(base_a["prediction"]["confidence"])
            s_payload["prediction"]["hard_pred"] = hp
            s_payload["prediction"]["soft_pred"] = sp
            s_payload["prediction"]["agreement"] = ag
            s_payload["prediction"]["disagreement"] = not ag
            s_payload["evaluation_metadata"]["detector_correct"] = bool(sp == y_pilot_true[i])
            dl_soft_alerts.append(s_payload)

        with (reports_dir / "pilot_dl_alerts_hard.json").open("w", encoding="utf-8") as f:
            json.dump(dl_hard_alerts, f, indent=2)
        with (reports_dir / "pilot_dl_alerts_soft.json").open("w", encoding="utf-8") as f:
            json.dump(dl_soft_alerts, f, indent=2)

        dl_valid_count = sum(1 for a in dl_alerts if validate_alert_payload(a, validator)[0])
        validation_results["dl_alerts_count"] = len(dl_alerts)
        validation_results["dl_valid"] = dl_valid_count == len(dl_alerts)

    # ==========================
    # TABULAR PILOT SUMMARY CSV
    # ==========================
    tabular_rows = []
    for i in range(len(pilot_test_df)):
        t_alert = traditional_alerts[i]
        t_hard = trad_hard_alerts[i]
        t_soft = trad_soft_alerts[i]
        row = {
            "flow_id": flow_ids[i],
            "timestamp": timestamps[i],
            "true_label": y_pilot_true[i],
            "alert_id_trad": t_alert["identity"]["alert_id"],
            "traditional_prediction": t_alert["prediction"]["predicted_label"],
            "traditional_confidence": t_alert["prediction"]["confidence"],
            "traditional_hard_prediction": t_hard["prediction"]["hard_pred"],
            "traditional_hard_confidence": t_hard["prediction"]["confidence"],
            "traditional_soft_prediction": t_soft["prediction"]["soft_pred"],
            "traditional_soft_confidence": t_soft["prediction"]["confidence"],
            "traditional_voting_agreement": t_hard["prediction"]["agreement"],
            "traditional_correct": t_alert["evaluation_metadata"]["detector_correct"],
        }
        if dl_alerts:
            d_alert = dl_alerts[i]
            d_hard = dl_hard_alerts[i]
            d_soft = dl_soft_alerts[i]
            row["alert_id_dl"] = d_alert["identity"]["alert_id"]
            row["dl_prediction"] = d_alert["prediction"]["predicted_label"]
            row["dl_confidence"] = d_alert["prediction"]["confidence"]
            row["dl_hard_prediction"] = d_hard["prediction"]["hard_pred"]
            row["dl_hard_confidence"] = d_hard["prediction"]["confidence"]
            row["dl_soft_prediction"] = d_soft["prediction"]["soft_pred"]
            row["dl_soft_confidence"] = d_soft["prediction"]["confidence"]
            row["dl_voting_agreement"] = d_hard["prediction"]["agreement"]
            row["dl_correct"] = d_alert["evaluation_metadata"]["detector_correct"]
        tabular_rows.append(row)

    df_pilot = pd.DataFrame(tabular_rows)
    df_pilot.to_csv(reports_dir / f"pilot_{len(pilot_test_df)}_alerts.csv", index=False)
    df_pilot.to_csv(reports_dir / "pilot_20_alerts.csv", index=False)

    # Save validation report
    val_report_path = reports_dir / "alert_validation_report.json"
    with val_report_path.open("w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    print("\n[PILOT] Pilot Export Summary:")
    print(f"  Traditional Alerts Exported: {len(traditional_alerts)} (Valid: {validation_results['traditional_valid']})")
    print(f"  DL Alerts Exported: {len(dl_alerts)} (Valid: {validation_results['dl_valid']})")
    print(f"  Validation Report Saved: {val_report_path}")
    print(f"  Pilot Manifest Saved: {reports_dir / 'pilot_manifest.json'}")

    if not validation_results["traditional_valid"] or (dl_alerts and not validation_results["dl_valid"]):
        raise ValueError(f"Alert schema validation failed! Details in {val_report_path}: {validation_results['errors']}")

    return {
        "traditional_alerts": traditional_alerts,
        "dl_alerts": dl_alerts,
        "validation_report": validation_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Export stratified 20-alert pilot with schema validation.")
    parser.add_argument("--config", default="config/reference.json", help="Path to config file")
    args = parser.parse_args()

    cfg = WorkstreamConfig.load(args.config)
    export_pilot_alerts(cfg)


if __name__ == "__main__":
    main()
