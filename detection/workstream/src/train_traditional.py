"""Train traditional ML ensemble (Random Forest, XGBoost, and MLP) with hard-voting consensus.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from .config import WorkstreamConfig
from .data_loader import load_dataset
from .preprocessing import prepare_data_and_splits, SplitData
from .ensemble import (
    LearnerVote,
    compute_ensemble_predictions,
    compute_comprehensive_metrics,
    assert_class_alignment,
    compute_soft_voting_predictions,
    compute_voting_discordance,
    compute_mcnemar_test,
    compute_paired_bootstrap_macro_f1,
)


def train_traditional_models(
    split_data: SplitData,
    config: WorkstreamConfig,
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train Random Forest, XGBoost, and MLP classifiers and compute ensemble predictions."""
    work_dir = base_dir or Path.cwd()
    reports_dir = config.resolve_path(work_dir, config.reports_dir)
    models_dir = config.resolve_path(work_dir, config.models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train = split_data.X_train
    y_train = split_data.y_train
    X_test = split_data.X_test
    y_test = split_data.y_test
    classes = split_data.class_names

    # 1. Random Forest
    rf = RandomForestClassifier(
        n_estimators=config.traditional.rf_n_estimators,
        max_depth=config.traditional.rf_max_depth,
        random_state=config.random_seed,
        n_jobs=-1,
    )
    print("\n[TRADITIONAL] Training Random Forest...")
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    rf_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    rf_probs = rf.predict_proba(X_test)
    rf_infer_time = time.perf_counter() - t0
    rf_preds_idx = np.argmax(rf_probs, axis=1)
    rf_confs = np.max(rf_probs, axis=1)
    rf_preds = [classes[i] for i in rf_preds_idx]

    # 2. XGBoost
    xgb = None
    xgb_train_time = 0.0
    xgb_infer_time = 0.0
    xgb_installed = False
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=config.traditional.xgb_n_estimators,
            max_depth=config.traditional.xgb_max_depth,
            learning_rate=config.traditional.xgb_learning_rate,
            eval_metric="mlogloss",
            random_state=config.random_seed,
            n_jobs=-1,
        )
        print("[TRADITIONAL] Training XGBoost...")
        t0 = time.perf_counter()
        xgb.fit(X_train, y_train)
        xgb_train_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        xgb_probs = xgb.predict_proba(X_test)
        xgb_infer_time = time.perf_counter() - t0
        xgb_preds_idx = np.argmax(xgb_probs, axis=1)
        xgb_confs = np.max(xgb_probs, axis=1)
        xgb_preds = [classes[i] for i in xgb_preds_idx]
        xgb_installed = True
    except ImportError:
        print("[WARN] XGBoost is not installed. To install: pip install xgboost")
        raise RuntimeError("XGBoost is required for the 3-model traditional ensemble. Install via: pip install xgboost")

    # 3. Multi-layer Perceptron (MLP)
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(config.traditional.mlp_hidden_layer_sizes),
        max_iter=config.traditional.mlp_max_iter,
        early_stopping=config.traditional.mlp_early_stopping,
        random_state=config.random_seed,
    )
    print("[TRADITIONAL] Training MLP Classifier...")
    t0 = time.perf_counter()
    mlp.fit(X_train, y_train)
    mlp_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    mlp_probs = mlp.predict_proba(X_test)
    mlp_infer_time = time.perf_counter() - t0
    mlp_preds_idx = np.argmax(mlp_probs, axis=1)
    mlp_confs = np.max(mlp_probs, axis=1)
    mlp_preds = [classes[i] for i in mlp_preds_idx]

    # Assert class alignment across all three members
    rf_classes = [classes[i] for i in rf.classes_]
    xgb_classes = [classes[i] for i in xgb.classes_]
    mlp_classes = [classes[i] for i in mlp.classes_]
    assert_class_alignment(
        {"RandomForest": rf_classes, "XGBoost": xgb_classes, "MLP": mlp_classes},
        classes
    )

    # 1. Hard Voting (Preserve existing implementation exactly)
    all_votes: List[List[LearnerVote]] = []
    for i in range(len(X_test)):
        sample_votes = [
            LearnerVote(learner_name="RandomForest", voted_label=rf_preds[i], learner_confidence=float(rf_confs[i])),
            LearnerVote(learner_name="XGBoost", voted_label=xgb_preds[i], learner_confidence=float(xgb_confs[i])),
            LearnerVote(learner_name="MLP", voted_label=mlp_preds[i], learner_confidence=float(mlp_confs[i])),
        ]
        all_votes.append(sample_votes)

    hard_preds, hard_confs = compute_ensemble_predictions(all_votes)

    # 2. Soft Voting (Uniform mean of posterior probability matrices)
    learner_probs = {
        "RandomForest": rf_probs,
        "XGBoost": xgb_probs,
        "MLP": mlp_probs,
    }
    soft_preds, soft_confs, mean_probs = compute_soft_voting_predictions(learner_probs, classes)

    true_labels = [classes[i] for i in y_test]
    eval_11_classes = [c for c in classes if c != "ARP Spoofing"]

    # Comparative Inference & Disagreement Analysis
    discordance_stats = compute_voting_discordance(hard_preds, soft_preds)
    mcnemar_stats = compute_mcnemar_test(true_labels, hard_preds, soft_preds)
    bootstrap_stats = compute_paired_bootstrap_macro_f1(
        true_labels, hard_preds, soft_preds, eval_classes=eval_11_classes, seed=config.random_seed
    )

    # Metrics calculation for both strategies
    hard_ensemble_metrics = compute_comprehensive_metrics(
        y_true=true_labels,
        y_pred=hard_preds,
        class_names=classes,
        train_duration=rf_train_time + xgb_train_time + mlp_train_time,
        infer_duration=rf_infer_time + xgb_infer_time + mlp_infer_time,
        excluded_from_primary="ARP Spoofing",
    )
    soft_ensemble_metrics = compute_comprehensive_metrics(
        y_true=true_labels,
        y_pred=soft_preds,
        class_names=classes,
        train_duration=rf_train_time + xgb_train_time + mlp_train_time,
        infer_duration=rf_infer_time + xgb_infer_time + mlp_infer_time,
        excluded_from_primary="ARP Spoofing",
    )

    rf_metrics = compute_comprehensive_metrics(true_labels, rf_preds, classes, rf_train_time, rf_infer_time, excluded_from_primary="ARP Spoofing")
    xgb_metrics = compute_comprehensive_metrics(true_labels, xgb_preds, classes, xgb_train_time, xgb_infer_time, excluded_from_primary="ARP Spoofing")
    mlp_metrics = compute_comprehensive_metrics(true_labels, mlp_preds, classes, mlp_train_time, mlp_infer_time, excluded_from_primary="ARP Spoofing")

    # Save per-model probability matrices
    np.savez_compressed(
        reports_dir / "traditional_probabilities.npz",
        rf_probs=rf_probs,
        xgb_probs=xgb_probs,
        mlp_probs=mlp_probs,
        mean_probs=mean_probs,
        classes=classes,
    )

    # Build predictions dataframe
    test_df = split_data.test_df.copy()
    test_df["true_label"] = true_labels
    test_df["ensemble_prediction"] = hard_preds
    test_df["ensemble_confidence"] = hard_confs
    test_df["hard_prediction"] = hard_preds
    test_df["hard_confidence"] = hard_confs
    test_df["soft_prediction"] = soft_preds
    test_df["soft_confidence"] = soft_confs
    test_df["voting_agreement"] = [h == s for h, s in zip(hard_preds, soft_preds)]
    test_df["rf_prediction"] = rf_preds
    test_df["rf_confidence"] = rf_confs
    test_df["xgb_prediction"] = xgb_preds
    test_df["xgb_confidence"] = xgb_confs
    test_df["mlp_prediction"] = mlp_preds
    test_df["mlp_confidence"] = mlp_confs
    test_df["ensemble_correct"] = test_df["true_label"] == test_df["ensemble_prediction"]
    test_df["hard_correct"] = test_df["true_label"] == test_df["hard_prediction"]
    test_df["soft_correct"] = test_df["true_label"] == test_df["soft_prediction"]

    pred_export_cols = [
        c for c in [
            "sample_id", "flow_id", "timestamp", "device_id", "true_label",
            "ensemble_prediction", "ensemble_confidence",
            "hard_prediction", "hard_confidence",
            "soft_prediction", "soft_confidence",
            "voting_agreement",
            "rf_prediction", "rf_confidence",
            "xgb_prediction", "xgb_confidence",
            "mlp_prediction", "mlp_confidence",
            "ensemble_correct", "hard_correct", "soft_correct"
        ] if c in test_df.columns
    ]
    test_predictions_path = reports_dir / "traditional_test_predictions.csv"
    test_df[pred_export_cols].to_csv(test_predictions_path, index=False)

    # Save root backward-compatible reports
    pd.DataFrame(hard_ensemble_metrics["classification_report"]).T.to_csv(
        reports_dir / "traditional_classification_report.csv", index=True, index_label="class"
    )
    pd.DataFrame(
        hard_ensemble_metrics["confusion_matrix"], index=classes, columns=classes
    ).to_csv(reports_dir / "traditional_confusion_matrix.csv")

    summary_metrics = {
        "ensemble": hard_ensemble_metrics,
        "ensemble_hard": hard_ensemble_metrics,
        "ensemble_soft": soft_ensemble_metrics,
        "inference_comparison": {
            "discordance_rate": discordance_stats["discordance_rate"],
            "discordant_count": discordance_stats["discordant_count"],
            "mcnemar_test": mcnemar_stats,
            "paired_bootstrap_11class": bootstrap_stats,
        },
        "learners": {
            "RandomForest": rf_metrics,
            "XGBoost": xgb_metrics,
            "MLP": mlp_metrics,
        },
        "config": {
            "dataset_name": config.dataset_name,
            "is_synthetic": config.is_synthetic,
            "features_used": split_data.feature_names,
            "classes": classes,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
    }

    with (reports_dir / "traditional_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # Tabular summary metrics CSV (both root and dual dirs)
    metric_rows = [
        {"model": "Ensemble (Hard Voting)", **{k: hard_ensemble_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
        {"model": "Ensemble (Soft Voting)", **{k: soft_ensemble_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
        {"model": "RandomForest", **{k: rf_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
        {"model": "XGBoost", **{k: xgb_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
        {"model": "MLP", **{k: mlp_metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_duration_seconds", "infer_duration_seconds"]}},
    ]
    pd.DataFrame(metric_rows).to_csv(reports_dir / "traditional_metrics.csv", index=False)

    # Export separate results/hard/ and results/soft/ directories
    results_hard_dir = reports_dir / "results" / "hard"
    results_soft_dir = reports_dir / "results" / "soft"
    results_hard_dir.mkdir(parents=True, exist_ok=True)
    results_soft_dir.mkdir(parents=True, exist_ok=True)

    # results/hard/
    with (results_hard_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(hard_ensemble_metrics, f, indent=2)
    pd.DataFrame(hard_ensemble_metrics["classification_report"]).T.to_csv(
        results_hard_dir / "classification_report.csv", index=True, index_label="class"
    )
    pd.DataFrame(
        hard_ensemble_metrics["confusion_matrix"], index=classes, columns=classes
    ).to_csv(results_hard_dir / "confusion_matrix.csv")
    test_df.to_csv(results_hard_dir / "test_predictions.csv", index=False)

    # results/soft/
    with (results_soft_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(soft_ensemble_metrics, f, indent=2)
    pd.DataFrame(soft_ensemble_metrics["classification_report"]).T.to_csv(
        results_soft_dir / "classification_report.csv", index=True, index_label="class"
    )
    pd.DataFrame(
        soft_ensemble_metrics["confusion_matrix"], index=classes, columns=classes
    ).to_csv(results_soft_dir / "confusion_matrix.csv")
    test_df.to_csv(results_soft_dir / "test_predictions.csv", index=False)

    # Run Manifest logging seeds, hashes, and comparative statistics
    run_manifest = {
        "dataset_name": config.dataset_name,
        "random_seed": config.random_seed,
        "detector_family": "traditional_ml",
        "split_method": "stratified_random_80_20",
        "leakage_limitation_note": "Split is stratified random per-flow with flow identifiers dropped. Temporal/session-window grouping was not enforced during traffic capture, so flows from the same scan activity can appear in both train and test splits.",
        "discordance_rate": discordance_stats["discordance_rate"],
        "discordant_count": discordance_stats["discordant_count"],
        "total_test_samples": len(X_test),
        "primary_inference_paired_bootstrap_11class": bootstrap_stats,
        "secondary_inference_mcnemar_test": mcnemar_stats,
        "bonferroni_adjusted_alpha": 0.025,
        "hard_metrics_summary": {
            "macro_f1_11class_primary": hard_ensemble_metrics.get("macro_f1_11class_primary"),
            "macro_f1_12class_secondary": hard_ensemble_metrics.get("macro_f1_12class_secondary"),
            "accuracy": hard_ensemble_metrics["accuracy"],
            "weighted_f1": hard_ensemble_metrics["weighted_f1"],
        },
        "soft_metrics_summary": {
            "macro_f1_11class_primary": soft_ensemble_metrics.get("macro_f1_11class_primary"),
            "macro_f1_12class_secondary": soft_ensemble_metrics.get("macro_f1_12class_secondary"),
            "accuracy": soft_ensemble_metrics["accuracy"],
            "weighted_f1": soft_ensemble_metrics["weighted_f1"],
        },
    }
    with (reports_dir / "results" / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    # Save models locally
    model_bundle = {
        "rf": rf,
        "xgb": xgb,
        "mlp": mlp,
        "preprocessor": split_data.preprocessor,
        "label_encoder": split_data.label_encoder,
        "feature_names": split_data.feature_names,
        "classes": classes,
    }
    joblib.dump(model_bundle, models_dir / "traditional_ensemble.joblib")

    print("\n[TRADITIONAL] Ensemble Training Completed Successfully!")
    print(f"  Hard Ensemble Accuracy: {hard_ensemble_metrics['accuracy']:.4f} | Macro F1: {hard_ensemble_metrics['macro_f1']:.4f}")
    print(f"  Soft Ensemble Accuracy: {soft_ensemble_metrics['accuracy']:.4f} | Macro F1: {soft_ensemble_metrics['macro_f1']:.4f}")
    print(f"  Voting Discordance Rate: {discordance_stats['discordance_rate']:.4%}")
    print(f"  Reports saved to {reports_dir}")

    return {
        "models": model_bundle,
        "test_df": test_df,
        "metrics": summary_metrics,
        "votes": all_votes,
        "soft_preds": soft_preds,
        "soft_confs": soft_confs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train traditional 3-model ensemble.")
    parser.add_argument("--config", default="config/reference.json", help="Path to config file")
    args = parser.parse_args()

    cfg = WorkstreamConfig.load(args.config)
    df, _ = load_dataset(cfg)
    splits = prepare_data_and_splits(df, cfg)
    train_traditional_models(splits, cfg)


if __name__ == "__main__":
    main()
