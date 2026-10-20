from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import OrdinalEncoder
from .common import find_csvs, unique_files_by_content, normalise_label

def sampled_network_table(files: list[Path], max_per_class: int, chunksize: int, seed: int) -> pd.DataFrame:
    """Sample each chunk per class, then cap each class globally to bound RAM."""
    rng = np.random.default_rng(seed)
    pieces = []
    # A bounded quota per chunk prevents intermediate memory from growing with every CSV chunk.
    # The final pass then caps each class at max_per_class.
    chunk_quota = min(max_per_class, max(100, max_per_class // 10))
    required = ["ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "type"]
    for path in files:
        print(f"[LOAD/SAMPLE] {path.name}")
        try:
            reader = pd.read_csv(path, usecols=lambda c: c in required, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                chunk["_source_file"] = path.name
                if "type" not in chunk.columns:
                    warnings.warn(f"Skipping {path}: no 'type' column")
                    break
                chunk["type"] = normalise_label(chunk["type"])
                chunk = chunk.dropna(subset=["type"])
                # Bound memory: each chunk contributes at most chunk_quota rows per label.
                sampled = []
                for _, group in chunk.groupby("type", sort=False):
                    n = min(len(group), chunk_quota)
                    if len(group) > n:
                        group = group.sample(n=n, random_state=int(rng.integers(0, 2**31-1)))
                    sampled.append(group)
                if sampled:
                    pieces.append(pd.concat(sampled, ignore_index=True))
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
    if not pieces:
        raise RuntimeError("No usable network rows found. Check --network-dir and CSV headers.")
    data = pd.concat(pieces, ignore_index=True)
    # Remove duplicate rows within the modelling sample; duplicate source files are removed earlier.
    data = data.drop_duplicates()
    # A final per-class cap prevents large chunks/files dominating the sample.
    final = []
    for label, group in data.groupby("type", sort=False):
        n = min(len(group), max_per_class)
        final.append(group.sample(n=n, random_state=seed) if len(group) > n else group)
    result = pd.concat(final, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return result

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Exclude raw IP addresses and timestamp in the first baseline to reduce identity/time leakage.
    out = pd.DataFrame(index=df.index)
    for col in ["src_port", "dst_port"]:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    out["proto"] = df["proto"].astype("string").fillna("unknown").str.strip().str.lower() if "proto" in df else "unknown"
    out["src_port"] = out["src_port"].fillna(-1).clip(-1, 65535)
    out["dst_port"] = out["dst_port"].fillna(-1).clip(-1, 65535)
    # Stable one-hot encoding allows unseen protocols at inference to be ignored.
    out = pd.get_dummies(out, columns=["proto"], prefix="proto", dtype=np.uint8)
    return out.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Train RF + XGBoost hard-voting network IDS baseline.")
    parser.add_argument("--network-dir", default="input_data/network")
    parser.add_argument("--max-per-class", type=int, default=20000)
    parser.add_argument("--chunksize", type=int, default=250000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_per_class < 2:
        raise ValueError("--max-per-class must be at least 2.")
    report_dir, model_dir = Path("reports"), Path("models")
    report_dir.mkdir(exist_ok=True); model_dir.mkdir(exist_ok=True)
    all_files = find_csvs(args.network_dir)
    if not all_files:
        raise FileNotFoundError(f"No CSVs found under {args.network_dir}. Extract the network dataset there.")
    files, duplicates = unique_files_by_content(all_files)
    print(f"Found {len(all_files)} CSVs; using {len(files)} unique files; skipped {len(duplicates)} exact duplicate files.")
    df = sampled_network_table(files, args.max_per_class, args.chunksize, args.seed)
    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    # Shared sample enables the traditional and DL experiments to use the same labelled rows.
    sample_columns = [c for c in ["sample_id", "_source_file", "ts", "src_port", "dst_port", "proto", "type"] if c in df.columns]
    df[sample_columns].rename(columns={"_source_file": "source_file"}).to_csv(report_dir / "modeling_sample.csv", index=False)
    if df["type"].nunique() < 2:
        raise RuntimeError("Need at least two classes for classification.")
    print("Sampled class counts:\n", df["type"].value_counts().to_string())

    X = make_features(df)
    y_text = normalise_label(df["type"]).astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    counts = pd.Series(y_text).value_counts()
    if counts.min() < 2:
        raise RuntimeError(f"At least one class has fewer than 2 rows after sampling: {counts[counts < 2].to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Models kept modest enough for a typical student laptop. Adjust n_estimators after first run.
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed
    )
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost is required. Run: pip install xgboost") from exc
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.9, objective="multi:softprob",
        num_class=len(encoder.classes_), eval_metric="mlogloss",
        tree_method="hist", n_jobs=-1, random_state=args.seed
    )
    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)
    print("Training XGBoost...")
    xgb.fit(X_train, y_train)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)], voting="hard", n_jobs=None
    )
    # Refit through VotingClassifier so the saved ensemble is self-contained.
    print("Fitting hard-voting ensemble...")
    ensemble.fit(X_train, y_train)
    pred = ensemble.predict(X_test)
    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)
    # Preserve sample IDs and labels for repeatable pilot construction.
    test_rows = df.loc[X_test.index, [c for c in ["sample_id", "_source_file", "ts", "src_port", "dst_port", "proto", "type"] if c in df.columns]].copy()
    test_rows = test_rows.rename(columns={"_source_file": "source_file", "type": "true_label"})
    test_rows["ensemble_prediction"] = encoder.inverse_transform(pred)
    test_rows["rf_prediction"] = encoder.inverse_transform(rf_pred)
    test_rows["xgb_prediction"] = encoder.inverse_transform(xgb_pred)
    test_rows["ensemble_correct"] = (test_rows["true_label"].astype(str) == test_rows["ensemble_prediction"].astype(str))
    test_rows.to_csv(report_dir / "traditional_test_predictions.csv", index=False)

    labels = list(range(len(encoder.classes_)))
    report = classification_report(y_test, pred, labels=labels, target_names=encoder.classes_,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "rows_sampled": int(len(df)),
        "features": list(X.columns),
        "classes": encoder.classes_.tolist(),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split": "stratified random baseline; not a temporal/generalisation guarantee",
        "shared_sample": "reports/modeling_sample.csv",
        "test_predictions": "reports/traditional_test_predictions.csv",
        "excluded_columns": ["ts", "src_ip", "dst_ip"],
        "note": "Results are empirical for this sampled data only; check time/source leakage before claiming real-world performance."
    }
    (report_dir / "traditional_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(report).T.to_csv(report_dir / "traditional_classification_report.csv", index=True, index_label="class")
    pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(report_dir / "traditional_confusion_matrix.csv")
    importance = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": rf.feature_importances_,
        "xgb_importance": xgb.feature_importances_
    }).sort_values("xgb_importance", ascending=False)
    importance.to_csv(report_dir / "traditional_feature_importance.csv", index=False)
    joblib.dump({"model": ensemble, "label_encoder": encoder, "feature_columns": list(X.columns),
                 "excluded_columns": ["ts", "src_ip", "dst_ip"]}, model_dir / "rf_xgb_hard_voting.joblib")
    pd.DataFrame({"type": y_text}).value_counts().rename("count").reset_index().to_csv(
        report_dir / "sampled_class_counts.csv", index=False
    )

    print("\nMetrics:")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "rows_sampled", "train_rows", "test_rows"]}, indent=2))
    print("\nPer-class report saved to reports/traditional_classification_report.csv")
    try:
        import shap
        # SHAP can be slow; use a small test subset and XGBoost tree explainer.
        sample_n = min(500, len(X_test))
        sample = X_test.sample(n=sample_n, random_state=args.seed)
        explainer = shap.TreeExplainer(xgb)
        values = explainer.shap_values(sample)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if isinstance(values, list):
            shap.summary_plot(values[0], sample, show=False, max_display=12)
        elif getattr(values, "ndim", 0) == 3:
            shap.summary_plot(values[:, :, 0], sample, show=False, max_display=12)
        else:
            shap.summary_plot(values, sample, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(report_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("SHAP summary saved to reports/shap_summary.png")
    except Exception as exc:
        print(f"[WARN] SHAP summary skipped (training/evaluation succeeded): {exc}")
    print("Done. Check reports/ and models/.")

if __name__ == "__main__":
    main()
