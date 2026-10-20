from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from .train_traditional import make_features


def main():
    p = argparse.ArgumentParser(description="Export a stratified 20-alert pilot with both detector predictions and explanations.")
    p.add_argument("--pilot-size", type=int, default=20)
    p.add_argument("--min-classes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    report = Path("reports"); dl_report = report / "dl"; model_dir = Path("models")
    needed = [report/"modeling_sample.csv", report/"traditional_test_predictions.csv",
              dl_report/"transformer_test_predictions.csv", dl_report/"transformer_test_windows.npz",
              model_dir/"rf_xgb_hard_voting.joblib", model_dir/"transformer_feasibility.pt"]
    missing = [str(x) for x in needed if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing required outputs. Run run_baseline.bat, then run_dl_feasibility.bat. Missing: " + ", ".join(missing))
    sample = pd.read_csv(report/"modeling_sample.csv", low_memory=False)
    trad = pd.read_csv(report/"traditional_test_predictions.csv")
    dl = pd.read_csv(dl_report/"transformer_test_predictions.csv")
    merged = trad.merge(dl[["sample_id", "transformer_prediction", "transformer_correct"]], on="sample_id", how="inner")
    merged["true_label"] = merged["true_label"].astype(str)
    merged["ensemble_prediction"] = merged["ensemble_prediction"].astype(str)
    merged["transformer_prediction"] = merged["transformer_prediction"].astype(str)
    classes = sorted(merged["true_label"].unique())
    if len(classes) < args.min_classes:
        raise RuntimeError(f"Only {len(classes)} classes intersect across both held-out outputs; need {args.min_classes}.")
    # Balanced, deterministic selection: take up to ceil(pilot_size/min_classes) per class, then fill if needed.
    rng = np.random.default_rng(args.seed)
    chosen = []
    per_class = max(1, int(np.ceil(args.pilot_size / len(classes))))
    for label in classes:
        g = merged[merged.true_label == label]
        if len(g):
            take = min(per_class, len(g))
            ids = rng.choice(g.index.to_numpy(), size=take, replace=False)
            chosen.extend(ids.tolist())
    if len(chosen) < args.pilot_size:
        remaining = merged.drop(index=chosen, errors="ignore")
        if len(remaining):
            extra = rng.choice(remaining.index.to_numpy(), size=min(args.pilot_size-len(chosen), len(remaining)), replace=False)
            chosen.extend(extra.tolist())
    pilot = merged.loc[chosen].drop_duplicates(subset=["sample_id"]).head(args.pilot_size).copy()
    if pilot["true_label"].nunique() < args.min_classes:
        raise RuntimeError(f"Could not create a {args.pilot_size}-row pilot covering {args.min_classes} classes; only {pilot['true_label'].nunique()} classes found.")
    pilot = pilot.merge(sample, on="sample_id", how="left", suffixes=("", "_sample"))
    pilot["traditional_correct"] = pilot["true_label"] == pilot["ensemble_prediction"]
    pilot["transformer_correct"] = pilot["true_label"] == pilot["transformer_prediction"]
    pilot["alert_id"] = [f"KOU-PILOT-{i+1:03d}" for i in range(len(pilot))]
    pilot_path = report / "pilot_20_alerts.csv"
    pilot.to_csv(pilot_path, index=False)

    # Traditional per-alert XGBoost SHAP values, explained for each sample's predicted XGBoost class.
    shap_out = report / "pilot_shap_values.csv"
    try:
        import shap
        bundle = joblib.load(model_dir/"rf_xgb_hard_voting.joblib")
        ensemble = bundle["model"]
        xgb = ensemble.named_estimators_["xgb"]
        feat_cols = bundle["feature_columns"]
        raw = pilot[[c for c in ["src_port", "dst_port", "proto", "ts", "type"] if c in pilot.columns]].copy()
        X = make_features(raw)
        X = X.reindex(columns=feat_cols, fill_value=0).astype(np.float32)
        pred_idx = xgb.predict(X).astype(int)
        explainer = shap.TreeExplainer(xgb)
        vals = explainer.shap_values(X)
        rows = []
        for i, (_, r) in enumerate(pilot.iterrows()):
            if isinstance(vals, list):
                class_vals = np.asarray(vals[pred_idx[i]])[i]
            else:
                arr = np.asarray(vals)
                class_vals = arr[i, :, pred_idx[i]] if arr.ndim == 3 else arr[i]
            for f, v in zip(X.columns, class_vals):
                rows.append({"alert_id": r["alert_id"], "sample_id": r["sample_id"], "detector": "XGBoost component of RF+XGBoost ensemble", "explained_class": r["xgb_prediction"], "feature": f, "shap_value": float(v)})

        # DL GradientExplainer values for the Transformer on the same held-out sample IDs.
        import torch
        from .train_dl import TransformerModel
        checkpoint = torch.load(model_dir/"transformer_feasibility.pt", map_location="cpu", weights_only=False)
        model = TransformerModel(checkpoint["n_features"], len(checkpoint["classes"]), checkpoint["window"])
        model.load_state_dict(checkpoint["state_dict"]); model.eval()
        z = np.load(dl_report/"transformer_test_windows.npz")
        id_to_i = {int(sid): i for i, sid in enumerate(z["sample_id"])}
        keep = [i for i, sid in enumerate(pilot["sample_id"].astype(int)) if sid in id_to_i]
        if keep:
            all_windows = z["X"]
            bg = torch.tensor(all_windows[:min(40, len(all_windows))], dtype=torch.float32)
            explain = torch.tensor(np.stack([all_windows[id_to_i[int(pilot.iloc[i]["sample_id"])]] for i in keep]), dtype=torch.float32)
            grad = shap.GradientExplainer(model, bg)
            sv = grad.shap_values(explain)
            if isinstance(sv, list):
                # list[class] -> samples x window x features
                for local_i, pilot_i in enumerate(keep):
                    pred_label = str(pilot.iloc[pilot_i]["transformer_prediction"])
                    class_idx = checkpoint["classes"].index(pred_label)
                    arr = np.asarray(sv[class_idx])[local_i]
                    mean_vals = arr.mean(axis=0)
                    feat_names = ["src_port", "dst_port"] + list(pd.get_dummies(sample["proto"].astype("string").fillna("unknown"), prefix="proto", dtype=np.uint8).columns)
                    for f, v in zip(feat_names, mean_vals):
                        rows.append({"alert_id": pilot.iloc[pilot_i]["alert_id"], "sample_id": int(pilot.iloc[pilot_i]["sample_id"]), "detector": "Transformer", "explained_class": pred_label, "feature": f, "shap_value": float(v)})
            else:
                arr = np.asarray(sv)
                for local_i, pilot_i in enumerate(keep):
                    pred_label = str(pilot.iloc[pilot_i]["transformer_prediction"])
                    class_idx = checkpoint["classes"].index(pred_label)
                    if arr.ndim == 4:
                        contribution = arr[local_i, :, :, class_idx]
                    else:
                        contribution = arr[local_i]
                    mean_vals = contribution.mean(axis=0)
                    feat_names = ["src_port", "dst_port"] + list(pd.get_dummies(sample["proto"].astype("string").fillna("unknown"), prefix="proto", dtype=np.uint8).columns)
                    for f, v in zip(feat_names, mean_vals):
                        rows.append({"alert_id": pilot.iloc[pilot_i]["alert_id"], "sample_id": int(pilot.iloc[pilot_i]["sample_id"]), "detector": "Transformer", "explained_class": pred_label, "feature": f, "shap_value": float(v)})
        else:
            print("[WARN] No pilot sample IDs matched the DL window output; DL SHAP rows not created.")
        pd.DataFrame(rows).to_csv(shap_out, index=False)
        if not any(r["detector"] == "Transformer" for r in rows):
            print("[WARN] Pilot predictions exported, but Transformer SHAP was not generated for matched IDs.")
    except Exception as exc:
        print(f"[WARN] SHAP export incomplete: {exc}")
        if not shap_out.exists():
            pd.DataFrame(columns=["alert_id", "sample_id", "detector", "explained_class", "feature", "shap_value"]).to_csv(shap_out, index=False)

    summary = {
        "pilot_rows": int(len(pilot)), "true_classes": sorted(pilot["true_label"].unique().tolist()),
        "traditional_accuracy_on_pilot": float(pilot["traditional_correct"].mean()),
        "transformer_accuracy_on_pilot": float(pilot["transformer_correct"].mean()),
        "predictions_file": str(pilot_path), "shap_file": str(shap_out),
        "warning": "Pilot examples are sampled from the shared stratified random holdout; they do not establish unseen-session generalisation. Inspect SHAP output and verify the actual feature columns before citing conclusions."
    }
    (report/"pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
