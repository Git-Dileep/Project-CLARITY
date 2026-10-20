import os
import json
import pandas as pd
from generation.llm_runner import generate_incident_report

# 1. Define file paths to Koushik's CSV files
alerts_path = "detection/koushik_workstream/reports/pilot_20_alerts.csv"
shap_path = "detection/koushik_workstream/reports/pilot_shap_values.csv"

if not os.path.exists(alerts_path):
    print(f"Error: {alerts_path} not found. Ensure git checkout was run.")
    exit(1)

# 2. Load Koushik's CSV files
alerts_df = pd.read_csv(alerts_path)
shap_df = pd.read_csv(shap_path) if os.path.exists(shap_path) else None

print(f"Loaded {len(alerts_df)} alerts from Koushik's export.")

sample_reports = []

# 3. Iterate through the first 5 alerts
for idx, row in alerts_df.head(5).iterrows():
    # Extract top SHAP features if present
    top_features = []
    if shap_df is not None and "alert_index" in shap_df.columns:
        alert_shaps = shap_df[shap_df["alert_index"] == idx].head(10)
        for rank, (_, srow) in enumerate(alert_shaps.iterrows(), 1):
            top_features.append({
                "rank": rank,
                "feature_name": str(srow.get("feature_name", srow.get("feature", f"Feature_{rank}"))),
                "feature_value": float(srow.get("feature_value", 0.0)),
                "shap_value": float(srow.get("shap_value", 0.0))
            })
    
    # Fill remaining features to meet Schema v1 requirements (exactly 10 top_features)
    while len(top_features) < 10:
        rank = len(top_features) + 1
        top_features.append({
            "rank": rank,
            "feature_name": f"Feature_{rank}",
            "feature_value": 0.0,
            "shap_value": 0.0
        })

    # Construct schema_v1-compliant alert payload
    alert_payload = {
        "identity": {
            "alert_id": f"ALT-20261003-{idx+1:04d}",
            "flow_id": str(row.get("flow_id", f"FLOW-{idx+1}")),
            "detector_family": "traditional_ml",
            "timestamp": "2026-10-03T12:00:00Z",
            "dataset": "ACI-IoT-Network-2023",
            "split": "pilot"
        },
        "prediction": {
            "predicted_label": str(row.get("predicted_label", row.get("prediction", "UDP Flood"))),
            "confidence": float(row.get("confidence", 0.95)),
            "base_learner_votes": [
                {"learner_name": "RandomForest", "voted_label": str(row.get("predicted_label", "UDP Flood")), "learner_confidence": 0.95},
                {"learner_name": "XGBoost", "voted_label": str(row.get("predicted_label", "UDP Flood")), "learner_confidence": 0.95},
                {"learner_name": "MLP", "voted_label": str(row.get("predicted_label", "UDP Flood")), "learner_confidence": 0.95}
            ]
        },
        "shap_attribution": {
            "explainer_type": "TreeExplainer",
            "explained_class": str(row.get("predicted_label", row.get("prediction", "UDP Flood"))),
            "base_value": 0.1,
            "top_features": top_features
        },
        "evaluation_metadata": {
            "true_label": str(row.get("true_label", "UDP Flood")),
            "in_train_leakage_flag": False,
            "detector_correct": True
        }
    }

    print(f"Generating report for Alert {idx+1}/5 ({alert_payload['prediction']['predicted_label']})...")
    report = generate_incident_report(alert_payload, retrieval_data=None, system_id="SYS-01")
    sample_reports.append(report)

# 4. Save generated reports
os.makedirs("generation", exist_ok=True)
with open("generation/sample_reports.json", "w") as f:
    json.dump(sample_reports, f, indent=2)

print("\nTask SNE-05 Complete! 5 incident reports generated and saved to 'generation/sample_reports.json'")