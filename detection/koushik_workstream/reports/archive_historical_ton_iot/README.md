# Historical TON_IoT Feasibility Archive

This directory contains preserved historical artifacts from the initial TON_IoT `GroundTruth_Network` feasibility run.

## Purpose & Provenance
These artifacts were generated during the preliminary exploratory phase on TON_IoT network CSVs:
- **Dataset:** TON_IoT `GroundTruth_Network` CSVs (18 chunk files).
- **Schema:** Only 7 fields (`ts, src_ip, src_port, dst_ip, dst_port, proto, type`).
- **Feature Limitation:** Excluded IPs and timestamps, leaving only 2 numeric ports (`src_port`, `dst_port`) and 3 protocol dummies (`proto_tcp`, `proto_udp`, `proto_icmp`). Total model features: **5 features**.
- **Schema Incompatibility:** `contracts/alert_schema_v1.json` strictly requires **exactly 10 ranked SHAP features** (`minItems: 10, maxItems: 10`) and valid alert IDs matching `^ALT-[0-9]{8}-[0-9]{4}$`. The historical pilot used non-standard IDs (e.g. `KOU-PILOT-001`) and could only provide 5 SHAP values.
- **Ensemble Limitation:** The historical baseline contained only 2 models (Random Forest + XGBoost) rather than the required 3 base learners.

## Archived Artifacts
- `traditional_classification_report.csv`: 2-model (RF + XGBoost) classification report (57.29% accuracy).
- `dl_metrics.json`: 3-epoch feasibility DL metrics.
- `pilot_20_alerts.csv`: 20-alert pilot with legacy `KOU-PILOT-001` IDs.
- `pilot_shap_values.csv`: 5-feature XGBoost SHAP attributions.
- `pilot_summary.json`: Summary metadata of the legacy pilot.
- `class_summary.csv`, `data_audit.csv`, `duplicate_files.csv`: Historical dataset audit artifacts.
- `modeling_sample.csv`, `traditional_confusion_matrix.csv`, `traditional_test_predictions.csv`: Auxiliary tables from historical runs.

## Current Active Run
The current reproducible, contract-compliant 10-flow pilot run is located in:
`reports/sample_pilot10/`
with configuration `config/run_sample_pilot10.json`.
Do not confuse these archived feasibility outputs with the active 10-flow pilot run.
