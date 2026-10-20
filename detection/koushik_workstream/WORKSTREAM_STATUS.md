# Koushik Detection Workstream — Evidence & Completion Status

This document records the exact state of workstream issues, code implementation, test verification, and real-data experimental gates for the Detection Workstream (`detection/koushik_workstream`) on branch `Koushik`.

---

## 1. Status Overview

| Issue ID | Title | Codebase Status | Real ACI Experiment Status | Current Verification Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **KOU-01** | Switch to ACI IoT 2023 & Audit | **COMPLETE** | **BLOCKED** | Configurable loader, schema checker, chunked audit, reports generated (`data_audit.json`, `data_audit.csv`, `label_counts.csv`, `audit_summary.md`), 3 audit unit tests passing. Real ACI dataset not yet downloaded. |
| **KOU-02** | Train Traditional Ensemble on ACI | **COMPLETE** | **BLOCKED** | 3-model majority voting ensemble (Random Forest + XGBoost + MLP) implemented, leakage-safe preprocessing, comprehensive metrics computed (`traditional_metrics.json`, `traditional_metrics.csv`, `traditional_test_predictions.csv`). Verified on synthetic reference data (Accuracy 99.7%, Macro-F1 99.6%). |
| **KOU-03** | Complete Deep-Learning Ensemble on ACI | **COMPLETE** | **BLOCKED** | PyTorch 1D-CNN, LSTM, and Transformer Encoder implemented, identical splits and label encoding as traditional ensemble, majority voting consensus, loss tracking & early stopping (`dl_metrics.json`, `dl_test_predictions.csv`). Verified on synthetic reference data (Accuracy 97.8%, Macro-F1 96.9%). |
| **KOU-04** | Generate SHAP Explanations (10 Features) | **COMPLETE** | **BLOCKED** | Local explainers implemented: `TreeExplainer` for XGBoost and `GradientExplainer` for Transformer. Ranks exactly 10 features strictly descending by absolute SHAP magnitude. Contract format verified via unit tests and pilot runs. |
| **KOU-05** | Export Stratified 20-Alert Pilot | **COMPLETE** | **BLOCKED** | Deterministic stratified pilot export implemented for held-out test split (covering at least 5 classes). Generates `pilot_traditional_alerts.json`, `pilot_dl_alerts.json`, `pilot_20_alerts.csv`. Validated 100% against `contracts/alert_schema_v1.json` (`alert_validation_report.json`). |
| **KOU-06** | Make `.bat` Run Scripts Reproducible | **COMPLETE** | **COMPLETE** | All Windows batch scripts implemented with error propagation, environment auto-detection, and argument forwarding: `run_all.bat`, `run_audit.bat`, `run_traditional.bat`, `run_deep_learning.bat`, `run_pilot.bat`. Verified in Windows environment. |

---

## 2. Evidence Gates & Deliverables

### A. Preprocessing & Data Integrity
- **Fitted Strictly on Train:** `src/preprocessing.py` fits scalers, imputers, and categorical one-hot encoders solely on training instances.
- **Split Manifest:** `reports/split_manifest.json` documents exact train/test sample IDs, class distributions, and feature indices.
- **Leakage Prevention:** Raw identifiers (`flow_id`, `timestamp`, `device_id`, IP addresses) and target-derived columns are excluded from model features.

### B. Traditional Ensemble (KOU-02)
- **Base Learners:**
  1. Random Forest (`RandomForestClassifier`)
  2. XGBoost (`XGBClassifier`)
  3. Multi-layer Perceptron (`MLPClassifier`)
- **Voting Mechanism:** Hard voting consensus across 3 learners with highest-confidence tie-breaking.
- **Outputs Generated:**
  - `reports/traditional_metrics.json`
  - `reports/traditional_metrics.csv`
  - `reports/traditional_classification_report.csv`
  - `reports/traditional_confusion_matrix.csv`
  - `reports/traditional_test_predictions.csv`

### C. Deep Learning Ensemble (KOU-03)
- **Base Learners:**
  1. 1D Convolutional Neural Network (`CNN1DModel`)
  2. Bidirectional LSTM (`LSTMModel`)
  3. Transformer Encoder (`TransformerModel`)
- **Shared Split:** Evaluated on the exact same test instances with identical target encodings as the traditional detector.
- **Outputs Generated:**
  - `reports/dl/dl_metrics.json`
  - `reports/dl/dl_metrics.csv`
  - `reports/dl/dl_classification_report.csv`
  - `reports/dl/dl_confusion_matrix.csv`
  - `reports/dl/dl_test_predictions.csv`

### D. Explainability & Alert Contract Validation (KOU-04 & KOU-05)
- **Authoritative Contract:** `contracts/alert_schema_v1.json` unmodified.
- **Explainers:** Genuine SHAP `TreeExplainer` for tree models; genuine SHAP `GradientExplainer` for PyTorch models.
- **Top-10 Ranking:** Exactly 10 feature entries ranked 1 through 10 in descending order of absolute attribution magnitude.
- **Validation Report:** `reports/alert_validation_report.json` records 20/20 valid traditional alerts and 20/20 valid DL alerts with 0 errors.

---

## 3. Important Boundaries & Remaining Blocked Tasks

1. **ACI IoT Network Dataset 2023 Acquisition:**
   - **Status:** **BLOCKED**
   - **Reason:** Real ACI dataset files (~1.17M flows, 71 features, 11 classes) are not present in the local workspace.
   - **Unblocking Action:** When CSV files are downloaded, place them in `input_data/aci_2023/`, copy `config/dataset_template.json` to `config/local_aci_2023.json`, and run `run_all.bat config\local_aci_2023.json`.
2. **Synthetic Data Separation:**
   - All synthetic reference outputs are clearly tagged with `"is_synthetic": true` and dataset name `"Reference-Synthetic-Network"`.
   - Synthetic metrics demonstrate code and contract validity; they do not represent final empirical detection metrics on real IoT traffic.
3. **Data & Artifact Hygiene:**
   - Model weights (`models/*.joblib`, `models/*.pt`) and raw dataset CSVs are excluded from Git commits via `.gitignore`.
