# KOU Detection Tasks — Status and Next Steps

## 1. Issue Status Matrix

| Task ID | Description | Status | Evidence & Acceptance Gates |
| :--- | :--- | :--- | :--- |
| **KOU-01** | Switch to ACI IoT Network Dataset 2023 and audit it | **PARTIAL / BLOCKED** | **Codebase:** COMPLETE. Chunked multi-CSV loader, schema compatibility verifier, JSON/CSV/MD audit generator implemented. Tested with 3 unit tests. **Real Data:** BLOCKED. Real ACI dataset not yet present on machine. |
| **KOU-02** | Train traditional ensemble on ACI | **PARTIAL / BLOCKED** | **Codebase:** COMPLETE. 3-model majority voting ensemble (Random Forest, XGBoost, MLP) implemented with leakage-free preprocessing pipeline. Verified on synthetic reference data (Ensemble Accuracy: 99.70%, Macro-F1: 99.57%). **Real Data:** BLOCKED until ACI CSVs supplied. |
| **KOU-03** | Complete deep-learning ensemble on ACI | **PARTIAL / BLOCKED** | **Codebase:** COMPLETE. 3 PyTorch models (1D-CNN, LSTM, Transformer Encoder) implemented with shared splits and labels, majority voting, loss logging, and early stopping. Verified on synthetic reference data (Ensemble Accuracy: 97.80%, Macro-F1: 96.88%). **Real Data:** BLOCKED until ACI CSVs supplied. |
| **KOU-04** | Generate SHAP explanations for both ensembles with 10 features | **PARTIAL / BLOCKED** | **Codebase:** COMPLETE. TreeExplainer (XGBoost) and GradientExplainer (Transformer) implemented. Exactly 10 features extracted and ranked 1..10 by descending absolute SHAP magnitude. Validated via unit tests and pilot runs. **Real Data:** BLOCKED until ACI CSVs supplied. |
| **KOU-05** | Export stratified 20-alert pilot for both detectors | **PARTIAL / BLOCKED** | **Codebase:** COMPLETE. Stratified selection from held-out test set spanning 6 classes. Validated 100% against `contracts/alert_schema_v1.json` (20 traditional alerts + 20 DL alerts pass Draft 2020-12 validator with 0 errors). **Real Data:** BLOCKED until ACI CSVs supplied. |
| **KOU-06** | Make `.bat` run scripts reproducible | **COMPLETE** | **Codebase & Runtime:** COMPLETE. `run_audit.bat`, `run_traditional.bat`, `run_deep_learning.bat`, `run_pilot.bat`, and `run_all.bat` implemented with venv auto-detection, error propagation (`exit /b %errorlevel%`), and config parameter support. Verified in Windows PowerShell and cmd. |

---

## 2. Immediate Next Steps (When ACI Dataset is Available)

1. **Acquire & Place ACI Dataset:**
   - Download the official ACI IoT Network Dataset 2023 CSV files.
   - Place them in `detection/koushik_workstream/input_data/aci_2023/`.
2. **Configure Dataset Path:**
   - Copy `config/dataset_template.json` to `config/local_aci_2023.json` (this file is git-ignored).
   - Verify column names in the ACI CSV files match `config/local_aci_2023.json` (e.g. target column `"Label"`).
3. **Execute Audit:**
   - Run `run_audit.bat config\local_aci_2023.json`.
   - Inspect `reports/data_audit.json` and `reports/audit_summary.md` to verify column counts, row counts, missing values, and class distributions.
4. **Train & Evaluate Ensembles:**
   - Run `run_traditional.bat config\local_aci_2023.json` to train the RF + XGBoost + MLP ensemble.
   - Run `run_deep_learning.bat config\local_aci_2023.json` to train the CNN + LSTM + Transformer ensemble.
   - Check per-class F1 scores and confusion matrices in `reports/` and `reports/dl/`.
5. **Export & Validate Pilot Alerts:**
   - Run `run_pilot.bat config\local_aci_2023.json`.
   - Verify `reports/alert_validation_report.json` passes with 0 errors.
   - Share `reports/pilot_traditional_alerts.json` and `reports/pilot_dl_alerts.json` with retrieval and generation teammates.
6. **Update Workstream Documentation:**
   - Switch KOU-01 through KOU-05 status from **PARTIAL / BLOCKED** to **COMPLETE** with real-data evidence.
