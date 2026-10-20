# KOU Detection Tasks — Status and Next Steps

## 1. Issue Status Matrix

| Task ID | Description | Status | Evidence & Acceptance Gates |
| :--- | :--- | :--- | :--- |
| **KOU-01** | Switch to ACI IoT Network Dataset 2023 and audit it | **PARTIAL / BLOCKED** | **Codebase:** COMPLETE. Chunked multi-CSV loader, schema compatibility verifier, JSON/CSV/MD audit generator implemented. Tested with 3 unit tests. **Real Data:** BLOCKED. Real ACI dataset not yet present on machine. |
| **KOU-02** | Train traditional ensemble on ACI | **PARTIAL / BLOCKED** | **Codebase:** COMPLETE. 3-model majority voting ensemble (Random Forest, XGBoost, MLP) implemented with leakage-free preprocessing pipeline. Verified on synthetic reference data (Ensemble Accuracy: 99.70%, Macro-F1: 99.57%). **Real Data:** BLOCKED until ACI CSVs supplied. |
| **KOU-03** | Complete deep-learning ensemble on ACI | **BLOCKED (Real ACI Data)** | **Codebase:** COMPLETE. 3 PyTorch models (1D-CNN, LSTM, Transformer Encoder) implemented with shared splits and labels, majority voting, loss logging, and early stopping. Verified on synthetic reference data (Ensemble Accuracy: 97.80%, Macro-F1: 96.88%). **Real Data:** BLOCKED until ACI CSVs supplied. Do NOT claim real ACI validation. |
| **KOU-04** | Generate SHAP explanations for both ensembles with 10 features | **SAMPLE VERIFIED** | **Codebase & Sample Run:** COMPLETE. TreeExplainer (XGBoost) and GradientExplainer (Transformer) implemented with stable index tie-breaking. Exactly 10 canonical features extracted and ranked 1..10 strictly descending by absolute SHAP magnitude. Validated in `RUN-20261020-SAMPLE-PILOT10`. Mathematical protocol in `docs/SHAP_EXPLANATION_PROTOCOL.md`. |
| **KOU-05** | Export stratified pilot for both detectors | **SAMPLE VERIFIED** | **Codebase & Sample Run:** COMPLETE. Reproducible 10-flow pilot (`RUN-20261020-SAMPLE-PILOT10`) with identical flow IDs across Traditional ML and DL detectors spanning 6 classes. Validated against `contracts/alert_schema_v1.json` via `scripts/validate_schemas.py` (0 schema/semantic errors) and audited via `src/leakage_audit.py` (0 train/test overlap). |
| **KOU-06** | Make `.bat` run scripts reproducible | **COMPLETE** | **Codebase & Runtime:** COMPLETE. `run_audit.bat`, `run_traditional.bat`, `run_deep_learning.bat`, `run_pilot.bat`, `run_all.bat`, and `run_sample_pilot10.bat` implemented with venv auto-detection, error propagation (`exit /b %errorlevel%`), and config parameter support. 38/38 automated tests passing. |

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
