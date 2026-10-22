# Detection Workstream Handoff

- **Ownership:** Managed through issue assignments and pull request reviews.
- **Scope:** Detection workstream located strictly within `detection/workstream/`. Authoritative shared contracts (`contracts/alert_schema_v1.json`) are preserved untouched.
- **Completed Implementation:**
  - Configurable multi-CSV data loader, schema compatibility checker, and robust audit generator (`src/audit_datasets.py`).
  - Three-model Traditional Ensemble (Random Forest + XGBoost + MLP) with majority voting (`src/train_traditional.py`).
  - Three-model Deep Learning Ensemble (1D-CNN + LSTM + Transformer Encoder) with shared leakage-free split (`src/train_dl.py`).
  - SHAP explainability engine ranking exactly 10 features strictly by descending absolute attribution magnitude (`src/explainability.py`).
  - Stratified 20-alert pilot export for held-out test examples (`src/export_pilot.py`), validated 100% against `contracts/alert_schema_v1.json` using Draft 2020-12 validator (`src/validate_alerts.py`).
  - Reproducible Windows batch run scripts with error propagation (`run_all.bat`, `run_audit.bat`, `run_traditional.bat`, `run_deep_learning.bat`, `run_pilot.bat`).
  - Comprehensive automated test suite (32 passing unit & integration tests under `tests/`).
- **Remaining Experimental Gate:**
  - Execute the pipeline on the physical ACI IoT Network Dataset 2023 files once downloaded to produce the final experiment benchmark.
  - The pipeline is completely dataset-independent and can be pointed to ACI files via configuration without code changes.
