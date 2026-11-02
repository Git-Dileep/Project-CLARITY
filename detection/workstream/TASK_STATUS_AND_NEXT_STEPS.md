# Detection Tasks — Status and Next Steps

## 1. Issue Status Matrix

| Task ID | Description | Status | Evidence & Acceptance Gates |
| :--- | :--- | :--- | :--- |
| **KOU-01** (Issue #7) | Switch to ACI IoT Network Dataset 2023 and audit it | **COMPLETE** | **Audited & Committed:** Full empirical audit deliverables committed in `reports/aci_dataset_audit/` (inventory, schema, class distribution, missing value report, duplicate report, manifest, summary). Verified 1,231,411 flows, 75 numerical features, 12 classes. Gate G1 satisfied (11 classes with $\ge 100$ test samples). |
| **KOU-02** (Issue #8) | Train traditional ensemble on ACI | **READY / KAGGLE RUNNABLE** | **Codebase & Pipeline:** COMPLETE. 3-model majority voting ensemble (Random Forest, XGBoost, MLP) implemented with leakage-free preprocessing pipeline. Verified via unit and integration tests. Turnkey Kaggle cloud notebook `scripts/kaggle_aci_pipeline.ipynb` and `scripts/process_aci_dataset.py` provided for cloud execution against raw `ACI-IoT-2023.csv`. |
| **KOU-03** (Issue #9) | Complete deep-learning ensemble on ACI | **READY / KAGGLE RUNNABLE** | **Codebase & Pipeline:** COMPLETE. 3 PyTorch models (1D-CNN, Bi-LSTM, Transformer Encoder) implemented with shared splits and labels, 2-of-3 majority voting, loss logging, and checkpointing. Verified via unit and integration tests. Cloud execution notebook provided. |
| **KOU-04** (Issue #10) | Generate SHAP explanations for both ensembles with 10 features | **VERIFIED** | **Codebase & Runtime:** COMPLETE. TreeExplainer (XGBoost) and GradientExplainer (Transformer) implemented with stable index tie-breaking. Exactly 10 canonical features extracted and ranked 1..10 strictly descending by absolute SHAP magnitude. Mathematical protocol in `docs/SHAP_EXPLANATION_PROTOCOL.md`. |
| **KOU-05** (Issue #11) | Export stratified pilot for both detectors | **VERIFIED** | **Codebase & Runtime:** COMPLETE. Reproducible 10-flow pilot with identical flow IDs across Traditional ML and DL detectors spanning multiple classes. Validated against `contracts/alert_schema_v1.json` via `scripts/validate_schemas.py` (0 errors) and audited via `src/leakage_audit.py` (0 train/test overlap). |
| **KOU-06** (Issue #12) | Make run scripts reproducible | **COMPLETE** | **Codebase & Runtime:** COMPLETE. Reproducible execution via `process_aci_dataset.py`, `kaggle_aci_pipeline.ipynb`, and `.bat` run scripts with environment auto-detection and error propagation. 44/44 automated tests passing. |

---

## 2. Immediate Next Steps & Execution Routes

1. **Option 1: Execute Cloud Pipeline on Kaggle (Zero Local Download)**
   - Open `detection/workstream/scripts/kaggle_aci_pipeline.ipynb` in Kaggle.
   - Attach the public Kaggle dataset `emilynack/aci-iot-network-traffic-dataset-2023`.
   - Run the notebook to generate full training logs, confusion matrices, and model weights directly in Kaggle cloud storage.
2. **Option 2: Local Execution with Downloaded CSV**
   - Place `ACI-IoT-2023.csv` (596.91 MB) into `detection/workstream/input_data/network/ACI-IoT-2023.csv`.
   - Run `python detection/workstream/scripts/process_aci_dataset.py`.
   - Run `scripts/validate_schemas.py` and `src/leakage_audit.py`.
