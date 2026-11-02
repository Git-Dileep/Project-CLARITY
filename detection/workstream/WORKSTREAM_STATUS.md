# Detection Workstream — Evidence & Completion Status

This document records the authoritative state of workstream issues, code implementation, test verification, dataset audit deliverables, and real-data experimental routes for the Detection Workstream (`detection/workstream`).

---

## 1. Status Overview

| Issue ID | GitHub Issue | Title | Codebase Status | Real ACI Experiment Status | Current Verification Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KOU-01** | **#7** | Switch to ACI IoT 2023 & Audit | **COMPLETE** | **AUDITED & COMMITTED** | Empirical audit complete (`reports/aci_dataset_audit/`). Verified 1,231,411 flows, 75 numerical features, 12 classes across 5 categories. 11 classes satisfy Gate G1 ($\ge 100$ test samples). Provenance and manifest documented. |
| **KOU-02** | **#8** | Train Traditional Ensemble on ACI | **COMPLETE** | **KAGGLE REPRODUCIBLE** | 3-model majority voting ensemble (Random Forest + XGBoost + MLP) implemented with 2-of-3 hard voting consensus. Verified via `tests/test_aci_pipeline.py`. Standalone cloud execution notebook `scripts/kaggle_aci_pipeline.ipynb` provided for Kaggle GPU/CPU execution against `ACI-IoT-2023.csv`. |
| **KOU-03** | **#9** | Complete Deep-Learning Ensemble on ACI | **COMPLETE** | **KAGGLE REPRODUCIBLE** | PyTorch 1D-CNN, Bi-LSTM, and Transformer Encoder implemented, 2-of-3 hard voting consensus, identical splits and label encoding as traditional detector. Standalone cloud execution notebook `scripts/kaggle_aci_pipeline.ipynb` provided for Kaggle GPU/CPU execution against `ACI-IoT-2023.csv`. |
| **KOU-04** | **#10** | Generate SHAP Explanations (10 Features) | **COMPLETE** | **VERIFIED** | SHAP `TreeExplainer` (XGBoost) and `GradientExplainer` (Transformer) implemented. Exactly 10 canonical features ranked 1..10 strictly descending by absolute SHAP magnitude with 0 gaps or duplicates. Protocol documented in `docs/SHAP_EXPLANATION_PROTOCOL.md`. |
| **KOU-05** | **#11** | Export Stratified Pilot for Both Detectors | **COMPLETE** | **VERIFIED** | 10 shared held-out flows evaluated across both Traditional ML and DL detectors spanning multiple classes. 100% compliant with `contracts/alert_schema_v1.json` via `scripts/validate_schemas.py` (0 errors) and 0 train/pilot flow leakage via `src/leakage_audit.py`. |
| **KOU-06** | **#12** | Make Run Scripts Reproducible | **COMPLETE** | **COMPLETE** | Automated reproducible scripts (`process_aci_dataset.py`, `kaggle_aci_pipeline.ipynb`, batch scripts). Full test suite passing 44/44 automated tests. |

---

## 2. Evidence Gates & Deliverables

### A. ACI Dataset Audit (KOU-01 / Issue #7)
- **Inventory & Manifest:** `reports/aci_dataset_audit/dataset_inventory.csv` and `dataset_manifest.json` record DOI `10.21227/qacj-3x32`, checksums, and storage breakdown (32.27 GB PCAP + 596.91 MB CSV).
- **Gate G1 Compliance:** `reports/aci_dataset_audit/class_distribution.csv` verifies that 11 out of 12 classes possess $\ge 100$ test samples in an 80/20 train/test split.
- **Missing Value Handling:** `reports/aci_dataset_audit/missing_value_report.csv` documents 3,848 missing/infinite cells in rate columns (`Flow Byts/s`, `Flow Pkts/s`) and their median imputation protocol.
- **Duplicate Analysis:** `reports/aci_dataset_audit/duplicate_report.json` documents 60,841 exact duplicate records and deduplication rules.
- **Audit Narrative:** `reports/aci_dataset_audit/dataset_audit_summary.md` and `docs/ACI_DATASET_AUDIT.md`.

### B. Traditional Ensemble (KOU-02 / Issue #8)
- **Learners:** Random Forest (`n_estimators=100`, `max_depth=12`), XGBoost (`n_estimators=100`, `learning_rate=0.1`), MLP (`hidden_layer_sizes=(128, 64)`).
- **Consensus:** 2-of-3 hard voting consensus.
- **Paper Reproduction Benchmark:** Target macro-F1 $\ge 98.2\%$ (within 1% of base paper's $99.2\%$).

### C. Deep Learning Ensemble (KOU-03 / Issue #9)
- **Learners:** 1D-CNN (`CNN1D`), Bidirectional LSTM (`BiLSTM`), Transformer Encoder (`TransformerEnc`).
- **Consensus:** 2-of-3 hard voting consensus over 75 active numerical features.
- **Shared Partition:** Strictly identical train/test splits and scalers as the traditional detector.

### D. Explainability & Alert Schema Validation (KOU-04 & KOU-05)
- **Authoritative Contract:** `contracts/alert_schema_v1.json` unmodified.
- **Explainers:** `TreeExplainer` on XGBoost and `GradientExplainer` on Transformer.
- **Validation:** 0 schema or semantic errors across all exported alerts.
- **Data Leakage:** 0 flow ID overlap between training partition and pilot evaluation flows.

---

## 3. Automated Testing Summary

- **Total Tests:** 44 tests (Unit, integration, schema validation, leakage audit).
- **Passed:** 44 (100%).
- **Execution Command:**
  ```powershell
  .\detection\workstream\.venv\Scripts\pytest detection/workstream/tests -v --ignore=detection/workstream/.venv --ignore=detection/koushik_workstream/.venv
  ```

---

## 4. Execution Routes & Dataset Hygiene

1. **Cloud Execution Route (Zero Local Download):**
   - Standalone Jupyter Notebook: `detection/workstream/scripts/kaggle_aci_pipeline.ipynb`
   - Mounts `emilynack/aci-iot-network-traffic-dataset-2023` in Kaggle's free GPU/CPU environment.
2. **Local Workstation Route:**
   - Place `ACI-IoT-2023.csv` into `detection/workstream/input_data/network/ACI-IoT-2023.csv`.
   - Run `detection/workstream/scripts/process_aci_dataset.py`.
3. **Hygiene Rules:**
   - Raw dataset files, `.pcap` dumps, and model binary weights are excluded via `.gitignore`.
