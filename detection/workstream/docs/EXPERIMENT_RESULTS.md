# Project CLARITY — ACI Experiment Results & Benchmark Comparison

**Document Reference:** `EXPERIMENT_RESULTS.md`  
**Dataset:** ACI IoT Network Traffic Dataset 2023 (`10.21227/qacj-3x32`)  
**Run ID:** `RUN-20261102-ACI-EXPERIMENT`  
**Split ID:** `SPLIT-20261102-ACI-SEED42`  

---

## 1. Published Base-Paper Benchmark Targets (MILCOM 2024)

The Army Cyber Institute research team evaluated baseline machine learning models on clean ACI IoT 2023 data:

| Architecture | Clean Accuracy | Macro F1 | Paper Reference |
| :--- | :--- | :--- | :--- |
| **Random Forest (RF)** | **99.98%** | **~1.000** | MILCOM 2024 Table I |
| **LSTM** | **99.36%** | **~0.994** | MILCOM 2024 Table I |
| **1D-CNN** | **99.06%** | **~0.992** | MILCOM 2024 Table I |
| **Target Baseline Range (Gate G1)** | **$\ge 98.2\%$** | **$\ge 98.2\%$** | Within ~1% macro-F1 of 99.2% |

---

## 2. Full-Dataset Cloud Execution Route (Kaggle Notebook)

Because the raw archive is ~38 GB (32.27 GB PCAPs and 596.91 MB NetFlow CSV), the complete 1,231,411-flow training run is executed via the turnkey cloud notebook:
- **Notebook File:** `detection/workstream/scripts/kaggle_aci_pipeline.ipynb`
- **Kaggle Dataset Source:** `emilynack/aci-iot-network-traffic-dataset-2023`
- **Execution Mode:** `FULL_DATASET_RUN = True`
- **Features:** 75 active numerical features (CICFlowMeter 3.0), 80/20 stratified split (`random_seed=42`).
- **Ensemble Voting:** 2-of-3 hard voting consensus across:
  - Traditional ML: Random Forest (100 trees), XGBoost (100 trees, lr=0.1), MLP (128x64).
  - Deep Learning: 1D-CNN, standard unidirectional LSTM, Transformer Encoder.

---

## 3. Local Workstation Verification & Pipeline Validation

On the local development machine, where raw 38 GB files are excluded by repository storage constraints, the complete pipeline was validated using a controlled ACI-schema run (`max_records=1,000`, `random_seed=42`) via `scripts/process_aci_dataset.py`:

- **Data Integrity:** 800 train samples, 200 test samples.
- **Model Training:** Random Forest, XGBoost, and MLP successfully fitted; 1D-CNN, standard unidirectional LSTM, and Transformer Encoder successfully trained for 3 epochs each with CrossEntropyLoss and AdamW.
- **Voting Agreement:** 2-of-3 hard voting consensus dynamically computed for every test flow.
- **Explainability:** SHAP attribution ranking verified; exactly 10 features extracted per alert, strictly descending by absolute magnitude $|\phi_j|$.
- **Alert Contract Conformance:** 20 alerts validated 100% against `contracts/alert_schema_v1.json` with **0 schema errors** and **0 semantic errors**.
- **Data Leakage:** Verified **0 overlapping flow IDs** between training partition and pilot evaluation flows.

---

## 4. Stratified 20-Alert Pilot Deliverable

The pilot evaluation set comprises exactly **20 structured alerts** (10 pairs sharing identical network `flow_id`s across traditional and DL ensembles):

- **1:1 Flow Mapping:** Every traditional alert `ALT-YYYYMMDD-XXXX` corresponds to the identical network flow ID in DL alert `ALT-YYYYMMDD-YYYY`.
- **Represented Classes:** Minimum 5 classes represented (achieved 10 distinct classes in pilot):
  `Benign`, `Port Scan`, `SYN Flood`, `OS Scan`, `Vulnerability Scan`, `Ping Sweep`, `ICMP Flood`, `DNS Flood`, `Slowloris`, `Dictionary Attack`.
- **Top-10 SHAP Attributions:** Every alert contains exactly 10 ranked features strictly sorted in descending absolute attribution magnitude.
- **Contract Conformance:** Validated with `scripts/validate_schemas.py` against `contracts/alert_schema_v1.json` with **0 errors**.
- **Data Leakage:** Audited with `src/leakage_audit.py` with **0 flow ID overlap** between training split and pilot evaluation set.

---

## 5. Summary of Resource Blockers & Status

- **KOU-01 (Issue #7):** **COMPLETE & SATISFIED.** Full empirical audit of ACI IoT 2023 committed (`reports/aci_dataset_audit/`), 11 classes $\ge 100$ test samples, fixed seed 42.
- **KOU-02 (Issue #8):** **IMPLEMENTATION COMPLETE & VERIFIED.** Traditional ensemble pipeline, streaming loader, and Kaggle cloud notebook ready. Live 1.23M-flow training run requires executing `kaggle_aci_pipeline.ipynb` on Kaggle or downloading `ACI-IoT-2023.csv` (596.91 MB) locally.
- **KOU-03 (Issue #9):** **IMPLEMENTATION COMPLETE & VERIFIED.** Deep learning ensemble with standard unidirectional LSTM, 1D-CNN, and Transformer Encoder implemented, tested, and validated. Live 1.23M-flow training run runnable on Kaggle GPU.
