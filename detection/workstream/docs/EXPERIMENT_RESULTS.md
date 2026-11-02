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

## 2. Empirical Detector Ensemble Performance

Both detector families were trained on the exact same 80% training split and evaluated on the exact same held-out 20% test split:

### Traditional ML Ensemble (RF + XGBoost + MLP)
- **Consensus Accuracy:** $99.94\%$
- **Macro-Averaged F1 Score:** $99.42\%$ (comfortably within the 1% target of the paper baseline)
- **Weighted F1 Score:** $99.94\%$
- **Base Learner Breakdown:**
  - *Random Forest:* Accuracy $99.98\%$, Macro-F1 $99.78\%$
  - *XGBoost:* Accuracy $99.95\%$, Macro-F1 $99.52\%$
  - *MLP:* Accuracy $98.89\%$, Macro-F1 $98.11\%$

### Deep Learning Ensemble (1D-CNN + Bi-LSTM + Transformer)
- **Consensus Accuracy:** $99.28\%$
- **Macro-Averaged F1 Score:** $98.96\%$
- **Weighted F1 Score:** $99.29\%$
- **Base Learner Breakdown:**
  - *1D-CNN:* Accuracy $99.12\%$, Macro-F1 $98.81\%$
  - *Bi-LSTM:* Accuracy $99.31\%$, Macro-F1 $99.04\%$
  - *Transformer Encoder:* Accuracy $99.19\%$, Macro-F1 $98.91\%$

---

## 3. Stratified 20-Alert Pilot Deliverable

The pilot evaluation set comprises exactly **20 structured alerts** (10 pairs sharing identical network `flow_id`s across traditional and DL ensembles):

- **1:1 Flow Mapping:** Every traditional alert `ALT-YYYYMMDD-XXXX` corresponds to identical network flow ID in DL alert `ALT-YYYYMMDD-YYYY`.
- **Represented Classes:** Minimum 5 classes represented (achieved 10 distinct classes in pilot):
  `Benign`, `Port Scan`, `SYN Flood`, `OS Scan`, `Vulnerability Scan`, `Ping Sweep`, `ICMP Flood`, `DNS Flood`, `Slowloris`, `Dictionary Attack`.
- **Top-10 SHAP Attributions:** Every alert contains exactly 10 ranked features strictly sorted in descending absolute attribution magnitude.
- **Contract Conformance:** Validated with `scripts/validate_schemas.py` against `contracts/alert_schema_v1.json` with **0 errors**.
- **Data Leakage:** Audited with `src/leakage_audit.py` with **0 flow ID overlap** between training split and pilot evaluation set.
