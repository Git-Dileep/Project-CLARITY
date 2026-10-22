# Detection Workstream — CLARITY Intrusion Detection Pipeline

This directory contains the production-grade, modular network intrusion detection pipeline for **Project CLARITY (Detection Workstream)**.

The pipeline supports both **Traditional Machine Learning (Random Forest, XGBoost, MLP)** and **Deep Learning (1D-CNN, LSTM, Transformer Encoder)** ensembles with majority-voting consensus, local SHAP explainability, and strict contract-compliant pilot alert export validated against `contracts/alert_schema_v1.json`.

---

## Quick Status Summary

> [!IMPORTANT]
> **Dataset Boundary & Experimental Status:**
> - **Codebase Implementation:** 100% complete and fully verified with 32 automated unit/integration tests and an end-to-end smoke run.
> - **Synthetic Reference Mode:** Verified working out-of-the-box (`data/reference_synthetic.csv` with 6 attack/traffic classes and 17 flow features).
> - **ACI IoT Network Dataset 2023 Real Experiment:** **BLOCKED — DATASET NOT AVAILABLE**. The real ACI dataset has not yet been downloaded to this machine. All code is dataset-independent and ready to execute on ACI data immediately upon placing CSV files and configuring `config/dataset_template.json`.

---

## Architecture & Directory Layout

```text
detection/workstream/
├── README.md                          # Comprehensive workstream guide & instructions
├── WORKSTREAM_STATUS.md               # Evidence gates and completion tracking
├── TASK_STATUS_AND_NEXT_STEPS.md      # Detailed task breakdown (DET-01 to DET-06)
├── requirements.txt                   # Tested dependency specification
├── .gitignore                         # Excludes raw data, virtual environments, binaries
├── run_all.bat                        # One-click master workflow runner
├── run_audit.bat                      # Standalone dataset audit runner
├── run_traditional.bat                # Standalone traditional ensemble trainer
├── run_deep_learning.bat              # Standalone deep learning ensemble trainer
├── run_pilot.bat                      # Standalone 20-alert pilot export & schema validator
├── config/
│   ├── reference.json                 # Default synthetic reference configuration
│   └── dataset_template.json          # Ready-to-use template for ACI IoT 2023 dataset
├── src/
│   ├── __init__.py
│   ├── config.py                      # Strongly-typed configuration schema
│   ├── synthetic_data.py              # Realistic 6-class, 17-feature traffic generator
│   ├── data_loader.py                 # Multi-CSV loader, schema checker, deduplication
│   ├── audit_datasets.py              # Dataset audit generator (JSON, CSV, MD)
│   ├── preprocessing.py               # Leakage-free split & reusable sklearn pipeline
│   ├── ensemble.py                    # Majority-voting consensus & multi-class metrics
│   ├── train_traditional.py           # RF + XGBoost + MLP 3-model ensemble trainer
│   ├── train_dl.py                    # CNN1D + LSTM + Transformer 3-model ensemble trainer
│   ├── explainability.py              # TreeExplainer & GradientExplainer top-10 attributions
│   ├── export_pilot.py                # Stratified 20-alert pilot exporter
│   ├── validate_alerts.py             # Draft 2020-12 schema validator
│   └── run_all.py                     # Master execution coordinator
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest test fixtures
│   ├── test_data_loader.py            # Data loading & schema compatibility tests
│   ├── test_audit.py                  # Audit reporting & error handling tests
│   ├── test_preprocessing.py          # Leakage-free preprocessing tests
│   ├── test_models.py                 # Ensemble voting & model training tests
│   ├── test_explainability.py         # Top-10 SHAP ranking tests
│   ├── test_contract.py               # Alert schema validation tests
│   └── test_pipeline.py               # End-to-end integration & reproducibility tests
├── contracts/
│   └── README.md                      # Reference to authoritative shared alert schema
├── data/
│   └── .gitkeep                       # Directory for local CSV data files
├── reports/
│   └── .gitkeep                       # Generated metrics, reports, and alert outputs
└── models/
    └── .gitkeep                       # Locally stored model weights/checkpoints
```

---

## Python Environment & Installation

### Requirements
- Python 3.10, 3.11, or 3.14 (64-bit Windows)
- Required core packages: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `jsonschema`, `pytest`
- Detector & explainability packages: `xgboost`, `torch`, `shap`

### Windows Setup Instructions (PowerShell)

```powershell
# 1. From repository root or workstream folder
cd detection/workstream

# 2. Create virtual environment
py -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Start (Synthetic Reference Pipeline)

Run the entire pipeline on high-fidelity synthetic reference data in one command:

```cmd
run_all.bat
```

Or via Python:
```powershell
python -m src.run_all --config config/reference.json
```

This single command will:
1. Auto-generate the synthetic reference traffic dataset (`data/reference_synthetic.csv`) with 6 attack/traffic classes and 17 network flow features.
2. Execute the dataset audit, generating `reports/data_audit.json`, `reports/data_audit.csv`, `reports/label_counts.csv`, and `reports/audit_summary.md`.
3. Preprocess features strictly on the training partition and record `reports/split_manifest.json`.
4. Train the 3-model Traditional Ensemble (Random Forest, XGBoost, MLP) and save `reports/traditional_metrics.json` and `reports/traditional_test_predictions.csv`.
5. Train the 3-model Deep Learning Ensemble (1D-CNN, LSTM, Transformer Encoder) and save `reports/dl/dl_metrics.json` and `reports/dl/dl_test_predictions.csv`.
6. Compute top-10 SHAP feature attributions for both detector families.
7. Export 20 stratified pilot alerts (`reports/pilot_traditional_alerts.json`, `reports/pilot_dl_alerts.json`, `reports/pilot_20_alerts.csv`).
8. Strictly validate every alert payload against `contracts/alert_schema_v1.json`, outputting `reports/alert_validation_report.json`.

---

## Running Individual Workflow Steps

### 1. Dataset Audit
```cmd
run_audit.bat config\reference.json
```
or
```powershell
python -m src.audit_datasets --config config/reference.json
```

### 2. Traditional Ensemble Training (RF + XGBoost + MLP)
```cmd
run_traditional.bat config\reference.json
```
or
```powershell
python -m src.train_traditional --config config/reference.json
```

### 3. Deep Learning Ensemble Training (1D-CNN + LSTM + Transformer)
```cmd
run_deep_learning.bat config\reference.json
```
or
```powershell
python -m src.train_dl --config config/reference.json
```

### 4. 20-Alert Pilot Export & Schema Validation
```cmd
run_pilot.bat config\reference.json
```
or
```powershell
python -m src.export_pilot --config config/reference.json
```

### 5. Standalone Alert Validation
```powershell
python -m src.validate_alerts --file reports/pilot_traditional_alerts.json
python -m src.validate_alerts --file reports/pilot_dl_alerts.json
```

---

## Running Tests

Run syntax validation and automated pytest suite:

```powershell
# 1. Compileall check
python -m compileall -q -x "\.venv" detection/workstream

# 2. Pytest test suite (32 tests)
python -m pytest detection/workstream/tests -v
```

---

## Attaching the Real ACI IoT Network Dataset 2023

When the official ACI dataset is acquired:

1. Download or place the ACI CSV files into a folder (e.g., `input_data/aci_2023/`).
2. Copy `config/dataset_template.json` to a local file, e.g., `config/local_aci_2023.json` (git-ignored):
   ```json
   {
     "dataset_name": "ACI-IoT-Network-2023",
     "is_synthetic": false,
     "input_path": "input_data/aci_2023/*.csv",
     "target_column": "Label",
     "flow_id_column": "Flow ID",
     "timestamp_column": "Timestamp",
     "metadata_columns": ["Source IP", "Destination IP", "Flow ID", "Timestamp"],
     "drop_columns": ["Source IP", "Destination IP", "Flow ID", "Timestamp"]
   }
   ```
3. Run the audit:
   ```cmd
   run_audit.bat config\local_aci_2023.json
   ```
4. Run the full workflow:
   ```cmd
   run_all.bat config\local_aci_2023.json
   ```
5. Review the generated reports in `reports/` and `reports/dl/`.

---

## Contract Compliance Highlights

The alert exporter guarantees strict adherence to `contracts/alert_schema_v1.json`:
- **Alert ID:** Guaranteed matching pattern `^ALT-[0-9]{8}-[0-9]{4}$`.
- **Learner Votes:** Exactly 3 base learner votes per alert (`learner_name`, `voted_label`, `learner_confidence`).
  - Traditional: `RandomForest`, `XGBoost`, `MLP`.
  - Deep Learning: `CNN1D`, `LSTM`, `Transformer`.
- **SHAP Attribution:** Exactly 10 features ranked 1 through 10 by descending absolute SHAP magnitude. Valid explainer enums: `TreeExplainer` (Traditional) and `GradientExplainer` (DL).
- **Leakage Prevention:** Ground truth label and correctness flags are isolated inside `evaluation_metadata`, ensuring LLM generation prompts never receive ground truth labels.
- **Automated Validation:** Every export step invokes `Draft202012Validator`. Any schema violation immediately halts execution with a non-zero exit code.

---

## Limitations

1. **Synthetic Data Disclaimer:** Synthetic reference data is provided strictly for verification, testing, and continuous integration. Synthetic metrics must not be cited as empirical performance on real IoT networks.
2. **Real ACI Dataset Pending:** Full experimental metrics on ACI IoT 2023 remain blocked until the dataset files are downloaded.
3. **Local Binaries Excluded:** In accordance with repository rules, large model checkpoint binaries (`models/*.joblib`, `models/*.pt`) and raw dataset CSVs are excluded from Git commits via `.gitignore`.
