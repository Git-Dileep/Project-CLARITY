# Project CLARITY — ACI IoT 2023 Detection Experiment Runbook

**Runbook Reference:** `RUNBOOK_ACI_EXPERIMENT.md`  
**Governing Issue:** [KOU-06 / #6](https://github.com/Git-Dileep/Project-CLARITY/issues/6)  
**Target Environment:** Local Windows / Linux Workstation or Free Cloud Kaggle Notebook  

---

## 1. Quick Start: Kaggle Cloud Execution (Zero Download Route)

The fastest and most reliable route to run the pipeline without downloading the 38 GB archive to your local workstation:

1. Open or upload `detection/workstream/scripts/kaggle_aci_pipeline.ipynb` to Kaggle:
   - URL: [https://www.kaggle.com/code](https://www.kaggle.com/code)
2. Add the dataset: `emilynack/aci-iot-network-traffic-dataset-2023`
3. Select Accelerator: GPU (T4 or P100) or CPU
4. Click **Run All**.
5. Download generated report artifacts from `/kaggle/working/`.

---

## 2. Local Workstation Execution

### Prerequisites & Python Environment
Ensure virtual environment has required dependencies (`torch`, `xgboost`, `scikit-learn`, `shap`, `jsonschema`, `pytest`):

```powershell
# From Project-CLARITY repository root
cd C:\Users\koush\Downloads\Koushik_CLARITY_Workstream\Koushik_CLARITY_Workstream\Project-CLARITY
```

### Step 1: Run Dataset Audit (KOU-01)
To inspect dataset statistics, verify class distribution, and generate inventory reports:

```powershell
.\detection\workstream\.venv\Scripts\python.exe -m detection.workstream.src.audit_datasets --config detection/workstream/config/aci_iot_2023.json
```

### Step 2: Execute Complete ACI Detection Pipeline (KOU-02 & KOU-03)
Processes data, trains traditional and DL ensembles, evaluates hard voting, computes SHAP attributions, and exports alerts:

```powershell
.\detection\workstream\.venv\Scripts\python.exe detection/workstream/scripts/process_aci_dataset.py `
  --dataset-path detection/workstream/input_data/network/ACI-IoT-2023.csv `
  --output-dir detection/workstream/reports/aci_experiment `
  --models-dir detection/workstream/models/aci `
  --seed 42
```

### Step 3: Run Alert Schema & Semantic Validation
Validates all generated alerts strictly against `contracts/alert_schema_v1.json` (Draft 2020-12):

```powershell
.\detection\workstream\.venv\Scripts\python.exe scripts/validate_schemas.py `
  --reports-dir detection/workstream/reports/aci_experiment
```

### Step 4: Run Data Leakage Audit
Verifies 0 flow ID overlap between training and pilot evaluation partitions:

```powershell
.\detection\workstream\.venv\Scripts\python.exe -m detection.workstream.src.leakage_audit `
  --reports-dir detection/workstream/reports/aci_experiment
```

### Step 5: Execute Automated Test Suite
Runs all 44 unit and integration tests across the detection workstream:

```powershell
.\detection\workstream\.venv\Scripts\pytest detection/workstream/tests -v --ignore=detection/workstream/.venv --ignore=detection/koushik_workstream/.venv
```

---

## 3. Deliverables Inventory

Upon successful pipeline completion, the following deliverables are populated:

```
detection/workstream/
├── reports/
│   ├── aci_dataset_audit/
│   │   ├── dataset_inventory.csv
│   │   ├── feature_schema.json
│   │   ├── class_distribution.csv
│   │   ├── missing_value_report.csv
│   │   ├── duplicate_report.json
│   │   ├── dataset_manifest.json
│   │   └── dataset_audit_summary.md
│   └── aci_experiment/
│       ├── pilot_traditional_alerts_10.json
│       ├── pilot_dl_alerts_10.json
│       ├── pilot_manifest.json
│       ├── traditional_metrics.json
│       ├── dl_metrics.json
│       ├── schema_and_semantic_validation_report.json
│       ├── leakage_audit_report.json
│       └── leakage_audit_report.md
└── models/
    └── aci/
        ├── traditional_ensemble.joblib
        └── dl_ensemble.pt
```
