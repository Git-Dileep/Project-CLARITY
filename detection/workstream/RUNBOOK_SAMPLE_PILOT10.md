# Project CLARITY — Reproducible 10-Flow Sample Pilot Runbook (DET-04 / DET-05)

**Run ID:** `RUN-20261020-SAMPLE-PILOT10`  
**Split ID:** `SPLIT-20261020-SEED42-V1`  
**Target Workstream:** `detection/workstream`
**Specification Reference:** `contracts/alert_schema_v1.json` (Draft 2020-12)  
**Execution Date:** October 2026  

---

> [!IMPORTANT]
> **SAMPLE-ONLY RUN DISCLAIMER:**  
> This run constitutes a **sample-only pilot** designed strictly to satisfy and verify the DET-04 (SHAP explainability) and DET-05 (alert export and validation) acceptance criteria. It uses the canonical reference synthetic dataset (`data/reference_synthetic.csv`) to validate end-to-end schemas, SHAP attributions, ensemble voter contracts, and zero data leakage.
> **This run does NOT claim validation on the real ACI IoT 2023 dataset.** DET-03 remains cleanly and honestly designated as **BLOCKED** pending acquisition and unblocking of the genuine ACI IoT dataset.

---

## 1. Reference Dataset & Environmental Invariants

To eliminate past ambiguity between historical 5-feature TON_IoT artifacts and the 24-feature contract, this run operates exclusively on the single agreed reference dataset:

- **Dataset File:** `data/reference_synthetic.csv`
- **Total Records:** 5,000 network flows
- **Feature Count:** 24 features (16 numeric, 2 categorical, 6 classes, 0 missing values)
- **Native Flow Identifier:** Column `flow_id` present on 100% of rows (`FLW-YYYYMMDD-XXXXXX`)
- **Dataset SHA-256 Checksum:** `c85c523d12610dc8554cdeee0eedc6a46b9e27a0ed74be12d0d362b8215ebceb`

---

## 2. Configuration & Execution Commands

The run is governed by `config/run_sample_pilot10.json`:
- `random_seed`: `42`
- `test_size`: `0.2` (4,000 train samples, 1,000 test samples)
- `pilot_size`: `10`
- `min_classes`: `5` (6 unique classes achieved)
- `reports_dir`: `reports/sample_pilot10`

### One-Click Reproduction (Windows Batch)
From `detection/workstream/`:
```bat
run_sample_pilot10.bat
```

### Manual CLI Execution
From `Project-CLARITY/` repository root:
```powershell
# 1. Run complete training, explanation, and pilot export pipeline
detection\workstream\.venv\Scripts\python.exe -m src.run_all --config config/run_sample_pilot10.json

# 2. Validate alert schema contracts (Draft 2020-12) and semantic invariants
detection\workstream\.venv\Scripts\python.exe scripts/validate_schemas.py --config detection/workstream/config/run_sample_pilot10.json

# 3. Perform train-vs-pilot data leakage audit
detection\workstream\.venv\Scripts\python.exe -m src.leakage_audit --config config/run_sample_pilot10.json

# 4. Run automated test suite
detection\workstream\.venv\Scripts\python.exe -m pytest detection/workstream/tests/test_pilot10_run.py -v
```

---

## 3. Train/Test Partition & Held-Out Pilot Selection

### Partition Manifest (`split_manifest.json`)
- **Train partition:** 4,000 flows (80%), recorded as explicit `train_flow_ids` and `train_indices`.
- **Test partition:** 1,000 flows (20%), recorded as explicit `test_flow_ids` and `test_indices`.
- **Determinism:** Governed by `SPLIT-20261020-SEED42-V1` with stratified class representation.

### Pilot Manifest (`pilot_manifest.json`)
- **Held-Out Isolation:** All 10 pilot flows are sampled exclusively from the 1,000 held-out test flows.
- **Shared Flows:** Both Traditional ML and Deep Learning ensembles evaluated the **exact same 10 flow IDs** in the same order.
- **Represented Classes (6):**
  1. `Benign` (3 flows)
  2. `BruteForce` (2 flows)
  3. `PortScan` (2 flows)
  4. `Infiltration` (1 flow)
  5. `DDoS_UDP` (1 flow)
  6. `Mirai_Botnet` (1 flow)

### Pilot Flow ID Inventory
1. `FLW-20261020-000657` (True Label: `Infiltration`)
2. `FLW-20261020-004125` (True Label: `BruteForce`)
3. `FLW-20261020-001603` (True Label: `PortScan`)
4. `FLW-20261020-001063` (True Label: `Benign`)
5. `FLW-20261020-000216` (True Label: `PortScan`)
6. `FLW-20261020-002464` (True Label: `DDoS_UDP`)
7. `FLW-20261020-004612` (True Label: `Benign`)
8. `FLW-20261020-003466` (True Label: `Mirai_Botnet`)
9. `FLW-20261020-001268` (True Label: `BruteForce`)
10. `FLW-20261020-000753` (True Label: `Benign`)

---

## 4. SHAP Attribution Protocol & Ensemble Rules

Attributions comply with `contracts/alert_schema_v1.json` and are documented mathematically in `docs/SHAP_EXPLANATION_PROTOCOL.md`:

### Traditional ML Ensemble (RF, XGBoost, MLP)
- **Consensus Rule:** Unweighted majority vote among the 3 base learners.
- **Attribution Engine:** `shap.TreeExplainer` applied to the gradient-boosted tree (XGBoost).
- **Target Class:** Explanations are computed specifically with respect to the ensemble consensus prediction $\hat{y}$.
- **Attribution Format:** Exactly 10 canonical features ranked 1..10 in descending order of $|\phi_j|$, with stable deterministic tie-breaking on feature index.

### Deep Learning Ensemble (CNN1D, LSTM, Transformer)
- **Consensus Rule:** Unweighted majority vote among the 3 neural architectures.
- **Attribution Engine:** `shap.GradientExplainer` applied to the Transformer architecture using a 100-sample representative background distribution from the training set.
- **Target Class:** Explanations computed with respect to consensus prediction $\hat{y}$.
- **Attribution Format:** Exactly 10 canonical features ranked 1..10 in descending order of $|\phi_j|$.

---

## 5. Verification Results

| Verification Dimension | Check / Invariant | Status | Result |
|---|---|---|---|
| **Schema Validation** | `Draft202012Validator` on all 20 alerts | **PASSED** | 0 schema violations |
| **Alert Identity Format** | Regex `^ALT-[0-9]{8}-[0-9]{4}$` | **PASSED** | 100% compliant |
| **Voter Card Invariant** | Exactly 3 base learners per alert | **PASSED** | Traditional: RF, XGB, MLP; DL: CNN, LSTM, Transformer |
| **Top Features Count** | Exactly 10 features per alert | **PASSED** | 10 features, ranks 1..10 |
| **SHAP Monotonicity** | $|\phi_1| \ge |\phi_2| \ge \dots \ge |\phi_{10}|$ | **PASSED** | Monotonic across all 20 alerts |
| **Identity / Target Leakage** | No ground-truth or ID features in top 10 | **PASSED** | 0 forbidden features |
| **Evaluation Leakage** | `in_train_leakage_flag == False` | **PASSED** | 100% held-out test flows |
| **Train/Pilot Overlap** | Train flows $\cap$ Pilot flows | **PASSED** | **0 flows overlap** |
| **Detector Consistency** | Traditional flows == DL flows | **PASSED** | Exact 1:1 match across all 10 flows |
| **Class Diversity** | Unique classes represented | **PASSED** | 6 classes ($\ge 5$ required) |
| **Automated Tests** | `pytest tests/test_pilot10_run.py` | **PASSED** | 6 passed in 0.12s (38/38 total passed) |

---

## 6. Directory Artifacts Reference

All generated outputs are located in `detection/workstream/reports/sample_pilot10/`:

```
reports/sample_pilot10/
├── alert_validation_report.json              # Fast internal validator output
├── audit_summary.md                          # DET-01 reference dataset audit summary
├── data_audit.csv                            # DET-01 feature statistics
├── data_audit.json                           # DET-01 complete dataset audit metadata
├── dl/
│   ├── dl_classification_report.csv          # Deep Learning per-class metrics
│   ├── dl_confusion_matrix.csv               # Deep Learning confusion matrix
│   ├── dl_metrics.json                       # Deep Learning ensemble evaluation scores
│   └── dl_test_predictions.csv              # Deep Learning test predictions
├── leakage_audit_report.json                 # Train-vs-pilot leakage audit JSON
├── leakage_audit_report.md                   # Train-vs-pilot leakage audit Markdown summary
├── pilot_traditional_alerts_10.json          # 10 schema-validated Traditional ML alerts (DET-05)
├── pilot_dl_alerts_10.json                   # 10 schema-validated DL alerts (DET-05)
├── pilot_manifest.json                       # Shared 10-flow pilot selection manifest
├── pilot_manifest.csv                        # CSV summary of 10 pilot flows
├── schema_and_semantic_validation_report.json # Comprehensive Draft 2020-12 validation report
├── split_manifest.json                       # Full train/test split manifest with SHA-256
├── traditional_classification_report.csv     # Traditional ML per-class metrics
├── traditional_confusion_matrix.csv          # Traditional ML confusion matrix
├── traditional_metrics.json                  # Traditional ML ensemble evaluation scores
└── traditional_test_predictions.csv          # Traditional ML test predictions
```

---

## 7. Archived Historical Artifacts

Legacy 5-feature TON_IoT artifacts generated prior to contract stabilization have been safely archived to:
`reports/archive_historical_ton_iot/`

Refer to `reports/archive_historical_ton_iot/README.md` for full provenance and rationale regarding why TON_IoT was retired in favor of the canonical 24-feature reference synthetic dataset.
