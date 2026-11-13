# Detection Tasks — Status and Next Steps

## 1. Primary Issue Status Matrix (Real ACI IoT 2023 Dataset)

| Task ID | Description | Status | Evidence & Acceptance Gates |
| :--- | :--- | :--- | :--- |
| **DET-01** | Switch to ACI IoT Network Dataset 2023 and audit it | **COMPLETE** | Audited on real ACI IoT 2023 dataset (88,784 flows across 12 classes). Class distribution and split manifest recorded in `reports/aci_iot_2023/`. |
| **DET-02** | Train traditional ensemble on ACI | **COMPLETE** | 3-model majority voting ensemble (Random Forest, XGBoost, MLP) trained and evaluated on 80/20 split (71,027 train / 17,757 test). Models persisted in `models/traditional_ensemble.joblib`. |
| **DET-03** | Complete deep-learning ensemble on ACI | **COMPLETE** | 3 PyTorch models (1D-CNN, LSTM, Transformer Encoder) trained and evaluated on identical test split. Checkpoints persisted in `models/dl_ensemble.pt`. |
| **DET-04** | Generate SHAP explanations for both ensembles with 10 features | **COMPLETE** | TreeExplainer and GradientExplainer generate 10 ranked SHAP attributions for pilot flows. Mathematical protocol verified. |
| **DET-05** | Export stratified pilot for both detectors | **COMPLETE** | 20-flow pilot alerts exported for both traditional and DL ensembles (`pilot_traditional_alerts.json`, `pilot_dl_alerts.json`). Schema validated against contract. Baseline snapshotted for regression testing. |
| **DET-06** | Make `.bat` run scripts reproducible | **COMPLETE** | Automated scripts and reproducible test suites verified across workstreams. |

---

## 2. Soft Voting Extension Workstream Status

| Task ID | GitHub Issue | Description | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **SOFT-01** | #32 | Continuous Posterior Probability Soft Voting in `ensemble.py` | **COMPLETE** | `compute_soft_voting_predictions`, `assert_class_alignment`, `compute_voting_discordance`, `compute_mcnemar_test`, `compute_paired_bootstrap_macro_f1`, and 11-class primary metric implemented and unit tested (13/13 passing in `test_models.py`). |
| **SOFT-REG** | - | Baseline Snapshot & Byte-Identity Regression Suite | **COMPLETE** | 16 baseline files snapshotted in `snapshots/baseline_hard_voting/manifest.json`. `tests/test_hard_voting_regression.py` validates 100% byte-identity. |
| **SOFT-SCH** | - | Alert Schema Additive v1.1 Evolution | **COMPLETE** | `contracts/alert_schema_v1_1.json` and `schema/alert_schema_v1_1.json` added with optional soft voting fields; passes `tests/test_contracts.py`. |
| **SOFT-02** | #33 | Integrate Soft Voting into Traditional ML Ensemble Pipeline | **NEXT UP** | Persist `predict_proba` matrix, derive Soft Voting from saved models, export dual `results/hard/` and `results/soft/`. |
| **SOFT-03** | #34 | Integrate Soft Voting into Deep Learning Ensemble Pipeline | **PENDING** | Symmetrically apply probability saving and dual export to DL pipeline. |
| **SOFT-04** | #35 | Export Dual Hard and Soft Voting Pilot Alerts with Schema Validation | **PENDING** | Keep default alerts byte-identical; generate `_hard.json` and `_soft.json` over frozen flow IDs. |
| **SOFT-05** | #36 | Update Results Aggregator and Evaluation Tables for Dual Voting | **PENDING** | Dual metrics comparison tables with pre-specified 11-class primary metric. |
| **SOFT-06** | #37 | Update Documentation, Experimental Protocol, and Paper Draft | **PENDING** | Neutral phrasing, all result values `[TBD]`. |

---

## 3. Immediate Resume Instructions

When resuming:
1. Models `traditional_ensemble.joblib` and `dl_ensemble.pt` are already saved in `detection/workstream/models/`.
2. Core mathematics and test suites are in place: run `python -m pytest tests/test_hard_voting_regression.py tests/test_contracts.py detection/workstream/tests/test_models.py` to confirm clean test passes.
3. Proceed directly to **SOFT-02 (Issue #33)** and **SOFT-03 (Issue #34)** to wire probability persistence and dual export (`results/hard/` and `results/soft/`) into traditional and DL evaluation pipelines.
