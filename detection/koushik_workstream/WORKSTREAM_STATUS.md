# Koushik Detection Workstream — Evidence and Completion Status

## Scope and dataset boundary

This workstream contains the existing TON_IoT-derived feasibility pipeline and evidence supplied for review. The parent Project CLARITY protocol currently names **ACI IoT Network Dataset 2023 (1,170,570 flow records, 71 features, 11 classes)** as the frozen primary dataset. The supplied sample in this workstream is from **TON_IoT-style GroundTruth_Network CSVs** with only `ts, src_ip, src_port, dst_ip, proto, type`-style fields. These are not interchangeable. Do not present TON_IoT metrics as the frozen ACI IoT experiment or claim the full CLARITY detector comparison is complete until the ACI dataset is supplied and the frozen pipeline is run on it.

## Evidence available in this folder

- `reports/modeling_sample.csv`: 83,010-row sampled network table, containing `sample_id`, `source_file`, `ts`, `src_port`, `dst_port`, `proto`, and `type`.
- `reports/traditional_classification_report.csv`: existing RF+XGBoost baseline per-class report. The corresponding historical summary was accuracy 57.29%, macro-F1 53.13%, weighted-F1 58.50%; only five encoded features were used (`src_port`, `dst_port`, and protocol one-hot columns).
- `reports/dl/dl_metrics.json`: 3-epoch feasibility outputs. CNN accuracy 45.04%, macro-F1 24.40%; LSTM accuracy 46.76%, macro-F1 29.30%; Transformer accuracy 58.52%, macro-F1 39.01%. These are feasibility results, not a completed full training run.
- `reports/pilot_20_alerts.csv`: 20-row pilot spanning 7 labels. Existing pilot accuracy: RF+XGBoost 40% (8/20); Transformer 10% (2/20).
- `reports/pilot_shap_values.csv`: local SHAP values from the XGBoost component and Transformer. XGBoost-component SHAP explains the XGBoost learner, not necessarily the hard-voting ensemble decision.
- `reports/pilot_summary.json`: pilot summary and caveat.

These outputs are historical evidence copied from prior runs. The scripts in `src/` can rerun the TON_IoT feasibility workflow after the dataset CSVs are placed in `input_data/network/` and (for the audit) `input_data/iot_devices/`.

## Task tracker

| ID | Status | Honest evidence / remaining gate |
|---|---|---|
| KOU-01 dataset audit/preprocessing/split | Partial | Audit/sample workflow exists. Split is random stratified; source-file/session holdout and primary ACI dataset verification remain. |
| KOU-02 RF + XGBoost hard voting | Initial baseline available | Existing TON_IoT-style baseline: 57.29% accuracy, 53.13% macro-F1, 58.50% weighted-F1. Feature set is limited. |
| KOU-03 CNN/LSTM/Transformer skeleton | Implemented | Model definitions and 3-epoch feasibility training exist. |
| KOU-04 traditional metrics + tree SHAP | Partial | Metrics and pilot XGBoost-component SHAP exist; distinguish component explanation from ensemble explanation. |
| KOU-05 first full DL run/checkpoint | Not complete | 3 epochs/model are feasibility runs only. Full run should wait for dataset and split validation. |
| KOU-06 DL metrics + DL-SHAP | Partial | Metrics and Transformer SHAP artifacts exist, but attribution output needs validation on the agreed primary dataset. |
| KOU-07 20-alert pilot with 5+ classes and both explanations | Pilot artifact exists; validation partial | 20 rows / 7 labels / both detector families present. Current pilot accuracies 40% and 10%; small and not independent session validation. |
| KOU-08 validate weak classes and SHAP | Partial | Existing pilot reveals Transformer predicts XSS for 15/20 alerts; inspect per-class metrics and SHAP before reporting. |
| KOU-09 results and dataset justification memo | Draft | `RESULTS_MEMO_DRAFT.md` is retained as a draft, not a final validated claim. |

## Important limitations

1. Do not commit raw datasets or trained model binaries unless repository owners explicitly request them and licensing/size are checked.
2. The current network schema lacks behavioural flow statistics such as duration, packet/byte counts, TCP flags, and flow rate. Avoid claiming robust traffic-behaviour detection from ports/protocol alone.
3. The 20-alert pilot is sampled from a random holdout and does not establish unseen-session/device generalization.
4. `pilot_shap_values.csv` reports the XGBoost component, not the RF+XGBoost voting decision, for traditional SHAP rows.
5. The root CLARITY contract expects three base learners and exactly ten top SHAP features per alert. The legacy KOU pipeline's two-model ensemble and five-feature SHAP table do not yet satisfy that contract. Do not connect its output directly to generation until an adapter is implemented and validated against `contracts/alert_schema_v1.json`.
