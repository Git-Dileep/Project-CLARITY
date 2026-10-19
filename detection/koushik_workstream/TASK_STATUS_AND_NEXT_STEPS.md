# KOU Task Status and Evidence Gates

This status is based on the first run logs supplied by Koushik, not on a rerun of the updated workflow.

| Task | First-run status | Evidence / remaining gate |
|---|---|---|
| KOU-01 Dataset, preprocessing, split | Partially complete | Audit counted 17 unique network files and 7 unique IoT device files. Need rerun with shared sample and document split limitations. |
| KOU-02 RF + XGBoost hard voting | Initial baseline run complete | Existing result: accuracy 57.29%, macro-F1 53.13%. Updated run should create `modeling_sample.csv` and `traditional_test_predictions.csv`. |
| KOU-03 CNN/LSTM/Transformer skeleton | Initial feasibility run complete | Existing run completed 3 epochs/model. Shared-sample retrain required for matched pilot IDs. |
| KOU-04 Traditional metrics + tree SHAP | Initial outputs generated | Classification report, confusion matrix, feature importance and SHAP summary exist. Verify after updated run. |
| KOU-05 First full DL training run | Not complete | Three epochs/model is a feasibility run, not a full training checkpoint. Increase epochs only after confirming data/split quality and runtime. |
| KOU-06 DL metrics, per-class F1 and DL-SHAP | Partially complete | Per-class metrics exist in the old output; rerun updated script to attempt Transformer GradientExplainer output. Inspect warnings and attribution quality. |
| KOU-07 Stratified 20-alert pilot | Not complete | Run `run_pilot.bat` after both models have been trained from the shared sample. It must cover at least five ground-truth classes and include explanations from both models. |
| KOU-08 Confirm detector results and SHAP sanity | Not complete | Manually inspect weak classes, confusion matrix, both SHAP outputs, and data leakage risks. |
| KOU-09 Results + dataset justification memo | Draft only | Use `RESULTS_MEMO_DRAFT.md`; finalize only after rerunning the workflow and verifying pilot outputs. |

## Important limitations

- The current network schema is `ts, src_ip, src_port, dst_ip, dst_port, proto, type`. It does not contain common flow-level behavioural metrics such as duration, bytes, packet counts or TCP flags.
- The IoT device schema shown in the audit is `ts, data, time, type`. Do not use `type` or any target-derived field as an input feature.
- The updated DL script shares the sampled rows and split seed with the traditional baseline, but its windows are built from sampled rows and are not guaranteed to be contiguous network events. Treat CNN/LSTM/Transformer metrics as feasibility results only until genuine chronological windows are constructed.
- The pilot script attempts SHAP for XGBoost and GradientExplainer for the Transformer. If a warning appears or the output is incomplete, the corresponding task is not complete.
- A random stratified holdout is useful for class-balanced baseline comparison, but it does not prove generalisation to unseen devices, attack sessions or time periods.
