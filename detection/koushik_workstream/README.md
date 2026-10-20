# Koushik Detection Workstream (legacy TON_IoT feasibility workspace)

This folder preserves the existing chunked audit, RF+XGBoost baseline, CNN/LSTM/Transformer feasibility scripts, and the most recent review evidence. It is intentionally nested under `detection/` so it can be committed on `feature/koushik-detection` without changing the shared contracts or teammates' modules.

> **Dataset boundary:** the current sample/results are from TON_IoT-style `GroundTruth_Network` CSVs. The parent CLARITY protocol specifies ACI IoT Network Dataset 2023 as the frozen primary dataset. These existing results are not the final CLARITY experiment.

## Windows setup

1. Install Python 3.10 or 3.11 and Git.
2. Obtain the approved dataset files separately. Do not add raw datasets to Git.
3. Put network CSVs in `input_data/network/`; optional IoT-device CSVs go in `input_data/iot_devices/`.
4. Double-click `run_setup.bat` to create `.venv` and install the workstream requirements.
5. Run `run_audit.bat`, then `run_all.bat` from this directory.
6. Review `reports/`, `reports/dl/`, and `WORKSTREAM_STATUS.md`; do not describe feasibility metrics as final results.

## Evidence included

- `reports/modeling_sample.csv`
- `reports/traditional_classification_report.csv`
- `reports/dl/dl_metrics.json`
- `reports/pilot_20_alerts.csv`
- `reports/pilot_shap_values.csv`
- `reports/pilot_summary.json`

The scripts may overwrite generated reports when rerun. `input_data/`, model checkpoints, and virtual environments are excluded from source control by `.gitignore`.

## Before sharing results

- Run `python -m compileall src` from this directory to check Python syntax.
- Run `python -m src.audit_datasets` after placing dataset files to regenerate audit outputs.
- Check `WORKSTREAM_STATUS.md` and parent `Koushik_Detection_Workstream.md` before citing any metric.
- Do not change `contracts/` without the repository lead's approval.
