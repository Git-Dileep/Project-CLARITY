# Koushik — Detection Workstream Handoff

- Owner: Koushik
- Intended branch: `feature/koushik-detection`
- Files changed: detection documentation and `detection/koushik_workstream/` only; shared `contracts/` schemas are not modified.
- Current contents: audit/training/pilot scripts plus copied historical evidence for the TON_IoT-style feasibility run.
- Remaining critical gate: run the agreed three-model traditional ensemble (RF + XGBoost + MLP) and three-model DL ensemble (CNN + LSTM + Transformer) on the frozen ACI IoT 2023 dataset; validate group/time split and export contract-compliant alerts with exactly ten SHAP features.

The current legacy results do not satisfy the parent repository's full experiment protocol or alert contract. The workstream status document records this explicitly to prevent overclaiming.
