# Detection Module

This module implements and evaluates the network flow detectors and SHAP attribution pipelines.

## Responsibilities
- Baseline 3-model ensemble: Random Forest + XGBoost + MLP (Paper reproduction)
- Deep learning ensemble: 1D-CNN + Bi-LSTM + Transformer Encoder
- Hard-voting majority consensus (2-out-of-3)
- Top-10 SHAP feature extraction (`shap.TreeExplainer` & `shap.GradientExplainer`)
- Emitting payloads conforming strictly to `contracts/alert_schema_v1.json`
