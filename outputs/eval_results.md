# Evaluation Results: ACI IoT Network Dataset 2023

> **Final study results evaluated on the real-world ACI IoT Network Dataset 2023.**

The ACI IoT 2023 dataset contains 1,231,411 rows across 12 attack/traffic classes. The table reports detector ensemble performance on the 17,757-row held-out test split.

| Detector | Accuracy | Macro F1 | Test rows |
|:---|---:|---:|---:|
| Traditional ML ensemble | 99.4% | 0.9075 | 17,757 |
| Deep learning ensemble | 98.2% | 0.8473 | 17,757 |

A separate blind review of five generated pilot reports is documented in `evaluation/human_eval.md`.
