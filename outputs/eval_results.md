# Provisional Results: Synthetic Reference Dataset

> **These are preliminary results from the synthetic reference dataset, not final study results. The models and end-to-end report evaluation must be rerun on the real ACI IoT Network Dataset 2023 once it is available.**

The synthetic reference dataset contains 5,000 rows across six classes. The table reports detector ensemble performance on the 1,000-row held-out test split.

| Detector | Accuracy | Macro F1 | Test rows |
|:---|---:|---:|---:|
| Traditional ML ensemble | 99.7% | 0.9957 | 1,000 |
| Deep learning ensemble | 97.8% | 0.9688 | 1,000 |

A separate blind review of five generated pilot reports is documented in `evaluation/human_eval.md`; it is a small-sample result and should also be repeated after the real-dataset run.
