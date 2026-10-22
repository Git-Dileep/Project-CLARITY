# Detector Results and Dataset Justification — Draft

## 1. Dataset choice

The selected first candidate is the TON_IoT network collection supplied to the project. The local audit found 18 network CSV files, 17 unique by content, and 7 IoT device CSV files, all unique by content. The network audit counted 17,801,188 rows including the exact duplicate file; class summaries exclude exact duplicate network files. The audited network labels are scanning, DDoS, password, XSS, backdoor, ransomware, MITM, DoS and injection.

The network files expose `ts`, `src_ip`, `src_port`, `dst_ip`, `dst_port`, `proto` and `type`. The initial classifier uses source port, destination port and one-hot protocol features, excluding raw IPs and timestamp. This is a deliberately limited baseline: it lacks packet counts, flow duration, byte totals and other behavioural features. The IoT device files expose `ts`, `data`, `time` and `type`; their `data` field must be inspected and semantically validated before it is used as a model feature. No claim of official provenance or licence validation is made by the local audit.

## 2. Initial traditional detector results

The first sampled RF + XGBoost hard-voting run used 83,010 rows (66,408 training, 16,602 test) and reported 57.29% accuracy, 53.13% macro-F1 and 58.50% weighted-F1. These values are preliminary, based on a stratified random row split, and should not be described as real-world generalisation performance.

Per-class F1 was strongest for ransomware (96.78%) and scanning (92.96%). The weakest classes were MITM (11.54%), injection (19.07%) and password (31.43%). The confusion matrix shows substantial confusion among injection, password and XSS, which should be discussed as a limitation rather than hidden.

## 3. Initial deep-learning feasibility results

The first three-epoch CPU feasibility run reported:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| CNN | 30.87% | 12.62% |
| LSTM | 39.83% | 23.18% |
| Transformer | 43.35% | 24.67% |

The Transformer performed best among the three initial DL models, but all were below the traditional ensemble in this experiment. These results are not a fair final architecture comparison: the run is short, the rows are sampled, and temporal continuity in the sequence windows has not been validated.

## 4. Explainability

The initial SHAP summary ranked `src_port` and `dst_port` as the most influential features, while protocol indicators had smaller influence. This is consistent with a model trained on only port and protocol metadata, but it also indicates a risk of over-reliance on port patterns. SHAP explains the model's behaviour; it does not prove that a feature is causal or that a prediction is correct.

## 5. Next evaluation gates

Before this memo is finalised:

1. Rerun the shared-sample workflow so both detector outputs can be matched by sample ID.
2. Inspect all per-class precision, recall and F1 values and the confusion matrix.
3. Verify the traditional and DL SHAP outputs, including the explained class and feature names.
4. Export a 20-alert pilot covering at least five ground-truth classes and include predictions and explanations from both detectors.
5. State explicitly that random-row and sampled-window evaluation do not establish unseen-session generalisation. If time permits, run a chronological or source/session-based evaluation and report any classes that cannot be represented in both partitions.

## 6. Conclusion (provisional)

The initial experiments establish that the supplied network CSVs can be used for a multi-class classification baseline and that the project scripts can train traditional and DL feasibility models. Current performance is not yet sufficient to claim a validated operational IIoT intrusion detector. The next milestone is a reproducible shared-sample evaluation, explainability sanity checks and a traceable pilot alert set.
