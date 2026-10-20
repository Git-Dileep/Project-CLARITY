# Human Blind Review Scoring Sheet (SUM-06)

- **Evaluator:** Sumit Lamani
- **Rubric Followed:** `eval/rubric_v0.md`
- **Source Data:** `generation/sample_reports.json` (SYS-01 pilot samples)
- **Target Issue:** Refs #18

---

## 1. Per-Sample Review & Categorical Breakdown

| Report ID | Flow ID | Groundedness | Faithfulness | Fabrication Rate | Hallucination Rate | Qualitative Findings & Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `ALT-20261003-0001` | FLOW-1 | 0.60 | 0.50 | 0.30 | 0.20 | Mentions ground-truth backdoor label not present in raw input alert. Top risk factors are raw placeholders (`Feature_1 (SHAP 0.0)`). Mitigations are technically sound for UDP floods, but generic. |
| `ALT-20261003-0002` | FLOW-2 | 0.65 | 0.55 | 0.25 | 0.20 | Cites ground truth backdoor intrusion without telemetry support in the input alert. UDP flood detection is grounded, but lacks concrete feature metric names. |
| `ALT-20261003-0003` | FLOW-3 | 0.70 | 0.60 | 0.20 | 0.20 | Captures 0.95 confidence faithfully. Introduces speculative dual-threat narrative based on ground-truth backdoor note. |
| `ALT-20261003-0004` | FLOW-4 | 0.85 | 0.80 | 0.10 | 0.10 | Well-aligned with UDP flood alert telemetry. Mitigations (scrubbing center, rate limits) are appropriate. Features remain unmapped `Feature_1..10`. |
| `ALT-20261003-0005` | FLOW-5 | 0.85 | 0.80 | 0.10 | 0.10 | High consistency with FLOW-4. Actionable mitigation recommendations. Needs actual feature column names instead of raw index placeholders. |

---

## 2. Summary & Automated Judge Agreement

- **Mean Groundedness:** 0.73
- **Mean Faithfulness:** 0.65
- **Mean Fabrication Rate:** 0.19
- **Mean Hallucination Rate:** 0.16
- **Automated Agreement:** High consensus with the automated judge regarding penalties for raw dummy feature names and ground-truth label leakage.
- **Scope Note:** Graded the 5 pilot reports currently available in `generation/sample_reports.json`. The remaining batch samples will be appended once multi-system generation completes.
