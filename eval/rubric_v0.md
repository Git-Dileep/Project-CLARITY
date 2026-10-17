# Evaluation Rubric v0 (Draft)

## Overview
This rubric provides the operational definitions and scoring guidelines for incident report reliability evaluation across systems SYS-01 to SYS-06.

### Dimensions
1. **Groundedness Score [0.0 – 1.0]**:
   - `1.0`: All stated claims, indicators, and metrics can be attributed directly to retrieved chunks or input SHAP attributes.
   - `0.5`: General technical deduction matches, but secondary claims lack explicit citation.
   - `0.0`: Major claims are completely ungrounded.

2. **Faithfulness Score [0.0 – 1.0]**:
   - `1.0`: No contradictory statements relative to detector confidence, label, or feature rankings.
   - `0.0`: Report contradicts detector output (e.g. asserts low risk when detector reported 0.94 confidence DDoS).

3. **Fabrication Rate [0.0 – 1.0]**:
   - Fraction of unsupported atomic assertions (e.g. fabricated CVE numbers, IP addresses, or malware families).

4. **Hallucination Severity**:
   - `None`: Fully supported by data.
   - `Minor`: Plausible extrapolations without operational harm.
   - `Severe`: Fabricated IOCs, false root causes, or misleading remediation directives.
