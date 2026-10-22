# Evaluation Module

This module implements the Rubric v0 evaluation engine, atomic claim scoring, and statistical synthesis.

## Responsibilities
- Implementing Rubric v0 atomic claim parsing and labeling (`FULL`, `PARTIAL`, `UNSUPPORTED`, `CONTRADICTED`, `FABRICATED`)
- Metric calculation: Faithfulness, Groundedness, Fabrication Rate, Hallucination Rate
- Wilcoxon signed-rank paired statistical tests (Gate G5)
- Automated report evaluation against Reference Evidence Set $E$
