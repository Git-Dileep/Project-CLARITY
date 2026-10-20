# Generation & Evaluation Workstream (Sneha)

## Completed Tasks Overview

### SNE-02: LLM Report Generator
- **File:** `generation/llm_runner.py`
- Integrated Groq API to generate Schema v1 compliant JSON incident reports.
- Model Configured: `openai/gpt-oss-20b` with `max_tokens=4096`.

### SNE-04: Evaluation Harness
- **File:** `evaluation/run_harness.py`
- Built automated evaluation harness to validate generated incident reports against schema specifications and ground truth metrics.

### SNE-05: Pilot Batch Report Generation
- **Files:** `run_pilot_5.py`, `generation/sample_reports.json`
- Ingested pilot alerts exported from `detection/koushik_workstream/reports/pilot_20_alerts.csv`.
- Generated 5 schema-compliant sample incident reports stored in `generation/sample_reports.json`.