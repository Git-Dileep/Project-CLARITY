# Generation & Evaluation Workstream (Sneha)

## Completed Tasks Overview

### SNE-01: LLM Report Generator
- **File:** `generation/llm_runner.py`
- Integrated Groq API using `llama-3.1-8b-instant` with `max_tokens=4096` for Schema v1 compliant JSON incident reports.
- Configured prompt for "expert IoT cybersecurity incident reporter".

### SNE-02: LLM Evaluation Judge
- **File:** `eval/score_reports.py`
- Built evaluation judge module imported via `from eval.score_reports import evaluate_report`.
- Evaluates generated reports against four core metrics: groundedness, faithfulness, fabrication, and hallucination.

### SNE-03: Sample Report Set Generation
- **Files:** `run_pilot_5.py`, `generation/sample_reports.json`
- Ingested pilot alerts from Koushik's export and generated schema-compliant sample incident reports saved in `generation/sample_reports.json`.