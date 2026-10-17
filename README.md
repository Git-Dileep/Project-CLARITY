# Incident Analysis RAG Study: Traditional ML vs Deep Learning Ensembles

## Research Question
> *"Does swapping a traditional-ML detector for a higher-capacity DL ensemble, in an otherwise identical RAG incident-analysis pipeline, change the reliability of the LLM-generated incident report?"*

## Architecture Overview
This study compares two detector families in an end-to-end Incident Analysis RAG pipeline:
1. **Traditional ML**: Random Forest + XGBoost (TreeExplainer)
2. **Deep Learning Ensemble**: 1D-CNN + LSTM + Transformer Encoder (GradientExplainer)

Outputs conform to `schema/alert_schema_v1.json` and are fed into the retrieval and grounding pipeline (`pipeline/run_rag.py`) evaluated under `eval/rubric_v0.md`.

## Directory Structure
- `protocol/`: Protocol definitions and constants (`protocol_v1.md`)
- `schema/`: Alert JSON schemas and example payloads (`alert_schema_v1.json`, `examples/`)
- `data/`: Pilot alerts, raw datasets, and evaluation test sets
- `detectors/`: Traditional ML and DL ensemble implementations
- `kb/`: Knowledge base persistence (ChromaDB vectors & Tavily cache)
- `pipeline/`: Incident analysis RAG pipeline
- `eval/`: Rubric definitions and automated evaluation scoring
- `results/`: LLM incident reports and benchmark outcomes
- `scripts/`: Validation and utility scripts

## Quickstart & Validation
Verify schema conformance:
```bash
python scripts/validate_schemas.py
```
