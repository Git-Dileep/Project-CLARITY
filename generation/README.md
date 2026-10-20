# Generation Module (Lead: Sneha)

This module handles prompt synthesis, LLM interaction, and structured incident report production.

## Responsibilities
- Implementing SYS-01 through SYS-06 RAG pipeline variations
- Generator LLM integration (`meta-llama/Llama-3.1-8b-instant`, temperature 0.0)
- **CRITICAL:** Explicitly stripping `evaluation_metadata` from the alert payload in prompt-building code to prevent evaluation leakage
- Formatting outputs conforming strictly to `contracts/report_schema_v1.json`
