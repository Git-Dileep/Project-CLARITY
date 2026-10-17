# Project CLARITY

> **Controlled study evaluating whether SHAP attributions from deep learning vs. traditional ML ensembles impact LLM groundedness and hallucination in RAG incident analysis.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contracts: Draft 2020-12](https://img.shields.io/badge/Contracts-Draft%202020--12-success.svg)](contracts/)
[![CI: Automated Contract Checks](https://img.shields.io/badge/CI-Automated-brightgreen.svg)](.github/workflows/ci.yml)

---

## 📌 Repository Overview & Team Governance

The `main` branch acts as the **central integration hub and source of truth** for all shared API contracts, data models, and team boundaries.

### Team Roles & Modular Ownership
- **Lead / Evaluation:** Dileep (`contracts/`, `evaluation/`)
- **Detection Module:** Koushik (`detection/`)
- **Retrieval Module:** Sumit & Sneha (`retrieval/`)
- **Generation & RAG Module:** Sneha (`generation/`)

---

## 🏛️ Shared API Contracts (`contracts/`)

All cross-module interactions are governed by canonical Draft 2020-12 JSON schemas in [`contracts/`](file:///contracts/):

| Contract File | Purpose | Producers | Consumers |
| :--- | :--- | :--- | :--- |
| [`alert_schema_v1.json`](contracts/alert_schema_v1.json) | Alert payload with predictions & top-10 SHAP attributions | `detection/` (Koushik) | `generation/` (Sneha), `evaluation/` (Dileep) |
| [`retrieval_schema_v1.json`](contracts/retrieval_schema_v1.json) | Retrieved dense vector chunks and cached web context | `retrieval/` (Sumit/Sneha) | `generation/` (Sneha) |
| [`report_schema_v1.json`](contracts/report_schema_v1.json) | LLM incident report output structure | `generation/` (Sneha) | `evaluation/` (Dileep) |

> [!CAUTION]
> **Prompt Boundary Enforcement:**  
> `alert_schema_v1.json` includes `evaluation_metadata` (`true_label`, `in_train_leakage_flag`, and `detector_correct`). The prompt-building code in `generation/` **must explicitly strip `evaluation_metadata`** prior to constructing the LLM prompt to prevent evaluation leakage.

---

## 🤝 Contribution Workflow

1. **Work on Your Own Feature Branches:**  
   Develop your module in your assigned directory (`detection/`, `retrieval/`, `generation/`, or `evaluation/`).
2. **Read Shared Contracts:**  
   Import and validate against schemas in `contracts/`. Do not modify files in `contracts/` for local testing; create module-specific test fixtures if needed.
3. **Proposing Contract Changes:**  
   If you need a change to a shared schema:
   - Reach out to Dileep with the reason, proposed diff, and affected modules.
   - Dileep will review, commit the canonical contract to `main`, and notify the team.
4. **Pull Requests Required:**  
   Direct pushes to `main` are disabled. All code enters `main` via PRs with passing automated CI status checks.

---

## 📂 Repository Layout

```text
├── contracts/                  # Canonical shared API contracts (Maintained by Dileep)
│   ├── alert_schema_v1.json
│   ├── retrieval_schema_v1.json
│   ├── report_schema_v1.json
│   └── README.md
├── detection/                  # ML/DL models & SHAP extraction (Koushik)
├── retrieval/                  # Dense vector index & Tavily caching (Sumit / Sneha)
├── generation/                 # RAG prompt construction & LLM generator (Sneha)
├── evaluation/                 # Scoring engine, Rubric v0 & metrics (Dileep)
├── tests/                      # Canonical contract validation test suite
├── .github/
│   ├── CODEOWNERS              # Centralized review rules for contracts/
│   └── workflows/ci.yml        # Automated contract validation CI workflow
├── requirements.txt            # Project dependencies
└── README.md
```

---

## ⚡ Quickstart

```bash
# Clone the repository
git clone https://github.com/Git-Dileep/Project-CLARITY.git
cd Project-CLARITY

# Install dependencies
pip install -r requirements.txt

# Verify shared contracts
python -m pytest tests/test_contracts.py -v
```
