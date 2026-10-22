# Project CLARITY: Incident Analysis RAG Reliability Study

> **Evaluating whether SHAP attributions from deep learning vs. traditional ML ensembles impact LLM groundedness and hallucination in RAG incident analysis.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Pilot Sprint Complete](https://img.shields.io/badge/Sprint%20Status-Pilot%20Sprint%20Complete-brightgreen.svg)]()
[![Schema: Draft 2020-12](https://img.shields.io/badge/Schema-Draft%202020--12-success.svg)]()
[![Gates: G1 G3 G5 GO](https://img.shields.io/badge/Gate%20Status-G1%20%7C%20G3%20%7C%20G5%20%3A%20GO-brightgreen.svg)](docs/decision_memo.md)

---

## 📌 Research Question & Core Premise

> *"Does swapping a traditional-ML detector for a higher-capacity DL ensemble, in an otherwise identical RAG incident-analysis pipeline, change the reliability of the LLM-generated incident report?"*

In automated Security Operations Center (SOC) pipelines, machine learning detectors generate alerts and feature attributions (SHAP) that feed into Retrieval-Augmented Generation (RAG) models to synthesize human-readable incident reports. **Project CLARITY** executes a controlled evaluation sprint measuring the direct downstream effects on LLM report reliability when altering only the detector architecture.

### Experimental Controls
- **Independent Variable:** Detector Subsystem producing the alert payload.
  - **Traditional ML Baseline:** 3-model ensemble (**Random Forest + XGBoost + MLP**) with 2-of-3 hard voting.
  - **Deep Learning Ensemble:** 3-model ensemble (**1D-CNN + Bi-LSTM + Transformer Encoder**) with 2-of-3 hard voting.
- **Controlled Invariants:** Identical network flows ($1:1$ mapping), reference dataset schema, retrieval engine ($k=5$), LLM generator (`meta-llama/Llama-3.1-8b-instant`), decoding temperature ($0.0$), prompt template, and random seed ($42$).

---

## 🏗️ Architecture & Study Systems (SYS-01 to SYS-06)

```
                            ┌──────────────────────────────────────────────┐
                            │            Raw Network Flow Event            │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                         [ Feature Extractor ]
                                                   │
                           ┌───────────────────────┴───────────────────────┐
                           ▼                                               ▼
                [ Traditional ML Baseline ]                     [ DL Ensemble Family ]
             ┌─────────────────────────────┐               ┌─────────────────────────────┐
             │  - Random Forest (100 trees)│               │  - 1D-CNN (Temporal Conv)   │
             │  - XGBoost (Grad Boosted)   │               │  - Bi-LSTM (2-layer Bi-Dir) │
             │  - MLP (Multi-Layer Percep) │               │  - Transformer Encoder      │
             │  - Explainer: TreeExplainer │               │  - Explainer: GradExplainer │
             └─────────────┬───────────────┘               └──────────────┬──────────────┘
                           │                                              │
                           ▼                                              ▼
                [ Hard Voting (2 of 3) ]                       [ Hard Voting (2 of 3) ]
                           │                                              │
                           └───────────────────────┬──────────────────────┘
                                                   ▼
                                     [ Schema v1 JSON Alert Payload ]
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                               ▼                               ▼
       [ Baseline (0-Shot RAG) ]         [ Standard Local RAG ]          [ Hybrid Local + Web RAG ]
       - SYS-01 (Trad ML)                - SYS-03 (Trad ML)              - SYS-05 (Trad ML)
       - SYS-02 (DL Ensemble)            - SYS-04 (DL Ensemble)          - SYS-06 (DL Ensemble)
                   │                               │                               │
                   └───────────────────────────────┼───────────────────────────────┘
                                                   ▼
                                    [ LLM Incident Report (Llama 3.1) ]
                                                   │
                                                   ▼
                                [ Evaluation Engine (Rubric v0) ]
```

| System ID | Detector Subsystem | Knowledge Sources | Retrieval Mechanism | Output Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **SYS-01** | Traditional ML (RF+XGB+MLP) | Alert Payload only (No RAG) | None (Baseline 0-shot) | Baseline LLM Report |
| **SYS-02** | DL Ensemble (CNN+LSTM+Trans) | Alert Payload only (No RAG) | None (Baseline 0-shot) | Baseline LLM Report |
| **SYS-03** | Traditional ML (RF+XGB+MLP) | Local KB (ChromaDB) | Dense Vector Search ($k=5$) | Standard RAG Report |
| **SYS-04** | DL Ensemble (CNN+LSTM+Trans) | Local KB (ChromaDB) | Dense Vector Search ($k=5$) | Standard RAG Report |
| **SYS-05** | Traditional ML (RF+XGB+MLP) | Local KB + Web (Tavily Cache) | Hybrid RAG ($k=5$) | Augmented RAG Report |
| **SYS-06** | DL Ensemble (CNN+LSTM+Trans) | Local KB + Web (Tavily Cache) | Hybrid RAG ($k=5$) | Augmented RAG Report |

---

## 📊 Evaluation Rubric v0 & Reliability Metrics

Reports are decomposed into disjoint atomic claims and graded strictly against Reference Evidence Set $E$ (alert metadata, top-10 SHAP features, and retrieved knowledge):

### Categorical Labels & Tie-Break Hierarchy
Claims are assigned exactly one label under strict tie-break precedence:
$$\mathbf{FABRICATED} \succ \mathbf{CONTRADICTED} \succ \mathbf{UNSUPPORTED} \succ \mathbf{FULL} \mid \mathbf{PARTIAL}$$

- **`FABRICATED`**: Invents specific unlisted CVEs, MITRE IDs, external IPs, quantitative metrics, or feature names not present in $E$.
- **`CONTRADICTED`**: Directly negates or conflicts with explicit facts or values in $E$.
- **`UNSUPPORTED`**: Speculative deduction or generic botnet trivia unsupported by $E$, with no fabricated identifiers.
- **`PARTIAL`**: Supported in concept or spirit, but over-generalized, imprecise, or hedged.
- **`FULL`**: 100% corroborated by explicit facts in $E$.

### Mathematical Formulations
$$\text{Groundedness} = \frac{N_{\text{FULL}} + N_{\text{PARTIAL}}}{N}$$
$$\text{Faithfulness} = \frac{N_{\text{FULL}}}{N}$$
$$\text{Fabrication Rate} = \frac{N_{\text{FABRICATED}}}{N}$$
$$\text{Hallucination Rate} = \frac{N_{\text{CONTRADICTED}} + N_{\text{UNSUPPORTED}}}{N}$$

> **Key Findings from Pilot Sprint:** See [`outputs/eval_results.md`](outputs/eval_results.md) and [`docs/decision_memo.md`](docs/decision_memo.md) for full metric breakdowns across all systems and attack classes. Standard RAG (SYS-03/04) drives Groundedness to near 1.0, making Faithfulness, Fabrication Rate, and Hallucination Rate the primary discriminatory metrics.

---

## 📂 Repository Structure

```text
├── contracts/                  # Frozen Draft 2020-12 JSON Schemas
│   ├── alert_schema_v1.json    # Canonical alert contract (metadata + top-10 SHAP)
│   ├── report_schema_v1.json   # Incident report structure schema
│   └── retrieval_schema_v1.json# Retrieval context chunk schema
├── detection/                  # Detection Workstream
│   └── workstream/                # Traditional & DL ensemble models, SHAP, and audit
│       ├── reports/            # Pilot alert payloads & leakage audit reports
│       └── src/                # Training, inference, SHAP, and export pipelines
├── retrieval/                  # Retrieval Workstream
│   ├── ingest.py               # ChromaDB document ingestion pipeline
│   ├── kb_manifest.json        # Curated 25-document IoT threat knowledge base
│   ├── retrieve.py             # Top-k vector retrieval engine
│   └── tools.py                # MITRE ATT&CK mapping & cached Tavily web search
├── generation/                 # Report Generation Workstream
│   ├── llm_runner.py           # Groq Llama 3.1 8B report generator
│   └── run_pilot.py            # End-to-end multi-system report generator
├── eval/                       # Evaluation & Scoring
│   ├── rubric_v0.md            # Rubric v0 specification & claim grading rules
│   ├── score_reports.py        # Pydantic automated claim scoring harness
│   ├── validate_judge.py       # LLM judge validation against ground truth
│   └── build_results.py        # Aggregate metric compilation
├── evaluation/                 # Human Evaluation & Verification
│   └── human_eval.md           # Blind human review & rubric calibration
├── outputs/                    # Sprint Results & Artifacts
│   ├── eval_results.md         # Final metric tables & hypothesis tests (H1-H3)
│   └── pilot_reports/          # Generated JSON & Markdown reports across systems
├── docs/                       # Project Governance & Milestones
│   └── decision_memo.md        # Formal Gate Decision Memo (G1, G3, G5 = GO)
├── protocol/                   # Experimental Protocol
│   └── protocol_v1.md          # Frozen experimental parameters & constants
├── scripts/                    # Automation & Verification
│   └── validate_schemas.py     # JSON Schema validation runner
├── tests/                      # Pytest Test Suites
│   ├── test_contracts.py       # Schema presence and Draft 2020-12 compliance
│   ├── test_scoring.py         # Rubric v0 calculations & tie-break tests
│   ├── test_tools.py           # MITRE & Tavily tool cache tests
│   └── test_retrieval.py       # Vector retrieval tests
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Setup & Execution

### 1. Installation
Clone the repository using its configured GitHub URL, then set up a virtual environment:
```bash
cd Project-CLARITY
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Validate Contracts & Schemas
Verify that all alert payloads and contracts conform strictly to JSON Schema Draft 2020-12:
```bash
python scripts/validate_schemas.py
```

### 3. Run Test Suite
Execute the unit and contract validation test suites:
```bash
python -m pytest tests/test_contracts.py tests/test_scoring.py tests/test_tools.py -v
```

### 4. Evaluate Generated Reports
Score incident reports against Rubric v0:
```bash
python eval/score_reports.py
```

---

## 📋 Milestone & Gate Status

All Sprint 1 milestone gates (G1, G3, G5) were evaluated and approved under **GO** decisions on the synthetic reference baseline. See [`docs/decision_memo.md`](docs/decision_memo.md) for full audit trails and decision matrices.

| Gate | Scope | Status | Notes |
| :--- | :--- | :---: | :--- |
| **G1** | Detection & Alert Schema | **GO** | 20 alerts exported (10 Trad + 10 DL), 0 schema errors, 0 data leakage |
| **G3** | Knowledge Base & Retrieval | **GO** | 25-doc KB, ChromaDB retrieval, MITRE mapping & cached Tavily operational |
| **G5** | Generation & Evaluation | **GO** | 80 pilot reports scored, LLM judge 10/10 agreement, 4 Rubric v0 metrics synthesized |

### Open Workstream Items (Dataset Dependent)
The following workstream tickets remain open pending full real-world ACI IoT 2023 dataset access:
- **Issue #7 [DET-01]:** Switch to real ACI IoT 2023 dataset and audit schema
- **Issue #8 [DET-02]:** Train traditional ensemble on real ACI IoT 2023
- **Issue #9 [DET-03]:** Complete deep learning ensemble training on real ACI IoT 2023
