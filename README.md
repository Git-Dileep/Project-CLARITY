# Project CLARITY: Incident Analysis RAG Reliability Study

> **Evaluating whether SHAP attributions from deep learning vs. traditional ML ensembles impact LLM groundedness and hallucination in RAG incident analysis.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: DIL-02 Complete](https://img.shields.io/badge/Sprint%20Status-DIL--02%20Complete-brightgreen.svg)]()
[![Schema: Draft 2020-12](https://img.shields.io/badge/Schema-Draft%202020--12-success.svg)]()

---

## 📌 Research Question & Core Premise

> *"Does swapping a traditional-ML detector for a higher-capacity DL ensemble, in an otherwise identical RAG incident-analysis pipeline, change the reliability of the LLM-generated incident report?"*

In automated Security Operations Center (SOC) pipelines, machine learning detectors generate alerts and feature attributions (SHAP) that feed into Retrieval-Augmented Generation (RAG) models to synthesize human-readable incident reports. **Project CLARITY** executes a controlled 5-day evaluation sprint measuring the direct downstream effects on LLM report reliability when altering only the detector architecture.

### Experimental Controls
- **Independent Variable:** Detector Subsystem producing the alert payload.
  - **Traditional ML Baseline:** 3-model ensemble (**Random Forest + XGBoost + MLP**) with 2-of-3 hard voting.
  - **Deep Learning Ensemble:** 3-model ensemble (**1D-CNN + Bi-LSTM + Transformer Encoder**) with 2-of-3 hard voting.
- **Controlled Invariants:** Identical network flows ($1:1$ mapping), primary dataset (**ACI IoT Network Dataset 2023**), retrieval engine ($k=5$), LLM generator (`meta-llama/Llama-3.1-8b-instant`), decoding temperature ($0.0$), prompt template, and random seed ($42$).

---

## 🏗️ Architecture & Study Systems (SYS-01 to SYS-06)

```
                            ┌──────────────────────────────────────────────┐
                            │          Raw Network Flow (ACI IoT 2023)     │
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

> **Note on Metric Sensitivity:** As observed in the base paper, Groundedness frequently saturates at $1.0$. Therefore, **Faithfulness**, **Fabrication Rate**, and **Hallucination Rate** carry the primary statistical discrimination.

---

## 📂 Repository Structure

```text
├── protocol/
│   └── protocol_v1.md          # Frozen experimental protocol & constants
├── schema/
│   ├── alert_schema_v1.json    # Strict Draft 2020-12 alert JSON schema
│   └── examples/
│       ├── alert_trad_example.json  # Traditional ML alert payload (RF+XGB+MLP)
│       └── alert_dl_example.json    # DL ensemble alert payload (CNN+LSTM+Trans)
├── data/
│   ├── raw/                    # Raw flow captures and PCAPs
│   ├── pilot_alerts/           # 20 stratified pilot alert payloads
│   └── test/                   # Evaluation test splits
├── detectors/
│   ├── traditional/            # RF, XGBoost, and MLP models
│   └── deep_learning/          # 1D-CNN, Bi-LSTM, and Transformer models
├── kb/
│   ├── chroma_db/              # Persistent ChromaDB vector collections
│   └── tavily_cache/           # External web verification cache
├── pipeline/
│   └── run_rag.py              # End-to-end RAG generator pipeline
├── eval/
│   ├── rubric_v0.md            # Complete Rubric v0 specification & worked examples
│   └── score_reports.py        # Pydantic evaluation harness & metric calculator
├── tests/
│   └── test_scoring.py         # Pytest suite for scoring, tie-breaks & schemas
├── results/
│   └── reports/                # Generated incident reports across SYS-01..06
├── scripts/
│   └── validate_schemas.py     # Draft 2020-12 schema validation runner
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Setup & Execution

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Git-Dileep/Project-CLARITY.git
cd Project-CLARITY
pip install -r requirements.txt
```

### 2. Validate Alert Schemas
Verify that all alert payloads strictly conform to Draft 2020-12 rules:
```bash
python scripts/validate_schemas.py
```

### 3. Run Test Suite
Execute the unit and schema test suite with `pytest`:
```bash
python -m pytest tests/test_scoring.py -v
```

---

## 📅 Sprint Roadmap (5-Day Plan)

- [x] **DIL-01:** Scaffold workspace, freeze `alert_schema_v1.json`, draft `protocol_v1.md`, and initialize Git tracking.
- [x] **DIL-02:** Finalize `rubric_v0.md`, implement `score_reports.py` Pydantic harness, integrate ACI IoT 2023 dataset specifications, and establish `tests/test_scoring.py`.
- [ ] **DIL-03:** Ingest ACI IoT 2023 pilot dataset, extract 20 stratified flow pairs, and execute Gate G1 validation.
- [ ] **DIL-04:** Execute pilot inference across SYS-01 through SYS-06 and evaluate Gate G3 consistency.
- [ ] **DIL-05:** Full test set inference, Wilcoxon signed-rank significance testing, Gate G5 statistical synthesis, and final report generation.
