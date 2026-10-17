# Project CLARITY

A controlled study evaluating whether local feature attributions (SHAP) from deep learning ensembles versus tree-dominant ensembles affect downstream LLM groundedness, faithfulness, fabrication, and hallucination in RAG-driven IoT incident analysis.

---

## 📌 Research Framing & Experimental Setup

* **Core Research Question:** When a baseline detector is replaced by a higher-capacity DL ensemble in an otherwise identical RAG incident-analysis pipeline, does the resulting SHAP attribution change the reliability of the LLM-generated incident report?
* **Single Manipulated Variable:** Detector Family
  * **Baseline Ensemble:** Random Forest + XGBoost + MLP (Hard voting majority rule; matches the base paper baseline).
  * **Deep Learning Ensemble:** 1D-CNN + LSTM + Transformer Encoder (Hard voting majority rule).
* **Constants (Frozen):**
  * **Primary Dataset:** ACI IoT Network Dataset 2023 (1,170,570 flow records, 71 features, 11 classes; IEEE DataPort).
  * **Attribution Method:** Local SHAP (averaged over agreeing majority learners).
  * **Knowledge Base:** Single fixed ChromaDB collection (Generic IoT threat profiles, MITRE ATT&CK generic mappings, and network defense guidance) + cached Tavily search snippets.
  * **Default LLM:** `meta-llama/Llama-3.1-8b-instant` at `temperature: 0.0` (matching the base paper's reference baseline).
  * **Target Metrics:** Groundedness, Faithfulness, Fabrication Rate, Hallucination Rate (evaluated via Rubric v0).

---

## 👥 Team Workstreams & Module Ownership

| Role | Workstream | Primary Directory | Owner | Key Responsibilities |
| :--- | :--- | :--- | :--- | :--- |
| **Lead / Eval** | Experimental Protocol & Evaluation | `contracts/`, `evaluation/` | **Dileep** (`@Git-Dileep`) | Architecture governance, protocol freeze, canonical schemas, Rubric v0 scoring harness. |
| **Detection** | Baseline & DL Ensembles, SHAP | `detection/` | **Koushik** | ACI IoT 2023 data processing, RF+XGB+MLP baseline, CNN+LSTM+Transformer, SHAP exporters. |
| **Retrieval** | Dense Store & Cached Tools | `retrieval/` | **Sumit & Sneha** | ChromaDB embedding collection, Tavily search caching, ATT&CK mapping tools. |
| **Generation** | Prompt Engineering & RAG Pipeline | `generation/` | **Sneha** | RAG orchestration, system prompt construction, Llama-3.1-8b runner, report export. |

---

## 🏛️ Shared API Contracts (`contracts/`)

All cross-module interfaces are governed strictly by Draft 2020-12 JSON schemas in `contracts/`.

| Contract File | Purpose | Producer | Consumer |
| :--- | :--- | :--- | :--- |
| [`alert_schema_v1.json`](contracts/alert_schema_v1.json) | Alert payload with classification votes & top-10 SHAP attributions | `detection/` | `generation/`, `evaluation/` |
| [`retrieval_schema_v1.json`](contracts/retrieval_schema_v1.json) | Retrieved dense vector chunks and cached web search context ($k=5$) | `retrieval/` | `generation/` |
| [`report_schema_v1.json`](contracts/report_schema_v1.json) | Generated incident report payload for metric evaluation | `generation/` | `evaluation/` |

> ⚠️ **Critical Prompt Boundary Rule:**
> `alert_schema_v1.json` contains `evaluation_metadata` (`true_label`, `in_train_leakage_flag`, `detector_correct`). The prompt-building code in `generation/` **must programmatically strip `evaluation_metadata`** prior to constructing the LLM prompt to prevent ground-truth data contamination.

---

## 🤝 Contribution & Governance Workflow

1. **Feature Branches:** Work exclusively on personal feature branches (e.g., `feature/koushik-detection`, `feature/sneha-generation`). Direct pushes to `main` are disabled.
2. **Schema Immutability:** Schemas inside `contracts/` are locked and protected by `.github/CODEOWNERS`. If a schema modification is needed:
   - File an RFC issue or notify Dileep with the rationale, proposed schema diff, and affected downstream modules.
   - Once approved, Dileep will update the schema on `main`.
3. **Pull Requests & CI Validation:** All PRs targeting `main` must pass the automated GitHub Actions contract validation test suite (`.github/workflows/ci.yml`).

---

## 📂 Repository Layout

```text
Project-CLARITY/
├── contracts/                  # Canonical shared API schemas (Maintained by Dileep)
│   ├── alert_schema_v1.json
│   ├── retrieval_schema_v1.json
│   ├── report_schema_v1.json
│   └── README.md
├── detection/                  # Detectors, training pipelines & SHAP extractors (Koushik)
├── retrieval/                  # ChromaDB vector store & Tavily web cache (Sumit / Sneha)
├── generation/                 # Prompt builder & Llama-3.1 RAG generation runner (Sneha)
├── evaluation/                 # Rubric v0 scoring harness & metric evaluation (Dileep)
├── tests/                      # Canonical contract validation test suite
│   └── test_contracts.py
├── .github/
│   ├── CODEOWNERS              # Governance: Dileep required for contracts/
│   └── workflows/ci.yml        # Automated CI workflow
├── requirements.txt            # Unified project dependencies
└── README.md

⚡ Quickstart
# 1. Clone the repository
git clone [https://github.com/Git-Dileep/Project-CLARITY.git](https://github.com/Git-Dileep/Project-CLARITY.git)
cd Project-CLARITY

# 2. Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify canonical shared schemas
python -m pytest tests/test_contracts.py -v
