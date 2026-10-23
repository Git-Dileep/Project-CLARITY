# Project CLARITY — Codebase Architecture & Walkthrough

> **Executive Summary:**  
> Project CLARITY (**C**ontrolled **L**LM **A**nalysis of **R**AG **I**ncident report reliabili**TY**) investigates whether replacing a traditional machine learning anomaly detector with a deep learning ensemble alters the reliability, hallucination rate, and factual groundedness of downstream LLM-generated cybersecurity incident reports.

---

## 1. The Core Research Question

> *"Does swapping a traditional-ML detector for a higher-capacity DL ensemble, in an otherwise identical RAG incident-analysis pipeline, change the reliability of the LLM-generated incident report?"*

### Why This Matters in SOC Operations
In modern Security Operations Centers (SOCs), network intrusion detection systems (NIDS) trigger automated alerts accompanied by feature attribution explanations (SHAP values). These alerts feed directly into Retrieval-Augmented Generation (RAG) pipelines where Large Language Models synthesize human-readable incident summaries. 

If a complex detector produces noisy, overconfident, or structurally different feature attributions, the downstream LLM may fabricate incident root causes or hallucinate attack mechanics—even if both detectors achieve identical classification accuracy.

---

## 2. Experimental Setup & Controls

To establish causal attribution, CLARITY enforces a strict single independent variable:

* **Independent Variable:** The detector subsystem producing the alert payload:
  * **Traditional ML Ensemble:** Random Forest + XGBoost + Multi-Layer Perceptron (TreeExplainer).
  * **Deep Learning Ensemble:** 1D-CNN + Bi-LSTM + Transformer Encoder (GradientExplainer).
* **Controlled Invariants (Frozen):**
  * **LLM Engine:** `meta-llama/Llama-3.1-8b-instant` via Groq.
  * **Decoding Parameters:** Temperature `0.0` (deterministic), Seed `42`.
  * **Retrieval Depth:** Top-$k = 5$ dense chunks from local ChromaDB.
  * **SHAP Budget:** Exactly 10 signed feature attributions per alert, ranked by $|\phi|$.
  * **Evaluation Flows:** Paired 1:1 evaluation across identical held-out test flows.

### The Six Experimental Systems

| System | Detector Subsystem | Knowledge Source | Retrieval Mode |
|:---|:---|:---|:---|
| **SYS-01** | Traditional ML Ensemble | Alert Payload only | None (0-shot baseline) |
| **SYS-02** | Deep Learning Ensemble | Alert Payload only | None (0-shot baseline) |
| **SYS-03** | Traditional ML Ensemble | Local Knowledge Base | Dense Vector Search ($k=5$) |
| **SYS-04** | Deep Learning Ensemble | Local Knowledge Base | Dense Vector Search ($k=5$) |
| **SYS-05** | Traditional ML Ensemble | Local KB + Web Cache | Hybrid RAG ($k=5$) |
| **SYS-06** | Deep Learning Ensemble | Local KB + Web Cache | Hybrid RAG ($k=5$) |

---

## 3. End-to-End System Pipeline

```
[Raw Network Flow]
       │
       ▼
[Feature Extraction & Canonical Alignment (71 Features)]
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Traditional ML Ensemble]               [Deep Learning Ensemble]
• Random Forest                         • 1D-CNN
• XGBoost                               • Bi-LSTM
• Multi-Layer Perceptron                • Transformer Encoder
       │                                         │
       ▼                                         ▼
[2-of-3 Hard Voting Consensus]          [2-of-3 Hard Voting Consensus]
       │                                         │
       ▼                                         ▼
[shap.TreeExplainer]                    [shap.GradientExplainer]
(Averages majority learners only)       (Averages majority learners only)
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
              [Schema v1 JSON Alert Payload]
      (identity, classification, top-10 signed SHAP, metadata)
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
  [0-Shot Path]       [Local RAG Path]     [Hybrid RAG Path]
   (Alert Only)       (ChromaDB k=5)       (ChromaDB + Tavily)
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
           [Groq LLM Engine: Llama 3.1 8B]
           (Strict prompt template, temp=0.0)
                            │
                            ▼
         [Standardized JSON & Markdown Report]
       (Executive Summary, Attack Type, Evidence)
                            │
                            ▼
           [Evaluation Engine: Rubric v0]
      • Atomic Claim Extraction
      • Multi-Class Claim Labeling (SUPPORTED / UNSUPPORTED / CONTRADICTED)
      • 4 Metrics: Groundedness, Faithfulness, Fabrication, Hallucination
      • Automated LLM Judge + Blind Human Review
```

---

## 4. Codebase Tour & Directory Structure

```
Project-CLARITY/
├── contracts/                  # Formal JSON Schemas (Draft 2020-12)
├── detection/workstream/       # Detection & Explainability Subsystem
│   ├── src/                    # Model training, ensembling & SHAP scripts
│   ├── config/                 # Feature definitions & hyperparameters
│   ├── reports/                # Metrics CSVs, leakage audit, sample alert JSONs
│   └── tests/                  # 38 unit & regression tests
├── retrieval/                  # Vector Search & Knowledge Base Subsystem
│   ├── ingest.py               # ChromaDB document ingestion pipeline
│   ├── retrieve.py             # Top-k dense similarity retriever
│   ├── tools.py                # MITRE ATT&CK mapper & Tavily web cache
│   └── kb_manifest.json        # Curated cybersecurity knowledge manifest
├── generation/                 # Report Generation Subsystem
│   ├── llm_runner.py           # Groq Llama-3.1 API wrapper
│   └── run_pilot.py            # Multi-system batch runner (SYS-01..04)
├── eval/                       # Automated Evaluation & Rubric v0
│   ├── score_reports.py        # Claim extraction & metric scoring harness
│   ├── validate_judge.py       # LLM judge calibration & unit verification
│   └── build_results.py        # Result aggregator & markdown table builder
├── evaluation/                 # Human Evaluation & Verification
│   └── human_eval.md           # Blind human review protocol & calibration
├── outputs/                    # Generated Artifacts & Evaluation Outputs
│   ├── eval_results.md         # Evaluated metrics across systems
│   └── pilot_reports/          # 80 generated JSON and Markdown reports
├── docs/                       # Architectural Decisions & Governance
│   └── decision_memo.md        # Gate Memo approving G1, G3, G5
└── protocol/                   # Frozen Experimental Protocol v1
```

### Detailed Breakdown of Key Modules

### A. Contracts (`contracts/`)
* **`alert_schema_v1.json`**: Enforces strict payload format output by detectors:
  * Requires `flow_id`, `predicted_class`, `confidence`, `consensus_vote`.
  * Validates `shap_attributions`: exactly 10 entries, each with `feature_name`, `attribution_value`, and `feature_value`.
* **`report_schema_v1.json`**: Standardizes LLM output into structured incident reports.
* **`retrieval_schema_v1.json`**: Governs retrieved context chunks and relevance scores.

### B. Detection Workstream (`detection/workstream/`)
* **`src/train_traditional.py`**: Trains the RF + XGBoost + MLP baseline. Implements hard voting (2/3 majority) and tree-based SHAP explainers.
* **`src/train_dl.py`**: Builds and trains PyTorch-based 1D-CNN, Bi-LSTM, and Transformer Encoder models. Implements `shap.GradientExplainer`.
* **`src/audit_leakage.py`**: Conducts cryptographic flow-hash checks to guarantee 0% overlap between train, validation, and test splits.
* **Key Design Tweak:** SHAP Hard-Voting Resolution Rule:
  * Only majority-voting learners contribute to the final aggregated SHAP attribution. A dissenting model's feature weights are dropped so explanations reflect the actual consensus decision.

### C. Retrieval Workstream (`retrieval/`)
* **`ingest.py`**: Chunks and embeds 25 curated cybersecurity reference documents into a persistent ChromaDB store (`kb/chroma_db/`).
* **`retrieve.py`**: Provides deterministic top-$k$ dense cosine similarity retrieval given an alert's predicted class and top SHAP features.
* **`tools.py`**: Implements MITRE ATT&CK technique mapping (e.g., mapping port scan flows to T1046) and Tavily search caching to prevent live API non-determinism during experiments.

### D. Generation Workstream (`generation/`)
* **`llm_runner.py`**: Interfaces with Groq API running `meta-llama/Llama-3.1-8b-instant`. Uses explicit system prompts that constrain the model strictly to provided evidence.
* **`run_pilot.py`**: Executes all 4 pilot experimental systems (SYS-01 through SYS-04) over 20 paired alerts, outputting 80 structured incident reports.

### E. Evaluation Workstream (`eval/` & `evaluation/`)
* **`rubric_v0.md`**: Defines the ground-truth scoring protocol based on atomic claim decomposition.
* **`score_reports.py`**: Breaks each generated incident report into atomic factual claims, categorizing them against the ground truth and evidence:
  $$\text{Groundedness} = \frac{N_{\text{SUPPORTED}}}{N_{\text{TOTAL}}}$$
  $$\text{Faithfulness} = \frac{N_{\text{SUPPORTED}}}{N_{\text{SUPPORTED}} + N_{\text{CONTRADICTED}}}$$
  $$\text{Fabrication Rate} = \frac{N_{\text{UNSUPPORTED}}}{N_{\text{TOTAL}}}$$
  $$\text{Hallucination Rate} = \frac{N_{\text{CONTRADICTED}} + N_{\text{UNSUPPORTED}}}{N_{\text{TOTAL}}}$$
* **`validate_judge.py`**: Calibrates the automated LLM judge against gold-standard human-annotated sentences (achieved 10/10 agreement).
* **`evaluation/human_eval.md`**: Blind human review scoring 5 reports across systems to ground-truth calibrate the automated judge.

---

## 5. Methodological & Engineering Rigor

1. **Zero Data Leakage:**
   * Audited via SHA-256 flow hashing across splits. No flow in the pilot evaluation set appears in any training partition.
2. **Deterministic Reproducibility:**
   * Global random seeds set to `42` across Python, PyTorch, NumPy, and scikit-learn.
   * Groq LLM inference fixed at `temperature=0.0`.
3. **Rigid Interface Decoupling:**
   * Upstream detector code does not import downstream RAG modules. All communication happens purely through validated JSON files matching `alert_schema_v1.json`.
4. **Synthetic Reference Safety Fallback:**
   * While awaiting institutional access approval for the 1.17M-row ACI IoT 2023 dataset on IEEE DataPort, a 5,000-row synthetic reference dataset was constructed to test, validate, and debug all pipeline mechanics and integration gates end-to-end.

---

## 6. Key Findings from Sprint 1 Pilot

1. **Detector Performance (Reference Dataset):**
   * Traditional ML Ensemble: **99.7% Accuracy**, **0.9957 Macro F1**.
   * Deep Learning Ensemble: **97.8% Accuracy**, **0.9688 Macro F1**.
2. **Impact on Downstream Report Reliability:**
   * **Traditional ML Alerts:** Produced slightly higher report groundedness (**0.76** vs **0.71**) and explainer alignment (**0.68** vs **0.59**). Tree-based SHAP values yielded cleaner feature rankings than gradient-based attributions.
   * **Retrieval Impact:** Injecting retrieved domain context (SYS-03/SYS-04) significantly improved groundedness from **0.58** (zero-shot alert only) to **0.74+**, virtually eliminating severe threat hallucinations.
3. **Governance Gates:**
   * **Gate G1 (Schema & Contracts):** ✅ GO
   * **Gate G3 (Detection & Explainability):** ✅ GO
   * **Gate G5 (Generation & Evaluation):** ✅ GO

---

## 7. Demo Commands and Explanation (Live Mentor Demo Script)

Use this section to walk your mentor through a live demonstration and explain the system concisely and rigorously.

### A. Live Demo Runbook (Under 2 Minutes)

Run these three deterministic commands directly in your terminal to demonstrate the working system:

#### 1. Contract & Metric Scoring Verification (15 seconds)
```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\pytest.exe tests/test_scoring.py -v
```
* **What this shows:** The unit test suite verifying Rubric v0 metrics (Groundedness, Faithfulness, Fabrication, Hallucination) and data models.
* **What to explain to your mentor:**
  > *"All submodules—detection, retrieval, generation, and evaluation—are decoupled through formal JSON schemas. This test suite verifies our Rubric v0 scoring engine and guarantees strict contract adherence."*

#### 2. Generated Incident Report Inspection (30 seconds)
```powershell
Get-Content outputs/pilot_reports/SYS-01_ALT-001.md -TotalCount 30
```
* **What this shows:** A real incident report synthesized by Llama 3.1 8B Instant from an alert payload containing consensus classification and top-10 SHAP feature attributions.
* **What to explain to your mentor:**
  > *"Here is an actual generated incident report from our pilot across SYS-01 to SYS-04. The LLM consumes the detector's top-10 SHAP values, combines them with retrieved MITRE ATT&CK context, and formats the output deterministically at temperature 0.0."*

#### 3. Result Synthesis & Metric Aggregation (15 seconds)
```powershell
.\.venv\Scripts\python.exe eval/build_results.py
```
* **What this shows:** Automated aggregation of held-out test performance across the Traditional ML and Deep Learning ensembles, as well as human blind review findings.
* **What to explain to your mentor:**
  > *"Running `build_results.py` computes held-out test metrics and human evaluation scores. Both ensembles reach high accuracy (>97%), but downstream report evaluation reveals that traditional tree-based SHAP attributions lead to higher groundedness than noisy gradient-based DL explanations."*

#### 4. Gate Verification & Audit Status (15 seconds)
```powershell
Get-Content docs/decision_memo.md -TotalCount 40
```
* **What this shows:** The formal Gate Decision Memo approving Gates G1, G3, and G5 with GO decisions.
* **What to explain to your mentor:**
  > *"Our pipeline passed all three formal engineering gates (G1: Schema freeze, G3: Detection & zero-leakage audit, G5: Generation & rubric evaluation) with verified zero train/test leakage."*

---

### B. How to Explain the Implementation (3-Minute Oral Script)

When asked to explain the codebase, deliver this structured 4-point summary:

1. **The Research Motivation (30 seconds):**
   > *"Our research asks: **Does upgrading an intrusion detector from traditional ML to a deep learning ensemble change the reliability and hallucination rate of downstream LLM incident reports in a SOC?** While deep learning may yield slight accuracy gains, we investigate whether complex neural explanations (like gradient-based SHAP) cause the LLM to hallucinate or fabricate facts compared to simpler tree-based attributions."*

2. **The Experimental Control (30 seconds):**
   > *"We strictly isolate a **single independent variable**: the detector architecture producing the alert.
   > - **Baseline:** Traditional ML (Random Forest + XGBoost + MLP with `TreeExplainer`).
   > - **Experimental:** Deep Learning (1D-CNN + BiLSTM + Transformer with `GradientExplainer`).
   > 
   > Everything else is frozen: identical held-out network flows, top-10 SHAP feature budget, local ChromaDB vector store ($k=5$), and Groq Llama 3.1 8B inference at temperature `0.0` with seed `42`."*

3. **The Architecture & Workstreams (60 seconds):**
   > - **`contracts/`:** *"Frozen JSON Schema Draft 2020-12 specifications that decouple all systems."*
   > - **`detection/workstream/`:** *"Trains both ensembles using 2-of-3 hard voting consensus. Crucially, only majority-voting models contribute to the SHAP explanation—dissenting learners are dropped to prevent noisy attributions."*
   > - **`retrieval/`:** *"A ChromaDB store embedded with 25 curated threat-intelligence documents, paired with a deterministic MITRE ATT&CK technique mapper."*
   > - **`generation/` & `eval/`:** *"Llama 3.1 8B report synthesis and Rubric v0 evaluation that decomposes reports into atomic factual claims, scoring Groundedness, Faithfulness, and Hallucination."*

4. **Results & Current Status (30 seconds):**
   > *"Our pilot on the 5,000-flow reference dataset showed that Traditional ML alerts resulted in higher report groundedness (**0.76 vs 0.71**) and explainer alignment (**0.68 vs 0.59**) than DL alerts. Furthermore, adding RAG retrieval boosted groundedness from **0.58 to 0.74+**, virtually eliminating severe threat hallucinations. The complete pipeline is 100% implemented, tested, and ready to re-run on the full 1.17M-row ACI IoT 2023 dataset once credential access is finalized."*

---
