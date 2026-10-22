# Protocol v1: Incident Analysis RAG Reliability Evaluation

**Status:** FROZEN  
**Date:** October 17, 2026 (Updated with ACI IoT 2023 & Rubric v0)
**Document ID:** PROTOCOL-V1

---

## 1. Research Question & Core Hypotheses

> **Core Research Question:**  
> *"Does swapping a traditional-ML detector for a higher-capacity DL ensemble, in an otherwise identical RAG incident-analysis pipeline, change the reliability of the LLM-generated incident report?"*

### Controlled Variable Strategy
To preserve strict experimental validity, this study enforces **exactly one manipulated variable**:
- **Independent Variable:** The detector subsystem producing the alert payload:
  - **Traditional ML (Paper's 3-Model Baseline):** Random Forest + XGBoost + Multilayer Perceptron (MLP) with majority hard voting.
  - **DL Ensemble:** 1D-CNN + Bi-LSTM + Transformer Encoder with majority hard voting.
- **Control Invariants:** Flow metadata, primary dataset, retrieval engine, knowledge base corpus, retrieval budget ($k=5$), LLM model version (`meta-llama/Llama-3.1-8b-instant`), prompt template, decoding temperature ($0.0$), and random seed ($42$).

---

## 2. Experimental Constants Table

| Constant | Value | Description / Scope | Validation |
| :--- | :--- | :--- | :--- |
| **Primary Dataset** | **ACI IoT Network Dataset 2023** | 1,170,570 flow records, 71 features, 11 classes (IEEE DataPort) | Confirmed |
| **Backup Datasets** | TON_IoT, IoT-23, BoT-IoT | Secondary fallback datasets if cross-domain test is run | Confirmed |
| **Target LLM Model**| `meta-llama/Llama-3.1-8b-instant` | Generator LLM (matching paper's 0.6391 faithfulness reference baseline) | Confirmed |
| **LLM Temperature** | `0.0` | Deterministic decoding mode | Confirmed |
| **Top-$k$ Retrieval** | `5` | Retrieved chunks from vector KB | Confirmed |
| **Top-$N$ SHAP Features**| `10` | Features passed into incident prompt | Confirmed |
| **Random Seed** | `42` | Global pseudo-random generator seed | Confirmed |
| **Embedding Model** | `text-embedding-3-small` / equivalent | Embedding engine for knowledge base chunks | To validate |
| **Vector DB** | ChromaDB (`chromadb>=0.4.22`) | Vector search database for local KB | Confirmed |
| **External Search** | Tavily Search API | Live verification cache / web lookup | To validate |
| **Feature Extraction** | 71 flow features (ACI IoT 2023) | Standardized tabular representation | To validate |

---

## 3. Detector Architecture Definitions

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
```

### 3.1 Detector 1: Traditional ML Baseline (`traditional_ml`)
- **Base Learners (Paper's 3-Model Baseline):**
  - Random Forest (`n_estimators=100`, `max_depth=12`)
  - XGBoost (`n_estimators=100`, `learning_rate=0.1`)
  - Multilayer Perceptron (MLP: `hidden_layer_sizes=(128, 64)`, `activation='relu'`)
- **Consensus Rule:** Hard voting majority (minimum 2 of 3 votes).
- **SHAP Tooling:** `shap.TreeExplainer` (for tree models) / unified majority attribution for the consensus class.

### 3.2 Detector 2: Deep Learning Ensemble (`dl_ensemble`)
- **Base Learners:**
  - 1D-CNN: Multi-channel 1D convolutions over sequence slices.
  - Bi-LSTM: 2-layer bidirectional LSTM capturing long-term packet dynamics.
  - Transformer Encoder: 4-layer multi-head self-attention module.
- **Consensus Rule:** Hard voting majority (minimum 2 of 3 votes).
- **SHAP Tooling:** `shap.GradientExplainer` (or `shap.DeepExplainer`).

---

## 4. Systems Table (SYS-01 through SYS-06)

| System ID | Detector Subsystem | Knowledge Sources | Retrieval Mechanism | Output Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **SYS-01** | Traditional ML (RF+XGB+MLP) | Alert Payload only (No RAG) | None (Baseline 0-shot) | Baseline LLM Report |
| **SYS-02** | DL Ensemble (CNN+LSTM+Trans) | Alert Payload only (No RAG) | None (Baseline 0-shot) | Baseline LLM Report |
| **SYS-03** | Traditional ML (RF+XGB+MLP) | Local KB (ChromaDB) | Dense Vector Search ($k=5$) | Standard RAG Report |
| **SYS-04** | DL Ensemble (CNN+LSTM+Trans) | Local KB (ChromaDB) | Dense Vector Search ($k=5$) | Standard RAG Report |
| **SYS-05** | Traditional ML (RF+XGB+MLP) | Local KB + Web (Tavily Cache) | Hybrid RAG ($k=5$) | Augmented RAG Report |
| **SYS-06** | DL Ensemble (CNN+LSTM+Trans) | Local KB + Web (Tavily Cache) | Hybrid RAG ($k=5$) | Augmented RAG Report |

---

## 5. Pilot Set Specification

- **Dataset:** ACI IoT Network Dataset 2023.
- **Sample Size:** Exactly 20 alerts (10 pairs across traditional ML vs DL ensemble).
- **1:1 Flow Mapping:** Every traditional alert pair `(ALT-xxxx, ALT-yyyy)` shares the identical network `flow_id`.
- **Stratification:** Minimum 5 distinct attack classes represented out of the 11 ACI IoT classes (e.g., `DDoS_SYN_Flood`, `Mirai_UDP_Flooding`, `Port_Scan`, `Brute_Force_SSH`, `Benign_Anomaly`).
- **Data Partition Integrity:**
  - $0\%$ train/test overlap (`in_train_leakage_flag: false`).
  - Strict disjoint IP subnets and temporal partitioning between training corpus and pilot flows.

---

## 6. Evaluation Metrics & Grounding Rubric

Reports are evaluated against `eval/rubric_v0.md` along four primary dimensions:
1. **Groundedness ($0.0 - 1.0$):** $(N_{\text{FULL}} + N_{\text{PARTIAL}}) / N$.
2. **Faithfulness ($0.0 - 1.0$):** $N_{\text{FULL}} / N$.
3. **Fabrication Rate ($0.0 - 1.0$):** $N_{\text{FABRICATED}} / N$.
4. **Hallucination Rate ($0.0 - 1.0$):** $(N_{\text{CONTRADICTED}} + N_{\text{UNSUPPORTED}}) / N$.

> [!NOTE]
> Groundedness saturated at 1.0 in the base paper. Faithfulness, Fabrication, and Hallucination carry the primary experimental distinction.

---

## 7. Decision Gates & Fallback Rules

```mermaid
flowchart TD
    G1{Gate G1: Schema, ACI-IoT & Baseline} -->|Pass: Zero Errors & Baseline ~99.2% F1| Exec[Run Pilot Inference SYS-01..06]
    G1 -->|Fail: Validation/Baseline Error| Hal1[Halt: Correct Schema & Retrain Baseline]
    Exec --> G3{Gate G3: Pilot RAG Consistency}
    G3 -->|Pass: Groundedness >= 0.70| Prod[Execute Full Test Evaluation]
    G3 -->|Fail: Groundedness < 0.70| FBack[Fallback: Clamp Prompt & Re-tune k]
    Prod --> G5{Gate G5: Statistical Synthesis}
    G5 -->|Pass: Significant Wilcoxon p < 0.05 on Faith/Fab/Halluc| Concl[Confirm Delta & Publish Finding]
    G5 -->|Fail: Null Outcome p >= 0.05| RepNull[Report Null Equivalent Hypothesis]
```

- **Gate G1 (End of Day 1):** 
  - Strict Schema freeze (`schema/alert_schema_v1.json`). Zero schema validation errors on pilot set; zero training leakage flags.
  - Class counts must match ACI IoT 2023.
  - Traditional baseline (RF + XGBoost + MLP) reproduction must land within ~1% macro-F1 of the paper (~99.2% macro-F1).
- **Gate G3 (End of Day 3):** Pilot RAG pipeline operational across SYS-01 through SYS-06. If Groundedness $< 0.70$, prompt template must be tightened to penalize speculative reasoning.
- **Gate G5 (End of Day 5):** 
  - Final comparative statistical test (Wilcoxon signed-rank test for paired flows).
  - **Special Constraint:** The Gate G5 decision **must not depend on groundedness delta** (due to saturation at 1.0). Primary decision metrics are Faithfulness, Fabrication Rate, and Hallucination Rate.

---

## 8. SHAP Hard-Voting Ensemble Resolution Rule

When resolving feature attributions for an ensemble using majority hard voting (e.g., 2-out-of-3 split):
1. **Consensus Majority Filtering:** Isolate the base learners whose vote matches the final consensus `predicted_label`.
2. **Signed Attribution Aggregation:**
   $$\Phi_i = \frac{1}{|M|} \sum_{m \in M} \phi_i^{(m)}$$
   where $M$ is the subset of learners supporting the majority class, and $\phi_i^{(m)}$ is the signed SHAP value of feature $i$ under learner $m$.
3. **Ranking & Slicing:**
   - Sort features in descending order of absolute magnitude $|\Phi_i|$.
   - Retain the top 10 ranked features.
4. **Architectural Trade-Offs:**
   - *Advantage:* Prevents minority dissident learners from contaminating the explanation of the winning decision.
   - *Limitation:* Disregards uncertainty expressed by dissenting models, which is preserved in `prediction.base_learner_votes`.

---

## 9. Out of Scope & Change Control

### Out of Scope for 5-Day Sprint
- End-to-end retraining of deep learning architectures with novel hyperparameter sweeps.
- Multi-turn interactive SOC analyst conversations.
- Dynamic autonomous agent tool-calling loops outside controlled retrieval.

### Change Control Policy
Any modification to `schema/alert_schema_v1.json` or `protocol/protocol_v1.md` requires review through an issue and pull request, with an incremental version tag (`v2`, `v3`).
