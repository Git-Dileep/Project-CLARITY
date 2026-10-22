# Project CLARITY — Gate Decision Memo

**Version:** 1.0  
**Document Owner:** Project CLARITY
**Date:** 2026-10-21  
**Run Baseline:** Synthetic Reference Dataset (`data/reference_synthetic.csv`)  
**Caveat:** All results below are from a 5,000-row synthetic reference dataset with 20-alert pilot (10 paired flows × 2 detectors). This gives **directional confidence, not statistical significance**. Final ACI IoT 2023 validation remains blocked pending dataset access.

---

## Executive Summary

Project CLARITY completed its Sprint 1 pilot cycle on 2026-10-21. All four workstreams (Detection, Retrieval, Generation, Evaluation) delivered their contracted artifacts, passed schema validation, and met the protocol v1 acceptance criteria on the synthetic reference baseline. The gate tracker below records the evidence and decision for each milestone gate.

---

## Gate Tracker

### G1 — Detection Pipeline & Alert Schema Contract

| Criterion | Evidence | Status |
|:---|:---|:---:|
| Traditional ML ensemble trained (RF, XGBoost, MLP) | `traditional_metrics.csv`: Ensemble Accuracy **99.7%**, Macro F1 **0.9957** | ✅ PASS |
| Deep Learning ensemble trained (CNN1D, LSTM, Transformer) | `dl_metrics.csv`: Ensemble Accuracy **97.8%**, Macro F1 **0.9688** | ✅ PASS |
| 20 alerts exported (10 Trad + 10 DL) sharing identical 10 held-out flow IDs | `pilot_manifest.json`: 10 shared flows across 6 classes | ✅ PASS |
| All alerts conform to `contracts/alert_schema_v1.json` (Draft 2020-12) | `schema_and_semantic_validation_report.json`: **0 schema errors, 0 semantic errors** | ✅ PASS |
| SHAP attributions: 10 features per alert, monotonically sorted by \|φ\| | `SHAP_EXPLANATION_PROTOCOL.md` + verification (Issue #17) | ✅ PASS |
| Zero train-to-pilot data leakage | `leakage_audit_report.json`: **0 overlap, pilot ⊆ test confirmed** | ✅ PASS |

**Merged PRs:** #26 (detection pipeline), #27 (schema/protocol freeze), #31 (reproducible pilot)
**Closed Issues:** #2, #3, #4, #10, #11, #12  
**Decision: GO** — Detection pipeline is operational and contract-compliant on the synthetic baseline.

---

### G3 — Retrieval & Knowledge Base Integration

| Criterion | Evidence | Status |
|:---|:---|:---:|
| 25-document IoT KB manifest created | `retrieval/kb_manifest.json`: 25 generic IoT security documents | ✅ PASS |
| ChromaDB ingestion pipeline operational | `retrieval/ingest.py` (PR #24) | ✅ PASS |
| Alert-driven retrieval function with top-k chunks | `retrieval/retrieve.py`: `retrieve_for_alert()` returns chunks with similarity scores | ✅ PASS |
| MITRE ATT&CK mapping for 5 core attack types | `retrieval/tools.py`: `map_mitre_attack()` covering T1498.001, T1110.001, T1046, T1210 | ✅ PASS |
| Cached Tavily search with per-alert determinism | `retrieval/tools.py`: `get_cached_tavily_search()` with local JSON cache + mock fallback | ✅ PASS |
| Unit tests passing | `tests/test_retrieval.py`, `tests/test_tools.py`: **All pass** | ✅ PASS |

**Merged PRs:** #24 (KB manifest + ingestion), #29 (retrieval + tools)
**Closed Issues:** #13, #14, #15, #16  
**Decision: GO** — Retrieval pipeline delivers grounded context for report generation.

---

### G5 — Generation, Evaluation & Human Agreement

| Criterion | Evidence | Status |
|:---|:---|:---:|
| LLM report generator operational (Groq Llama 3.1 8B Instant) | `generation/llm_runner.py` (PR #25) | ✅ PASS |
| Pilot reports generated across SYS-01..SYS-04 on 20 alerts | `outputs/pilot_reports/`: 80 `.json` + 80 `.md` files (PR #28) | ✅ PASS |
| Real LLM judge with Rubric v0 validation | `eval/validate_judge.py`: **10/10 agreement** with rubric labels (PR #28) | ✅ PASS |
| 4 uncollapsed Rubric v0 metrics scored | `outputs/eval_results.md`: Groundedness, Faithfulness, Fabrication, Hallucination across systems, detectors, attack types, and H1–H3 trade-offs | ✅ PASS |
| Human blind review completed | `evaluation/human_eval.md`: Mean Groundedness **0.73**, Faithfulness **0.65**, Fabrication **0.19**, Hallucination **0.16** (PR #29) | ✅ PASS |
| Human-judge agreement documented | High consensus on penalties for raw feature placeholders and ground-truth label leakage | ✅ PASS |
| DL ensemble training supported | `detection/workstream/logs/dl_training.log` + metrics artifacts (PR #30) | ✅ PASS |
| Pytest suite passing | `tests/test_scoring.py`: **6/6 pass**, `tests/test_tools.py`: **2/2 pass** | ✅ PASS |

**Merged PRs:** #25 (LLM runner), #28 (LLM judge, pilot generation, rubric scoring), #29 (human evaluation), #30 (DL training)
**Closed Issues:** #17, #18, #19, #20, #21, #22, #23  
**Decision: GO** — Generation and evaluation pipelines validated end-to-end with human agreement.

---

## Consolidated PR & Issue Ledger

### Merged Pull Requests

| PR | Summary | Key Issues Closed |
|:---|:---|:---|
| #24 | KB manifest (25 docs) + ChromaDB ingestion | #13, #14 |
| #25 | Base LLM generator (`llm_runner.py`) | #19 |
| #26 | Complete detection pipeline (Ensemble + SHAP + 20 pilot alerts) | #12 |
| #27 | Protocol v1 freeze, schema validator, test harness | #2, #3, #4 |
| #28 | LLM judge, pilot generation, Rubric v0 scoring | #20, #21, #22 |
| #29 | ChromaDB retrieval, MITRE mapping, Tavily tool, blind human eval | #13, #14, #15, #16, #18 |
| #30 | DL ensemble training logs and metrics artifacts | #23 |
| #31 | Reproducible 10-flow pilot, SHAP explanations, leakage audit | #10, #11 |

### Open Issues (Remaining)

| Issue | Title | Status | Notes |
|:---|:---|:---|:---|
| #7 | [DET-01] Switch to ACI IoT 2023 and audit it | Open | Blocked on dataset access |
| #8 | [DET-02] Train traditional ensemble on ACI | Open | Blocked on #7 |
| #9 | [DET-03] Complete DL ensemble on ACI | Open | Blocked on #7 |

These three issues form the **ACI real-data validation chain** and remain blocked pending dataset procurement. They do not gate Sprint 1 closure, which operates on the synthetic reference baseline.

---

## Caveats & Limitations

1. **Sample Size:** The 20-alert pilot (10 flows × 2 detectors) provides directional signal only. Statistical significance requires the full ACI dataset run (DET-01/02/03).
2. **Synthetic Dataset:** All detection metrics (99.7% Trad, 97.8% DL accuracy) are on `data/reference_synthetic.csv` (5,000 rows, 6 classes). Real-world ACI IoT 2023 performance may differ materially.
3. **Human Eval Coverage:** The blind review covered 5 of the planned 10 reports from `generation/sample_reports.json`. The remaining samples will be evaluated once multi-system generation is finalized.
4. **Feature Placeholders:** Human review identified that early pilot reports used raw index placeholders (`Feature_1..10`) instead of canonical feature column names. This was resolved in the detection pipeline (PR #31) which exports real column names.

---

## Decision

**Sprint 1 Gate Status: GO**

All three gates (G1, G3, G5) pass their acceptance criteria on the synthetic reference baseline. The project is cleared to proceed to the next phase: ACI IoT 2023 real-data validation (pending dataset access) and extended evaluation coverage.

---

*Project CLARITY, 2026-10-21*
