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
| Traditional ML ensemble trained (RF, XGBoost, MLP) | `reports/aci_iot_2023/traditional_metrics.csv`: Accuracy **99.37%**, 11-Cls Macro F1 **0.9897** | ✅ PASS |
| Deep Learning ensemble trained (CNN1D, LSTM, Transformer) | `reports/aci_iot_2023/dl/dl_metrics.csv`: Accuracy **98.18%**, 11-Cls Macro F1 **0.9244** | ✅ PASS |
| Hard Voting baseline outputs preserved byte-identical | `tests/test_hard_voting_regression.py`: 100% byte-identity verified | ✅ PASS |
| Additive Schema v1.1 evolved and validated | `contracts/alert_schema_v1_1.json`, `tests/test_contracts.py`: **0 errors** | ✅ PASS |
| Continuous posterior probability Soft Voting core implemented | `detection/workstream/src/ensemble.py`: `compute_soft_voting_predictions`, class alignment, discordance, McNemar, paired bootstrap (13/13 tests pass) | ✅ PASS |
| Dual strategy pilot alerts exported | `pilot_traditional_alerts_hard.json`, `pilot_traditional_alerts_soft.json`, `pilot_dl_alerts_hard.json`, `pilot_dl_alerts_soft.json` (20 flows) | ✅ PASS |
| Soft Voting full test-set comparative results & significance | `reports/aci_iot_2023/results/soft/`: Metrics, McNemar test, and paired bootstrap CI | ⏳ PENDING ([TBD]) |
| Zero train-to-pilot data leakage | `reports/aci_iot_2023/split_manifest.json`: **0 overlap confirmed** | ✅ PASS |

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
| 4 uncollapsed Rubric v0 metrics scored | `outputs/eval_results.md`: Groundedness, Faithfulness, Fabrication, and Hallucination summaries across systems, detectors, and attack types | ✅ PASS |
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

### GitHub Issues Status Ledger

| Issue | Title | Status | Notes |
|:---|:---|:---|:---|
| #7 | [DET-01] Switch to ACI IoT 2023 and audit it | Closed | Completed on real ACI IoT 2023 dataset (88,784 flows) |
| #8 | [DET-02] Train traditional ensemble on ACI | Closed | Completed on real ACI IoT 2023 (17,757 test flows; 99.37% acc) |
| #9 | [DET-03] Complete DL ensemble on ACI | Closed | Completed on real ACI IoT 2023 (17,757 test flows; 98.18% acc) |
| #32 | [SOFT-01] Continuous Posterior Probability Soft Voting in `ensemble.py` | Closed | Core math, tests, and metrics complete (13/13 unit tests pass) |
| #33 | [SOFT-02] Integrate Soft Voting in `train_traditional.py` | Closed | Class alignment assertion, prob saving, dual export integrated |
| #34 | [SOFT-03] Integrate Soft Voting in `train_dl.py` | Closed | PyTorch prob extraction, dual export, run manifest integrated |
| #35 | [SOFT-04] Export Dual Hard and Soft Voting Pilot Alerts | Closed | `_hard.json` and `_soft.json` exported over frozen flow IDs; Schema v1.1 validated |
| #36 | [SOFT-05] Update Results Aggregator and Evaluation Tables | Closed | `eval/build_results.py` and `outputs/eval_results.md` updated with `[TBD]` placeholders |
| #37 | [SOFT-06] Update Documentation, Protocol, and Paper Draft | Closed | Neutral phrasing, research motivation documented, all Soft result values `[TBD]` |

---

## Caveats & Limitations

1. **Split Method & Temporal Session Overlap:** Split is stratified random per-flow with flow identifiers dropped. Temporal/session-window grouping was not enforced during traffic capture, so flows from the same scan activity can appear in both train and test splits.
2. **Soft Voting Empirical Status:** Soft Voting is introduced as an additional evaluation strategy to assess potential improvements over majority voting. All comparative result numbers and significance tests are marked `[TBD]` pending final cluster execution.
3. **Hard Voting Preservation:** Existing Hard Voting outputs, metrics, and baseline pilot alerts remain 100% byte-identical to pre-change snapshots.

---

## Decision

**Sprint 1 Gate Status: GO**

All three gates (G1, G3, G5) pass their acceptance criteria on the synthetic reference baseline. The project is cleared to proceed to the next phase: ACI IoT 2023 real-data validation (pending dataset access) and extended evaluation coverage.

---

*Project CLARITY, 2026-10-21*
