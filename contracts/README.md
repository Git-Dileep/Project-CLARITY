# Shared API Contracts Governance & Change Procedure

This directory contains the **canonical API contracts and JSON schemas** for Project CLARITY.  
All modules across detection, retrieval, generation, and evaluation interact through these standardized boundaries.

---

## 1. Governance Rules

1. **Centralized Authority:**  
   The schemas in `contracts/` on the `main` branch represent the **single source of truth**. Only Dileep (Study Lead) approves and merges changes to files in `contracts/`.
2. **Branch Protection:**  
   Direct pushes to `main` are disabled. All changes must arrive via reviewed pull requests.
3. **Decoupled Development:**  
   Team members (Koushik, Sneha, Sumit) work in their respective module folders (`detection/`, `retrieval/`, `generation/`, `evaluation/`) on their own feature branches. You do **not** need to modify contracts to implement your code.
4. **Test Fixtures:**  
   If you need mock or sample data for unit tests, place them in your module's test directories or fixtures. Do not modify the canonical schemas in `contracts/` for local testing convenience.

---

## 2. Shared Contracts Overview

| Contract File | Purpose | Consumers / Producers |
| :--- | :--- | :--- |
| [`alert_schema_v1.json`](file:///C:/Users/DILEEP/.gemini/antigravity-ide/scratch/Project%20CLARITY/contracts/alert_schema_v1.json) | Alert payload with predictions & top-10 SHAP attributions | **Produced by:** Koushik (`detection/`)<br>**Consumed by:** Sneha (`generation/`), Evaluation (`evaluation/`) |
| [`retrieval_schema_v1.json`](file:///C:/Users/DILEEP/.gemini/antigravity-ide/scratch/Project%20CLARITY/contracts/retrieval_schema_v1.json) | Retrieved dense vector chunks and cached web context | **Produced by:** Sumit / Sneha (`retrieval/`)<br>**Consumed by:** Sneha (`generation/`) |
| [`report_schema_v1.json`](file:///C:/Users/DILEEP/.gemini/antigravity-ide/scratch/Project%20CLARITY/contracts/report_schema_v1.json) | LLM incident report output structure | **Produced by:** Sneha (`generation/`)<br>**Consumed by:** Dileep (`evaluation/`) |

---

## 3. How to Propose a Contract Change

If your implementation genuinely requires a change to a shared schema:
1. **Coordinate First:** Do not assume schema modifications without discussion.
2. **Submit a Contract RFC:** Send Dileep an issue or message with:
   - **Reason:** Why the current schema is insufficient.
   - **Proposed Diff:** The specific additions, deletions, or type alterations.
   - **Compatibility Impact:** Backwards compatibility evaluation.
   - **Affected Modules:** List of team members whose code will be impacted.
3. **Review & Merge:** Once reviewed and accepted, Dileep will update the schema in `main` and notify the team.

---

## 4. Critical Architecture Constraint: Prompt Boundary Enforcement

> [!CAUTION]
> **Documentation is not a security boundary.**  
> While `alert_schema_v1.json` defines `evaluation_metadata` (containing `true_label`, `in_train_leakage_flag`, and `detector_correct`), the **prompt-building code in `generation/` MUST explicitly strip `evaluation_metadata`** prior to formatting the LLM generation prompt. Passing ground-truth labels to the generator LLM invalidates the study.
