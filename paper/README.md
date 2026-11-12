# Project CLARITY Research Paper Guide

This directory contains the expanded, publication-ready **IEEE Conference Paper** for **Project CLARITY** (6+ pages with the three requested figure categories).

---

## 📁 Image Placement Guide

Place your three images inside the **`paper/figures/`** directory with the following exact filenames:

1. **`figures/solution_overview.png`**  
   * **Paper Section:** Section III (*Proposed Idea / Solution*, Figure 1)  
   * **Content:** High-level Project CLARITY solution diagram (traffic ingestion, structured alert, RAG knowledge layer, LLM incident analysis, evidence-grounded report).

2. **`figures/detailed_methodology.png`**  
   * **Paper Section:** Section IV (*Proposed Research / Methodology*, Figure 2)  
   * **Content:** Parallel detection ensembles, pre-detection filtering, 2-of-3 hard voting, consensus SHAP explainability, and downstream RAG evaluation.

3. **`figures/process_flow_diagram.png`**  
   * **Paper Section:** Section V (*Process / Flow / Design Diagram*, Figure 3)  
   * **Content:** Detailed process execution flow diagram (agentic workflow, subgraphs, vector store retrieval, and RAGAs/Rubric v0 evaluation).

---

## 🚀 How to Compile in Overleaf

1. Go to [Overleaf.com](https://www.overleaf.com).
2. Open your project.
3. Replace your `main.tex` with the newly updated `paper/main.tex`.
4. In Overleaf's file explorer on the left, create a folder named `figures`.
5. Upload your three images (`solution_overview.png`, `detailed_methodology.png`, `process_flow_diagram.png`) into that `figures/` folder.
6. Click **Recompile**.

The paper will render as a complete **6 to 7-page IEEE double-column conference paper**.
