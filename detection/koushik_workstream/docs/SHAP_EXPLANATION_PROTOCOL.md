# SHAP Explanation Protocol for Detection Ensembles

**Project CLARITY — Detection Workstream**  
**Document ID:** `DOC-SHAP-PROTOCOL-V1`  
**Applicability:** Traditional ML Ensemble & Deep Learning Ensemble Alert Exports  

---

## 1. Mathematical Background & Ensemble Attribution Rationale

In Project CLARITY, both the Traditional Detector and the Deep Learning Detector operate as **majority-voting ensembles** (hard voting across exactly 3 base learners) to eliminate tie votes and maximize classification robustness:
- **Traditional Ensemble:** Random Forest (`RandomForestClassifier`), XGBoost (`XGBClassifier`), Multi-layer Perceptron (`MLPClassifier`).
- **Deep Learning Ensemble:** 1D-CNN (`CNN1DModel`), LSTM (`LSTMModel`), Transformer Encoder (`TransformerModel`).

### Why Component-Level Explanations are Used for Majority-Voting Consensus
Shapley values require a well-defined, continuous characteristic function \( v(S) \) satisfying the efficiency, symmetry, dummy, and additivity axioms. A discrete majority vote over three heterogeneous model families (e.g., decision trees, non-linear neural networks, and attention mechanisms) produces a step-wise, non-differentiable consensus decision surface \( \mathcal{C}(x) \in \{0, 1\}^K \). 

Averaging Shapley values computed across disparate mathematical spaces (e.g., tree log-odds from `TreeExplainer` versus integrated gradients from `GradientExplainer`) lacks a unified baseline \( \mathbb{E}[f(x)] \) and violates the additive efficiency axiom.

Therefore, the **CLARITY Detection Explanation Protocol** explicitly defines the following attribution rules:

1. **Traditional Detector (`traditional_ml`):**
   - **Explainer:** Genuine SHAP `TreeExplainer` (Lundberg et al., Nature MI 2020).
   - **Target Model:** Gradient Boosted Decision Tree (`XGBoost`) member of the ensemble.
   - **Explained Target:** The consensus predicted class \( \hat{y}_{\text{consensus}} \).
   - **Baseline (\( \phi_0 \)):** Model expected value \( \mathbb{E}[f(X_{\text{train}})] \) for the target class.
   - **Rationale:** `TreeExplainer` provides exact, polynomial-time Shapley values without sampling variance.

2. **Deep Learning Detector (`dl_ensemble`):**
   - **Explainer:** Genuine SHAP `GradientExplainer` (Sundararajan et al., ICML 2017).
   - **Target Model:** `TransformerModel` (Multi-Head Self-Attention Encoder) member of the ensemble.
   - **Explained Target:** The consensus predicted class \( \hat{y}_{\text{consensus}} \).
   - **Baseline (\( \phi_0 \)):** Empirical mean output computed over a representative background dataset (\( N=50 \)) sampled exclusively from the training partition.
   - **Rationale:** Evaluates integrated feature gradients along the path from baseline inputs to the observed flow representation.

---

## 2. Top-10 Feature Selection and Ranking Protocol

In strict conformance with [`contracts/alert_schema_v1.json`](../contracts/alert_schema_v1.json), each exported alert payload must contain **exactly 10 ranked feature attributions**:

```json
"top_features": [
  {
    "rank": 1,
    "feature_name": "<canonical_feature_name>",
    "feature_value": <float>,
    "shap_value": <signed_float>
  },
  ... (ranks 1 through 10)
]
```

### Strict Ranking Rules:
1. **Absolute Magnitude Ordering:** Features are sorted in strictly non-increasing order of absolute attribution:
   \[
   |\phi_{(1)}| \ge |\phi_{(2)}| \ge \dots \ge |\phi_{(10)}|
   \]
   In the event of exact absolute equality, ties are deterministically resolved by ascending canonical feature index.
2. **Contiguous Ranking:** Ranks are sequential integers \( 1, 2, \dots, 10 \) with no gaps, skips, or duplicate ranks.
3. **Signed Preservation:** While ordering is governed by \( |\phi_j| \), the exported `shap_value` retains its original mathematical sign:
   - \( \phi_j > 0 \): Feature pushed the prediction toward the explained class.
   - \( \phi_j < 0 \): Feature pushed the prediction away from the explained class.
4. **Canonical Vocabulary Alignment:** Feature names must originate exclusively from the fitted training preprocessor (`preprocessor.feature_names_`). No synthetic padding or fabricated features are permitted.

---

## 3. Preprocessing & Leakage Integrity

1. **Fitted Exclusively on Train:** The `FeaturePreprocessor` pipeline (median imputation, standard scaling, and one-hot encoding with `handle_unknown='ignore'`) is fitted strictly on the training partition. Test and pilot flows are transformed through the frozen preprocessor.
2. **Identity & Label Exclusion:** All non-behavioral identifiers (`flow_id`, `timestamp`, `device_id`, IP addresses) and target labels (`true_label`, `label`, `type`) are strictly excluded from the feature matrix \( X \).
3. **Evaluation Metadata Isolation:** Ground-truth labels and detector correctness flags are strictly sequestered inside `evaluation_metadata`, ensuring that downstream retrieval and LLM prompt generation components remain free of ground-truth leakage.

---

## 4. Limitations & Truthful Reporting

- **Local Scope:** SHAP attributions quantify the local contribution of features to an individual flow decision; they should not be extrapolated as global feature importance across arbitrary traffic distributions.
- **Reference Data Boundary:** Attributions generated in this run reflect the `Reference-Synthetic-Network` dataset. When the frozen ACI IoT Network Dataset 2023 is attached, the identical protocol will execute without architectural modification.
