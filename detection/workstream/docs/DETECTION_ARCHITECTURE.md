# Project CLARITY — Detection Architecture Reference (KOU-02 & KOU-03)

**Document Reference:** `DETECTION_ARCHITECTURE.md`  
**Governing Specification:** `protocol/protocol_v1.md`  
**Target Contract:** `contracts/alert_schema_v1.json`  

---

## 1. System Topology & Dual-Ensemble Paradigm

To evaluate whether detector capacity impacts downstream LLM reliability in RAG incident analysis, Project CLARITY implements two competing detector families across the exact same network flows:

```
                            ┌──────────────────────────────────────────────┐
                            │      Cleaned Network Flow Vector (75 Feats)  │
                            └──────────────────────┬───────────────────────┘
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
            [ Traditional ML Baseline ]                           [ DL Ensemble Family ]
         ┌───────────────────────────────┐                   ┌───────────────────────────────┐
         │ - Random Forest (100 trees)   │                   │ - 1D-CNN (Temporal Conv)      │
         │ - XGBoost (Grad Boosted Tree) │                   │ - Bi-LSTM (2-layer Bidirect)  │
         │ - MLP (128x64 Dense ReLu)     │                   │ - Transformer Encoder (4-Head)│
         └───────────────┬───────────────┘                   └───────────────┬───────────────┘
                         │                                                   │
                         ▼                                                   ▼
              [ 2-of-3 Hard Voting ]                              [ 2-of-3 Hard Voting ]
                         │                                                   │
                         ▼                                                   ▼
              [ TreeExplainer (XGB) ]                             [ GradExplainer (Trans) ]
                         │                                                   │
                         └─────────────────────────┬─────────────────────────┘
                                                   ▼
                                    [ Schema v1 JSON Alert Payload ]
```

---

## 2. Traditional ML Ensemble (`traditional_ml`) — KOU-02 / Issue #8

### Base Learners
1. **Random Forest (`RandomForestClassifier`):**
   - Trees: `n_estimators=100`
   - Max Depth: `12`
   - Splitting Criterion: Gini impurity
   - Parallelism: `n_jobs=-1`
2. **XGBoost (`XGBClassifier`):**
   - Estimators: `n_estimators=100`
   - Learning Rate: `0.1`
   - Max Depth: `6`
   - Objective: `multi:softprob` (`mlogloss`)
3. **Multilayer Perceptron (`MLPClassifier`):**
   - Hidden Layers: `(128, 64)`
   - Activation: `relu`
   - Optimizer: Adam (`max_iter=60`, early stopping enabled)

### Consensus & Voting Mechanism
- **Rule:** Unweighted 2-of-3 majority vote across the discrete predictions $\{v_{\text{RF}}, v_{\text{XGB}}, v_{\text{MLP}}\}$.
- **Tie-Breaking:** If a 3-way tie occurs, the winning prediction is resolved in favor of the learner with highest calibrated posterior probability $\max_m \hat{P}_m(y = v_m \mid x)$.

---

## 3. Deep Learning Ensemble (`dl_ensemble`) — KOU-03 / Issue #9

### Base Learners
1. **1D-CNN (`CNN1D`):**
   - Convolutional layers with kernel size 3, batch normalization, ReLU activation, adaptive pooling, and dropout (0.2).
   - Designed to detect localized flow feature correlations.
2. **Long Short-Term Memory (`LSTMModel`):**
   - 2-layer standard unidirectional LSTM network with hidden dimension 32 and dropout (0.1).
   - Captures sequential packet and inter-arrival dynamics across ordered flow metrics.
3. **Transformer Encoder (`TransformerEnc`):**
   - Linear feature projection to dimension $d_{\text{model}} = 32$, 4-head multi-head self-attention module, feedforward dimension 64, and dropout (0.1).
   - Identifies non-linear cross-feature interactions without positional decay.

### Consensus & Voting Mechanism
- **Rule:** Unweighted 2-of-3 hard voting consensus across $\{v_{\text{CNN}}, v_{\text{BiLSTM}}, v_{\text{Trans}}\}$.

---

## 4. Explainability & SHAP Attribution Protocol

- **Traditional Attribution:** Evaluated via `shap.TreeExplainer` on the gradient-boosted tree learner (`XGBoost`), explaining the consensus predicted class.
- **Deep Learning Attribution:** Evaluated via `shap.GradientExplainer` on the `TransformerEncoder` member with respect to an empirical background distribution ($N=50$) sampled from the training split.
- **Contract Enforcement:**
  - Exactly 10 features returned per alert payload (`minItems: 10, maxItems: 10`).
  - Strict descending ordering by absolute magnitude: $|\phi_1| \ge |\phi_2| \ge \dots \ge |\phi_{10}|$.
  - Ranks assigned consecutively $1, 2, \dots, 10$.
  - Feature names belong strictly to the canonical 75-feature vocabulary.
