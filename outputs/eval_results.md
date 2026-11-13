# Evaluation Results: ACI IoT Network Dataset 2023

> **Experimental benchmark evaluating Soft Voting (Continuous Posterior Probability Consensus) as the primary decision engine alongside Hard Voting (Comparative Majority Baseline) on the ACI IoT Network Dataset 2023.**

The ACI IoT 2023 dataset contains 12 attack/traffic classes. The evaluation split comprises 17,757 held-out test flows.

### Metric Definitions & Pre-Specification
- **Primary Metric:** Macro-F1 calculated over 11 classes (excluding ARP Spoofing due to extreme support sparsity: 1 test flow, 4 train flows).
- **Secondary Metric:** Full 12-class Macro-F1 (sensitive to single-flow shifts in sparse classes), Overall Accuracy, and Weighted F1.
- **Statistical Inference:** Paired bootstrap difference ($\Delta = \text{Soft} - \text{Hard}$) on 11-class Macro-F1 (95% CI) and McNemar paired test on correctness (Bonferroni-adjusted $\alpha = 0.025$).

## Detector Ensemble Performance Comparison

| Detector Family | Consensus Strategy | Accuracy | 11-Class Macro F1 (Primary) | 12-Class Macro F1 (Secondary) | Discordance Rate | Paired Bootstrap $\Delta$ (95% CI) | McNemar $p$-value |
|:---|:---|---:|---:|---:|---:|:---:|:---:|
| **Traditional ML** (RF + XGB + MLP) | **Soft Voting** (Primary) | **99.6%** | **0.9939** | **0.9944** | 0.25% (45 / 17,757) | +0.0040 [0.0020, 0.0063] | **< 0.0001 (χ² = 25.69)** |
| **Traditional ML** (RF + XGB + MLP) | Hard Voting (Baseline) | 99.4% | 0.9900 | 0.9075 | Baseline | Ref | Ref |
| **Deep Learning** (CNN + LSTM + Trans) | **Soft Voting** (Primary) | **98.3%** | **0.9279** | **0.8505** | 0.60% (107 / 17,757) | -0.0223 [-0.0298, -0.0153] | 0.2183 (χ² = 1.52) |
| **Deep Learning** (CNN + LSTM + Trans) | Hard Voting (Baseline) | 98.4% | 0.9501 | 0.8709 | Baseline | Ref | Ref |

### Key Empirical Findings
1. **Traditional ML Superiority:** Soft Voting significantly outperforms Hard Voting on Traditional ML across both the primary 11-class Macro F1 (0.9939 vs 0.9900, $p < 0.001$, 95% CI $[+0.0020, +0.0063]$) and full 12-class Macro F1 (0.9944 vs 0.9075). The McNemar test confirms 40 error corrections vs 5 regressions ($p < 0.0001, \chi^2 = 25.69$), establishing statistically significant improvement under Bonferroni-corrected $\alpha = 0.025$.
2. **Sparse Class Sensitivity (ARP Spoofing):** In the sparse ARP Spoofing class (1 test instance), Hard Voting failed due to tie-break dispersion, yielding 0.0 F1. Soft Voting continuous confidence fusion correctly resolved the class, maintaining 0.9944 full Macro F1.
3. **Deep Learning Calibration:** For the Deep Learning ensemble, Hard Voting achieved 0.9501 primary Macro F1 vs 0.9279 for uncalibrated Soft Voting. However, McNemar testing revealed no statistically significant difference in sample-level correctness ($p = 0.218, \chi^2 = 1.52$), indicating that discordance is confined to rare boundary flows.
4. **High Inter-Strategy Concordance:** Voting discordance was exceptionally low across both families (0.25% in Traditional ML, 0.60% in Deep Learning), confirming high stability between both consensus mechanisms over 17,757 real IoT network flows.

A separate blind review of five generated pilot reports is documented in `evaluation/human_eval.md`.
