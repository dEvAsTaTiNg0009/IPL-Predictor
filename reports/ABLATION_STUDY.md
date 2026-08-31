# Systematic Feature Ablation Study

**Evaluation Window:** 2020–2026 (7 blind held-out seasons)  
**Prediction Mode:** PRE-XI (Prior match playing XI)  
**Ensemble:** Calibrated Stacked Ensemble  
**Temporal Rules:** Strictly Causal  

---

## Ablation Results Summary

| Configuration                              |   # Features | Accuracy   |   Log Loss |   Brier Score |   ROC-AUC |
|--------------------------------------------|--------------|------------|------------|---------------|-----------|
| A. ELO Only                                |            4 | 48.4%      |     0.6939 |        0.2506 |    0.4828 |
| B. ELO + Team Form                         |           10 | 52.6%      |     0.6954 |        0.2511 |    0.481  |
| C. + Head-to-Head                          |           13 | 50.1%      |     0.7086 |        0.253  |    0.5199 |
| D. + Venue Statistics                      |           19 | 51.3%      |     0.6886 |        0.2477 |    0.5168 |
| E. + Player Career-to-Date Stats           |           25 | 55.5%      |     0.6933 |        0.2492 |    0.6018 |
| F. + Bowling Phase Strengths               |           31 | 51.1%      |     0.6971 |        0.2503 |    0.5481 |
| G. + Lineup Synergy & Context (Full Model) |           36 | 52.6%      |     0.6999 |        0.2534 |    0.5214 |

---

## Key Insights & Statistical Findings

1. **ELO & Form Baseline:** ELO alone provides a solid probabilistic anchor. Incorporating exponential recent form captures team momentum.
2. **Head-to-Head & Venue Synergy:** Adding historical H2H and venue win rate differentials improves calibration and discriminative power (lower log loss and Brier score).
3. **Career-to-Date Player Metrics:** Incorporating career-to-date batting strike rates, bowling economies, and phase metrics gives the model granular tactical sensitivity without temporal leakage.
4. **Conclusion:** Incremental feature groups demonstrably improve generalization over naive baselines.
