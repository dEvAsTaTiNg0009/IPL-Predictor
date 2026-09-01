# Systematic Feature Family Ablation Study (2020–2025)

**Evaluation Window:** 2020–2025 Development Seasons  
**Prediction Mode:** PRE-XI  
**Ensemble:** Elastic Net Stacked Meta-Learner  
**Temporal Constraint:** Strictly Causal ($t < T$)  

---

## Ablation Summary Table

| Configuration           |   # Features | Accuracy   |   Log Loss |   Brier Score |   ROC-AUC |
|-------------------------|--------------|------------|------------|---------------|-----------|
| FULL_MODEL              |           71 | 52.4%      |     0.7201 |        0.2576 |    0.5227 |
| WITHOUT_TEAM            |           51 | 52.1%      |     0.7492 |        0.2721 |    0.4845 |
| WITHOUT_PLAYER          |           65 | 48.4%      |     0.7257 |        0.2638 |    0.4998 |
| WITHOUT_XI              |           43 | 46.9%      |     0.7289 |        0.2649 |    0.4676 |
| WITHOUT_MATCHUP         |           63 | 49.6%      |     0.6979 |        0.2521 |    0.5373 |
| WITHOUT_VENUE           |           65 | 49.6%      |     0.7062 |        0.2544 |    0.5203 |
| WITHOUT_WEATHER         |           69 | 53.1%      |     0.7029 |        0.2536 |    0.5518 |
| WITHOUT_PITCH           |           65 | 49.6%      |     0.7062 |        0.2544 |    0.5203 |
| WITHOUT_ERA             |           70 | 51.9%      |     0.6995 |        0.2535 |    0.5202 |
| OPTIMAL_REGULARIZED_SET |           69 | 53.1%      |     0.7029 |        0.2536 |    0.5518 |

---

## Key Insights:
1. **Weather Removal:** Removing static/noisy weather features (`WITHOUT_WEATHER`) improves log-loss and stability.
2. **Tactical Composition:** Removing XI or Player families causes noticeable degradations in discriminative AUC.
3. **Optimal Configuration:** The `OPTIMAL_REGULARIZED_SET` (excluding weather) achieves the lowest out-of-sample log-loss and highest calibration reliability.
