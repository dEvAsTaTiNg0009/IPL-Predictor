# IPL 2026 True Holdout Blind Test Report 🏏

**Training Period:** 2008–2025 (1,169 matches)  
**Blind Test Season:** 2026 (Held-out season)  
**Prediction Mode:** `PRE_XI` (Prior Match Playing XI)  
**Evaluation Standard:** Strictly Causal, Sequential Step Simulation  

---

## 2026 Blind Benchmark Metrics

- **Number of Matches Evaluated:** **6**
- **Correct Predictions:** **3**
- **Incorrect Predictions:** **3**
- **Accuracy:** **50.0%**
- **Log Loss:** **0.7180**
- **Brier Score:** **0.2624**
- **ROC-AUC:** **0.4000**

---

## Match-by-Match Audit Log

|   Match ID | Date       | Fixture    | Venue                                                                 | Predicted Winner (Prob)   | Actual Winner   | Correct?   |   T1 XI Source |   T2 XI Source |
|------------|------------|------------|-----------------------------------------------------------------------|---------------------------|-----------------|------------|----------------|----------------|
|    1527674 | 2026-03-28 | SRH vs RCB | M Chinnaswamy Stadium, Bengaluru                                      | SRH (54.0% T1)            | RCB             | NO         |        1473505 |        1473511 |
|    1527675 | 2026-03-29 | KKR vs MI  | Wankhede Stadium, Mumbai                                              | KKR (54.0% T1)            | MI              | NO         |        1473505 |        1473510 |
|    1527676 | 2026-03-30 | CSK vs RR  | Barsapara Cricket Stadium, Guwahati                                   | RR (48.7% T1)             | RR              | YES        |        1473504 |        1473500 |
|    1527677 | 2026-03-31 | GT vs PBKS | Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur    | PBKS (48.7% T1)           | PBKS            | YES        |        1473509 |        1473511 |
|    1527678 | 2026-04-01 | LSG vs DC  | Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow | LSG (54.0% T1)            | DC              | NO         |        1473507 |        1485779 |
|    1527679 | 2026-04-02 | SRH vs KKR | Eden Gardens, Kolkata                                                 | SRH (52.6% T1)            | SRH             | YES        |        1527674 |        1527675 |

---

## Methodology & Verification

1. The model was trained strictly on 2008–2025 data. Zero 2026 match outcomes, player statistics, or venue results were visible during training, hyperparameter selection, scaling, or calibration.
2. For each 2026 match, prediction probabilities were logged *before* the match outcome was revealed to the historical state tracker.
3. PRE-XI mode ensured that team line-ups were derived from the most recent prior match played by that franchise.
