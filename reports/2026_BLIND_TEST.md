# IPL 2026 True Holdout Blind Test Report 🏏

**Training Data:** 2008–2025 (1,146 completed matches)  
**Holdout Season:** 2026 (Completely untouched during development)  
**Prediction Mode:** `PRE_XI` (Lineups from franchise's prior match)  
**Artifact Status:** Frozen & SHA-256 Verified  

---

## Benchmark Performance Metrics

- **Total Matches Evaluated:** **6**
- **Correct Predictions:** **2**
- **Incorrect Predictions:** **4**
- **Blind Accuracy:** **33.3%** (Wilson 95% CI: [9.7%, 70.0%])
- **Log Loss:** **0.7460**
- **Brier Score:** **0.2763**
- **ROC-AUC:** **0.2000**

---

## Match-by-Match Sequential Audit Log

|   Match ID | Date       | Fixture    | Venue                                                                 | Predicted Winner (Prob)   | Actual Winner   | Correct?   |   T1 XI Source |   T2 XI Source |
|------------|------------|------------|-----------------------------------------------------------------------|---------------------------|-----------------|------------|----------------|----------------|
|    1527674 | 2026-03-28 | SRH vs RCB | M Chinnaswamy Stadium, Bengaluru                                      | SRH (55.8% T1)            | RCB             | NO         |        1473505 |        1473511 |
|    1527675 | 2026-03-29 | KKR vs MI  | Wankhede Stadium, Mumbai                                              | MI (47.2% T1)             | MI              | YES        |        1473505 |        1473510 |
|    1527676 | 2026-03-30 | CSK vs RR  | Barsapara Cricket Stadium, Guwahati                                   | RR (47.2% T1)             | RR              | YES        |        1473504 |        1473500 |
|    1527677 | 2026-03-31 | GT vs PBKS | Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur    | GT (55.8% T1)             | PBKS            | NO         |        1473509 |        1473511 |
|    1527678 | 2026-04-01 | LSG vs DC  | Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow | LSG (55.8% T1)            | DC              | NO         |        1473507 |        1485779 |
|    1527679 | 2026-04-02 | SRH vs KKR | Eden Gardens, Kolkata                                                 | KKR (47.2% T1)            | SRH             | NO         |        1527674 |        1527675 |

---

## Statistical Context & Sample Size Limitation

> [!NOTE]
> The 2026 holdout season contains **6 completed fixtures** to date. With $N=6$, empirical accuracy is subject to high statistical variance. Model hyperparameters, feature selection, and Elastic Net coefficients were finalized strictly on 2008–2025 data.
