# Walk-Forward Blind Evaluation Report (2016–2026)

**Prediction Mode:** `PRE_XI` (Playing XI derived strictly from most recent prior match)  
**Temporal Integrity:** Strict Causality ($t < T$); No future match outcomes, player stats, or ELO ratings accessible.  
**Synthetic Data:** Disabled (`use_synthetic=False`)  
**Calibration:** Isotonic Regression on inner expanding-window CV  

---

## Executive Summary

Across all blind test seasons from **2016 to 2026** (encompassing 644 fully held-out matches):
- **Overall Blind Accuracy:** **50.9%** (328/644 matches) [95% CI: [47.1%, 54.8%]]
- **Overall Balanced Accuracy:** **51.0%**
- **Overall ROC-AUC:** **0.4957**
- **Overall Log Loss:** **0.7127**
- **Overall Brier Score:** **0.2587**
- **Expected Calibration Error (ECE):** **0.0647**

---

## Season-by-Season Blind Walk-Forward Results

| Train Window   |   Test Year |   Matches |   Correct | Accuracy   | Bal. Acc   |   ROC-AUC |   Log Loss |   Brier | ELO Base   | Bayes Base   |
|----------------|-------------|-----------|-----------|------------|------------|-----------|------------|---------|------------|--------------|
| 2007–2015      |        2016 |        60 |        26 | 43.3%      | 45.3%      |    0.3817 |     0.7557 |  0.28   | 50.0%      | 51.7%        |
| 2007–2016      |        2017 |        58 |        32 | 55.2%      | 55.0%      |    0.5793 |     0.6761 |  0.2416 | 51.7%      | 56.9%        |
| 2007–2017      |        2018 |        60 |        28 | 46.7%      | 49.5%      |    0.5737 |     0.7133 |  0.2587 | 43.3%      | 45.0%        |
| 2007–2018      |        2019 |        57 |        28 | 49.1%      | 53.5%      |    0.5351 |     0.7293 |  0.2672 | 45.6%      | 45.6%        |
| 2007–2019      |        2020 |        56 |        28 | 50.0%      | 49.9%      |    0.4981 |     0.717  |  0.2615 | 53.6%      | 53.6%        |
| 2007–2020      |        2021 |        59 |        33 | 55.9%      | 48.3%      |    0.551  |     0.7219 |  0.2618 | 54.2%      | 54.2%        |
| 2007–2021      |        2022 |        74 |        38 | 51.3%      | 51.3%      |    0.4993 |     0.706  |  0.2563 | 51.3%      | 48.6%        |
| 2007–2022      |        2023 |        73 |        37 | 50.7%      | 52.6%      |    0.5102 |     0.7224 |  0.2608 | 52.0%      | 57.5%        |
| 2007–2023      |        2024 |        71 |        37 | 52.1%      | 51.9%      |    0.502  |     0.7061 |  0.2561 | 47.9%      | 50.7%        |
| 2007–2024      |        2025 |        70 |        38 | 54.3%      | 53.5%      |    0.5307 |     0.6857 |  0.2464 | 45.7%      | 44.3%        |
| 2007–2025      |        2026 |         6 |         3 | 50.0%      | 70.0%      |    0.4    |     0.7014 |  0.2541 | 33.3%      | 50.0%        |

---

## Reliability & Probability Calibration

| Probability Range | Sample Count | Mean Predicted Prob | Empirical Win Rate | Calibration Gap |
|---|---|---|---|---|
| 0.0-0.1 | 1 | 0.091 | 1.000 | 0.909 |
| 0.1-0.2 | 1 | 0.143 | 1.000 | 0.857 |
| 0.3-0.4 | 33 | 0.368 | 0.576 | 0.207 |
| 0.4-0.5 | 274 | 0.466 | 0.445 | 0.021 |
| 0.5-0.6 | 218 | 0.529 | 0.472 | 0.057 |
| 0.6-0.7 | 112 | 0.627 | 0.527 | 0.100 |
| 0.7-0.8 | 4 | 0.724 | 0.000 | 0.724 |
| 0.8-0.9 | 1 | 0.800 | 0.000 | 0.800 |

---

## Comparison Against Established Baselines

| Model Architecture | Overall Accuracy | Log Loss | Brier Score | ROC-AUC |
|---|---|---|---|---|
| **Calibrated Stacked Ensemble (Ours)** | **50.9%** | **0.7127** | **0.2587** | **0.4957** |
| Dynamic Bayesian Bradley-Terry Model | 50.8% | 0.6931 | 0.2500 | 0.5000 |
| Dynamic ELO-Only Baseline | 49.4% | 0.6720 | 0.2395 | 0.5840 |
| Stronger Historical Team Baseline | 53.3% | 0.6931 | 0.2500 | 0.5210 |
| Random 50-50 Baseline | 50.0% | 0.6931 | 0.2500 | 0.5000 |

---

## Audit Trail & Reproducibility Verification

Every single match prediction recorded in `reports/match_predictions.csv` contains explicit audit cutoffs confirming:
1. All player career statistics, ELO ratings, and venue win rates were computed prior to match datetime.
2. In `PRE-XI` mode, playing XIs were retrieved from each franchise's previous match.
3. No test season match outcome was revealed to the model until after the prediction was committed.
