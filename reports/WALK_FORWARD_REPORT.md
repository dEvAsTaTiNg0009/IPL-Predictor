# Walk-Forward Blind Evaluation Report (2016–2026)

**Prediction Mode:** `PRE_XI` (Playing XI derived strictly from most recent prior match)  
**Temporal Integrity:** Strict Causality ($t < T$); No future match outcomes, player stats, or ELO ratings accessible.  
**Synthetic Data:** Disabled (`use_synthetic=False`)  
**Calibration:** Isotonic Regression on inner expanding-window CV  

---

## Executive Summary

Across all blind test seasons from **2016 to 2026** (encompassing 644 fully held-out matches):
- **Overall Blind Accuracy:** **50.2%** (323/644 matches)
- **Overall Balanced Accuracy:** **50.7%**
- **Overall ROC-AUC:** **0.4879**
- **Overall Log Loss:** **0.7070**
- **Overall Brier Score:** **0.2567**
- **Expected Calibration Error (ECE):** **0.0544**

---

## Season-by-Season Blind Walk-Forward Results

| Train Window   |   Test Year |   Matches |   Correct | Accuracy   | Bal. Acc   |   ROC-AUC |   Log Loss |   Brier | ELO Base   |
|----------------|-------------|-----------|-----------|------------|------------|-----------|------------|---------|------------|
| 2007–2015      |        2016 |        60 |        26 | 43.3%      | 41.7%      |    0.3806 |     0.7437 |  0.2742 | 50.0%      |
| 2007–2016      |        2017 |        58 |        32 | 55.2%      | 50.0%      |    0.4573 |     0.6995 |  0.2529 | 51.7%      |
| 2007–2017      |        2018 |        60 |        28 | 46.7%      | 49.1%      |    0.4364 |     0.7378 |  0.2706 | 43.3%      |
| 2007–2018      |        2019 |        57 |        22 | 38.6%      | 50.0%      |    0.511  |     0.7027 |  0.2551 | 45.6%      |
| 2007–2019      |        2020 |        56 |        31 | 55.4%      | 55.9%      |    0.6073 |     0.68   |  0.2437 | 53.6%      |
| 2007–2020      |        2021 |        59 |        39 | 66.1%      | 61.9%      |    0.6155 |     0.6765 |  0.2416 | 54.2%      |
| 2007–2021      |        2022 |        74 |        30 | 40.5%      | 40.5%      |    0.4467 |     0.718  |  0.2627 | 51.3%      |
| 2007–2022      |        2023 |        73 |        30 | 41.1%      | 43.1%      |    0.4511 |     0.7309 |  0.2686 | 52.0%      |
| 2007–2023      |        2024 |        71 |        39 | 54.9%      | 55.0%      |    0.5012 |     0.7051 |  0.2557 | 47.9%      |
| 2007–2024      |        2025 |        70 |        43 | 61.4%      | 62.0%      |    0.6282 |     0.6708 |  0.2389 | 45.7%      |
| 2007–2025      |        2026 |         6 |         3 | 50.0%      | 70.0%      |    0.4    |     0.718  |  0.2624 | 33.3%      |

---

## Reliability & Probability Calibration

| Probability Range | Sample Count | Mean Predicted Prob | Empirical Win Rate | Calibration Gap |
|---|---|---|---|---|
| 0.4-0.5 | 258 | 0.454 | 0.465 | 0.011 |
| 0.5-0.6 | 322 | 0.537 | 0.475 | 0.061 |
| 0.6-0.7 | 41 | 0.667 | 0.463 | 0.203 |
| 0.7-0.8 | 22 | 0.731 | 0.545 | 0.186 |
| 0.9-1.0 | 1 | 0.980 | 1.000 | 0.020 |

---

## Comparison Against Established Baselines

| Model Architecture | Overall Accuracy | Log Loss | Brier Score | ROC-AUC |
|---|---|---|---|---|
| **Calibrated Stacked Ensemble (Ours)** | **50.2%** | **0.7070** | **0.2567** | **0.4879** |
| Stronger Historical Team Baseline | 47.4% | 0.6931 | 0.2500 | 0.5210 |
| ELO-Only Baseline | 49.4% | 0.6720 | 0.2395 | 0.5840 |

---

## Audit Trail & Reproducibility Verification

Every single match prediction recorded in `reports/match_predictions.csv` contains explicit audit cutoffs confirming:
1. All player career statistics, ELO ratings, and venue win rates were computed prior to match datetime.
2. In `PRE-XI` mode, playing XIs were retrieved from each franchise's previous match.
3. No test season match outcome was revealed to the model until after the prediction was committed.
