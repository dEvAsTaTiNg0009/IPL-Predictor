# Leak-Free Stochastic Cricket Prediction (IPL) 🏏

> A research-grade, strictly causal match prediction framework for the Indian Premier League (IPL) trained on 18 seasons (2008–2025) of Cricsheet ball-by-ball records and evaluated via sequential walk-forward blind backtesting and holdout testing on the 2026 season.

---

## 🛡️ The Causal Temporal Data Contract

In sports predictive modeling, subtle future leakages (lookahead in rolling statistics, global ELO ratings, target-season playing XI knowledge, and full-dataset normalization) often produce artificially inflated evaluation scores. 

This repository implements a **zero-leakage temporal contract**:
> **Strict Causality Rule:** For any match occurring at timestamp $T$, absolutely no information from $T$ or after $T$ is accessible to feature engineering, normalization/scaling, ELO rating updates, Bayesian priors, model selection, probability calibration, or ensemble meta-learning.

### Prediction Modes

1. **MODE A: PRE-XI (Primary Benchmark)**
   - Prediction occurs **before** the official toss and playing XI announcement.
   - Lineups are strictly resolved from each franchise's **most recent prior match** before timestamp $T$.
   - Evaluates the true real-world forecasting task faced hours before match start.

2. **MODE B: POST-XI (Tactical Benchmark)**
   - Prediction occurs **after** the official toss and XI announcement.
   - Evaluates the model when provided the exact announced target-match playing XIs and toss decisions.

---

## 📈 Benchmark Walk-Forward Evaluation (2016–2026)

Evaluated sequentially across **644 fully blind held-out matches** spanning 11 consecutive IPL seasons without synthetic data or temporal leakage.

### Season-by-Season Blind Results (PRE-XI Mode)

| Outer Train Window | Test Season | Matches | Correct | Accuracy | Balanced Acc | ROC-AUC | Log Loss | Brier Score | ELO Baseline | Bayesian Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2008–2015** | 2016 | 60 | 26 | 43.3% | 45.3% | 0.3817 | 0.7557 | 0.2800 | 50.0% | 51.7% |
| **2008–2016** | 2017 | 58 | 32 | 55.2% | 55.0% | 0.5793 | 0.6761 | 0.2416 | 51.7% | 56.9% |
| **2008–2017** | 2018 | 60 | 28 | 46.7% | 49.5% | 0.5737 | 0.7133 | 0.2587 | 43.3% | 45.0% |
| **2008–2018** | 2019 | 57 | 28 | 49.1% | 53.5% | 0.5351 | 0.7293 | 0.2672 | 45.6% | 45.6% |
| **2008–2019** | 2020 | 56 | 28 | 50.0% | 49.9% | 0.4981 | 0.7170 | 0.2615 | 53.6% | 53.6% |
| **2008–2020** | 2021 | 59 | 33 | 55.9% | 48.3% | 0.5510 | 0.7219 | 0.2618 | 54.2% | 54.2% |
| **2008–2021** | 2022 | 74 | 38 | 51.3% | 51.3% | 0.4993 | 0.7060 | 0.2563 | 51.3% | 48.6% |
| **2008–2022** | 2023 | 73 | 37 | 50.7% | 52.6% | 0.5102 | 0.7224 | 0.2608 | 52.0% | 57.5% |
| **2008–2023** | 2024 | 71 | 37 | 52.1% | 51.9% | 0.5020 | 0.7061 | 0.2561 | 47.9% | 50.7% |
| **2008–2024** | 2025 | 70 | 38 | 54.3% | 53.5% | 0.5307 | 0.6857 | 0.2464 | 45.7% | 44.3% |
| **2008–2025** | 2026 | 6 | 3 | **50.0%** | 70.0% | 0.4000 | 0.7014 | 0.2541 | 33.3% | 50.0% |
| **TOTAL / OVERALL** | **2016–2026** | **644** | **328** | **50.9%** | **51.0%** | **0.4957** | **0.7127** | **0.2587** | **49.4%** | **50.8%** |

---

## 🎯 IPL 2026 True Holdout Blind Test

Models were trained strictly on **2008–2025 data (1,146 completed matches)** and evaluated match-by-match on the 2026 holdout season.

| Match ID | Date | Fixture | Venue | Predicted Winner (Prob) | Actual Winner | Result | T1 XI Source | T2 XI Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1527674** | 2026-03-28 | SRH vs RCB | M Chinnaswamy Stadium, Bengaluru | SRH (53.4%) | RCB | ❌ Incorrect | Match 1473505 | Match 1473511 |
| **1527675** | 2026-03-29 | KKR vs MI | Wankhede Stadium, Mumbai | KKR (53.4%) | MI | ❌ Incorrect | Match 1473505 | Match 1473510 |
| **1527676** | 2026-03-30 | CSK vs RR | Barsapara Cricket Stadium, Guwahati | RR (54.5%) | RR | ✅ **Correct** | Match 1473504 | Match 1473500 |
| **1527677** | 2026-03-31 | GT vs PBKS | Maharaja Yadavindra Singh Stadium, Mullanpur | PBKS (54.5%) | PBKS | ✅ **Correct** | Match 1473509 | Match 1473511 |
| **1527678** | 2026-04-01 | LSG vs DC | BRSABV Ekana Stadium, Lucknow | LSG (54.9%) | DC | ❌ Incorrect | Match 1473507 | Match 1485779 |
| **1527679** | 2026-04-02 | SRH vs KKR | Eden Gardens, Kolkata | SRH (51.1%) | SRH | ✅ **Correct** | Match 1527674 | Match 1527675 |

**2026 Benchmark Metrics:** Accuracy: **50.0% (3/6)** | Log Loss: **0.7014** | Brier Score: **0.2541** | Dynamic ELO Baseline: **33.3%** | Stronger Team Baseline: **16.7%**

---

## 🧪 Systematic 11-Step Feature Ablation Study (2020–2026)

To understand feature group contributions, sequential walk-forward evaluation was conducted across 11 incremental feature configurations:

| Config | Feature Group Added | # Feats | Accuracy | Log Loss | Brier Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Dynamic Pre-Match ELO Only | 4 | 50.86% | 0.6970 | 0.2515 | 0.5081 |
| **B** | + Multi-Window Team Exponential Form | 14 | 52.57% | 0.6993 | 0.2528 | 0.5247 |
| **C** | + Player Career-to-Date Strengths (Batting/Bowling) | 20 | 50.61% | 0.7050 | 0.2558 | 0.4491 |
| **D** | + Venue Statistics & Chase Win Rates | 26 | 51.10% | 0.6895 | 0.2485 | 0.5273 |
| **E** | + Head-to-Head Historical Dynamics | 29 | 50.86% | 0.6965 | 0.2516 | 0.5257 |
| **F** | + Playing XI Tactical Composition (Top/Mid/Finish/Phase) | 52 | 50.37% | 0.7056 | 0.2555 | 0.5706 |
| **G** | + Batter vs Bowler & Style Matchup Matrix | 57 | 53.30% | 0.7009 | 0.2507 | 0.5323 |
| **H** | + Player Continuity & Workload / Rest Days | 62 | 55.75% | 0.6956 | 0.2508 | 0.5178 |
| **I** | + Era & Phase Adjustments (Optimal Set) | 69 | **56.23%** | **0.6896** | **0.2483** | **0.5323** |
| **J** | Weather Ablation (Without Weather) | 71 | 50.61% | 0.7117 | 0.2567 | 0.4666 |
| **K** | Full Model (with Context Flags) | 71 | 50.61% | 0.7117 | 0.2567 | 0.4666 |

---

## 🏗️ Architecture & Model Pipeline

```
                               ┌──────────────────────────────────────────────┐
                               │       CHRONOLOGICAL DATA INGESTION           │
                               │  1,175 Cricsheet matches (2008–2026)         │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │         HISTORICAL STATE TRACKER             │
                               │  Dynamic ELO · Bayesian player ratings       │
                               │  Multi-window form · Venue & H2H registries  │
                               │  Batter vs bowling style interaction matrix  │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │         TEMPORAL FEATURE ENGINE              │
                               │  Strict pre-match cutoff (PRE-XI / POST-XI)  │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                     ┌──────────────────────────────────────────────────────────────────┐
                     │              EXPANDING-WINDOW INNER CV ENSEMBLE                  │
                     │  StandardScaler (fold-isolated)                                  │
                     │  XGBoost · LightGBM · ExtraTrees · GradientBoosting · Logistic  │
                     │  Meta-Learner: Ridge / Logistic Stacking                         │
                     │  Calibration: Out-of-fold Isotonic Regression                    │
                     └────────────────────────────────┬─────────────────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │           EVALUATION & PREDICTION            │
                               │  Calibrated Win Probabilities & Metrics      │
                               └──────────────────────────────────────────────┘
```

---

## 🔬 Automated Temporal Leakage Test Suite

Run the automated test suite to mathematically verify feature immutability and causality:

```bash
PYTHONPATH=. ./.venv/bin/python3 -m unittest tests/test_temporal_leakage.py
```

### Verified Test Cases:
- `test_a_feature_immutability_future_matches_added`: Verifies that adding future matches to the dataset produces byte-for-byte identical features for historical matches.
- `test_b_player_performance_future_modification`: Verifies that modifying a player's future match performance does not leak into prior match ratings.
- `test_c_elo_future_mutation_resistance`: Verifies that past team ELO ratings are completely invariant to future match outcomes.
- `test_d_venue_statistics_future_independence`: Verifies venue win rates and scoring averages only reflect matches completed prior to datetime $T$.
- `test_e_target_match_outcome_independence`: Verifies target match winner/margin labels are inaccessible during feature creation.
- `test_f_pre_xi_isolation`: Verifies that `PRE-XI` mode uses only lineups from matches played strictly before the target match.
- `test_g_scaler_and_pipeline_transformation_isolation`: Verifies feature scaling and probability calibration parameters are never computed across train/test boundaries.
- `test_h_red_team_synthetic_future_perturbation`: Injects synthetic anomalous matches with extreme results and verifies 100% feature invariance for all prior matches.

---

## 🚀 Quick Start & CLI Usage

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Walk-Forward Evaluation & Ablation Study
```bash
# Fast evaluation of recent seasons (2021–2026)
python fast_eval.py

# Full walk-forward backtest (2016–2026) and 11-step ablation study
python walk_forward_backtest.py --run-all --run-ablation
```

### Run Interactive Match Predictor
```bash
python ipl_predictor.py
```

### Run Automated Temporal Leakage Suite
```bash
python -m unittest tests/test_temporal_leakage.py
```

---

## 📁 Repository Structure

```
├── ipl_temporal.py             # Canonical temporal data structures, state tracker & feature engine
├── ipl_models_pipeline.py      # Leak-free ensemble, expanding-window CV, calibration & metrics
├── walk_forward_backtest.py    # Walk-forward blind evaluator, 11-step ablation suite & markdown reporter
├── fast_eval.py                # Fast leak-free evaluator for recent seasons
├── feature_audit.py            # Feature sensitivity, transparency & attribution audit
├── ipl_predictor.py            # Interactive CLI, pitch modeling, score & player projections
├── ipl_stats_module.py         # Squad rosters, fallbacks & player database
├── tests/
│   └── test_temporal_leakage.py# 8 automated leakage test cases
├── reports/
│   ├── WALK_FORWARD_RESULTS.csv# Season-by-season walk-forward performance
│   ├── match_predictions.csv   # Match-by-match predictions with full audit trail
│   ├── WALK_FORWARD_REPORT.md  # Detailed walk-forward analysis & calibration curves
│   ├── 2026_BLIND_TEST.md      # IPL 2026 true holdout evaluation report
│   ├── ABLATION_RESULTS.csv    # 11-configuration feature ablation study results
│   ├── ABLATION_STUDY.md       # Ablation markdown report
│   └── FEATURE_IMPORTANCE.csv  # Normalized tree & ensemble feature importances
├── docs/
│   ├── FINAL_LEAKAGE_AUDIT.md  # Complete second-stage forensic audit
│   └── LEAKAGE_AUDIT.md        # Initial forensic leakage audit document
├── ipl_data/cricsheet/         # 1,175 raw Cricsheet IPL match CSVs (2008–2026)
└── README.md                   # System documentation & evaluation benchmarks
```

---

## 📜 Scientific Integrity & Limitations

1. **Realistic Accuracy Bounds**: True pre-match IPL forecasting accuracy legitimately operates in the **50–58% range**. Cricket matches possess substantial inherent stochasticity (toss outcome, weather variations, dropped catches, umpire calls). Any claim of >75% pre-match accuracy in professional T20 cricket is indicative of temporal leakage.
2. **Pre-XI Uncertainty**: Pre-match predictions rely on the previous match playing XI. Sudden tactical lineup rotations or late injuries announced at toss time cannot be foreseen prior to match announcement.
3. **Score Projections**: T20 score prediction standard error is approximately $\pm 22$ runs due to boundary variance and death-over acceleration.
