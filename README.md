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

| Outer Train Window | Test Season | Matches | Correct | Accuracy | Balanced Acc | ROC-AUC | Log Loss | Brier Score | ELO Baseline | Stronger Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2008–2015** | 2016 | 60 | 26 | 43.3% | 41.7% | 0.3806 | 0.7437 | 0.2742 | 50.0% | 46.7% |
| **2008–2016** | 2017 | 58 | 32 | 55.2% | 50.0% | 0.4573 | 0.6995 | 0.2529 | 51.7% | 55.2% |
| **2008–2017** | 2018 | 60 | 28 | 46.7% | 49.1% | 0.4364 | 0.7378 | 0.2706 | 43.3% | 46.7% |
| **2008–2018** | 2019 | 57 | 22 | 38.6% | 50.0% | 0.5110 | 0.7027 | 0.2551 | 45.6% | 38.6% |
| **2008–2019** | 2020 | 56 | 31 | 55.4% | 55.9% | 0.6073 | 0.6800 | 0.2437 | 53.6% | 48.2% |
| **2008–2020** | 2021 | 59 | 39 | **66.1%** | 61.9% | 0.6155 | 0.6765 | 0.2416 | 54.2% | 37.3% |
| **2008–2021** | 2022 | 74 | 30 | 40.5% | 40.5% | 0.4467 | 0.7180 | 0.2627 | 51.4% | 50.0% |
| **2008–2022** | 2023 | 73 | 30 | 41.1% | 43.1% | 0.4511 | 0.7309 | 0.2686 | 52.1% | 54.8% |
| **2008–2023** | 2024 | 71 | 39 | **54.9%** | 55.0% | 0.5012 | 0.7051 | 0.2557 | 47.9% | 49.3% |
| **2008–2024** | 2025 | 70 | 43 | **61.4%** | 62.0% | 0.6282 | 0.6708 | 0.2389 | 45.7% | 47.1% |
| **2008–2025** | 2026 | 6 | 3 | **50.0%** | 70.0% | 0.4000 | 0.7180 | 0.2624 | 33.3% | 16.7% |
| **TOTAL / OVERALL** | **2016–2026** | **644** | **323** | **50.2%** | **50.7%** | **0.4879** | **0.7070** | **0.2567** | **49.4%** | **47.4%** |

---

## 🎯 IPL 2026 True Holdout Blind Test

Models were trained strictly on **2008–2025 data (1,146 completed matches)** and evaluated match-by-match on the 2026 holdout season.

| Match ID | Date | Fixture | Venue | Predicted Winner (Prob) | Actual Winner | Result | T1 XI Source | T2 XI Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1527674** | 2026-03-28 | SRH vs RCB | M Chinnaswamy Stadium, Bengaluru | SRH (54.0%) | RCB | ❌ Incorrect | Match 1473505 | Match 1473511 |
| **1527675** | 2026-03-29 | KKR vs MI | Wankhede Stadium, Mumbai | KKR (54.0%) | MI | ❌ Incorrect | Match 1473505 | Match 1473510 |
| **1527676** | 2026-03-30 | CSK vs RR | Barsapara Cricket Stadium, Guwahati | RR (51.3%) | RR | ✅ **Correct** | Match 1473504 | Match 1473500 |
| **1527677** | 2026-03-31 | GT vs PBKS | Maharaja Yadavindra Singh Stadium, Mullanpur | PBKS (51.3%) | PBKS | ✅ **Correct** | Match 1473509 | Match 1473511 |
| **1527678** | 2026-04-01 | LSG vs DC | BRSABV Ekana Stadium, Lucknow | LSG (54.0%) | DC | ❌ Incorrect | Match 1473507 | Match 1485779 |
| **1527679** | 2026-04-02 | SRH vs KKR | Eden Gardens, Kolkata | SRH (52.6%) | SRH | ✅ **Correct** | Match 1527674 | Match 1527675 |

**2026 Benchmark Metrics:** Accuracy: **50.0% (3/6)** | Log Loss: **0.7180** | Brier Score: **0.2624** | ELO Baseline: **33.3%** | Stronger Team Baseline: **16.7%**

---

## 🧪 Systematic Feature Ablation Study (2020–2026)

To understand feature group contributions, sequential walk-forward evaluation was conducted across 7 incremental feature configurations:

| Config | Feature Group Added | # Feats | Accuracy | Log Loss | Brier Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | Dynamic Pre-Match ELO Only | 4 | 48.4% | 0.6939 | 0.2506 | 0.4828 |
| **B** | + Team Exponential Form & Win Rates | 10 | 52.6% | 0.6954 | 0.2511 | 0.4810 |
| **C** | + Head-to-Head Historical Dynamics | 13 | 50.1% | 0.7086 | 0.2530 | 0.5199 |
| **D** | + Venue Statistics & Chase Win Rates | 19 | 51.3% | 0.6886 | 0.2477 | 0.5168 |
| **E** | + Player Career-to-Date Batting/Bowling Ratings | 25 | **55.5%** | 0.6933 | 0.2492 | **0.6018** |
| **F** | + Bowling Phase Strengths (Powerplay, Death) | 31 | 51.1% | 0.6971 | 0.2503 | 0.5481 |
| **G** | + Lineup Synergy & Matchup Context (Full Model) | 36 | 52.6% | 0.6999 | 0.2534 | 0.5214 |

---

## 🏗️ Architecture & Model Pipeline

```
                               ┌──────────────────────────────────────────────┐
                               │       CRONOLOGICAL DATA INGESTION            │
                               │  1,175 Cricsheet matches (2008–2026)         │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │         HISTORICAL STATE TRACKER             │
                               │  Pre-match ELO · Bayesian player ratings     │
                               │  Exponential form · Venue & H2H registries   │
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
PYTHONPATH=. pytest tests/test_temporal_leakage.py -v
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
# Fast evaluation of recent seasons (2024–2026)
python fast_eval.py

# Full walk-forward backtest (2016–2026) and 7-config ablation study
python walk_forward_backtest.py --run-all --run-ablation
```

### Run Interactive Match Predictor
```bash
python ipl_predictor.py
```

### Run Feature Sensitivity & Transparency Audit
```bash
python feature_audit.py
```

---

## 📁 Repository Structure

```
├── ipl_temporal.py             # Canonical temporal data structures, state tracker & feature engine
├── ipl_models_pipeline.py      # Leak-free ensemble, expanding-window CV, calibration & metrics
├── walk_forward_backtest.py    # Walk-forward blind evaluator, ablation suite & markdown reporter
├── fast_eval.py                # Fast leak-free evaluator for recent seasons
├── feature_audit.py            # Feature sensitivity, transparency & attribution audit
├── ipl_predictor.py            # Interactive CLI, pitch modeling, score & player projections
├── ipl_stats_module.py         # Squad rosters, fallbacks & player database
├── tests/
│   └── test_temporal_leakage.py# 8 automated leakage test cases
├── reports/
│   ├── walk_forward_results.csv# Season-by-season walk-forward performance
│   ├── match_predictions.csv   # Match-by-match predictions with full audit trail
│   ├── WALK_FORWARD_REPORT.md  # Detailed walk-forward analysis & calibration curves
│   ├── 2026_BLIND_TEST.md      # IPL 2026 true holdout evaluation report
│   └── ABLATION_STUDY.md       # 7-configuration feature ablation study
├── docs/
│   └── LEAKAGE_AUDIT.md        # Complete forensic leakage audit document
├── ipl_data/cricsheet/         # 1,175 raw Cricsheet IPL match CSVs (2008–2026)
└── README.md                   # System documentation & evaluation benchmarks
```

---

## 📜 Scientific Integrity & Limitations

1. **Realistic Accuracy Bounds**: True pre-match IPL forecasting accuracy legitimately operates in the **50–65% range**. Cricket matches possess substantial inherent stochasticity (toss outcome, weather variations, dropped catches, umpire calls). Any claim of >75% pre-match accuracy in professional T20 cricket is indicative of temporal leakage.
2. **Pre-XI Uncertainty**: Pre-match predictions rely on the previous match playing XI. Sudden tactical lineup rotations or late injuries announced at toss time cannot be foreseen prior to match announcement.
3. **Score Projections**: T20 score prediction standard error is approximately $\pm 22$ runs due to boundary variance and death-over acceleration.
