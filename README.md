# Stochastic Cricket Prediction (IPL) 🏏

> A research-grade, strictly causal match prediction framework for the Indian Premier League (IPL) trained on 18 seasons (2008–2025) of Cricsheet ball-by-ball records and evaluated via sequential walk-forward blind backtesting and holdout testing on the 2026 season.

---

## 📌 Overview

This project investigates the limits of pre-match probabilistic forecasting in professional T20 cricket (IPL). By enforcing a strict chronological data contract, dynamic estimation of training priors, family-level feature regularization, and Elastic Net meta-learning, the framework produces scientifically defensible, leak-free predictions without artificial accuracy inflation.

---

## 🛡️ Core Research Problem & Key Design Principle

In sports modeling, subtle lookahead leakages—such as global ELO calculations, full-dataset normalization, retrospective playing XI knowledge, or tuning models on future holdouts—routinely produce artificially inflated evaluation metrics.

### The Causal Data Contract
> **Fundamental Causality Rule:** For any target match $M$ occurring at timestamp $T$, the available information set is:
> $$\mathcal{I}(M_T) = \{d \in \mathcal{D} \mid \text{timestamp}(d) < T\}$$
> No data from $t \ge T$ may influence feature engineering, dynamic priors, player ratings, venue statistics, scaling transformations, hyperparameter selection, Elastic Net weights, or probability calibration.

```
  Historical Information (t < T)
               │
               ▼
      Frozen Historical State
               │
               ▼
     Causal Feature Engine
               │
               ▼
    Regularized Base Models
               │
               ▼
   Chronological OOF Predictions
               │
               ▼
   Elastic Net Meta-Learner
               │
               ▼
    Isotonic Calibration
               │
               ▼
    Predicted Win Probability ──► Logged & Committed
                                         │
                                         ▼
                               Reveal Match Outcome (t = T)
                                         │
                                         ▼
                               Update Historical State (t > T)
```

---

## ⚔️ Benchmark Prediction Modes

| Benchmark Mode | Lineup Source | Toss Information | Use Case |
| :--- | :--- | :--- | :--- |
| **MODE A: PRE-XI (Primary)** | Most recent prior match played by franchise | **Excluded** | Real-world forecasting hours before match start |
| **MODE B: POST-XI (Tactical)** | Officially announced target match XI | **Included** | Post-toss tactical matchup evaluation |

*Results from PRE-XI and POST-XI modes are strictly benchmarked and reported separately.*

---

## 🧬 Explicit Feature Families

Features are organized into 7 explicit, interpretable families:

1. **`TEAM_FAMILY` (20 features):** Dynamic pre-match ELO, multi-window exponential form ($\lambda=0.25$ over 3, 5, 8 matches), historical win rates, phase run-rates (Powerplay, Death).
2. **`PLAYER_FAMILY` (6 features):** Career-to-date and recent form batting/bowling composite ratings with Bayesian shrinkage.
3. **`XI_FAMILY` (26 features):** Segmented XI strength (top order 1–3, middle order 4–6, finishing 6–8, death bowling, powerplay bowling, pace/spin strength, all-rounder depth, XI continuity, rest days).
4. **`MATCHUP_FAMILY` (8 features):** Head-to-head encounters, batter vs bowling style interaction matrix (RHB/LHB vs Pace/Spin/SLA/LBG/OB).
5. **`VENUE_FAMILY` (6 features):** Historical first innings scoring averages, chase win rates, team-at-venue win rates.
6. **`WEATHER_FAMILY` (2 features):** Temperature and humidity (ablated during feature selection to eliminate noise).
7. **`ERA_FAMILY` (1 feature):** Impact Player rule indicator (2023+).

---

## 📈 Development Walk-Forward Evaluation (2016–2025)

Evaluated sequentially across **638 fully blind held-out matches** across 10 consecutive IPL seasons using the optimal regularized feature set (PRE-XI Mode).

| Outer Train Window | Test Season | Matches | Correct | Accuracy | Balanced Acc | ROC-AUC | Log Loss | Brier Score | ELO Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2008–2015** | 2016 | 60 | 31 | 51.7% | 52.9% | 0.5067 | 0.7304 | 0.2661 | 50.0% |
| **2008–2016** | 2017 | 58 | 33 | 56.9% | 52.6% | 0.5198 | 0.6910 | 0.2489 | 51.7% |
| **2008–2017** | 2018 | 60 | 31 | 51.7% | 53.6% | 0.5954 | 0.7181 | 0.2612 | 43.3% |
| **2008–2018** | 2019 | 57 | 24 | 42.1% | 47.8% | 0.4565 | 0.7852 | 0.2940 | 45.6% |
| **2008–2019** | 2020 | 56 | 27 | 48.2% | 49.4% | 0.5192 | 0.7274 | 0.2650 | 53.6% |
| **2008–2020** | 2021 | 59 | 37 | **62.7%** | 55.5% | 0.5682 | 0.7276 | 0.2520 | 54.2% |
| **2008–2021** | 2022 | 74 | 38 | 51.4% | 51.4% | 0.5000 | 0.7023 | 0.2556 | 51.4% |
| **2008–2022** | 2023 | 73 | 40 | **54.8%** | 56.4% | 0.5432 | 0.6997 | 0.2531 | 52.1% |
| **2008–2023** | 2024 | 71 | 39 | **54.9%** | 54.7% | 0.5532 | 0.7552 | 0.2626 | 47.9% |
| **2008–2024** | 2025 | 70 | 30 | 42.9% | 45.1% | 0.4521 | 0.7086 | 0.2575 | 45.7% |
| **TOTAL / OVERALL** | **2016–2025** | **638** | **330** | **51.7%** | **51.9%** | **0.5214** | **0.7245** | **0.2616** | **49.5%** |

*Overall 95% Wilson Confidence Interval for Accuracy: [47.8%, 55.6%]*

---

## 🔬 Feature Selection & Base Model Pruning

### Elastic Net Base Model Pruning (`reports/MODEL_SELECTION.csv`)
An Elastic Net Logistic Regression meta-learner (`penalty="elasticnet", solver="saga"`) was tuned via chronological inner cross-validation. Models with near-zero coefficients were pruned:

| Base Model | Elastic Net Coefficient | Validation Contribution | Final Status |
| :--- | :---: | :---: | :---: |
| **BayesianBradleyTerry** | **+2.3829** | Strongly Positive | ✅ **RETAINED** |
| **NeuralNet (MLP)** | **+0.4775** | Positive | ✅ **RETAINED** |
| **GradientBoosting** | **+0.2524** | Positive | ✅ **RETAINED** |
| **LogisticRegression** | **-1.5535** | Calibrating / Regularizing | ✅ **RETAINED** |
| **XGBoost** | 0.0000 | Neutral / Redundant | ❌ **PRUNED** |
| **LightGBM** | 0.0000 | Neutral / Redundant | ❌ **PRUNED** |
| **ExtraTrees** | 0.0000 | Neutral / Redundant | ❌ **PRUNED** |
| **ElasticNetLogistic** | 0.0000 | Neutral / Redundant | ❌ **PRUNED** |

---

## 🧪 Systematic Feature Family Ablation Study (2020–2025)

| Configuration | # Feats | Accuracy | Log Loss | Brier Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OPTIMAL_REGULARIZED_SET (No Weather)** | **69** | **53.1%** | **0.7029** | **0.2536** | **0.5518** |
| FULL_MODEL | 71 | 52.4% | 0.7201 | 0.2576 | 0.5227 |
| WITHOUT_WEATHER | 69 | 53.1% | 0.7029 | 0.2536 | 0.5518 |
| WITHOUT_ERA | 70 | 51.9% | 0.6995 | 0.2535 | 0.5202 |
| WITHOUT_TEAM | 51 | 52.1% | 0.7492 | 0.2721 | 0.4845 |
| WITHOUT_MATCHUP | 63 | 49.6% | 0.6979 | 0.2521 | 0.5373 |
| WITHOUT_VENUE | 65 | 49.6% | 0.7062 | 0.2544 | 0.5203 |
| WITHOUT_PLAYER | 65 | 48.4% | 0.7257 | 0.2638 | 0.4998 |
| WITHOUT_XI | 43 | 46.9% | 0.7289 | 0.2649 | 0.4676 |

---

## 🎯 IPL 2026 True Blind Holdout Test

The 2026 season was **strictly isolated** during development. All features, base models, Elastic Net parameters, and scalers were frozen into `artifacts/final_2026_model/` with SHA-256 cryptographic checksums before sequentially predicting 2026 matches:

| Match ID | Date | Fixture | Venue | Predicted Winner (Prob) | Actual Winner | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **1527674** | 2026-03-28 | SRH vs RCB | M Chinnaswamy Stadium, Bengaluru | SRH (55.8%) | RCB | ❌ |
| **1527675** | 2026-03-29 | KKR vs MI | Wankhede Stadium, Mumbai | MI (52.8%) | MI | ✅ **Correct** |
| **1527676** | 2026-03-30 | CSK vs RR | Barsapara Stadium, Guwahati | RR (52.8%) | RR | ✅ **Correct** |
| **1527677** | 2026-03-31 | GT vs PBKS | Maharaja Yadavindra Singh Stadium, Mullanpur | GT (55.8%) | PBKS | ❌ |
| **1527678** | 2026-04-01 | LSG vs DC | BRSABV Ekana Stadium, Lucknow | LSG (55.8%) | DC | ❌ |
| **1527679** | 2026-04-02 | SRH vs KKR | Eden Gardens, Kolkata | KKR (52.8%) | SRH | ❌ |

**2026 Benchmark Metrics:** Accuracy: **33.3% (2/6)** | Log Loss: **0.7460** | Brier Score: **0.2763** | ROC-AUC: **0.2000**

*Note: With $N=6$ completed fixtures, the 2026 holdout has wide confidence intervals ([9.7%, 70.0%]) and should be evaluated across the full season as additional matches complete.*

---

## 🔬 Automated Red-Team Test Suite (10/10 Passing)

Verify feature immutability and temporal causality:

```bash
PYTHONPATH=. ./.venv/bin/python3 -m unittest tests/test_temporal_leakage.py
```

- `test_a_feature_immutability_adding_future_matches`: Adding future matches does not change past features.
- `test_b_future_player_performance_mutation_immunity`: Modifying future player performance leaves past ratings invariant.
- `test_c_future_match_results_do_not_alter_historical_elo`: Changing future match results does not alter past ELO.
- `test_d_future_venue_results_do_not_alter_historical_venue_stats`: Future venue outcomes do not leak into past venue stats.
- `test_e_future_h2h_results_do_not_alter_historical_h2h`: Future head-to-head fixtures leave past H2H stats invariant.
- `test_f_target_match_outcome_column_independence`: Modifying target match outcome produces identical pre-match features.
- `test_g_pre_xi_mode_does_not_access_target_playing_xi`: PRE-XI mode uses strictly prior match playing XI.
- `test_h_red_team_synthetic_future_injection`: Synthetic future matches leave prior features completely unchanged.
- `test_i_preprocessing_fitted_only_on_training_data`: Scalers and calibrators are fitted strictly on training data.
- `test_j_development_cannot_access_2026`: Development selection code strictly excludes seasons $\ge 2026$.

---

## 🚀 Quick Start & Reproduction

```bash
# Setup Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Unit Tests (10/10)
python -m unittest tests/test_temporal_leakage.py

# Run Walk-Forward Evaluation & Ablation Suite
python walk_forward_backtest.py --run-all --run-ablation
```

---

## 📁 Repository Structure

```
├── ipl_temporal.py             # Canonical chronological state tracker & feature families
├── ipl_models_pipeline.py      # Elastic Net ensemble, model pruning & calibration
├── walk_forward_backtest.py    # Walk-forward evaluator, feature stability & ablation engine
├── fast_eval.py                # Quick evaluator for recent seasons
├── tests/
│   └── test_temporal_leakage.py# 10 automated red-team leakage tests
├── artifacts/
│   └── final_2026_model/       # Frozen 2008–2025 model artifacts & SHA-256 manifest
├── reports/
│   ├── WALK_FORWARD_RESULTS.csv# Development season results (2016–2025)
│   ├── FEATURE_SELECTION.csv   # Feature stability, importance & pruning decisions
│   ├── MODEL_SELECTION.csv     # Base model pruning & Elastic Net selection status
│   ├── META_MODEL_COEFFICIENTS.csv # Elastic Net meta-learner weights
│   ├── ABLATION_RESULTS.csv    # Family-level ablation metrics
│   ├── ABLATION_STUDY.md       # Ablation analysis markdown
│   ├── MATCH_PREDICTIONS.csv   # Match audit log with provenance cutoffs
│   ├── EXPERIMENT_REGISTRY.csv # Reproducible experiment registry
│   └── 2026_BLIND_TEST.md      # 2026 holdout evaluation report
├── docs/
│   ├── LEAKAGE_AUDIT_FINAL.md  # Forensic second-stage leakage audit
│   └── BLIND_TEST_PROTOCOL.md  # Strict blind evaluation protocol
└── README.md                   # System documentation & benchmarks
```
