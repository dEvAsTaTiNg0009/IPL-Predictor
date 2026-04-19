# IPL 2026 AI Match Predictor 🏏

> An end-to-end cricket match prediction system trained on 17 years of IPL ball-by-ball data (2008–2025), with real-time weather integration, pitch modeling, and player-level projections.

---

## What It Does

Given two IPL teams and a venue, the system produces:

- **Win probability** for each team (ensemble of 4 ML models)
- **Projected team score** with realistic range (calibrated to IPL 2022–2025 averages: 130–220 runs)
- **Individual batting projections** — runs, balls faced, strike rate for all 11 players
- **Individual bowling projections** — wickets, economy, pitch suitability per bowler
- **Toss recommendation** based on dew risk, pitch deterioration, venue chase history
- **Match factors breakdown** — which features are actually driving the prediction
- **Pitch report** — pace/spin index, expected score, predicted type (TURNER / FAST & BOUNCY / BALANCED etc.)

---

## Quick Start


### Local
```bash
pip install -r requirements.txt
python ipl_predictor.py
```

### Commands
```
> predict   — Run full match analysis (prompts for teams/venue/date)
> teams     — List all 10 IPL teams with captains
> venues    — List all 21 IPL venues with pitch characteristics
> quit      — Exit
```

### Example
```
Team 1: MI
Team 2: CSK
Venue:  Wankhede
Date:   2026-04-15
Time:   19:30
Match # at venue: 2
```

---

## Data Sources

| Source | What it provides | Free? |
|--------|-----------------|-------|
| [Cricsheet.org](https://cricsheet.org) | 1,169 IPL matches, ball-by-ball (2008–2025) | ✅ Free, auto-downloaded |
| [Open-Meteo API](https://open-meteo.com) | Real-time hourly weather forecast per venue | ✅ Free, no API key |
| [ESPNcricinfo Statsguru](https://stats.espncricinfo.com) | Player career stats (IPL + T20I + List-A) | ✅ Free, scraped |
| [iplt20.com](https://www.iplt20.com) | Current squad rosters (live scrape) | ✅ Free, scraped |
| Cricbuzz / Howstat | Squad fallbacks when iplt20 fails | ✅ Free, scraped |

All scraped data is cached in SQLite (`ipl_data/`) and refreshes every 12–24 hours.

---

## How Each Feature Actually Works

This is not a black box. Here is exactly how each input category affects the prediction:

### Weather → Pitch → Score

```
Open-Meteo API
     │
     ├─ humidity > 78%  → pace_index += 0.6  (damp = seam movement)
     ├─ temp > 36°C     → spin_index += 0.3  (baked = turn)
     ├─ rain_prob > 50% → pace_index += 0.7  (soft pitch = swing)
     │
     └─ Pitch type determined → affects:
           • batting avg multiplier per player  (turner hurts RHB)
           • bowler economy modifier            (spin pitch drops spinner eco 10%)
           • team batting/bowling strength      (used in ML feature vector)
```

**Proof it works**: Run `feature_audit.py` → Audit D shows win probability shifting
3–8% across different weather scenarios for the same match.

### Player Stats → Team Strength → Win Probability

```
PLAYER_DB (scraped or blended from IPL + T20I + List-A)
     │
     ├─ bat_avg × 0.55 + bat_sr × 0.14 = batting score per player
     ├─ 1/bowl_eco × 7.5 × 32/bowl_avg = bowling score per bowler
     │
     ├─ Pitch modifier applied:
     │     spin > 7.5 and RHB batter → batting score × 0.90
     │     pace > 6.5 and RF bowler  → bowling score × 1.14
     │
     └─ Average across XI = t1_bat, t1_bowl (ML features)
           bat_diff  = t1_bat - t2_bat
           bowl_diff = t1_bowl - t2_bowl
```

### New Player Bootstrap (< 5 IPL seasons)

Players with limited IPL data get a Bayesian blend of formats:
- T20I stats: 0.90 batting avg discount (IPL is harder than international T20)
- Domestic T20: 0.83 discount
- List-A: 0.70 discount (big county/Ranji averages don't translate well)
- Extra 14% penalty if List-A avg > 38 but < 10 IPL innings (catches inflated stats)
- Output tagged with confidence: `HIGH / MEDIUM / LOW / VERY_LOW`

### Playing XI Selection

The system uses squad role keys directly — not a flat list:

```
squad["wk"]           → 1 wicketkeeper (mandatory)
squad["batters"]      → up to 4 specialist batters
squad["all_rounders"] → 3–4 all-rounders
squad["bowlers"]      → 3–4 specialist bowlers
```

**Constraint enforced**: every XI must have ≥ 3 bowlers and ≥ 1 wicketkeeper. If this fails, the system pulls from the next category (e.g. extra all-rounder instead of missing bowler).

Bowling projections are computed **only** for players who actually bowl — pure batters and WK-batters are excluded from bowling analysis.

---

## Model Architecture

```
                    ┌─────────────────────────────────────┐
                    │         INPUT FEATURES (33+)         │
                    │  venue · weather · pitch · player    │
                    │  stats · form · H2H · matchups       │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           ▼                       ▼                       ▼
     XGBoost                  LightGBM               ExtraTrees
    (boosted trees)         (fast gradient)        (randomised trees)
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │ out-of-fold predictions
                                   ▼
                         Logistic Meta-learner
                         (stacked generalization)
                                   │
                                   ▼
                          Win Probability %
                    (calibrated with isotonic scaling)
```

**Training**: Time-series cross-validation (train on 2008–N, validate on N+1). Never leaks future data. Supplemented with 1,200 synthetic samples when real data is sparse.

**Realistic accuracy**: 62–68% on held-out IPL matches. The remaining 32–38% is genuine randomness — dropped catches, DRS reviews, random form days. Bookmakers operate at 65–70%.

---

## Score Projection Calibration

The old model predicted ~100 runs. The new model is calibrated to IPL 2022–2025:

| Venue type | Predicted range | Real IPL range |
|---|---|---|
| Batting paradise (Wankhede, Chinnaswamy) | 182–210 | 175–220 |
| Average venue | 162–185 | 155–195 |
| Spin track (Chepauk, Nagpur) | 145–168 | 140–175 |

Architecture: **65% physics model** (venue base × team strength × powerplay × death overs × dew) + **35% individual sum** (all 11 players projected separately). Bounds: min 128, max 235.

---

## All 21 Venues Supported

| Venue | City | Pace | Spin | Avg 1st inn |
|---|---|---|---|---|
| Wankhede Stadium | Mumbai | 7.2 | 4.5 | 178 |
| M. Chinnaswamy Stadium | Bengaluru | 5.5 | 5.8 | 183 |
| MA Chidambaram Stadium | Chennai | 4.2 | 8.5 | 162 |
| Eden Gardens | Kolkata | 6.2 | 6.0 | 170 |
| Narendra Modi Stadium | Ahmedabad | 6.5 | 6.0 | 175 |
| Arun Jaitley Stadium | Delhi | 6.5 | 6.5 | 172 |
| Rajiv Gandhi Stadium | Hyderabad | 6.0 | 6.5 | 176 |
| Sawai Mansingh Stadium | Jaipur | 7.0 | 5.0 | 174 |
| BRSABV Ekana Stadium | Lucknow | 6.8 | 5.5 | 170 |
| PCA IS Bindra Stadium | Mohali | 7.5 | 4.5 | 168 |
| HPCA Stadium | Dharamsala | 7.8 | 4.0 | 162 |
| Holkar Stadium | Indore | 6.0 | 6.5 | 180 |
| + 9 more (Vizag, Pune, Ranchi, Raipur, Cuttack, Nagpur, Kanpur, DY Patil, Brabourne) | | | | |

---

## File Structure

```
ipl_predictor.py          ← Main system (run this)
ipl_stats_module.py       ← Live squad + player stats scraper
ipl_fixes.py              ← IPL 2026 squad data + score model fixes
snippet_1_squad_scraper.py   ← Live squad scraper (priority: 3 sources)
snippet_2_stats_scraper.py   ← ESPNcricinfo player stats scraper
snippet_3_score_model.py     ← Calibrated T20 score projection model
snippet_4_accuracy_features.py ← ELO, form, phase bowling, matchup matrix
snippet_5_model_accuracy.py  ← Backtesting + accuracy evaluator
feature_audit.py             ← Prove all features are working correctly
vscode_prompt_xi_fix.md      ← Paste into VS Code AI to fix the XI bug
requirements.txt
ipl_data/                 ← Auto-created: Cricsheet CSVs + SQLite caches
ipl_models/               ← Auto-created: trained model pickle
```

---

## Running the Feature Audit

To verify the system is not just returning hardcoded values:

```python
from ipl_predictor import setup_system
from feature_audit import FeatureAudit

analyzer, squads = setup_system()
audit = FeatureAudit(analyzer)
audit.run_full_audit("MI", "CSK", "Wankhede Stadium")
```

This runs 5 audits:
1. Shows every computed feature value with its live data source
2. Changes one input at a time and shows the prediction shift
3. Attribution — which features drove this specific prediction
4. Weather scenarios — same match in 4 weather conditions, different predictions
5. XI validation — confirms bowlers/batters are correctly separated

---

## Known Limitations

- **Squad data**: IPL 2026 post-auction rosters are hardcoded as fallback. Live scraping hits iplt20.com → ESPNcricinfo → Howstat in order; Cloudflare blocks these intermittently.
- **Player stats**: Career averages only. No in-season form updates unless you run the stats scraper nightly.
- **New players**: Players with < 10 IPL innings get `VERY_LOW` confidence stats — predictions involving them are less reliable.
- **Score prediction**: ±22 runs RMSE is realistic for T20. No model predicts exact scores.
- **Injuries**: No live injury feed. If a key player is ruled out after you run the prediction, it won't reflect that.

---

## Dependencies

```
pandas · numpy · scikit-learn · xgboost · lightgbm
requests · beautifulsoup4 · lxml · tqdm · joblib
scipy · tabulate · colorama
```

All installed automatically on first run via `pip`.
