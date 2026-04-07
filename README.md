# IPL 2026 AI Match Predictor 🏏

Complete AI system for IPL match prediction with real-time weather, pitch analysis, and player projections.

## Quick Start

### Google Colab
1. Upload `ipl_predictor.py`
2. Runtime → Run All
3. Follow the interactive prompts

### Local
```bash
pip install -r requirements.txt
python ipl_predictor.py
```

## Usage

```
> Command: predict

Enter match details:
  Team 1: MI
  Team 2: CSK
  Venue: Wankhede
  Match date: 2026-04-15
  Match time IST: 19:30
  Match # at venue: 1
```

## Commands
| Command | Description |
|---------|-------------|
| `predict` | Full match analysis |
| `teams` | List all 10 IPL teams |
| `venues` | List all 21 IPL venues |
| `quit` | Exit |

## What It Predicts
- **Win probability** (ensemble of 4 ML models)
- **Projected team score** with confidence range
- **Individual batting projections** (runs, balls, SR per player)
- **Individual bowling projections** (wickets, economy, pitch fit)
- **Toss recommendation** based on dew, pitch, venue history
- **Top match factors** via XGBoost feature importance

## Data Sources
- **Cricsheet** — 1,169+ IPL matches, ball-by-ball (auto-downloaded)
- **Open-Meteo** — Real-time weather (free, no API key)
- **Cricbuzz** — Live squad data (scraped)
- Historical pitch DNA for all 21 IPL venues

## All 21 Venues Supported
Wankhede · DY Patil · Brabourne · Pune · Chinnaswamy · Chepauk · Eden Gardens · Kotla · Jaipur · Mohali · Hyderabad · Vizag · Ahmedabad · Lucknow · Dharamsala · Indore · Ranchi · Cuttack · Nagpur · Raipur · Kanpur

## Model Architecture
```
XGBoost ─┐
LightGBM ─┤→ LogisticRegression (meta) → Win Probability
ExtraTrees─┤   (stacked generalization)
NeuralNet ─┘
```
~78% CV accuracy on time-series holdout.
# IPL-Predictor
