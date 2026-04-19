import csv
import numpy as np
import pandas as pd
from pathlib import Path
from ipl_predictor import FeatureEngineer, ModelTrainer, PitchPredictor, WeatherModule, FALLBACK_SQUADS, _resolve_team, TEAMS

base = Path('ipl_data/cricsheet')

# Load info
info_rows = []
for f in sorted(base.glob('*_info.csv')):
    mid = f.stem.replace('_info', '')
    with open(f, 'r', encoding='utf-8', newline='') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 3:
                continue
            info_rows.append({
                'type': row[0],
                'key': row[1],
                'value': row[2],
                'player': row[3] if len(row) > 3 else '',
                'match_id': mid,
            })
info_df = pd.DataFrame(info_rows)

# Load balls
ball_dfs = []
for f in sorted(base.glob('*.csv')):
    if '_info' in f.name:
        continue
    try:
        df = pd.read_csv(f)
        df['match_id'] = f.stem
        ball_dfs.append(df)
    except Exception:
        pass
ball_df = pd.concat(ball_dfs, ignore_index=True) if ball_dfs else pd.DataFrame()

fe = FeatureEngineer(ball_df, info_df)
trainer = ModelTrainer()
trainer.load(Path('ipl_models/ipl_ensemble.pkl'))

pitch = PitchPredictor()
def_w = WeatherModule()._default()

test_years = set(range(2021, 2026))
results = []

match_ids = info_df[info_df['key'] == 'winner']['match_id'].unique()
for mid in sorted(match_ids):
    mi = info_df[info_df['match_id'] == mid]
    winner_rows = mi[mi['key'] == 'winner']['value'].values
    team_rows = mi[mi['key'] == 'team']['value'].tolist()
    date_rows = mi[mi['key'] == 'date']['value'].values
    venue_rows = mi[mi['key'] == 'venue']['value'].values

    if len(winner_rows) < 1 or len(team_rows) < 2 or len(date_rows) < 1:
        continue

    try:
        year = int(str(date_rows[0])[:4])
    except Exception:
        continue
    if year not in test_years:
        continue

    t1 = _resolve_team(team_rows[0])
    t2 = _resolve_team(team_rows[1])
    if t1 not in TEAMS or t2 not in TEAMS:
        continue

    venue = venue_rows[0] if len(venue_rows) else 'Wankhede Stadium'
    p = pitch.predict(venue, str(date_rows[0]).replace('/', '-'), def_w)

    sq1 = FALLBACK_SQUADS.get(t1, {})
    sq2 = FALLBACK_SQUADS.get(t2, {})

    fv = fe.build(t1, t2, venue, def_w, p, {t1: sq1, t2: sq2})
    pred = trainer.predict(fv)

    prob_t1 = pred['win_prob_t1']
    actual = 1.0 if TEAMS.get(t1, t1).lower() in str(winner_rows[0]).lower() else 0.0
    results.append((prob_t1, actual))

probs = np.array([r[0] for r in results], dtype=float)
actuals = np.array([r[1] for r in results], dtype=float)
preds = (probs > 0.5).astype(float)

accuracy = float(np.mean(preds == actuals)) if len(results) else 0.0
brier = float(np.mean((probs - actuals) ** 2)) if len(results) else 0.0
eps = 1e-7
log_loss = float(-np.mean(actuals * np.log(probs + eps) + (1 - actuals) * np.log(1 - probs + eps))) if len(results) else 0.0

print({'n_matches': len(results), 'accuracy': round(accuracy, 4), 'brier': round(brier, 4), 'log_loss': round(log_loss, 4)})
