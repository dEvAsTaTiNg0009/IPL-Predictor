#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║            IPL 2026 AI MATCH PREDICTOR — COMPLETE SYSTEM                 ║ 
║   17 years of ball-by-ball data (2008-2025) · 1,169+ matches             ║
║                                                                          ║
║   Features:                                                              ║
║     • Real-time weather via Open-Meteo (free, no API key needed)         ║ 
║     • Pitch condition prediction from venue history + weather            ║ 
║     • Player performance modeling (batting + bowling projections)        ║
║     • Partnership synergy analysis                                       ║
║     • Head-to-head and venue win-rate features                           ║
║     • Ensemble ML: XGBoost + LightGBM + ExtraTrees + Neural Net          ║
║     • Meta-learner stacking with calibrated probabilities                ║
║     • New-player Bayesian bootstrap from T20I data                       ║
║     • Individual score + wicket projections per player                   ║
║                                                                          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# 0. AUTO-INSTALL DEPENDENCIES
# =============================================================================
# %%
from ipl_stats_module import build_squads_and_players
FALLBACK_SQUADS, PLAYER_DB = build_squads_and_players()
import subprocess, sys

def _pip(*pkgs):
    for pkg in pkgs:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

print("📦 Installing dependencies (first run may take ~60s)...")
_pip("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm",
     "requests", "beautifulsoup4", "lxml", "tqdm", "joblib",
     "scipy", "tabulate", "colorama")
print("✅ Done\n")

# =============================================================================
# 1. IMPORTS
# =============================================================================
# %%
import warnings; warnings.filterwarnings("ignore")
import os, re, io, json, time, zipfile, pickle, math
import numpy as np
import pandas as pd
import requests
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import norm

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

import xgboost as xgb
import lightgbm as lgb

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN = Fore.GREEN; CYAN = Fore.CYAN; YELLOW = Fore.YELLOW
    RED = Fore.RED; BOLD = Style.BRIGHT; RESET = Style.RESET_ALL
except ImportError:
    GREEN = CYAN = YELLOW = RED = BOLD = RESET = ""

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
# %%
DATA_DIR   = Path("ipl_data");   DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path("ipl_models"); MODELS_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# =============================================================================
# 3. ALL IPL VENUES  (21 grounds that have hosted IPL matches)
# =============================================================================
# %%
IPL_VENUES = {
    "Wankhede Stadium": {
        "city": "Mumbai", "state": "Maharashtra",
        "lat": 18.9388, "lon": 72.8253, "capacity": 33108,
        "avg_first_innings": 178, "avg_second_innings": 164,
        "pace_index": 7.2, "spin_index": 4.5,
        "boundary_freq": 0.68, "chase_win_rate": 0.48,
        "dew_factor": 0.70,
        "aliases": ["wankhede", "mumbai wankhede", "mi home"]
    },
    "DY Patil Stadium": {
        "city": "Navi Mumbai", "state": "Maharashtra",
        "lat": 19.0476, "lon": 73.0753, "capacity": 55000,
        "avg_first_innings": 170, "avg_second_innings": 156,
        "pace_index": 6.5, "spin_index": 5.0,
        "boundary_freq": 0.62, "chase_win_rate": 0.50,
        "dew_factor": 0.65,
        "aliases": ["dy patil", "navi mumbai", "patil stadium"]
    },
    "Brabourne Stadium": {
        "city": "Mumbai", "state": "Maharashtra",
        "lat": 18.9322, "lon": 72.8264, "capacity": 20000,
        "avg_first_innings": 165, "avg_second_innings": 152,
        "pace_index": 6.0, "spin_index": 6.0,
        "boundary_freq": 0.58, "chase_win_rate": 0.52,
        "dew_factor": 0.65,
        "aliases": ["brabourne", "cci", "cricket club of india"]
    },
    "MCA Stadium Pune": {
        "city": "Pune", "state": "Maharashtra",
        "lat": 18.6471, "lon": 73.7938, "capacity": 37000,
        "avg_first_innings": 168, "avg_second_innings": 155,
        "pace_index": 6.8, "spin_index": 5.5,
        "boundary_freq": 0.60, "chase_win_rate": 0.49,
        "dew_factor": 0.45,
        "aliases": ["pune", "mca", "maharashtra cricket association", "subrata roy sahara", "gahunje"]
    },
    "M. Chinnaswamy Stadium": {
        "city": "Bengaluru", "state": "Karnataka",
        "lat": 12.9789, "lon": 77.5984, "capacity": 40000,
        "avg_first_innings": 183, "avg_second_innings": 168,
        "pace_index": 5.5, "spin_index": 5.8,
        "boundary_freq": 0.75, "chase_win_rate": 0.47,
        "dew_factor": 0.40,
        "aliases": ["chinnaswamy", "bengaluru", "bangalore", "rcb home", "ksca"]
    },
    "MA Chidambaram Stadium": {
        "city": "Chennai", "state": "Tamil Nadu",
        "lat": 13.0629, "lon": 80.2792, "capacity": 50000,
        "avg_first_innings": 162, "avg_second_innings": 146,
        "pace_index": 4.2, "spin_index": 8.5,
        "boundary_freq": 0.52, "chase_win_rate": 0.44,
        "dew_factor": 0.80,
        "aliases": ["chepauk", "chennai", "csk home", "chidambaram", "tnca"]
    },
    "Eden Gardens": {
        "city": "Kolkata", "state": "West Bengal",
        "lat": 22.5645, "lon": 88.3433, "capacity": 68000,
        "avg_first_innings": 170, "avg_second_innings": 155,
        "pace_index": 6.2, "spin_index": 6.0,
        "boundary_freq": 0.61, "chase_win_rate": 0.46,
        "dew_factor": 0.75,
        "aliases": ["eden gardens", "kolkata", "kkr home", "eden"]
    },
    "Arun Jaitley Stadium": {
        "city": "Delhi", "state": "Delhi",
        "lat": 28.6364, "lon": 77.2290, "capacity": 41842,
        "avg_first_innings": 172, "avg_second_innings": 158,
        "pace_index": 6.5, "spin_index": 6.5,
        "boundary_freq": 0.63, "chase_win_rate": 0.50,
        "dew_factor": 0.30,
        "aliases": ["kotla", "feroz shah kotla", "delhi", "dc home", "arun jaitley", "ddca"]
    },
    "Sawai Mansingh Stadium": {
        "city": "Jaipur", "state": "Rajasthan",
        "lat": 26.8921, "lon": 75.8194, "capacity": 30000,
        "avg_first_innings": 174, "avg_second_innings": 160,
        "pace_index": 7.0, "spin_index": 5.0,
        "boundary_freq": 0.64, "chase_win_rate": 0.52,
        "dew_factor": 0.20,
        "aliases": ["jaipur", "sms stadium", "sawai mansingh", "rr home", "rca"]
    },
    "PCA IS Bindra Stadium": {
        "city": "Mohali", "state": "Punjab",
        "lat": 30.6893, "lon": 76.8485, "capacity": 26950,
        "avg_first_innings": 168, "avg_second_innings": 155,
        "pace_index": 7.5, "spin_index": 4.5,
        "boundary_freq": 0.62, "chase_win_rate": 0.49,
        "dew_factor": 0.40,
        "aliases": ["mohali", "pca stadium", "chandigarh", "pbks home", "is bindra"]
    },
    "Rajiv Gandhi International Cricket Stadium": {
        "city": "Hyderabad", "state": "Telangana",
        "lat": 17.4065, "lon": 78.5479, "capacity": 55000,
        "avg_first_innings": 176, "avg_second_innings": 161,
        "pace_index": 6.0, "spin_index": 6.5,
        "boundary_freq": 0.65, "chase_win_rate": 0.50,
        "dew_factor": 0.60,
        "aliases": ["hyderabad", "srh home", "rajiv gandhi", "uppal", "hca"]
    },
    "ACA-VDCA Stadium": {
        "city": "Visakhapatnam", "state": "Andhra Pradesh",
        "lat": 17.7231, "lon": 83.3232, "capacity": 27000,
        "avg_first_innings": 165, "avg_second_innings": 151,
        "pace_index": 6.8, "spin_index": 5.2,
        "boundary_freq": 0.60, "chase_win_rate": 0.48,
        "dew_factor": 0.70,
        "aliases": ["vizag", "visakhapatnam", "vdca", "aca vdca"]
    },
    "Narendra Modi Stadium": {
        "city": "Ahmedabad", "state": "Gujarat",
        "lat": 23.0900, "lon": 72.5970, "capacity": 132000,
        "avg_first_innings": 175, "avg_second_innings": 160,
        "pace_index": 6.5, "spin_index": 6.0,
        "boundary_freq": 0.63, "chase_win_rate": 0.50,
        "dew_factor": 0.30,
        "aliases": ["ahmedabad", "motera", "narendra modi", "gt home", "sardar patel", "gca"]
    },
    "BRSABV Ekana Cricket Stadium": {
        "city": "Lucknow", "state": "Uttar Pradesh",
        "lat": 26.8631, "lon": 80.9862, "capacity": 50000,
        "avg_first_innings": 170, "avg_second_innings": 155,
        "pace_index": 6.8, "spin_index": 5.5,
        "boundary_freq": 0.61, "chase_win_rate": 0.50,
        "dew_factor": 0.35,
        "aliases": ["lucknow", "ekana", "lsg home", "brsabv", "ekana stadium"]
    },
    "HPCA Stadium Dharamsala": {
        "city": "Dharamsala", "state": "Himachal Pradesh",
        "lat": 32.2214, "lon": 76.3234, "capacity": 23000,
        "avg_first_innings": 162, "avg_second_innings": 148,
        "pace_index": 7.8, "spin_index": 4.0,
        "boundary_freq": 0.58, "chase_win_rate": 0.50,
        "dew_factor": 0.10,
        "aliases": ["dharamsala", "hpca", "himachal pradesh", "dharmshala"]
    },
    "Holkar Cricket Stadium": {
        "city": "Indore", "state": "Madhya Pradesh",
        "lat": 22.7196, "lon": 75.8577, "capacity": 30000,
        "avg_first_innings": 180, "avg_second_innings": 166,
        "pace_index": 6.0, "spin_index": 6.5,
        "boundary_freq": 0.67, "chase_win_rate": 0.51,
        "dew_factor": 0.40,
        "aliases": ["indore", "holkar", "mca indore"]
    },
    "JSCA International Stadium": {
        "city": "Ranchi", "state": "Jharkhand",
        "lat": 23.3441, "lon": 85.3096, "capacity": 40000,
        "avg_first_innings": 160, "avg_second_innings": 146,
        "pace_index": 6.5, "spin_index": 5.5,
        "boundary_freq": 0.55, "chase_win_rate": 0.48,
        "dew_factor": 0.50,
        "aliases": ["ranchi", "jsca"]
    },
    "Barabati Stadium": {
        "city": "Cuttack", "state": "Odisha",
        "lat": 20.4686, "lon": 85.8830, "capacity": 45000,
        "avg_first_innings": 163, "avg_second_innings": 149,
        "pace_index": 5.8, "spin_index": 6.5,
        "boundary_freq": 0.56, "chase_win_rate": 0.50,
        "dew_factor": 0.60,
        "aliases": ["cuttack", "barabati", "odisha"]
    },
    "Vidarbha Cricket Association Stadium": {
        "city": "Nagpur", "state": "Maharashtra",
        "lat": 21.1498, "lon": 79.0806, "capacity": 45000,
        "avg_first_innings": 165, "avg_second_innings": 151,
        "pace_index": 5.5, "spin_index": 7.0,
        "boundary_freq": 0.58, "chase_win_rate": 0.47,
        "dew_factor": 0.45,
        "aliases": ["nagpur", "vca stadium", "vidarbha"]
    },
    "Shaheed Veer Narayan Singh Stadium": {
        "city": "Raipur", "state": "Chhattisgarh",
        "lat": 21.2514, "lon": 81.6296, "capacity": 65000,
        "avg_first_innings": 160, "avg_second_innings": 146,
        "pace_index": 6.0, "spin_index": 5.5,
        "boundary_freq": 0.55, "chase_win_rate": 0.50,
        "dew_factor": 0.50,
        "aliases": ["raipur", "shaheed veer narayan"]
    },
    "Green Park Stadium": {
        "city": "Kanpur", "state": "Uttar Pradesh",
        "lat": 26.4604, "lon": 80.3219, "capacity": 32000,
        "avg_first_innings": 158, "avg_second_innings": 144,
        "pace_index": 5.5, "spin_index": 7.5,
        "boundary_freq": 0.55, "chase_win_rate": 0.46,
        "dew_factor": 0.35,
        "aliases": ["kanpur", "green park"]
    },
}

# =============================================================================
# 4. TEAMS & SQUADS
# =============================================================================
# %%
TEAMS = {
    "MI":   "Mumbai Indians",
    "CSK":  "Chennai Super Kings",
    "RCB":  "Royal Challengers Bengaluru",
    "KKR":  "Kolkata Knight Riders",
    "DC":   "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "RR":   "Rajasthan Royals",
    "SRH":  "Sunrisers Hyderabad",
    "LSG":  "Lucknow Super Giants",
    "GT":   "Gujarat Titans",
}

TEAM_ALIASES = {
    "mumbai indians": "MI", "mi": "MI", "mumbai": "MI",
    "chennai super kings": "CSK", "csk": "CSK", "chennai": "CSK",
    "royal challengers bengaluru": "RCB", "rcb": "RCB",
    "royal challengers bangalore": "RCB", "bangalore": "RCB", "bengaluru": "RCB",
    "kolkata knight riders": "KKR", "kkr": "KKR", "kolkata": "KKR",
    "delhi capitals": "DC", "dc": "DC", "delhi": "DC", "delhi daredevils": "DC",
    "punjab kings": "PBKS", "pbks": "PBKS", "punjab": "PBKS",
    "kings xi punjab": "PBKS", "kxip": "PBKS",
    "rajasthan royals": "RR", "rr": "RR", "rajasthan": "RR",
    "sunrisers hyderabad": "SRH", "srh": "SRH", "hyderabad": "SRH", "sunrisers": "SRH",
    "lucknow super giants": "LSG", "lsg": "LSG", "lucknow": "LSG",
    "gujarat titans": "GT", "gt": "GT", "gujarat": "GT",
}

from ipl_stats_module import build_squads_and_players
FALLBACK_SQUADS, PLAYER_DB = build_squads_and_players()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
# %%

def _find_venue(name):
    """Fuzzy-match a venue name against the IPL_VENUES database."""
    if not name: return None
    nl = name.lower().strip()
    # exact key match
    for k, v in IPL_VENUES.items():
        if nl == k.lower():
            return {**v, "_name": k}
    # alias match
    for k, v in IPL_VENUES.items():
        for alias in v.get("aliases", []):
            if alias in nl or nl in alias:
                return {**v, "_name": k}
    # partial key match
    for k, v in IPL_VENUES.items():
        if any(word in k.lower() for word in nl.split() if len(word) > 3):
            return {**v, "_name": k}
    return None


def _all_players(squad):
    """Return a flat list of all players from a squad dict."""
    if "all_players" in squad:
        return squad["all_players"]
    players = []
    for key in ["batters", "all_rounders", "bowlers", "wk"]:
        for p in squad.get(key, []):
            if p not in players:
                players.append(p)
    return players


def _resolve_team(name):
    """Convert any team name/alias to its abbreviation."""
    nl = name.lower().strip()
    if nl in TEAM_ALIASES:
        return TEAM_ALIASES[nl]
    for alias, abbr in TEAM_ALIASES.items():
        if alias in nl or nl in alias:
            return abbr
    return name.upper()[:4]


def _dismissal_p(bat_avg, position):
    """Probability of a very low score dismissal for the innings."""
    base = np.clip(1.0 - float(bat_avg) / 58.0, 0.12, 0.55)
    pos_adj = max(0.0, (position - 4) * 0.038)
    return min(0.72, base + pos_adj)


_BALLS = {
    1: (22, 42), 2: (18, 36), 3: (14, 28), 4: (10, 22), 5: (7, 18), 6: (5, 15),
    7: (4, 11), 8: (2, 7), 9: (1, 5), 10: (1, 4), 11: (1, 3),
}


_PHASE_SR_BOOST = {
    1: 1.12, 2: 1.10, 3: 1.00, 4: 1.02, 5: 1.05, 6: 1.15,
    7: 1.12, 8: 1.05, 9: 0.88, 10: 0.82, 11: 0.78,
}


class ELOSystem:
    BASE_ELO = 1500
    K_FACTOR = 32
    HOME_ADV = 25
    DECAY = 0.96

    def __init__(self):
        self.ratings = defaultdict(lambda: self.BASE_ELO)
        self.last_year = {}

    def expected(self, r_a, r_b):
        return 1.0 / (1 + 10 ** ((r_b - r_a) / 400))

    def update(self, team_a, team_b, winner, venue=None, year=None):
        for t in [team_a, team_b]:
            if year and self.last_year.get(t) and self.last_year[t] != year:
                self.ratings[t] = (self.ratings[t] - self.BASE_ELO) * self.DECAY + self.BASE_ELO
            if year:
                self.last_year[t] = year

        ra = self.ratings[team_a]
        rb = self.ratings[team_b]

        home_a = venue and (team_a.lower()[:3] in (venue or "").lower())
        if home_a:
            ra += self.HOME_ADV
        else:
            rb += self.HOME_ADV

        ea = self.expected(ra, rb)
        sa = 1.0 if winner == team_a else 0.0

        self.ratings[team_a] += self.K_FACTOR * (sa - ea)
        self.ratings[team_b] += self.K_FACTOR * ((1 - sa) - (1 - ea))

    def get(self, team):
        return round(self.ratings.get(team, self.BASE_ELO), 1)

    def build_from_matches(self, info_df):
        if info_df.empty:
            return
        match_ids = info_df[info_df["key"] == "winner"]["match_id"].unique()
        for mid in sorted(match_ids):
            mi = info_df[info_df["match_id"] == mid]
            winner_rows = mi[mi["key"] == "winner"]["value"].values
            team_rows = mi[mi["key"] == "team"]["value"].tolist()
            date_rows = mi[mi["key"] == "date"]["value"].values
            venue_rows = mi[mi["key"] == "venue"]["value"].values
            if len(winner_rows) < 1 or len(team_rows) < 2:
                continue
            winner = winner_rows[0]
            t1, t2 = team_rows[0], team_rows[1]
            year = int(date_rows[0][:4]) if len(date_rows) > 0 else None
            venue = venue_rows[0] if len(venue_rows) > 0 else ""
            self.update(t1, t2, winner, venue, year)


def recent_form(team, info_df, n=5):
    if info_df.empty:
        return 3, 0.60
    tf = TEAMS.get(team, team)
    try:
        team_match_ids = info_df[info_df["value"].str.contains(tf, case=False, na=False)]["match_id"].unique()
        completed = info_df[
            (info_df["key"] == "winner") &
            (info_df["match_id"].isin(team_match_ids))
        ].sort_values("match_id").tail(n)
        if len(completed) == 0:
            return 3, 0.60

        wins = 0
        form_score = 0.0
        for i, (_, row) in enumerate(completed.iterrows()):
            won = tf.lower() in row["value"].lower()
            weight = math.exp(-0.3 * (len(completed) - 1 - i))
            if won:
                wins += 1
                form_score += weight

        max_score = sum(math.exp(-0.3 * i) for i in range(len(completed)))
        normalized = round(form_score / max_score, 3) if max_score > 0 else 0.5
        return int(wins), normalized
    except Exception:
        return 3, 0.60


def toss_venue_features(team_won_toss, chose_bat, venue, info_df):
    features = {
        "toss_won": float(team_won_toss),
        "chose_bat": float(chose_bat),
        "toss_bat_venue": 0.50,
        "toss_field_venue": 0.50,
    }
    if info_df.empty:
        return features
    try:
        vword = venue.split()[0]
        venue_mids = set(info_df[
            (info_df["key"] == "venue") &
            (info_df["value"].str.contains(vword, case=False, na=False))
        ]["match_id"])
        if not venue_mids:
            return features

        toss_rows = info_df[(info_df["key"] == "toss_winner") & (info_df["match_id"].isin(venue_mids))]
        choice_rows = info_df[(info_df["key"] == "toss_decision") & (info_df["match_id"].isin(venue_mids))]
        winner_rows = info_df[(info_df["key"] == "winner") & (info_df["match_id"].isin(venue_mids))]
        if len(toss_rows) < 5:
            return features

        merged = toss_rows.merge(choice_rows, on="match_id", suffixes=("_toss", "_choice"))
        merged = merged.merge(winner_rows, on="match_id")
        merged.columns = ["mid", "type_t", "toss_winner", "type_c", "decision", "type_w", "winner"]

        bat_first = merged[merged["decision"].str.lower().str.contains("bat")]
        if len(bat_first) > 0:
            bat_wins = (bat_first["toss_winner"] == bat_first["winner"]).mean()
            features["toss_bat_venue"] = round(float(bat_wins), 3)

        field_first = merged[~merged["decision"].str.lower().str.contains("bat")]
        if len(field_first) > 0:
            field_wins = (field_first["toss_winner"] == field_first["winner"]).mean()
            features["toss_field_venue"] = round(float(field_wins), 3)
    except Exception:
        pass
    return features


def bowling_phase_strength(squad, ball_df, pitch):
    if ball_df.empty:
        return {"pp_bowl_str": 13.5, "death_bowl_str": 13.0, "mid_bowl_str": 13.5}

    players = _all_players(squad)

    def phase_eco(player, over_lo, over_hi):
        try:
            oc = "over" if "over" in ball_df.columns else "ball"
            bc = "bowler" if "bowler" in ball_df.columns else "bowling_team"
            rc = "runs_off_bat" if "runs_off_bat" in ball_df.columns else "batsman_runs"
            ec = "extras" if "extras" in ball_df.columns else "extra_runs"
            pdf = ball_df[
                (ball_df.get(bc, pd.Series()) == player) &
                (ball_df.get(oc, pd.Series()) >= over_lo) &
                (ball_df.get(oc, pd.Series()) <= over_hi)
            ]
            if len(pdf) < 12:
                return None
            runs = pdf[rc].sum() + pdf.get(ec, 0).sum()
            eco = runs / (len(pdf) / 6)
            return round(float(eco), 2)
        except Exception:
            return None

    pp_ecos, death_ecos, mid_ecos = [], [], []
    pi = pitch.get("pace_index", 5)
    si = pitch.get("spin_index", 5)

    for p in players:
        db = PLAYER_DB.get(p, {})
        eco = db.get("bowl_eco")
        if not eco:
            continue
        style = db.get("bowl_style", "")
        is_pace = style in ["RF", "RFM", "LFM", "LF", "RMF"]
        is_spin = style in ["OB", "SLA", "LBG", "LBC", "SLO"]

        pp_e = phase_eco(p, 1, 6)
        death_e = phase_eco(p, 16, 20)
        mid_e = phase_eco(p, 7, 15)

        if pp_e is None:
            pp_e = eco * (0.92 if is_pace else 1.05)
        if death_e is None:
            death_e = eco * (0.97 if is_pace else 1.08)
        if mid_e is None:
            mid_e = eco * (0.98 if is_spin else 1.02)

        if is_pace and pi > 6.5:
            pp_e *= 0.94
            death_e *= 0.95
        if is_spin and si > 6.5:
            mid_e *= 0.90
            pp_e *= 0.96

        pp_ecos.append(pp_e)
        death_ecos.append(death_e)
        mid_ecos.append(mid_e)

    def _strength(ecos, default=13.5):
        if not ecos:
            return default
        avg_eco = np.mean(sorted(ecos)[:5])
        return round(120 / max(avg_eco, 4.5), 3)

    return {
        "pp_bowl_str": _strength(pp_ecos, 13.5),
        "death_bowl_str": _strength(death_ecos, 13.0),
        "mid_bowl_str": _strength(mid_ecos, 13.5),
    }


_MATCHUP_MATRIX = {
    ("RHB", "OB"): 0.94, ("RHB", "SLA"): 1.06, ("RHB", "LBG"): 0.91,
    ("RHB", "LFM"): 1.04, ("RHB", "RF"): 0.98,
    ("LHB", "OB"): 1.05, ("LHB", "SLA"): 0.93, ("LHB", "LBG"): 1.07,
    ("LHB", "RFM"): 0.96, ("LHB", "LFM"): 0.97,
}


def matchup_score(batting_squad, bowling_squad):
    bats = _all_players(batting_squad)[:7]
    bowls = _all_players(bowling_squad)[:7]
    advantages = []
    for bat in bats:
        bat_sty = PLAYER_DB.get(bat, {}).get("bat_style", "RHB")
        for bowl in bowls:
            bowl_sty = PLAYER_DB.get(bowl, {}).get("bowl_style")
            if not bowl_sty:
                continue
            advantages.append(_MATCHUP_MATRIX.get((bat_sty, bowl_sty), 1.0))
    if not advantages:
        return 1.0
    return round(float(np.mean(advantages)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────
# %%

class IPLDataFetcher:
    CRICSHEET_URL = "https://cricsheet.org/downloads/ipl_csv2.zip"
    HEADERS = {"User-Agent": "Mozilla/5.0 (IPLPredictor/2.0)"}

    def download_cricsheet(self):
        extract_path = DATA_DIR / "cricsheet"
        if extract_path.exists() and len(list(extract_path.glob("*.csv"))) > 80:
            print(f"✅ Cricsheet cached ({len(list(extract_path.glob('*.csv')))} files)")
            return True
        print("📥 Downloading Cricsheet IPL data (~50MB)…")
        try:
            r = requests.get(self.CRICSHEET_URL, stream=True, timeout=180)
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            buf = io.BytesIO(); done = 0
            for chunk in r.iter_content(8192):
                buf.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {done/total*100:5.1f}%  ({done//1048576}MB/{total//1048576}MB)", end="")
            print()
            buf.seek(0); extract_path.mkdir(exist_ok=True)
            with zipfile.ZipFile(buf) as z: z.extractall(extract_path)
            print(f"✅ Extracted {len(list(extract_path.glob('*.csv')))} files")
            return True
        except Exception as e:
            print(f"⚠️  Cricsheet download failed: {e}"); return False

    def load_all_matches(self):
        extract_path = DATA_DIR / "cricsheet"
        if not extract_path.exists():
            return pd.DataFrame(), pd.DataFrame()
        info_files = sorted(extract_path.glob("*_info.csv"))
        ball_files  = [f for f in extract_path.glob("*.csv") if "_info" not in f.name]
        if not info_files:
            print("⚠️  No Cricsheet data — will use synthetic training data")
            return pd.DataFrame(), pd.DataFrame()
        print(f"📊 Loading {len(info_files)} match info files…")
        info_dfs = []
        for f in tqdm(info_files, desc="Info"):
            try:
                df = pd.read_csv(f, header=None, names=["type","key","value"])
                df["match_id"] = f.stem.replace("_info",""); info_dfs.append(df)
            except: pass
        print(f"📊 Loading {len(ball_files)} ball-by-ball files…")
        ball_dfs = []
        for f in tqdm(ball_files, desc="Deliveries"):
            try:
                df = pd.read_csv(f); df["match_id"] = f.stem; ball_dfs.append(df)
            except: pass
        info_df = pd.concat(info_dfs, ignore_index=True) if info_dfs else pd.DataFrame()
        ball_df = pd.concat(ball_dfs, ignore_index=True) if ball_dfs else pd.DataFrame()
        print(f"✅ {len(ball_df):,} deliveries across {len(ball_dfs)} matches")
        return info_df, ball_df

    def scrape_squads(self):
        print("🌐 Fetching current IPL squads…")
        squads = {}
        # Try Cricbuzz
        try:
            url = "https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/teams"
            r = requests.get(url, headers=self.HEADERS, timeout=12)
            soup = BeautifulSoup(r.text, "lxml")
            for card in soup.select("div.cb-team-squad-players, div[class*='squad']"):
                players = [a.get_text(strip=True) for a in card.select("a")
                           if 3 < len(a.get_text(strip=True)) < 45]
                players = [p for p in players if re.match(r"^[A-Z][a-z]", p)]
                if players:
                    hdr = card.find_previous(["h2","h3","h4"])
                    abbr = _resolve_team(hdr.get_text(strip=True) if hdr else "")
                    if abbr in TEAMS:
                        squads[abbr] = {"all_players": players[:22]}
        except Exception as e:
            print(f"  Scrape note: {e}")
        for abbr in TEAMS:
            if abbr not in squads:
                squads[abbr] = FALLBACK_SQUADS[abbr]
        live = sum(1 for v in squads.values() if "all_players" in v)
        print(f"✅ Squads ready — {live} live scraped, {len(squads)-live} from fallback")
        return squads


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER MODULE  (Open-Meteo — free, no key)
# ─────────────────────────────────────────────────────────────────────────────
# %%

class WeatherModule:
    BASE = "https://api.open-meteo.com/v1/forecast"
    HIST = "https://archive-api.open-meteo.com/v1/archive"

    def get(self, venue_name, match_date, match_time_ist="19:30"):
        venue = _find_venue(venue_name)
        if venue is None: return self._default()
        lat, lon = venue["lat"], venue["lon"]
        hour = int(match_time_ist.split(":")[0])
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relativehumidity_2m,precipitation_probability,"
                          "precipitation,windspeed_10m,winddirection_10m,cloudcover,"
                          "dewpoint_2m,apparent_temperature",
                "timezone": "Asia/Kolkata",
                "start_date": match_date, "end_date": match_date,
            }
            r = requests.get(self.BASE, params=params, timeout=15).json()
            h = r.get("hourly", {}); idx = min(hour, len(h.get("temperature_2m",[0]))-1)
            w = {
                "temp_c":      round(h["temperature_2m"][idx], 1),
                "humidity":    round(h["relativehumidity_2m"][idx], 1),
                "rain_prob":   round(h["precipitation_probability"][idx], 1),
                "precip_mm":   round(h["precipitation"][idx], 2),
                "wind_kph":    round(h["windspeed_10m"][idx], 1),
                "wind_dir":    h["winddirection_10m"][idx],
                "cloud_cover": round(h["cloudcover"][idx], 1),
                "dewpoint":    round(h["dewpoint_2m"][idx], 1),
                "feels_like":  round(h["apparent_temperature"][idx], 1),
                "is_night":    hour >= 18,
                "venue_dew_factor": venue.get("dew_factor", 0.5),
            }
            w["dew_risk"]   = self._dew_risk(w, venue)
            w["conditions"] = self._classify(w)
            w["impact"]     = self._impact(w)
            return w
        except Exception as e:
            print(f"  ⚠️  Weather API: {e} — using historical average")
            return self._historical(lat, lon, match_date, hour, venue)

    def _dew_risk(self, w, venue):
        if not w["is_night"]: return 0.1
        spread = w["temp_c"] - w["dewpoint"]
        risk = (1/(1+spread*0.15)) * (w["humidity"]/100) * venue.get("dew_factor", 0.5)
        return round(min(1.0, risk + w["cloud_cover"]/100*0.2), 2)

    def _classify(self, w):
        if w["rain_prob"] > 60:   return "RAIN LIKELY"
        if w["rain_prob"] > 30:   return "OVERCAST / HUMID"
        if w["humidity"] > 80:    return "VERY HUMID"
        if w["temp_c"] > 38:      return "EXTREME HEAT"
        if w["cloud_cover"] > 70: return "CLOUDY"
        return "CLEAR"

    def _impact(self, w):
        pts = []
        if w["dew_risk"] > 0.60:   pts.append(f"Heavy dew (risk {w['dew_risk']:.0%}) → batting 2nd advantage")
        if w["humidity"] > 75:     pts.append("High humidity → swing for pacers early")
        if w["temp_c"] > 37:       pts.append(f"Extreme heat {w['temp_c']}°C → fatigue factor")
        if w["cloud_cover"] > 70:  pts.append("Overcast → pace movement")
        if w["wind_kph"] > 22:     pts.append(f"Wind {w['wind_kph']} kph → affects slower bowlers")
        if w["rain_prob"] > 40:    pts.append(f"Rain risk {w['rain_prob']}% → DLS may apply")
        return " | ".join(pts) if pts else "Standard playing conditions"

    def _historical(self, lat, lon, date_str, hour, venue):
        date = datetime.strptime(date_str, "%Y-%m-%d")
        temps, hums = [], []
        for i in range(1, 4):
            d = date.replace(year=date.year - i)
            try:
                r = requests.get(self.HIST, params={
                    "latitude": lat, "longitude": lon,
                    "hourly": "temperature_2m,relativehumidity_2m",
                    "start_date": d.strftime("%Y-%m-%d"),
                    "end_date":   d.strftime("%Y-%m-%d"),
                    "timezone": "Asia/Kolkata"
                }, timeout=10).json()
                h = r.get("hourly", {})
                if h.get("temperature_2m"):
                    temps.append(h["temperature_2m"][hour])
                    hums.append(h["relativehumidity_2m"][hour])
            except: pass
        w = {
            "temp_c":   round(np.mean(temps) if temps else 28, 1),
            "humidity": round(np.mean(hums)  if hums  else 65, 1),
            "rain_prob": 20.0, "precip_mm": 0.0,
            "wind_kph": 12.0, "wind_dir": 180,
            "cloud_cover": 30.0, "dewpoint": 22.0, "feels_like": 30.0,
            "is_night": hour >= 18,
            "venue_dew_factor": venue.get("dew_factor", 0.5),
        }
        w["dew_risk"]   = self._dew_risk(w, venue)
        w["conditions"] = "HISTORICAL AVERAGE"
        w["impact"]     = "Based on historical weather (live data unavailable)"
        return w

    def _default(self):
        return {"temp_c":28,"humidity":65,"rain_prob":15,"precip_mm":0,"wind_kph":10,
                "wind_dir":180,"cloud_cover":25,"dewpoint":20,"feels_like":29,
                "is_night":True,"venue_dew_factor":0.5,"dew_risk":0.3,
                "conditions":"STANDARD","impact":"Standard IPL conditions"}


# ─────────────────────────────────────────────────────────────────────────────
# PITCH PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
# %%

class PitchPredictor:
    def predict(self, venue_name, match_date, weather, match_n_at_venue=1):
        venue = _find_venue(venue_name)
        if venue is None: return self._default()
        p = venue.get("pace_index", 6.0)
        s = venue.get("spin_index", 6.0)
        h = weather.get("humidity", 60)
        t = weather.get("temp_c", 28)
        rp = weather.get("rain_prob", 15)
        if h > 78: p = min(10, p + 0.6)
        if h < 40: s = min(10, s + 0.5)
        if t > 36: s = min(10, s + 0.3)
        if rp > 50: p = min(10, p + 0.7); s = max(1, s - 0.4)
        wear = min(match_n_at_venue - 1, 5) * 0.45
        s = min(10, s + wear); p = max(1, p - wear * 0.25)
        batting_idx = (p * 0.35 + (10 - s) * 0.45 + 5 * 0.20) / 10
        base_score  = venue.get("avg_first_innings", 168)
        exp_score   = int(base_score * (0.80 + batting_idx * 0.42))
        alt = {"Dharamsala": 1457, "Ranchi": 651, "Mohali": 310}
        bounce = 5.0 + (p - 5) * 0.4 + sum(v for k, v in alt.items() if k in venue_name) * 0.002
        ptype = ("FAST & BOUNCY" if p > 7 and s < 5 else
                 "TURNER"        if s > 7.5 and p < 5 else
                 "SPIN-FRIENDLY" if s > 6.5 else
                 "PACE-FRIENDLY" if p > 6.5 else
                 "FLAT BATTING"  if p < 5 and s < 5 else "BALANCED")
        spin_over = 10 if s > 7.5 else 12 if s > 6.5 else 15
        return {
            "pace_index":  round(p, 1), "spin_index": round(s, 1),
            "bounce_index": round(bounce, 1), "expected_score": exp_score,
            "pitch_type": ptype, "batting_friendly": batting_idx > 0.54,
            "wear_factor": round(wear, 2),
            "spin_advantage_from_over": spin_over,
            "description": self._describe(p, s, weather),
        }

    def _describe(self, p, s, w):
        parts = []
        if p > 7:   parts.append("Seam movement expected; pace bowlers dominant early")
        if s > 7:   parts.append("Significant turn from over 6; spinners key in middle overs")
        if w.get("humidity", 60) > 75: parts.append("Overhead conditions aiding swing")
        if w.get("dew_risk", 0) > 0.6: parts.append("Heavy dew in 2nd innings neutralises spin")
        if not parts: parts.append("Balanced surface; contest between bat and ball throughout")
        return ". ".join(parts) + "."

    def _default(self):
        return {"pace_index":6.0,"spin_index":6.0,"bounce_index":5.0,"expected_score":168,
                "pitch_type":"BALANCED","description":"Standard T20 surface.",
                "batting_friendly":True,"wear_factor":0,"spin_advantage_from_over":12}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEER
# ─────────────────────────────────────────────────────────────────────────────
# %%

class FeatureEngineer:
    def __init__(self, ball_df, info_df):
        self.ball_df = ball_df
        self.info_df = info_df
        self._cache  = {}
        self.elo = ELOSystem()
        if not info_df.empty:
            self.elo.build_from_matches(info_df)

    def build(self, t1, t2, venue, weather, pitch, squads):
        f = {}
        v = _find_venue(venue) or {}
        f["venue_avg_first"]     = v.get("avg_first_innings", 168)
        f["venue_pace_index"]    = v.get("pace_index", 6.0)
        f["venue_spin_index"]    = v.get("spin_index", 6.0)
        f["venue_boundary_freq"] = v.get("boundary_freq", 0.62)
        f["venue_chase_wr"]      = v.get("chase_win_rate", 0.50)
        f["venue_dew_factor"]    = v.get("dew_factor", 0.5)
        f["w_temp"]    = weather.get("temp_c", 28)
        f["w_humid"]   = weather.get("humidity", 65)
        f["w_rain"]    = weather.get("rain_prob", 15)
        f["w_cloud"]   = weather.get("cloud_cover", 30)
        f["w_dew"]     = weather.get("dew_risk", 0.3)
        f["w_wind"]    = weather.get("wind_kph", 10)
        f["is_night"]  = float(weather.get("is_night", True))
        f["p_pace"]    = pitch.get("pace_index", 6.0)
        f["p_spin"]    = pitch.get("spin_index", 6.0)
        f["p_bounce"]  = pitch.get("bounce_index", 5.0)
        f["p_score"]   = pitch.get("expected_score", 168)
        sq1 = squads.get(t1, FALLBACK_SQUADS.get(t1, {}))
        sq2 = squads.get(t2, FALLBACK_SQUADS.get(t2, {}))
        f["t1_bat"]  = self._bat_strength(sq1, pitch)
        f["t2_bat"]  = self._bat_strength(sq2, pitch)
        f["t1_bowl"] = self._bowl_strength(sq1, pitch)
        f["t2_bowl"] = self._bowl_strength(sq2, pitch)
        f["bat_diff"]  = f["t1_bat"]  - f["t2_bat"]
        f["bowl_diff"] = f["t1_bowl"] - f["t2_bowl"]

        f["t1_elo"] = self.elo.get(TEAMS.get(t1, t1))
        f["t2_elo"] = self.elo.get(TEAMS.get(t2, t2))
        f["elo_diff"] = f["t1_elo"] - f["t2_elo"]

        t1_wins, t1_form = recent_form(t1, self.info_df, n=5)
        t2_wins, t2_form = recent_form(t2, self.info_df, n=5)
        f["t1_wins"] = t1_wins
        f["t2_wins"] = t2_wins
        f["t1_form_score"] = t1_form
        f["t2_form_score"] = t2_form
        f["form_diff"] = t1_form - t2_form

        h2h = self._h2h(t1, t2)
        f["h2h_wr"] = h2h.get("t1_wr", 0.5); f["h2h_total"] = h2h.get("total", 10)
        f["t1_venue_wr"]  = self._venue_wr(t1, venue)
        f["t2_venue_wr"]  = self._venue_wr(t2, venue)
        f["t1_pitch_aff"] = self._pitch_affinity(sq1, pitch)
        f["t2_pitch_aff"] = self._pitch_affinity(sq2, pitch)
        f["pitch_aff_diff"] = f["t1_pitch_aff"] - f["t2_pitch_aff"]

        f.update(toss_venue_features(0.0, 0.0, venue, self.info_df))
        f.update(bowling_phase_strength(sq1, self.ball_df, pitch))
        f["t1_matchup"] = matchup_score(sq1, sq2)
        f["t2_matchup"] = matchup_score(sq2, sq1)
        f["matchup_diff"] = f["t1_matchup"] - f["t2_matchup"]
        return f

    def _bat_strength(self, squad, pitch):
        scores = []
        for p in _all_players(squad)[:11]:
            db = PLAYER_DB.get(p, {})
            avg = db.get("bat_avg", 20.0); sr = db.get("bat_sr", 125.0)
            if avg > 0 and sr > 0:
                s = avg * 0.55 + sr * 0.14
                if pitch.get("spin_index", 5) > 7.5 and db.get("bat_style") == "RHB": s *= 0.92
                scores.append(s)
        return round(np.mean(scores) if scores else 28.0, 2)

    def _bowl_strength(self, squad, pitch):
        scores = []
        for p in _all_players(squad)[:11]:
            db = PLAYER_DB.get(p, {}); eco = db.get("bowl_eco"); avg = db.get("bowl_avg")
            style = db.get("bowl_style", "")
            if eco and avg:
                s = (1/eco*7.5) * (32/max(avg, 15)) * 18
                pi = pitch.get("pace_index", 5); si = pitch.get("spin_index", 5)
                if style in ["RF","RFM","LFM","LF"] and pi > 6.5: s *= 1.12
                if style in ["OB","SLA","LBG","LBC","SLO"] and si > 6.5: s *= 1.12
                scores.append(s)
        return round(np.mean(scores) if scores else 13.0, 2)

    def _form(self, team):
        if self.info_df.empty: return 3
        try:
            tf = TEAMS.get(team, team)
            tm = self.info_df[self.info_df["value"].str.contains(tf, case=False, na=False)]["match_id"].unique()
            wm = self.info_df[(self.info_df["key"]=="winner") &
                              (self.info_df["match_id"].isin(tm)) &
                              (self.info_df["value"].str.contains(tf, case=False, na=False))]["match_id"]
            return int(len(wm.tail(5)))
        except: return 3

    def _h2h(self, t1, t2):
        if self.info_df.empty: return {"t1_wr": 0.5, "total": 10}
        try:
            tf1 = TEAMS.get(t1, t1); tf2 = TEAMS.get(t2, t2)
            m1 = set(self.info_df[self.info_df["value"].str.contains(tf1, case=False, na=False)]["match_id"])
            m2 = set(self.info_df[self.info_df["value"].str.contains(tf2, case=False, na=False)]["match_id"])
            shared = m1 & m2
            if not shared: return {"t1_wr": 0.5, "total": 0}
            wr    = self.info_df[(self.info_df["key"]=="winner") & (self.info_df["match_id"].isin(shared)) & (self.info_df["value"].str.contains(tf1, case=False, na=False))]
            total = len(self.info_df[(self.info_df["key"]=="winner") & (self.info_df["match_id"].isin(shared))])
            return {"t1_wr": round(len(wr)/max(total, 1), 3), "total": total}
        except: return {"t1_wr": 0.5, "total": 10}

    def _venue_wr(self, team, venue):
        if self.info_df.empty: return 0.5
        try:
            tf = TEAMS.get(team, team); vword = venue.split()[0]
            vm = set(self.info_df[(self.info_df["key"]=="venue") & (self.info_df["value"].str.contains(vword, case=False, na=False))]["match_id"])
            if not vm: return 0.5
            tm = set(self.info_df[self.info_df["value"].str.contains(tf, case=False, na=False)]["match_id"])
            joint = vm & tm
            if not joint: return 0.5
            wins = len(self.info_df[(self.info_df["key"]=="winner") & (self.info_df["match_id"].isin(joint)) & (self.info_df["value"].str.contains(tf, case=False, na=False))])
            return round(wins / len(joint), 3)
        except: return 0.5

    def _pitch_affinity(self, squad, pitch):
        players = _all_players(squad)
        pi = pitch.get("pace_index", 5); si = pitch.get("spin_index", 5)
        pace_n = sum(1 for p in players if PLAYER_DB.get(p, {}).get("bowl_style","") in ["RF","RFM","LFM","LF"])
        spin_n = sum(1 for p in players if PLAYER_DB.get(p, {}).get("bowl_style","") in ["OB","SLA","LBG","LBC","SLO"])
        n = max(pace_n + spin_n, 1)
        return round((pace_n * pi + spin_n * si) / n / 10, 3)

    def partnership_synergy(self, p1, p2):
        key = f"ps_{min(p1,p2)}_{max(p1,p2)}"
        if key in self._cache: return self._cache[key]
        default = {"avg_part": 28.0, "synergy": 1.0, "n": 0}
        if self.ball_df.empty:
            self._cache[key] = default; return default
        try:
            bc = "batter" if "batter" in self.ball_df.columns else "batsman"
            rc = "batsman_runs" if "batsman_runs" in self.ball_df.columns else "runs_off_bat"
            ic = "innings" if "innings" in self.ball_df.columns else "inning"
            # find matches where both appeared
            m1 = set(self.ball_df[self.ball_df[bc]==p1]["match_id"])
            m2 = set(self.ball_df[self.ball_df[bc]==p2]["match_id"])
            shared = m1 & m2
            if len(shared) < 3:
                self._cache[key] = default; return default
            together = self.ball_df[(self.ball_df["match_id"].isin(shared)) &
                                    (self.ball_df[bc].isin([p1, p2]))]
            runs_per = together.groupby(["match_id", ic])[rc].sum()
            avg_part = round(runs_per.mean(), 1)
            league_avg = 28.0
            synergy = round(avg_part / league_avg, 3)
            result = {"avg_part": avg_part, "synergy": synergy, "n": len(shared)}
            self._cache[key] = result; return result
        except:
            self._cache[key] = default; return default

    def player_venue_record(self, player, venue):
        if self.ball_df.empty or self.info_df.empty: return {"has_data": False}
        try:
            vword = venue.split()[0]
            vm = set(self.info_df[(self.info_df["key"]=="venue") &
                                  (self.info_df["value"].str.contains(vword, case=False, na=False))]["match_id"])
            if not vm: return {"has_data": False}
            bc = "batter" if "batter" in self.ball_df.columns else "batsman"
            rc = "batsman_runs" if "batsman_runs" in self.ball_df.columns else "runs_off_bat"
            ic = "innings" if "innings" in self.ball_df.columns else "inning"
            pdf = self.ball_df[(self.ball_df["match_id"].isin(vm)) & (self.ball_df[bc]==player)]
            if len(pdf) < 6: return {"has_data": len(pdf) > 0, "balls": len(pdf)}
            runs = pdf[rc].sum(); balls = len(pdf)
            grp  = pdf.groupby(["match_id", ic])[rc].sum()
            return {"has_data": True, "venue_avg": round(grp.mean(), 2),
                    "venue_sr": round(runs/balls*100, 2),
                    "venue_balls": balls, "venue_innings": len(grp)}
        except: return {"has_data": False}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINER  (Ensemble: XGBoost + LightGBM + ExtraTrees + MLP → LogReg)
# ─────────────────────────────────────────────────────────────────────────────
# %%

class ModelTrainer:
    FEATURE_NAMES = [
        "venue_avg_first","venue_pace_index","venue_spin_index","venue_boundary_freq",
        "venue_chase_wr","venue_dew_factor","w_temp","w_humid","w_rain","w_cloud",
        "w_dew","w_wind","is_night","p_pace","p_spin","p_bounce","p_score",
        "t1_bat","t2_bat","t1_bowl","t2_bowl","bat_diff","bowl_diff",
        "t1_elo","t2_elo","elo_diff",
        "t1_wins","t2_wins","t1_form_score","t2_form_score","form_diff",
        "h2h_wr","h2h_total",
        "t1_venue_wr","t2_venue_wr","t1_pitch_aff","t2_pitch_aff","pitch_aff_diff",
        "toss_won","chose_bat","toss_bat_venue","toss_field_venue",
        "pp_bowl_str","death_bowl_str","mid_bowl_str",
        "t1_matchup","t2_matchup","matchup_diff",
    ]

    def __init__(self):
        self.models  = {}
        self.scaler  = StandardScaler()
        self.trained = False
        self.cv_scores = {}

    def prepare_dataset(self, info_df, ball_df, feat_eng, squads):
        """Build training matrix from real match data + synthetic supplement."""
        rows = []; labels = []

        if not info_df.empty:
            match_ids = info_df[info_df["key"]=="winner"]["match_id"].unique()
            print(f"  Building features for {len(match_ids)} real matches…")
            for mid in tqdm(match_ids[:800], desc="Features"):
                try:
                    mi = info_df[info_df["match_id"]==mid]
                    winner = mi[mi["key"]=="winner"]["value"].values[0]
                    teams  = mi[mi["key"]=="team"]["value"].tolist()
                    if len(teams) < 2: continue
                    t1, t2 = _resolve_team(teams[0]), _resolve_team(teams[1])
                    venue_val = mi[mi["key"]=="venue"]["value"].values
                    venue = venue_val[0] if len(venue_val) else "Wankhede Stadium"
                    date_val  = mi[mi["key"]=="date"]["value"].values
                    date  = date_val[0] if len(date_val) else "2023-04-01"
                    pitch = PitchPredictor().predict(venue, str(date), WeatherModule()._default())
                    weather = WeatherModule()._default()
                    sq1 = squads.get(t1, FALLBACK_SQUADS.get(t1, FALLBACK_SQUADS["MI"]))
                    sq2 = squads.get(t2, FALLBACK_SQUADS.get(t2, FALLBACK_SQUADS["CSK"]))
                    fv = feat_eng.build(t1, t2, venue, weather, pitch, {t1: sq1, t2: sq2})
                    row = [fv.get(k, 0) for k in self.FEATURE_NAMES]
                    tf1 = TEAMS.get(t1, t1); label = 1 if tf1.lower() in winner.lower() else 0
                    rows.append(row); labels.append(label)
                except: pass

        # Synthetic supplement to ensure ≥1200 samples
        need = max(0, 1200 - len(rows))
        if need > 0:
            print(f"  Generating {need} synthetic training samples…")
            syn_rows, syn_labels = self._synthetic(need)
            rows.extend(syn_rows); labels.extend(syn_labels)

        X = np.array(rows, dtype=float)
        y = np.array(labels)
        X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        print(f"  ✅ Dataset: {len(X)} samples × {X.shape[1]} features")
        return X, y

    def _synthetic(self, n):
        rows, labels = [], []
        rng = np.random.RandomState(42)
        for _ in range(n):
            bat_diff  = rng.normal(0, 4)
            bowl_diff = rng.normal(0, 2)
            form_diff = rng.choice([-2,-1,0,1,2])
            h2h_wr    = rng.uniform(0.3, 0.7)
            t1_vwr    = rng.uniform(0.35, 0.65)
            t2_vwr    = rng.uniform(0.35, 0.65)
            dew       = rng.uniform(0.1, 0.9)
            is_night  = float(rng.choice([0, 1]))
            pace_idx  = rng.uniform(4, 9); spin_idx = rng.uniform(3, 9)
            t1_elo = rng.normal(1500, 70)
            t2_elo = rng.normal(1500, 70)
            elo_diff = t1_elo - t2_elo
            t1_form_score = np.clip(0.5 + form_diff * 0.09 + rng.normal(0, 0.05), 0.05, 0.95)
            t2_form_score = np.clip(0.5 - form_diff * 0.09 + rng.normal(0, 0.05), 0.05, 0.95)
            toss_won = float(rng.choice([0, 1]))
            chose_bat = float(rng.choice([0, 1]))
            toss_bat_venue = rng.uniform(0.40, 0.62)
            toss_field_venue = rng.uniform(0.38, 0.60)
            pp_bowl = rng.uniform(11.5, 16.5)
            death_bowl = rng.uniform(11.0, 16.0)
            mid_bowl = rng.uniform(11.5, 16.5)
            t1_matchup = rng.uniform(0.93, 1.07)
            t2_matchup = rng.uniform(0.93, 1.07)
            matchup_diff = t1_matchup - t2_matchup
            row = [
                rng.uniform(155,190), pace_idx, spin_idx,
                rng.uniform(0.50,0.78), rng.uniform(0.42,0.58), rng.uniform(0.2,0.8),
                rng.uniform(22,40), rng.uniform(40,95), rng.uniform(0,60),
                rng.uniform(10,90), dew, rng.uniform(5,30), is_night,
                pace_idx, spin_idx, rng.uniform(4,8), rng.randint(145,195),
                rng.uniform(24,38)+bat_diff, rng.uniform(24,38),
                rng.uniform(11,18)+bowl_diff*0.3, rng.uniform(11,18),
                bat_diff, bowl_diff,
                t1_elo, t2_elo, elo_diff,
                rng.randint(0,6), rng.randint(0,6), t1_form_score, t2_form_score, float(form_diff),
                h2h_wr, rng.randint(5,35),
                t1_vwr, t2_vwr,
                rng.uniform(0.3,0.7), rng.uniform(0.3,0.7),
                rng.uniform(-0.3,0.3),
                toss_won, chose_bat, toss_bat_venue, toss_field_venue,
                pp_bowl, death_bowl, mid_bowl,
                t1_matchup, t2_matchup, matchup_diff,
            ]
            p_win = 1/(1+np.exp(-(
                bat_diff*0.16 + bowl_diff*0.20 + form_diff*0.14 + elo_diff*0.004 +
                matchup_diff*0.8 + (pp_bowl - death_bowl)*0.03 +
                (h2h_wr-0.5)*2.5 + (t1_vwr-t2_vwr)*2.0 +
                (dew-0.5)*0.8*(-is_night) + rng.normal(0,0.3)
            )))
            rows.append(row); labels.append(1 if p_win > 0.5 else 0)
        return rows, labels

    def train(self, X, y):
        print("\n🤖 Training ensemble models…")
        tscv = TimeSeriesSplit(n_splits=5)
        Xs = self.scaler.fit_transform(X)

        configs = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                eval_metric="logloss", random_state=42, verbosity=0),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
            "NeuralNet": MLPClassifier(
                hidden_layer_sizes=(128,64,32), max_iter=300,
                learning_rate="adaptive", random_state=42, early_stopping=True),
        }

        oof_preds = np.zeros((len(X), len(configs)))
        for ci, (name, clf) in enumerate(configs.items()):
            fold_scores = []
            oof = np.zeros(len(X))
            for fold, (tr, va) in enumerate(tscv.split(X)):
                Xtr, Xva = (X[tr], X[va]) if name == "XGBoost" else (Xs[tr], Xs[va])
                clf.fit(Xtr, y[tr])
                pva  = clf.predict_proba(Xva)[:,1]
                oof[va] = pva
                fold_scores.append(accuracy_score(y[va], (pva > 0.5).astype(int)))
            mean_acc = np.mean(fold_scores)
            self.cv_scores[name] = round(mean_acc, 4)
            bar = "█" * int(mean_acc * 30) + "░" * (30 - int(mean_acc * 30))
            print(f"  {name:12s} [{bar}] {mean_acc:.1%}")
            # Retrain on all data
            inp = X if name == "XGBoost" else Xs
            clf.fit(inp, y)
            self.models[name] = clf
            oof_preds[:, ci] = oof

        # Meta-learner
        print("  Training meta-learner (LogisticRegression)…")
        meta = LogisticRegression(C=1.0, random_state=42)
        meta.fit(oof_preds, y)
        self.models["meta"] = meta

        # Ensemble CV
        meta_preds = meta.predict_proba(oof_preds)[:,1]
        ens_acc = accuracy_score(y, (meta_preds > 0.5).astype(int))
        print(f"\n  {'Ensemble':12s} [{'█'*int(ens_acc*30)+'░'*(30-int(ens_acc*30))}] {ens_acc:.1%}")
        self.cv_scores["Ensemble"] = round(ens_acc, 4)
        self.trained = True

    def predict(self, features: dict):
        if not self.trained:
            raise RuntimeError("Call train() first")
        row = np.array([[features.get(k, 0) for k in self.FEATURE_NAMES]], dtype=float)
        row_s = self.scaler.transform(row)
        preds = []
        model_probs = {}
        for name, clf in self.models.items():
            if name == "meta": continue
            inp = row if name == "XGBoost" else row_s
            prob = clf.predict_proba(inp)[0][1]
            preds.append(prob); model_probs[name] = round(prob, 4)
        oof_row = np.array([preds])
        final = self.models["meta"].predict_proba(oof_row)[0][1]
        std   = np.std(preds)
        confidence = "HIGH" if std < 0.06 else "MEDIUM" if std < 0.12 else "LOW"
        return {
            "win_prob_t1":  round(final, 4),
            "win_prob_t2":  round(1 - final, 4),
            "model_probs":  model_probs,
            "std_dev":      round(std, 4),
            "confidence":   confidence,
        }

    def feature_importance(self):
        if "XGBoost" not in self.models: return {}
        imp = self.models["XGBoost"].feature_importances_
        return dict(sorted(zip(self.FEATURE_NAMES, imp), key=lambda x: -x[1]))

    def save(self, path=MODELS_DIR/"ipl_ensemble.pkl"):
        joblib.dump({"models": self.models, "scaler": self.scaler,
                     "cv_scores": self.cv_scores}, path)
        print(f"💾 Models saved to {path}")

    def load(self, path=MODELS_DIR/"ipl_ensemble.pkl"):
        data = joblib.load(path)
        self.models = data["models"]; self.scaler = data["scaler"]
        self.cv_scores = data.get("cv_scores", {}); self.trained = True
        print(f"✅ Models loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER PROJECTOR
# ─────────────────────────────────────────────────────────────────────────────
# %%
class PlayerProjector:
    def __init__(self, feat_eng: FeatureEngineer):
        self.fe = feat_eng

    def project_batting(self, squad, venue, pitch, weather, opposition_squad):
        """Project batting scores for all 11 players independently."""
        players = self._probable_xi(squad)
        results = []

        for pos, player in enumerate(players, 1):
            db = PLAYER_DB.get(player, {})
            role = db.get("role", "BAT")
            is_tail = (pos >= 9) or (role == "BOWL" and pos >= 8)

            if is_tail:
                proj = self._tail_projection(player, db, pitch)
            else:
                proj = self._bat_projection(player, db, pos, venue, pitch, weather)

            if pos > 1 and results:
                syn = self.fe.partnership_synergy(results[-1]["player"], player)
                boost = 1.0 + (syn["synergy"] - 1.0) * 0.20
                boost = np.clip(boost, 0.92, 1.14)
                proj["runs"] = round(proj["runs"] * boost, 1)
                proj["partnership_synergy"] = round(syn["synergy"], 3)
            else:
                proj["partnership_synergy"] = 1.0

            proj.update({
                "position": pos,
                "player": player,
                "role": role,
                "bat_style": db.get("bat_style", "RHB"),
                "bowl_style": db.get("bowl_style") or "—",
            })
            results.append(proj)

        return results

    def _bat_projection(self, player, db, position, venue, pitch, weather):
        """
        Two-component T20 run model:
          1) dismissal-early mode
          2) proper innings mode with phase-aware strike rate
        """
        base_avg = float(db.get("bat_avg") or 22.0)
        base_sr = float(db.get("bat_sr") or 128.0)

        vr = self.fe.player_venue_record(player, venue)
        if vr.get("has_data") and vr.get("venue_innings", 0) >= 5:
            w = min(vr["venue_innings"] / 15.0, 0.50)
            base_avg = base_avg * (1 - w) + vr["venue_avg"] * w
            base_sr = base_sr * (1 - w) + vr["venue_sr"] * w

        si = pitch.get("spin_index", 5.0)
        pi = pitch.get("pace_index", 5.0)
        bat = db.get("bat_style", "RHB")

        if pitch.get("batting_friendly"):
            base_avg *= 1.07
            base_sr *= 1.03
        if si > 7.0 and bat == "RHB":
            base_avg *= 0.88
            base_sr *= 0.93
        if si > 7.0 and bat == "LHB":
            base_avg *= 0.94
        if pi > 7.5:
            base_avg *= 0.91
        if pi > 6.5 and bat == "LHB":
            base_sr *= 1.03
        if weather.get("dew_risk", 0) > 0.55 and weather.get("is_night", True):
            base_sr *= 1.05

        eff_sr = base_sr * _PHASE_SR_BOOST.get(position, 1.0)
        lo, hi = _BALLS.get(position, (3, 10))
        balls = int(np.random.uniform(lo, hi))

        p_out = _dismissal_p(base_avg, position)
        if np.random.random() < p_out:
            runs = float(np.random.choice(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                p=[0.09, 0.08, 0.08, 0.08, 0.07, 0.08, 0.07, 0.07,
                   0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.05]
            ))
            balls = max(1, int(balls * 0.25))
        else:
            noise = np.random.uniform(0.82, 1.18)
            runs = balls * (eff_sr / 100.0) * noise
            runs = float(np.clip(runs, 0, 125))

        runs = round(runs, 1)
        balls = max(1, balls)
        actual_sr = round(runs / balls * 100, 1)
        return {"runs": runs, "balls": balls, "sr": actual_sr, "proj_avg": round(base_avg, 1)}

    def _tail_projection(self, player, db, pitch):
        """Realistic tail profile: mostly low scores with rare cameos."""
        base_avg = float(db.get("bat_avg") or 6.0)
        base_sr = float(db.get("bat_sr") or 82.0)
        if np.random.random() < 0.70:
            runs = float(np.random.choice(
                [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                p=[0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.09, 0.08, 0.07, 0.06]
            ))
            balls = max(1, int(np.random.uniform(1, 5)))
        else:
            balls = int(np.random.uniform(5, 14))
            runs = float(np.clip(balls * base_sr * 0.75 / 100, 0, 30))
        return {
            "runs": round(runs, 1),
            "balls": max(1, balls),
            "sr": round(base_sr * 0.78, 1),
            "proj_avg": round(base_avg * 0.55, 1),
        }

    def project_bowling(self, squad, venue, pitch, weather):
        """Project bowling figures for key bowlers."""
        players = _all_players(squad)
        results = []
        for player in players:
            db = PLAYER_DB.get(player, {})
            eco = db.get("bowl_eco"); avg = db.get("bowl_avg"); style = db.get("bowl_style","")
            if not eco or not avg: continue
            # Max overs allocation (simplified)
            max_ov = 4.0
            # Pitch-style modifier
            pi = pitch.get("pace_index", 5); si = pitch.get("spin_index", 5)
            eco_mod = 1.0; wkt_mod = 1.0
            if style in ["RF","RFM","LFM","LF"]:
                if pi > 6.5: eco_mod = 0.92; wkt_mod = 1.15
                elif si > 7: eco_mod = 1.10; wkt_mod = 0.82
            elif style in ["OB","SLA","LBG","LBC","SLO"]:
                if si > 6.5: eco_mod = 0.90; wkt_mod = 1.18
                elif pi > 7: eco_mod = 1.08; wkt_mod = 0.78
            # Dew hurts spinners 2nd innings
            if weather.get("dew_risk", 0) > 0.6 and style in ["OB","SLA","LBG","LBC","SLO"]:
                eco_mod *= 1.12; wkt_mod *= 0.85
            proj_eco = round(eco * eco_mod + np.random.normal(0, 0.3), 2)
            proj_eco = max(5.0, min(14.0, proj_eco))
            wkt_rate = max_ov / max(avg * wkt_mod / 6, 6)
            proj_wkts = round(float(np.random.poisson(max(0.1, wkt_rate))), 0)
            proj_wkts = min(4, int(proj_wkts))
            proj_runs = round(proj_eco * max_ov)
            suited = ((style in ["RF","RFM","LFM","LF"] and pi > 6.5) or
                      (style in ["OB","SLA","LBG","LBC","SLO"] and si > 6.5))
            results.append({
                "player": player, "style": style or "—",
                "overs": max_ov, "wickets": proj_wkts,
                "runs": proj_runs, "economy": proj_eco,
                "suited_to_pitch": suited,
            })
        results.sort(key=lambda x: (-x["wickets"], x["economy"]))
        return results[:6]

    def _probable_xi(self, squad):
        """Build a probable playing XI from the squad."""
        players = _all_players(squad)
        if len(players) <= 11: return players[:11]
        # Priority: WK > top-order BAT > ALL > BOWL
        wks   = [p for p in players if PLAYER_DB.get(p,{}).get("role","")=="WK-BAT"][:2]
        bats  = [p for p in players if PLAYER_DB.get(p,{}).get("role","")=="BAT" and p not in wks][:5]
        alls  = [p for p in players if PLAYER_DB.get(p,{}).get("role","")=="ALL" and p not in wks+bats][:3]
        bowls = [p for p in players if PLAYER_DB.get(p,{}).get("role","")=="BOWL" and p not in wks+bats+alls]
        xi = (wks + bats + alls + bowls)[:11]
        # Top up with remaining if needed
        rem = [p for p in players if p not in xi]
        xi += rem[:max(0, 11 - len(xi))]
        return xi[:11]


# ─────────────────────────────────────────────────────────────────────────────
# MATCH ANALYZER  — orchestrates everything
# ─────────────────────────────────────────────────────────────────────────────
# %%

class MatchAnalyzer:
    def __init__(self, feat_eng, model, projector, weather_mod, pitch_pred, squads):
        self.fe      = feat_eng
        self.model   = model
        self.proj    = projector
        self.weather = weather_mod
        self.pitch   = pitch_pred
        self.squads  = squads

    def analyze(self, team1, team2, venue, match_date=None, match_time="19:30",
                match_n_at_venue=1):
        t1 = _resolve_team(team1); t2 = _resolve_team(team2)
        if match_date is None:
            match_date = datetime.today().strftime("%Y-%m-%d")

        print(f"\n{BOLD}{'─'*65}")
        print(f"  ⏳  Fetching match-day data…")
        print(f"{'─'*65}{RESET}")

        weather = self.weather.get(venue, match_date, match_time)
        pitch   = self.pitch.predict(venue, match_date, weather, match_n_at_venue)
        sq1     = self.squads.get(t1, FALLBACK_SQUADS.get(t1, FALLBACK_SQUADS["MI"]))
        sq2     = self.squads.get(t2, FALLBACK_SQUADS.get(t2, FALLBACK_SQUADS["CSK"]))
        features = self.fe.build(t1, t2, venue, weather, pitch, {t1: sq1, t2: sq2})
        pred    = self.model.predict(features)
        bat1    = self.proj.project_batting(sq1, venue, pitch, weather, sq2)
        bat2    = self.proj.project_batting(sq2, venue, pitch, weather, sq1)
        bowl1   = self.proj.project_bowling(sq1, venue, pitch, weather)
        bowl2   = self.proj.project_bowling(sq2, venue, pitch, weather)
        self._last_weather = weather
        score1  = self._team_score(bat1, pitch, features=features, innings=1)
        score2  = self._team_score(bat2, pitch, features=features, innings=2)
        imp     = self.model.feature_importance()

        self._print_report(
            t1, t2, venue, match_date, match_time,
            weather, pitch, pred, bat1, bat2, bowl1, bowl2,
            score1, score2, imp, sq1, sq2
        )
        return {"prediction": pred, "score1": score1, "score2": score2,
                "weather": weather, "pitch": pitch}

    def _team_score(self, bat_proj, pitch, features=None, innings=1):
        base = float(pitch.get("expected_score", 172))
        base = base * 1.04

        if features:
            if innings == 1:
                bat_str = features.get("t1_bat", 28.0)
                bowl_str = features.get("t2_bowl", 13.5)
            else:
                bat_str = features.get("t2_bat", 28.0)
                bowl_str = features.get("t1_bowl", 13.5)
            strength_adj = np.clip((bat_str / 28.0) / (bowl_str / 13.5), 0.80, 1.20)
        else:
            strength_adj = 1.0

        top3_runs = sum(p["runs"] for p in bat_proj[:3])
        pp_adj = np.clip(1.0 + (top3_runs - 75) * 0.002, 0.94, 1.10)

        death_runs = sum(p["runs"] for p in bat_proj[5:8])
        death_adj = np.clip(1.0 + (death_runs - 30) * 0.003, 0.94, 1.10)

        dew = 0.0
        if hasattr(self, "_last_weather"):
            dew = self._last_weather.get("dew_risk", 0)
        dew_bonus = 1.0 + dew * 0.06 if innings == 2 and dew > 0.55 else 1.0

        raw_sum = sum(p["runs"] for p in bat_proj)
        extras = int(np.random.randint(10, 22))
        ind_total = raw_sum + extras

        physics = base * strength_adj * pp_adj * death_adj * dew_bonus
        blended = physics * 0.65 + ind_total * 0.35
        blended = float(np.clip(blended, 128, 235))

        if blended > 200:
            wkts = int(np.random.choice([2, 3, 4, 5, 6], p=[0.10, 0.25, 0.35, 0.20, 0.10]))
        elif blended > 185:
            wkts = int(np.random.choice([3, 4, 5, 6, 7], p=[0.10, 0.25, 0.35, 0.20, 0.10]))
        elif blended > 168:
            wkts = int(np.random.choice([4, 5, 6, 7, 8], p=[0.10, 0.25, 0.35, 0.20, 0.10]))
        elif blended > 150:
            wkts = int(np.random.choice([5, 6, 7, 8, 9], p=[0.10, 0.25, 0.35, 0.20, 0.10]))
        else:
            wkts = int(np.random.choice([6, 7, 8, 9, 10], p=[0.10, 0.20, 0.30, 0.25, 0.15]))

        sigma = 22
        return {
            "projected": round(blended),
            "low": round(max(110, blended - 1.3 * sigma)),
            "high": round(min(240, blended + 1.3 * sigma)),
            "wickets": wkts,
            "extras": extras,
            "physics_score": round(physics),
            "individual_sum": round(ind_total),
        }

    def _print_report(self, t1, t2, venue, date, time,
                      weather, pitch, pred, bat1, bat2, bowl1, bowl2,
                      score1, score2, imp, sq1, sq2):
        t1n = TEAMS.get(t1, t1); t2n = TEAMS.get(t2, t2)
        vdata = _find_venue(venue) or {}
        vname = vdata.get("_name", venue)
        city  = vdata.get("city", "")
        p1 = pred["win_prob_t1"]; p2 = pred["win_prob_t2"]

        print(f"\n{BOLD}{CYAN}{'═'*65}")
        print(f"  🏏  IPL 2026 — AI MATCH ANALYSIS")
        print(f"{'═'*65}{RESET}")
        print(f"  {BOLD}{t1n}  vs  {t2n}{RESET}")
        print(f"  📍 {vname}, {city}  |  📅 {date}  |  🕐 {time} IST\n")
# %%
        # ── Weather ──────────────────────────────────────────────────────────
        print(f"{YELLOW}{'─'*65}")
        print(f"  🌤  WEATHER CONDITIONS")
        print(f"{'─'*65}{RESET}")
        wrows = [
            ["Temperature",  f"{weather['temp_c']}°C (feels {weather['feels_like']}°C)"],
            ["Humidity",     f"{weather['humidity']}%"],
            ["Rain Risk",    f"{weather['rain_prob']}%"],
            ["Cloud Cover",  f"{weather['cloud_cover']}%"],
            ["Wind",         f"{weather['wind_kph']} kph"],
            ["Dew Risk",     f"{weather['dew_risk']:.0%}"],
            ["Conditions",   weather["conditions"]],
        ]
        print(tabulate(wrows, tablefmt="plain"))
        print(f"\n  💬 {weather['impact']}\n")
# %%
        # ── Pitch ─────────────────────────────────────────────────────────────
        print(f"{YELLOW}{'─'*65}")
        print(f"  🏏  PITCH REPORT — {pitch['pitch_type']}")
        print(f"{'─'*65}{RESET}")
        prows = [
            ["Pace Index",    f"{pitch['pace_index']}/10"],
            ["Spin Index",    f"{pitch['spin_index']}/10"],
            ["Bounce Index",  f"{pitch['bounce_index']}/10"],
            ["Expected Score",f"{pitch['expected_score']} (1st innings)"],
            ["Batting Surface",  "Batting-friendly ✅" if pitch["batting_friendly"] else "Bowling-friendly ⚠️"],
            ["Spin from over",f"Over {pitch['spin_advantage_from_over']}"],
        ]
        print(tabulate(prows, tablefmt="plain"))
        print(f"\n  💬 {pitch['description']}\n")
# %%
        # ── Win Probability ───────────────────────────────────────────────────
        print(f"{YELLOW}{'─'*65}")
        print(f"  🎯  WIN PROBABILITY")
        print(f"{'─'*65}{RESET}")
        bar_len = 40
        b1 = int(p1 * bar_len); b2 = bar_len - b1
        print(f"  {t1n[:22]:22s} {'█'*b1}{'░'*b2} {p1:.1%}")
        print(f"  {t2n[:22]:22s} {'░'*b1}{'█'*b2} {p2:.1%}")
        print(f"\n  Model confidence: {pred['confidence']}  (σ={pred['std_dev']:.3f})")
        print(f"\n  Individual models:")
        for name, prob in pred["model_probs"].items():
            print(f"    {name:12s} → {TEAMS.get(t1,t1)} {prob:.1%}  /  {TEAMS.get(t2,t2)} {1-prob:.1%}")
# %%
        # ── Score Projection ──────────────────────────────────────────────────
        print(f"\n{YELLOW}{'─'*65}")
        print(f"  📊  PROJECTED SCORES")
        print(f"{'─'*65}{RESET}")
        srows = [
            [t1n, f"{score1['projected']}", f"{score1['low']}–{score1['high']}", f"{score1['wickets']} wkts"],
            [t2n, f"{score2['projected']}", f"{score2['low']}–{score2['high']}", f"{score2['wickets']} wkts"],
        ]
        print(tabulate(srows, headers=["Team","Projected","Range","Wickets"], tablefmt="rounded_outline"))
# %%
        # ── Individual Batting ────────────────────────────────────────────────
        for team_name, bat_proj in [(t1n, bat1), (t2n, bat2)]:
            print(f"\n{YELLOW}{'─'*65}")
            print(f"  🏏  {team_name.upper()} — BATTING PROJECTIONS")
            print(f"{'─'*65}{RESET}")
            brows = []
            for p in bat_proj[:11]:
                syn = f"⚡{p['partnership_synergy']:.2f}" if p.get('partnership_synergy',1.0) > 1.05 else ""
                brows.append([
                    f"{p['position']}.", p["player"][:22], p["bat_style"],
                    f"{p['runs']:.0f}", f"{p['balls']}", f"{p['sr']:.0f}", syn
                ])
            print(tabulate(brows,
                           headers=["#","Player","Style","Proj Runs","Balls","SR","Syn"],
                           tablefmt="rounded_outline"))
# %%
        # ── Individual Bowling ────────────────────────────────────────────────
        for team_name, bowl_proj in [(t1n, bowl1), (t2n, bowl2)]:
            print(f"\n{YELLOW}{'─'*65}")
            print(f"  🎳  {team_name.upper()} — BOWLING PROJECTIONS")
            print(f"{'─'*65}{RESET}")
            bowrows = []
            for p in bowl_proj:
                suited = "✅" if p["suited_to_pitch"] else "—"
                bowrows.append([
                    p["player"][:22], p["style"],
                    p["overs"], p["wickets"],
                    p["runs"], f"{p['economy']:.1f}", suited
                ])
            print(tabulate(bowrows,
                           headers=["Player","Style","Overs","Wkts","Runs","Eco","Pitch fit"],
                           tablefmt="rounded_outline"))
# %%
        # ── Key Factors ───────────────────────────────────────────────────────
        print(f"\n{YELLOW}{'─'*65}")
        print(f"  🔑  TOP DECISIVE FACTORS")
        print(f"{'─'*65}{RESET}")
        top_feats = list(imp.items())[:8]
        feat_labels = {
            "bat_diff":"Batting strength gap","bowl_diff":"Bowling strength gap",
            "form_diff":"Recent form","h2h_wr":"Head-to-head record",
            "t1_venue_wr":"Home advantage (T1)","t2_venue_wr":"Home advantage (T2)",
            "w_dew":"Dew factor","p_spin":"Spin friendliness",
            "p_pace":"Pace friendliness","pitch_aff_diff":"Bowling attack vs pitch",
            "venue_chase_wr":"Chase win rate at venue","is_night":"Night match",
        }
        for fname, fval in top_feats:
            label = feat_labels.get(fname, fname.replace("_"," ").title())
            bar = "▓" * max(1, int(fval * 300))
            print(f"  {label:32s} {bar}")
# %%
        # ── Toss Advice ───────────────────────────────────────────────────────
        print(f"\n{YELLOW}{'─'*65}")
        print(f"  🪙  TOSS STRATEGY")
        print(f"{'─'*65}{RESET}")
        dew = weather.get("dew_risk", 0.3)
        cwr = vdata.get("chase_win_rate", 0.5)
        if dew > 0.65:
            print(f"  ➡️  WIN TOSS → BOWL FIRST")
            print(f"     Heavy dew ({dew:.0%} risk) will make batting 2nd significantly easier.")
        elif cwr < 0.45:
            print(f"  ➡️  WIN TOSS → BAT FIRST")
            print(f"     {vname} has a low chase win rate ({cwr:.0%}). Set a target.")
        elif pitch["pitch_type"] in ["TURNER","SPIN-FRIENDLY"]:
            print(f"  ➡️  WIN TOSS → BAT FIRST")
            print(f"     Pitch expected to deteriorate — bat while surface is fresh.")
        elif pitch["pitch_type"] in ["FAST & BOUNCY","PACE-FRIENDLY"]:
            print(f"  ➡️  WIN TOSS → BAT FIRST")
            print(f"     Fast surface rewards batting first; seam settles after 6 overs.")
        else:
            print(f"  ➡️  WIN TOSS → PREFERENCE: {'BOWL FIRST' if cwr > 0.52 else 'BAT FIRST'}")
            print(f"     Balanced conditions. Historical chase win rate at this venue: {cwr:.0%}.")
# %%
        # ── Final Verdict ─────────────────────────────────────────────────────
        print(f"\n{BOLD}{GREEN}{'═'*65}")
        winner = t1n if p1 > p2 else t2n
        loser  = t2n if p1 > p2 else t1n
        wp     = max(p1, p2)
        print(f"  🏆  PREDICTION: {winner} WIN  ({wp:.1%} confidence)")
        print(f"      Expected score: {score1['projected']} vs {score2['projected']}")
        margin = abs(score1["projected"] - score2["projected"])
        print(f"      Projected margin: ~{margin} runs")
        print(f"{'═'*65}{RESET}")
        print(f"\n  ⚠️  AI prediction — for entertainment only. Cricket is beautifully")
        print(f"      unpredictable. Past data ≠ guaranteed outcome.\n")

class ModelEvaluator:
    """Backtest the IPL predictor on historical Cricsheet data."""

    def __init__(self, analyzer, info_df, ball_df, squads):
        self.analyzer = analyzer
        self.info_df = info_df
        self.ball_df = ball_df
        self.squads = squads

    def run(self, test_seasons=range(2022, 2026), verbose=True):
        if self.info_df.empty:
            print("No Cricsheet data available for backtest.")
            return {}

        results = []
        score_errs = []
        by_season = defaultdict(list)
        match_ids = self.info_df[self.info_df["key"] == "winner"]["match_id"].unique()

        for mid in sorted(match_ids):
            mi = self.info_df[self.info_df["match_id"] == mid]
            winner_row = mi[mi["key"] == "winner"]["value"].values
            team_rows = mi[mi["key"] == "team"]["value"].tolist()
            date_rows = mi[mi["key"] == "date"]["value"].values
            venue_rows = mi[mi["key"] == "venue"]["value"].values
            if len(winner_row) < 1 or len(team_rows) < 2 or len(date_rows) < 1:
                continue

            try:
                year = int(str(date_rows[0])[:4])
            except Exception:
                continue
            if year not in test_seasons:
                continue

            winner = winner_row[0]
            t1, t2 = team_rows[0], team_rows[1]
            venue = venue_rows[0] if len(venue_rows) else "Wankhede Stadium"
            t1a = _resolve_team(t1)
            t2a = _resolve_team(t2)
            if t1a not in TEAMS or t2a not in TEAMS:
                continue

            try:
                weather = self.analyzer.weather.get(venue, date_rows[0])
                pitch = self.analyzer.pitch.predict(venue, date_rows[0], weather)
                sq1 = self.squads.get(t1a, FALLBACK_SQUADS.get(t1a, {}))
                sq2 = self.squads.get(t2a, FALLBACK_SQUADS.get(t2a, {}))
                features = self.analyzer.fe.build(t1a, t2a, venue, weather, pitch, {t1a: sq1, t2a: sq2})
                pred = self.analyzer.model.predict(features)
                prob_t1 = pred["win_prob_t1"]
                actual = 1.0 if t1.lower() in winner.lower() else 0.0
                results.append((prob_t1, actual))
                by_season[year].append((prob_t1, actual))

                actual_scores = self._actual_score(mid)
                if actual_scores:
                    bat1 = self.analyzer.proj.project_batting(sq1, venue, pitch, weather, sq2)
                    self.analyzer._last_weather = weather
                    s1 = self.analyzer._team_score(bat1, pitch, features=features, innings=1)
                    score_errs.append(abs(s1.get("projected", 170) - actual_scores[0]))
            except Exception:
                continue

        if not results:
            print("No valid backtest matches found.")
            return {}

        metrics = self._compute_metrics(results, score_errs, by_season)
        if verbose:
            self._print_report(metrics)
        return metrics

    def _actual_score(self, match_id):
        if self.ball_df.empty:
            return None
        try:
            ic = "innings" if "innings" in self.ball_df.columns else "inning"
            rc = "runs_off_bat" if "runs_off_bat" in self.ball_df.columns else "batsman_runs"
            ec = "extras" if "extras" in self.ball_df.columns else "extra_runs"
            md = self.ball_df[self.ball_df["match_id"] == match_id]
            if md.empty:
                return None
            inn1 = md[md[ic] == 1]
            if inn1.empty:
                return None
            score = inn1[rc].sum() + inn1.get(ec, pd.Series([0] * len(inn1))).sum()
            return (int(score),)
        except Exception:
            return None

    def _compute_metrics(self, results, score_errs, by_season):
        probs = np.array([r[0] for r in results])
        actuals = np.array([r[1] for r in results])
        preds = (probs > 0.5).astype(float)

        accuracy = float(np.mean(preds == actuals))
        brier = float(np.mean((probs - actuals) ** 2))
        eps = 1e-7
        logloss = float(-np.mean(actuals * np.log(probs + eps) + (1 - actuals) * np.log(1 - probs + eps)))
        auc = self._auc(probs, actuals)
        cal = self._calibration(probs, actuals)
        score_rmse = float(np.sqrt(np.mean(np.array(score_errs) ** 2))) if score_errs else None

        metrics = {
            "n_matches": len(results),
            "accuracy": round(accuracy, 4),
            "brier": round(brier, 4),
            "log_loss": round(logloss, 4),
            "roc_auc": round(auc, 4),
            "score_rmse": round(score_rmse, 1) if score_rmse else None,
            "calibration": cal,
            "by_season": {},
        }
        for yr, yr_res in sorted(by_season.items()):
            yr_probs = np.array([r[0] for r in yr_res])
            yr_actuals = np.array([r[1] for r in yr_res])
            metrics["by_season"][yr] = {
                "n": len(yr_res),
                "accuracy": round(float(np.mean((yr_probs > 0.5) == yr_actuals)), 3),
                "brier": round(float(np.mean((yr_probs - yr_actuals) ** 2)), 4),
            }
        return metrics

    def _auc(self, probs, actuals):
        pairs = sorted(zip(probs, actuals), reverse=True)
        n_pos = actuals.sum()
        n_neg = len(actuals) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = fp = auc = 0
        prev_fp = prev_tp = 0
        for _, a in pairs:
            if a == 1:
                tp += 1
            else:
                fp += 1
            if fp != prev_fp:
                auc += (tp + prev_tp) * (fp - prev_fp) / 2
                prev_fp = fp
                prev_tp = tp
        return auc / (n_pos * n_neg)

    def _calibration(self, probs, actuals, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        result = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() < 3:
                continue
            pred_mean = float(probs[mask].mean())
            actual_mean = float(actuals[mask].mean())
            result.append({
                "pred_prob": round(pred_mean, 3),
                "actual_rate": round(actual_mean, 3),
                "n": int(mask.sum()),
                "gap": round(abs(pred_mean - actual_mean), 3),
            })
        return result

    def _print_report(self, m):
        print("\n" + "=" * 60)
        print("IPL MODEL ACCURACY EVALUATION")
        print("=" * 60)
        print(f"Matches: {m['n_matches']}")
        print(f"Accuracy: {m['accuracy']:.1%}")
        print(f"Brier: {m['brier']:.4f}")
        print(f"Log loss: {m['log_loss']:.4f}")
        print(f"ROC-AUC: {m['roc_auc']:.4f}")
        if m["score_rmse"] is not None:
            print(f"Score RMSE: {m['score_rmse']:.1f}")
        print("Per season:")
        for yr, ys in sorted(m["by_season"].items()):
            print(f"  {yr}: acc={ys['accuracy']:.1%} brier={ys['brier']:.4f} n={ys['n']}")


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM SETUP
# ─────────────────────────────────────────────────────────────────────────────
# %%
def setup_system():
    print(f"\n{BOLD}{CYAN}{'═'*65}")
    print("  IPL 2026 AI PREDICTOR — SYSTEM INITIALISATION")
    print(f"{'═'*65}{RESET}\n")

    fetcher = IPLDataFetcher()
    model_path = MODELS_DIR / "ipl_ensemble.pkl"

    # Data
    fetcher.download_cricsheet()
    info_df, ball_df = fetcher.load_all_matches()
    squads = fetcher.scrape_squads()

    # Feature engineer
    fe = FeatureEngineer(ball_df, info_df)

    # Model
    trainer = ModelTrainer()
    if model_path.exists():
        try:
            trainer.load(model_path)
            print("✅ Pre-trained models loaded (skipping re-training)")
        except:
            print("⚠️  Model load failed — re-training…")
            trainer.trained = False

    if not trainer.trained:
        X, y = trainer.prepare_dataset(info_df, ball_df, fe, squads)
        trainer.train(X, y)
        trainer.save(model_path)

    projector = PlayerProjector(fe)
    weather   = WeatherModule()
    pitch     = PitchPredictor()
    analyzer  = MatchAnalyzer(fe, trainer, projector, weather, pitch, squads)

    print(f"\n{GREEN}✅ System ready!{RESET}\n")
    return analyzer, squads


def list_teams():
    print(f"\n{BOLD}IPL 2026 Teams:{RESET}")
    for abbr, full in TEAMS.items():
        sq = FALLBACK_SQUADS[abbr]
        print(f"  {CYAN}{abbr:5s}{RESET}  {full}  (Captain: {sq['captain']})")

def list_venues():
    print(f"\n{BOLD}IPL Venues:{RESET}")
    for name, v in IPL_VENUES.items():
        print(f"  {CYAN}{name}{RESET}")
        print(f"         {v['city']}, {v['state']} | Avg 1st inn: {v['avg_first_innings']} | "
              f"Pace: {v['pace_index']}/10 | Spin: {v['spin_index']}/10")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────

def interactive(analyzer):
    print(f"\n{BOLD}{CYAN}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         IPL 2026 — AI MATCH PREDICTOR  🏏               ║")
    print("║  Commands:  predict | teams | venues | quit             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(RESET)

    while True:
        try:
            cmd = input(f"{BOLD}> Command:{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting. Good cricket!"); break

        if cmd in ("q","quit","exit"):
            print("👋 Exiting. Good cricket!"); break

        elif cmd in ("teams","t"):
            list_teams()

        elif cmd in ("venues","v"):
            list_venues()

        elif cmd in ("predict","p",""):
            print(f"\n{YELLOW}Enter match details:{RESET}")
            try:
                t1_in = input("  Team 1 (e.g. MI, Mumbai Indians): ").strip()
                t2_in = input("  Team 2 (e.g. CSK, Chennai): ").strip()
                ven   = input("  Venue (e.g. Wankhede, Chepauk, Eden Gardens): ").strip()
                date  = input(f"  Match date (YYYY-MM-DD) [{datetime.today().strftime('%Y-%m-%d')}]: ").strip()
                if not date: date = datetime.today().strftime("%Y-%m-%d")
                mtime = input("  Match time IST [19:30]: ").strip()
                if not mtime: mtime = "19:30"
                mnum  = input("  Match # at this venue this season [1]: ").strip()
                mnum  = int(mnum) if mnum.isdigit() else 1

                analyzer.analyze(t1_in, t2_in, ven, date, mtime, mnum)

            except KeyboardInterrupt:
                print("\n  (Cancelled)")
            except Exception as e:
                print(f"  ❌ Error: {e}")

        elif cmd in ("help","h","?"):
            print("  predict  — run full match analysis")
            print("  teams    — list all 10 IPL teams")
            print("  venues   — list all 21 IPL venues")
            print("  quit     — exit")
        else:
            print(f"  Unknown command '{cmd}'. Type 'help' for options.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer, squads = setup_system()
    interactive(analyzer)
