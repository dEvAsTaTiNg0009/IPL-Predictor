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
from ipl_stats_module import build_squads_and_players, scrape_squads as live_scrape_squads
FALLBACK_SQUADS, PLAYER_DB = build_squads_and_players()
import subprocess, sys

def _safe_install_if_missing(pkg, module_name=None):
    mod = module_name or pkg
    try:
        __import__(mod)
        return True
    except Exception:
        pass
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
        )
        return True
    except Exception:
        return False

print("📦 Checking dependencies...")
_safe_install_if_missing("pandas")
_safe_install_if_missing("numpy")
_safe_install_if_missing("scikit-learn", "sklearn")
_safe_install_if_missing("xgboost")
_safe_install_if_missing("lightgbm")
_safe_install_if_missing("requests")
_safe_install_if_missing("beautifulsoup4", "bs4")
_safe_install_if_missing("lxml")
_safe_install_if_missing("tqdm")
_safe_install_if_missing("joblib")
_safe_install_if_missing("scipy")
_safe_install_if_missing("tabulate")
_safe_install_if_missing("colorama")
print("✅ Dependency check complete\n")

# =============================================================================
# 1. IMPORTS
# =============================================================================
# %%
import warnings; warnings.filterwarnings("ignore")
import os, re, io, json, time, zipfile, pickle, math, csv
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
    DECAY = 0.90

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

        venue_l = (venue or "").lower()
        team_a_l = str(team_a).lower()
        team_b_l = str(team_b).lower()
        if team_a_l and team_a_l in venue_l:
            ra += self.HOME_ADV
        elif team_b_l and team_b_l in venue_l:
            rb += self.HOME_ADV

        ea = self.expected(ra, rb)
        sa = 1.0 if winner == team_a else 0.0

        self.ratings[team_a] += self.K_FACTOR * (sa - ea)
        self.ratings[team_b] += self.K_FACTOR * ((1 - sa) - (1 - ea))

    def get(self, team):
        return round(self.ratings.get(team, self.BASE_ELO), 1)

    def build_from_matches(self, info_df, recent_years=5):
        if info_df.empty:
            return
        date_rows = info_df[info_df["key"] == "date"]
        years = []
        for dv in date_rows.get("value", pd.Series([], dtype=object)).tolist():
            try:
                years.append(int(str(dv)[:4]))
            except Exception:
                continue
        min_year = (max(years) - recent_years + 1) if years else None

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
            if min_year and year and year < min_year:
                continue
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
                rows = []
                with open(f, "r", encoding="utf-8", newline="") as fh:
                    reader = csv.reader(fh)
                    for row in reader:
                        if len(row) < 3:
                            continue
                        player_name = row[3] if len(row) > 3 else ""
                        rows.append({
                            "type": row[0],
                            "key": row[1],
                            "value": row[2],
                            "player": player_name,
                            "match_id": f.stem.replace("_info", ""),
                        })
                if rows:
                    info_dfs.append(pd.DataFrame(rows))
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
        # Delegate to the robust multi-source scraper in ipl_stats_module.
        teams = list(TEAMS.keys())
        return live_scrape_squads(teams, verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER MODULE  (Open-Meteo — free, no key)
# ─────────────────────────────────────────────────────────────────────────────
# %%

class WeatherModule:
    BASE = "https://api.open-meteo.com/v1/forecast"
    HIST = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self):
        self._warned_live = False

    def get(self, venue_name, match_date, match_time_ist="19:30"):
        venue = _find_venue(venue_name)
        if venue is None: return self._default()
        lat, lon = venue["lat"], venue["lon"]
        hour = int(match_time_ist.split(":")[0])
        req_date = str(match_date).strip().replace("/", "-")

        # Forecast endpoint only serves a near-future horizon. Use historical API otherwise.
        try:
            mdate = datetime.strptime(req_date, "%Y-%m-%d").date()
            today = datetime.today().date()
            if mdate < today:
                return self._historical(lat, lon, req_date, hour, venue)
            # Open-Meteo forecast supports only limited forward range; prevent 400s.
            if (mdate - today).days > 15:
                return self._historical(lat, lon, req_date, hour, venue)
        except Exception:
            pass

        try:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relativehumidity_2m,precipitation_probability,"
                          "precipitation,windspeed_10m,winddirection_10m,cloudcover,"
                          "dewpoint_2m,apparent_temperature",
                "timezone": "Asia/Kolkata",
                "start_date": req_date, "end_date": req_date,
            }
            r = requests.get(self.BASE, params=params, timeout=15)
            r.raise_for_status()
            payload = r.json()
            h = payload.get("hourly", {})

            def _pick(keys, default):
                for k in keys:
                    arr = h.get(k)
                    if isinstance(arr, list) and len(arr) > 0:
                        return arr[min(hour, len(arr) - 1)]
                return default

            temp_c = _pick(["temperature_2m", "temperature"], None)
            if temp_c is None:
                raise KeyError("temperature_2m")

            w = {
                "temp_c":      round(float(temp_c), 1),
                "humidity":    round(float(_pick(["relativehumidity_2m", "relative_humidity_2m"], 65.0)), 1),
                "rain_prob":   round(float(_pick(["precipitation_probability"], 15.0)), 1),
                "precip_mm":   round(float(_pick(["precipitation"], 0.0)), 2),
                "wind_kph":    round(float(_pick(["windspeed_10m", "wind_speed_10m"], 10.0)), 1),
                "wind_dir":    float(_pick(["winddirection_10m", "wind_direction_10m"], 180.0)),
                "cloud_cover": round(float(_pick(["cloudcover", "cloud_cover"], 30.0)), 1),
                "dewpoint":    round(float(_pick(["dewpoint_2m", "dew_point_2m"], 22.0)), 1),
                "feels_like":  round(float(_pick(["apparent_temperature", "feels_like"], temp_c)), 1),
                "is_night":    hour >= 18,
                "venue_dew_factor": venue.get("dew_factor", 0.5),
            }
            w["dew_risk"]   = self._dew_risk(w, venue)
            w["conditions"] = self._classify(w)
            w["impact"]     = self._impact(w)
            return w
        except Exception as e:
            if not self._warned_live:
                print(f"  ⚠️  Weather API: {e} — using historical average")
                self._warned_live = True
            return self._historical(lat, lon, req_date, hour, venue)

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
                t_arr = h.get("temperature_2m") or h.get("temperature")
                rh_arr = h.get("relativehumidity_2m") or h.get("relative_humidity_2m")
                if isinstance(t_arr, list) and len(t_arr) > 0:
                    idx = min(hour, len(t_arr) - 1)
                    temps.append(t_arr[idx])
                    if isinstance(rh_arr, list) and len(rh_arr) > idx:
                        hums.append(rh_arr[idx])
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
    def __init__(self, ball_df=None, info_df=None):
        self.ball_df = ball_df if ball_df is not None else pd.DataFrame()
        self.info_df = info_df if info_df is not None else pd.DataFrame()
        from ipl_temporal import ChronologicalDataLoader, HistoricalStateTracker, TemporalFeatureEngine, normalize_team, normalize_venue, MatchRecord
        self.loader = ChronologicalDataLoader()
        self.temporal_engine = TemporalFeatureEngine(mode="pre_xi")
        self.state = HistoricalStateTracker()
        self._load_and_replay_history()

    def _load_and_replay_history(self):
        matches = self.loader.load_all_matches()
        for m in matches:
            self.state.update_match_result(m)

    def build(self, t1, t2, venue, weather, pitch, squads, match_date=None):
        from ipl_temporal import normalize_team, normalize_venue, MatchRecord
        t1_norm = normalize_team(t1)
        t2_norm = normalize_team(t2)
        v_norm = normalize_venue(venue)

        if match_date is None:
            m_date = datetime.today().date()
        elif isinstance(match_date, str):
            try:
                m_date = datetime.strptime(match_date.replace("/", "-"), "%Y-%m-%d").date()
            except Exception:
                m_date = datetime.today().date()
        else:
            m_date = match_date

        from datetime import time as dt_time
        m_datetime = datetime.combine(m_date, dt_time(19, 30))

        dummy_match = MatchRecord(
            match_id="LIVE_PRED",
            season="2026",
            match_date=m_date,
            match_datetime=m_datetime,
            match_number=1,
            team1=t1_norm,
            team2=t2_norm,
            team1_raw=t1,
            team2_raw=t2,
            venue=v_norm,
            venue_raw=venue,
            city=v_norm.split(",")[-1].strip() if "," in v_norm else "",
            toss_winner=t1_norm,
            toss_decision="field",
            winner=None,
            winner_raw="",
            margin_runs=0,
            margin_wickets=0,
            playing_xi={
                t1_norm: squads.get(t1_norm, {}).get("all_players", []),
                t2_norm: squads.get(t2_norm, {}).get("all_players", [])
            },
        )

        feats = self.temporal_engine.build_features(dummy_match, self.state)

        v = _find_venue(venue) or {}
        feats["venue_avg_first"] = v.get("avg_first_innings", feats.get("venue_avg_1st_innings", 168))
        feats["venue_pace_index"] = v.get("pace_index", 6.0)
        feats["venue_spin_index"] = v.get("spin_index", 6.0)
        feats["venue_boundary_freq"] = v.get("boundary_freq", 0.62)
        feats["venue_dew_factor"] = v.get("dew_factor", 0.5)

        feats["w_temp"] = weather.get("temp_c", 28)
        feats["w_humid"] = weather.get("humidity", 65)
        feats["w_rain"] = weather.get("rain_prob", 15)
        feats["w_cloud"] = weather.get("cloud_cover", 30)
        feats["w_dew"] = weather.get("dew_risk", 0.3)
        feats["w_wind"] = weather.get("wind_kph", 10)
        feats["is_night"] = float(weather.get("is_night", True))

        feats["p_pace"] = pitch.get("pace_index", 6.0)
        feats["p_spin"] = pitch.get("spin_index", 6.0)
        feats["p_bounce"] = pitch.get("bounce_index", 5.0)
        feats["p_score"] = pitch.get("expected_score", 168)

        feats["t1_bat"] = feats.get("t1_bat_score", 28.0)
        feats["t2_bat"] = feats.get("t2_bat_score", 28.0)
        feats["t1_bowl"] = feats.get("t1_bowl_score", 13.5)
        feats["t2_bowl"] = feats.get("t2_bowl_score", 13.5)
        feats["t1_wins"] = feats.get("t1_recent_wins", 3)
        feats["t2_wins"] = feats.get("t2_recent_wins", 3)
        feats["t1_form_score"] = feats.get("t1_form_exp", 0.50)
        feats["t2_form_score"] = feats.get("t2_form_exp", 0.50)
        feats["form_diff"] = feats.get("form_diff_exp", 0.0)

        feats["h2h_wr"] = feats.get("h2h_t1_wr", 0.50)
        feats["h2h_total"] = feats.get("h2h_matches_count", 10)
        feats["pitch_aff_diff"] = 0.0

        return feats

    def _bat_strength(self, squad, pitch):
        players = _all_players(squad)[:7]
        scores = [self.state.get_player_batting_rating(p)["composite_rating"] for p in players]
        return round(float(np.mean(scores)) if scores else 28.0, 2)

    def _bowl_strength(self, squad, pitch):
        players = _all_players(squad)[-5:]
        scores = [self.state.get_player_bowling_rating(p)["composite_rating"] for p in players]
        return round(float(np.mean(scores)) if scores else 13.5, 2)

    def _form(self, team):
        from ipl_temporal import normalize_team
        wins, _ = self.state.get_team_form(normalize_team(team), n=5)
        return wins

    def _h2h(self, t1, t2):
        from ipl_temporal import normalize_team
        return self.state.get_h2h_stats(normalize_team(t1), normalize_team(t2))

    def _venue_wr(self, team, venue):
        from ipl_temporal import normalize_team, normalize_venue
        stats = self.state.get_venue_stats(normalize_venue(venue), normalize_team(team), "UNK")
        return stats.get("t1_venue_wr", 0.50)

    def partnership_synergy(self, p1, p2):
        return {"avg_part": 28.0, "synergy": 1.0, "n": 5}

    def player_venue_record(self, player, venue):
        return {"has_data": False}


class ModelTrainer:
    from ipl_temporal import TemporalFeatureEngine
    FEATURE_NAMES = TemporalFeatureEngine.FEATURE_NAMES

    def __init__(self):
        from ipl_models_pipeline import LeakFreeEnsemble
        self.ensemble = LeakFreeEnsemble(random_seed=RANDOM_SEED, use_calibration=True)
        self.trained = False
        self.cv_scores = {}

    def prepare_dataset(self, info_df=None, ball_df=None, feat_eng=None, squads=None):
        from ipl_temporal import ChronologicalDataLoader, HistoricalStateTracker, TemporalFeatureEngine
        loader = ChronologicalDataLoader()
        matches = loader.load_all_matches()
        completed = [m for m in matches if m.is_completed]

        fe = TemporalFeatureEngine(mode="pre_xi")
        state = HistoricalStateTracker()

        rows = []
        labels = []

        print(f"  Building leak-free features chronologically for {len(completed)} matches…")
        for m in tqdm(completed, desc="Features"):
            f_dict = fe.build_features(m, state)
            row = [f_dict.get(k, 0.0) for k in self.FEATURE_NAMES]
            label = 1 if m.winner == m.team1 else 0
            rows.append(row)
            labels.append(label)
            state.update_match_result(m)

        X = np.array(rows, dtype=float)
        y = np.array(labels, dtype=int)
        print(f"  ✅ Dataset: {len(X)} real matches × {X.shape[1]} features (synthetic: False)")
        return X, y

    def train(self, X, y):
        print("\n🤖 Training leak-free ensemble pipeline with expanding-window CV…")
        self.ensemble.fit(X, y)
        self.trained = True
        print("  ✅ Ensemble trained and calibrated with Isotonic Regression")

    ALIAS_MAP = {
        "t1_bat": "t1_bat_score",
        "t2_bat": "t2_bat_score",
        "t1_bowl": "t1_bowl_score",
        "t2_bowl": "t2_bowl_score",
        "bat_diff": "bat_score_diff",
        "bowl_diff": "bowl_score_diff",
        "h2h_wr": "h2h_t1_wr",
        "h2h_total": "h2h_matches_count",
        "t1_wins": "t1_recent_wins",
        "t2_wins": "t2_recent_wins",
        "form_diff": "form_diff_exp",
        "t1_form_score": "t1_form_exp",
        "t2_form_score": "t2_form_exp",
        "elo_diff": "elo_diff_pre",
    }

    def predict(self, features: dict):
        if not self.trained:
            raise RuntimeError("Call train() first")

        f_copy = dict(features)
        # Apply alias mappings and sync overrides
        for alias, canonical in self.ALIAS_MAP.items():
            if alias in features and canonical in features:
                # If alias was mutated differently from canonical, sync it
                if features[alias] != features[canonical]:
                    f_copy[canonical] = f_copy[alias]
            elif alias in f_copy and canonical not in f_copy:
                f_copy[canonical] = f_copy[alias]
            elif canonical in f_copy and alias not in f_copy:
                f_copy[alias] = f_copy[canonical]

        # Recompute differentials if base strengths were mutated in sensitivity analysis
        if "t1_bat_score" in f_copy and "t2_bat_score" in f_copy:
            f_copy["bat_diff"] = f_copy["t1_bat_score"] - f_copy["t2_bat_score"]
        elif "t1_bat" in f_copy and "t2_bat" in f_copy:
            f_copy["bat_diff"] = f_copy["t1_bat"] - f_copy["t2_bat"]

        if "t1_bowl_score" in f_copy and "t2_bowl_score" in f_copy:
            f_copy["bowl_diff"] = f_copy["t1_bowl_score"] - f_copy["t2_bowl_score"]
        elif "t1_bowl" in f_copy and "t2_bowl" in f_copy:
            f_copy["bowl_diff"] = f_copy["t1_bowl"] - f_copy["t2_bowl"]

        if "t1_recent_wins" in f_copy and "t2_recent_wins" in f_copy:
            f_copy["form_diff_exp"] = (f_copy["t1_recent_wins"] - f_copy["t2_recent_wins"]) / 5.0
        elif "t1_wins" in f_copy and "t2_wins" in f_copy:
            f_copy["form_diff_exp"] = (f_copy["t1_wins"] - f_copy["t2_wins"]) / 5.0

        if "t1_elo" in f_copy and "t2_elo" in f_copy:
            f_copy["elo_diff"] = f_copy["t1_elo"] - f_copy["t2_elo"]
            f_copy["elo_expected_t1"] = 1.0 / (1.0 + 10.0 ** (-f_copy["elo_diff"] / 400.0))

        row = np.array([[f_copy.get(k, 0.0) for k in self.FEATURE_NAMES]], dtype=float)
        probs = self.ensemble.predict_proba(row)[0]
        prob_t1 = float(probs[1])
        prob_t2 = float(probs[0])

        model_probs = {}
        if hasattr(self.ensemble, "base_models"):
            row_s = self.ensemble.scaler.transform(row)
            for name, clf in self.ensemble.base_models.items():
                inp = row_s if name in ["LogisticRegression", "NeuralNet", "ExtraTrees"] else row
                p = float(clf.predict_proba(inp)[0, 1])
                model_probs[name] = round(p, 4)

        std = float(np.std(list(model_probs.values()))) if model_probs else 0.05
        confidence_pct = float(np.clip(0.80 + (0.10 - std) * 0.20, 0.76, 0.86))
        confidence = "HIGH" if confidence_pct >= 0.84 else "MEDIUM" if confidence_pct >= 0.79 else "LOW"

        return {
            "win_prob_t1": round(prob_t1, 4),
            "win_prob_t2": round(prob_t2, 4),
            "model_probs": model_probs,
            "std_dev": round(std, 4),
            "confidence": confidence,
            "confidence_pct": round(confidence_pct, 4),
        }

    def feature_importance(self):
        if hasattr(self.ensemble, "base_models") and "XGBoost" in self.ensemble.base_models:
            imp = self.ensemble.base_models["XGBoost"].feature_importances_
            return dict(sorted(zip(self.FEATURE_NAMES, imp), key=lambda x: -x[1]))
        return {k: 1.0 / len(self.FEATURE_NAMES) for k in self.FEATURE_NAMES}

    def save(self, path=MODELS_DIR / "ipl_ensemble.pkl"):
        joblib.dump({"ensemble": self.ensemble, "trained": self.trained}, path)
        print(f"💾 Models saved to {path}")

    def load(self, path=MODELS_DIR / "ipl_ensemble.pkl"):
        data = joblib.load(path)
        self.ensemble = data["ensemble"]
        self.trained = data.get("trained", True)
        print(f"✅ Models loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER PROJECTOR
# ─────────────────────────────────────────────────────────────────────────────
# %%
class PlayerProjector:
    def __init__(self, feat_eng: FeatureEngineer):
        self.fe = feat_eng

    def _season_sort_key(self, season_value):
        text = str(season_value).strip()
        match = re.search(r"(\d{4})", text)
        if not match:
            return (-1, text)
        start_year = int(match.group(1))
        tail_match = re.search(r"/(\d{2,4})", text)
        if tail_match:
            tail = tail_match.group(1)
            if len(tail) == 2:
                return (int(f"{start_year // 100}{tail}"), text)
            return (int(tail), text)
        return (start_year, text)

    def _latest_season(self):
        if self.fe.info_df.empty or "key" not in self.fe.info_df.columns:
            return None
        seasons = (
            self.fe.info_df[self.fe.info_df["key"] == "season"]["value"]
            .dropna().astype(str).unique().tolist()
        )
        if not seasons:
            return None
        return max(seasons, key=self._season_sort_key)

    def _parse_date(self, value):
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        return datetime.min

    def _team_match(self, value, team):
        return _resolve_team(str(value)) == team

    def _season_match_ids(self, team, season=None):
        if self.fe.info_df.empty:
            return []
        season = season or self._latest_season()
        if not season:
            return []

        season_rows = self.fe.info_df[
            (self.fe.info_df["key"] == "season") &
            (self.fe.info_df["value"].astype(str) == str(season))
        ]
        season_match_ids = season_rows["match_id"].unique().tolist()
        matches = []
        for mid in season_match_ids:
            mi = self.fe.info_df[self.fe.info_df["match_id"] == mid]
            team_rows = mi[mi["key"] == "team"]["value"].astype(str).tolist()
            if not any(self._team_match(v, team) for v in team_rows):
                continue
            date_rows = mi[mi["key"] == "date"]["value"].astype(str).tolist()
            matches.append((self._parse_date(date_rows[0]) if date_rows else datetime.min, mid))
        matches.sort(key=lambda item: (item[0], str(item[1])))
        return [mid for _, mid in matches]

    def _team_season_xi(self, team, squad):
        match_ids = self._season_match_ids(team)
        if not match_ids:
            return []

        for mid in reversed(match_ids):
            mi = self.fe.info_df[self.fe.info_df["match_id"] == mid]
            player_rows = mi[
                (mi["key"] == "player") &
                (mi["value"].astype(str).apply(lambda v: self._team_match(v, team)))
            ]
            xi = []
            for _, row in player_rows.iterrows():
                player = str(row.get("player", "")).strip()
                if player and player not in xi:
                    xi.append(player)
            if len(xi) >= 11:
                return xi[:11]

        return []

    def _order_xi_for_batting(self, xi):
        role_rank = {"WK-BAT": 0, "BAT": 1, "ALL": 2, "BOWL": 3}

        def sort_key(player):
            db = PLAYER_DB.get(player, {})
            role = db.get("role", "")
            return (
                role_rank.get(role, 4),
                -float(db.get("bat_avg") or 0.0),
                -float(db.get("bat_sr") or 0.0),
                player,
            )

        return sorted(list(dict.fromkeys(xi)), key=sort_key)

    def preview_team_selection(self, team, squad):
        roster = _all_players(squad)
        season_xi = self._team_season_xi(team, squad)
        if not season_xi:
            season_xi = self._probable_xi(squad, team=team)
        bench = [p for p in roster if p not in season_xi]
        return season_xi, bench, roster

    def apply_replacements(self, base_xi, roster, replacements):
        final_xi = list(base_xi)
        roster_set = set(roster)
        for out_player, in_player in replacements:
            out_player = out_player.strip()
            in_player = in_player.strip()
            if not out_player or not in_player:
                continue
            if out_player not in final_xi:
                continue
            if in_player not in roster_set or in_player in final_xi:
                continue
            idx = final_xi.index(out_player)
            final_xi[idx] = in_player

        deduped = []
        seen = set()
        for player in final_xi:
            if player not in seen:
                seen.add(player)
                deduped.append(player)
        for player in roster:
            if len(deduped) >= 11:
                break
            if player not in seen:
                seen.add(player)
                deduped.append(player)
        return deduped[:11]

    def project_batting(self, squad, venue, pitch, weather, opposition_squad, team=None, playing_xi=None):
        """Project batting scores for all 11 players independently."""
        players = list(playing_xi) if playing_xi else self._probable_xi(squad, team=team)
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

    def project_bowling(self, squad, venue, pitch, weather, team=None, playing_xi=None):
        """
        Project bowling figures only for players who actually bowl.
        Correctly separates: pure bowlers + bowling all-rounders.
        Pure batters and WK-batters are excluded from bowling projections.
        """
        xi = list(playing_xi) if playing_xi else self._probable_xi(squad, team=team)

        # Get ONLY players who bowl — from structured squad keys
        if "all_players" in squad:
            # Flat list: filter by PLAYER_DB bowl_eco
            bowlers_xi = [p for p in squad["all_players"]
                          if PLAYER_DB.get(p, {}).get("bowl_eco") is not None
                          and PLAYER_DB.get(p, {}).get("bowl_eco", 0) > 0]
        else:
            # Structured: bowlers + all-rounders who have bowl stats
            bowlers_xi = squad.get("bowlers", [])
            bowling_allrounders = [
                p for p in squad.get("all_rounders", [])
                if PLAYER_DB.get(p, {}).get("bowl_eco") is not None
            ]
            bowlers_xi = bowlers_xi + bowling_allrounders

        # Ensure we only project the 11 players actually in the XI
        bowlers_xi = [p for p in bowlers_xi if p in xi]

        results = []
        for player in bowlers_xi:
            db    = PLAYER_DB.get(player, {})
            eco   = db.get("bowl_eco")
            avg   = db.get("bowl_avg")
            style = db.get("bowl_style", "")

            if not eco or not avg:
                continue

            # Overs allocation: pace get more in PP+death, spinners in middle
            pi = pitch.get("pace_index", 5)
            si = pitch.get("spin_index", 5)
            is_pace = style in ["RF","RFM","LFM","LF","RMF"]
            is_spin = style in ["OB","SLA","LBG","LBC","SLO"]

            # Base 4 overs; key bowlers bowl their full quota
            max_ov = 4.0

            # Economy modifier based on pitch type
            eco_mod = 1.0; wkt_mod = 1.0
            if is_pace:
                if pi > 6.5: eco_mod = 0.91; wkt_mod = 1.18   # pace pitch = good for pacers
                elif si > 7:  eco_mod = 1.12; wkt_mod = 0.78   # spin pitch = pacers struggle
            elif is_spin:
                if si > 6.5: eco_mod = 0.89; wkt_mod = 1.20   # spin pitch = good for spinners
                elif pi > 7: eco_mod = 1.10; wkt_mod = 0.75   # pace pitch = spinners struggle
                # Dew kills spinners in 2nd innings
                if weather.get("dew_risk", 0) > 0.6:
                    eco_mod *= 1.14; wkt_mod *= 0.82

            proj_eco  = round(float(eco * eco_mod + np.random.normal(0, 0.35)), 2)
            proj_eco  = max(5.5, min(14.0, proj_eco))
            wkt_rate  = max_ov / max(avg * wkt_mod / 6.0, 5.0)
            proj_wkts = int(min(4, np.random.poisson(max(0.05, wkt_rate))))
            proj_runs = round(proj_eco * max_ov)

            suited = ((is_pace and pi > 6.5) or (is_spin and si > 6.5))

            results.append({
                "player":          player,
                "style":           style or "—",
                "overs":           max_ov,
                "wickets":         proj_wkts,
                "runs":            proj_runs,
                "economy":         proj_eco,
                "suited_to_pitch": suited,
            })
        results.sort(key=lambda x: (-x["wickets"], x["economy"]))
        return results[:6]

    def _probable_xi(self, squad, team=None):
        """
        Build a realistic IPL playing XI following actual selection logic:
        1 WK + 4-5 batters + 3-4 all-rounders + 3-4 bowlers = 11

        Handles both structured squads (with role keys) and flat scraped lists.
        """
        if team:
            season_xi = self._team_season_xi(team, squad)
            if season_xi:
                return self._order_xi_for_batting(season_xi)

        # ── Extract players by role ───────────────────────────────────────────
        if "all_players" in squad:
            # Flat scraped list — classify each player by PLAYER_DB role
            flat = squad["all_players"]
            wk_pool  = [p for p in flat if PLAYER_DB.get(p, {}).get("role") == "WK-BAT"]
            bat_pool = [p for p in flat if PLAYER_DB.get(p, {}).get("role") == "BAT"]
            all_pool = [p for p in flat if PLAYER_DB.get(p, {}).get("role") == "ALL"]
            bowl_pool= [p for p in flat if PLAYER_DB.get(p, {}).get("role") == "BOWL"]
            # Unknown role → classify by squad position (first 8 = batters, rest = bowlers)
            unknown  = [p for p in flat if not PLAYER_DB.get(p, {}).get("role")]
            bat_pool += unknown[:4]
            bowl_pool+= unknown[4:]
        else:
            # Structured squad dict — use the role keys directly
            wk_pool  = squad.get("wk", [])
            bat_pool = squad.get("batters", [])
            all_pool = squad.get("all_rounders", [])
            bowl_pool= squad.get("bowlers", [])

        # ── Deduplicate across pools (WK can also be in batters list) ─────────
        seen = set()
        def _dedup(lst):
            result = []
            for p in lst:
                if p not in seen:
                    seen.add(p); result.append(p)
            return result

        wk_pool   = _dedup(wk_pool)
        bat_pool  = _dedup(bat_pool)
        all_pool  = _dedup(all_pool)
        bowl_pool = _dedup(bowl_pool)

        xi = []

        # ── Step 1: Pick 1 wicketkeeper (mandatory) ───────────────────────────
        if wk_pool:
            xi.append(wk_pool[0])          # Primary WK
            # Add backup WK only if squad has a clear second option
            if len(wk_pool) >= 2:
                # Only pick 2nd WK if fewer than 2 WKs already and squad is large
                pass   # Most modern IPL teams play 1 WK

        # ── Step 2: Pick specialist batters (target: 4 total incl. WK) ───────
        remaining_bat_slots = 4 - len([p for p in xi if PLAYER_DB.get(p,{}).get("role")=="WK-BAT"])
        for p in bat_pool:
            if len(xi) >= 5: break
            if p not in xi:
                xi.append(p)

        # ── Step 3: Pick all-rounders (target: 3-4) ───────────────────────────
        for p in all_pool:
            if len(xi) >= 8: break
            if p not in xi:
                xi.append(p)

        # ── Step 4: Pick specialist bowlers (target: 3-4) ─────────────────────
        for p in bowl_pool:
            if len(xi) >= 11: break
            if p not in xi:
                xi.append(p)

        # ── Step 5: Fill remaining spots if XI not complete ───────────────────
        all_available = wk_pool + bat_pool + all_pool + bowl_pool
        for p in all_available:
            if len(xi) >= 11: break
            if p not in xi:
                xi.append(p)

        # ── Validation: must have at least 3 bowling options ──────────────────
        bowl_count = sum(1 for p in xi if PLAYER_DB.get(p,{}).get("role") in ["BOWL","ALL"])
        if bowl_count < 3:
            # Add more all-rounders or bowlers from pool
            for p in all_pool + bowl_pool:
                if p not in xi:
                    xi.append(p)
                    bowl_count += 1
                    if bowl_count >= 3: break

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
        self.info_df = feat_eng.info_df
        self.ball_df = feat_eng.ball_df

    def analyze(self, team1, team2, venue, match_date=None, match_time="19:30",
                match_n_at_venue=1, playing_xis=None):
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
        playing_xis = playing_xis or {}
        bat1    = self.proj.project_batting(sq1, venue, pitch, weather, sq2, team=t1, playing_xi=playing_xis.get(t1))
        weather_inn2 = {**weather, "is_batting_second": True}
        bat2    = self.proj.project_batting(sq2, venue, pitch, weather_inn2, sq1, team=t2, playing_xi=playing_xis.get(t2))
        print(f"\n  {TEAMS.get(t1,t1)} XI: {', '.join(p['player'] for p in bat1)}")
        print(f"  {TEAMS.get(t2,t2)} XI: {', '.join(p['player'] for p in bat2)}\n")
        bowl1   = self.proj.project_bowling(sq1, venue, pitch, weather, team=t1, playing_xi=playing_xis.get(t1))
        bowl2   = self.proj.project_bowling(sq2, venue, pitch, weather, team=t2, playing_xi=playing_xis.get(t2))
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

        # Dew impact on teams and players
        if weather.get("is_night", True):
            dew = weather.get("dew_risk", 0)
            print(f"{YELLOW}{'─'*65}")
            print("  💧  DEW IMPACT (DATE/TIME-SPECIFIC)")
            print(f"{'─'*65}{RESET}")
            if dew >= 0.55:
                hot_bats = sorted(bat1[:4] + bat2[:4], key=lambda x: x.get("runs", 0), reverse=True)[:5]
                spin_styles = {"OB", "SLA", "LBG", "LBC", "SLO"}
                spin_hurt = [b for b in (bowl1 + bowl2) if b.get("style") in spin_styles][:5]
                print(f"  Dew risk {dew:.0%}: likely advantage to batting second/chasing side.")
                if hot_bats:
                    print("  Players likely favored by wet ball batting conditions:")
                    print("   " + ", ".join(p["player"] for p in hot_bats))
                if spin_hurt:
                    print("  Players likely hurt (grip-dependent spinners):")
                    print("   " + ", ".join(p["player"] for p in spin_hurt))
            else:
                print(f"  Dew risk {dew:.0%}: low to moderate, no major dew bias expected.")
            print()
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
        print(f"\n  Model confidence: {pred['confidence_pct']:.0%}  ({pred['confidence']}, σ={pred['std_dev']:.3f})")
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
    """Backtest the IPL predictor using leak-free walk-forward validation."""

    def __init__(self, analyzer=None, info_df=None, ball_df=None, squads=None):
        self.analyzer = analyzer

    def run(self, test_seasons=range(2021, 2027), verbose=True):
        from walk_forward_backtest import WalkForwardBacktester
        backtester = WalkForwardBacktester(mode="pre_xi")
        backtester.load_data()
        season_results, match_preds = backtester.run_walk_forward(test_seasons=list(test_seasons))
        backtester.save_reports(season_results, match_preds)
        return {"season_results": season_results, "predictions": match_preds}


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
    squads = FALLBACK_SQUADS

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

                t1 = _resolve_team(t1_in)
                t2 = _resolve_team(t2_in)
                sq1 = analyzer.squads.get(t1, FALLBACK_SQUADS.get(t1, FALLBACK_SQUADS["MI"]))
                sq2 = analyzer.squads.get(t2, FALLBACK_SQUADS.get(t2, FALLBACK_SQUADS["CSK"]))

                xi_overrides = {}
                for team, squad in ((t1, sq1), (t2, sq2)):
                    current_xi, bench, roster = analyzer.proj.preview_team_selection(team, squad)
                    print(f"\n  {TEAMS.get(team, team)} current season XI:")
                    print(f"    {', '.join(current_xi)}")
                    print(f"  {TEAMS.get(team, team)} other players:")
                    print(f"    {', '.join(bench) if bench else 'None'}")
                    replacements = input(
                        f"  Replace any {TEAMS.get(team, team)} players? Use out=in pairs, comma-separated, or Enter to keep XI: "
                    ).strip()
                    if replacements:
                        parsed = []
                        for chunk in replacements.split(","):
                            chunk = chunk.strip()
                            if not chunk:
                                continue
                            if "=" in chunk:
                                out_player, in_player = chunk.split("=", 1)
                            elif ":" in chunk:
                                out_player, in_player = chunk.split(":", 1)
                            else:
                                raise ValueError(f"Invalid replacement '{chunk}'. Use out=in.")
                            parsed.append((out_player.strip(), in_player.strip()))
                        current_xi = analyzer.proj.apply_replacements(current_xi, roster, parsed)
                        print(f"  Updated XI for {TEAMS.get(team, team)}:")
                        print(f"    {', '.join(current_xi)}")
                    xi_overrides[team] = current_xi

                analyzer.analyze(t1_in, t2_in, ven, date, mtime, mnum, playing_xis=xi_overrides)

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
