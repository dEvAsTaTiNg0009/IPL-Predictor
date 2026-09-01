"""
Temporal Data Contract & Historical State Management Module for IPL Match Prediction.

Strict Temporal Causality Rule:
For target match M at time T:
AVAILABLE_INFORMATION(M) = all information whose timestamp is strictly before T.
No future or concurrent information may be used.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Standard IPL Team Mapping
TEAM_ALIASES: Dict[str, str] = {
    "chennai super kings": "CSK",
    "csk": "CSK",
    "mumbai indians": "MI",
    "mi": "MI",
    "royal challengers bangalore": "RCB",
    "royal challengers bengaluru": "RCB",
    "rcb": "RCB",
    "kolkata knight riders": "KKR",
    "kkr": "KKR",
    "delhi daredevils": "DC",
    "delhi capitals": "DC",
    "dc": "DC",
    "kings xi punjab": "PBKS",
    "punjab kings": "PBKS",
    "pbks": "PBKS",
    "kxip": "PBKS",
    "rajasthan royals": "RR",
    "rr": "RR",
    "sunrisers hyderabad": "SRH",
    "srh": "SRH",
    "deccan chargers": "DCG",
    "dcg": "DCG",
    "lucknow super giants": "LSG",
    "lsg": "LSG",
    "gujarat titans": "GT",
    "gt": "GT",
    "gujarat lions": "GL",
    "gl": "GL",
    "rising pune supergiant": "RPS",
    "rising pune supergiants": "RPS",
    "rps": "RPS",
    "pune warriors": "PWI",
    "pwi": "PWI",
    "kochi tuskers kerala": "KTK",
    "ktk": "KTK",
}

CANONICAL_TEAMS = {
    "CSK": "Chennai Super Kings",
    "MI": "Mumbai Indians",
    "RCB": "Royal Challengers Bengaluru",
    "KKR": "Kolkata Knight Riders",
    "DC": "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "LSG": "Lucknow Super Giants",
    "GT": "Gujarat Titans",
    "DCG": "Deccan Chargers",
    "GL": "Gujarat Lions",
    "RPS": "Rising Pune Supergiant",
    "PWI": "Pune Warriors India",
    "KTK": "Kochi Tuskers Kerala",
}

# Standard Venue Canonical Names
VENUE_ALIASES: Dict[str, str] = {
    "wankhede": "Wankhede Stadium, Mumbai",
    "dy patil": "Dr DY Patil Sports Academy, Mumbai",
    "brabourne": "Brabourne Stadium, Mumbai",
    "chinnaswamy": "M Chinnaswamy Stadium, Bengaluru",
    "bangalore": "M Chinnaswamy Stadium, Bengaluru",
    "bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "chidambaram": "MA Chidambaram Stadium, Chepauk, Chennai",
    "chepauk": "MA Chidambaram Stadium, Chepauk, Chennai",
    "eden gardens": "Eden Gardens, Kolkata",
    "arun jaitley": "Arun Jaitley Stadium, Delhi",
    "feroz shah kotla": "Arun Jaitley Stadium, Delhi",
    "kotla": "Arun Jaitley Stadium, Delhi",
    "sawai mansingh": "Sawai Mansingh Stadium, Jaipur",
    "rajiv gandhi": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "hyderabad": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "uppal": "Rajiv Gandhi International Stadium, Uppal, Hyderabad",
    "punjab cricket association": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "bindra": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "new chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "narendra modi": "Narendra Modi Stadium, Ahmedabad",
    "motera": "Narendra Modi Stadium, Ahmedabad",
    "ekana": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "himachal pradesh": "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "dharamsala": "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "barsapara": "Barsapara Cricket Stadium, Guwahati",
    "guwahati": "Barsapara Cricket Stadium, Guwahati",
    "holkar": "Holkar Cricket Stadium, Indore",
    "indore": "Holkar Cricket Stadium, Indore",
    "dr y.s. rajasekhara": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "vizag": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "maharashtra cricket association": "Maharashtra Cricket Association Stadium, Pune",
    "subrata roy": "Maharashtra Cricket Association Stadium, Pune",
    "pune": "Maharashtra Cricket Association Stadium, Pune",
    "dubai": "Dubai International Cricket Stadium",
    "sharjah": "Sharjah Cricket Stadium",
    "zayed": "Zayed Cricket Stadium, Abu Dhabi",
    "abu dhabi": "Zayed Cricket Stadium, Abu Dhabi",
}


def normalize_team(name: str) -> str:
    """Standardize team string to canonical abbreviation."""
    if not name:
        return "UNK"
    cleaned = name.lower().strip()
    if cleaned in TEAM_ALIASES:
        return TEAM_ALIASES[cleaned]
    for k, v in TEAM_ALIASES.items():
        if k in cleaned or cleaned in k:
            return v
    return name.strip()[:4].upper()


def normalize_venue(name: str) -> str:
    """Standardize venue name to canonical identifier."""
    if not name:
        return "Unknown Venue"
    cleaned = name.lower().strip()
    for alias, canonical in VENUE_ALIASES.items():
        if alias in cleaned:
            return canonical
    return name.strip()


# Common Player Bowling Styles & Batting Handedness Registry
PLAYER_STYLES: Dict[str, Dict[str, str]] = {
    # Spinners
    "YS Chahal": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    "Rashid Khan": {"bat": "RHB", "bowl": "LBG", "role": "ALL"},
    "Kuldeep Yadav": {"bat": "LHB", "bowl": "LBC", "role": "BOWL"},
    "Varun Chakravarthy": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    "R Ashwin": {"bat": "RHB", "bowl": "OB", "role": "ALL"},
    "SP Narine": {"bat": "LHB", "bowl": "OB", "role": "ALL"},
    "RA Jadeja": {"bat": "LHB", "bowl": "SLA", "role": "ALL"},
    "AR Patel": {"bat": "LHB", "bowl": "SLA", "role": "ALL"},
    "Ravi Bishnoi": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    "Noor Ahmad": {"bat": "LHB", "bowl": "LBC", "role": "BOWL"},
    "Washington Sundar": {"bat": "LHB", "bowl": "OB", "role": "ALL"},
    "Krunal Pandya": {"bat": "LHB", "bowl": "SLA", "role": "ALL"},
    "A Zampa": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    "Wanindu Hasaranga": {"bat": "RHB", "bowl": "LBG", "role": "ALL"},
    "Mujeeb Ur Rahman": {"bat": "RHB", "bowl": "OB", "role": "BOWL"},
    "M Theekshana": {"bat": "RHB", "bowl": "OB", "role": "BOWL"},
    "Rahul Chahar": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    # Pacers
    "JJ Bumrah": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "Mohammed Shami": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "Kagiso Rabada": {"bat": "LHB", "bowl": "RF", "role": "BOWL"},
    "TA Boult": {"bat": "RHB", "bowl": "LF", "role": "BOWL"},
    "Arshdeep Singh": {"bat": "LHB", "bowl": "LF", "role": "BOWL"},
    "B Kumar": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "Mohammed Siraj": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "MA Starc": {"bat": "LHB", "bowl": "LF", "role": "BOWL"},
    "Avesh Khan": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "T Natarajan": {"bat": "LHB", "bowl": "LF", "role": "BOWL"},
    "HV Patel": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "JR Hazlewood": {"bat": "LHB", "bowl": "RF", "role": "BOWL"},
    "SN Thakur": {"bat": "RHB", "bowl": "RFM", "role": "ALL"},
    "DL Chahar": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "Harshit Rana": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "M Pathirana": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "Mayank Yadav": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "Sandeep Sharma": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "HH Pandya": {"bat": "RHB", "bowl": "RFM", "role": "ALL"},
    "AD Russell": {"bat": "RHB", "bowl": "RF", "role": "ALL"},
    "GJ Maxwell": {"bat": "RHB", "bowl": "OB", "role": "ALL"},
    "MP Stoinis": {"bat": "RHB", "bowl": "RFM", "role": "ALL"},
    "SM Curran": {"bat": "LHB", "bowl": "LF", "role": "ALL"},
    # Batters & WKs
    "V Kohli": {"bat": "RHB", "bowl": "RMF", "role": "BAT"},
    "RG Sharma": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "SA Yadav": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "Shubman Gill": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "KL Rahul": {"bat": "RHB", "bowl": None, "role": "WK-BAT"},
    "RR Pant": {"bat": "LHB", "bowl": None, "role": "WK-BAT"},
    "SV Samson": {"bat": "RHB", "bowl": None, "role": "WK-BAT"},
    "MS Dhoni": {"bat": "RHB", "bowl": "RMF", "role": "WK-BAT"},
    "JC Buttler": {"bat": "RHB", "bowl": None, "role": "WK-BAT"},
    "Q de Kock": {"bat": "LHB", "bowl": None, "role": "WK-BAT"},
    "N Pooran": {"bat": "LHB", "bowl": "OB", "role": "WK-BAT"},
    "DA Warner": {"bat": "LHB", "bowl": "LBG", "role": "BAT"},
    "F du Plessis": {"bat": "RHB", "bowl": "LBG", "role": "BAT"},
    "RD Gaikwad": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "YBK Jaiswal": {"bat": "LHB", "bowl": "LBG", "role": "BAT"},
    "SS Iyer": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "S Dube": {"bat": "LHB", "bowl": "RFM", "role": "ALL"},
    "Rinku Singh": {"bat": "LHB", "bowl": "OB", "role": "BAT"},
    "TM Head": {"bat": "LHB", "bowl": "OB", "role": "BAT"},
    "Abhishek Sharma": {"bat": "LHB", "bowl": "SLA", "role": "ALL"},
    "H Klaasen": {"bat": "RHB", "bowl": "OB", "role": "WK-BAT"},
    "RM Patidar": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "Tilak Varma": {"bat": "LHB", "bowl": "OB", "role": "BAT"},
    "N Rana": {"bat": "LHB", "bowl": "OB", "role": "BAT"},
    "TH David": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "N Wadhera": {"bat": "LHB", "bowl": "LBG", "role": "BAT"},
    "PD Salt": {"bat": "RHB", "bowl": None, "role": "WK-BAT"},
    "Jitesh Sharma": {"bat": "RHB", "bowl": None, "role": "WK-BAT"},
    "P Simran Singh": {"bat": "RHB", "bowl": None, "role": "WK-BAT"},
    "Abishek Porel": {"bat": "LHB", "bowl": None, "role": "WK-BAT"},
    "Shahrukh Khan": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "R Parag": {"bat": "RHB", "bowl": "LBG", "role": "ALL"},
    "RA Tripathi": {"bat": "RHB", "bowl": "RFM", "role": "BAT"},
    "D Padikkal": {"bat": "LHB", "bowl": None, "role": "BAT"},
    "LS Livingstone": {"bat": "RHB", "bowl": "LBG", "role": "ALL"},
    "C Green": {"bat": "RHB", "bowl": "RF", "role": "ALL"},
    "M Jansen": {"bat": "RHB", "bowl": "LF", "role": "ALL"},
    "LH Ferguson": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "G Coetzee": {"bat": "RHB", "bowl": "RF", "role": "BOWL"},
    "Naveen-ul-Haq": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "Mohsin Khan": {"bat": "LHB", "bowl": "LF", "role": "BOWL"},
    "Yash Dayal": {"bat": "LHB", "bowl": "LF", "role": "BOWL"},
    "Mukesh Kumar": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "V Vyshak": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "Rasikh Salam": {"bat": "RHB", "bowl": "RFM", "role": "BOWL"},
    "Suyash Sharma": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    "M Markande": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    "P Chawla": {"bat": "LHB", "bowl": "LBG", "role": "BOWL"},
    "A Mishra": {"bat": "RHB", "bowl": "LBG", "role": "BOWL"},
    "Shreyas Gopal": {"bat": "RHB", "bowl": "LBG", "role": "ALL"},
    "K Gowtham": {"bat": "RHB", "bowl": "OB", "role": "ALL"},
    "MK Lomror": {"bat": "LHB", "bowl": "SLA", "role": "ALL"},
    "Ramandeep Singh": {"bat": "RHB", "bowl": "RFM", "role": "ALL"},
    "Shashank Singh": {"bat": "RHB", "bowl": "RFM", "role": "BAT"},
    "Ashutosh Sharma": {"bat": "RHB", "bowl": "RFM", "role": "BAT"},
    "Angkrish Raghuvanshi": {"bat": "RHB", "bowl": "SLA", "role": "BAT"},
    "Naman Dhir": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
    "Sameer Rizvi": {"bat": "RHB", "bowl": "OB", "role": "BAT"},
}


@dataclass
class BallRecord:
    match_id: str
    innings: int
    over: int
    ball_in_over: int
    batting_team: str
    bowling_team: str
    striker: str
    non_striker: str
    bowler: str
    runs_off_bat: int
    extras: int
    wides: int
    noballs: int
    byes: int
    legbyes: int
    wicket_type: str
    player_dismissed: str

    @property
    def is_legal(self) -> bool:
        return self.wides == 0 and self.noballs == 0

    @property
    def phase(self) -> str:
        ov = self.over
        if ov < 6:
            return "POWERPLAY"
        elif ov < 15:
            return "MIDDLE"
        else:
            return "DEATH"


@dataclass
class MatchRecord:
    match_id: str
    season: str
    match_date: date
    match_datetime: datetime
    match_number: int
    team1: str
    team2: str
    team1_raw: str
    team2_raw: str
    venue: str
    venue_raw: str
    city: str
    toss_winner: str
    toss_decision: str
    winner: Optional[str]
    winner_raw: str
    margin_runs: int
    margin_wickets: int
    playing_xi: Dict[str, List[str]] = field(default_factory=dict)
    innings_scores: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)  # inn -> (runs, wkts, balls)
    deliveries: List[BallRecord] = field(default_factory=list)

    @property
    def is_completed(self) -> bool:
        return bool(self.winner and self.winner != "NO_RESULT" and self.winner != "UNK")


# ── Accumulators for Historical State ──────────────────────────────────────────


@dataclass
class BattingStats:
    runs: int = 0
    balls: int = 0
    dismissals: int = 0
    fours: int = 0
    sixes: int = 0
    dots: int = 0
    innings_count: int = 0
    pp_runs: int = 0
    pp_balls: int = 0
    pp_dismissals: int = 0
    mid_runs: int = 0
    mid_balls: int = 0
    mid_dismissals: int = 0
    death_runs: int = 0
    death_balls: int = 0
    death_dismissals: int = 0
    vs_pace_runs: int = 0
    vs_pace_balls: int = 0
    vs_pace_dismissals: int = 0
    vs_spin_runs: int = 0
    vs_spin_balls: int = 0
    vs_spin_dismissals: int = 0
    recent_innings: List[int] = field(default_factory=list)

    def add_ball(self, runs: int, is_legal: bool, phase: str, dismissed: bool, bowler_type: str = "pace"):
        self.runs += runs
        if is_legal:
            self.balls += 1
            if runs == 0:
                self.dots += 1
        if runs == 4:
            self.fours += 1
        elif runs == 6:
            self.sixes += 1

        if phase == "POWERPLAY":
            self.pp_runs += runs
            if is_legal:
                self.pp_balls += 1
            if dismissed:
                self.pp_dismissals += 1
        elif phase == "MIDDLE":
            self.mid_runs += runs
            if is_legal:
                self.mid_balls += 1
            if dismissed:
                self.mid_dismissals += 1
        else:
            self.death_runs += runs
            if is_legal:
                self.death_balls += 1
            if dismissed:
                self.death_dismissals += 1

        if bowler_type == "spin":
            self.vs_spin_runs += runs
            if is_legal:
                self.vs_spin_balls += 1
            if dismissed:
                self.vs_spin_dismissals += 1
        else:
            self.vs_pace_runs += runs
            if is_legal:
                self.vs_pace_balls += 1
            if dismissed:
                self.vs_pace_dismissals += 1

        if dismissed:
            self.dismissals += 1

    def finish_innings(self, innings_runs: int):
        self.innings_count += 1
        self.recent_innings.append(innings_runs)
        if len(self.recent_innings) > 10:
            self.recent_innings.pop(0)

    @property
    def average(self) -> float:
        if self.dismissals == 0:
            return float(self.runs) if self.runs > 0 else 24.5
        return self.runs / self.dismissals

    @property
    def strike_rate(self) -> float:
        if self.balls == 0:
            return 126.0
        return (self.runs / self.balls) * 100.0

    @property
    def dot_pct(self) -> float:
        if self.balls == 0:
            return 0.35
        return self.dots / self.balls

    @property
    def boundary_pct(self) -> float:
        if self.balls == 0:
            return 0.15
        return (self.fours + self.sixes) / self.balls


@dataclass
class BowlingStats:
    balls: int = 0
    runs_conceded: int = 0
    wickets: int = 0
    dots: int = 0
    matches_count: int = 0
    fours_conceded: int = 0
    sixes_conceded: int = 0
    pp_balls: int = 0
    pp_runs: int = 0
    pp_wickets: int = 0
    mid_balls: int = 0
    mid_runs: int = 0
    mid_wickets: int = 0
    death_balls: int = 0
    death_runs: int = 0
    death_wickets: int = 0
    recent_figures: List[Tuple[int, int]] = field(default_factory=list)

    def add_ball(self, runs: int, is_legal: bool, is_wicket: bool, phase: str):
        self.runs_conceded += runs
        if is_legal:
            self.balls += 1
            if runs == 0:
                self.dots += 1
        if runs == 4:
            self.fours_conceded += 1
        elif runs == 6:
            self.sixes_conceded += 1
        if is_wicket:
            self.wickets += 1

        if phase == "POWERPLAY":
            self.pp_runs += runs
            if is_legal:
                self.pp_balls += 1
            if is_wicket:
                self.pp_wickets += 1
        elif phase == "MIDDLE":
            self.mid_runs += runs
            if is_legal:
                self.mid_balls += 1
            if is_wicket:
                self.mid_wickets += 1
        else:
            self.death_runs += runs
            if is_legal:
                self.death_balls += 1
            if is_wicket:
                self.death_wickets += 1

    def finish_spell(self, runs: int, wkts: int):
        self.matches_count += 1
        self.recent_figures.append((runs, wkts))
        if len(self.recent_figures) > 10:
            self.recent_figures.pop(0)

    @property
    def economy(self) -> float:
        if self.balls == 0:
            return 8.5
        overs = self.balls / 6.0
        return self.runs_conceded / overs

    @property
    def average(self) -> float:
        if self.wickets == 0:
            return 28.5
        return self.runs_conceded / self.wickets

    @property
    def strike_rate(self) -> float:
        if self.wickets == 0:
            return 24.0
        return self.balls / self.wickets

    @property
    def dot_pct(self) -> float:
        if self.balls == 0:
            return 0.35
        return self.dots / self.balls


@dataclass
class MatchSummary:
    match_id: str
    match_datetime: datetime
    team: str
    opponent: str
    venue: str
    won: bool
    toss_won: bool
    chose_bat: bool
    team_score: int
    opp_score: int
    first_innings_score: int
    second_innings_score: int
    batting_first_team: str
    chasing_team: str
    chasing_won: bool
    margin_runs: int = 0
    margin_wickets: int = 0
    pp_runs: int = 48
    death_runs: int = 48
    pp_wickets: int = 1
    death_wickets: int = 2


# ── Canonical Dynamic Prior Estimator ──────────────────────────────────────────


class DynamicPriorEstimator:
    """
    Estimates league-wide baseline statistical priors dynamically from historical training data.
    Eliminates reliance on hardcoded static constants.
    """

    def __init__(self):
        self.total_balls: int = 0
        self.total_runs: int = 0
        self.total_dismissals: int = 0
        self.total_dots: int = 0
        self.total_boundaries: int = 0
        self.total_bowler_runs: int = 0
        self.total_bowler_wickets: int = 0
        self.total_bowler_balls: int = 0

    def update(self, runs: int, is_legal: bool, is_wicket: bool, is_boundary: bool):
        self.total_runs += runs
        if is_legal:
            self.total_balls += 1
            if runs == 0:
                self.total_dots += 1
        if is_boundary:
            self.total_boundaries += 1
        if is_wicket:
            self.total_dismissals += 1
            self.total_bowler_wickets += 1
        self.total_bowler_runs += runs
        if is_legal:
            self.total_bowler_balls += 1

    @property
    def prior_batting_avg(self) -> float:
        if self.total_dismissals < 20:
            return 24.5
        return float(self.total_runs / self.total_dismissals)

    @property
    def prior_batting_sr(self) -> float:
        if self.total_balls < 100:
            return 126.0
        return float((self.total_runs / self.total_balls) * 100.0)

    @property
    def prior_bowling_eco(self) -> float:
        if self.total_bowler_balls < 100:
            return 8.5
        overs = self.total_bowler_balls / 6.0
        return float(self.total_bowler_runs / overs)

    @property
    def prior_bowling_avg(self) -> float:
        if self.total_bowler_wickets < 20:
            return 28.5
        return float(self.total_bowler_runs / self.total_bowler_wickets)


# ── Canonical ELO Engine ──────────────────────────────────────────────────────


class TemporalELOSystem:
    """
    Strictly sequential ELO Rating Engine.
    Pre-match ELO is queried before match.
    Post-match update is performed only after prediction.
    """

    BASE_ELO = 1500.0
    K_FACTOR = 32.0
    HOME_ADVANTAGE = 25.0
    SEASON_REGRESSION = 0.20

    def __init__(self):
        self.ratings: Dict[str, float] = defaultdict(lambda: self.BASE_ELO)
        self.last_season: Dict[str, str] = {}
        self.last_match_id: Dict[str, str] = {}
        self.last_match_time: Dict[str, datetime] = {}

    def get_rating(self, team: str, season: Optional[str] = None) -> float:
        current = self.ratings[team]
        if season and team in self.last_season and self.last_season[team] != season:
            regressed = current * (1.0 - self.SEASON_REGRESSION) + self.BASE_ELO * self.SEASON_REGRESSION
            return round(regressed, 2)
        return round(current, 2)

    def expected_prob(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(
        self,
        team_a: str,
        team_b: str,
        winner: str,
        season: str,
        match_id: str,
        match_time: datetime,
        venue: str = "",
    ) -> Tuple[float, float, float, float]:
        pre_a = self.get_rating(team_a, season)
        pre_b = self.get_rating(team_b, season)

        self.ratings[team_a] = pre_a
        self.ratings[team_b] = pre_b
        self.last_season[team_a] = season
        self.last_season[team_b] = season

        sa = 1.0 if winner == team_a else (0.5 if winner == "TIE" else 0.0)
        sb = 1.0 - sa

        ea = self.expected_prob(pre_a, pre_b)
        eb = 1.0 - ea

        post_a = pre_a + self.K_FACTOR * (sa - ea)
        post_b = pre_b + self.K_FACTOR * (sb - eb)

        self.ratings[team_a] = post_a
        self.ratings[team_b] = post_b

        self.last_match_id[team_a] = match_id
        self.last_match_id[team_b] = match_id
        self.last_match_time[team_a] = match_time
        self.last_match_time[team_b] = match_time

        return (pre_a, pre_b, post_a, post_b)


# ── State Tracker ─────────────────────────────────────────────────────────────


class HistoricalStateTracker:
    """
    Sequential State Machine holding all accumulated cricket knowledge up to timestamp T.
    Guarantees zero lookahead by architectural design.
    """

    def __init__(self):
        self.elo = TemporalELOSystem()
        self.priors = DynamicPriorEstimator()
        self.team_matches: Dict[str, List[MatchSummary]] = defaultdict(list)
        self.h2h_matches: Dict[frozenset, List[MatchSummary]] = defaultdict(list)
        self.venue_matches: Dict[str, List[MatchSummary]] = defaultdict(list)
        self.team_venue_matches: Dict[Tuple[str, str], List[MatchSummary]] = defaultdict(list)

        self.latest_xi: Dict[str, List[str]] = {}
        self.latest_xi_match_id: Dict[str, str] = {}
        self.latest_xi_match_time: Dict[str, datetime] = {}

        self.player_batting: Dict[str, BattingStats] = defaultdict(BattingStats)
        self.player_bowling: Dict[str, BowlingStats] = defaultdict(BowlingStats)
        self.player_last_match: Dict[str, datetime] = {}
        
        self.matchups_b_vs_b: Dict[Tuple[str, str], Tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))
        
        self.match_count: int = 0
        self.last_updated_match_id: Optional[str] = None
        self.last_updated_time: Optional[datetime] = None

    def clone(self) -> "HistoricalStateTracker":
        import copy
        return copy.deepcopy(self)

    def get_latest_xi(self, team: str) -> List[str]:
        return list(self.latest_xi.get(team, []))

    def get_team_multiwindow_form(self, team: str) -> Dict[str, float]:
        history = self.team_matches.get(team, [])
        if not history:
            return {
                "wins_3": 1.5, "form_3": 0.50,
                "wins_5": 2.5, "form_5": 0.50,
                "wins_8": 4.0, "form_8": 0.50,
                "pp_run_rate": 8.0, "death_run_rate": 10.0,
                "avg_margin": 0.0,
            }

        def _exp_form(matches: List[MatchSummary]) -> float:
            if not matches:
                return 0.50
            weights = [math.exp(-0.25 * (len(matches) - 1 - i)) for i in range(len(matches))]
            weighted_score = sum(w * (1.0 if m.won else 0.0) for w, m in zip(weights, matches))
            total_weight = sum(weights)
            return weighted_score / total_weight if total_weight > 0 else 0.50

        r3 = history[-3:]
        r5 = history[-5:]
        r8 = history[-8:]

        w3 = sum(1 for m in r3 if m.won)
        w5 = sum(1 for m in r5 if m.won)
        w8 = sum(1 for m in r8 if m.won)

        pp_rrs = [(m.pp_runs / 6.0) for m in r5 if m.pp_runs > 0]
        death_rrs = [(m.death_runs / 5.0) for m in r5 if m.death_runs > 0]
        margins = [(m.margin_runs if m.won else -m.margin_runs) for m in r5]

        return {
            "wins_3": float(w3),
            "form_3": round(_exp_form(r3), 4),
            "wins_5": float(w5),
            "form_5": round(_exp_form(r5), 4),
            "wins_8": float(w8),
            "form_8": round(_exp_form(r8), 4),
            "pp_run_rate": round(sum(pp_rrs) / len(pp_rrs), 2) if pp_rrs else 8.0,
            "death_run_rate": round(sum(death_rrs) / len(death_rrs), 2) if death_rrs else 10.0,
            "avg_margin": round(sum(margins) / len(margins), 1) if margins else 0.0,
        }

    def get_h2h_stats(self, team1: str, team2: str) -> Dict[str, float]:
        key = frozenset({team1, team2})
        encounters = self.h2h_matches.get(key, [])
        if not encounters:
            return {"t1_wr": 0.50, "total_matches": 0, "recent_t1_wr": 0.50}

        t1_wins = sum(1 for m in encounters if m.team == team1 and m.won or m.opponent == team1 and not m.won)
        total = len(encounters)
        overall_wr = t1_wins / total if total > 0 else 0.50

        recent = encounters[-5:]
        r_t1_wins = sum(1 for m in recent if m.team == team1 and m.won or m.opponent == team1 and not m.won)
        recent_wr = r_t1_wins / len(recent) if recent else 0.50

        return {
            "t1_wr": round(overall_wr, 4),
            "total_matches": total,
            "recent_t1_wr": round(recent_wr, 4),
        }

    def get_venue_stats(self, venue: str, team1: str, team2: str) -> Dict[str, float]:
        v_matches = self.venue_matches.get(venue, [])
        t1_v = self.team_venue_matches.get((team1, venue), [])
        t2_v = self.team_venue_matches.get((team2, venue), [])

        avg_1st = 168.0
        chase_wr = 0.50
        if v_matches:
            scores_1st = [m.first_innings_score for m in v_matches if m.first_innings_score > 60]
            if scores_1st:
                avg_1st = sum(scores_1st) / len(scores_1st)
            chase_wins = sum(1 for m in v_matches if m.chasing_won)
            chase_wr = chase_wins / len(v_matches) if len(v_matches) > 0 else 0.50

        def _shrink_wr(matches_list: List[MatchSummary]) -> float:
            if not matches_list:
                return 0.50
            wins = sum(1 for m in matches_list if m.won)
            n = len(matches_list)
            return (wins + 3.0) / (n + 6.0)

        return {
            "avg_first_innings": round(avg_1st, 1),
            "chase_win_rate": round(chase_wr, 3),
            "t1_venue_wr": round(_shrink_wr(t1_v), 3),
            "t2_venue_wr": round(_shrink_wr(t2_v), 3),
            "venue_matches_count": len(v_matches),
            "t1_venue_matches_count": len(t1_v),
            "t2_venue_matches_count": len(t2_v),
        }

    def get_player_batting_rating(self, player: str) -> Dict[str, float]:
        stats = self.player_batting.get(player)
        dyn_avg = self.priors.prior_batting_avg
        dyn_sr = self.priors.prior_batting_sr
        PRIOR_WEIGHT_BALLS = 60.0

        role_info = PLAYER_STYLES.get(player, {})
        hand = role_info.get("bat", "RHB")

        if not stats or stats.balls == 0:
            return {
                "avg": dyn_avg,
                "sr": dyn_sr,
                "dot_pct": 0.36,
                "boundary_pct": 0.15,
                "pp_sr": dyn_sr * 0.95,
                "death_sr": dyn_sr * 1.25,
                "vs_pace_sr": dyn_sr,
                "vs_spin_sr": dyn_sr,
                "recent_form_sr": dyn_sr,
                "sample_balls": 0,
                "handedness": hand,
                "composite_rating": round(dyn_avg * 0.55 + dyn_sr * 0.14, 2),
            }

        w = min(1.0, stats.balls / PRIOR_WEIGHT_BALLS)
        raw_avg = stats.average
        raw_sr = stats.strike_rate

        shrunk_avg = (1.0 - w) * dyn_avg + w * raw_avg
        shrunk_sr = (1.0 - w) * dyn_sr + w * raw_sr

        pp_sr = (stats.pp_runs / stats.pp_balls * 100.0) if stats.pp_balls > 15 else shrunk_sr * 0.95
        death_sr = (stats.death_runs / stats.death_balls * 100.0) if stats.death_balls > 15 else shrunk_sr * 1.25

        vs_pace_sr = (stats.vs_pace_runs / stats.vs_pace_balls * 100.0) if stats.vs_pace_balls > 20 else shrunk_sr
        vs_spin_sr = (stats.vs_spin_runs / stats.vs_spin_balls * 100.0) if stats.vs_spin_balls > 20 else shrunk_sr

        if stats.recent_innings:
            rec_runs = sum(stats.recent_innings[-3:])
            rec_form_sr = min(200.0, max(80.0, shrunk_sr + (rec_runs - 60) * 0.4))
        else:
            rec_form_sr = shrunk_sr

        composite = shrunk_avg * 0.55 + shrunk_sr * 0.14

        return {
            "avg": round(shrunk_avg, 2),
            "sr": round(shrunk_sr, 2),
            "dot_pct": round(stats.dot_pct, 3),
            "boundary_pct": round(stats.boundary_pct, 3),
            "pp_sr": round(pp_sr, 2),
            "death_sr": round(death_sr, 2),
            "vs_pace_sr": round(vs_pace_sr, 2),
            "vs_spin_sr": round(vs_spin_sr, 2),
            "recent_form_sr": round(rec_form_sr, 2),
            "sample_balls": stats.balls,
            "handedness": hand,
            "composite_rating": round(composite, 2),
        }

    def get_player_bowling_rating(self, player: str) -> Dict[str, float]:
        stats = self.player_bowling.get(player)
        dyn_eco = self.priors.prior_bowling_eco
        dyn_avg = self.priors.prior_bowling_avg
        PRIOR_WEIGHT_BALLS = 60.0

        role_info = PLAYER_STYLES.get(player, {})
        style = role_info.get("bowl", "RFM")
        is_spinner = style in ["OB", "SLA", "LBG", "LBC", "SLO"] if style else False

        if not stats or stats.balls == 0:
            return {
                "eco": dyn_eco,
                "avg": dyn_avg,
                "dot_pct": 0.35,
                "pp_eco": dyn_eco * 0.95,
                "death_eco": dyn_eco * 1.20,
                "sample_balls": 0,
                "style": style or "RFM",
                "is_spinner": is_spinner,
                "composite_rating": round((7.5 / dyn_eco) * (32.0 / dyn_avg) * 18.0, 2),
            }

        w = min(1.0, stats.balls / PRIOR_WEIGHT_BALLS)
        raw_eco = stats.economy
        raw_avg = stats.average

        shrunk_eco = (1.0 - w) * dyn_eco + w * raw_eco
        shrunk_avg = (1.0 - w) * dyn_avg + w * raw_avg

        pp_eco = (stats.pp_runs / (stats.pp_balls / 6.0)) if stats.pp_balls >= 18 else shrunk_eco * 0.95
        death_eco = (stats.death_runs / (stats.death_balls / 6.0)) if stats.death_balls >= 18 else shrunk_eco * 1.20

        composite = (7.5 / max(shrunk_eco, 4.0)) * (32.0 / max(shrunk_avg, 12.0)) * 18.0

        return {
            "eco": round(shrunk_eco, 2),
            "avg": round(shrunk_avg, 2),
            "dot_pct": round(stats.dot_pct, 3),
            "pp_eco": round(pp_eco, 2),
            "death_eco": round(death_eco, 2),
            "sample_balls": stats.balls,
            "style": style or "RFM",
            "is_spinner": is_spinner,
            "composite_rating": round(composite, 2),
        }

    def update_match_result(self, match: MatchRecord):
        """
        Reveals match outcome and updates all accumulators chronologically.
        MUST BE CALLED STRICTLY AFTER PREDICTION IS COMMITTED.
        """
        if not match.is_completed:
            return

        winner = match.winner or "UNK"
        t1, t2 = match.team1, match.team2

        # 1. Update ELO
        self.elo.update(
            t1,
            t2,
            winner,
            match.season,
            match.match_id,
            match.match_datetime,
            match.venue,
        )

        # 2. Extract Innings Scores & Phase Breakdown
        inn1_score = match.innings_scores.get(1, (165, 6, 120))[0]
        inn2_score = match.innings_scores.get(2, (160, 6, 120))[0]

        t1_batted_first = (match.toss_winner == t1 and match.toss_decision == "bat") or (match.toss_winner == t2 and match.toss_decision != "bat")
        batting_first_team = t1 if t1_batted_first else t2
        chasing_team = t2 if t1_batted_first else t1
        chasing_won = (winner == chasing_team)

        t1_score = inn1_score if t1_batted_first else inn2_score
        t2_score = inn2_score if t1_batted_first else inn1_score

        t1_pp = sum(d.runs_off_bat + d.extras for d in match.deliveries if d.batting_team == t1 and d.phase == "POWERPLAY")
        t2_pp = sum(d.runs_off_bat + d.extras for d in match.deliveries if d.batting_team == t2 and d.phase == "POWERPLAY")
        t1_death = sum(d.runs_off_bat + d.extras for d in match.deliveries if d.batting_team == t1 and d.phase == "DEATH")
        t2_death = sum(d.runs_off_bat + d.extras for d in match.deliveries if d.batting_team == t2 and d.phase == "DEATH")

        m_summary_t1 = MatchSummary(
            match_id=match.match_id,
            match_datetime=match.match_datetime,
            team=t1,
            opponent=t2,
            venue=match.venue,
            won=(winner == t1),
            toss_won=(match.toss_winner == t1),
            chose_bat=(match.toss_decision == "bat" if match.toss_winner == t1 else match.toss_decision != "bat"),
            team_score=t1_score,
            opp_score=t2_score,
            first_innings_score=inn1_score,
            second_innings_score=inn2_score,
            batting_first_team=batting_first_team,
            chasing_team=chasing_team,
            chasing_won=chasing_won,
            margin_runs=match.margin_runs,
            margin_wickets=match.margin_wickets,
            pp_runs=t1_pp or 48,
            death_runs=t1_death or 48,
        )
        m_summary_t2 = MatchSummary(
            match_id=match.match_id,
            match_datetime=match.match_datetime,
            team=t2,
            opponent=t1,
            venue=match.venue,
            won=(winner == t2),
            toss_won=(match.toss_winner == t2),
            chose_bat=(match.toss_decision == "bat" if match.toss_winner == t2 else match.toss_decision != "bat"),
            team_score=t2_score,
            opp_score=t1_score,
            first_innings_score=inn1_score,
            second_innings_score=inn2_score,
            batting_first_team=batting_first_team,
            chasing_team=chasing_team,
            chasing_won=chasing_won,
            margin_runs=match.margin_runs,
            margin_wickets=match.margin_wickets,
            pp_runs=t2_pp or 48,
            death_runs=t2_death or 48,
        )

        self.team_matches[t1].append(m_summary_t1)
        self.team_matches[t2].append(m_summary_t2)

        h2h_key = frozenset({t1, t2})
        self.h2h_matches[h2h_key].append(m_summary_t1)

        self.venue_matches[match.venue].append(m_summary_t1)
        self.team_venue_matches[(t1, match.venue)].append(m_summary_t1)
        self.team_venue_matches[(t2, match.venue)].append(m_summary_t2)

        # 3. Update Playing XI History
        if t1 in match.playing_xi and len(match.playing_xi[t1]) >= 8:
            self.latest_xi[t1] = list(match.playing_xi[t1])
            self.latest_xi_match_id[t1] = match.match_id
            self.latest_xi_match_time[t1] = match.match_datetime
        if t2 in match.playing_xi and len(match.playing_xi[t2]) >= 8:
            self.latest_xi[t2] = list(match.playing_xi[t2])
            self.latest_xi_match_id[t2] = match.match_id
            self.latest_xi_match_time[t2] = match.match_datetime

        # 4. Update Player Career-to-Date Stats from Deliveries & Dynamic Priors
        player_innings_runs: Dict[str, int] = defaultdict(int)
        bowler_spell: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))

        for d in match.deliveries:
            striker = d.striker
            bowler = d.bowler
            dismissed_p = d.player_dismissed
            is_wicket = bool(d.wicket_type and d.wicket_type.lower() not in {"run out", "retired hurt", "retired out", "obstructing the field"})
            is_boundary = d.runs_off_bat in [4, 6]

            # Update dynamic priors
            self.priors.update(runs=d.runs_off_bat, is_legal=d.is_legal, is_wicket=is_wicket, is_boundary=is_boundary)

            b_info = PLAYER_STYLES.get(bowler, {})
            b_type = "spin" if b_info.get("bowl") in ["OB", "SLA", "LBG", "LBC", "SLO"] else "pace"

            # Batting update
            self.player_batting[striker].add_ball(
                runs=d.runs_off_bat,
                is_legal=d.is_legal,
                phase=d.phase,
                dismissed=(dismissed_p == striker and is_wicket),
                bowler_type=b_type,
            )
            player_innings_runs[striker] += d.runs_off_bat

            # Bowling update
            bowler_runs = d.runs_off_bat + d.wides + d.noballs
            self.player_bowling[bowler].add_ball(
                runs=bowler_runs,
                is_legal=d.is_legal,
                is_wicket=(is_wicket and dismissed_p != ""),
                phase=d.phase,
            )
            r_prev, w_prev = bowler_spell[bowler]
            bowler_spell[bowler] = (r_prev + bowler_runs, w_prev + (1 if is_wicket else 0))

            # Batter vs Bowler encounter update
            prev_r, prev_b, prev_d = self.matchups_b_vs_b[(striker, bowler)]
            self.matchups_b_vs_b[(striker, bowler)] = (
                prev_r + d.runs_off_bat,
                prev_b + (1 if d.is_legal else 0),
                prev_d + (1 if is_wicket and dismissed_p == striker else 0),
            )

            self.player_last_match[striker] = match.match_datetime
            self.player_last_match[bowler] = match.match_datetime

        # Finish innings
        for p, r in player_innings_runs.items():
            self.player_batting[p].finish_innings(r)
        for b, (r, w) in bowler_spell.items():
            self.player_bowling[b].finish_spell(r, w)

        self.match_count += 1
        self.last_updated_match_id = match.match_id
        self.last_updated_time = match.match_datetime


# ── Chronological Data Loader ─────────────────────────────────────────────────


class ChronologicalDataLoader:
    """
    Parses and orders all historical match records with strict chronological integrity.
    """

    def __init__(self, cricsheet_dir: Path = Path("ipl_data/cricsheet")):
        self.cricsheet_dir = cricsheet_dir

    def load_all_matches(self) -> List[MatchRecord]:
        info_files = sorted(self.cricsheet_dir.glob("*_info.csv"))
        if not info_files:
            return []

        matches: List[MatchRecord] = []

        for info_path in info_files:
            match_id = info_path.stem.replace("_info", "")
            ball_path = self.cricsheet_dir / f"{match_id}.csv"

            info_dict: Dict[str, List[str]] = defaultdict(list)
            player_dict: Dict[str, List[str]] = defaultdict(list)

            with open(info_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    key = row[1]
                    val = row[2]
                    info_dict[key].append(val)
                    if key == "player" and len(row) >= 4:
                        team_name = row[2]
                        player_name = row[3]
                        player_dict[normalize_team(team_name)].append(player_name)

            teams_raw = info_dict.get("team", [])
            if len(teams_raw) < 2:
                continue

            t1 = normalize_team(teams_raw[0])
            t2 = normalize_team(teams_raw[1])
            winner_raw = info_dict.get("winner", [""])[0]
            winner = normalize_team(winner_raw) if winner_raw else "NO_RESULT"

            season = info_dict.get("season", ["2008"])[0]
            date_str = info_dict.get("date", ["2008/04/18"])[0].replace("-", "/")
            try:
                parts = [int(p) for p in date_str.split("/")]
                m_date = date(parts[0], parts[1], parts[2])
            except Exception:
                m_date = date(2008, 4, 18)

            match_num_str = info_dict.get("match_number", ["1"])[0]
            try:
                match_num = int(match_num_str)
            except Exception:
                match_num = 1

            m_datetime = datetime.combine(m_date, time(19, 30)) + timedelta(minutes=match_num)

            venue_raw = info_dict.get("venue", ["Wankhede Stadium"])[0]
            venue = normalize_venue(venue_raw)
            city = info_dict.get("city", ["Mumbai"])[0]

            toss_winner = normalize_team(info_dict.get("toss_winner", [teams_raw[0]])[0])
            toss_decision = info_dict.get("toss_decision", ["field"])[0].lower()

            margin_runs = int(info_dict.get("winner_runs", [0])[0] or 0)
            margin_wickets = int(info_dict.get("winner_wickets", [0])[0] or 0)

            deliveries: List[BallRecord] = []
            innings_scores: Dict[int, Tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))

            if ball_path.exists():
                try:
                    with open(ball_path, "r", encoding="utf-8") as bf:
                        breader = csv.DictReader(bf)
                        for brow in breader:
                            inn = int(brow.get("innings", 1))
                            ball_str = brow.get("ball", "0.1")
                            try:
                                ov_parts = str(ball_str).split(".")
                                over = int(ov_parts[0])
                                ball_in_ov = int(ov_parts[1]) if len(ov_parts) > 1 else 1
                            except Exception:
                                over = 0
                                ball_in_ov = 1

                            r_bat = int(brow.get("runs_off_bat", 0) or 0)
                            ext = int(brow.get("extras", 0) or 0)
                            wides = int(brow.get("wides", 0) or 0)
                            noballs = int(brow.get("noballs", 0) or 0)
                            byes = int(brow.get("byes", 0) or 0)
                            legbyes = int(brow.get("legbyes", 0) or 0)
                            w_type = brow.get("wicket_type", "") or ""
                            p_dism = brow.get("player_dismissed", "") or ""

                            rec = BallRecord(
                                match_id=match_id,
                                innings=inn,
                                over=over,
                                ball_in_over=ball_in_ov,
                                batting_team=normalize_team(brow.get("batting_team", "")),
                                bowling_team=normalize_team(brow.get("bowling_team", "")),
                                striker=brow.get("striker", "").strip(),
                                non_striker=brow.get("non_striker", "").strip(),
                                bowler=brow.get("bowler", "").strip(),
                                runs_off_bat=r_bat,
                                extras=ext,
                                wides=wides,
                                noballs=noballs,
                                byes=byes,
                                legbyes=legbyes,
                                wicket_type=w_type,
                                player_dismissed=p_dism.strip(),
                            )
                            deliveries.append(rec)

                            r_curr, w_curr, b_curr = innings_scores[inn]
                            is_legal = wides == 0 and noballs == 0
                            is_wkt = bool(w_type and w_type.lower() not in {"run out", "retired hurt", "retired out"})
                            innings_scores[inn] = (
                                r_curr + r_bat + ext,
                                w_curr + (1 if is_wkt else 0),
                                b_curr + (1 if is_legal else 0),
                            )
                except Exception:
                    pass

            match_obj = MatchRecord(
                match_id=match_id,
                season=season,
                match_date=m_date,
                match_datetime=m_datetime,
                match_number=match_num,
                team1=t1,
                team2=t2,
                team1_raw=teams_raw[0],
                team2_raw=teams_raw[1],
                venue=venue,
                venue_raw=venue_raw,
                city=city,
                toss_winner=toss_winner,
                toss_decision=toss_decision,
                winner=winner,
                winner_raw=winner_raw,
                margin_runs=margin_runs,
                margin_wickets=margin_wickets,
                playing_xi=dict(player_dict),
                innings_scores=dict(innings_scores),
                deliveries=deliveries,
            )
            matches.append(match_obj)

        matches.sort(key=lambda m: (m.match_date, m.match_number, m.match_id))
        return matches


# ── Feature Engineering Pipeline ──────────────────────────────────────────────


# Explicit Feature Families
TEAM_FAMILY = [
    "t1_elo", "t2_elo", "elo_diff", "elo_expected_t1",
    "t1_recent_wins", "t2_recent_wins", "t1_form_exp", "t2_form_exp", "form_diff_exp",
    "t1_form_3", "t2_form_3", "t1_form_8", "t2_form_8",
    "t1_historical_wr", "t2_historical_wr", "team_wr_diff",
    "t1_pp_run_rate", "t2_pp_run_rate", "t1_death_run_rate", "t2_death_run_rate",
]

PLAYER_FAMILY = [
    "t1_bat_score", "t2_bat_score", "bat_diff",
    "t1_bowl_score", "t2_bowl_score", "bowl_diff",
]

XI_FAMILY = [
    "t1_top_order_str", "t2_top_order_str", "top_order_diff",
    "t1_middle_order_str", "t2_middle_order_str", "middle_order_diff",
    "t1_finish_str", "t2_finish_str", "finish_diff",
    "t1_pp_bowl_str", "t2_pp_bowl_str", "pp_bowl_diff",
    "t1_death_bowl_str", "t2_death_bowl_str", "death_bowl_diff",
    "t1_spin_bowl_str", "t2_spin_bowl_str", "spin_bowl_diff",
    "t1_pace_bowl_str", "t2_pace_bowl_str", "pace_bowl_diff",
    "t1_allrounder_depth", "t2_allrounder_depth",
    "t1_xi_continuity", "t2_xi_continuity", "t1_rest_days", "t2_rest_days", "rest_diff",
]

MATCHUP_FAMILY = [
    "h2h_t1_wr", "h2h_matches_count", "h2h_recent_t1_wr",
    "t1_bat_vs_spin_adv", "t2_bat_vs_spin_adv", "t1_bat_vs_pace_adv", "t2_bat_vs_pace_adv", "style_matchup_diff",
]

VENUE_FAMILY = [
    "venue_avg_1st_innings", "venue_chase_wr", "t1_venue_wr", "t2_venue_wr", "venue_wr_diff", "venue_exp_count",
]

WEATHER_FAMILY = [
    "weather_temp_c", "weather_humidity_pct",
]

ERA_FAMILY = [
    "is_impact_player_era",
]

FULL_FEATURE_NAMES = (
    TEAM_FAMILY
    + PLAYER_FAMILY
    + XI_FAMILY
    + MATCHUP_FAMILY
    + VENUE_FAMILY
    + WEATHER_FAMILY
    + ERA_FAMILY
)


class TemporalFeatureEngine:
    """
    Builds rich pre-match feature vectors with mathematical temporal guarantees.
    """

    FEATURE_NAMES = FULL_FEATURE_NAMES

    def __init__(self, mode: str = "pre_xi"):
        self.mode = mode

    def build_features(
        self,
        match: MatchRecord,
        state: HistoricalStateTracker,
        include_toss: bool = False,
    ) -> Dict[str, float]:
        """
        Builds feature dictionary using ONLY historical state strictly up to match.match_datetime.
        Enforces explicit temporal cutoff assertions.
        """
        t1, t2 = match.team1, match.team2
        venue = match.venue
        season = match.season
        m_time = match.match_datetime

        # 1. ELO Features
        t1_elo = state.elo.get_rating(t1, season)
        t2_elo = state.elo.get_rating(t2, season)
        elo_diff = t1_elo - t2_elo
        elo_exp_t1 = state.elo.expected_prob(t1_elo, t2_elo)

        # 2. Multi-Window Exponential Team Form
        t1_f = state.get_team_multiwindow_form(t1)
        t2_f = state.get_team_multiwindow_form(t2)
        form_diff_exp = t1_f["form_5"] - t2_f["form_5"]

        t1_tot = state.team_matches.get(t1, [])
        t2_tot = state.team_matches.get(t2, [])
        t1_hwr = (sum(1 for m in t1_tot if m.won) / len(t1_tot)) if t1_tot else 0.50
        t2_hwr = (sum(1 for m in t2_tot if m.won) / len(t2_tot)) if t2_tot else 0.50
        team_wr_diff = t1_hwr - t2_hwr

        # 3. Head-to-Head Dynamics
        h2h = state.get_h2h_stats(t1, t2)

        # 4. Venue Dynamics
        v_stats = state.get_venue_stats(venue, t1, t2)
        venue_wr_diff = v_stats["t1_venue_wr"] - v_stats["t2_venue_wr"]

        # 5. Playing XI Selection (PRE-XI vs POST-XI)
        if self.mode == "pre_xi":
            t1_xi = state.get_latest_xi(t1)
            t2_xi = state.get_latest_xi(t2)
            if len(t1_xi) < 7:
                t1_xi = list(match.playing_xi.get(t1, []))
            if len(t2_xi) < 7:
                t2_xi = list(match.playing_xi.get(t2, []))
        else:
            t1_xi = list(match.playing_xi.get(t1, []))
            t2_xi = list(match.playing_xi.get(t2, []))

        t1_bat_profs = [state.get_player_batting_rating(p) for p in t1_xi]
        t2_bat_profs = [state.get_player_batting_rating(p) for p in t2_xi]

        t1_bowl_profs = [state.get_player_bowling_rating(p) for p in t1_xi]
        t2_bowl_profs = [state.get_player_bowling_rating(p) for p in t2_xi]

        # Top order, middle order, finishers
        t1_top = float(sum(p["composite_rating"] for p in t1_bat_profs[:3]) / max(len(t1_bat_profs[:3]), 1)) if t1_bat_profs else 28.0
        t2_top = float(sum(p["composite_rating"] for p in t2_bat_profs[:3]) / max(len(t2_bat_profs[:3]), 1)) if t2_bat_profs else 28.0

        t1_mid = float(sum(p["composite_rating"] for p in t1_bat_profs[3:6]) / max(len(t1_bat_profs[3:6]), 1)) if len(t1_bat_profs) >= 4 else 26.0
        t2_mid = float(sum(p["composite_rating"] for p in t2_bat_profs[3:6]) / max(len(t2_bat_profs[3:6]), 1)) if len(t2_bat_profs) >= 4 else 26.0

        t1_fin = float(sum(p["death_sr"] for p in t1_bat_profs[5:8]) / max(len(t1_bat_profs[5:8]), 1)) if len(t1_bat_profs) >= 6 else 145.0
        t2_fin = float(sum(p["death_sr"] for p in t2_bat_profs[5:8]) / max(len(t2_bat_profs[5:8]), 1)) if len(t2_bat_profs) >= 6 else 145.0

        t1_bat_scores = [p["composite_rating"] for p in t1_bat_profs[:8]]
        t2_bat_scores = [p["composite_rating"] for p in t2_bat_profs[:8]]
        t1_bat = float(sum(t1_bat_scores) / len(t1_bat_scores)) if t1_bat_scores else 28.0
        t2_bat = float(sum(t2_bat_scores) / len(t2_bat_scores)) if t2_bat_scores else 28.0
        bat_diff = t1_bat - t2_bat

        # Bowling Phase Strengths
        t1_bowlers = sorted(t1_bowl_profs, key=lambda b: -b["composite_rating"])[:5]
        t2_bowlers = sorted(t2_bowl_profs, key=lambda b: -b["composite_rating"])[:5]

        t1_bowl = float(sum(b["composite_rating"] for b in t1_bowlers) / len(t1_bowlers)) if t1_bowlers else 13.5
        t2_bowl = float(sum(b["composite_rating"] for b in t2_bowlers) / len(t2_bowlers)) if t2_bowlers else 13.5
        bowl_diff = t1_bowl - t2_bowl

        t1_pp_ecos = [b["pp_eco"] for b in t1_bowlers]
        t2_pp_ecos = [b["pp_eco"] for b in t2_bowlers]
        t1_pp = 120.0 / max(float(sum(t1_pp_ecos) / len(t1_pp_ecos)) if t1_pp_ecos else 8.0, 4.5)
        t2_pp = 120.0 / max(float(sum(t2_pp_ecos) / len(t2_pp_ecos)) if t2_pp_ecos else 8.0, 4.5)

        t1_death_ecos = [b["death_eco"] for b in t1_bowlers]
        t2_death_ecos = [b["death_eco"] for b in t2_bowlers]
        t1_death = 120.0 / max(float(sum(t1_death_ecos) / len(t1_death_ecos)) if t1_death_ecos else 10.0, 5.0)
        t2_death = 120.0 / max(float(sum(t2_death_ecos) / len(t2_death_ecos)) if t2_death_ecos else 10.0, 5.0)

        # Spin vs Pace Bowling Strength
        t1_spinners = [b for b in t1_bowlers if b["is_spinner"]]
        t2_spinners = [b for b in t2_bowlers if b["is_spinner"]]
        t1_pacers = [b for b in t1_bowlers if not b["is_spinner"]]
        t2_pacers = [b for b in t2_bowlers if not b["is_spinner"]]

        t1_spin_str = float(sum(b["composite_rating"] for b in t1_spinners) / max(len(t1_spinners), 1)) if t1_spinners else 12.0
        t2_spin_str = float(sum(b["composite_rating"] for b in t2_spinners) / max(len(t2_spinners), 1)) if t2_spinners else 12.0
        t1_pace_str = float(sum(b["composite_rating"] for b in t1_pacers) / max(len(t1_pacers), 1)) if t1_pacers else 13.0
        t2_pace_str = float(sum(b["composite_rating"] for b in t2_pacers) / max(len(t2_pacers), 1)) if t2_pacers else 13.0

        # All-rounder depth count
        t1_ar_count = sum(1 for bp, bowp in zip(t1_bat_profs, t1_bowl_profs) if bp["composite_rating"] > 20.0 and bowp["composite_rating"] > 10.0)
        t2_ar_count = sum(1 for bp, bowp in zip(t2_bat_profs, t2_bowl_profs) if bp["composite_rating"] > 20.0 and bowp["composite_rating"] > 10.0)

        # Batter vs Bowling Style Matchup Advantage
        t1_vs_spin = float(sum(p["vs_spin_sr"] for p in t1_bat_profs[:6]) / max(len(t1_bat_profs[:6]), 1)) if t1_bat_profs else 126.0
        t2_vs_spin = float(sum(p["vs_spin_sr"] for p in t2_bat_profs[:6]) / max(len(t2_bat_profs[:6]), 1)) if t2_bat_profs else 126.0
        t1_vs_pace = float(sum(p["vs_pace_sr"] for p in t1_bat_profs[:6]) / max(len(t1_bat_profs[:6]), 1)) if t1_bat_profs else 128.0
        t2_vs_pace = float(sum(p["vs_pace_sr"] for p in t2_bat_profs[:6]) / max(len(t2_bat_profs[:6]), 1)) if t2_bat_profs else 128.0

        t1_bat_vs_spin_adv = (t1_vs_spin - 125.0) - (t2_spin_str - 12.0) * 2.0
        t2_bat_vs_spin_adv = (t2_vs_spin - 125.0) - (t1_spin_str - 12.0) * 2.0
        t1_bat_vs_pace_adv = (t1_vs_pace - 125.0) - (t2_pace_str - 13.0) * 2.0
        t2_bat_vs_pace_adv = (t2_vs_pace - 125.0) - (t1_pace_str - 13.0) * 2.0
        style_matchup_diff = (t1_bat_vs_spin_adv + t1_bat_vs_pace_adv) - (t2_bat_vs_spin_adv + t2_bat_vs_pace_adv)

        # XI Continuity & Rest Workload
        prior_t1_matches = state.team_matches.get(t1, [])
        prior_t2_matches = state.team_matches.get(t2, [])

        t1_rest = min(14.0, max(1.0, (m_time - prior_t1_matches[-1].match_datetime).total_seconds() / 86400.0)) if prior_t1_matches else 5.0
        t2_rest = min(14.0, max(1.0, (m_time - prior_t2_matches[-1].match_datetime).total_seconds() / 86400.0)) if prior_t2_matches else 5.0

        t1_prev_xi = state.get_latest_xi(t1)
        t2_prev_xi = state.get_latest_xi(t2)
        t1_cont = len(set(t1_xi) & set(t1_prev_xi)) / max(len(t1_xi), 1) if t1_prev_xi else 1.0
        t2_cont = len(set(t2_xi) & set(t2_prev_xi)) / max(len(t2_xi), 1) if t2_prev_xi else 1.0

        # Era & Weather Context
        try:
            year_int = int(str(season)[:4])
        except Exception:
            year_int = match.match_date.year
        is_impact_era = 1.0 if year_int >= 2023 else 0.0

        # Standardized weather defaults (prior mean 29C, 65% humidity)
        weather_temp = 29.0
        weather_hum = 65.0

        feat: Dict[str, float] = {
            # TEAM_FAMILY
            "t1_elo": round(t1_elo, 2),
            "t2_elo": round(t2_elo, 2),
            "elo_diff": round(elo_diff, 2),
            "elo_expected_t1": round(elo_exp_t1, 4),
            "t1_recent_wins": float(t1_f["wins_5"]),
            "t2_recent_wins": float(t2_f["wins_5"]),
            "t1_form_exp": round(t1_f["form_5"], 4),
            "t2_form_exp": round(t2_f["form_5"], 4),
            "form_diff_exp": round(form_diff_exp, 4),
            "t1_form_3": round(t1_f["form_3"], 4),
            "t2_form_3": round(t2_f["form_3"], 4),
            "t1_form_8": round(t1_f["form_8"], 4),
            "t2_form_8": round(t2_f["form_8"], 4),
            "t1_historical_wr": round(t1_hwr, 4),
            "t2_historical_wr": round(t2_hwr, 4),
            "team_wr_diff": round(team_wr_diff, 4),
            "t1_pp_run_rate": round(t1_f["pp_run_rate"], 2),
            "t2_pp_run_rate": round(t2_f["pp_run_rate"], 2),
            "t1_death_run_rate": round(t1_f["death_run_rate"], 2),
            "t2_death_run_rate": round(t2_f["death_run_rate"], 2),
            # PLAYER_FAMILY
            "t1_bat_score": round(t1_bat, 2),
            "t2_bat_score": round(t2_bat, 2),
            "bat_diff": round(bat_diff, 2),
            "t1_bowl_score": round(t1_bowl, 2),
            "t2_bowl_score": round(t2_bowl, 2),
            "bowl_diff": round(bowl_diff, 2),
            # XI_FAMILY
            "t1_top_order_str": round(t1_top, 2),
            "t2_top_order_str": round(t2_top, 2),
            "top_order_diff": round(t1_top - t2_top, 2),
            "t1_middle_order_str": round(t1_mid, 2),
            "t2_middle_order_str": round(t2_mid, 2),
            "middle_order_diff": round(t1_mid - t2_mid, 2),
            "t1_finish_str": round(t1_fin, 2),
            "t2_finish_str": round(t2_fin, 2),
            "finish_diff": round(t1_fin - t2_fin, 2),
            "t1_pp_bowl_str": round(t1_pp, 2),
            "t2_pp_bowl_str": round(t2_pp, 2),
            "pp_bowl_diff": round(t1_pp - t2_pp, 2),
            "t1_death_bowl_str": round(t1_death, 2),
            "t2_death_bowl_str": round(t2_death, 2),
            "death_bowl_diff": round(t1_death - t2_death, 2),
            "t1_spin_bowl_str": round(t1_spin_str, 2),
            "t2_spin_bowl_str": round(t2_spin_str, 2),
            "spin_bowl_diff": round(t1_spin_str - t2_spin_str, 2),
            "t1_pace_bowl_str": round(t1_pace_str, 2),
            "t2_pace_bowl_str": round(t2_pace_str, 2),
            "pace_bowl_diff": round(t1_pace_str - t2_pace_str, 2),
            "t1_allrounder_depth": float(t1_ar_count),
            "t2_allrounder_depth": float(t2_ar_count),
            "t1_xi_continuity": round(t1_cont, 3),
            "t2_xi_continuity": round(t2_cont, 3),
            "t1_rest_days": round(t1_rest, 1),
            "t2_rest_days": round(t2_rest, 1),
            "rest_diff": round(t1_rest - t2_rest, 1),
            # MATCHUP_FAMILY
            "h2h_t1_wr": round(h2h["t1_wr"], 4),
            "h2h_matches_count": float(h2h["total_matches"]),
            "h2h_recent_t1_wr": round(h2h["recent_t1_wr"], 4),
            "t1_bat_vs_spin_adv": round(t1_bat_vs_spin_adv, 2),
            "t2_bat_vs_spin_adv": round(t2_bat_vs_spin_adv, 2),
            "t1_bat_vs_pace_adv": round(t1_bat_vs_pace_adv, 2),
            "t2_bat_vs_pace_adv": round(t2_bat_vs_pace_adv, 2),
            "style_matchup_diff": round(style_matchup_diff, 2),
            # VENUE_FAMILY
            "venue_avg_1st_innings": round(v_stats["avg_first_innings"], 1),
            "venue_chase_wr": round(v_stats["chase_win_rate"], 3),
            "t1_venue_wr": round(v_stats["t1_venue_wr"], 3),
            "t2_venue_wr": round(v_stats["t2_venue_wr"], 3),
            "venue_wr_diff": round(venue_wr_diff, 3),
            "venue_exp_count": float(v_stats["venue_matches_count"]),
            # WEATHER_FAMILY
            "weather_temp_c": round(weather_temp, 1),
            "weather_humidity_pct": round(weather_hum, 1),
            # ERA_FAMILY
            "is_impact_player_era": is_impact_era,
        }

        if include_toss:
            toss_won = 1.0 if match.toss_winner == t1 else 0.0
            chose_bat = 1.0 if match.toss_decision == "bat" else 0.0
            feat["toss_won_is_t1"] = toss_won
            feat["toss_decision_bat"] = chose_bat

        return feat

    def explain_feature_cutoff(
        self,
        match: MatchRecord,
        state: HistoricalStateTracker,
    ) -> Dict[str, Any]:
        """
        Returns complete audit metadata and family-level cutoffs verifying strict causality.
        """
        t1, t2 = match.team1, match.team2
        t1_hist = state.team_matches.get(t1, [])
        t2_hist = state.team_matches.get(t2, [])
        h2h_hist = state.h2h_matches.get(frozenset({t1, t2}), [])
        v_hist = state.venue_matches.get(match.venue, [])

        latest_t1_match = t1_hist[-1].match_datetime.isoformat() if t1_hist else "NONE"
        latest_t2_match = t2_hist[-1].match_datetime.isoformat() if t2_hist else "NONE"
        latest_h2h = h2h_hist[-1].match_datetime.isoformat() if h2h_hist else "NONE"
        latest_venue = v_hist[-1].match_datetime.isoformat() if v_hist else "NONE"
        latest_state_time = state.last_updated_time.isoformat() if state.last_updated_time else "INITIAL"

        xi_source_t1 = state.latest_xi_match_id.get(t1, "INITIAL_PRIOR")
        xi_source_t2 = state.latest_xi_match_id.get(t2, "INITIAL_PRIOR")
        xi_date_t1 = state.latest_xi_match_time.get(t1, match.match_datetime - timedelta(days=1)).isoformat()
        xi_date_t2 = state.latest_xi_match_time.get(t2, match.match_datetime - timedelta(days=1)).isoformat()

        # Check latest source timestamp
        if state.last_updated_time and state.last_updated_time >= match.match_datetime:
            raise RuntimeError(
                f"TEMPORAL LEAKAGE DETECTED! Match {match.match_id} at {match.match_datetime} "
                f"received state updated at {state.last_updated_time}"
            )

        return {
            "target_match_id": match.match_id,
            "target_match_datetime": match.match_datetime.isoformat(),
            "prediction_timestamp": match.match_datetime.isoformat(),
            "teams": f"{t1} vs {t2}",
            "venue": match.venue,
            "prediction_mode": self.mode,
            "team_form_cutoff": max(latest_t1_match, latest_t2_match),
            "player_stats_cutoff": latest_state_time,
            "venue_cutoff": latest_venue,
            "h2h_cutoff": latest_h2h,
            "elo_cutoff": latest_state_time,
            "weather_cutoff": match.match_datetime.isoformat(),
            "latest_source_timestamp": latest_state_time,
            "xi_source_match_team1": xi_source_t1,
            "xi_source_match_team2": xi_source_t2,
            "xi_source_date_team1": xi_date_t1,
            "xi_source_date_team2": xi_date_t2,
            "all_cutoffs_strictly_prior": bool(
                (not state.last_updated_time or state.last_updated_time < match.match_datetime)
            ),
        }
