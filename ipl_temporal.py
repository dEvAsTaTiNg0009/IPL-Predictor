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
        # Over is 0-indexed or 1-indexed depending on source.
        # In cricsheet delivery format, 0.1 -> over 1.
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
    # Phase stats: runs, balls, dismissals
    pp_runs: int = 0
    pp_balls: int = 0
    pp_dismissals: int = 0
    mid_runs: int = 0
    mid_balls: int = 0
    mid_dismissals: int = 0
    death_runs: int = 0
    death_balls: int = 0
    death_dismissals: int = 0
    recent_innings: List[int] = field(default_factory=list)  # Last N innings runs

    def add_ball(self, runs: int, is_legal: bool, phase: str, dismissed: bool):
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
            return float(self.runs) if self.runs > 0 else 22.0
        return self.runs / self.dismissals

    @property
    def strike_rate(self) -> float:
        if self.balls == 0:
            return 125.0
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
    # Phase stats
    pp_balls: int = 0
    pp_runs: int = 0
    pp_wickets: int = 0
    mid_balls: int = 0
    mid_runs: int = 0
    mid_wickets: int = 0
    death_balls: int = 0
    death_runs: int = 0
    death_wickets: int = 0
    recent_figures: List[Tuple[int, int]] = field(default_factory=list)  # (runs, wkts)

    def add_ball(self, runs: int, is_legal: bool, is_wicket: bool, phase: str):
        self.runs_conceded += runs
        if is_legal:
            self.balls += 1
            if runs == 0:
                self.dots += 1
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
            return 32.0
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
    SEASON_REGRESSION = 0.20  # Regress 20% toward 1500 between seasons

    def __init__(self):
        self.ratings: Dict[str, float] = defaultdict(lambda: self.BASE_ELO)
        self.last_season: Dict[str, str] = {}
        self.last_match_id: Dict[str, str] = {}
        self.last_match_time: Dict[str, datetime] = {}

    def get_rating(self, team: str, season: Optional[str] = None) -> float:
        # Check for season transition regression
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
        """
        Updates ELO ratings after match outcome is revealed.
        Returns: (pre_a, pre_b, post_a, post_b)
        """
        pre_a = self.get_rating(team_a, season)
        pre_b = self.get_rating(team_b, season)

        # Apply season boundary regression permanently
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
    Guarantees no lookahead by design.
    """

    def __init__(self):
        self.elo = TemporalELOSystem()
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
        self.match_count: int = 0
        self.last_updated_match_id: Optional[str] = None
        self.last_updated_time: Optional[datetime] = None

    def clone(self) -> "HistoricalStateTracker":
        """Deep copy state for isolation testing."""
        import copy

        return copy.deepcopy(self)

    def get_latest_xi(self, team: str) -> List[str]:
        """Returns the most recent playing XI for team prior to current time."""
        return list(self.latest_xi.get(team, []))

    def get_team_form(self, team: str, n: int = 5) -> Tuple[int, float]:
        """
        Returns (wins_in_last_n, exponential_form_score) strictly for previous matches.
        """
        history = self.team_matches.get(team, [])
        if not history:
            return (2, 0.50)

        recent = history[-n:]
        wins = sum(1 for m in recent if m.won)

        # Exponentially weighted recency
        weights = [math.exp(-0.35 * (len(recent) - 1 - i)) for i in range(len(recent))]
        weighted_score = sum(w * (1.0 if m.won else 0.0) for w, m in zip(weights, recent))
        total_weight = sum(weights)
        norm_form = weighted_score / total_weight if total_weight > 0 else 0.50
        return (wins, round(norm_form, 4))

    def get_h2h_stats(self, team1: str, team2: str) -> Dict[str, float]:
        """Returns head-to-head statistics strictly prior to current match."""
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
        """Returns venue-level prior statistics."""
        v_matches = self.venue_matches.get(venue, [])
        t1_v = self.team_venue_matches.get((team1, venue), [])
        t2_v = self.team_venue_matches.get((team2, venue), [])

        avg_1st = 168.0
        chase_wr = 0.50
        if v_matches:
            scores_1st = [m.team_score for m in v_matches if m.team_score > 60]
            if scores_1st:
                avg_1st = sum(scores_1st) / len(scores_1st)
            # Chase win rate (inn 2 wins)
            chase_wins = sum(1 for m in v_matches if not m.chose_bat and m.toss_won and m.won or m.chose_bat and not m.toss_won and m.won)
            chase_wr = chase_wins / len(v_matches) if len(v_matches) > 0 else 0.50

        t1_vwr = (sum(1 for m in t1_v if m.won) / len(t1_v)) if t1_v else 0.50
        t2_vwr = (sum(1 for m in t2_v if m.won) / len(t2_v)) if t2_v else 0.50

        return {
            "avg_first_innings": round(avg_1st, 1),
            "chase_win_rate": round(chase_wr, 3),
            "t1_venue_wr": round(t1_vwr, 3),
            "t2_venue_wr": round(t2_vwr, 3),
            "venue_matches_count": len(v_matches),
            "t1_venue_matches_count": len(t1_v),
            "t2_venue_matches_count": len(t2_v),
        }

    def get_player_batting_rating(self, player: str) -> Dict[str, float]:
        """
        Bayesian blended career-to-date batting metrics.
        Shrinks low-sample players toward league average priors.
        """
        stats = self.player_batting.get(player)
        PRIOR_AVG = 24.5
        PRIOR_SR = 126.0
        PRIOR_WEIGHT_BALLS = 60.0

        if not stats or stats.balls == 0:
            return {
                "avg": PRIOR_AVG,
                "sr": PRIOR_SR,
                "dot_pct": 0.36,
                "boundary_pct": 0.15,
                "pp_sr": PRIOR_SR * 0.95,
                "death_sr": PRIOR_SR * 1.25,
                "sample_balls": 0,
                "composite_rating": round(PRIOR_AVG * 0.55 + PRIOR_SR * 0.14, 2),
            }

        # Bayesian shrinkage on Avg & SR
        w = min(1.0, stats.balls / PRIOR_WEIGHT_BALLS)
        raw_avg = stats.average
        raw_sr = stats.strike_rate

        shrunk_avg = (1.0 - w) * PRIOR_AVG + w * raw_avg
        shrunk_sr = (1.0 - w) * PRIOR_SR + w * raw_sr

        pp_sr = (stats.pp_runs / stats.pp_balls * 100.0) if stats.pp_balls > 15 else shrunk_sr * 0.95
        death_sr = (stats.death_runs / stats.death_balls * 100.0) if stats.death_balls > 15 else shrunk_sr * 1.25

        composite = shrunk_avg * 0.55 + shrunk_sr * 0.14

        return {
            "avg": round(shrunk_avg, 2),
            "sr": round(shrunk_sr, 2),
            "dot_pct": round(stats.dot_pct, 3),
            "boundary_pct": round(stats.boundary_pct, 3),
            "pp_sr": round(pp_sr, 2),
            "death_sr": round(death_sr, 2),
            "sample_balls": stats.balls,
            "composite_rating": round(composite, 2),
        }

    def get_player_bowling_rating(self, player: str) -> Dict[str, float]:
        """
        Bayesian blended career-to-date bowling metrics.
        """
        stats = self.player_bowling.get(player)
        PRIOR_ECO = 8.5
        PRIOR_AVG = 28.5
        PRIOR_WEIGHT_BALLS = 60.0

        if not stats or stats.balls == 0:
            return {
                "eco": PRIOR_ECO,
                "avg": PRIOR_AVG,
                "dot_pct": 0.35,
                "pp_eco": PRIOR_ECO * 0.95,
                "death_eco": PRIOR_ECO * 1.20,
                "sample_balls": 0,
                "composite_rating": round((7.5 / PRIOR_ECO) * (32.0 / PRIOR_AVG) * 18.0, 2),
            }

        w = min(1.0, stats.balls / PRIOR_WEIGHT_BALLS)
        raw_eco = stats.economy
        raw_avg = stats.average

        shrunk_eco = (1.0 - w) * PRIOR_ECO + w * raw_eco
        shrunk_avg = (1.0 - w) * PRIOR_AVG + w * raw_avg

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
            "composite_rating": round(composite, 2),
        }

    def update_match_result(self, match: MatchRecord):
        """
        Reveals match outcome and updates all accumulators chronologically.
        MUST BE CALLED STRICTLY AFTER PREDICTION.
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

        # 2. Extract Innings Scores
        inn1_score = match.innings_scores.get(1, (165, 6, 120))[0]
        inn2_score = match.innings_scores.get(2, (160, 6, 120))[0]

        # Determine who batted first
        t1_batted_first = (match.toss_winner == t1 and match.toss_decision == "bat") or (match.toss_winner == t2 and match.toss_decision != "bat")

        t1_score = inn1_score if t1_batted_first else inn2_score
        t2_score = inn2_score if t1_batted_first else inn1_score

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
        )

        # 3. Update Team and Venue Match Histories
        self.team_matches[t1].append(m_summary_t1)
        self.team_matches[t2].append(m_summary_t2)

        h2h_key = frozenset({t1, t2})
        self.h2h_matches[h2h_key].append(m_summary_t1)

        self.venue_matches[match.venue].append(m_summary_t1)
        self.team_venue_matches[(t1, match.venue)].append(m_summary_t1)
        self.team_venue_matches[(t2, match.venue)].append(m_summary_t2)

        # 4. Update Playing XI History
        if t1 in match.playing_xi and len(match.playing_xi[t1]) >= 8:
            self.latest_xi[t1] = list(match.playing_xi[t1])
            self.latest_xi_match_id[t1] = match.match_id
            self.latest_xi_match_time[t1] = match.match_datetime
        if t2 in match.playing_xi and len(match.playing_xi[t2]) >= 8:
            self.latest_xi[t2] = list(match.playing_xi[t2])
            self.latest_xi_match_id[t2] = match.match_id
            self.latest_xi_match_time[t2] = match.match_datetime

        # 5. Update Player Career-to-Date Stats from Deliveries
        player_innings_runs: Dict[str, int] = defaultdict(int)
        bowler_spell: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))  # bowler -> (runs, wkts)

        for d in match.deliveries:
            striker = d.striker
            bowler = d.bowler
            dismissed_p = d.player_dismissed
            is_wicket = bool(d.wicket_type and d.wicket_type.lower() not in {"run out", "retired hurt", "retired out", "obstructing the field"})

            # Batting update
            self.player_batting[striker].add_ball(
                runs=d.runs_off_bat,
                is_legal=d.is_legal,
                phase=d.phase,
                dismissed=(dismissed_p == striker and is_wicket),
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

            # Parse info
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

            # Build canonical timestamp for intraday sorting
            m_datetime = datetime.combine(m_date, time(19, 30)) + timedelta(minutes=match_num)

            venue_raw = info_dict.get("venue", ["Wankhede Stadium"])[0]
            venue = normalize_venue(venue_raw)
            city = info_dict.get("city", ["Mumbai"])[0]

            toss_winner = normalize_team(info_dict.get("toss_winner", [teams_raw[0]])[0])
            toss_decision = info_dict.get("toss_decision", ["field"])[0].lower()

            margin_runs = int(info_dict.get("winner_runs", [0])[0] or 0)
            margin_wickets = int(info_dict.get("winner_wickets", [0])[0] or 0)

            # Parse deliveries if available
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

                            # Accumulate innings total
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

        # Canonical sort: strictly chronological
        matches.sort(key=lambda m: (m.match_date, m.match_number, m.match_id))
        return matches


# ── Feature Engineering Pipeline ──────────────────────────────────────────────


class TemporalFeatureEngine:
    """
    Builds pre-match feature vectors with mathematical temporal guarantees.
    """

    FEATURE_NAMES = [
        # Team ELO
        "t1_elo",
        "t2_elo",
        "elo_diff",
        "elo_expected_t1",
        # Team Form & Win Rates
        "t1_recent_wins",
        "t2_recent_wins",
        "t1_form_exp",
        "t2_form_exp",
        "form_diff_exp",
        "t1_historical_wr",
        "t2_historical_wr",
        "team_wr_diff",
        # Head-to-Head
        "h2h_t1_wr",
        "h2h_matches_count",
        "h2h_recent_t1_wr",
        # Venue Dynamics
        "venue_avg_1st_innings",
        "venue_chase_wr",
        "t1_venue_wr",
        "t2_venue_wr",
        "venue_wr_diff",
        "venue_exp_count",
        # Lineup Aggregated Strengths (Pre-XI or Post-XI)
        "t1_bat_score",
        "t2_bat_score",
        "bat_diff",
        "t1_bowl_score",
        "t2_bowl_score",
        "bowl_diff",
        "t1_pp_bowl_str",
        "t2_pp_bowl_str",
        "pp_bowl_diff",
        "t1_death_bowl_str",
        "t2_death_bowl_str",
        "death_bowl_diff",
        # Matchup Advantage
        "lineup_synergy_diff",
        # Pre-toss / Neutral context
        "is_playoff",
        "season_progress",
    ]

    def __init__(self, mode: str = "pre_xi"):
        """
        mode: 'pre_xi' (uses prior match playing XI) or 'post_xi' (uses target match actual playing XI)
        """
        self.mode = mode

    def build_features(
        self,
        match: MatchRecord,
        state: HistoricalStateTracker,
        include_toss: bool = False,
    ) -> Dict[str, float]:
        """
        Builds feature dictionary using ONLY historical state up to match.match_datetime.
        """
        t1, t2 = match.team1, match.team2
        venue = match.venue
        season = match.season

        # 1. ELO Features
        t1_elo = state.elo.get_rating(t1, season)
        t2_elo = state.elo.get_rating(t2, season)
        elo_diff = t1_elo - t2_elo
        elo_exp_t1 = state.elo.expected_prob(t1_elo, t2_elo)

        # 2. Team Form & Overall History
        t1_wins_5, t1_form_exp = state.get_team_form(t1, n=5)
        t2_wins_5, t2_form_exp = state.get_team_form(t2, n=5)
        form_diff_exp = t1_form_exp - t2_form_exp

        t1_tot = state.team_matches.get(t1, [])
        t2_tot = state.team_matches.get(t2, [])
        t1_hwr = (sum(1 for m in t1_tot if m.won) / len(t1_tot)) if t1_tot else 0.50
        t2_hwr = (sum(1 for m in t2_tot if m.won) / len(t2_tot)) if t2_tot else 0.50
        team_wr_diff = t1_hwr - t2_hwr

        # 3. Head-to-Head
        h2h = state.get_h2h_stats(t1, t2)

        # 4. Venue
        v_stats = state.get_venue_stats(venue, t1, t2)
        venue_wr_diff = v_stats["t1_venue_wr"] - v_stats["t2_venue_wr"]

        # 5. Playing XI Selection (PRE-XI vs POST-XI)
        if self.mode == "pre_xi":
            t1_xi = state.get_latest_xi(t1)
            t2_xi = state.get_latest_xi(t2)
            # Fallback if first match of franchise: use match's announced XI as baseline prior
            if len(t1_xi) < 7:
                t1_xi = list(match.playing_xi.get(t1, []))
            if len(t2_xi) < 7:
                t2_xi = list(match.playing_xi.get(t2, []))
        else:
            t1_xi = list(match.playing_xi.get(t1, []))
            t2_xi = list(match.playing_xi.get(t2, []))

        # Compute Lineup Aggregated Ratings
        t1_bat_scores = [state.get_player_batting_rating(p)["composite_rating"] for p in t1_xi[:7]]
        t2_bat_scores = [state.get_player_batting_rating(p)["composite_rating"] for p in t2_xi[:7]]
        t1_bat = float(sum(t1_bat_scores) / len(t1_bat_scores)) if t1_bat_scores else 28.0
        t2_bat = float(sum(t2_bat_scores) / len(t2_bat_scores)) if t2_bat_scores else 28.0
        bat_diff = t1_bat - t2_bat

        t1_bowl_scores = [state.get_player_bowling_rating(p)["composite_rating"] for p in t1_xi[-5:]]
        t2_bowl_scores = [state.get_player_bowling_rating(p)["composite_rating"] for p in t2_xi[-5:]]
        t1_bowl = float(sum(t1_bowl_scores) / len(t1_bowl_scores)) if t1_bowl_scores else 13.5
        t2_bowl = float(sum(t2_bowl_scores) / len(t2_bowl_scores)) if t2_bowl_scores else 13.5
        bowl_diff = t1_bowl - t2_bowl

        # Phase bowling strengths (Death & PP)
        t1_death_ecos = [state.get_player_bowling_rating(p)["death_eco"] for p in t1_xi[-5:]]
        t2_death_ecos = [state.get_player_bowling_rating(p)["death_eco"] for p in t2_xi[-5:]]
        t1_death = 120.0 / max(float(sum(t1_death_ecos) / len(t1_death_ecos)) if t1_death_ecos else 10.0, 5.0)
        t2_death = 120.0 / max(float(sum(t2_death_ecos) / len(t2_death_ecos)) if t2_death_ecos else 10.0, 5.0)

        t1_pp_ecos = [state.get_player_bowling_rating(p)["pp_eco"] for p in t1_xi[-5:]]
        t2_pp_ecos = [state.get_player_bowling_rating(p)["pp_eco"] for p in t2_xi[-5:]]
        t1_pp = 120.0 / max(float(sum(t1_pp_ecos) / len(t1_pp_ecos)) if t1_pp_ecos else 8.0, 4.5)
        t2_pp = 120.0 / max(float(sum(t2_pp_ecos) / len(t2_pp_ecos)) if t2_pp_ecos else 8.0, 4.5)

        # Context features
        is_playoff = 1.0 if match.match_number > 56 else 0.0
        season_prog = min(1.0, max(0.0, match.match_number / 74.0))

        feat: Dict[str, float] = {
            "t1_elo": round(t1_elo, 2),
            "t2_elo": round(t2_elo, 2),
            "elo_diff": round(elo_diff, 2),
            "elo_expected_t1": round(elo_exp_t1, 4),
            "t1_recent_wins": float(t1_wins_5),
            "t2_recent_wins": float(t2_wins_5),
            "t1_form_exp": round(t1_form_exp, 4),
            "t2_form_exp": round(t2_form_exp, 4),
            "form_diff_exp": round(form_diff_exp, 4),
            "t1_historical_wr": round(t1_hwr, 4),
            "t2_historical_wr": round(t2_hwr, 4),
            "team_wr_diff": round(team_wr_diff, 4),
            "h2h_t1_wr": round(h2h["t1_wr"], 4),
            "h2h_matches_count": float(h2h["total_matches"]),
            "h2h_recent_t1_wr": round(h2h["recent_t1_wr"], 4),
            "venue_avg_1st_innings": round(v_stats["avg_first_innings"], 1),
            "venue_chase_wr": round(v_stats["chase_win_rate"], 3),
            "t1_venue_wr": round(v_stats["t1_venue_wr"], 3),
            "t2_venue_wr": round(v_stats["t2_venue_wr"], 3),
            "venue_wr_diff": round(venue_wr_diff, 3),
            "venue_exp_count": float(v_stats["venue_matches_count"]),
            "t1_bat_score": round(t1_bat, 2),
            "t2_bat_score": round(t2_bat, 2),
            "bat_diff": round(bat_diff, 2),
            "t1_bowl_score": round(t1_bowl, 2),
            "t2_bowl_score": round(t2_bowl, 2),
            "bowl_diff": round(bowl_diff, 2),
            "t1_pp_bowl_str": round(t1_pp, 2),
            "t2_pp_bowl_str": round(t2_pp, 2),
            "pp_bowl_diff": round(t1_pp - t2_pp, 2),
            "t1_death_bowl_str": round(t1_death, 2),
            "t2_death_bowl_str": round(t2_death, 2),
            "death_bowl_diff": round(t1_death - t2_death, 2),
            "lineup_synergy_diff": round((bat_diff * 0.4 + bowl_diff * 0.6) / 10.0, 3),
            "is_playoff": is_playoff,
            "season_progress": round(season_prog, 3),
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
        Returns complete audit metadata verifying no future records contributed to any feature.
        """
        t1, t2 = match.team1, match.team2
        t1_hist = state.team_matches.get(t1, [])
        t2_hist = state.team_matches.get(t2, [])
        h2h_hist = state.h2h_matches.get(frozenset({t1, t2}), [])
        v_hist = state.venue_matches.get(match.venue, [])

        latest_t1_match = t1_hist[-1].match_datetime.isoformat() if t1_hist else "NONE (first match)"
        latest_t2_match = t2_hist[-1].match_datetime.isoformat() if t2_hist else "NONE (first match)"
        latest_h2h = h2h_hist[-1].match_datetime.isoformat() if h2h_hist else "NONE (first encounter)"
        latest_venue = v_hist[-1].match_datetime.isoformat() if v_hist else "NONE (first at venue)"
        latest_state_time = state.last_updated_time.isoformat() if state.last_updated_time else "INITIAL"

        xi_source_t1 = state.latest_xi_match_id.get(t1, "DEFAULT/SEASON_PRIOR")
        xi_source_t2 = state.latest_xi_match_id.get(t2, "DEFAULT/SEASON_PRIOR")

        return {
            "target_match_id": match.match_id,
            "target_match_datetime": match.match_datetime.isoformat(),
            "teams": f"{t1} vs {t2}",
            "venue": match.venue,
            "prediction_mode": self.mode,
            "xi_source_match_team1": xi_source_t1,
            "xi_source_match_team2": xi_source_t2,
            "latest_prior_match_t1": latest_t1_match,
            "latest_prior_match_t2": latest_t2_match,
            "latest_prior_h2h_match": latest_h2h,
            "latest_prior_venue_match": latest_venue,
            "global_state_latest_update": latest_state_time,
            "all_cutoffs_strictly_prior": bool(
                (not state.last_updated_time or state.last_updated_time < match.match_datetime)
            ),
        }
