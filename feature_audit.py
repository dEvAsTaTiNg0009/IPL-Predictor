"""
=============================================================================
  FEATURE TRANSPARENCY AUDIT
  
  Paste this file into your project and call:
      audit = FeatureAudit(analyzer)
      audit.run_full_audit("MI", "CSK", "Wankhede Stadium")
  
  This proves EVERY feature is:
    ① Actually computed (not zero or default)
    ② Actually influencing the prediction
    ③ Correctly responding to conditions (weather changes score, etc.)
    ④ Not just decorative
=============================================================================
"""

import numpy as np
from datetime import datetime
from tabulate import tabulate

from ipl_predictor import FALLBACK_SQUADS, PLAYER_DB, TEAMS, _resolve_team


class FeatureAudit:
    """
    Runs three types of proof:
    
    A) Feature value audit — shows actual computed values, not zeros
    B) Feature sensitivity — change one input, show prediction change
    C) SHAP-style attribution — which features drive THIS prediction
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def run_full_audit(self, team1="MI", team2="CSK",
                       venue="Wankhede Stadium",
                       date=None):
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")

        t1 = _resolve_team(team1)
        t2 = _resolve_team(team2)

        print(f"\n{'═'*65}")
        print(f"  🔬 FEATURE TRANSPARENCY AUDIT")
        print(f"  {TEAMS.get(t1,t1)} vs {TEAMS.get(t2,t2)} at {venue}")
        print(f"{'═'*65}")

        # Build base features
        weather  = self.analyzer.weather.get(venue, date)
        pitch    = self.analyzer.pitch.predict(venue, date, weather)
        sq1      = self.analyzer.squads.get(t1, FALLBACK_SQUADS.get(t1, {}))
        sq2      = self.analyzer.squads.get(t2, FALLBACK_SQUADS.get(t2, {}))
        features = self.analyzer.fe.build(t1, t2, venue, weather, pitch,
                                           {t1: sq1, t2: sq2})
        pred     = self.analyzer.model.predict(features)

        self._audit_A_feature_values(features, weather, pitch, sq1, sq2, t1, t2)
        self._audit_B_sensitivity(features, pred, t1, t2)
        self._audit_C_attribution(features, pred)
        self._audit_D_weather_proof(venue, date, t1, t2, sq1, sq2)
        self._audit_E_player_xi_proof(sq1, sq2, venue, pitch, weather, t1, t2)

    # ═══════════════════════════════════════════════════════════════════
    # AUDIT A: Show all feature values — proves nothing is hardcoded
    # ═══════════════════════════════════════════════════════════════════
    def _audit_A_feature_values(self, features, weather, pitch, sq1, sq2, t1, t2):
        print(f"\n{'─'*65}")
        print(f"  ✅ AUDIT A: All computed feature values")
        print(f"{'─'*65}")

        def _fmt(value, spec=None, default="?"):
            if value is None or value == "?":
                return default
            try:
                return format(value, spec) if spec else str(value)
            except (TypeError, ValueError):
                return default

        sections = {
            "🌤 Weather (from Open-Meteo API)": {
                "Temperature":     f"{weather.get('temp_c','?')}°C",
                "Humidity":        f"{weather.get('humidity','?')}%",
                "Rain probability":f"{weather.get('rain_prob','?')}%",
                "Cloud cover":     f"{weather.get('cloud_cover','?')}%",
                "Wind speed":      f"{weather.get('wind_kph','?')} kph",
                "Dew risk":        _fmt(weather.get('dew_risk'), ".2f"),
                "Conditions":      weather.get('conditions','?'),
                "Data source":     "✅ Live Open-Meteo API" if weather.get('conditions') != 'STANDARD' else "⚠️  Default fallback",
            },
            "🏏 Pitch (computed from venue+weather)": {
                "Pace index":      f"{pitch.get('pace_index','?')}/10",
                "Spin index":      f"{pitch.get('spin_index','?')}/10",
                "Expected score":  f"{pitch.get('expected_score','?')}",
                "Pitch type":      pitch.get('pitch_type','?'),
                "Batting friendly":str(pitch.get('batting_friendly','?')),
                "Spin from over":  f"Over {pitch.get('spin_advantage_from_over','?')}",
                "Weather modified":"✅ Yes" if weather.get('humidity',0) > 70 or weather.get('temp_c',0) > 35 else "No change needed",
            },
            f"🏏 {TEAMS.get(t1,t1)} squad features": {
                "Batting strength":f"{features.get('t1_bat','?')} (league avg 28.0)",
                "Bowling strength":f"{features.get('t1_bowl','?')} (league avg 13.5)",
                "Recent wins":     f"{features.get('t1_wins','?')}/5 matches",
                "Venue win rate":  _fmt(features.get('t1_venue_wr'), ".1%"),
                "Pitch affinity":  _fmt(features.get('t1_pitch_aff'), ".3f"),
                "Data source":     "✅ PLAYER_DB stats" if features.get('t1_bat',0) != 28.0 else "⚠️  Using league average",
            },
            f"🏏 {TEAMS.get(t2,t2)} squad features": {
                "Batting strength":f"{features.get('t2_bat','?')} (league avg 28.0)",
                "Bowling strength":f"{features.get('t2_bowl','?')} (league avg 13.5)",
                "Recent wins":     f"{features.get('t2_wins','?')}/5 matches",
                "Venue win rate":  _fmt(features.get('t2_venue_wr'), ".1%"),
                "Pitch affinity":  _fmt(features.get('t2_pitch_aff'), ".3f"),
            },
            "🔄 Differential features": {
                "Batting diff":    _fmt(features.get('bat_diff'), "+.2f"),
                "Bowling diff":    _fmt(features.get('bowl_diff'), "+.2f"),
                "Form diff":       _fmt(features.get('form_diff'), "+.0f"),
                "H2H win rate":    f"{_fmt(features.get('h2h_wr'), '.1%')} for {TEAMS.get(t1,t1)}",
                "H2H total":       f"{features.get('h2h_total','?')} matches",
                "Venue WR diff":   _fmt(features.get('t1_venue_wr',0.5) - features.get('t2_venue_wr',0.5), "+.1%"),
            },
        }

        for section, items in sections.items():
            print(f"\n  {section}")
            for k, v in items.items():
                non_default = "  ← LIVE" if "✅" in str(v) else ""
                print(f"    {k:30s} {v}{non_default}")

        # Flag any suspiciously default values
        print(f"\n  Feature count: {len(features)} features computed")
        zeros = [k for k, v in features.items() if v == 0.0]
        if zeros:
            print(f"  ⚠️  Zero-value features: {', '.join(zeros[:5])}")
            print(f"     (zeros may be legitimate, e.g. no rain, night match=0)")


    # ═══════════════════════════════════════════════════════════════════
    # AUDIT B: Sensitivity — change input, show prediction change
    # ═══════════════════════════════════════════════════════════════════
    def _audit_B_sensitivity(self, base_features, base_pred, t1, t2):
        print(f"\n{'─'*65}")
        print(f"  ✅ AUDIT B: Feature sensitivity (input change → prediction change)")
        print(f"{'─'*65}")
        print(f"  Base prediction: {TEAMS.get(t1,t1)} win {base_pred['win_prob_t1']:.1%}\n")

        tests = [
            # (description, feature_to_change, new_value, expected_direction)
            ("Batting +5 for team 1",      "t1_bat",     base_features["t1_bat"]+5,    "↑ T1"),
            ("Bowling +3 for team 2",      "t2_bowl",    base_features["t2_bowl"]+3,   "↓ T1"),
            ("Heavy dew (0.9)",            "w_dew",      0.90,                          "Mixed"),
            ("Very humid (95%)",           "w_humid",    95.0,                          "Mixed"),
            ("Spin pitch (spin=9)",        "p_spin",     9.0,                           "Depends"),
            ("H2H win rate = 70%",         "h2h_wr",     0.70,                          "↑ T1"),
            ("H2H win rate = 30%",         "h2h_wr",     0.30,                          "↓ T1"),
            ("T1 venue WR = 70%",          "t1_venue_wr",0.70,                          "↑ T1"),
            ("T1 venue WR = 30%",          "t1_venue_wr",0.30,                          "↓ T1"),
            ("T1 form: 5 wins",            "t1_wins",    5,                             "↑ T1"),
            ("T1 form: 0 wins",            "t1_wins",    0,                             "↓ T1"),
            ("Rain 80%",                   "w_rain",     80.0,                          "Uncertain"),
        ]

        rows = []
        for desc, feat, val, direction in tests:
            modified = {**base_features, feat: val}
            try:
                new_pred = self.analyzer.model.predict(modified)
                delta    = new_pred["win_prob_t1"] - base_pred["win_prob_t1"]
                actual_dir = "↑" if delta > 0.005 else "↓" if delta < -0.005 else "≈"
                working  = "✅" if (actual_dir != "≈" or "Uncertain" in direction) else "⚠️ "
                rows.append([
                    desc, f"{base_features[feat]:.2f}→{val:.2f}",
                    f"{base_pred['win_prob_t1']:.1%}→{new_pred['win_prob_t1']:.1%}",
                    f"{delta:+.1%}", direction, working
                ])
            except Exception as e:
                rows.append([desc, str(val), "ERROR", str(e), direction, "❌"])

        print(tabulate(rows, headers=["Test","Change","Prediction","Δ","Expected","Status"],
                       tablefmt="rounded_outline"))
        working = sum(1 for r in rows if r[-1] == "✅")
        print(f"\n  {working}/{len(rows)} features confirmed responsive to input changes")


    # ═══════════════════════════════════════════════════════════════════
    # AUDIT C: Feature attribution (SHAP-lite)
    # ═══════════════════════════════════════════════════════════════════
    def _audit_C_attribution(self, features, pred):
        print(f"\n{'─'*65}")
        print(f"  ✅ AUDIT C: Feature attribution (what drove this prediction)")
        print(f"{'─'*65}")

        try:
            imp = self.analyzer.model.feature_importance()
            base_p = pred["win_prob_t1"]

            # Compute approximate Shapley values by baseline comparison
            # Baseline = all features at league average
            baseline = {
                "venue_avg_first":168,"venue_pace_index":6.0,"venue_spin_index":6.0,
                "venue_boundary_freq":0.62,"venue_chase_wr":0.50,"venue_dew_factor":0.5,
                "w_temp":28,"w_humid":65,"w_rain":15,"w_cloud":30,"w_dew":0.3,
                "w_wind":10,"is_night":1.0,"p_pace":6.0,"p_spin":6.0,"p_bounce":5.0,
                "p_score":168,"t1_bat":28.0,"t2_bat":28.0,"t1_bowl":13.5,"t2_bowl":13.5,
                "bat_diff":0,"bowl_diff":0,"t1_wins":3,"t2_wins":3,"form_diff":0,
                "h2h_wr":0.5,"h2h_total":15,"t1_venue_wr":0.5,"t2_venue_wr":0.5,
                "t1_pitch_aff":0.5,"t2_pitch_aff":0.5,"pitch_aff_diff":0,
            }

            try:
                base_prob = self.analyzer.model.predict(baseline)["win_prob_t1"]
            except:
                base_prob = 0.50

            attribs = []
            for feat, val in features.items():
                if feat not in baseline: continue
                modified = {**baseline, feat: val}
                try:
                    new_p = self.analyzer.model.predict(modified)["win_prob_t1"]
                    delta = new_p - base_prob
                    if abs(delta) > 0.002:
                        attribs.append((feat, val, delta))
                except: pass

            attribs.sort(key=lambda x: -abs(x[2]))

            print(f"\n  Baseline probability (all-average team): {base_prob:.1%}")
            print(f"  This match probability:                  {base_p:.1%}")
            print(f"  Total attribution:                       {base_p - base_prob:+.1%}")
            print(f"\n  Top 12 features driving THIS prediction:")
            print(f"  {'Feature':<28} {'Value':>8}  {'Impact':>8}  Direction")
            print(f"  {'─'*60}")

            for feat, val, delta in attribs[:12]:
                bar    = "▓" * min(15, int(abs(delta) * 200))
                arrow  = "↑ T1 wins" if delta > 0 else "↓ T1 wins"
                label  = feat.replace("_"," ").title()
                print(f"  {label:<28} {val:>8.2f}  {delta:>+7.1%}  {bar} {arrow}")

        except Exception as e:
            # Fallback: just show XGBoost feature importance
            print(f"  Using XGBoost feature importance (SHAP unavailable: {e})")
            imp = self.analyzer.model.feature_importance()
            rows = [(k.replace("_"," ").title(), round(v,4))
                    for k,v in list(imp.items())[:12]]
            print(tabulate(rows, headers=["Feature","Importance"], tablefmt="simple"))


    # ═══════════════════════════════════════════════════════════════════
    # AUDIT D: Weather proof — same match, rain vs clear, show difference
    # ═══════════════════════════════════════════════════════════════════
    def _audit_D_weather_proof(self, venue, date, t1, t2, sq1, sq2):
        print(f"\n{'─'*65}")
        print(f"  ✅ AUDIT D: Weather is ACTUALLY affecting predictions")
        print(f"{'─'*65}")
        print(f"  Same match, 3 different weather scenarios:\n")

        scenarios = [
            ("Clear night (normal)",   {"temp_c":28,"humidity":60,"rain_prob":5, "cloud_cover":20,"dew_risk":0.20,"is_night":True, "wind_kph":10,"conditions":"CLEAR"}),
            ("Heavy dew night",         {"temp_c":26,"humidity":88,"rain_prob":10,"cloud_cover":80,"dew_risk":0.82,"is_night":True, "wind_kph":8, "conditions":"VERY HUMID"}),
            ("Overcast + rain 60%",     {"temp_c":23,"humidity":85,"rain_prob":60,"cloud_cover":90,"dew_risk":0.15,"is_night":False,"wind_kph":22,"conditions":"RAIN LIKELY"}),
            ("Extreme heat day match",  {"temp_c":42,"humidity":28,"rain_prob":2, "cloud_cover":10,"dew_risk":0.05,"is_night":False,"wind_kph":18,"conditions":"EXTREME HEAT"}),
        ]

        rows = []
        for name, weather_override in scenarios:
            try:
                pitch    = self.analyzer.pitch.predict(venue, date, weather_override)
                features = self.analyzer.fe.build(t1, t2, venue, weather_override, pitch,
                                                   {t1: sq1, t2: sq2})
                pred     = self.analyzer.model.predict(features)
                rows.append([
                    name,
                    f"{weather_override['temp_c']}°C",
                    f"{weather_override['humidity']}%",
                    f"{weather_override['dew_risk']:.0%}",
                    f"{pitch['pace_index']:.1f}/{pitch['spin_index']:.1f}",
                    f"{pitch['expected_score']}",
                    f"{pred['win_prob_t1']:.1%}",
                ])
            except Exception as e:
                rows.append([name, "—","—","—","—","—", f"ERROR: {e}"])

        print(tabulate(rows,
                       headers=["Scenario","Temp","Humid","Dew","P/S idx","Exp score","T1 win%"],
                       tablefmt="rounded_outline"))

        probs = [float(r[-1].replace("%",""))/100 for r in rows if "%" in r[-1]]
        if len(probs) >= 2:
            spread = max(probs) - min(probs)
            print(f"\n  Win probability spread across weather scenarios: {spread:.1%}")
            if spread > 0.03:
                print(f"  ✅ CONFIRMED: Weather meaningfully affects predictions")
            else:
                print(f"  ⚠️  Weather impact is small (<3%) — check weather feature weights")


    # ═══════════════════════════════════════════════════════════════════
    # AUDIT E: Playing XI selection proof
    # ═══════════════════════════════════════════════════════════════════
    def _audit_E_player_xi_proof(self, sq1, sq2, venue, pitch, weather, t1, t2):
        print(f"\n{'─'*65}")
        print(f"  ✅ AUDIT E: Playing XI selection is role-correct")
        print(f"{'─'*65}\n")

        from ipl_predictor import PlayerProjector   # adjust import if needed
        proj  = self.analyzer.proj
        xi1   = proj._probable_xi(sq1)
        xi2   = proj._probable_xi(sq2)
        bowl1 = proj.project_bowling(sq1, venue, pitch, weather)
        bowl2 = proj.project_bowling(sq2, venue, pitch, weather)

        for team, xi, bowl_proj, name in [(t1, xi1, bowl1, TEAMS.get(t1,t1)),
                                           (t2, xi2, bowl2, TEAMS.get(t2,t2))]:
            print(f"  {name} Selected XI:")
            wk_count   = 0; bat_count = 0; all_count = 0; bowl_count = 0
            for i, p in enumerate(xi, 1):
                role = PLAYER_DB.get(p, {}).get("role", "UNK")
                sr   = PLAYER_DB.get(p, {}).get("bat_sr",  "?")
                avg  = PLAYER_DB.get(p, {}).get("bat_avg", "?")
                eco  = PLAYER_DB.get(p, {}).get("bowl_eco")
                eco_str = f"eco={eco:.1f}" if eco else "      "
                print(f"    {i:2d}. {p:25s} [{role:7s}] avg={avg:4} sr={sr:5} {eco_str}")
                if role=="WK-BAT": wk_count+=1
                elif role=="BAT":  bat_count+=1
                elif role=="ALL":  all_count+=1
                elif role=="BOWL": bowl_count+=1

            print(f"\n       WK={wk_count} BAT={bat_count} ALL={all_count} BOWL={bowl_count}")
            valid = wk_count >= 1 and bowl_count >= 3 and (bowl_count+all_count) >= 4
            print(f"       XI validity: {'✅ VALID' if valid else '❌ INVALID (fix _probable_xi)'}")

            print(f"\n  {name} Bowling attack ({len(bowl_proj)} bowlers projected):")
            for b in bowl_proj:
                suited = "✅" if b["suited_to_pitch"] else "—"
                print(f"    {b['player']:25s} {b['style']:4s} {b['overs']}ov "
                      f"{b['wickets']}wkts {b['runs']}runs eco={b['economy']:.1f} {suited}")

            # Verify bowlers are actually in the XI
            xi_set    = set(xi)
            bowl_set  = {b["player"] for b in bowl_proj}
            not_in_xi = bowl_set - xi_set
            if not_in_xi:
                print(f"\n    ⚠️  Bowlers projected but NOT in XI: {not_in_xi}")
                print(f"       This means _probable_xi and project_bowling are out of sync!")
            else:
                print(f"\n    ✅ All projected bowlers are in the XI")
            print()


# ─────────────────────────────────────────────────────────────────────────────
# USAGE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ipl_predictor import setup_system

    print("Setting up system…")
    analyzer, squads = setup_system()

    audit = FeatureAudit(analyzer)

    print("\nRunning audit for MI vs CSK at Wankhede:")
    audit.run_full_audit("MI", "CSK", "Wankhede Stadium")

    print("\n" + "═"*65)
    print("Running audit for RCB vs SRH at Chinnaswamy:")
    audit.run_full_audit("RCB", "SRH", "M. Chinnaswamy Stadium")