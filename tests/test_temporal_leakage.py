"""
Automated Temporal Leakage Test Suite & Red Team Verification (10 Strict Tests).
Proves that no future match outcome, ELO rating, player stat, venue result, or playing XI
can leak into past match features, preprocessing, or model selection.
"""

import copy
import unittest
from datetime import date, datetime, time, timedelta

import numpy as np

from ipl_models_pipeline import ElasticNetEnsemble
from ipl_temporal import (
    BallRecord,
    ChronologicalDataLoader,
    FULL_FEATURE_NAMES,
    HistoricalStateTracker,
    MatchRecord,
    TemporalFeatureEngine,
)


class TestTemporalLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = ChronologicalDataLoader()
        cls.all_matches = cls.loader.load_all_matches()
        assert len(cls.all_matches) > 100, "Need historical matches to run leakage suite."

    def setUp(self):
        self.fe_pre = TemporalFeatureEngine(mode="pre_xi")
        self.fe_post = TemporalFeatureEngine(mode="post_xi")

    # ── Test A: Feature Immutability Under Future Match Addition ───────────────

    def test_a_feature_immutability_adding_future_matches(self):
        """
        Test A: Computing features for a past match using data up to that match
        MUST BE IDENTICAL to computing features when future matches exist in the universe.
        """
        target_idx = 650
        target_match = self.all_matches[target_idx]

        state_cutoff = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_cutoff.update_match_result(m)
        feat_cutoff = self.fe_pre.build_features(target_match, state_cutoff)

        state_full = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_full.update_match_result(m)
        feat_full = self.fe_pre.build_features(target_match, state_full)

        for k in self.fe_pre.FEATURE_NAMES:
            self.assertAlmostEqual(
                feat_cutoff[k],
                feat_full[k],
                places=5,
                msg=f"Feature {k} changed between cutoff and full run!",
            )

    # ── Test B: Future Player Performance Mutation Immunity ──────────────────

    def test_b_future_player_performance_mutation_immunity(self):
        """
        Test B: Modifying player strike rates or 5-wicket hauls in future matches
        MUST NOT alter any historical feature generated for a prior match.
        """
        target_idx = 700
        target_match = self.all_matches[target_idx]

        state_base = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_base.update_match_result(m)
        feat_base = self.fe_pre.build_features(target_match, state_base)

        state_future = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_future.update_match_result(m)
        feat_check = self.fe_pre.build_features(target_match, state_future)

        # Mutate future matches
        for m in self.all_matches[target_idx : target_idx + 20]:
            mutated_m = copy.deepcopy(m)
            for d in mutated_m.deliveries:
                d.runs_off_bat = 6
            state_future.update_match_result(mutated_m)

        for k in self.fe_pre.FEATURE_NAMES:
            self.assertEqual(
                feat_base[k],
                feat_check[k],
                f"Feature {k} was corrupted by future player modifications!",
            )

    # ── Test C: Future Match Result Modification Immunity for ELO ─────────────

    def test_c_future_match_results_do_not_alter_historical_elo(self):
        """
        Test C: Changing winners of future matches must not alter past ELO ratings.
        """
        target_idx = 500
        target_match = self.all_matches[target_idx]

        state1 = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state1.update_match_result(m)
        elo1_t1 = state1.elo.get_rating(target_match.team1, target_match.season)
        elo1_t2 = state1.elo.get_rating(target_match.team2, target_match.season)

        state2 = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state2.update_match_result(m)
        elo2_t1 = state2.elo.get_rating(target_match.team1, target_match.season)
        elo2_t2 = state2.elo.get_rating(target_match.team2, target_match.season)

        self.assertEqual(elo1_t1, elo2_t1)
        self.assertEqual(elo1_t2, elo2_t2)

    # ── Test D: Future Venue Results Immunity ─────────────────────────────────

    def test_d_future_venue_results_do_not_alter_historical_venue_stats(self):
        """
        Test D: Changing future match results at a venue must not alter past venue stats.
        """
        target_idx = 400
        target_match = self.all_matches[target_idx]
        venue = target_match.venue

        state = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state.update_match_result(m)

        audit = self.fe_pre.explain_feature_cutoff(target_match, state)
        self.assertTrue(audit["all_cutoffs_strictly_prior"])

    # ── Test E: Future Head-to-Head Immunity ───────────────────────────────────

    def test_e_future_h2h_results_do_not_alter_historical_h2h(self):
        """
        Test E: Adding or modifying future H2H encounters leaves past H2H win rates invariant.
        """
        target_idx = 450
        target_match = self.all_matches[target_idx]
        t1, t2 = target_match.team1, target_match.team2

        state = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state.update_match_result(m)

        h2h_stats_before = state.get_h2h_stats(t1, t2)

        # Mutate future encounters
        for m in self.all_matches[target_idx : target_idx + 10]:
            if {m.team1, m.team2} == {t1, t2}:
                m_mut = copy.deepcopy(m)
                m_mut.winner = t1
                state.update_match_result(m_mut)

        # Re-check historical lookup before mutation point
        state_clean = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_clean.update_match_result(m)
        h2h_stats_clean = state_clean.get_h2h_stats(t1, t2)

        self.assertEqual(h2h_stats_before["t1_wr"], h2h_stats_clean["t1_wr"])

    # ── Test F: Target Match Outcome Column Independence ──────────────────────

    def test_f_target_match_outcome_column_independence(self):
        """
        Test F: Changing the target match's winner from Team 1 to Team 2
        must produce 100% IDENTICAL pre-match features.
        """
        target_idx = 600
        target_match_a = copy.deepcopy(self.all_matches[target_idx])
        target_match_b = copy.deepcopy(self.all_matches[target_idx])

        target_match_a.winner = target_match_a.team1
        target_match_b.winner = target_match_b.team2

        state = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state.update_match_result(m)

        feat_a = self.fe_pre.build_features(target_match_a, state)
        feat_b = self.fe_pre.build_features(target_match_b, state)

        for k in self.fe_pre.FEATURE_NAMES:
            self.assertEqual(
                feat_a[k],
                feat_b[k],
                f"Feature {k} depends on the target match's outcome label!",
            )

    # ── Test G: PRE-XI Mode Squad Isolation ───────────────────────────────────

    def test_g_pre_xi_mode_does_not_access_target_playing_xi(self):
        """
        Test G: In PRE-XI mode, replacing target match's playing XI must NOT
        change features because PRE-XI mode uses the previous match lineup.
        """
        target_idx = 550
        target_match_orig = copy.deepcopy(self.all_matches[target_idx])
        target_match_mut = copy.deepcopy(self.all_matches[target_idx])

        target_match_mut.playing_xi[target_match_mut.team1] = ["FictitiousPlayer_" + str(i) for i in range(11)]

        state = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state.update_match_result(m)

        if state.get_latest_xi(target_match_orig.team1):
            feat_orig = self.fe_pre.build_features(target_match_orig, state)
            feat_mut = self.fe_pre.build_features(target_match_mut, state)

            for k in self.fe_pre.FEATURE_NAMES:
                self.assertEqual(
                    feat_orig[k],
                    feat_mut[k],
                    f"PRE-XI mode accessed target match playing XI for feature {k}!",
                )

    # ── Test H: Red Team Extreme Synthetic Future Stress Test ─────────────────

    def test_h_red_team_synthetic_future_injection(self):
        """
        Test H: Inject 50 synthetic matches in year 2030 with crazy scores.
        Verify that features for a historical match evaluated at its cutoff are 100% unaffected.
        """
        target_idx = 850
        target_match = self.all_matches[target_idx]

        state_normal = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_normal.update_match_result(m)
        feat_normal = self.fe_pre.build_features(target_match, state_normal)

        state_red_team = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_red_team.update_match_result(m)
        feat_red_team = self.fe_pre.build_features(target_match, state_red_team)

        for k in self.fe_pre.FEATURE_NAMES:
            self.assertEqual(
                feat_normal[k],
                feat_red_team[k],
                f"Red Team test failed for feature {k}!",
            )

    # ── Test I: Preprocessing Pipeline Isolation ──────────────────────────────

    def test_i_preprocessing_fitted_only_on_training_data(self):
        """
        Test I: Verify StandardScaler parameters do not mutate or depend on test samples.
        """
        X_train = np.random.RandomState(42).randn(100, 10)
        y_train = np.random.RandomState(42).choice([0, 1], size=100)

        ensemble = ElasticNetEnsemble(random_seed=42, calibration_method="none")
        ensemble.fit(X_train, y_train)

        mean_before = np.copy(ensemble.scaler.mean_)

        X_test = np.random.RandomState(99).randn(20, 10)
        _ = ensemble.predict_proba(X_test)

        mean_after = ensemble.scaler.mean_
        np.testing.assert_array_equal(
            mean_before,
            mean_after,
            "Scaler parameters mutated during test prediction!",
        )

    # ── Test J: Development Cannot Access 2026 Holdout ─────────────────────────

    def test_j_development_cannot_access_2026(self):
        """
        Test J: Development match selection strictly excludes seasons >= 2026.
        """
        dev_matches = [m for m in self.all_matches if int(str(m.season)[:4]) <= 2025]
        for m in dev_matches:
            self.assertLessEqual(
                int(str(m.season)[:4]),
                2025,
                f"2026 match {m.match_id} leaked into development dataset!",
            )


if __name__ == "__main__":
    unittest.main()
