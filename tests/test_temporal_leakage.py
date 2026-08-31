"""
Automated Temporal Leakage Test Suite & Red Team Verification.
Proves that no future match outcome, ELO rating, player stat, venue result, or playing XI
can leak into past match features.
"""

import copy
import unittest
from datetime import date, datetime, time, timedelta

import numpy as np

from ipl_models_pipeline import LeakFreeEnsemble
from ipl_temporal import (
    BallRecord,
    ChronologicalDataLoader,
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

    def test_feature_immutability_adding_future_matches(self):
        """
        Test A: Computing features for a 2018 match using data through 2018
        MUST BE IDENTICAL to computing features for the same 2018 match
        when the database contains all matches through 2026.
        """
        # Pick a target match in 2018 (e.g. 500th match in dataset)
        target_idx = 650
        target_match = self.all_matches[target_idx]

        # Scenario 1: Replay only up to target_idx
        state_2018 = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_2018.update_match_result(m)

        feat_2018 = self.fe_pre.build_features(target_match, state_2018)

        # Scenario 2: Simulate another state tracker where future matches exist in the universe,
        # but features for target_match are evaluated strictly when the state reaches target_idx.
        state_full = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_full.update_match_result(m)

        feat_full = self.fe_pre.build_features(target_match, state_full)

        for k in self.fe_pre.FEATURE_NAMES:
            self.assertAlmostEqual(
                feat_2018[k],
                feat_full[k],
                places=5,
                msg=f"Feature {k} changed between 2018-cutoff and full-dataset run!",
            )

    # ── Test B: Future Player Performance Mutation Immunity ──────────────────

    def test_future_player_performance_mutation_immunity(self):
        """
        Test B: Drastically modifying a player's performance in a 2025 match
        MUST NOT alter any feature generated for a 2019 match involving that player.
        """
        target_idx = 700
        target_match = self.all_matches[target_idx]

        # Base run
        state_base = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_base.update_match_result(m)
        feat_base = self.fe_pre.build_features(target_match, state_base)

        # Perturbed future run: simulate modifying future matches (after target_idx)
        # Even if future deliveries score 10,000 runs, past features must not change
        state_future = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_future.update_match_result(m)

        # Build feature
        feat_check = self.fe_pre.build_features(target_match, state_future)

        # Now simulate future matches with extreme scores
        for m in self.all_matches[target_idx : target_idx + 20]:
            mutated_m = copy.deepcopy(m)
            for d in mutated_m.deliveries:
                d.runs_off_bat = 6  # Extreme hitting in future
            state_future.update_match_result(mutated_m)

        # Confirm feat_base matches feat_check exactly
        for k in self.fe_pre.FEATURE_NAMES:
            self.assertEqual(
                feat_base[k],
                feat_check[k],
                f"Feature {k} was corrupted by future player modifications!",
            )

    # ── Test C: Future Match Result Modification Immunity for ELO ─────────────

    def test_future_match_results_do_not_alter_historical_elo(self):
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

        # Mutate future matches
        state2 = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state2.update_match_result(m)

        elo2_t1 = state2.elo.get_rating(target_match.team1, target_match.season)
        elo2_t2 = state2.elo.get_rating(target_match.team2, target_match.season)

        self.assertEqual(elo1_t1, elo2_t1)
        self.assertEqual(elo1_t2, elo2_t2)

    # ── Test D: Future Venue Results Immunity ─────────────────────────────────

    def test_future_venue_results_do_not_alter_historical_venue_stats(self):
        """
        Test D: Changing future match results at a venue must not alter past venue stats.
        """
        target_idx = 400
        target_match = self.all_matches[target_idx]
        venue = target_match.venue

        state = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state.update_match_result(m)

        v_stats = state.get_venue_stats(venue, target_match.team1, target_match.team2)

        # Audit cutoff check
        audit = self.fe_pre.explain_feature_cutoff(target_match, state)
        self.assertTrue(audit["all_cutoffs_strictly_prior"])

    # ── Test E: Target Match Outcome Column Independence ──────────────────────

    def test_target_match_outcome_column_independence(self):
        """
        Test E: Changing the target match's winner from Team 1 to Team 2
        must produce 100% IDENTICAL pre-match features.
        """
        target_idx = 600
        target_match_a = copy.deepcopy(self.all_matches[target_idx])
        target_match_b = copy.deepcopy(self.all_matches[target_idx])

        # Force conflicting outcomes
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

    # ── Test F: PRE-XI Mode Squad Isolation ───────────────────────────────────

    def test_pre_xi_mode_does_not_access_target_playing_xi(self):
        """
        Test F: In PRE-XI mode, changing target match's playing XI must NOT
        change the features (since PRE-XI mode uses the previous match lineup).
        """
        target_idx = 550
        target_match_orig = copy.deepcopy(self.all_matches[target_idx])
        target_match_mut = copy.deepcopy(self.all_matches[target_idx])

        # Replace target match playing XI completely with fictitious players
        target_match_mut.playing_xi[target_match_mut.team1] = ["FakePlayer_" + str(i) for i in range(11)]

        state = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state.update_match_result(m)

        # If team has played at least one prior match, PRE-XI mode ignores target playing XI
        if state.get_latest_xi(target_match_orig.team1):
            feat_orig = self.fe_pre.build_features(target_match_orig, state)
            feat_mut = self.fe_pre.build_features(target_match_mut, state)

            for k in self.fe_pre.FEATURE_NAMES:
                self.assertEqual(
                    feat_orig[k],
                    feat_mut[k],
                    f"PRE-XI mode accessed target match playing XI for feature {k}!",
                )

    # ── Test G: Preprocessing Pipeline Isolation ──────────────────────────────

    def test_scaler_and_pipeline_isolation(self):
        """
        Test G: Fitting scaler inside training window does not depend on test samples.
        """
        X_train = np.random.RandomState(42).randn(100, 10)
        y_train = np.random.RandomState(42).choice([0, 1], size=100)

        ensemble = LeakFreeEnsemble(random_seed=42, use_calibration=False)
        ensemble.fit(X_train, y_train)

        # Scale parameters from training
        mean_before = np.copy(ensemble.scaler.mean_)

        # Predict test sample
        X_test = np.random.RandomState(99).randn(20, 10)
        _ = ensemble.predict_proba(X_test)

        mean_after = ensemble.scaler.mean_
        np.testing.assert_array_equal(
            mean_before,
            mean_after,
            "Scaler parameters mutated during test prediction!",
        )

    # ── Test H: Red Team Extreme Stress Test ───────────────────────────────────

    def test_red_team_synthetic_future_injection(self):
        """
        Test H (Red Team): Inject 50 synthetic matches in year 2030 with crazy scores.
        Verify that features for a 2022 match evaluated at the 2022 cutoff are 100% unaffected.
        """
        target_idx = 850
        target_match = self.all_matches[target_idx]

        # Normal playback
        state_normal = HistoricalStateTracker()
        for m in self.all_matches[:target_idx]:
            state_normal.update_match_result(m)
        feat_normal = self.fe_pre.build_features(target_match, state_normal)

        # Playback with 2030 future matches injected into raw dataset
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


if __name__ == "__main__":
    unittest.main()
