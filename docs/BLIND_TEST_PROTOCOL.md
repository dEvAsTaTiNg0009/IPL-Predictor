# Strict Blind Testing & Temporal Isolation Protocol 🛡️

**Project:** Stochastic Cricket Prediction (IPL)  
**Standard:** Strict Chronological Causality ($t < T$)  
**Development Window:** 2008–2025  
**Final True Holdout:** 2026  

---

## 1. Core Invariants & Information Boundaries

1. **Strict Temporal Causality:**  
   For any match $M$ occurring at timestamp $T$, the available information set is:
   $$\mathcal{I}(M_T) = \{d \in \mathcal{D} \mid \text{timestamp}(d) < T\}$$
   Under no circumstances may any data $d$ with $\text{timestamp}(d) \ge T$ enter feature extraction, dynamic ELO ratings, Bayesian priors, player statistics, venue histories, scaling transformations, hyperparameter selection, Elastic Net weights, or probability calibration.

2. **Strict Isolation of the 2026 Season:**  
   The 2026 IPL season serves as the **final untouched blind holdout**.
   - 2026 data is **NEVER** included in development walk-forward experiments, feature selection, base model pruning, hyperparameter optimization, Elastic Net meta-learner tuning, or calibration selection.
   - All design decisions, feature sets, hyperparameters, and ensemble structures are finalized and frozen strictly on 2008–2025 data.
   - The final model pipeline is trained on 2008–2025 and saved to `artifacts/final_2026_model/` along with a cryptographic SHA-256 manifest.
   - The 2026 season is then evaluated sequentially match-by-match loading the frozen artifacts.

---

## 2. Match Processing Sequence

Every match in both development walk-forward and the 2026 holdout must follow this exact sequence:

```
  1. Determine prediction timestamp T (e.g. 19:30 on match_date).
  2. Freeze HistoricalState up to T.
  3. Determine evaluation mode (PRE-XI vs POST-XI).
  4. In PRE-XI mode, query franchise's most recent prior match played before T to retrieve candidate XI.
  5. Compute causal features using ONLY frozen HistoricalState.
  6. Apply StandardScaler fitted strictly on the outer training fold (2008..Y-1).
  7. Generate base model predictions and Elastic Net meta-learner probability.
  8. Apply calibration fitted strictly on inner out-of-fold predictions.
  9. Record prediction and audit metadata (feature cutoffs and source timestamps).
 10. Reveal actual match result.
 11. Update player stats, team stats, ELO, H2H, venue records, and XI state.
 12. Advance to match T+1.
```

---

## 3. Benchmark Mode Definitions

- **MODE A: PRE-XI (Primary Benchmark):**
  - Evaluates pre-toss forecasting hours before the match.
  - Lineups are strictly resolved from each franchise's most recent prior match before timestamp $T$.
  - Target match actual playing XI, substitutes, toss winner, toss decision, and outcome are strictly sealed.
  
- **MODE B: POST-XI (Tactical Benchmark):**
  - Evaluates post-announcement forecasting after official XIs and toss decisions are declared.
  - Allowed: Announced target-match XI and toss decision.
  - Forbidden: Match outcome, deliveries, and in-game performance.

---

## 4. Frozen Artifact Manifest Protocol

Before evaluating the 2026 holdout:
1. All pipelines are trained on the complete 2008–2025 dataset.
2. Artifacts are serialized into `artifacts/final_2026_model/`:
   - `feature_config.json`: Selected feature list.
   - `selected_models.json`: Pruned base models.
   - `hyperparameters.json`: Base learner and Elastic Net hyperparameters.
   - `scaler.pkl`: StandardScaler fitted on 2008–2025.
   - `models/`: Trained base model checkpoints.
   - `meta_model.pkl`: Fitted Elastic Net Logistic Regression meta-learner.
   - `calibrator.pkl`: Out-of-fold probability calibrator.
   - `priors.json`: Dynamic priors computed on 2008–2025.
   - `manifest.json`: SHA-256 checksums of all serialized artifacts.
3. The 2026 evaluator validates the checksums before making predictions.
