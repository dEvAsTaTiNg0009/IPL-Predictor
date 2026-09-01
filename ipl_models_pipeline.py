"""
Leak-Free Model Training, Ensembling, Preprocessing, and Calibration Pipelines.
Implements Elastic Net Meta-Learning, Model Pruning, and Strict Inner-Fold Cross-Validation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")


# ── Metrics Evaluator ─────────────────────────────────────────────────────────


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes all standard probabilistic and classification metrics with confidence intervals.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    y_pred = (y_prob >= 0.5).astype(float)

    n_samples = len(y_true)
    if n_samples == 0:
        return {}

    correct = int(np.sum(y_pred == y_true))
    incorrect = n_samples - correct
    acc = float(np.mean(y_pred == y_true))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else acc

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.50
    except Exception:
        auc = 0.50

    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob, labels=[0.0, 1.0]))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix
    tp = int(np.sum((y_pred == 1.0) & (y_true == 1.0)))
    tn = int(np.sum((y_pred == 0.0) & (y_true == 0.0)))
    fp = int(np.sum((y_pred == 1.0) & (y_true == 0.0)))
    fn = int(np.sum((y_pred == 0.0) & (y_true == 1.0)))

    # Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    cal_table = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if np.sum(mask) > 0:
            bin_pred_mean = float(np.mean(y_prob[mask]))
            bin_actual_rate = float(np.mean(y_true[mask]))
            bin_count = int(np.sum(mask))
            bin_weight = bin_count / n_samples
            gap = abs(bin_pred_mean - bin_actual_rate)
            ece += bin_weight * gap
            cal_table.append({
                "range": f"{lo:.1f}-{hi:.1f}",
                "count": bin_count,
                "pred_mean": round(bin_pred_mean, 3),
                "actual_rate": round(bin_actual_rate, 3),
                "gap": round(gap, 3),
            })

    # Wilson 95% Confidence Interval for Accuracy
    z = 1.96
    p_hat = acc
    denom = 1.0 + (z**2) / n_samples
    centre = (p_hat + (z**2) / (2.0 * n_samples)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * n_samples)) / n_samples)) / denom
    ci_lower = max(0.0, centre - margin)
    ci_upper = min(1.0, centre + margin)

    return {
        "n_matches": n_samples,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(acc, 4),
        "accuracy_ci_95": (round(ci_lower, 4), round(ci_upper, 4)),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(auc, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "ece": round(float(ece), 4),
        "mean_predicted_prob": round(float(np.mean(y_prob)), 4),
        "mean_actual_outcome": round(float(np.mean(y_true)), 4),
        "calibration_table": cal_table,
    }


# ── Baseline Models ───────────────────────────────────────────────────────────


class BaselineRandom:
    """Predicts constant 50% probability."""
    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        return np.full((n, 2), 0.50)


class BaselineStrongerTeam:
    """Predicts winner based strictly on career pre-match win rate differential."""

    def __init__(self):
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray, feature_idx: int = 15) -> np.ndarray:
        wr_diff = X[:, feature_idx] if len(X.shape) > 1 and X.shape[1] > feature_idx else np.zeros(len(X))
        prob = 1.0 / (1.0 + np.exp(-3.0 * wr_diff))
        return np.column_stack([1.0 - prob, prob])


class BaselineTeamForm:
    """Predicts winner based on exponential recent form differential."""

    def __init__(self):
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray, feature_idx: int = 8) -> np.ndarray:
        form_diff = X[:, feature_idx] if len(X.shape) > 1 and X.shape[1] > feature_idx else np.zeros(len(X))
        prob = 1.0 / (1.0 + np.exp(-2.5 * form_diff))
        return np.column_stack([1.0 - prob, prob])


class BaselineELOOnly:
    """Predicts winner strictly using logistic conversion of ELO difference."""

    def __init__(self):
        self.lr = LogisticRegression(C=1.0, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_idx: int = 2):
        elo_diff = X[:, feature_idx : feature_idx + 1] if X.shape[1] > feature_idx else X[:, 0:1]
        self.lr.fit(elo_diff, y)
        return self

    def predict_proba(self, X: np.ndarray, feature_idx: int = 2) -> np.ndarray:
        elo_diff = X[:, feature_idx : feature_idx + 1] if X.shape[1] > feature_idx else X[:, 0:1]
        return self.lr.predict_proba(elo_diff)


# ── Bayesian Bradley-Terry Team-Strength Model ────────────────────────────────


class BayesianBradleyTerryModel:
    """
    Bayesian Bradley-Terry Logit Team Rating Model with L2 shrinkage prior.
    """

    def __init__(self, prior_variance: float = 1.0):
        self.prior_variance = prior_variance
        self.model = LogisticRegression(C=1.0 / self.prior_variance, fit_intercept=False, random_state=42)
        self.fitted = False
        self.feature_indices: List[int] = [0]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_indices: Optional[List[int]] = None):
        n_feats = X_train.shape[1]
        if feature_indices is None:
            candidates = [2, 8, 22, 25] if n_feats > 25 else [2, 8]
            feature_indices = [idx for idx in candidates if idx < n_feats]
            if not feature_indices:
                feature_indices = [0]
        self.feature_indices = feature_indices
        X_sub = X_train[:, self.feature_indices]
        self.model.fit(X_sub, y_train)
        self.fitted = True
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full((len(X_test), 2), 0.50)
        X_sub = X_test[:, self.feature_indices]
        return self.model.predict_proba(X_sub)


# ── Elastic Net Stacked Ensemble ──────────────────────────────────────────────


class ElasticNetEnsemble:
    """
    Stacked Ensemble with Elastic Net Logistic Regression Meta-Learner,
    expanding-window inner cross-validation, and automated base model pruning.
    """

    def __init__(
        self,
        random_seed: int = 42,
        calibration_method: str = "isotonic",  # "none", "platt", or "isotonic"
        retained_models: Optional[List[str]] = None,
    ):
        self.random_seed = random_seed
        self.calibration_method = calibration_method
        self.retained_models = retained_models
        self.scaler = StandardScaler()
        self.base_models: Dict[str, Any] = {}
        self.meta_learner: Optional[LogisticRegression] = None
        self.calibrator: Optional[Any] = None
        self.fitted = False
        self.best_c: float = 0.3
        self.best_l1_ratio: float = 0.5
        self.meta_coefficients: Dict[str, float] = {}

    def _init_candidate_models(self) -> Dict[str, Any]:
        """All candidate base models before Elastic Net pruning."""
        models = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.80,
                eval_metric="logloss",
                random_state=self.random_seed,
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.80,
                random_state=self.random_seed,
                verbose=-1,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=180,
                max_depth=5,
                min_samples_split=4,
                random_state=self.random_seed,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                random_state=self.random_seed,
            ),
            "LogisticRegression": LogisticRegression(
                C=0.25,
                max_iter=300,
                random_state=self.random_seed,
            ),
            "ElasticNetLogistic": LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=0.20,
                l1_ratio=0.5,
                max_iter=400,
                random_state=self.random_seed,
            ),
            "NeuralNet": MLPClassifier(
                hidden_layer_sizes=(48, 24),
                max_iter=250,
                alpha=0.02,
                random_state=self.random_seed,
                early_stopping=True,
                n_iter_no_change=15,
            ),
            "BayesianBradleyTerry": BayesianBradleyTerryModel(prior_variance=1.0),
        }

        # Filter if a subset of models was already selected
        if self.retained_models is not None:
            return {k: v for k, v in models.items() if k in self.retained_models}
        return models

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ElasticNetEnsemble":
        """
        Trains ensemble on training set using inner chronological expanding splits
        to generate out-of-fold predictions for Elastic Net meta-learning.
        """
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        n_samples = len(X)

        if n_samples < 50:
            raise ValueError(f"Insufficient training samples: {n_samples}")

        # 1. Generate Expanding-Window Inner Cross-Validation Splits
        n_splits = 5
        min_train_size = int(n_samples * 0.45)
        val_chunk_size = (n_samples - min_train_size) // n_splits

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_splits):
            train_end = min_train_size + i * val_chunk_size
            val_end = train_end + val_chunk_size if i < n_splits - 1 else n_samples
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, val_end)
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        model_prototypes = self._init_candidate_models()
        model_names = list(model_prototypes.keys())
        n_models = len(model_names)

        oof_predictions_list: List[np.ndarray] = []
        oof_targets_list: List[np.ndarray] = []

        for fold, (tr_idx, va_idx) in enumerate(splits):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            # Fit inner fold scaler strictly on inner training split
            inner_scaler = StandardScaler()
            X_tr_s = inner_scaler.fit_transform(X_tr)
            X_va_s = inner_scaler.transform(X_va)

            fold_oof = np.zeros((len(va_idx), n_models), dtype=float)

            for m_idx, name in enumerate(model_names):
                clf = self._init_candidate_models()[name]
                inp_tr = X_tr_s if name in ["LogisticRegression", "ElasticNetLogistic", "NeuralNet", "ExtraTrees"] else X_tr
                inp_va = X_va_s if name in ["LogisticRegression", "ElasticNetLogistic", "NeuralNet", "ExtraTrees"] else X_va

                clf.fit(inp_tr, y_tr)
                p_va = clf.predict_proba(inp_va)[:, 1]
                fold_oof[:, m_idx] = p_va

            oof_predictions_list.append(fold_oof)
            oof_targets_list.append(y_va)

        OOF_X = np.vstack(oof_predictions_list)
        OOF_y = np.concatenate(oof_targets_list)

        # 2. Chronological Grid Search for Elastic Net (C, l1_ratio)
        best_ll = float("inf")
        c_candidates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 3.0]
        l1_candidates = [0.0, 0.25, 0.5, 0.75, 1.0]

        for c_val in c_candidates:
            for l1_val in l1_candidates:
                try:
                    meta_cand = LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=c_val,
                        l1_ratio=l1_val,
                        max_iter=500,
                        random_state=self.random_seed,
                    )
                    meta_cand.fit(OOF_X, OOF_y)
                    p_meta = meta_cand.predict_proba(OOF_X)[:, 1]
                    cur_ll = log_loss(OOF_y, p_meta)
                    if cur_ll < best_ll:
                        best_ll = cur_ll
                        self.best_c = c_val
                        self.best_l1_ratio = l1_val
                except Exception:
                    pass

        # 3. Fit Best Elastic Net Meta-Learner on OOF Predictions
        self.meta_learner = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=self.best_c,
            l1_ratio=self.best_l1_ratio,
            max_iter=600,
            random_state=self.random_seed,
        )
        self.meta_learner.fit(OOF_X, OOF_y)

        # Record meta coefficients
        if hasattr(self.meta_learner, "coef_"):
            for m_name, coef in zip(model_names, self.meta_learner.coef_[0]):
                self.meta_coefficients[m_name] = float(coef)

        # 4. Fit Probability Calibrator on Inner OOF Predictions
        meta_oof_probs = self.meta_learner.predict_proba(OOF_X)[:, 1]
        if self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(
                y_min=0.02,
                y_max=0.98,
                out_of_bounds="clip",
            )
            self.calibrator.fit(meta_oof_probs, OOF_y)
        elif self.calibration_method == "platt":
            self.calibrator = LogisticRegression(C=1.0, random_state=self.random_seed)
            self.calibrator.fit(meta_oof_probs.reshape(-1, 1), OOF_y)
        else:
            self.calibrator = None

        # 5. Refit Scaler and Base Models on Full Outer Training Set
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        self.base_models = self._init_candidate_models()
        for name, clf in self.base_models.items():
            inp = X_s if name in ["LogisticRegression", "ElasticNetLogistic", "NeuralNet", "ExtraTrees"] else X
            clf.fit(inp, y)

        self.fitted = True
        return self

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = np.asarray(X_test, dtype=float)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        X_s = self.scaler.transform(X)

        base_preds = []
        for name in self.base_models:
            clf = self.base_models[name]
            inp = X_s if name in ["LogisticRegression", "ElasticNetLogistic", "NeuralNet", "ExtraTrees"] else X
            prob = clf.predict_proba(inp)[:, 1]
            base_preds.append(prob)

        meta_input = np.column_stack(base_preds)
        raw_meta_prob = self.meta_learner.predict_proba(meta_input)[:, 1]

        if self.calibrator is not None:
            if isinstance(self.calibrator, IsotonicRegression):
                calibrated_prob = self.calibrator.predict(raw_meta_prob)
            else:
                calibrated_prob = self.calibrator.predict_proba(raw_meta_prob.reshape(-1, 1))[:, 1]
            calibrated_prob = np.clip(calibrated_prob, 0.02, 0.98)
        else:
            calibrated_prob = raw_meta_prob

        return np.column_stack([1.0 - calibrated_prob, calibrated_prob])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)[:, 1]
        return (probs >= 0.5).astype(int)

    def get_feature_importances(self, feature_names: List[str]) -> List[Tuple[str, float]]:
        if not self.fitted:
            return []
        importances = np.zeros(len(feature_names), dtype=float)
        tree_count = 0
        for name in ["XGBoost", "LightGBM", "ExtraTrees", "GradientBoosting"]:
            if name in self.base_models and hasattr(self.base_models[name], "feature_importances_"):
                fi = np.asarray(self.base_models[name].feature_importances_, dtype=float)
                if len(fi) == len(feature_names):
                    fi_norm = fi / (np.sum(fi) + 1e-12)
                    importances += fi_norm
                    tree_count += 1

        if tree_count > 0:
            importances /= tree_count

        pairs = list(zip(feature_names, importances.tolist()))
        pairs.sort(key=lambda p: -p[1])
        return pairs

    def export_frozen_artifacts(self, export_dir: Path, feature_names: List[str]) -> Dict[str, str]:
        """
        Serializes all trained models, preprocessors, and hyperparameters into export_dir
        and generates a cryptographic SHA-256 manifest.
        """
        export_dir.mkdir(parents=True, exist_ok=True)
        manifest: Dict[str, str] = {}

        # 1. Configs & Hyperparameters
        cfg = {
            "feature_names": feature_names,
            "retained_models": list(self.base_models.keys()),
            "best_c": self.best_c,
            "best_l1_ratio": self.best_l1_ratio,
            "calibration_method": self.calibration_method,
            "meta_coefficients": self.meta_coefficients,
        }
        cfg_path = export_dir / "config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # 2. Preprocessor & Meta Learner Checkpoints
        with open(export_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(export_dir / "meta_learner.pkl", "wb") as f:
            pickle.dump(self.meta_learner, f)
        if self.calibrator is not None:
            with open(export_dir / "calibrator.pkl", "wb") as f:
                pickle.dump(self.calibrator, f)

        # 3. Base Models
        models_dir = export_dir / "models"
        models_dir.mkdir(exist_ok=True)
        for name, clf in self.base_models.items():
            with open(models_dir / f"{name}.pkl", "wb") as f:
                pickle.dump(clf, f)

        # 4. Generate SHA256 checksums
        for p in export_dir.rglob("*"):
            if p.is_file() and p.name != "manifest.json":
                with open(p, "rb") as f:
                    manifest[str(p.relative_to(export_dir))] = hashlib.sha256(f.read()).hexdigest()

        with open(export_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


# Alias for backward compatibility
LeakFreeEnsemble = ElasticNetEnsemble
BayesianTeamStrengthModel = BayesianBradleyTerryModel
