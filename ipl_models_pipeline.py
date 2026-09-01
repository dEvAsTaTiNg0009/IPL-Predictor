"""
Leak-Free Model Training, Ensembling, Preprocessing, and Calibration Pipelines.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
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
    Computes all standard probabilistic and classification metrics.
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
    """Predicts a constant 50% probability."""
    def fit(self, X: np.ndarray, y: np.ndarray):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        p = np.full((n, 2), 0.50)
        return p


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
        elo_diff = X[:, feature_idx : feature_idx + 1]
        self.lr.fit(elo_diff, y)
        return self

    def predict_proba(self, X: np.ndarray, feature_idx: int = 2) -> np.ndarray:
        elo_diff = X[:, feature_idx : feature_idx + 1]
        return self.lr.predict_proba(elo_diff)


# ── Bayesian Bradley-Terry Team-Strength Model ────────────────────────────────


class BayesianTeamStrengthModel:
    """
    Bayesian Bradley-Terry Logit Team Rating Model.
    Shrinks team strengths towards a common league mean using L2 regularization prior.
    """

    def __init__(self, prior_variance: float = 1.0):
        self.prior_variance = prior_variance
        self.model = LogisticRegression(C=1.0 / self.prior_variance, fit_intercept=False, random_state=42)
        self.fitted = False
        self.feature_indices = [0]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_indices: Optional[List[int]] = None):
        n_feats = X_train.shape[1]
        if feature_indices is None:
            candidates = [2, 8, 32, 35] if n_feats > 35 else [2, 8]
            feature_indices = [idx for idx in candidates if idx < n_feats]
            if not feature_indices:
                feature_indices = [min(2, n_feats - 1)]
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


# ── Leak-Free Stacked Ensemble ────────────────────────────────────────────────


class LeakFreeEnsemble:
    """
    Stacked Ensemble with expanding-window inner cross-validation,
    strict preprocessing isolation, and isotonic probability calibration.
    """

    def __init__(self, random_seed: int = 42, use_calibration: bool = True):
        self.random_seed = random_seed
        self.use_calibration = use_calibration
        self.scaler = StandardScaler()
        self.base_models: Dict[str, Any] = {}
        self.meta_learner: Optional[LogisticRegression] = None
        self.calibrator: Optional[IsotonicRegression] = None
        self.fitted = False
        self.inner_cv_scores: Dict[str, float] = {}

    def _init_base_models(self) -> Dict[str, Any]:
        return {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=160,
                max_depth=4,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.80,
                eval_metric="logloss",
                random_state=self.random_seed,
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=160,
                max_depth=4,
                learning_rate=0.025,
                subsample=0.85,
                colsample_bytree=0.80,
                random_state=self.random_seed,
                verbose=-1,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=200,
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
            "NeuralNet": MLPClassifier(
                hidden_layer_sizes=(48, 24),
                max_iter=250,
                alpha=0.02,
                random_state=self.random_seed,
                early_stopping=True,
                n_iter_no_change=15,
            ),
        }

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LeakFreeEnsemble":
        """
        Trains the ensemble strictly on the provided training set using
        expanding-window walk-forward splits to generate out-of-fold predictions.
        """
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)
        n_samples = len(X)

        if n_samples < 50:
            raise ValueError(f"Insufficient training samples for nested ensemble: {n_samples}")

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

        model_prototypes = self._init_base_models()
        model_names = list(model_prototypes.keys())
        n_models = len(model_names)

        # Store OOF validation predictions
        val_indices_all: List[int] = []
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
                clf = self._init_base_models()[name]
                inp_tr = X_tr_s if name in ["LogisticRegression", "NeuralNet", "ExtraTrees"] else X_tr
                inp_va = X_va_s if name in ["LogisticRegression", "NeuralNet", "ExtraTrees"] else X_va

                clf.fit(inp_tr, y_tr)
                p_va = clf.predict_proba(inp_va)[:, 1]
                fold_oof[:, m_idx] = p_va

            val_indices_all.extend(va_idx.tolist())
            oof_predictions_list.append(fold_oof)
            oof_targets_list.append(y_va)

        OOF_X = np.vstack(oof_predictions_list)
        OOF_y = np.concatenate(oof_targets_list)

        # 2. Train Meta-Learner strictly on valid OOF predictions
        self.meta_learner = LogisticRegression(
            C=0.25,
            random_state=self.random_seed,
        )
        self.meta_learner.fit(OOF_X, OOF_y)

        # 3. Fit Isotonic Calibration on OOF meta predictions
        meta_oof_probs = self.meta_learner.predict_proba(OOF_X)[:, 1]
        if self.use_calibration:
            self.calibrator = IsotonicRegression(
                y_min=0.02,
                y_max=0.98,
                out_of_bounds="clip",
            )
            self.calibrator.fit(meta_oof_probs, OOF_y)
        else:
            self.calibrator = None

        # 4. Refit Scaler and Base Models on Full Outer Training Set
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        self.base_models = self._init_base_models()
        for name, clf in self.base_models.items():
            inp = X_s if name in ["LogisticRegression", "NeuralNet", "ExtraTrees"] else X
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
            inp = X_s if name in ["LogisticRegression", "NeuralNet", "ExtraTrees"] else X
            prob = clf.predict_proba(inp)[:, 1]
            base_preds.append(prob)

        meta_input = np.column_stack(base_preds)
        raw_meta_prob = self.meta_learner.predict_proba(meta_input)[:, 1]

        if self.calibrator is not None:
            calibrated_prob = self.calibrator.predict(raw_meta_prob)
            calibrated_prob = np.clip(calibrated_prob, 0.02, 0.98)
        else:
            calibrated_prob = raw_meta_prob

        return np.column_stack([1.0 - calibrated_prob, calibrated_prob])

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X_test)[:, 1]
        return (probs >= 0.5).astype(int)

    def get_feature_importances(self, feature_names: List[str]) -> List[Tuple[str, float]]:
        """Computes normalized feature importance across tree base models."""
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
