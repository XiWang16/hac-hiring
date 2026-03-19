"""
Logistic regression baseline with temperature scaling.

Temperature scaling follows Guo et al. (2017) "On Calibration of Modern
Neural Networks". A scalar temperature T is fitted on the validation set by
minimising negative log-likelihood (NLL).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid
from sklearn.linear_model import LogisticRegression

from src.evaluation.calibration import compute_ece


class LogisticBaseline:
    """Logistic regression with post-hoc temperature scaling.

    Attributes:
        lr_        : fitted sklearn LogisticRegression
        temperature_ : scalar temperature T (default 1.0 before calibration)
    """

    def __init__(self, random_state: int = 42):
        self.lr_ = LogisticRegression(
            max_iter=1000, random_state=random_state, solver="lbfgs"
        )
        self.temperature_ = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticBaseline":
        """Fit logistic regression on training data.

        Args:
            X_train : shape (n, d)
            y_train : shape (n,) binary labels

        Returns:
            self
        """
        self.lr_.fit(X_train, y_train)
        return self

    def _logits(self, X: np.ndarray) -> np.ndarray:
        """Return raw log-odds for P(y=1) — shape (n,)."""
        return self.lr_.decision_function(X)

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> "LogisticBaseline":
        """Fit scalar temperature T by minimising NLL on the validation set.

        Implements Lecture 1 temperature scaling (post-hoc calibration).
        Uses scipy.optimize.minimize_scalar(method='brent') over T in (0.01, 10).

        Logs pre- and post-calibration ECE for diagnostic purposes.

        Args:
            X_val : shape (n, d)
            y_val : shape (n,) binary labels

        Returns:
            self
        """
        logits = self._logits(X_val)

        def nll(T: float) -> float:
            probs = expit(logits / T)
            probs = np.clip(probs, 1e-9, 1 - 1e-9)
            return -np.mean(y_val * np.log(probs) + (1 - y_val) * np.log(1 - probs))

        # Pre-calibration ECE
        raw_probs = expit(logits)
        ece_before = compute_ece(raw_probs, y_val)
        print(f"[LogisticBaseline] ECE before calibration: {ece_before:.4f}")

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.temperature_ = float(result.x)

        # Post-calibration ECE
        cal_probs = expit(logits / self.temperature_)
        ece_after = compute_ece(cal_probs, y_val)
        print(
            f"[LogisticBaseline] Temperature T={self.temperature_:.4f} | "
            f"ECE after calibration: {ece_after:.4f}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities, shape (n, 2).

        Column 0 = P(y=0), column 1 = P(y=1).
        Applies temperature scaling: p = sigmoid(logit / T).

        Implements Lecture 1 temperature-scaled probability.
        """
        logits = self._logits(X)
        p1 = expit(logits / self.temperature_)
        return np.stack([1 - p1, p1], axis=1)

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Return uncalibrated probabilities (temperature = 1.0), shape (n, 2)."""
        logits = self._logits(X)
        p1 = expit(logits)
        return np.stack([1 - p1, p1], axis=1)
