"""
RAPS conformal prediction policy.

Implements Lecture 1, Sections 1.7.3 and 1.7.4.

RAPS = Regularised Adaptive Prediction Sets (Angelopoulos et al., 2021).
Reference: "Uncertainty Sets for Image Classifiers using Conformal Prediction".
ICLR 2021.

How it works as a COLLABORATION POLICY (not just a calibration tool):
1. Fit conformal threshold tau_hat on the CALIBRATION split (val set).
2. At test time, build a RAPS prediction set C_alpha(x) for each candidate.
3. Routing rule:
       |C_alpha(x)| <= RAPS_DEFER_THRESHOLD -> automate (model is certain)
       |C_alpha(x)| >  RAPS_DEFER_THRESHOLD -> defer to human with action a3
          (full explanation shown — appropriate when the model is uncertain)

Coverage guarantee (Lecture 1 Eq 1.9):
    P(y in C_alpha(x)) >= 1 - alpha

The penalty lambda_reg discourages large prediction sets; k_reg is the rank
at which the penalty begins to apply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


class RAPSPolicy:
    """RAPS conformal prediction policy for human-AI collaboration.

    Parameters
    ----------
    alpha          : target miscoverage level. Implements Lecture 1 Eq 1.9.
    lambda_reg     : regularisation penalty weight (RAPS parameter).
    k_reg          : rank at which penalty begins.
    defer_threshold: defer if |prediction set| > this value.
    """

    def __init__(
        self,
        alpha: float = config.RAPS_ALPHA,
        lambda_reg: float = config.RAPS_LAMBDA,
        k_reg: int = config.RAPS_K_REG,
        defer_threshold: int = config.RAPS_DEFER_THRESHOLD,
    ):
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        self.defer_threshold = defer_threshold
        self.tau_hat_: float | None = None

    def _nonconformity_score(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Compute RAPS nonconformity scores for calibration examples.

        For each example i with true class y_i:
          1. Sort class probabilities in descending order.
          2. Accumulate (probability + lambda penalty for ranks > k_reg)
             up to and including the rank of the true class.

        Implements Lecture 1 Section 1.7.3 (RAPS score definition).

        Args:
            probs  : shape (n, 2), column k = P(y=k).
            labels : shape (n,), binary ground truth.

        Returns:
            shape (n,) nonconformity scores.
        """
        n = len(labels)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]   # descending by prob
            true_class = labels[i]
            rank = int(np.where(sorted_idx == true_class)[0][0]) + 1   # 1-indexed
            cumsum = 0.0
            for j in range(1, rank + 1):
                class_j = sorted_idx[j - 1]
                cumsum += probs[i][class_j]
                if j > self.k_reg:
                    cumsum += self.lambda_reg
            scores[i] = cumsum
        return scores

    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray) -> "RAPSPolicy":
        """Compute conformal quantile tau_hat from calibration (val) set.

        tau_hat = quantile at level ceil((1-alpha)(n+1))/n of calibration scores.
        Implements coverage guarantee from Lecture 1 Eq 1.9.

        Args:
            val_probs  : shape (n, 2) calibrated probabilities on val set.
            val_labels : shape (n,) binary ground truth on val set.

        Returns:
            self
        """
        scores = self._nonconformity_score(val_probs, val_labels)
        n = len(scores)
        level = np.ceil((1 - self.alpha) * (n + 1)) / n
        level = min(level, 1.0)
        self.tau_hat_ = float(np.quantile(scores, level))
        print(f"[RAPSPolicy] tau_hat = {self.tau_hat_:.4f} (alpha={self.alpha})")
        return self

    def predict_set(self, probs: np.ndarray) -> list[list[int]]:
        """Build RAPS prediction sets for test points.

        Include the highest-probability classes in order until the penalised
        cumulative score exceeds tau_hat. Implements Lecture 1 Eq 1.11.

        Args:
            probs : shape (n, 2), column k = P(y=k).

        Returns:
            List of n lists, each containing 0, 1, or both class indices.

        Raises:
            RuntimeError: if fit() has not been called.
        """
        if self.tau_hat_ is None:
            raise RuntimeError("Call fit() before predict_set().")
        sets: list[list[int]] = []
        for i in range(len(probs)):
            sorted_idx = np.argsort(probs[i])[::-1]
            pred_set: list[int] = []
            cumsum = 0.0
            for j, class_j in enumerate(sorted_idx, start=1):
                cumsum += probs[i][class_j]
                if j > self.k_reg:
                    cumsum += self.lambda_reg
                pred_set.append(int(class_j))
                if cumsum >= self.tau_hat_:
                    break
            sets.append(pred_set)
        return sets

    def run(
        self,
        test_df: pd.DataFrame,
        model,
        human,
        test_probs: np.ndarray,
    ) -> pd.DataFrame:
        """Execute the RAPS collaboration policy on the test set.

        Routing rule (Lecture 1 Section 1.7.4):
            |C_alpha(x)| <= defer_threshold -> automate: predict argmax(probs)
            |C_alpha(x)| >  defer_threshold -> defer: human decides with action a3
                (full explanation shown — maximises human accuracy for uncertain cases)

        Args:
            test_df    : test split DataFrame.
            model      : unused (probs already computed); kept for API parity.
            human      : HumanReviewer instance.
            test_probs : shape (n, 2) calibrated ensemble probabilities.

        Returns:
            pd.DataFrame with columns:
                true_label, model_prob, set_size, decision, final_pred,
                correct, subgroup
        """
        sets = self.predict_set(test_probs)
        true_labels = test_df["pass_bar"].values
        results = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            pred_set = sets[i]
            set_size = len(pred_set)
            true_label = int(true_labels[i])

            if set_size <= self.defer_threshold:
                decision = "automate"
                # Use the highest-probability class (first in sorted order)
                final_pred = pred_set[0]
            else:
                decision = "defer"
                final_pred = human.predict(row, "a3", true_label)

            results.append(
                {
                    "true_label": true_label,
                    "model_prob": float(test_probs[i, 1]),
                    "set_size":   set_size,
                    "decision":   decision,
                    "final_pred": final_pred,
                    "correct":    int(final_pred == true_label),
                    "subgroup":   row["subgroup"],
                }
            )
        return pd.DataFrame(results)

    def empirical_coverage(
        self, test_probs: np.ndarray, test_labels: np.ndarray
    ) -> float:
        """Verify the marginal coverage guarantee on the test set.

        Computes P(true label in C_alpha(x)) on the test set.
        Should be >= 1 - alpha (Lecture 1 Eq 1.9).
        Report as a calibration diagnostic in the paper.

        Args:
            test_probs  : shape (n, 2) calibrated ensemble probabilities.
            test_labels : shape (n,) binary ground truth.

        Returns:
            Empirical coverage as a float in [0, 1].
        """
        sets = self.predict_set(test_probs)
        covered = sum(
            int(test_labels[i]) in sets[i] for i in range(len(test_labels))
        )
        return covered / len(test_labels)
