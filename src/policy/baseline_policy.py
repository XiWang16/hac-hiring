"""
Confidence threshold collaboration policy.

Implements the cost-based deferral threshold from Lecture 1, Section 1.6.
The optimal threshold balances the cost of an automated error (C_ERR) against
the cost of escalating to a human reviewer (C_ESC).

Threshold derivation (Lecture 1 Eq 1.8):
    tau_star = 1 - C_esc / C_err

At tau_star = 0.85 (with C_esc=0.15, C_err=1.0):
  - If the model's calibrated confidence P(y=1|x) >= tau_star: automate.
  - Otherwise: defer to the human reviewer (action a1 — no AI support shown).

This provides a simple, interpretable baseline. The proposed policies (RAPS,
Mozannar-Sontag) aim to improve on this by routing more intelligently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


class BaselinePolicy:
    """Confidence threshold deferral policy.

    The threshold tau_star is derived from the cost structure:
        tau_star = 1 - C_esc / C_err       (Lecture 1 Eq 1.8)

    If calibrated P(y=1|x) >= tau_star: model decides (automate).
    Otherwise: human decides using action a1 (no support provided).

    Parameters
    ----------
    c_err : float
        Cost of an automated wrong decision. Default: config.C_ERR.
    c_esc : float
        Cost of escalating to the human reviewer. Default: config.C_ESC.
    """

    def __init__(self, c_err: float = config.C_ERR, c_esc: float = config.C_ESC):
        self.c_err = c_err
        self.c_esc = c_esc
        self.tau_star = 1.0 - c_esc / c_err

    def run(
        self,
        test_df: pd.DataFrame,
        model,
        human,
        probs: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Execute the confidence threshold policy on the test set.

        Implements Lecture 1 Section 1.6 routing rule.

        Args:
            test_df : test split DataFrame (must contain 'pass_bar', 'subgroup').
            model   : fitted model with predict_proba(X) -> shape (n, 2).
                      If probs is provided, model is not called.
            human   : HumanReviewer instance.
            probs   : optional pre-computed probabilities shape (n, 2).
                      Useful when the ensemble probabilities were already computed.

        Returns:
            pd.DataFrame with columns:
                true_label, model_prob, set_size, decision, final_pred,
                correct, subgroup
            (set_size is None for the baseline — column kept for schema parity
             with RAPS and MS policies so evaluation functions are policy-agnostic.)
        """
        from src.data.lsac import get_X, FEATURE_COLS  # lazy import to avoid circularity

        if probs is None:
            # model must support predict_proba(X) -> (n, 2)
            from src.data.lsac import load_lsac
            raise ValueError(
                "Pass pre-computed probs to BaselinePolicy.run() to avoid "
                "reloading/refitting. Use ensemble.predict_proba(X_test) "
                "stacked to shape (n, 2) first."
            )

        true_labels = test_df["pass_bar"].values
        results = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            true_label = int(true_labels[i])
            model_prob = float(probs[i, 1])

            if model_prob >= self.tau_star:
                decision = "automate"
                final_pred = int(np.argmax(probs[i]))
            else:
                decision = "defer"
                final_pred = human.predict(row, "a1", true_label)

            results.append(
                {
                    "true_label": true_label,
                    "model_prob": model_prob,
                    "set_size":   None,
                    "decision":   decision,
                    "final_pred": final_pred,
                    "correct":    int(final_pred == true_label),
                    "subgroup":   row["subgroup"],
                }
            )

        return pd.DataFrame(results)
