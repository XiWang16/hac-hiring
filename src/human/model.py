"""
Simulated human reviewer for the HAC hiring pipeline.

The human model captures subgroup-varying accuracy and action-dependent
support effects. Noise is additive Gaussian (clipped to [0.05, 0.98]).

Actions:
    a1 : no decision support (human decides without AI output)
    a2 : model prediction shown (human sees AI recommendation)
    a3 : full explanation shown (e.g. SHAP values + prediction)

Subgroup definitions and accuracy parameters are specified in src/config.py.
Human accuracy levels are loosely calibrated to empirical findings from:
    Lai et al. (2020) "On Human Predictions with Explanations and Predictions
    of Machine Learning Models: A Case Study on Deception Detection". FAccT 2020.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.lsac import get_subgroup


class HumanReviewer:
    """Simulated human reviewer with subgroup-varying accuracy.

    Parameters
    ----------
    seed : int
        RNG seed for reproducibility. Defaults to config.RANDOM_SEED.
    """

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.rng = np.random.default_rng(seed)

    def predict(self, row: pd.Series, action: str, true_label: int) -> int:
        """Simulate a single human prediction.

        Computes the probability of a correct decision as:
            p_correct = clip(base_accuracy + support_effect + noise, 0.05, 0.98)

        where:
            base_accuracy  = config.HUMAN_ACCURACY[subgroup]
            support_effect = config.SUPPORT_EFFECT[(subgroup, action)]
            noise          ~ Normal(0, HUMAN_NOISE_SCALE)

        Args:
            row        : pd.Series representing a single candidate (must contain
                         columns needed by get_subgroup: race1, cluster, lsat_pct,
                         ugpa_pct).
            action     : one of {'a1', 'a2', 'a3'}.
            true_label : ground-truth label (0 or 1).

        Returns:
            Predicted label (0 or 1).
        """
        sg = get_subgroup(row)
        base = config.HUMAN_ACCURACY[sg]
        effect = config.SUPPORT_EFFECT.get((sg, action), 0.0)
        noise = self.rng.normal(0, config.HUMAN_NOISE_SCALE)
        p_correct = float(np.clip(base + effect + noise, 0.05, 0.98))
        return true_label if self.rng.random() < p_correct else 1 - true_label

    def predict_batch(
        self,
        df: pd.DataFrame,
        action: str,
        true_labels: np.ndarray,
    ) -> np.ndarray:
        """Vectorised version of predict().

        Args:
            df          : DataFrame of candidates (same schema as train/val/test_df).
            action      : one of {'a1', 'a2', 'a3'}.
            true_labels : shape (n,) array of ground-truth labels.

        Returns:
            np.ndarray of shape (n,) containing predicted labels (0 or 1).
        """
        return np.array(
            [
                self.predict(row, action, int(y))
                for (_, row), y in zip(df.iterrows(), true_labels)
            ]
        )
