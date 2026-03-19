"""
Risk-coverage curve evaluation for human-AI collaboration policies.

A risk-coverage curve sweeps a deferral threshold and plots:
    coverage        = fraction of test instances the system handles automatically
    selective_risk  = error rate on the automated subset

Lower risk at the same coverage indicates a better deferral policy.

Reference: HAC_Lecture1.pdf — selective prediction and coverage-risk trade-off.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


def compute_risk_coverage_curve(
    results_df: pd.DataFrame,
    thresholds: np.ndarray = np.linspace(0, 1, 200),
) -> pd.DataFrame:
    """Compute risk-coverage curve for confidence-threshold-based policies.

    For each threshold t, treats instances with model_prob >= t as automated
    and measures error rate on that subset.

    Applies to both BaselinePolicy and RAPSPolicy outputs (both store
    model_prob in the results DataFrame).

    Args:
        results_df : output DataFrame from BaselinePolicy.run() or
                     RAPSPolicy.run() with columns 'model_prob', 'correct'.
        thresholds : array of confidence thresholds to sweep.

    Returns:
        pd.DataFrame with columns: threshold, coverage, selective_risk
    """
    model_probs = results_df["model_prob"].values
    correct = results_df["correct"].values
    n = len(results_df)

    rows = []
    for t in thresholds:
        automated = model_probs >= t
        cov = float(automated.sum() / n)
        if automated.any():
            risk = float(1.0 - correct[automated].mean())
        else:
            risk = float("nan")
        rows.append({"threshold": t, "coverage": cov, "selective_risk": risk})

    return pd.DataFrame(rows)


def compute_ms_risk_coverage_curve(
    logits: np.ndarray,
    true_labels: np.ndarray,
    human,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute risk-coverage curve for the Mozannar-Sontag policy.

    MS policy defers when logit[defer] > max(logit[0], logit[1]).
    The deferral margin is: logit[2] - max(logit[0], logit[1]).
    A higher threshold means fewer deferrals (more automation).

    Sweeps a threshold on the NEGATIVE deferral margin (so higher threshold
    = more automation), which aligns with the baseline curve's semantics.

    Args:
        logits      : shape (n, 3) raw logits from DeferralHead.
        true_labels : shape (n,) binary ground truth.
        human       : HumanReviewer instance (to get deferred predictions).
        test_df     : test split DataFrame (needed for human.predict).

    Returns:
        pd.DataFrame with columns: threshold, coverage, selective_risk
    """
    logits_t = torch.FloatTensor(logits)
    probs = F.softmax(logits_t, dim=1).numpy()  # (n, 3)

    # Deferral margin: positive means "prefer defer"
    deferral_margin = logits[:, 2] - np.maximum(logits[:, 0], logits[:, 1])

    # Model predictions (ignoring deferral class)
    model_preds = np.argmax(logits[:, :2], axis=1)  # 0 or 1
    n = len(true_labels)

    # Sweep: threshold on negative margin (so "automate if margin < t")
    thresholds = np.linspace(deferral_margin.min() - 0.1,
                              deferral_margin.max() + 0.1, 200)
    rows = []
    for t in thresholds:
        automated_mask = deferral_margin < t  # negative margin = prefer not defer
        cov = float(automated_mask.sum() / n)
        if automated_mask.any():
            correct = (model_preds[automated_mask] == true_labels[automated_mask])
            risk = float(1.0 - correct.mean())
        else:
            risk = float("nan")
        rows.append({"threshold": t, "coverage": cov, "selective_risk": risk})

    return pd.DataFrame(rows)


def plot_risk_coverage_curves(
    baseline_rc: pd.DataFrame,
    proposed_rc: pd.DataFrame,
    policy_name: str,
    save_path: Path,
    baseline_op: dict | None = None,
    proposed_op: dict | None = None,
) -> None:
    """Plot risk-coverage curves for baseline and proposed policy on shared axes.

    Annotates the operating point (actual threshold used in evaluation) with
    a marker on each curve.

    Reference: HAC_Lecture1.pdf — selective prediction trade-off.

    Args:
        baseline_rc  : DataFrame from compute_risk_coverage_curve() for baseline.
        proposed_rc  : DataFrame from compute_risk_coverage_curve() for proposed.
        policy_name  : display name for the proposed policy (e.g. 'RAPS').
        save_path    : Path (without extension); PDF + PNG saved at 300 dpi.
        baseline_op  : optional dict with keys 'coverage', 'selective_risk' for
                       the actual operating point of the baseline policy.
        proposed_op  : optional dict for the proposed policy operating point.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Drop NaN rows for plotting
    bl = baseline_rc.dropna(subset=["selective_risk"])
    pr = proposed_rc.dropna(subset=["selective_risk"])

    ax.plot(bl["coverage"], bl["selective_risk"], label="Baseline (threshold)",
            color="#4c72b0", linewidth=2)
    ax.plot(pr["coverage"], pr["selective_risk"], label=policy_name,
            color="#dd8452", linewidth=2, linestyle="--")

    # Operating point markers
    if baseline_op is not None:
        ax.scatter(
            baseline_op["coverage"], baseline_op["selective_risk"],
            color="#4c72b0", s=80, zorder=5, marker="o",
            label=f"Baseline operating point"
        )
    if proposed_op is not None:
        ax.scatter(
            proposed_op["coverage"], proposed_op["selective_risk"],
            color="#dd8452", s=80, zorder=5, marker="s",
            label=f"{policy_name} operating point"
        )

    ax.set_xlabel("Coverage (fraction automated)", fontsize=12)
    ax.set_ylabel("Selective risk (error rate on automated)", fontsize=12)
    ax.set_title(f"Risk–Coverage Curve: Baseline vs {policy_name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[risk_coverage] Curve saved to {save_path}.{{pdf,png}}")
