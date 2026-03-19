"""
Calibration evaluation utilities.

Implements Expected Calibration Error (ECE) and reliability diagrams following:
    Guo et al. (2017) "On Calibration of Modern Neural Networks". ICML 2017.
    Lecture 1 Eq 1.7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error using equal-width bins.

    Implements Lecture 1 Eq 1.7:
        ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    where B_b is the set of predictions in bin b, acc is mean accuracy,
    and conf is mean predicted probability.

    Args:
        probs  : shape (n,) predicted probabilities for the positive class.
        labels : shape (n,) binary ground truth labels.
        n_bins : number of equal-width bins in [0, 1]. Default: 15.

    Returns:
        ECE as a float.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def plot_reliability_diagram(
    probs_raw: np.ndarray,
    probs_cal: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: Path,
    n_bins: int = 15,
) -> None:
    """Plot reliability diagrams before and after temperature scaling.

    Creates two side-by-side panels showing the gap between the diagonal
    (perfect calibration) and the actual accuracy per bin.

    Args:
        probs_raw  : shape (n,) uncalibrated probabilities for positive class.
        probs_cal  : shape (n,) calibrated probabilities for positive class.
        labels     : shape (n,) binary ground truth.
        model_name : string label for the title (e.g. 'DeepEnsemble').
        save_path  : Path to save the figure (without extension; both PDF and
                     PNG are saved at 300 dpi).
        n_bins     : number of bins. Default: 15.
    """
    sns.set_style("whitegrid")

    def _bin_stats(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        accs, confs, counts = [], [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.any():
                accs.append(labels[mask].mean())
                confs.append(probs[mask].mean())
            else:
                accs.append(np.nan)
                confs.append(np.nan)
            counts.append(mask.sum())
        return np.array(accs), np.array(confs), np.array(counts)

    ece_raw = compute_ece(probs_raw, labels, n_bins)
    ece_cal = compute_ece(probs_cal, labels, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

    for ax, probs, ece, title_suffix in zip(
        axes,
        [probs_raw, probs_cal],
        [ece_raw, ece_cal],
        ["Before calibration", "After temperature scaling"],
    ):
        accs, confs, counts = _bin_stats(probs)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Bar chart of accuracy per bin
        valid = ~np.isnan(accs)
        ax.bar(
            centres[valid],
            accs[valid],
            width=1.0 / n_bins * 0.9,
            color="#4c72b0",
            alpha=0.7,
            label="Accuracy",
        )
        # Gap bars (calibration error)
        ax.bar(
            centres[valid],
            np.abs(confs[valid] - accs[valid]),
            bottom=np.minimum(accs[valid], confs[valid]),
            width=1.0 / n_bins * 0.9,
            color="#dd8452",
            alpha=0.5,
            label="Calibration gap",
        )
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of positives", fontsize=11)
        ax.set_title(f"{model_name}\n{title_suffix}\nECE = {ece:.4f}", fontsize=11)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Reliability Diagram — {model_name}", fontsize=13, y=1.02)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[calibration] Reliability diagram saved to {save_path}.{{pdf,png}}")


def report_raps_coverage(coverage: float, alpha: float = config.RAPS_ALPHA) -> None:
    """Print RAPS empirical coverage diagnostic.

    Should be called when PROPOSED_POLICY == 'raps'. Checks whether the
    coverage guarantee P(y in C_alpha(x)) >= 1 - alpha holds on the test set.

    Args:
        coverage : empirical coverage from RAPSPolicy.empirical_coverage().
        alpha    : miscoverage level from config.RAPS_ALPHA.
    """
    target = 1.0 - alpha
    status = "OK" if coverage >= target else "WARNING: below target"
    print(
        f"[calibration] Empirical coverage at alpha={alpha:.2f}: "
        f"{coverage:.3f} (target: {target:.2f}) [{status}]"
    )
