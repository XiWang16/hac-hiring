"""
Complementarity evaluation for human-AI collaboration.

Computes delta_complementarity per subgroup: how much better the joint
human-AI oracle is compared to the better solo agent.

Reference: HAC_Lecture2.pdf — complementarity metric.
    delta_comp = min(model_error, human_error) - oracle_error

A positive delta_comp indicates the human and model are complementary —
their errors are not perfectly correlated, so the oracle (knowing which
agent to trust per instance) outperforms either agent alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_delta_comp(
    test_df: pd.DataFrame,
    model_probs: np.ndarray,
    human,
) -> pd.DataFrame:
    """Compute complementarity metric per subgroup.

    For each subgroup g (HAC_Lecture2.pdf complementarity definition):
        model_error  = mean(argmax(model_probs_g) != true_labels_g)
        human_error  = mean(human.predict_batch(df_g, 'a1', labels_g) != labels_g)
        oracle_error = fraction of instances where BOTH model AND human are wrong
        delta_comp   = min(model_error, human_error) - oracle_error

    Action 'a1' (no AI support) is used for the human to measure their
    unassisted performance — the natural baseline for complementarity.

    Args:
        test_df     : test split DataFrame with columns 'pass_bar', 'subgroup'.
        model_probs : shape (n,) calibrated P(y=1) from the ensemble.
        human       : HumanReviewer instance.

    Returns:
        pd.DataFrame with columns:
            subgroup, model_error, human_error, oracle_error, delta_comp
    """
    true_labels = test_df["pass_bar"].values
    model_preds = (model_probs >= 0.5).astype(int)

    # Compute human predictions once over the full test set
    human_preds = human.predict_batch(test_df, "a1", true_labels)

    subgroups = sorted(test_df["subgroup"].unique())
    rows = []
    for sg in subgroups:
        mask = test_df["subgroup"].values == sg
        if not mask.any():
            continue

        labels_g = true_labels[mask]
        model_preds_g = model_preds[mask]
        human_preds_g = human_preds[mask]

        model_err = float(np.mean(model_preds_g != labels_g))
        human_err = float(np.mean(human_preds_g != labels_g))

        # Oracle error: instances where BOTH model AND human are wrong
        oracle_err = float(
            np.mean((model_preds_g != labels_g) & (human_preds_g != labels_g))
        )

        delta_comp = min(model_err, human_err) - oracle_err

        rows.append(
            {
                "subgroup":     sg,
                "model_error":  model_err,
                "human_error":  human_err,
                "oracle_error": oracle_err,
                "delta_comp":   delta_comp,
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def plot_complementarity_heatmap(
    delta_comp_df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plot a seaborn heatmap of complementarity metrics per subgroup.

    Each column is a subgroup; rows show model_error, human_error,
    oracle_error, and delta_comp. Cells are annotated with numeric values.

    Args:
        delta_comp_df : DataFrame from compute_delta_comp().
        save_path     : Path (without extension); both PDF and PNG are saved
                        at 300 dpi to outputs/figures/.
    """
    sns.set_style("whitegrid")

    display_df = delta_comp_df.set_index("subgroup")[
        ["model_error", "human_error", "oracle_error", "delta_comp"]
    ].T

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        display_df.astype(float),
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        vmin=0.0,
        cbar_kws={"label": "Value"},
    )
    ax.set_title("Complementarity Analysis by Subgroup", fontsize=13)
    ax.set_xlabel("Subgroup", fontsize=11)
    ax.set_ylabel("Metric", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[complementarity] Heatmap saved to {save_path}.{{pdf,png}}")
