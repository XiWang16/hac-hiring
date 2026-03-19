"""
Standalone evaluation script: loads data, trains models, runs all evaluations,
and saves figures and results.

Usage:
    python scripts/evaluate.py

To switch policy, edit src/config.py: PROPOSED_POLICY = "mozannar_sontag"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import config
from src.config import FIGURES_DIR, PROPOSED_POLICY, RESULTS_DIR
from src.data.lsac import get_X, load_lsac
from src.evaluation.calibration import compute_ece, plot_reliability_diagram
from src.evaluation.complementarity import (
    compute_delta_comp,
    plot_complementarity_heatmap,
)
from src.evaluation.risk_coverage import (
    compute_ms_risk_coverage_curve,
    compute_risk_coverage_curve,
    plot_risk_coverage_curves,
)
from src.human.model import HumanReviewer
from src.models.baseline import LogisticBaseline
from src.models.ensemble import DeepEnsemble
from src.policy.baseline_policy import BaselinePolicy

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_df, val_df, test_df, scaler = load_lsac()
    X_train = get_X(train_df, scaler)
    X_val = get_X(val_df, scaler)
    X_test = get_X(test_df, scaler)
    y_train = train_df["pass_bar"].values
    y_val = val_df["pass_bar"].values
    y_test = test_df["pass_bar"].values

    human = HumanReviewer()

    # ── Models ────────────────────────────────────────────────────────────────
    print("Training and calibrating models...")
    baseline_model = LogisticBaseline()
    baseline_model.fit(X_train, y_train)
    baseline_model.calibrate(X_val, y_val)

    ensemble = DeepEnsemble(input_dim=X_train.shape[1])
    ensemble.fit(X_train, y_train)
    ensemble.calibrate(X_val, y_val)

    # ── Calibration plots ─────────────────────────────────────────────────────
    print("Plotting reliability diagrams...")
    bl_raw = baseline_model.predict_proba_raw(X_test)[:, 1]
    bl_cal = baseline_model.predict_proba(X_test)[:, 1]
    plot_reliability_diagram(
        bl_raw, bl_cal, y_test,
        model_name="LogisticBaseline",
        save_path=FIGURES_DIR / "reliability_baseline",
    )

    ens_raw = ensemble.predict_proba_raw(X_test)
    ens_cal = ensemble.predict_proba(X_test)
    plot_reliability_diagram(
        ens_raw, ens_cal, y_test,
        model_name="DeepEnsemble",
        save_path=FIGURES_DIR / "reliability_ensemble",
    )

    # ── Complementarity ───────────────────────────────────────────────────────
    print("Computing complementarity...")
    delta_comp_df = compute_delta_comp(test_df, ens_cal, human)
    print(delta_comp_df.to_string(index=False))
    plot_complementarity_heatmap(
        delta_comp_df,
        save_path=FIGURES_DIR / "complementarity_heatmap",
    )
    delta_comp_df.to_csv(RESULTS_DIR / "complementarity.csv", index=False)

    # ── Ensemble probs as (n, 2) for policies ─────────────────────────────────
    test_probs_2d = np.stack([1 - ens_cal, ens_cal], axis=1)
    val_probs_2d = np.stack(
        [1 - ensemble.predict_proba(X_val), ensemble.predict_proba(X_val)], axis=1
    )

    # ── Baseline policy ───────────────────────────────────────────────────────
    print("Running baseline policy...")
    baseline_policy = BaselinePolicy()
    baseline_results = baseline_policy.run(test_df, ensemble, human, probs=test_probs_2d)
    baseline_results.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

    baseline_acc = baseline_results["correct"].mean()
    baseline_defer_rate = (baseline_results["decision"] == "defer").mean()
    print(f"  Baseline acc={baseline_acc:.3f}  defer_rate={baseline_defer_rate:.3f}")

    baseline_rc = compute_risk_coverage_curve(baseline_results)

    # ── Proposed policy ───────────────────────────────────────────────────────
    if PROPOSED_POLICY == "raps":
        from src.evaluation.calibration import report_raps_coverage
        from src.policy.raps_policy import RAPSPolicy

        print("Running RAPS policy...")
        raps = RAPSPolicy()
        raps.fit(val_probs_2d, y_val)
        proposed_results = raps.run(test_df, ensemble, human, test_probs_2d)
        coverage = raps.empirical_coverage(test_probs_2d, y_test)
        report_raps_coverage(coverage)

        proposed_results.to_csv(RESULTS_DIR / "raps_results.csv", index=False)
        proposed_rc = compute_risk_coverage_curve(proposed_results)
        policy_label = "RAPS"

    elif PROPOSED_POLICY == "mozannar_sontag":
        from src.policy.mozannar_sontag_policy import MozannarSontagPolicy

        print("Running Mozannar-Sontag policy...")
        ms = MozannarSontagPolicy(input_dim=X_train.shape[1])
        ms.fit(train_df, X_train, y_train, human)
        proposed_results = ms.run(test_df, X_test, human)
        proposed_results.to_csv(RESULTS_DIR / "ms_results.csv", index=False)

        logits = ms.get_logits(X_test)
        proposed_rc = compute_ms_risk_coverage_curve(logits, y_test, human, test_df)
        policy_label = "Mozannar-Sontag"

    else:
        raise ValueError(f"Unknown PROPOSED_POLICY: {PROPOSED_POLICY!r}")

    proposed_acc = proposed_results["correct"].mean()
    proposed_defer = (proposed_results["decision"] == "defer").mean()
    print(f"  {policy_label} acc={proposed_acc:.3f}  defer_rate={proposed_defer:.3f}")

    # Operating points
    baseline_op = {
        "coverage": 1.0 - baseline_defer_rate,
        "selective_risk": 1.0 - baseline_results[
            baseline_results["decision"] == "automate"
        ]["correct"].mean(),
    }
    proposed_op = {
        "coverage": 1.0 - proposed_defer,
        "selective_risk": 1.0 - proposed_results[
            proposed_results["decision"] == "automate"
        ]["correct"].mean(),
    }

    plot_risk_coverage_curves(
        baseline_rc, proposed_rc, policy_label,
        save_path=FIGURES_DIR / f"risk_coverage_{PROPOSED_POLICY}",
        baseline_op=baseline_op,
        proposed_op=proposed_op,
    )

    print("\nEvaluation complete. Outputs saved to outputs/")


if __name__ == "__main__":
    main()
