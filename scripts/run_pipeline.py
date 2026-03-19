"""
End-to-end pipeline. Reads PROPOSED_POLICY from src/config.py.

Usage:
    python scripts/run_pipeline.py

To switch policy, edit src/config.py:
    PROPOSED_POLICY = "mozannar_sontag"   # or "raps"

Steps:
    1. Load data
    2. Train + calibrate LogisticBaseline and DeepEnsemble
    3. Evaluate calibration: reliability diagrams + ECE for both models
    4. Compute complementarity heatmap
    5. Run baseline policy (always)
    6. Run proposed policy (branches on PROPOSED_POLICY)
    7. Risk-coverage curves for baseline vs proposed
    8. Print summary table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config
from src.config import (
    FIGURES_DIR,
    PROPOSED_POLICY,
    RAPS_ALPHA,
    RESULTS_DIR,
)
from src.data.lsac import get_X, load_lsac
from src.evaluation.calibration import (
    compute_ece,
    plot_reliability_diagram,
    report_raps_coverage,
)
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

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"Pipeline starting | PROPOSED_POLICY = {PROPOSED_POLICY!r}")
print("=" * 60)

print("\n[1/8] Loading LSAC data...")
train_df, val_df, test_df, scaler = load_lsac()
X_train = get_X(train_df, scaler)
X_val   = get_X(val_df,   scaler)
X_test  = get_X(test_df,  scaler)
y_train = train_df["pass_bar"].values
y_val   = val_df["pass_bar"].values
y_test  = test_df["pass_bar"].values
print(f"  Train {len(train_df):,}  Val {len(val_df):,}  Test {len(test_df):,}")

human = HumanReviewer()

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train + calibrate models
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Training LogisticBaseline...")
baseline_model = LogisticBaseline()
baseline_model.fit(X_train, y_train)
baseline_model.calibrate(X_val, y_val)

print("\n[2/8] Training DeepEnsemble...")
ensemble = DeepEnsemble(input_dim=X_train.shape[1])
ensemble.fit(X_train, y_train)
ensemble.calibrate(X_val, y_val)

# Precompute probabilities used throughout the pipeline
bl_raw_test     = baseline_model.predict_proba_raw(X_test)[:, 1]
bl_cal_test     = baseline_model.predict_proba(X_test)[:, 1]
ens_raw_test    = ensemble.predict_proba_raw(X_test)
ens_cal_test    = ensemble.predict_proba(X_test)

# Shape (n, 2): column 0 = P(fail), column 1 = P(pass)
test_probs_2d   = np.stack([1 - ens_cal_test, ens_cal_test], axis=1)
val_probs_2d    = np.stack(
    [1 - ensemble.predict_proba(X_val), ensemble.predict_proba(X_val)], axis=1
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Calibration evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Calibration evaluation...")

ece_bl_raw = compute_ece(bl_raw_test, y_test)
ece_bl_cal = compute_ece(bl_cal_test, y_test)
ece_ens_raw = compute_ece(ens_raw_test, y_test)
ece_ens_cal = compute_ece(ens_cal_test, y_test)
print(f"  Baseline  ECE raw={ece_bl_raw:.4f}  cal={ece_bl_cal:.4f}")
print(f"  Ensemble  ECE raw={ece_ens_raw:.4f}  cal={ece_ens_cal:.4f}")

plot_reliability_diagram(
    bl_raw_test, bl_cal_test, y_test,
    model_name="LogisticBaseline",
    save_path=FIGURES_DIR / "reliability_baseline",
)
plot_reliability_diagram(
    ens_raw_test, ens_cal_test, y_test,
    model_name="DeepEnsemble",
    save_path=FIGURES_DIR / "reliability_ensemble",
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Complementarity heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Complementarity analysis...")
delta_comp_df = compute_delta_comp(test_df, ens_cal_test, human)
print(delta_comp_df.to_string(index=False))
plot_complementarity_heatmap(
    delta_comp_df,
    save_path=FIGURES_DIR / "complementarity_heatmap",
)
delta_comp_df.to_csv(RESULTS_DIR / "complementarity.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Baseline policy
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Running BaselinePolicy...")
baseline_policy = BaselinePolicy()
baseline_results = baseline_policy.run(
    test_df, ensemble, human, probs=test_probs_2d
)
baseline_results.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

bl_acc        = baseline_results["correct"].mean()
bl_defer_rate = (baseline_results["decision"] == "defer").mean()
bl_coverage   = 1.0 - bl_defer_rate
bl_auto       = baseline_results[baseline_results["decision"] == "automate"]
bl_auto_risk  = 1.0 - bl_auto["correct"].mean() if len(bl_auto) > 0 else float("nan")
print(f"  acc={bl_acc:.3f}  coverage={bl_coverage:.3f}  "
      f"selective_risk={bl_auto_risk:.3f}")

baseline_rc = compute_risk_coverage_curve(baseline_results)

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Proposed policy
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6/8] Running proposed policy: {PROPOSED_POLICY!r}...")

if PROPOSED_POLICY == "raps":
    from src.policy.raps_policy import RAPSPolicy

    raps = RAPSPolicy()
    raps.fit(val_probs_2d, y_val)
    proposed_results = raps.run(test_df, ensemble, human, test_probs_2d)

    coverage_guarantee = raps.empirical_coverage(test_probs_2d, y_test)
    print(f"  RAPS empirical coverage: {coverage_guarantee:.3f} "
          f"(target: {1 - RAPS_ALPHA:.2f})")
    report_raps_coverage(coverage_guarantee)

    proposed_results.to_csv(RESULTS_DIR / "raps_results.csv", index=False)
    proposed_rc = compute_risk_coverage_curve(proposed_results)
    policy_label = "RAPS"

elif PROPOSED_POLICY == "mozannar_sontag":
    from src.policy.mozannar_sontag_policy import MozannarSontagPolicy

    ms = MozannarSontagPolicy(input_dim=X_train.shape[1])
    ms.fit(train_df, X_train, y_train, human)
    proposed_results = ms.run(test_df, X_test, human)
    proposed_results.to_csv(RESULTS_DIR / "ms_results.csv", index=False)

    logits = ms.get_logits(X_test)
    proposed_rc = compute_ms_risk_coverage_curve(logits, y_test, human, test_df)
    policy_label = "Mozannar-Sontag"

else:
    raise ValueError(f"Unknown PROPOSED_POLICY: {PROPOSED_POLICY!r}")

prop_acc        = proposed_results["correct"].mean()
prop_defer_rate = (proposed_results["decision"] == "defer").mean()
prop_coverage   = 1.0 - prop_defer_rate
prop_auto       = proposed_results[proposed_results["decision"] == "automate"]
prop_auto_risk  = (
    1.0 - prop_auto["correct"].mean() if len(prop_auto) > 0 else float("nan")
)
print(f"  acc={prop_acc:.3f}  coverage={prop_coverage:.3f}  "
      f"selective_risk={prop_auto_risk:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Risk-coverage curves
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Plotting risk-coverage curves...")
plot_risk_coverage_curves(
    baseline_rc,
    proposed_rc,
    policy_label,
    save_path=FIGURES_DIR / f"risk_coverage_{PROPOSED_POLICY}",
    baseline_op={"coverage": bl_coverage, "selective_risk": bl_auto_risk},
    proposed_op={"coverage": prop_coverage, "selective_risk": prop_auto_risk},
)

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] Summary")
print("=" * 60)
summary = pd.DataFrame(
    [
        {
            "Policy":         "Baseline (threshold)",
            "Accuracy":       f"{bl_acc:.3f}",
            "Coverage":       f"{bl_coverage:.3f}",
            "Selective Risk": f"{bl_auto_risk:.3f}",
            "ECE (ensemble)": f"{ece_ens_cal:.4f}",
        },
        {
            "Policy":         policy_label,
            "Accuracy":       f"{prop_acc:.3f}",
            "Coverage":       f"{prop_coverage:.3f}",
            "Selective Risk": f"{prop_auto_risk:.3f}",
            "ECE (ensemble)": f"{ece_ens_cal:.4f}",
        },
    ]
)
print(summary.to_string(index=False))

summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
print(f"\nAll outputs saved to {RESULTS_DIR.parent}/")
