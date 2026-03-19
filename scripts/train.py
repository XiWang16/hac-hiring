"""
Train and calibrate baseline (LogisticRegression) and ensemble (DeepEnsemble).

Usage:
    python scripts/train.py

Saves nothing to disk — training is re-run in run_pipeline.py. This script
is a standalone entry point for quick experimentation or debugging.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURES_DIR, RESULTS_DIR
from src.data.lsac import get_X, load_lsac
from src.models.baseline import LogisticBaseline
from src.models.ensemble import DeepEnsemble

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 60)
    print("Step 1: Loading LSAC data")
    print("=" * 60)
    train_df, val_df, test_df, scaler = load_lsac()
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    X_train = get_X(train_df, scaler)
    X_val = get_X(val_df, scaler)
    X_test = get_X(test_df, scaler)
    y_train = train_df["pass_bar"].values
    y_val = val_df["pass_bar"].values

    print("\n" + "=" * 60)
    print("Step 2: Training LogisticBaseline")
    print("=" * 60)
    baseline = LogisticBaseline()
    baseline.fit(X_train, y_train)
    baseline.calibrate(X_val, y_val)
    print(f"  Baseline temperature: {baseline.temperature_:.4f}")

    print("\n" + "=" * 60)
    print("Step 3: Training DeepEnsemble")
    print("=" * 60)
    ensemble = DeepEnsemble(input_dim=X_train.shape[1])
    ensemble.fit(X_train, y_train)
    ensemble.calibrate(X_val, y_val)
    print(f"  Ensemble temperature: {ensemble.temperature_:.4f}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
