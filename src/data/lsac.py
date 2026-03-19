"""
LSAC Bar Passage dataset loader.

Dataset: Law School Admission Council (LSAC) bar passage study.
Download bar_pass_prediction.csv from Kaggle into data/.
Target: pass_bar (binary, 1 = passed the bar exam).

Notes on race1:
  - The column is already string-valued: ['white', 'hisp', 'asian', 'black',
    'other', NaN]. No numeric mapping is applied.
  - A separate numeric 'race' column exists in the CSV but is ignored entirely.
  - Rows where race1 is NaN are dropped (very few affected).
  - Subgroup logic uses the literal string 'hisp' for Hispanic applicants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from src import config

# Features used for the model (numeric only)
FEATURE_COLS = ["lsat", "ugpa", "zfygpa", "zgpa", "cluster", "age", "fam_inc", "fulltime"]


def _encode_race(series: pd.Series) -> pd.Series:
    """Normalise the race1 string column.

    race1 is already a string column in the CSV. We lowercase and strip
    whitespace only — no numeric mapping is applied. The dataset uses:
        'white', 'hisp', 'asian', 'black', 'other'
    NaN rows are handled upstream by dropna(subset=['race1']).
    """
    return series.astype(str).str.lower().str.strip()


def get_subgroup(row: pd.Series) -> str:
    """Assign a subgroup label to a single row.

    Priority order (first match wins):
      1. minority_low_cluster : race1 in ['black', 'hisp'] AND cluster <= 2
      2. ambiguous_middle     : 35 <= lsat_pct <= 65 AND 35 <= ugpa_pct <= 65
      3. strong_credentials   : lsat_pct >= 75 AND ugpa_pct >= 75
      4. other                : everything else

    Note: 'hisp' is the literal value in the LSAC CSV (not 'hispanic').
    """
    race = row["race1"]
    cluster = row["cluster"]
    lsat_pct = row["lsat_pct"]
    ugpa_pct = row["ugpa_pct"]

    if race in ("black", "hisp") and cluster <= 2:
        return "minority_low_cluster"
    if 35 <= lsat_pct <= 65 and 35 <= ugpa_pct <= 65:
        return "ambiguous_middle"
    if lsat_pct >= 75 and ugpa_pct >= 75:
        return "strong_credentials"
    return "other"


def _assign_subgroups(df: pd.DataFrame) -> pd.Series:
    """Apply get_subgroup row-wise. Requires lsat_pct, ugpa_pct, race1, cluster."""
    return df.apply(get_subgroup, axis=1)


def _compute_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rank-based percentiles for lsat and ugpa across the full dataset.

    Percentiles are fitted on *all* data before splitting so they reflect the
    full population distribution (consistent with how admissions benchmarks work).
    """
    df = df.copy()
    n = len(df)
    df["lsat_pct"] = rankdata(df["lsat"], method="average") / n * 100
    df["ugpa_pct"] = rankdata(df["ugpa"], method="average") / n * 100
    return df


def load_lsac() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Load and split the LSAC bar passage dataset.

    Steps:
      1. Read CSV, encode race1, impute fam_inc with median.
      2. Drop rows where pass_bar is NaN.
      3. Compute lsat_pct / ugpa_pct on full dataset (before splitting).
      4. Assign subgroup labels.
      5. Stratified split on pass_bar.
      6. Fit StandardScaler on train features only.

    Returns:
        train_df, val_df, test_df : pd.DataFrame (each with all original columns
                                     plus 'subgroup', 'lsat_pct', 'ugpa_pct')
        scaler                    : StandardScaler fitted on FEATURE_COLS of train_df
    """
    df = pd.read_csv(config.LSAC_CSV)

    # Encode race (lowercase + strip; race1 is already string in CSV)
    df["race1"] = _encode_race(df["race1"])

    # Drop rows where race1 is NaN (very few; must precede subgroup assignment)
    df = df.dropna(subset=["race1"]).reset_index(drop=True)

    # Impute fam_inc with median
    fam_inc_median = df["fam_inc"].median()
    df["fam_inc"] = df["fam_inc"].fillna(fam_inc_median)

    # Drop rows with missing target
    df = df.dropna(subset=["pass_bar"]).reset_index(drop=True)
    df["pass_bar"] = df["pass_bar"].astype(int)

    # Rank-based percentiles on full dataset (before split)
    df = _compute_percentiles(df)

    # Subgroup assignment
    df["subgroup"] = _assign_subgroups(df)

    # Stratified split: train / val / test
    test_frac = 1.0 - config.TRAIN_FRAC - config.VAL_FRAC
    train_df, temp_df = train_test_split(
        df,
        train_size=config.TRAIN_FRAC,
        stratify=df["pass_bar"],
        random_state=config.RANDOM_SEED,
    )
    # Val fraction relative to temp_df
    val_frac_of_temp = config.VAL_FRAC / (config.VAL_FRAC + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_frac_of_temp,
        stratify=temp_df["pass_bar"],
        random_state=config.RANDOM_SEED,
    )

    # Fit scaler on train only
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        scaler,
    )


def get_X(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Return the scaled feature matrix for a given split.

    Uses FEATURE_COLS (numeric columns only) standardised with a pre-fitted
    StandardScaler (fitted on train_df only).

    Args:
        df     : one of train_df / val_df / test_df from load_lsac()
        scaler : StandardScaler returned by load_lsac()

    Returns:
        np.ndarray of shape (n_samples, len(FEATURE_COLS))
    """
    return scaler.transform(df[FEATURE_COLS].values)
