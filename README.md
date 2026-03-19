# hac-hiring

Code for studying human-AI collaboration (HAC) policies.

---

## Dataset setup

1. Download `bar_pass_prediction.csv` from Kaggle:
   [LSAC National Longitudinal Bar Passage Study](https://www.kaggle.com/datasets/danofer/law-school-admissions-bar-passage)

2. Place the file at:
   ```
   data/bar_pass_prediction.csv
   ```

The `data/` directory is gitignored for CSV files; only `.gitkeep` is tracked.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the pipeline

```bash
python scripts/run_pipeline.py
```

This runs the full end-to-end pipeline:
1. Load and split the LSAC data (70/15/15 stratified split)
2. Train and calibrate `LogisticBaseline` and `DeepEnsemble`
3. Plot reliability diagrams (before/after temperature scaling)
4. Compute complementarity heatmap per subgroup
5. Run the **baseline policy** (confidence threshold)
6. Run the **proposed policy** (controlled by `PROPOSED_POLICY` flag)
7. Plot risk-coverage curves
8. Print and save a summary table

### Switching between policies

Edit `src/config.py`:

```python
PROPOSED_POLICY = "raps"           # Conformal prediction (HAC_Lecture1.pdf)
# PROPOSED_POLICY = "mozannar_sontag"  # Consistent surrogate (HAC_Lecture2.pdf)
```

Both implementations are always present; only the active one is evaluated.

---

## Output figures

All figures are saved to `outputs/figures/` as both PDF and PNG at 300 dpi.

| File | Description |
|------|-------------|
| `reliability_baseline.{pdf,png}` | Reliability diagram for LogisticBaseline (before/after calibration) |
| `reliability_ensemble.{pdf,png}` | Reliability diagram for DeepEnsemble (before/after calibration) |
| `complementarity_heatmap.{pdf,png}` | Delta-complementarity per subgroup (model vs human vs oracle error) |
| `risk_coverage_{policy}.{pdf,png}` | Risk-coverage curves: baseline vs proposed policy |

Results tables are saved to `outputs/results/` as CSV.

---

## Standalone scripts

```bash
python scripts/train.py      # Train + calibrate models only
python scripts/evaluate.py   # Full evaluation (same as run_pipeline.py)
```

---

## Subgroups

The pipeline assigns each candidate to exactly one subgroup (priority order):

| Subgroup | Definition |
|----------|-----------|
| `minority_low_cluster` | `race1 ∈ {black, hisp}` AND `cluster ≤ 2` |
| `ambiguous_middle` | `35 ≤ lsat_pct ≤ 65` AND `35 ≤ ugpa_pct ≤ 65` |
| `strong_credentials` | `lsat_pct ≥ 75` AND `ugpa_pct ≥ 75` |
| `other` | everything else |

`lsat_pct` and `ugpa_pct` are rank-based percentiles computed on the full
dataset before splitting.

---

## Policy descriptions

### Baseline — confidence threshold (`baseline_policy.py`)
Defers to the human when the ensemble's calibrated P(pass_bar=1) falls below:

    τ* = 1 − C_esc / C_err = 0.85   (HAC_Lecture1.pdf Eq 1.8)

Human uses **action a1** (no AI support shown).

### RAPS — conformal prediction (`raps_policy.py`)
Builds prediction sets using the RAPS nonconformity score (HAC_Lecture1.pdf
Eqs 1.11–1.12). Defers when `|C_α(x)| > RAPS_DEFER_THRESHOLD`.
Human uses **action a3** (full explanation shown — appropriate for uncertain cases).
Provides a marginal coverage guarantee: P(y ∈ C_α(x)) ≥ 1 − α.

### Mozannar-Sontag — consistent surrogate (`mozannar_sontag_policy.py`)
Trains a three-output MLP jointly on classification and deferral using the
surrogate loss from HAC_Lecture2.pdf Eq 2.8. Defers when the deferral logit
exceeds both class logits. Human uses **action a2** (model prediction shown).
