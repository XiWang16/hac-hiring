# CLAUDE.md — Implementation notes for hac-hiring

This file documents non-obvious implementation decisions, parameter choices,
and project structure for future Claude Code sessions.

---

## Project structure

```
hac-hiring/
├── src/
│   ├── config.py                    # All hyperparameters + PROPOSED_POLICY toggle
│   ├── data/lsac.py                 # Dataset loading, subgroup assignment
│   ├── models/baseline.py           # LogisticRegression + temperature scaling
│   ├── models/ensemble.py           # DeepEnsemble + uncertainty decomposition
│   ├── human/model.py               # Simulated HumanReviewer
│   ├── policy/
│   │   ├── baseline_policy.py       # Confidence threshold (always runs)
│   │   ├── raps_policy.py           # RAPS conformal prediction
│   │   └── mozannar_sontag_policy.py# Consistent surrogate deferral
│   └── evaluation/
│       ├── calibration.py           # ECE, reliability diagrams
│       ├── complementarity.py       # Delta-comp per subgroup
│       └── risk_coverage.py         # Risk-coverage curves
├── scripts/
│   ├── train.py                     # Standalone training
│   ├── evaluate.py                  # Standalone evaluation
│   └── run_pipeline.py              # Full end-to-end pipeline
├── data/                            # Place bar_pass_prediction.csv here
└── outputs/
    ├── figures/                     # PDF + PNG at 300 dpi
    └── results/                     # CSV result tables
```

---

## Policy toggle

In `src/config.py`:

```python
PROPOSED_POLICY = "raps"           # or "mozannar_sontag"
```

- **Baseline policy** (confidence threshold) always runs regardless of the flag.
- Only the selected proposed policy is trained and evaluated.
- Both implementations are always present in the codebase.
- Every evaluation module imports `PROPOSED_POLICY` and branches accordingly.

---

## Dataset: LSAC race1 column

**Critical**: `race1` is already a string column in the CSV with values:
`['white', 'hisp', 'asian', 'black', 'other', NaN]`

- **Do NOT apply any numeric mapping** — there is no integer encoding needed.
- Normalisation: lowercase + strip whitespace only.
- Rows where `race1 is NaN` are dropped (very few; handled before subgroup assignment).
- A separate numeric `race` column exists in the CSV — **ignore it entirely**.
  All subgroup logic uses `race1` exclusively.
- Hispanic applicants appear as `'hisp'` (not `'hispanic'`).

---

## Subgroup definitions

Priority order (first match wins):

| Subgroup | Condition |
|----------|-----------|
| `minority_low_cluster` | `race1 in ['black', 'hisp']` AND `cluster <= 2` |
| `ambiguous_middle` | `35 <= lsat_pct <= 65` AND `35 <= ugpa_pct <= 65` |
| `strong_credentials` | `lsat_pct >= 75` AND `ugpa_pct >= 75` |
| `other` | everything else |

`lsat_pct` and `ugpa_pct` are rank-based percentiles fitted on the **full
dataset** before any train/val/test split (population-level benchmarks).

---

## Human model parameters

### Base accuracy (`config.HUMAN_ACCURACY`)

| Subgroup | Accuracy |
|----------|----------|
| `minority_low_cluster` | 0.78 |
| `ambiguous_middle` | 0.61 |
| `strong_credentials` | 0.83 |
| `other` | 0.71 |

These values are loosely calibrated to empirical findings from:
- Lai et al. (2020) "On Human Predictions with Explanations and Predictions of
  Machine Learning Models". FAccT 2020.
- Green & Chen (2019) "Disparate Interactions: An Algorithm-in-the-Loop
  Analysis of Fairness in Risk Assessments". FAccT 2019.

### Support effects (`config.SUPPORT_EFFECT`)

| (subgroup, action) | Effect |
|--------------------|--------|
| `(minority_low_cluster, a3)` | +0.08 |
| `(ambiguous_middle, a2)` | +0.03 |
| `(other, a3)` | +0.04 |
| `(strong_credentials, a2)` | −0.03 |
| `(minority_low_cluster, a2)` | −0.05 |
| (all others) | 0.00 |

**Actions**:
- `a1`: no AI support (human decides alone)
- `a2`: model prediction shown
- `a3`: full explanation shown (e.g. feature importances + prediction)

Negative effects for `a2` on certain subgroups model automation bias — humans
over-relying on model output when the model is less reliable for that group.

### Noise model
`noise ~ Normal(0, HUMAN_NOISE_SCALE=0.04)`, clipped so `p_correct ∈ [0.05, 0.98]`.
Gaussian noise introduces trial-by-trial variability matching empirical spread.

---

## Cost structure

```
C_err  = 1.0   # cost of wrong automated decision
C_esc  = 0.15  # cost of escalating to human
```

Implies `τ* = 1 − 0.15/1.0 = 0.85` for the baseline threshold policy
(HAC_Lecture1.pdf Eq 1.8). Low escalation cost reflects that human review
is relatively cheap compared to a bad hiring decision.

---

## Equation references

Equations are cited in docstrings using the format `HAC_LectureN.pdf Eq X.Y`:

| Module | Reference |
|--------|-----------|
| `baseline_policy.py` | HAC_Lecture1.pdf Eq 1.8 (threshold derivation) |
| `models/baseline.py` | HAC_Lecture1.pdf (temperature scaling, ECE) |
| `evaluation/calibration.py` | HAC_Lecture1.pdf Eq 1.7 (ECE formula) |
| `policy/raps_policy.py` | HAC_Lecture1.pdf Eqs 1.9, 1.11, 1.12 (RAPS) |
| `policy/mozannar_sontag_policy.py` | HAC_Lecture2.pdf Eqs 2.5, 2.7, 2.8 (MS surrogate) |
| `evaluation/complementarity.py` | HAC_Lecture2.pdf (complementarity metric) |
| `evaluation/risk_coverage.py` | HAC_Lecture1.pdf (selective prediction) |
| `models/ensemble.py` | HAC_Lecture1.pdf Eq 1.5 (uncertainty decomposition) |

HAC_Lecture3.pdf covers bandit routing and resource constraints — not
implemented in this scaffold but referenced for future work.

---

## Non-obvious implementation decisions

### RAPS probs shape
`RAPSPolicy` expects `probs` of shape `(n, 2)` (both class probabilities),
not just P(y=1). This is needed to compute nonconformity scores for binary
classification. The ensemble's `predict_proba()` returns `(n,)`, so callers
must stack: `np.stack([1-p, p], axis=1)`.

### DeepEnsemble temperature scaling
Temperature scaling is applied to the **logit** of the ensemble mean (not
the probability directly) for numerical stability:
`p_cal = sigmoid(logit_mean / T)`.

### BaselinePolicy `run()` signature
Takes `probs` as an explicit argument (shape `(n, 2)`) to avoid re-running
the ensemble inside the policy. Pass `test_probs_2d` from the pipeline.

### MozannarSontagPolicy deferral weights
`w_defer` is estimated by running `human.predict_batch` 5 times and averaging
to reduce Monte Carlo variance. This increases training time slightly but
improves stability of the deferral threshold.

### Results DataFrame schema
All three policies return a DataFrame with identical columns:
`true_label, model_prob, set_size, decision, final_pred, correct, subgroup`

`set_size` is `None` for BaselinePolicy and MozannarSontagPolicy (kept for
schema parity so evaluation functions are policy-agnostic).

### Risk-coverage for MS policy
The MS policy does not use a scalar confidence threshold, so `model_prob`
alone is not a valid sweep variable. `compute_ms_risk_coverage_curve()` sweeps
the **deferral margin** `logit[defer] − max(logit[0], logit[1])` instead,
which monotonically controls the deferral rate.

---

## Reproducibility

All random operations use explicit seeds:
- `config.RANDOM_SEED = 42` for data splits and HumanReviewer
- `torch.manual_seed(seed)` for each ensemble member (seeds 0..N_ENSEMBLE-1)
- `np.random.default_rng(seed)` for HumanReviewer RNG
