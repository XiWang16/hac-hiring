from pathlib import Path

# ── Policy selection ──────────────────────────────────────────────────────────
# Set to "raps" or "mozannar_sontag". Controls which proposed policy is trained,
# evaluated, and plotted against the baseline in all scripts.
PROPOSED_POLICY = "raps"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data"
OUTPUTS_DIR   = ROOT / "outputs"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
RESULTS_DIR   = OUTPUTS_DIR / "results"

# ── Data ──────────────────────────────────────────────────────────────────────
LSAC_CSV      = DATA_DIR / "bar_pass_prediction.csv"
RANDOM_SEED   = 42
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# test = remainder

# ── Cost structure ────────────────────────────────────────────────────────────
C_ERR         = 1.0     # cost of an automated wrong decision
C_ESC         = 0.15    # cost of escalating to human reviewer

# ── Ensemble ──────────────────────────────────────────────────────────────────
N_ENSEMBLE    = 5
HIDDEN_DIM    = 64
N_LAYERS      = 2
LR            = 1e-3
EPOCHS        = 50
BATCH_SIZE    = 256

# ── RAPS conformal prediction ─────────────────────────────────────────────────
RAPS_ALPHA    = 0.10    # target miscoverage level (1 - alpha = 90% coverage)
RAPS_LAMBDA   = 0.01    # regularisation penalty weight
RAPS_K_REG   = 1        # rank at which penalty begins
# Deferral rule: defer if |prediction set| > RAPS_DEFER_THRESHOLD
RAPS_DEFER_THRESHOLD = 1

# ── Mozannar-Sontag deferral ──────────────────────────────────────────────────
MS_LR         = 1e-3
MS_EPOCHS     = 80
MS_HIDDEN_DIM = 64
MS_N_LAYERS   = 2

# ── Human model ───────────────────────────────────────────────────────────────
HUMAN_NOISE_SCALE = 0.04

HUMAN_ACCURACY = {
    "minority_low_cluster": 0.78,
    "ambiguous_middle":     0.61,
    "strong_credentials":   0.83,
    "other":                0.71,
}

SUPPORT_EFFECT = {
    ("minority_low_cluster", "a1"):  0.00,
    ("minority_low_cluster", "a2"): -0.05,
    ("minority_low_cluster", "a3"):  0.08,
    ("ambiguous_middle",     "a1"):  0.00,
    ("ambiguous_middle",     "a2"):  0.03,
    ("ambiguous_middle",     "a3"):  0.02,
    ("strong_credentials",   "a1"):  0.00,
    ("strong_credentials",   "a2"): -0.03,
    ("strong_credentials",   "a3"):  0.00,
    ("other",                "a1"):  0.00,
    ("other",                "a2"):  0.02,
    ("other",                "a3"):  0.04,
}
