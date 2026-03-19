"""
Deep ensemble of small MLPs with uncertainty decomposition.

Implements the deep ensemble approach from:
    Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles". NeurIPS 2017.

Uncertainty decomposition follows Lecture 1 Eq 1.5:
    total_uncertainty    = H[ E_theta[p(y|x,theta)] ]       (predictive entropy)
    aleatoric_uncertainty = E_theta[ H[p(y|x,theta)] ]       (mean member entropy)
    epistemic_uncertainty = total - aleatoric
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.evaluation.calibration import compute_ece


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy H(p) = -p log p - (1-p) log(1-p), clipped for stability."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


class SingleMLP(nn.Module):
    """Small MLP for binary classification.

    Architecture: input -> [Linear(d, H) -> ReLU] * N_LAYERS -> Linear(H, 1) -> Sigmoid
    Trained with BCELoss.
    """

    def __init__(self, input_dim: int,
                 hidden_dim: int = config.HIDDEN_DIM,
                 n_layers: int = config.N_LAYERS):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns P(y=1), shape (batch, 1)."""
        return self.net(x)


class DeepEnsemble:
    """Deep ensemble of N_ENSEMBLE SingleMLP models.

    Each member is trained from a different random seed to promote diversity.
    Post-hoc temperature scaling is applied to the ensemble mean probability.

    Uncertainty decomposition implements Lecture 1 Eq 1.5.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.members_: list[SingleMLP] = []
        self.temperature_ = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DeepEnsemble":
        """Train N_ENSEMBLE SingleMLPs, each from a different random seed.

        Uses Adam optimiser with LR, BCELoss, EPOCHS epochs, BATCH_SIZE.
        Seeds 0 .. N_ENSEMBLE-1 ensure reproducible diversity across members.

        Args:
            X_train : shape (n, d)
            y_train : shape (n,) binary labels

        Returns:
            self
        """
        X_t = torch.FloatTensor(X_train)
        y_t = torch.FloatTensor(y_train).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)

        self.members_ = []
        for seed in range(config.N_ENSEMBLE):
            torch.manual_seed(seed)
            model = SingleMLP(self.input_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
            criterion = nn.BCELoss()
            loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

            model.train()
            for _ in range(config.EPOCHS):
                for X_b, y_b in loader:
                    optimizer.zero_grad()
                    preds = model(X_b)
                    loss = criterion(preds, y_b)
                    loss.backward()
                    optimizer.step()

            model.eval()
            self.members_.append(model)

        return self

    def predict_member_probas(self, X: np.ndarray) -> np.ndarray:
        """Return per-member probabilities P(y=1), shape (N_ENSEMBLE, n).

        Used by RAPSPolicy for per-member prediction sets.
        """
        X_t = torch.FloatTensor(X)
        member_probs = []
        with torch.no_grad():
            for model in self.members_:
                p = model(X_t).squeeze(1).numpy()
                member_probs.append(p)
        return np.stack(member_probs, axis=0)

    def calibrate(self, X_val: np.ndarray, y_val: np.ndarray) -> "DeepEnsemble":
        """Fit a single scalar temperature T on the ensemble mean probability.

        Minimises NLL on validation set via scipy.optimize.minimize_scalar.
        Logs pre- and post-calibration ECE.

        Implements Lecture 1 temperature scaling applied to ensemble mean.

        Args:
            X_val : shape (n, d)
            y_val : shape (n,) binary labels

        Returns:
            self
        """
        raw_mean = self.predict_member_probas(X_val).mean(axis=0)

        def nll(T: float) -> float:
            p = np.clip(raw_mean ** (1 / T), 1e-9, 1 - 1e-9)
            # Temperature scaling on probabilities: renormalise after scaling
            # Use logit-based temperature for numerical stability
            logits = np.log(raw_mean / np.clip(1 - raw_mean, 1e-9, None))
            p_scaled = 1 / (1 + np.exp(-logits / T))
            p_scaled = np.clip(p_scaled, 1e-9, 1 - 1e-9)
            return -np.mean(y_val * np.log(p_scaled) + (1 - y_val) * np.log(1 - p_scaled))

        ece_before = compute_ece(raw_mean, y_val)
        print(f"[DeepEnsemble] ECE before calibration: {ece_before:.4f}")

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.temperature_ = float(result.x)

        cal_probs = self.predict_proba(X_val)
        ece_after = compute_ece(cal_probs, y_val)
        print(
            f"[DeepEnsemble] Temperature T={self.temperature_:.4f} | "
            f"ECE after calibration: {ece_after:.4f}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated P(y=1), shape (n,).

        Applies temperature scaling to the ensemble mean logit.
        Implements: p_cal = sigmoid(logit_mean / T).
        """
        raw_mean = self.predict_member_probas(X).mean(axis=0)
        logits = np.log(raw_mean / np.clip(1 - raw_mean, 1e-9, None))
        return 1 / (1 + np.exp(-logits / self.temperature_))

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        """Return uncalibrated ensemble mean P(y=1), shape (n,)."""
        return self.predict_member_probas(X).mean(axis=0)

    def decompose_uncertainty(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Decompose predictive uncertainty into aleatoric and epistemic components.

        Implements Lecture 1 Eq 1.5:
            total      = H[ (1/N) sum_i p_i(y|x) ]  — predictive entropy
            aleatoric  = (1/N) sum_i H[ p_i(y|x) ]  — mean member entropy
            epistemic  = total - aleatoric

        This decomposes the total uncertainty into:
          - Aleatoric: irreducible noise inherent to the task.
          - Epistemic: model uncertainty reducible with more data.

        Args:
            X : shape (n, d)

        Returns:
            dict with keys 'total', 'aleatoric', 'epistemic', each shape (n,)
        """
        member_probs = self.predict_member_probas(X)   # (N, n)

        # Total uncertainty: entropy of the mean predictive distribution
        mean_probs = member_probs.mean(axis=0)         # (n,)
        total = _binary_entropy(mean_probs)             # (n,)

        # Aleatoric: mean of per-member entropies
        member_entropies = _binary_entropy(member_probs)  # (N, n)
        aleatoric = member_entropies.mean(axis=0)          # (n,)

        # Epistemic: residual (Lecture 1 Eq 1.5)
        epistemic = total - aleatoric

        return {
            "total":      total,
            "aleatoric":  aleatoric,
            "epistemic":  epistemic,
        }
