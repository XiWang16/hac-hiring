"""
Mozannar-Sontag consistent surrogate for learning to defer.

Implements HAC_Lecture2.pdf, Sections 2.2.1 and 2.2.2.

How it works:
1. Augment the label space with a deferral class: Y_aug = {0, 1, defer}.
2. Define target weights (HAC_Lecture2.pdf Eq 2.7):
       w_k(x, y)     = 1{y == k}           for k in {0, 1}
       w_defer(x, y) = ell(h(x), y)        human error probability on (x, y)
   The deferral weight equals the human's expected error, estimated by running
   HumanReviewer under action a2 (model prediction shown) N_SAMPLES times.
3. Train a joint scoring function f_tilde: X -> R^3 with the cross-entropy
   surrogate (HAC_Lecture2.pdf Eq 2.8). At the minimiser, the system defers
   on x iff f_tilde[defer](x) > max(f_tilde[0](x), f_tilde[1](x)).
4. Consistent: converges to the Bayes-optimal deferral rule (Eq 2.5) as
   n -> infinity under mild conditions.

Reference:
    Mozannar & Sontag (2020) "Consistent Estimators for Learning to Defer to
    an Expert". ICML 2020.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src import config


class DeferralHead(nn.Module):
    """Three-output MLP: logits for class 0, class 1, and deferral.

    Architecture: input -> [Linear(d, H) -> ReLU] * N_LAYERS -> Linear(H, 3)

    The three outputs are raw logits; softmax is applied during training and
    argmax at inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = config.MS_HIDDEN_DIM,
        n_layers: int = config.MS_N_LAYERS,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 3))   # 3 outputs: 0, 1, defer
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, 3).

        Implements HAC_Lecture2.pdf Eq 2.8 scoring function f_tilde.
        """
        return self.net(x)


class MozannarSontagPolicy:
    """Mozannar-Sontag consistent deferral policy.

    Learns a joint classifier-deferral model by minimising a surrogate loss
    that is consistent with the Bayes-optimal deferral rule.

    Reference: HAC_Lecture2.pdf Sections 2.2.1–2.2.2.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = DeferralHead(input_dim)
        self.optimizer: torch.optim.Optimizer | None = None
        self._last_logits: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_deferral_weights(
        self,
        df: pd.DataFrame,
        true_labels: np.ndarray,
        human,
    ) -> np.ndarray:
        """Estimate w_defer(x, y) = human error probability per training instance.

        Uses action a2 (human sees model prediction) — the deferral scenario
        where the human is shown AI output. Runs HumanReviewer N_SAMPLES=5
        times and takes the mean error rate to reduce Monte Carlo noise.

        Implements HAC_Lecture2.pdf Eq 2.7 deferral weight.

        Args:
            df          : DataFrame of training instances.
            true_labels : shape (n,) binary ground truth.
            human       : HumanReviewer instance.

        Returns:
            shape (n,) float array in [0, 1] — estimated human error rate.
        """
        N_SAMPLES = 5
        errors = np.zeros(len(df), dtype=float)
        for _ in range(N_SAMPLES):
            preds = human.predict_batch(df, "a2", true_labels)
            errors += (preds != true_labels).astype(float)
        return errors / N_SAMPLES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        X_train: np.ndarray,
        y_train: np.ndarray,
        human,
    ) -> "MozannarSontagPolicy":
        """Train the joint deferral model using the MS cross-entropy surrogate.

        Loss (HAC_Lecture2.pdf Eq 2.8):
            L = -E_{x,y} [ sum_{k in {0,1,defer}} w_k(x,y) * log softmax(f_k(x)) ]

        where:
            w_0(x,y)     = 1{y==0}
            w_1(x,y)     = 1{y==1}
            w_defer(x,y) = human error rate on (x, y) under action a2

        This loss is NOT standard cross-entropy: the deferral logit is
        calibrated to the human's expected error, not to label correctness.
        The consistent property (Eq 2.5) relies on this specific form.

        Args:
            train_df : training split DataFrame (for human model).
            X_train  : shape (n, d) scaled feature matrix.
            y_train  : shape (n,) binary labels.
            human    : HumanReviewer instance.

        Returns:
            self
        """
        w_defer = self._compute_deferral_weights(train_df, y_train, human)

        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        w_tensor = torch.FloatTensor(w_defer)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.MS_LR
        )
        dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        self.model.train()
        for epoch in range(config.MS_EPOCHS):
            epoch_loss = 0.0
            for X_b, y_b, w_b in loader:
                self.optimizer.zero_grad()
                logits = self.model(X_b)                 # (batch, 3)
                log_softmax = F.log_softmax(logits, dim=1)

                batch_size = X_b.size(0)
                # Target weights per instance (HAC_Lecture2.pdf Eq 2.7)
                w_target = torch.zeros(batch_size, 3)
                w_target[torch.arange(batch_size), y_b] = 1.0  # class weights
                w_target[:, 2] = w_b                            # deferral weights

                # Weighted cross-entropy (HAC_Lecture2.pdf Eq 2.8)
                loss = -(w_target * log_softmax).sum(dim=1).mean()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"[MS] epoch {epoch+1}/{config.MS_EPOCHS} "
                      f"loss={epoch_loss/len(loader):.4f}")

        return self

    def run(
        self,
        test_df: pd.DataFrame,
        X_test: np.ndarray,
        human,
    ) -> pd.DataFrame:
        """Execute the Mozannar-Sontag policy on the test set.

        Routing rule (HAC_Lecture2.pdf Eq 2.5):
            argmax(logits) in {0, 1} -> automate with that prediction
            argmax(logits) == 2      -> defer; human uses action a2
                                        (consistent with deferral weight fit)

        Args:
            test_df : test split DataFrame.
            X_test  : shape (n, d) scaled feature matrix.
            human   : HumanReviewer instance.

        Returns:
            pd.DataFrame with columns:
                true_label, model_prob, set_size, decision, final_pred,
                correct, subgroup
            (set_size is None — column kept for schema parity with other policies.)
        """
        self.model.eval()
        with torch.no_grad():
            logits_t = self.model(torch.FloatTensor(X_test))   # (n, 3)
            decisions_idx = logits_t.argmax(dim=1).numpy()     # 0, 1, or 2

        self._last_logits = logits_t.numpy()

        true_labels = test_df["pass_bar"].values
        results = []
        for i, (_, row) in enumerate(test_df.iterrows()):
            true_label = int(true_labels[i])
            if decisions_idx[i] in (0, 1):
                decision = "automate"
                final_pred = int(decisions_idx[i])
            else:
                decision = "defer"
                final_pred = human.predict(row, "a2", true_label)

            results.append(
                {
                    "true_label": true_label,
                    "model_prob": float(
                        F.softmax(logits_t[i], dim=0)[1].item()
                    ),
                    "set_size":   None,
                    "decision":   decision,
                    "final_pred": final_pred,
                    "correct":    int(final_pred == true_label),
                    "subgroup":   row["subgroup"],
                }
            )
        return pd.DataFrame(results)

    def get_logits(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits for shape (n, 3). Useful for risk-coverage curves.

        Args:
            X : shape (n, d) scaled feature matrix.

        Returns:
            np.ndarray of shape (n, 3).
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X)).numpy()
