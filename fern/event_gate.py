"""(7) Event-driven attention / salience gating.

Humans don't spend equal compute on every word. A cheap salience head scores
each token; high-salience tokens (new entities, contradictions, planning cues)
get promoted to "global anchors" that everyone can attend to, and they also
ponder longer in the fractal core. Low-salience tokens ride a cheap residual.

The salience signal is learned end-to-end from the task loss plus an optional
"surprise" signal (how much a token's representation changes after one step).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FERNConfig


class EventGate(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        self.score = nn.Sequential(
            nn.Linear(config.d_model, config.event_salience_dim),
            nn.GELU(),
            nn.Linear(config.event_salience_dim, 1),
        )

    def forward(self, x: torch.Tensor, surprise: torch.Tensor | None = None):
        """x: [B, T, D] -> salience [B, T] in (0,1), global_idx [B, K]."""
        B, T, _ = x.shape
        logits = self.score(x).squeeze(-1)            # [B, T]
        if surprise is not None:
            logits = logits + surprise
        salience = torch.sigmoid(logits)

        k = min(self.config.event_global_topk, T)
        # Most-salient tokens become global anchors for sparse attention.
        global_idx = torch.topk(salience, k=k, dim=1).indices  # [B, k]
        return salience, global_idx
