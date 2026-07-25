"""(8) Predictive state compression + (11) continuous-time latent dynamics.

A million-token chat shouldn't live as a million key/value pairs. We carry a
fixed set of `n_latent_tokens` latent "concept" vectors. Each segment, the
latents (as queries) read the new hidden states (cross-attention) and update.
The update is an Euler step of a learned ODE so the latent state evolves
*continuously* — compute happens in proportion to how much the state actually
changes, not per discrete layer.

These latents are the model's compressed working state: they are fed back as
global memory to attention and persist across segments (recurrent memory).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FERNConfig


class StateCompressor(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        D, M = config.d_model, config.n_latent_tokens
        self.init_latent = nn.Parameter(torch.randn(M, D) * 0.02)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        # the ODE vector field f(latent, context)
        self.f = nn.Sequential(
            nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D)
        )
        self.norm = nn.LayerNorm(D)
        self.h = config.n_heads

    def initial_state(self, batch: int, device) -> torch.Tensor:
        return self.init_latent.unsqueeze(0).expand(batch, -1, -1).contiguous().to(device)

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """latent: [B,M,D], hidden: [B,T,D] -> updated latent [B,M,D]."""
        B, M, D = latent.shape
        dh = D // self.h
        q = self.q(latent).view(B, M, self.h, dh).transpose(1, 2)
        k = self.k(hidden).view(B, -1, self.h, dh).transpose(1, 2)
        v = self.v(hidden).view(B, -1, self.h, dh).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) / (dh ** 0.5), dim=-1)
        context = (att @ v).transpose(1, 2).contiguous().view(B, M, D)

        deriv = self.f(torch.cat([latent, context], dim=-1))   # dlatent/dt
        if self.config.use_continuous_time:
            latent = latent + self.config.ct_dt * deriv         # Euler step
        else:
            latent = latent + deriv
        return self.norm(latent)
