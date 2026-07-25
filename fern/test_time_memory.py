"""(NEW) Test-time learning memory — weights that update DURING inference.

This is the "living model" idea made concrete and trainable. The memory is a
fast-weight matrix S (one per head) that is rewritten as the model reads the
sequence, using the **delta rule**:

    read :  o_t = q_t · S
    write:  S <- S + beta_t · k_tᵀ (v_t − k_t · S)     (error-correcting)

That update is *learning* — the network fits an associative map to the current
context on the fly, with no optimizer and no backprop-through-time at deploy.
It's the linear-attention / DeltaNet form of test-time training (Schmidhuber's
fast weights → Yang et al. 2024 DeltaNet → Google "Titans" 2025), chosen over
nonlinear gradient-TTT because it trains stably from scratch on one GPU.

For a coding model this means: as it reads your file, it *learns* the local
names, types and conventions and recalls them for the rest of the generation.
`beta` (the write strength / per-step learning rate) is itself predicted from
the token and scaled by the neuromodulator's mem_write gate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FERNConfig


class TestTimeMemory(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        self.h = config.ttm_heads
        self.d = config.d_model
        self.dh = self.d // self.h
        self.qkv = nn.Linear(self.d, 3 * self.d, bias=False)
        self.beta = nn.Linear(self.d, self.h)      # per-head write strength
        self.out = nn.Linear(self.d, self.d, bias=False)
        self.norm = nn.LayerNorm(self.d)

    def forward(self, x: torch.Tensor, write_gate: torch.Tensor | None = None,
                S: torch.Tensor | None = None):
        """x: [B,T,D] -> (o [B,T,D], S [B,h,dh,dh]).  S carries across segments."""
        B, T, D = x.shape
        xn = self.norm(x)
        q, k, v = self.qkv(xn).chunk(3, dim=-1)
        shape = (B, T, self.h, self.dh)
        q = F.normalize(q.view(shape), dim=-1)     # normalised keys/queries
        k = F.normalize(k.view(shape), dim=-1)     # stabilises the delta rule
        v = v.view(shape)
        beta = torch.sigmoid(self.beta(xn))        # [B,T,h]
        if write_gate is not None:
            beta = beta * write_gate.view(B, 1, 1)

        if S is None:
            S = x.new_zeros(B, self.h, self.dh, self.dh)

        outs = []
        # recurrent delta-rule scan (test-time learning, one step per token)
        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]          # [B,h,dh]
            ot = torch.einsum("bhd,bhde->bhe", qt, S)        # read
            pred = torch.einsum("bhd,bhde->bhe", kt, S)      # current memory of kt
            err = vt - pred                                  # prediction error
            S = S + beta[:, t].unsqueeze(-1).unsqueeze(-1) * \
                torch.einsum("bhd,bhe->bhde", kt, err)       # error-correcting write
            outs.append(ot)
        o = torch.stack(outs, dim=1).reshape(B, T, D)
        return self.out(o), S
