"""(NEW) Neuromodulator — one global "brain-state" that gates the whole network.

Biological brains have global chemical signals (dopamine, noradrenaline,
acetylcholine) that put the whole cortex into a mode — alert vs relaxed, explore
vs exploit, focused vs diffuse. Here a small recurrent controller reads a coarse
summary of the current input (mean activity + mean salience) and emits a handful
of scalar gates that modulate everything at once:

    think   : how hard to reason   -> scales equilibrium iterations / depth
    mem_read/mem_write : how much to trust / commit to fast-weight memory
    route_temp : sharp vs exploratory expert routing

Instead of five independent controllers, the model learns a single low-dim
"cognitive mode" that coordinates them. Cheap (a tiny GRU) and brain-inspired.
"""

import torch
import torch.nn as nn

from .config import FERNConfig


class Neuromodulator(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        D, G = config.d_model, config.neuromod_dim
        # input: mean hidden (D) + [mean salience, salience std] (2)
        self.cell = nn.GRUCell(D + 2, G)
        self.heads = nn.Linear(G, 4)   # think, mem_read, mem_write, route_temp
        self.register_buffer("g0", torch.zeros(1, G), persistent=True)

    def forward(self, hidden: torch.Tensor, salience: torch.Tensor,
                state: torch.Tensor | None = None):
        """hidden: [B,T,D], salience: [B,T] -> (gates dict, new_state [B,G])."""
        B = hidden.shape[0]
        summary = torch.cat([
            hidden.mean(dim=1),
            salience.mean(dim=1, keepdim=True),
            salience.std(dim=1, keepdim=True),
        ], dim=-1)
        if state is None:
            state = self.g0.expand(B, -1).contiguous()
        g = self.cell(summary, state)
        raw = self.heads(g)
        gates = {
            "think": torch.sigmoid(raw[:, 0]),        # 0..1
            "mem_read": torch.sigmoid(raw[:, 1]),
            "mem_write": torch.sigmoid(raw[:, 2]),
            "route_temp": 0.5 + torch.sigmoid(raw[:, 3]),  # 0.5..1.5
        }
        return gates, g
