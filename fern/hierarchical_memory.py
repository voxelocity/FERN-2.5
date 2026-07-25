"""(12) Hierarchical memory: working / short / long / archive tiers.

A shared associative memory bank. Compressed latents are written into the
`working` tier; as tiers overflow, vectors cascade down to slower, larger,
more-decayed tiers — mirroring working -> short-term -> long-term -> archive.

Reads are a differentiable attention over everything currently stored, so the
model learns *how to query* its own memory. Stored vectors are detached: the
bank is a non-parametric memory, not part of the backprop graph through time
(like a retrieval cache that happens to live inside the model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FERNConfig


class HierarchicalMemory(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        D = config.d_model
        self.tiers = config.mem_tiers
        self.caps = config.mem_tier_capacity
        self.decays = config.mem_tier_decay
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.gate = nn.Linear(D, D)
        # one storage tensor per tier, created lazily on first write
        self._store: list[torch.Tensor | None] = [None] * len(self.tiers)

    def reset(self):
        self._store = [None] * len(self.tiers)

    @torch.no_grad()
    def write(self, vectors: torch.Tensor):
        """vectors: [B, M, D] -> push flattened set into the working tier."""
        v = vectors.reshape(-1, vectors.shape[-1]).detach()
        self._push(0, v)

    @torch.no_grad()
    def _push(self, tier: int, v: torch.Tensor):
        if tier >= len(self.tiers):
            return  # falls off the end of archive — truly forgotten
        cur = self._store[tier]
        cur = v if cur is None else torch.cat([cur, v], dim=0)
        if self.decays[tier] > 0 and cur.shape[0] > 0:
            cur = cur * (1.0 - self.decays[tier])  # gentle forgetting
        cap = self.caps[tier]
        if cur.shape[0] > cap:
            overflow, cur = cur[:-cap], cur[-cap:]  # oldest cascade down
            self._push(tier + 1, overflow)
        self._store[tier] = cur

    def _all(self, device) -> torch.Tensor | None:
        parts = [s.to(device) for s in self._store if s is not None and s.numel()]
        if not parts:
            return None
        return torch.cat(parts, dim=0)  # [S, D]

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """query: [B,T,D] -> memory-conditioned vector [B,T,D] (zeros if empty)."""
        bank = self._all(query.device)
        if bank is None:
            return torch.zeros_like(query)
        B, T, D = query.shape
        q = self.q(query)                          # [B,T,D]
        k = self.k(bank)                           # [S,D]
        att = F.softmax((q @ k.t()) / (D ** 0.5), dim=-1)  # [B,T,S]
        read = att @ bank                          # [B,T,D]
        return torch.sigmoid(self.gate(query)) * read  # learned read gate

    def stats(self) -> dict:
        return {t: (0 if s is None else int(s.shape[0]))
                for t, s in zip(self.tiers, self._store)}
