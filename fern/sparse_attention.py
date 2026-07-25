"""(1) Sparse attention with (7) event-driven global anchors.

Each token attends to:
  * a local causal window of `attn_local_window` neighbours   (cheap, O(T*w))
  * a small set of event-selected global anchor tokens         (long-range)
  * any external "memory" vectors passed in (latent state,     (always global)
    hierarchical-memory reads, retrieved knowledge)

We express the pattern as an additive attention mask. The prototype still runs
a dense matmul under the hood — the mask defines the *algorithmic* sparsity. A
production build swaps this for a block-sparse / flash-sparse kernel so the FLOPs
actually drop; the rest of the model is unchanged. That swap is the single
biggest lever for "activate 500M of 500B params per token".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FERNConfig

NEG_INF = -1e9


class SparseAttention(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        self.h = config.n_heads
        self.d = config.d_model
        self.dh = self.d // self.h
        self.q = nn.Linear(self.d, self.d, bias=False)
        self.kv = nn.Linear(self.d, 2 * self.d, bias=False)
        self.o = nn.Linear(self.d, self.d, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def _seq_mask(self, T: int, global_idx: torch.Tensor, device,
                  block_size: int | None = None) -> torch.Tensor:
        """Boolean [B, T, T]: True where query i may attend to key j.

        AR mode (`block_size is None`): strictly causal — local causal window +
        causal global anchors. (FERN-1 behaviour, unchanged.)

        Block-diffusion mode (`block_size` set): **block-causal**. A token may
        attend bidirectionally to its WHOLE current block (so the denoiser sees
        both sides of a hole), and causally to earlier blocks via the local
        window or global anchors. This is the only attention change block
        diffusion needs; the reasoning core is untouched.
        """
        B = global_idx.shape[0]
        i = torch.arange(T, device=device)
        rel = i[:, None] - i[None, :]                       # [T, T] (>0 = past)

        is_global = torch.zeros(B, T, dtype=torch.bool, device=device)
        is_global.scatter_(1, global_idx, True)             # [B, T]

        if block_size is None:
            # local causal window
            local = (rel >= 0) & (rel < self.config.attn_local_window)
            local = local.unsqueeze(0).expand(B, T, T).clone()
            causal = (rel >= 0).unsqueeze(0)                # [1, T, T]
            global_allowed = is_global[:, None, :] & causal
            return local | global_allowed

        # block-causal: same block fully visible; earlier blocks causal+sparse
        blk = (i // block_size)                             # [T] block id
        same_block = (blk[:, None] == blk[None, :])         # [T, T]
        prev_block = (blk[:, None] > blk[None, :])          # strictly earlier blocks
        local = prev_block & (rel < self.config.attn_local_window)
        seq = same_block | local                            # [T, T]
        seq = seq.unsqueeze(0).expand(B, T, T).clone()
        global_allowed = is_global[:, None, :] & prev_block.unsqueeze(0)
        return seq | global_allowed

    def forward(self, x: torch.Tensor, global_idx: torch.Tensor,
                memory: torch.Tensor | None = None,
                block_size: int | None = None) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device

        kv_src = x if memory is None else torch.cat([x, memory], dim=1)
        Tk = kv_src.shape[1]

        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)        # [B,h,T,dh]
        k, v = self.kv(kv_src).chunk(2, dim=-1)
        k = k.view(B, Tk, self.h, self.dh).transpose(1, 2)               # [B,h,Tk,dh]
        v = v.view(B, Tk, self.h, self.dh).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)            # [B,h,T,Tk]

        # build [B, T, Tk] allow-mask: sequence part sparse, memory part open
        seq_allow = self._seq_mask(T, global_idx, device, block_size)   # [B,T,T]
        if memory is not None:
            M = memory.shape[1]
            mem_allow = torch.ones(B, T, M, dtype=torch.bool, device=device)
            allow = torch.cat([seq_allow, mem_allow], dim=2)            # [B,T,Tk]
        else:
            allow = seq_allow
        scores = scores.masked_fill(~allow.unsqueeze(1), NEG_INF)

        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                                  # [B,h,T,dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(out)
