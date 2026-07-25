"""(4) Fractal network with two reasoning modes.

  "ponder"      — the original PonderNet adaptive-depth recursion (robust).
  "equilibrium" — Deep-Equilibrium latent reasoning: loop the SAME block to a
                  fixed point z* = Block(z*, x). This is "think in latent space
                  until the thought stops changing". Effective depth is
                  unbounded but memory is constant, and we use Jacobian-Free
                  Backprop (gradient only through the last step) so it trains
                  stably from scratch on one GPU. The neuromodulator's `think`
                  gate sets how many iterations to spend — cheap inputs settle
                  fast, hard reasoning gets more iterations. This is the single
                  biggest reasoning-ceiling lever for a small coding model.
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import FERNConfig
from .sparse_attention import SparseAttention
from .sparse_moe import CognitiveMoE


class ReasoningBlock(nn.Module):
    """The one block reused at every level / iteration of abstraction."""

    def __init__(self, config: FERNConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = SparseAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.moe = CognitiveMoE(config)
        self.depth_emb = nn.Embedding(config.max_fractal_depth, config.d_model)
        # equilibrium needs the SAME function each step -> inject the constant
        # input `inj` rather than a step-varying depth code.
        self.inject = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, z, global_idx, memory, step=None, inj=None, route_temp=None,
                block_size=None):
        h = z
        if step is not None:
            h = h + self.depth_emb(torch.tensor(step, device=z.device))
        if inj is not None:
            h = h + self.inject(inj)        # fixed input injection (DEQ)
        z = z + self.attn(self.norm1(h), global_idx, memory, block_size=block_size)
        moe_out, aux = self.moe(self.norm2(z), temperature=route_temp)
        z = z + moe_out
        return z, aux


class FractalCore(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        self.block = ReasoningBlock(config)
        self.halt = nn.Linear(config.d_model, 1)

    def _run_block(self, z, global_idx, memory, **kw):
        """Apply the reasoning block once. With config.reversible the block's
        internal activations are recomputed on the backward pass instead of
        cached (gradient checkpointing), so peak activation VRAM stops scaling
        with recursion depth — the FERN-2.5 "memory-free depth" feature. This
        is numerically exact (unlike a hand-written RevNet inverse) and a no-op
        when we're already inside torch.no_grad (the equilibrium solve phase)."""
        if self.config.reversible and torch.is_grad_enabled():
            # checkpoint needs tensor args positional; fold the kwargs into a
            # closure so non-tensor side inputs (memory, ints, None) pass through.
            def run(z_, gidx_):
                return self.block(z_, gidx_, memory, **kw)
            return checkpoint(run, z, global_idx, use_reentrant=False)
        return self.block(z, global_idx, memory, **kw)

    # ---- mode dispatch -----------------------------------------------------
    def forward(self, x, global_idx, memory, salience, infer=False,
                think_gate=None, route_temp=None, block_size=None):
        if self.config.reasoning_mode == "equilibrium":
            return self._equilibrium(x, global_idx, memory, think_gate, route_temp,
                                     block_size)
        return self._ponder(x, global_idx, memory, salience, infer, route_temp,
                            block_size)

    # ---- Deep-Equilibrium latent reasoning --------------------------------
    def _equilibrium(self, x, global_idx, memory, think_gate, route_temp,
                     block_size=None):
        cfg = self.config
        max_iters = cfg.deq_max_iters
        if think_gate is not None:                 # neuromodulator sets effort
            g = float(think_gate.mean().detach())
            if math.isfinite(g):                    # ignore NaN/Inf gates
                max_iters = max(2, int(round(max_iters * g)))

        z = torch.zeros_like(x)
        residual = x.new_tensor(0.0)
        damp = cfg.deq_damping
        iters = 0
        # 1) find the fixed point WITHOUT tracking gradients (cheap, deep).
        #    Damped Picard iteration z<-(1-d)z+d*Block(z) converges far more
        #    reliably than raw iteration for a learned, not-yet-contractive map.
        with torch.no_grad():
            for i in range(max_iters):
                fz, _ = self.block(z, global_idx, memory, inj=x,
                                   route_temp=route_temp, block_size=block_size)
                z_new = (1 - damp) * z + damp * fz
                residual = (z_new - z).norm() / (z.norm() + 1e-6)
                z = z_new
                iters = i + 1
                if residual < cfg.deq_tol:
                    break
        # 2) Jacobian-free backprop: re-run the last few steps WITH gradients
        z = z.detach()
        for _ in range(cfg.deq_grad_steps):
            fz, aux = self._run_block(z, global_idx, memory, inj=x,
                                      route_temp=route_temp, block_size=block_size)
            z = (1 - damp) * z + damp * fz
        # DEQ needs no halting regulariser; report convergence instead.
        ponder_loss = x.new_zeros(())
        info = {"avg_steps": float(iters), "steps_run": iters,
                "residual": float(residual), "mode": "equilibrium"}
        return z, aux, ponder_loss, info

    # ---- PonderNet adaptive depth (original) ------------------------------
    def _ponder(self, x, global_idx, memory, salience, infer, route_temp,
                block_size=None):
        B, T, D = x.shape
        N = self.config.max_fractal_depth
        y = torch.zeros_like(x)
        cum_p = torch.zeros(B, T, device=x.device)
        still = torch.ones(B, T, device=x.device)
        aux_total = x.new_zeros(())
        halt_dist = []
        steps_used = 0
        state = x
        for n in range(N):
            state, aux = self._run_block(state, global_idx, memory, step=n,
                                         route_temp=route_temp, block_size=block_size)
            aux_total = aux_total + aux
            steps_used = n + 1
            lam = torch.sigmoid(self.halt(state).squeeze(-1) - 3.0 * salience)
            if n == N - 1:
                lam = torch.ones_like(lam)
            p_n = still * lam
            y = y + p_n.unsqueeze(-1) * state
            halt_dist.append(p_n)
            cum_p = cum_p + p_n
            still = still * (1.0 - lam)
            if infer and (cum_p > self.config.inference_halt_threshold).all():
                break
        p = torch.stack(halt_dist, dim=0).clamp_min(1e-8)
        steps = torch.arange(1, p.shape[0] + 1, device=x.device).float()
        # geometric prior over halting step. Cap pg<1 so EVERY step keeps some
        # mass — at small depth pg=1 would zero out later steps and log(0)=-inf
        # would blow the loss to NaN.
        pg = min(0.9, 1.0 / max(1.0, N / 2.0))
        prior = (1 - pg) ** (steps - 1) * pg
        prior = (prior / prior.sum()).clamp_min(1e-6).view(-1, 1, 1)
        ponder_loss = (p * (p.log() - prior.log())).sum(0).mean()
        avg_steps = (p * steps.view(-1, 1, 1)).sum(0).mean().detach()
        info = {"avg_steps": float(avg_steps), "steps_run": steps_used,
                "mode": "ponder"}
        return y, aux_total, ponder_loss, info
