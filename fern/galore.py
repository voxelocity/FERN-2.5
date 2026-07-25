"""GaLore — Gradient Low-Rank Projection (FERN-2.5 eco, BUILD_FERN_OPTIMIZED §5).

Adam keeps two full-size moment tensors (m, v) per parameter — for a big 2D
weight that optimizer state is ~2x the weight bytes. GaLore instead projects
each 2D gradient into a rank-`r` subspace (via a periodic SVD of the gradient),
runs Adam's moments in rank-`r`, and projects the low-rank update back up. The
moment tensors then live in r*(m+n) instead of m*n — roughly halving optimizer
VRAM at r=128 on the eco/small sizes, which is exactly the memory a 12GB card
runs out of first.

This is the tractable, proven-at-LLM-scale member of the "train in a compressed
space" family. 1D params (norms, biases, region_bias) get plain AdamW — only
the large 2D matrices are worth projecting. Composes with bf16/fp8 and offload;
does NOT compose with runtime expert growth (keep maintenance off), same caveat
as the vectorized MoE.

Reference: Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient
Low-Rank Projection" (2024). This is a compact, self-contained reimplementation.
"""

import torch
from torch.optim import Optimizer


class _GaLoreProjector:
    """Holds the projection matrix P for one 2D weight and refreshes it every
    `gap` steps from the current gradient's top-r singular subspace."""

    def __init__(self, rank: int, gap: int, scale: float):
        self.rank = rank
        self.gap = gap
        self.scale = scale
        self.proj = None          # the P matrix (kept in the grad's dtype/device)
        self.side = None          # "left" (project rows) or "right" (project cols)

    def _svd_subspace(self, grad: torch.Tensor):
        # SVD wants fp32 for stability; cast back to the grad dtype afterwards.
        m, n = grad.shape
        U, _, Vh = torch.linalg.svd(grad.float(), full_matrices=False)
        r = min(self.rank, m, n)
        if m >= n:
            # tall/square: project the row space -> P is [m, r], back-proj P @ x
            self.side = "left"
            self.proj = U[:, :r].to(grad.dtype)
        else:
            # wide: project the column space -> P is [n, r], back-proj x @ P^T
            self.side = "right"
            self.proj = Vh[:r, :].t().to(grad.dtype)

    def project(self, grad: torch.Tensor, step: int) -> torch.Tensor:
        if self.proj is None or step % self.gap == 0:
            self._svd_subspace(grad)
        if self.side == "left":
            return self.proj.t() @ grad          # [r, n]
        return grad @ self.proj                  # [m, r]

    def project_back(self, low: torch.Tensor) -> torch.Tensor:
        if self.side == "left":
            return self.scale * (self.proj @ low)      # [m, n]
        return self.scale * (low @ self.proj.t())      # [m, n]


class GaLoreAdamW(Optimizer):
    """AdamW where 2D params tagged `galore=True` optimize in a low-rank
    subspace. Build the param groups with `build_galore_param_groups`."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, rank=128, update_gap=200, scale=0.25):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=rank, update_gap=update_gap, scale=scale,
                        galore=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            use_galore = group.get("galore", False)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = None       # lazily sized (low-rank if galore)
                    state["exp_avg_sq"] = None
                    if use_galore:
                        state["proj"] = _GaLoreProjector(
                            group["rank"], group["update_gap"], group["scale"])
                state["step"] += 1
                t = state["step"]

                g = grad
                projector = None
                if use_galore and grad.ndim == 2:
                    projector = state["proj"]
                    g = projector.project(grad, t)

                if state["exp_avg"] is None:
                    state["exp_avg"] = torch.zeros_like(g)
                    state["exp_avg_sq"] = torch.zeros_like(g)
                m, v = state["exp_avg"], state["exp_avg_sq"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                update = (m / bc1) / denom            # normalized step (low-rank if galore)

                if projector is not None:
                    update = projector.project_back(update)   # back to full shape

                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


def build_galore_param_groups(model, rank=128, update_gap=200, scale=0.25):
    """Split the model's params: large 2D matrices -> GaLore group, everything
    else (norms, biases, embeddings, 1D) -> plain AdamW group. Embeddings/lm_head
    are 2D but tied and vocab-shaped; we still project them since they are among
    the largest tensors and benefit most, but skip anything with a tiny min-dim
    where low-rank buys nothing."""
    galore, plain = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and min(p.shape) > rank:
            galore.append(p)
        else:
            plain.append(p)
    groups = [
        {"params": galore, "galore": True, "rank": rank,
         "update_gap": update_gap, "scale": scale},
        {"params": plain, "galore": False},
    ]
    return groups, len(galore), len(plain)
