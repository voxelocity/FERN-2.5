"""(1)(6) Sparse Mixture-of-Experts organised into cognitive regions,
with (5) dynamic growth and (3) variable-precision hooks.

Only `moe_top_k` experts fire per token. FERN-1 implemented this with a Python
`for`-loop over experts, which fired many tiny GPU kernels and left the card
dispatch-bound (~59% util on a 5070). **FERN-2 vectorizes it** (ROADMAP Phase 0a):

  * The fast path (`_forward_batched`, used when experts are NOT offloaded) does
    a **capacity-based batched GEMM** — the GShard / Switch-Transformer dispatch.
    Tokens are routed into a dense `[E, capacity, D]` buffer, then ALL experts run
    as exactly two big `baddbmm` calls over stacked expert weights. No per-expert
    Python dispatch; the GPU sees a few large kernels instead of E*2 tiny ones.
  * The offload path (`_forward_offload`) keeps the streaming loop, but only over
    experts that actually received tokens — offload is bandwidth-bound by design,
    so the loop is not the bottleneck there.

Experts are grouped into named regions (language / math / code / ...) so routing
is interpretable and a region can be grown/pruned at runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from .config import FERNConfig
from .precision import VariablePrecisionLinear, fake_quant


class Expert(nn.Module):
    """A small gated MLP. Its two linears are variable-precision aware."""

    def __init__(self, config: FERNConfig):
        super().__init__()
        self.fc1 = VariablePrecisionLinear(config.d_model, config.expert_hidden)
        self.fc2 = VariablePrecisionLinear(config.expert_hidden, config.d_model)
        self.act = nn.GELU()
        # (3) usage stats drive precision assignment later.
        self.register_buffer("tokens_seen", torch.zeros(1), persistent=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CognitiveMoE(nn.Module):
    def __init__(self, config: FERNConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        # learned per-region bias lets the model specialise routing by region.
        self.region_bias = nn.Parameter(torch.zeros(config.n_experts))
        self.top_k = config.moe_top_k
        # ---- offloading state ----
        self.offload = False
        self.offload_device = torch.device("cpu")
        self._eval_cache: dict = {}     # expert idx -> moved param/buffer dict

    # ---- expert offloading -------------------------------------------------
    def enable_offload(self, offload_device="cpu", pin=True):
        """Park all experts on `offload_device` (CPU); they stream to the GPU
        on demand inside forward. The optimizer keeps updating the CPU master
        weights (grads flow back through the device copy), so this is training-
        safe, just bandwidth-bound."""
        self.offload = True
        self.offload_device = torch.device(offload_device)
        self.experts.to(self.offload_device)
        if pin and self.offload_device.type == "cpu" and torch.cuda.is_available():
            for p in self.experts.parameters():
                try:
                    p.data = p.data.pin_memory()
                except Exception:
                    pass
        self._eval_cache.clear()

    def _run_expert(self, e: int, x: torch.Tensor) -> torch.Tensor:
        """Compute expert `e` on x's device, streaming weights if offloaded."""
        expert = self.experts[e]
        if not self.offload:
            return expert(x)
        dev = x.device
        # in eval the master weights are frozen -> cache the GPU copies
        if not self.training and not torch.is_grad_enabled():
            pb = self._eval_cache.get(e)
            if pb is None:
                pb = {n: t.to(dev, non_blocking=True)
                      for n, t in (*expert.named_parameters(), *expert.named_buffers())}
                self._eval_cache[e] = pb
            return functional_call(expert, pb, (x,))
        # training: copy fresh each step so grads reach the CPU master leaves
        pb = {n: t.to(dev, non_blocking=True)
              for n, t in (*expert.named_parameters(), *expert.named_buffers())}
        return functional_call(expert, pb, (x,))

    # ---- (5) dynamic model growth --------------------------------------
    @torch.no_grad()
    def add_expert(self, region: str | None = None):
        """Append a fresh expert (optionally tagged to a region) at runtime."""
        dev = self.offload_device if self.offload else self.router.weight.device
        new = Expert(self.config).to(dev)
        self.experts.append(new)
        self._eval_cache.clear()
        # grow the router output dim, preserving learned weights.
        old = self.router
        self.router = nn.Linear(old.in_features, old.out_features + 1, bias=False).to(old.weight.device)
        self.router.weight.data[:-1] = old.weight.data
        self.router.weight.data[-1].normal_(0, 0.02)
        self.region_bias = nn.Parameter(
            torch.cat([self.region_bias.data, torch.zeros(1, device=self.region_bias.device)])
        )

    @torch.no_grad()
    def prune_expert(self, idx: int):
        """Remove an under-used expert and shrink the router."""
        del self.experts[idx]
        keep = [i for i in range(self.router.out_features) if i != idx]
        old = self.router
        self.router = nn.Linear(old.in_features, len(keep), bias=False).to(old.weight.device)
        self.router.weight.data = old.weight.data[keep]
        self.region_bias = nn.Parameter(self.region_bias.data[keep])
        self._eval_cache.clear()

    # ---- routing + dispatch ------------------------------------------------
    def forward(self, x: torch.Tensor, temperature=None):
        """x: [B,T,D] -> (out [B,T,D], aux_loss scalar).

        `temperature` (from the neuromodulator) sharpens/softens routing: <1
        sharp/exploit, >1 soft/explore.
        """
        B, T, D = x.shape
        E = len(self.experts)
        flat = x.reshape(-1, D)                                  # [N, D]
        N = flat.shape[0]

        logits = self.router(flat) + self.region_bias[:E]        # [N, E]
        if temperature is not None:
            t = temperature.view(B, 1, 1).expand(B, T, 1).reshape(N, 1)
            logits = logits / t.clamp_min(1e-3)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)       # [N, k]
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)    # renormalise

        # load-balancing aux loss (Switch-Transformer style) — identical in both paths
        importance = probs.mean(dim=0)                           # [E]
        load = (topi.reshape(-1).bincount(minlength=E).float() / max(N, 1))
        aux = self.config.load_balance_weight * E * (importance * load).sum()

        if self.offload:
            out = self._forward_offload(flat, topv, topi, E)
        else:
            out = self._forward_batched(flat, topv, topi, E)

        return out.view(B, T, D), aux

    # ---- fast path: capacity-based batched GEMM (no per-expert loop) --------
    def _forward_batched(self, flat: torch.Tensor, topv: torch.Tensor,
                         topi: torch.Tensor, E: int) -> torch.Tensor:
        """All experts run as two big batched matmuls over a dense dispatch
        buffer. This is the GShard/Switch formulation: each expert gets a fixed
        `capacity`; tokens beyond it are dropped (their residual still carries
        them through the block, so dropping only removes that expert's edit)."""
        N, D = flat.shape
        k = self.top_k
        dev = flat.device
        S = N * k
        capacity = max(1, int(self.config.moe_capacity_factor * N * k / E))

        expert_idx = topi.reshape(-1)                            # [S]
        gate = topv.reshape(-1)                                  # [S]
        token_idx = torch.arange(N, device=dev).repeat_interleave(k)  # [S]

        # position of each (token,slot) within its expert's queue, so we can pack
        # into a dense [E, capacity, D] buffer and enforce capacity.
        counts = torch.bincount(expert_idx, minlength=E)         # [E]
        group_start = torch.cumsum(counts, 0) - counts           # exclusive prefix
        sort_val, sort_perm = torch.sort(expert_idx)             # group by expert
        ranks_sorted = torch.arange(S, device=dev) - group_start[sort_val]
        pos = torch.empty(S, dtype=torch.long, device=dev)
        pos[sort_perm] = ranks_sorted                            # rank within expert
        keep = pos < capacity                                    # [S] capacity mask

        ke = expert_idx[keep]                                    # [n_keep]
        kp = pos[keep]
        kt = token_idx[keep]
        kg = gate[keep]

        dispatch = flat.new_zeros(E, capacity, D)
        dispatch[ke, kp] = flat[kt]                              # scatter inputs

        # Run experts in CHUNKS. Stacking ALL expert weights into one [E,H,D]
        # tensor makes a transient copy of every expert (≈10 GB at the `large`
        # preset, held through backward) that risks OOM on an 80 GB card. Chunks
        # bound that copy to `chunk` experts at a time; the result is bit-for-bit
        # identical because each chunk writes a disjoint slice of `eo`.
        chunk = self.config.moe_stack_chunk or E
        eo = flat.new_empty(E, capacity, D)                     # small: E*cap*D
        for lo in range(0, E, chunk):
            hi = min(lo + chunk, E)
            experts = [self.experts[i] for i in range(lo, hi)]
            W1 = torch.stack([fake_quant(e.fc1.weight, int(e.fc1.bits.item()))
                              for e in experts])                 # [c, H, D]
            b1 = torch.stack([e.fc1.bias for e in experts])      # [c, H]
            W2 = torch.stack([fake_quant(e.fc2.weight, int(e.fc2.bits.item()))
                              for e in experts])                 # [c, D, H]
            b2 = torch.stack([e.fc2.bias for e in experts])      # [c, D]
            h = torch.baddbmm(b1.unsqueeze(1), dispatch[lo:hi],
                              W1.transpose(1, 2))                # [c,cap,H]
            h = F.gelu(h)
            eo[lo:hi] = torch.baddbmm(b2.unsqueeze(1), h,
                                      W2.transpose(1, 2))        # [c,cap,D]

        contrib = eo[ke, kp] * kg.unsqueeze(-1)                  # [n_keep, D]
        out = flat.new_zeros(N, D)
        out.index_add_(0, kt, contrib)

        # (3) usage stats for later precision assignment
        kept_counts = torch.bincount(ke, minlength=E).to(flat.dtype)
        for e in range(E):
            self.experts[e].tokens_seen += kept_counts[e]
        return out

    # ---- offload path: stream only the experts that got tokens --------------
    def _forward_offload(self, flat: torch.Tensor, topv: torch.Tensor,
                         topi: torch.Tensor, E: int) -> torch.Tensor:
        """Bandwidth-bound by design (experts live in CPU RAM), so we keep the
        gather-per-expert loop but skip experts with no routed tokens."""
        out = torch.zeros_like(flat)
        for e in range(E):
            sel = (topi == e)                                    # [N, k]
            if not sel.any():
                continue
            tok_mask = sel.any(dim=-1)                           # [N]
            idx = tok_mask.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            gate = (topv * sel).sum(dim=-1)[idx].unsqueeze(-1)   # [n,1]
            y = self._run_expert(e, flat[idx])                   # streams from CPU
            out[idx] += gate * y
            self.experts[e].tokens_seen += idx.numel()
        return out

    @torch.no_grad()
    def expert_usage(self) -> torch.Tensor:
        return torch.stack([e.tokens_seen for e in self.experts]).squeeze(-1)
