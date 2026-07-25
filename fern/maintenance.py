"""(3) precision assignment + (5) dynamic growth, run out-of-band.

These are the "living filesystem" operations you call periodically during
training or deployment — not part of the forward pass. They read the usage
statistics the MoE accumulates and reshape the network: rare experts get
crushed to 4-bit, dead ones get pruned, overloaded regions grow new experts.
"""

import torch
from .model import FERN
from .precision import VariablePrecisionLinear


@torch.no_grad()
def assign_precision_by_usage(model: FERN) -> dict:
    """High-traffic experts keep 16-bit; rare ones drop to 8/4-bit."""
    moe = model.core.block.moe
    usage = moe.expert_usage()                      # [E]
    if usage.numel() == 0:
        return {}
    order = torch.argsort(usage, descending=True)
    buckets = model.config.precision_buckets        # e.g. [16, 8, 4]
    n = len(order)
    assigned = {}
    for rank, e in enumerate(order.tolist()):
        frac = rank / max(n - 1, 1)
        bits = buckets[min(int(frac * len(buckets)), len(buckets) - 1)]
        for mod in moe.experts[e].modules():
            if isinstance(mod, VariablePrecisionLinear):
                mod.set_precision(bits)
        assigned[e] = bits
    return assigned


@torch.no_grad()
def model_memory_bytes(model: FERN) -> dict:
    """Footprint accounting that respects each expert's assigned precision."""
    vp_bytes, other = 0.0, 0.0
    for m in model.modules():
        if isinstance(m, VariablePrecisionLinear):
            vp_bytes += m.memory_bytes()
    seen = set(id(m) for m in model.modules() if isinstance(m, VariablePrecisionLinear)
               for p in m.parameters())
    for p in model.parameters():
        if id(p) not in seen:
            other += p.numel() * 2  # everything else at fp16
    total = vp_bytes + other
    return {"expert_bytes": vp_bytes, "other_bytes": other,
            "total_mb": total / 1e6}


@torch.no_grad()
def evolve_experts(model: FERN, grow_threshold: float = 3.0,
                   prune_threshold: float = 0.05) -> dict:
    """Grow overloaded experts, prune idle ones. Call every N steps."""
    moe = model.core.block.moe
    usage = moe.expert_usage().float()
    if usage.numel() == 0 or usage.sum() == 0:
        return {"grew": 0, "pruned": 0}
    share = usage / usage.sum()
    mean = 1.0 / usage.numel()
    grew = pruned = 0
    if (share.max() > grow_threshold * mean).item():
        region = model.config.region_of_expert(int(share.argmax()))
        moe.add_expert(region)
        grew = 1
    # prune at most one idle expert per call, keep at least top_k+1 experts
    if len(moe.experts) > model.config.moe_top_k + 1:
        idle = (share < prune_threshold * mean).nonzero(as_tuple=True)[0]
        if idle.numel() > 0:
            moe.prune_expert(int(idle[0]))
            pruned = 1
    # reset counters after an evolution step
    for e in moe.experts:
        e.tokens_seen.zero_()
    return {"grew": grew, "pruned": pruned, "experts": len(moe.experts)}
