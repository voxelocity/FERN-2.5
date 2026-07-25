"""(3) Variable-precision neurons.

Most frameworks give every weight the same dtype. Here each linear carries its
own assigned bit-width. Frequently-used experts keep high precision; rarely-used
ones are quantised hard; dormant ones can be dropped to 4-bit. We simulate this
with straight-through fake-quantisation so it is also quantisation-aware during
training, and expose a realistic memory estimate so you can watch the footprint
shrink as precision is reassigned.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 16:
        return w
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max().clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax) * scale
    # straight-through estimator: gradients flow as if no quantisation
    return w + (q - w).detach()


class VariablePrecisionLinear(nn.Module):
    def __init__(self, in_f: int, out_f: int, bias: bool = True, bits: int = 16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, 0, 0.02)
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.in_features, self.out_features = in_f, out_f
        # not a Parameter — just an assigned precision level
        self.register_buffer("bits", torch.tensor(bits), persistent=True)

    def set_precision(self, bits: int):
        self.bits.fill_(bits)

    def forward(self, x):
        w = fake_quant(self.weight, int(self.bits.item()))
        return F.linear(x, w, self.bias)

    def memory_bytes(self) -> float:
        b = int(self.bits.item())
        nbytes = self.weight.numel() * b / 8.0
        if self.bias is not None:
            nbytes += self.bias.numel() * 2  # bias kept at fp16
        return nbytes
