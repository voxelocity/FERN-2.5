"""(9) Neural cache for reasoning traces.

If a million users ask "What's Newton's second law?", recomputing the full
reasoning every time is wasteful. We hash a coarse signature of the reasoning
state and memoise the resulting output, like a CPU instruction cache. Coarse
quantisation makes near-identical states collide (a feature: similar questions
reuse the same trace). Exact-match only; an approximate-NN cache is the obvious
upgrade.
"""

from collections import OrderedDict
import torch

from .config import FERNConfig


class NeuralCache:
    def __init__(self, config: FERNConfig):
        self.config = config
        self.store: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _sig(self, vec: torch.Tensor) -> tuple:
        # quantise to a few buckets so similar states share a key
        levels = 2 ** self.config.cache_signature_bits
        v = torch.tanh(vec.detach().float())              # bound to (-1,1)
        q = torch.round((v * 0.5 + 0.5) * (levels - 1)).to(torch.int32)
        return tuple(q.tolist())

    def get(self, vec: torch.Tensor):
        key = self._sig(vec)
        if key in self.store:
            self.hits += 1
            self.store.move_to_end(key)
            return self.store[key]
        self.misses += 1
        return None

    def put(self, vec: torch.Tensor, value: torch.Tensor):
        key = self._sig(vec)
        self.store[key] = value.detach()
        self.store.move_to_end(key)
        while len(self.store) > self.config.cache_max_entries:
            self.store.popitem(last=False)

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0,
                "size": len(self.store)}
