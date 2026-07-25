"""(2) Permanent knowledge storage outside the weights.

The network learns *reasoning*; facts live in an external store the model learns
to *query*. Here the store is a simple in-memory vector index (swap for FAISS /
a real vector DB / a 100TB knowledge graph — the interface is the same: given a
query vector, return the top-k fact vectors and their payloads).

`KnowledgeRetriever` projects the model's hidden state into a query, pulls the
top-k facts, and fuses them back through a learned gate. When the store is empty
it is a no-op, so the model trains fine before you've loaded any knowledge.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FERNConfig


class KnowledgeStore:
    """Non-neural external memory. Persisted/loaded as plain tensors + text."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: torch.Tensor | None = None   # [S, dim], L2-normalised
        self.payloads: List[str] = []

    def __len__(self):
        return 0 if self.vectors is None else self.vectors.shape[0]

    def add(self, vector: torch.Tensor, payload: str):
        v = F.normalize(vector.detach().float().view(1, -1), dim=-1)
        self.vectors = v if self.vectors is None else torch.cat([self.vectors, v], 0)
        self.payloads.append(payload)

    def query(self, q: torch.Tensor, k: int) -> Tuple[torch.Tensor, List[List[str]]]:
        """q: [N, dim] -> values [N, k, dim], list of payload lists."""
        if self.vectors is None:
            return torch.zeros(q.shape[0], k, self.dim, device=q.device), [[] for _ in range(q.shape[0])]
        qn = F.normalize(q.float(), dim=-1)
        sims = qn @ self.vectors.to(q.device).t()        # [N, S]
        kk = min(k, self.vectors.shape[0])
        topv, topi = torch.topk(sims, kk, dim=-1)
        vals = self.vectors.to(q.device)[topi]           # [N, kk, dim]
        if kk < k:                                       # pad
            pad = torch.zeros(q.shape[0], k - kk, self.dim, device=q.device)
            vals = torch.cat([vals, pad], dim=1)
        texts = [[self.payloads[i] for i in row.tolist()] for row in topi]
        return vals, texts

    def save(self, path: str):
        torch.save({"vectors": self.vectors, "payloads": self.payloads}, path)

    def load(self, path: str):
        d = torch.load(path, weights_only=False)
        self.vectors, self.payloads = d["vectors"], d["payloads"]


class KnowledgeRetriever(nn.Module):
    def __init__(self, config: FERNConfig, store: KnowledgeStore):
        super().__init__()
        self.config = config
        self.store = store
        D, K = config.d_model, config.knowledge_dim
        self.to_query = nn.Linear(D, K, bias=False)
        self.from_value = nn.Linear(K, D, bias=False)
        self.gate = nn.Linear(D, D)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: [B,T,D] -> knowledge-conditioned vector [B,T,D]."""
        if not self.config.use_knowledge or len(self.store) == 0:
            return torch.zeros_like(hidden)
        B, T, D = hidden.shape
        q = self.to_query(hidden).reshape(B * T, -1)
        vals, _ = self.store.query(q, self.config.knowledge_topk)   # [BT,k,K]
        q3 = q.unsqueeze(1)                                         # [BT,1,K]
        att = F.softmax((q3 @ vals.transpose(1, 2)) / (self.config.knowledge_dim ** 0.5), dim=-1)
        fused = (att @ vals).squeeze(1)                            # [BT,K]
        out = self.from_value(fused).view(B, T, D)
        return torch.sigmoid(self.gate(hidden)) * out
