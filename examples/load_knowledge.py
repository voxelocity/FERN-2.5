"""Demonstrates (2) external knowledge storage.

Encodes fact strings into the model's query space and stores them in the
external KnowledgeStore. At inference the model retrieves and fuses them, so the
facts never have to live in the weights. Swap KnowledgeStore for FAISS / a real
vector DB to scale to a 100TB knowledge graph without changing the model.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fern import FERN, FERNConfig, ByteTokenizer


@torch.no_grad()
def encode_fact(model: FERN, tok: ByteTokenizer, text: str) -> torch.Tensor:
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    pos = torch.arange(ids.shape[1]).clamp_max(model.config.max_seq_len - 1)
    x = model.token_emb(ids) + model.pos_emb(pos)[None]
    # project into the same space the retriever queries with
    q = model.retriever.to_query(x).mean(dim=1)           # [1, K]
    return q.squeeze(0)


def main():
    config = FERNConfig()
    model = FERN(config).eval()
    tok = ByteTokenizer(config)

    facts = [
        "Newton's second law: force equals mass times acceleration.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The capital of France is Paris.",
        "Python lists are mutable; tuples are immutable.",
    ]
    for f in facts:
        model.store.add(encode_fact(model, tok, f), payload=f)

    print(f"stored {len(model.store)} facts in the external knowledge graph")

    # retrieve for a query
    q = encode_fact(model, tok, "what is the boiling point of water")
    vals, texts = model.store.query(q.unsqueeze(0), k=2)
    print("top retrieved:", texts[0])


if __name__ == "__main__":
    main()
