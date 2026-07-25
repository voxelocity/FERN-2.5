"""FERN — Fractal Event-driven Retrieval Network.

An ultra-lightweight LLM architecture built around sparse, brain-inspired
computation. See README.md for the mapping of modules -> design principles.
"""

from .config import FERNConfig
from .tokenizer import ByteTokenizer, BPETokenizer, make_tokenizer
from .model import FERN
from .knowledge_store import KnowledgeStore, KnowledgeRetriever
from .maintenance import (
    assign_precision_by_usage,
    model_memory_bytes,
    evolve_experts,
)
from .galore import GaLoreAdamW, build_galore_param_groups

__all__ = [
    "FERNConfig", "ByteTokenizer", "BPETokenizer", "make_tokenizer", "FERN",
    "KnowledgeStore", "KnowledgeRetriever",
    "assign_precision_by_usage", "model_memory_bytes", "evolve_experts",
    "GaLoreAdamW", "build_galore_param_groups",
]
__version__ = "0.2.5"
