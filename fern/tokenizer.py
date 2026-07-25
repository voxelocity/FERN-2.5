"""Tokenizers for FERN-2.

Two interchangeable tokenizers share ONE interface and ONE id layout, so the
model code never changes when you switch:

  * `ByteTokenizer`  — dependency-free byte-level (vocab 268 = 256 bytes + 12
                       specials). The robust fallback / smoke-test default.
  * `BPETokenizer`   — a trained code-aware byte-level BPE (ROADMAP's #1 build:
                       byte-level wasted ~3-5x the steps to coherence). Wraps a
                       HuggingFace `tokenizers` model.

The 12 special tokens always live at the TOP of the vocab (ids
`[vocab_size-12, vocab_size)`), in the fixed order below, and the regular
("base") tokens occupy `[0, vocab_size-12)`. For bytes the base is exactly the
256 byte values; for BPE it's the learned word-pieces. `decode` strips every id
>= the first special id (== `config.pad_id`).
"""

from typing import List
from .config import FERNConfig

# Fixed order — must match FERNConfig.__post_init__'s assignment.
SPECIAL_ORDER = [
    "pad", "bos", "eos", "unk",
    "fim_prefix", "fim_suffix", "fim_middle",
    "sys", "user", "asst", "end", "mask",
]


def special_ids(config: FERNConfig) -> dict:
    return {
        "pad": config.pad_id, "bos": config.bos_id, "eos": config.eos_id,
        "unk": config.unk_id, "fim_prefix": config.fim_prefix_id,
        "fim_suffix": config.fim_suffix_id, "fim_middle": config.fim_middle_id,
        "sys": config.sys_id, "user": config.user_id, "asst": config.asst_id,
        "end": config.end_id, "mask": config.mask_id,
    }


class ByteTokenizer:
    def __init__(self, config: FERNConfig):
        self.config = config

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.config.bos_id] + ids
        if add_eos:
            ids = ids + [self.config.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        # base (non-special) ids are exactly the byte values 0..255
        base = self.config.n_base_tokens
        raw = bytes([i for i in ids if 0 <= i < base])
        return raw.decode("utf-8", errors="replace")


class BPETokenizer:
    """Wraps a trained `tokenizers.Tokenizer` (byte-level BPE) and layers the 12
    FERN special tokens on top at ids `[n_base, vocab_size)`. The underlying HF
    tokenizer knows ONLY the base pieces (ids `[0, n_base)`); specials are
    managed here so their ids are fully under FERN's control."""

    def __init__(self, config: FERNConfig, path: str | None = None):
        from tokenizers import Tokenizer  # local import: optional dependency
        self.config = config
        path = path or config.tokenizer_path
        if not path:
            raise ValueError("BPETokenizer needs a tokenizer_path (train one with "
                             "train_tokenizer.py)")
        self.tok = Tokenizer.from_file(path)
        self.n_base = config.n_base_tokens
        vs = self.tok.get_vocab_size()
        if vs != self.n_base:
            raise ValueError(
                f"tokenizer base vocab {vs} != config.n_base_tokens {self.n_base}. "
                f"Set config.vocab_size = {vs + config.N_SPECIAL} (base + 12 specials).")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        ids = self.tok.encode(text).ids
        if add_bos:
            ids = [self.config.bos_id] + ids
        if add_eos:
            ids = ids + [self.config.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        # drop all specials (ids >= n_base), decode the rest with the BPE model
        base = [int(i) for i in ids if 0 <= int(i) < self.n_base]
        return self.tok.decode(base)


def make_tokenizer(config: FERNConfig):
    """Return the tokenizer selected by `config.tokenizer_kind`."""
    if config.tokenizer_kind == "bpe":
        return BPETokenizer(config)
    return ByteTokenizer(config)
