"""Fetch a code training corpus for FERN and tokenize it to a memory-mapped
.bin file (fast, near-zero RAM at train time — the standard approach).

Output is a single uint16 .bin on drive E (vocab 267 fits in uint16). train.py
memory-maps it and samples windows, so a 50 GB corpus costs no extra RAM.

Examples
--------
HuggingFace code dataset (needs `pip install datasets`):
    python prepare_data.py --dataset codeparrot/codeparrot-clean \
        --column content --max_mb 4000 --out E:\fern_data\py.bin

Local files instead (no download): point --src at a folder; it walks source
files and tokenizes them:
    python prepare_data.py --src E:\some\code --out E:\fern_data\local.bin
"""

import os
os.environ.setdefault("HF_HOME", r"E:\hf_cache")    # keep HF cache off C:

import argparse
import numpy as np

from fern import FERNConfig, make_tokenizer
from fern.data import CODE_EXTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="HF dataset id")
    ap.add_argument("--config", default=None, help="HF dataset config/name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--column", default="content")
    ap.add_argument("--src", default=None, help="local folder to tokenize instead")
    ap.add_argument("--tokenizer", default=None,
                    help="BPE tokenizer.json (train_tokenizer.py). MUST match the "
                         "one you train the model with. Omitted = byte-level.")
    ap.add_argument("--max_mb", type=float, default=2000.0,
                    help="token budget in millions (uint16 -> ~2 bytes/token on disk)")
    ap.add_argument("--out", default=r"E:\fern_data\corpus.bin")
    args = ap.parse_args()

    if args.tokenizer:
        from tokenizers import Tokenizer
        vs = Tokenizer.from_file(args.tokenizer).get_vocab_size() + FERNConfig.N_SPECIAL
        cfg = FERNConfig.bpe(args.tokenizer, vocab_size=vs)
    else:
        cfg = FERNConfig()
    tok = make_tokenizer(cfg)
    assert cfg.vocab_size <= 65536, "vocab exceeds uint16; widen the .bin dtype"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    budget = int(args.max_mb * 1e6)         # ~1 token per byte at byte level
    written = 0
    n = 0

    def docs():
        if args.src:
            for root, _, files in os.walk(args.src):
                if any(p in root for p in (".git", "node_modules", "__pycache__")):
                    continue
                for fn in files:
                    if fn.endswith(CODE_EXTS):
                        try:
                            with open(os.path.join(root, fn), "r",
                                      encoding="utf-8", errors="ignore") as f:
                                yield f.read()
                        except OSError:
                            continue
        else:
            from datasets import load_dataset
            ds = load_dataset(args.dataset, args.config, split=args.split,
                              streaming=True)
            for ex in ds:
                t = ex.get(args.column)
                if t:
                    yield t

    with open(args.out, "wb") as fbin:
        for text in docs():
            ids = tok.encode(text, add_bos=True, add_eos=True)
            np.asarray(ids, dtype=np.uint16).tofile(fbin)
            written += len(ids)
            n += 1
            if n % 1000 == 0:
                print(f"  {n} docs, {written/1e6:.1f}M tokens")
            if written >= budget:
                break

    print(f"done: {n} docs, {written/1e6:.1f}M tokens -> {args.out}")
    print(f"train with:  python train.py --bin {args.out} ...")


if __name__ == "__main__":
    main()
