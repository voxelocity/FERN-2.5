"""Train a code-aware byte-level BPE tokenizer for FERN-2.

ROADMAP / RESEARCH_FINDINGS flagged this as the #1 build: FERN-1 was byte-level,
which wasted ~3-5x the training steps to reach coherence and is the main source
of mid-word "drift". A BPE vocab makes each step predict a whole word-piece.

We train ONLY the base pieces (byte-level BPE, so there is never an OOV byte).
FERN's 12 special tokens are layered on top at fixed ids by the tokenizer
wrapper, so we ask the trainer for `vocab_size - 12` base pieces.

Examples
--------
Quick offline smoke (trains on FERN-2's own source):
    python train_tokenizer.py --vocab_size 2048 --out E:\\fern\\tok_smoke.json

Real code tokenizer from a HF dataset:
    python train_tokenizer.py --dataset codeparrot/codeparrot-clean --column content \\
        --max_docs 200000 --vocab_size 16384 --out E:\\fern\\tok_code.json

From a local folder of source:
    python train_tokenizer.py --src C:\\path\\to\\repo --vocab_size 16384 \\
        --out E:\\fern\\tok_code.json
"""

import os
os.environ.setdefault("HF_HOME", r"E:\fern\hf_cache")

import argparse

from fern.config import FERNConfig
from fern.data import CODE_EXTS

N_SPECIAL = FERNConfig.N_SPECIAL


def iter_local(paths, max_docs):
    n = 0
    for root, _, files in os.walk(paths):
        if any(p in root for p in (".git", "node_modules", "__pycache__", ".venv")):
            continue
        for fn in files:
            if not fn.endswith(CODE_EXTS):
                continue
            try:
                with open(os.path.join(root, fn), "r", encoding="utf-8",
                          errors="ignore") as f:
                    yield f.read()
            except OSError:
                continue
            n += 1
            if max_docs and n >= max_docs:
                return


def iter_hf(dataset, split, column, max_docs):
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    n = 0
    for ex in ds:
        text = ex.get(column)
        if not text:
            continue
        yield text
        n += 1
        if max_docs and n >= max_docs:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="HF dataset id")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--column", type=str, default="content")
    ap.add_argument("--src", type=str, default=None, help="local folder of code")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = no limit")
    ap.add_argument("--vocab_size", type=int, default=16384,
                    help="TOTAL vocab incl. 12 specials; base pieces = this - 12")
    ap.add_argument("--out", type=str, default=r"E:\fern\tok_code.json")
    args = ap.parse_args()

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    base_vocab = args.vocab_size - N_SPECIAL
    if base_vocab < 256:
        raise SystemExit("vocab_size too small (need >= 256 + 12 for byte-level)")

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=base_vocab, show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes -> no OOV
    )

    if args.dataset:
        src = iter_hf(args.dataset, args.split, args.column, args.max_docs)
        print(f"training BPE from HF dataset {args.dataset} (col={args.column})")
    elif args.src:
        src = iter_local(args.src, args.max_docs)
        print(f"training BPE from local folder {args.src}")
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        src = iter_local(here, args.max_docs)
        print(f"[smoke] no --dataset/--src given; training on FERN-2 source at {here}")

    tok.train_from_iterator(src, trainer=trainer)
    got = tok.get_vocab_size()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tok.save(args.out)

    total = got + N_SPECIAL
    print(f"\nsaved -> {args.out}")
    print(f"base pieces       : {got}")
    print(f"total vocab_size  : {total}  (base + {N_SPECIAL} specials)")
    print("\nUse it like:")
    print(f'    cfg = FERNConfig.bpe(r"{args.out}", vocab_size={total}, preset="small")')


if __name__ == "__main__":
    main()
