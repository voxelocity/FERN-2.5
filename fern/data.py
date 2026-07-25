"""Fill-in-the-Middle (FIM) data transform — code-editing without diffusion.

Autoregressive models only learn to *continue* text. But code is *edited*: you
need to fill a hole given what's before AND after it. FIM (Bavarian et al. 2022;
used by StarCoder / DeepSeek-Coder) teaches infilling cheaply: split a document
into prefix / middle / suffix and reorder it so the model learns to predict the
middle from both sides:

    [PRE] prefix [SUF] suffix [MID] middle

Train with a mix of normal left-to-right and FIM examples and the model gains
the single most useful coding skill — completing inside existing code — while
staying a plain next-token predictor. This captures most of diffusion's
code-editing advantage at a fraction of the training risk.
"""

import os
import random
import torch
from .config import FERNConfig

CODE_EXTS = (".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".h", ".cpp", ".cc",
             ".hpp", ".rs", ".go", ".java", ".cs", ".rb", ".lua", ".sh",
             ".html", ".css", ".json", ".md", ".txt", ".sql", ".toml", ".yaml")


def load_bin(path: str):
    """Memory-map a uint16 .bin produced by prepare_data.py (near-zero RAM).
    Returned object supports len() and slicing, so make_batch uses it directly."""
    import numpy as np
    return np.memmap(path, dtype=np.uint16, mode="r")


def load_code_dir(path: str, config: FERNConfig, exts=CODE_EXTS,
                  max_bytes: int | None = None) -> torch.Tensor:
    """Walk a directory, concatenate source files (EOS-separated) into one
    byte-level token tensor ready for `make_batch`. Skips binaries/huge files."""
    from .tokenizer import make_tokenizer
    tok = make_tokenizer(config)
    ids: list[int] = []
    for root, _, files in os.walk(path):
        if any(p in root for p in (".git", "node_modules", "__pycache__", ".venv")):
            continue
        for fn in files:
            if not fn.endswith(exts):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except (OSError, ValueError):
                continue
            ids.extend(tok.encode(text, add_bos=True, add_eos=True))
            if max_bytes and len(ids) >= max_bytes:
                break
        if max_bytes and len(ids) >= max_bytes:
            break
    if not ids:
        raise RuntimeError(f"no source files found under {path!r}")
    return torch.tensor(ids, dtype=torch.long)


def fim_transform(ids, config: FERNConfig, rng: random.Random):
    """ids: list[int] (one document) -> possibly FIM-reordered list[int]."""
    if rng.random() > config.fim_rate or len(ids) < 4:
        return ids
    a, b = sorted(rng.sample(range(1, len(ids)), 2))
    prefix, middle, suffix = ids[:a], ids[a:b], ids[b:]
    return ([config.fim_prefix_id] + prefix +
            [config.fim_suffix_id] + suffix +
            [config.fim_middle_id] + middle)


# ---------------------------------------------------------------------------
# Chat / instruction-tuning support
# ---------------------------------------------------------------------------

# A tiny built-in chat set so you can run sft.py -> chat.py end-to-end with ZERO
# downloads, just to prove the loop works. Replace with a real dataset for quality.
TOY_CHAT = [
    [{"role": "user", "content": "Write a Python function to add two numbers."},
     {"role": "assistant", "content": "def add(a, b):\n    return a + b"}],
    [{"role": "user", "content": "How do I reverse a list in Python?"},
     {"role": "assistant", "content": "Use slicing: reversed_list = my_list[::-1]"}],
    [{"role": "user", "content": "Write a function that returns the square of n."},
     {"role": "assistant", "content": "def square(n):\n    return n * n"}],
]


def render_chat(messages, config, system: str | None = None,
                add_generation_prompt: bool = False):
    """Turn a list of {role, content} into (ids, trainable_mask).

    Layout:  [BOS] <sys> .. <end> <user> .. <end> <assistant> .. <end> ...
    trainable_mask is 1 only on assistant content + its closing <end>, so SFT
    learns to *answer* (and to stop) but isn't trained to echo the prompt.
    Set add_generation_prompt=True at inference to end on <assistant> and let
    the model generate the reply.
    """
    from .tokenizer import ByteTokenizer
    tok = ByteTokenizer(config)
    role_tok = {"system": config.sys_id, "user": config.user_id,
                "assistant": config.asst_id}
    ids = [config.bos_id]
    trainable = [0]

    def emit(role, content):
        ids.append(role_tok[role]); trainable.append(0)
        train = 1 if role == "assistant" else 0
        for cid in tok.encode(content, add_bos=False):
            ids.append(cid); trainable.append(train)
        ids.append(config.end_id); trainable.append(train)

    if system:
        emit("system", system)
    for m in messages:
        emit(m["role"], m["content"])
    if add_generation_prompt:
        ids.append(config.asst_id); trainable.append(0)
    return ids, trainable


def make_sft_batch(examples, block, batch, config, device, rng, system=None):
    """Sample `batch` conversations, render + mask, pad/truncate to `block`.

    Targets on non-assistant positions are set to pad_id so the loss (which
    ignores pad_id) is computed only on the assistant's tokens.
    """
    xs, ys = [], []
    for _ in range(batch):
        ex = rng.choice(examples)
        ids, train = render_chat(ex, config, system=system)
        ids, train = ids[:block + 1], train[:block + 1]
        if len(ids) < block + 1:
            pad = block + 1 - len(ids)
            ids += [config.pad_id] * pad
            train += [0] * pad
        seq = torch.tensor(ids, dtype=torch.long)
        x = seq[:-1]
        y = seq[1:].clone()
        mask = torch.tensor(train[1:], dtype=torch.bool)
        y[~mask] = config.pad_id            # ignored by the loss
        xs.append(x); ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def load_instruction_dataset(dataset, split, instr_col, output_col,
                             input_col=None, max_examples=None):
    """Load an alpaca-style instruction dataset from HF into chat message pairs."""
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split, streaming=True)
    out = []
    for ex in ds:
        instr = ex.get(instr_col, "")
        if input_col and ex.get(input_col):
            instr = f"{instr}\n\n{ex[input_col]}"
        ans = ex.get(output_col, "")
        if not instr or not ans:
            continue
        out.append([{"role": "user", "content": instr},
                    {"role": "assistant", "content": ans}])
        if max_examples and len(out) >= max_examples:
            break
    return out


def make_batch(data_tokens, block, batch, config, device, rng, fim=True):
    """Sample `batch` windows of length `block`, optionally FIM-transformed.

    `data_tokens` is a 1-D LongTensor of the full corpus. For FIM we slice a
    window, transform it, then pad/truncate back to `block` so shapes stay fixed.
    """
    xs, ys = [], []
    n = len(data_tokens)
    for _ in range(batch):
        i = rng.randint(0, n - block - 2)
        window = data_tokens[i:i + block + 1].tolist()
        if fim and config.fim_rate > 0:
            window = fim_transform(window, config, rng)
        window = window[:block + 1]
        if len(window) < block + 1:
            window += [config.pad_id] * (block + 1 - len(window))
        seq = torch.tensor(window, dtype=torch.long)
        xs.append(seq[:-1])
        ys.append(seq[1:])
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def make_diffusion_batch(data_tokens, block, batch, config, device, rng):
    """Sample `batch` CLEAN windows of length `block` for block diffusion.

    No FIM (infilling is native to diffusion) and no shift — diffusion.corrupt()
    does the masking and the model predicts tokens in place. `block` should be a
    multiple of config.diff_block_size (train.py enforces this)."""
    n = len(data_tokens)
    xs = []
    for _ in range(batch):
        i = rng.randint(0, n - block - 1)
        window = data_tokens[i:i + block].tolist()
        if len(window) < block:
            window += [config.pad_id] * (block - len(window))
        xs.append(torch.tensor(window, dtype=torch.long))
    return torch.stack(xs).to(device)
