"""Supervised fine-tuning (SFT) — turn a pretrained FERN into a coding chatbot.

Pretraining (train.py) teaches the model to model code. SFT teaches it to
*answer* using the chat template, with the loss masked to the assistant's tokens
only. After this you can talk to it via chat.py.

Large downloads (datasets + HF cache) are routed to drive E.

Quick end-to-end test with NO downloads (uses the built-in toy chat set):
    python sft.py --toy --steps 400 --out E:\fern\fern_chat.pt

Real run on a pretrained base, with a coding instruction dataset:
    python sft.py --resume E:\fern\fern_base.pt --offload --amp \
        --dataset iamtarun/python_code_instructions_18k_alpaca \
        --instr_col instruction --input_col input --output_col output \
        --steps 20000 --out E:\fern\fern_chat.pt
"""

import os
os.environ.setdefault("HF_HOME", r"E:\hf_cache")    # keep HF cache off C:

import argparse
import math
import random
import torch

from fern import FERN, FERNConfig, ByteTokenizer
from fern.data import make_sft_batch, load_instruction_dataset, TOY_CHAT

SYSTEM = "You are FERN, a helpful coding assistant. Answer with correct, concise code."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None, help="pretrained base checkpoint")
    ap.add_argument("--preset", type=str, default=None,
                    choices=["small", "base", "large"])
    ap.add_argument("--toy", action="store_true", help="use built-in toy chat set")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--instr_col", type=str, default="instruction")
    ap.add_argument("--input_col", type=str, default=None)
    ap.add_argument("--output_col", type=str, default="output")
    ap.add_argument("--max_examples", type=int, default=50000)
    ap.add_argument("--reasoning_mode", type=str, default="ponder",
                    choices=["equilibrium", "ponder"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--out", type=str, default=r"E:\fern\fern_chat.pt")
    args = ap.parse_args()

    # ---- config: inherit the base checkpoint's config if resuming ----
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = ck["config"]
        config.reasoning_mode = args.reasoning_mode
        config.offload_experts = args.offload
    else:
        config = (FERNConfig.preset(args.preset, reasoning_mode=args.reasoning_mode,
                                    offload_experts=args.offload)
                  if args.preset else FERNConfig(reasoning_mode=args.reasoning_mode,
                                                 offload_experts=args.offload))
        ck = None

    tok = ByteTokenizer(config)

    # ---- data ----
    if args.toy or not args.dataset:
        examples = TOY_CHAT
        print(f"using built-in toy chat set ({len(examples)} convos)")
    else:
        examples = load_instruction_dataset(
            args.dataset, "train", args.instr_col, args.output_col,
            input_col=args.input_col, max_examples=args.max_examples)
        print(f"loaded {len(examples)} conversations from {args.dataset}")

    # ---- model ----
    model = FERN(config).to(args.device)
    if ck is not None:
        model.load_state_dict(ck["model"])
        print(f"loaded base weights from {args.resume}")
    if args.offload:
        model.enable_offload("cpu")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_dtype = torch.bfloat16 if args.amp else None
    use_amp = args.amp and args.device == "cuda"
    rng = random.Random(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        for _ in range(args.grad_accum):
            x, y = make_sft_batch(examples, args.block, args.batch, config,
                                  args.device, rng, system=SYSTEM)
            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if use_amp else torch.enable_grad())
            with ctx:
                out = model(x, targets=y)
                loss = out["loss"] / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == 1:
            print(f"step {step:5d} | loss {out['loss'].item():.3f} "
                  f"| ce {out['ce'].item():.3f} ppl {math.exp(min(out['ce'].item(),20)):.1f}")
        if step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "config": config, "step": step}, args.out)
            print(f"  [checkpoint] -> {args.out}")

    torch.save({"model": model.state_dict(), "config": config, "step": args.steps}, args.out)
    print(f"\nsaved chat model -> {args.out}\nnow run:  python chat.py --model {args.out}")


if __name__ == "__main__":
    main()
