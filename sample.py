"""Sample code completions from a trained FERN checkpoint.

This is the right way to eyeball a BASE model (chat.py wraps input in chat
tokens the pretrained model has never seen). Give it a code prefix and it
continues, using whichever paradigm the checkpoint was trained with.

Examples
--------
    python sample.py --model E:\\fern\\fern25_bpe2.pt
    python sample.py --model E:\\fern\\fern25_bpe2.pt --prompt "def fib(n):"
    python sample.py --model E:\\fern\\fern25_bpe2.pt --steps 32 --temperature 0.9
    python sample.py --model E:\\fern\\fern25_bpe2.pt --infill_test --bin E:\\fern_data\\py_bpe.bin
"""

import argparse

import torch
import torch.nn.functional as F

from fern import FERN, make_tokenizer

DEFAULT_PROMPTS = [
    "def add(a, b):\n    return",
    "class Dog:\n    def __init__(self",
    "for i in range(10):\n",
    "x = [1, 2, 3]\nprint(",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a .pt checkpoint")
    ap.add_argument("--prompt", default=None,
                    help="single prompt; omit to run the built-in set")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--steps", type=int, default=None,
                    help="denoising steps per block (diffusion only; "
                         "default = the config's diff_steps)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--infill_test", action="store_true",
                    help="also measure teacher-forced infill accuracy on real "
                         "data — the metric that tracks training progress")
    ap.add_argument("--bin", default=None, help="corpus .bin for --infill_test")
    args = ap.parse_args()

    ck = torch.load(args.model, map_location=args.device, weights_only=False)
    cfg = ck["config"]
    model = FERN(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = make_tokenizer(cfg)

    print(f"checkpoint : {args.model}  (step {ck.get('step')})")
    print(f"mode       : {cfg.gen_mode} | tokenizer {cfg.tokenizer_kind} "
          f"(vocab {cfg.vocab_size}) | attn window {cfg.attn_local_window}\n")

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    for p in prompts:
        ids = torch.tensor([tok.encode(p, add_bos=True)], device=args.device)
        gen = model.generate(ids, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             steps=args.steps)
        txt = tok.decode(gen[0].tolist()).encode("ascii", "replace").decode()
        print("-" * 64)
        print(txt)
    print("-" * 64)

    if args.infill_test:
        if not args.bin:
            print("\n[infill_test] needs --bin pointing at the corpus")
            return
        import numpy as np
        data = np.memmap(args.bin, dtype=np.uint16, mode="r")
        T, B = 512, 8
        starts = np.random.RandomState(0).randint(0, len(data) - T, size=B)
        x = torch.tensor(np.stack([np.array(data[s:s + T], dtype=np.int64)
                                   for s in starts])).to(args.device)
        torch.manual_seed(0)
        mask = torch.rand(x.shape, device=x.device) < 0.5
        xn = torch.where(mask, torch.full_like(x, cfg.mask_id), x)
        with torch.no_grad():
            logits = model(xn, block_size=cfg.diff_block_size)["logits"]
        acc = (logits.argmax(-1)[mask] == x[mask]).float().mean().item()
        ce = F.cross_entropy(logits[mask], x[mask]).item()
        print(f"\n== teacher-forced infill (50% masked, real code) ==")
        print(f"  accuracy : {acc*100:.2f}%   (random {100/cfg.vocab_size:.4f}%)")
        print(f"  raw CE   : {ce:.3f}        (random {np.log(cfg.vocab_size):.3f})")
        print("  ^ this is the number that tracks training progress")


if __name__ == "__main__":
    main()
