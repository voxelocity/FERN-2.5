"""(Phase -1) Cheap local proof BEFORE you rent an H100.

The dangerous FERN failures are silent-correctness ones — great training loss,
garbage output — and they're free to catch on the 5070. This script runs the
diagnostic battery that would have caught FERN-1's worst bug (ppl 1.7 yet
`generate()` emitted `o o o o`), plus a block-diffusion infilling check, and
prints a go/no-go verdict.

Two ways to run:
  * On a checkpoint you trained:
        python validate_local.py --model E:\\fern\\fern2_base.pt
  * Self-contained demo (briefly trains a tiny model on the toy corpus so the
    diagnostics have something to chew on):
        python validate_local.py --gen_mode ar --steps 600
        python validate_local.py --gen_mode diffusion --steps 600

The AR diagnostics (teacher-forced accuracy vs fresh-latent greedy vs the
shipped decoder) are the FERN-1 lesson made into a test. The diffusion
diagnostic masks a block in a known snippet and measures how well the denoiser
recovers it conditioned on the surrounding context.
"""

import argparse
import random

import torch

from fern import FERN, FERNConfig, make_tokenizer
from fern.config import backfill_config
from fern.data import make_batch, make_diffusion_batch
from fern.diffusion import diffusion_loss

HELD = "def add(a, b):\n    return a + b\n"
PREFIX = "def add(a, b):\n    "

TOY = (
    "def add(a, b):\n    return a + b\n\n"
    "def boil_point():\n    # water boils at 100 C at sea level\n    return 100\n\n"
    "the cat sat on the mat. newton's second law: force = mass * acceleration.\n"
) * 200


def brief_train(cfg, steps, device):
    model = FERN(cfg).to(device)
    tok = make_tokenizer(cfg)
    data = torch.tensor(tok.encode(TOY, add_bos=False), dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    rng = random.Random(0)
    diffusion = cfg.gen_mode == "diffusion"
    block = 64
    model.train()
    for step in range(1, steps + 1):
        opt.zero_grad()
        if diffusion:
            x = make_diffusion_batch(data, block, 16, cfg, device, rng)
            o = diffusion_loss(model, x, cfg)
        else:
            x, y = make_batch(data, block, 16, cfg, device, rng)
            o = model(x, targets=y)
        o["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            ce = o["ce"].item()
            print(f"  train step {step:4d} | ce {ce:.3f} | ppl {min(2.718**ce,1e6):.1f}")
    model.eval()
    return model, tok


@torch.no_grad()
def diagnose_ar(model, tok, cfg, device):
    print("\n== AR decoder diagnostic (the FERN-1 lesson) ==")
    ids = tok.encode(HELD, add_bos=True)
    x = torch.tensor([ids], device=device)
    out = model(x[:, :-1])
    pred = out["logits"].argmax(-1)[0]
    acc = (pred == x[0, 1:]).float().mean().item()
    print(f"  teacher-forced argmax accuracy : {acc:.3f}  "
          f"(high => the model learned)")

    def greedy(carry_latent, infer):
        cur = torch.tensor([tok.encode(PREFIX, add_bos=True)], device=device)
        latent = None
        for _ in range(40):
            o = model(cur[:, -cfg.max_seq_len:], latent=latent, infer=infer)
            if carry_latent:
                latent = o["latent"]
            nxt = o["logits"][:, -1].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], 1)
        return tok.decode(cur[0].tolist())

    fresh = greedy(carry_latent=False, infer=False)   # training-matched
    carried = greedy(carry_latent=True, infer=True)    # the FERN-1 bug pattern
    shipped = tok.decode(model.generate(
        torch.tensor([tok.encode(PREFIX, add_bos=True)], device=device),
        max_new_tokens=40, temperature=1e-4, top_k=1)[0].tolist())

    def show(label, s):
        print(f"  {label}: {repr(s.encode('ascii','backslashreplace').decode())[:90]}")
    show("fresh-latent greedy (train-matched)", fresh)
    show("carried-latent infer=True (OOD)    ", carried)
    show("shipped model.generate()           ", shipped)

    coherent = sum(ch.isalpha() or ch in " ()_:\n+=" for ch in shipped) / max(len(shipped), 1)
    verdict = []
    if acc >= 0.5:
        verdict.append("model learned (teacher-forced acc OK)")
    else:
        verdict.append("LOW teacher-forced acc - train longer / more data")
    # decoder-mismatch detector: train-matched path coherent but shipped not
    if acc >= 0.5 and ("return" in fresh or "+" in fresh) and "return" not in shipped \
            and "+" not in shipped:
        verdict.append("WARNING: shipped decoder diverges from train-matched path "
                       "-> decoder/statefulness bug (the FERN-1 trap)")
    return acc, verdict


@torch.no_grad()
def diagnose_diffusion(model, tok, cfg, device):
    print("\n== block-diffusion infilling diagnostic ==")
    L = cfg.diff_block_size
    ids = tok.encode(HELD, add_bos=False)
    # trim to a whole number of blocks, keep at least 2 blocks
    n_blocks = max(2, len(ids) // L)
    ids = ids[:n_blocks * L]
    x = torch.tensor([ids], device=device)
    # mask one interior block and measure recovery against the original
    blk = n_blocks // 2
    lo, hi = blk * L, (blk + 1) * L
    noised = x.clone()
    noised[0, lo:hi] = cfg.mask_id
    out = model(noised, block_size=L)
    pred = out["logits"][0, lo:hi].argmax(-1)
    recov = (pred == x[0, lo:hi]).float().mean().item()
    print(f"  masked block recovery accuracy : {recov:.3f}  "
          f"(teacher-forced; high => denoiser learned local structure)")

    prompt = torch.tensor([tok.encode(PREFIX, add_bos=True)], device=device)
    gen = model.generate(prompt, max_new_tokens=32, steps=cfg.diff_steps,
                         temperature=0.7, top_k=20)
    sample = tok.decode(gen[0].tolist())
    print(f"  block-diffusion sample         : "
          f"{repr(sample.encode('ascii','backslashreplace').decode())[:90]}")
    verdict = ["denoiser recovering structure" if recov >= 0.4
               else "LOW recovery - train longer / more data"]
    return recov, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="checkpoint to diagnose")
    ap.add_argument("--gen_mode", default="ar", choices=["ar", "diffusion"])
    ap.add_argument("--steps", type=int, default=600,
                    help="brief training steps when no --model is given")
    ap.add_argument("--tokenizer", default=None, help="BPE tokenizer.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.model:
        ck = torch.load(args.model, map_location=args.device, weights_only=False)
        cfg = backfill_config(ck["config"])
        model = FERN(cfg).to(args.device)
        model.load_state_dict(ck["model"])
        model.eval()
        if getattr(cfg, "offload_experts", False):
            model.enable_offload("cpu")
        tok = make_tokenizer(cfg)
        print(f"loaded {args.model} (step {ck.get('step','?')}) "
              f"| gen_mode={cfg.gen_mode}")
    else:
        over = dict(max_fractal_depth=2, reasoning_mode="ponder",
                    use_test_time_memory=False, gen_mode=args.gen_mode,
                    max_seq_len=128, diff_block_size=16)
        if args.tokenizer:
            from tokenizers import Tokenizer
            vs = Tokenizer.from_file(args.tokenizer).get_vocab_size() + FERNConfig.N_SPECIAL
            over.update(tokenizer_kind="bpe", tokenizer_path=args.tokenizer, vocab_size=vs)
        cfg = FERNConfig(d_model=256, n_heads=4, **over)
        print(f"[demo] briefly training a tiny model ({args.gen_mode}, "
              f"{args.steps} steps) on the toy corpus...")
        model, tok = brief_train(cfg, args.steps, args.device)

    if cfg.gen_mode == "diffusion":
        score, verdict = diagnose_diffusion(model, tok, cfg, args.device)
    else:
        score, verdict = diagnose_ar(model, tok, cfg, args.device)

    print("\n== VERDICT ==")
    for v in verdict:
        print("  - " + v)
    print("\n(Run the eval_harness.py for the actual pass@k number. This script "
          "only checks the model learned and decodes the way it trained.)")


if __name__ == "__main__":
    main()
