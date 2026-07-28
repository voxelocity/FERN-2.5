"""End-to-end training loop for FERN-2 — supports both generation paradigms.

  --gen_mode ar         : token-by-token autoregressive (FERN-1, kept for A/B).
  --gen_mode diffusion  : BD3-LM block diffusion (FERN-2 default substrate).

Also supports the code-aware BPE tokenizer (`--tokenizer tok.json`), scale
presets, expert offloading, mixed precision, gradient accumulation, and the
FERN-1 safety guards (NaN guard, finite-only saves, step-tagged immutable
backups, LR warmup). Maintenance (precision/grow-prune) stays OFF by default.

Examples
--------
AR sanity run (CPU ok):
    python train.py --steps 300

Block-diffusion run with the BPE tokenizer on your box:
    python train.py --preset small --amp --gen_mode diffusion \\
        --tokenizer E:\\fern\\tok_code.json --bin E:\\fern_data\\py.bin \\
        --block 512 --batch 24 --steps 150000 --save_every 2000 \\
        --out E:\\fern\\fern2_base.pt
"""

import argparse
import math
import random
import torch

from fern import (
    FERN, FERNConfig, make_tokenizer,
    assign_precision_by_usage, model_memory_bytes, evolve_experts,
    GaLoreAdamW, build_galore_param_groups,
)
from fern.data import make_batch, make_diffusion_batch, load_code_dir, load_bin
from fern.diffusion import diffusion_loss

TOY = (
    "def add(a, b):\n    return a + b\n\n"
    "def boil_point():\n    # water boils at 100 C at sea level\n    return 100\n\n"
    "the cat sat on the mat. newton's second law: force = mass * acceleration.\n"
) * 200


def _bpe_vocab_size(path: str) -> int:
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path).get_vocab_size() + FERNConfig.N_SPECIAL


def build_config(args) -> FERNConfig:
    overrides = dict(
        reasoning_mode=args.reasoning_mode,
        offload_experts=args.offload,
        max_seq_len=max(256, args.block),
        use_test_time_memory=not args.no_ttm,
        use_neuromodulator=not args.no_neuromod,
    )
    # only force these when explicitly passed, so a preset that sets them (eco =
    # reversible + block diffusion) isn't silently overridden back to the CLI
    # defaults.
    if args.reversible:
        overrides["reversible"] = True
    if args.gen_mode is not None:
        overrides["gen_mode"] = args.gen_mode
    if args.diff_block is not None:
        overrides["diff_block_size"] = args.diff_block
    if args.attn_window is not None:
        overrides["attn_local_window"] = args.attn_window
    if args.depth is not None:
        overrides["max_fractal_depth"] = args.depth
    if args.capacity_factor is not None:
        overrides["moe_capacity_factor"] = args.capacity_factor
    if args.load_balance is not None:
        overrides["load_balance_weight"] = args.load_balance
    if args.tokenizer:
        overrides.update(tokenizer_kind="bpe", tokenizer_path=args.tokenizer,
                         vocab_size=_bpe_vocab_size(args.tokenizer))
    if args.preset:
        return FERNConfig.preset(args.preset, **overrides)
    return FERNConfig(**overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=str, default=None,
                    help="memory-mapped .bin from prepare_data.py (recommended)")
    ap.add_argument("--data", type=str, default=None, help="single text file")
    ap.add_argument("--data_dir", type=str, default=None, help="directory of code")
    ap.add_argument("--preset", type=str, default=None,
                    choices=["eco", "small", "base", "large"])
    ap.add_argument("--tokenizer", type=str, default=None,
                    help="path to a BPE tokenizer.json (train_tokenizer.py); "
                         "omitted = byte-level")
    ap.add_argument("--gen_mode", type=str, default=None,
                    choices=["ar", "diffusion"],
                    help="override the paradigm; default follows the preset "
                         "(eco = diffusion, others = ar)")
    ap.add_argument("--diff_block", type=int, default=None,
                    help="block size for block diffusion (default: preset's)")
    ap.add_argument("--reasoning_mode", type=str, default="ponder",
                    choices=["equilibrium", "ponder"])
    ap.add_argument("--depth", type=int, default=None,
                    help="fractal reasoning steps; lower = much faster "
                         "(default: preset's; eco=4). NOTE: changes depth_emb's "
                         "shape, so pass the same value when resuming.")
    ap.add_argument("--no_ttm", action="store_true",
                    help="disable test-time-memory (forced off under diffusion)")
    ap.add_argument("--no_neuromod", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=300, help="LR warmup steps")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 mixed precision")
    ap.add_argument("--capacity_factor", type=float, default=None,
                    help="MoE expert capacity (default: preset's; eco=4.0). "
                         "Too low silently DROPS routed tokens; buffers scale "
                         "with batch*block, so lower it if VRAM is tight.")
    ap.add_argument("--load_balance", type=float, default=None,
                    help="load-balancing aux weight (default: preset's; "
                         "eco=0.05). Raise if experts stay imbalanced.")
    ap.add_argument("--attn_window", type=int, default=None,
                    help="local attention window = the REAL context bound "
                         "(default: preset's; eco=256). Attention is a dense "
                         "masked matmul, so raising --block past this buys "
                         "almost no context and costs O(T^2).")
    ap.add_argument("--reversible", action="store_true",
                    help="recompute reasoning-block activations on backward "
                         "(memory-free depth); on by default under --preset eco")
    ap.add_argument("--galore", action="store_true",
                    help="GaLore low-rank optimizer subspace (~halves optimizer "
                         "VRAM on the big 2D matrices)")
    ap.add_argument("--galore_rank", type=int, default=128)
    ap.add_argument("--galore_every", type=int, default=200,
                    help="steps between SVD refreshes of the projection")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--maintain_every", type=int, default=0,
                    help="0 = OFF (recommended). Destabilises long training.")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--keep_backups", type=int, default=3,
                    help="how many step-tagged backups to retain (0 = keep all). "
                         "Each holds model+optimizer (~11GB base / ~29GB large).")
    ap.add_argument("--out", type=str, default="fern2.pt")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    config = build_config(args)

    # block diffusion needs the window to be a whole number of blocks. Resolve
    # against the built config so preset-set diffusion (eco) is honoured even
    # when --gen_mode isn't passed.
    diffusion = config.gen_mode == "diffusion"
    if diffusion and args.block % config.diff_block_size != 0:
        args.block = (args.block // config.diff_block_size) * config.diff_block_size
        args.block = max(config.diff_block_size, args.block)
        config.max_seq_len = max(config.max_seq_len, args.block)
        print(f"[diffusion] rounded --block down to {args.block} "
              f"(multiple of diff_block={config.diff_block_size})")

    tok = make_tokenizer(config)

    if args.bin:
        data = load_bin(args.bin)                # memory-mapped, near-zero RAM
    elif args.data_dir:
        data = load_code_dir(args.data_dir, config)
    elif args.data:
        with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
            data = torch.tensor(tok.encode(f.read(), add_bos=False), dtype=torch.long)
    else:
        data = torch.tensor(tok.encode(TOY, add_bos=False), dtype=torch.long)
    print(f"corpus tokens     : {len(data):,}")

    model = FERN(config).to(args.device)
    if args.offload:
        model.enable_offload("cpu")     # AFTER .to(device)
    rng = random.Random(0)

    rep = model.param_report()
    vram = model.vram_estimate()
    print(f"total params      : {rep['total_params']:,}")
    print(f"active / token    : {rep['active_params_per_token']:,} "
          f"({rep['sparsity']*100:.1f}% sparse)")
    print(f"experts           : {rep['experts']} (top-{rep['active_experts_per_token']} fire)")
    print(f"tokenizer         : {config.tokenizer_kind} (vocab {config.vocab_size})")
    print(f"gen_mode={config.gen_mode}"
          + (f" (block {config.diff_block_size})" if diffusion else "")
          + f" | reasoning={config.reasoning_mode} offload={args.offload} "
          f"amp={args.amp} block={args.block}\n")

    if args.galore:
        groups, n_gl, n_pl = build_galore_param_groups(
            model, rank=args.galore_rank, update_gap=args.galore_every)
        opt = GaLoreAdamW(groups, lr=args.lr, rank=args.galore_rank,
                          update_gap=args.galore_every)
        print(f"optimizer         : GaLore r={args.galore_rank} "
              f"({n_gl} low-rank matrices / {n_pl} plain tensors)")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp_dtype = torch.bfloat16 if args.amp else None
    use_amp = args.amp and args.device == "cuda"

    start_step = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        sd = ck["model"]
        cur = model.state_dict()
        # Adapt an older checkpoint to the current architecture: keep every
        # tensor whose shape still matches, truncate depth_emb if the reasoning
        # depth shrank (the leading rows are the ones actually used), and drop
        # tensors for subsystems that no longer exist. Dropping only ever
        # removes modules measured to contribute exactly zero, so the learned
        # function is preserved -- but report it loudly rather than silently.
        keep, dropped, resized, missing = {}, [], [], []
        for k, v in sd.items():
            if k not in cur:
                dropped.append(k)
            elif cur[k].shape == v.shape:
                keep[k] = v
            elif (v.dim() == cur[k].dim()
                  and all(c <= o for c, o in zip(cur[k].shape, v.shape))):
                sl = tuple(slice(0, c) for c in cur[k].shape)
                keep[k] = v[sl].clone()
                resized.append(f"{k} {tuple(v.shape)}->{tuple(cur[k].shape)}")
            else:
                dropped.append(f"{k} {tuple(v.shape)} vs {tuple(cur[k].shape)}")
        missing = [k for k in cur if k not in keep]
        model.load_state_dict(keep, strict=False)
        exact = not dropped and not resized and not missing
        if "opt" in ck and exact:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0) + 1
        print(f"resumed from {args.resume} at step {start_step}")
        print(f"  loaded {len(keep)} tensors"
              + (f" | resized {len(resized)}" if resized else "")
              + (f" | dropped {len(dropped)}" if dropped else "")
              + (f" | freshly initialised {len(missing)}" if missing else ""))
        for r in resized:
            print(f"    resized: {r}")
        for d in dropped[:8]:
            print(f"    dropped: {d}")
        if not exact:
            print("  [resume] architecture changed -> optimizer state NOT "
                  "restored (Adam moments would not match). Expect a brief "
                  "loss bump while the moments rebuild.")

    import os, glob, re
    stem, ext = os.path.splitext(args.out)

    def save(step):
        finite = all(torch.isfinite(p).all() for p in model.parameters())
        if not finite:
            print("  [checkpoint] SKIPPED — model has NaN/Inf (not saving)")
            return False
        payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                   "config": config, "step": step}
        torch.save(payload, args.out)
        torch.save(payload, f"{stem}_step{step}{ext}")   # immutable backup
        # Rotate step-tagged backups so storage stays bounded. Each backup holds
        # model+optimizer (~11 GB for `base`, ~29 GB for `large`); without this,
        # saving every N steps for a long run fills the volume and crashes the run.
        if args.keep_backups > 0:
            backs = glob.glob(f"{stem}_step*{ext}")
            backs = sorted(backs, key=lambda p: int(re.search(r"_step(\d+)", p).group(1)))
            for old in backs[:-args.keep_backups]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        return True

    def train_step():
        """One forward/backward over grad_accum micro-batches. Returns
        (loss_acc, stats) where stats has ce/ppl/info/extra for logging."""
        loss_acc = 0.0
        stats = {}
        for _ in range(args.grad_accum):
            ctx = (torch.autocast(device_type="cuda", dtype=amp_dtype)
                   if use_amp else torch.enable_grad())
            with ctx:
                if diffusion:
                    x = make_diffusion_batch(data, args.block, args.batch,
                                             config, args.device, rng)
                    o = diffusion_loss(model, x, config)
                else:
                    x, y = make_batch(data, args.block, args.batch, config,
                                      args.device, rng)
                    o = model(x, targets=y)
                loss = o["loss"] / args.grad_accum
            loss.backward()
            loss_acc += loss.item()
            stats = o
        return loss_acc, stats

    import time
    tokens_per_step = args.batch * args.grad_accum * args.block
    base_lr = args.lr
    model.train()
    opt.zero_grad()
    t_last = time.time()
    nan_skips = 0
    for step in range(start_step, args.steps + 1):
        lr = base_lr * min(1.0, step / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr

        loss_acc, o = train_step()

        # NaN guard: throw away a bad batch's grads instead of poisoning weights
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(loss_acc) or not torch.isfinite(gnorm):
            opt.zero_grad()
            nan_skips += 1
            if nan_skips <= 5 or nan_skips % 50 == 0:
                print(f"  [nan-guard] skipped step {step} (total skips {nan_skips})")
            continue
        opt.step()
        opt.zero_grad()

        if step % 20 == 0 or step == 1:
            now = time.time()
            n = 1 if step == 1 else 20
            sec_per_step = (now - t_last) / n
            t_last = now
            toks = tokens_per_step / max(sec_per_step, 1e-6)
            eta_h = sec_per_step * (args.steps - step) / 3600
            i = o["info"]
            ce = o["ce"].item()
            if diffusion:
                extra = f"mask_rate {o['mask_rate'].item():.2f}"
            elif "residual" in i:
                extra = f"residual {i['residual']:.3f}"
            else:
                extra = f"ponder {o['ponder'].item():.3f}"
            print(f"step {step:5d} | loss {loss_acc:.3f} "
                  f"| ce {ce:.3f} ppl {math.exp(min(ce,20)):.1f} "
                  f"| {i['mode']} iters {i['avg_steps']:.1f} | {extra} "
                  f"| {toks:,.0f} tok/s | ETA {eta_h:.1f}h")

        if args.maintain_every and step % args.maintain_every == 0:
            prec = assign_precision_by_usage(model)
            ev = evolve_experts(model)
            mem = model_memory_bytes(model)
            print(f"  [maintain] precision={sorted(set(prec.values()), reverse=True)} "
                  f"grew={ev['grew']} pruned={ev['pruned']} experts={ev.get('experts','?')} "
                  f"| footprint {mem['total_mb']:.1f} MB")

        if step % args.save_every == 0:
            if save(step):
                print(f"  [checkpoint] saved -> {args.out} (+ backup) @ step {step}")

    save(args.steps)
    print(f"\nsaved -> {args.out}")

    prompt = torch.tensor([tok.encode("def ")], dtype=torch.long, device=args.device)
    gen = model.generate(prompt, max_new_tokens=60)
    sample = tok.decode(gen[0].tolist()).encode("ascii", "backslashreplace").decode()
    print("sample:", repr(sample))


if __name__ == "__main__":
    main()
