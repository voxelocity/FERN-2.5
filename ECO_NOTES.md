# FERN-2.5 "eco" — train-fast-on-a-5070 fork

Fork of FERN-2. Adds the *training-efficiency* methods from `BUILD_FERN_OPTIMIZED.md`
that actually help a 12GB Blackwell card, plus a tiny `eco` preset. The FERN-2
baseline (BPE tokenizer + vectorized MoE + block diffusion) is untouched — every
addition below is an opt-in toggle on the working baseline.

## What was added

| # (doc) | Method | Where | Flag | Win |
|---------|--------|-------|------|-----|
| new | `eco` scale preset | `fern/config.py` | `--preset eco` | **Block-diffusion** model by default (gen_mode="diffusion", 16-tok blocks). d=256, 24 experts → ~8M total (byte) / ~2.5M active. Fits 12GB with room to spare. |
| §6 | Reversible ReasoningBlock | `fern/fractal_core.py` | `--reversible` | Recomputes the reasoning block on the backward pass (gradient checkpointing) → peak activation VRAM stops scaling with recursion depth. **Bit-identical** to the plain block (parity check in `smoke_test.py`). On by default under `--preset eco`. |
| §5 | GaLore low-rank optimizer | `fern/galore.py` | `--galore --galore_rank N` | Adam moments for the big 2D matrices live in a rank-`N` subspace (periodic SVD of the gradient) → ~halves optimizer VRAM. |

Implementation note on §6: the spec suggests a hand-written RevNet inverse. This
fork uses gradient checkpointing instead — same "activations recomputed on
backward" memory profile, but numerically **exact** (no risk of a learned inverse
silently diverging), and it composes cleanly with the MoE aux loss and the DEQ /
ponder loops. It's a no-op inside the equilibrium solver's `no_grad` phase.

## Recommended 5070 recipe

The eco preset **is a block-diffusion model** — `--gen_mode` is not needed:

```bash
python train.py --preset eco --reversible --galore --amp \
    --bin E:\fern_data\py.bin \
    --block 1024 --batch 24 --grad_accum 1 \
    --reasoning_mode ponder --steps 100000 --save_every 2000 \
    --out E:\fern\fern25_eco.pt
```

- `--block` is auto-rounded down to a multiple of the 16-token diffusion block.
- `py.bin` on this box is **byte-level** (no BPE tokenizer exists), so omit `--tokenizer`. Under diffusion, infilling is native and FIM/TTM are skipped automatically.
- `--galore_rank` trades memory for fidelity smoothly; 64–128 is the sweet spot at this size. If ce lags full-Adam by more than a few %, raise the rank.
- `--reversible` keeps activation VRAM flat as you raise `--depth`.
- To A/B against autoregressive, add `--gen_mode ar` explicitly.

## Verified (RTX-agnostic, on this box)

- `python smoke_test.py` → `ALL SUBSYSTEMS OK`, incl. reversible parity: `dloss=0 max|dgrad|=0`.
- `--preset eco --reversible --galore` (block diffusion, default) trains on the toy
  corpus: ce **7.7 → 4.65** in 180 steps at high mask rate, grads flow, ~6k tok/s CPU.
  GaLore projects 69 low-rank matrices / 80 plain tensors at rank 64.

## Not done (out of eco scope, still available from the doc)

µP (§4), FP8 (§3 — Blackwell supports it, but torchao/TE on consumer Blackwell is
driver-fragile; land and measure separately), distillation SFT (§7), and the
research forks (§8–10). GaLore + reversible were chosen because they are the two
methods that directly cut the VRAM a 12GB card runs out of first, and both are
pure-PyTorch and verifiable.
