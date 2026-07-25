"""Fast end-to-end sanity check for FERN-2: every subsystem runs, grads flow,
and the FERN-2 additions (vectorized MoE, BPE tokenizer, block diffusion) are
exercised. CPU-only, a few seconds.

    python smoke_test.py
"""

import math
import os
import tempfile

import torch
import torch.nn.functional as F

from fern import (
    FERN, FERNConfig, ByteTokenizer, make_tokenizer,
    assign_precision_by_usage, model_memory_bytes, evolve_experts,
)
from fern.sparse_moe import CognitiveMoE
from fern.diffusion import diffusion_loss, generate_diffusion


def check_moe_vectorization():
    """Vectorized batched MoE must match a per-token reference (no drops)."""
    cfg = FERNConfig(d_model=64, moe_capacity_factor=100.0)
    moe = CognitiveMoE(cfg).eval()
    x = torch.randn(2, 16, 64)
    out, aux = moe(x)
    flat = x.reshape(-1, 64); E = len(moe.experts)
    logits = moe.router(flat) + moe.region_bias[:E]
    probs = F.softmax(logits, -1)
    topv, topi = torch.topk(probs, moe.top_k, -1)
    topv = topv / (topv.sum(-1, keepdim=True) + 1e-9)
    ref = torch.zeros_like(flat)
    for n in range(flat.shape[0]):
        for j in range(moe.top_k):
            e = int(topi[n, j])
            ref[n] += topv[n, j] * moe.experts[e](flat[n:n + 1]).squeeze(0)
    diff = (out - ref.view(2, 16, 64)).abs().max().item()
    print(f"  vectorized-MoE max |diff| vs reference: {diff:.2e}")
    assert diff < 1e-4, "MoE vectorization parity FAILED"


def check_reversible_parity():
    """FERN-2.5: the reversible (recompute-on-backward) core must produce the
    same loss AND the same gradients as the plain core on a fixed seed — the
    memory win must be numerically free. Item 12/step-2 of the OPTIMIZED spec."""
    def run(reversible):
        torch.manual_seed(0)
        cfg = FERNConfig(d_model=64, n_heads=4, reasoning_mode="ponder",
                         max_fractal_depth=4, reversible=reversible)
        torch.manual_seed(1)
        model = FERN(cfg)
        torch.manual_seed(2)
        x = torch.randint(0, cfg.n_base_tokens, (2, 24))
        out = model(x, targets=x)
        out["loss"].backward()
        g = torch.cat([p.grad.reshape(-1) for p in model.parameters()
                       if p.grad is not None])
        return out["loss"].item(), g

    l0, g0 = run(False)
    l1, g1 = run(True)
    dloss = abs(l0 - l1)
    dgrad = (g0 - g1).abs().max().item()
    print(f"  reversible parity: dloss={dloss:.2e} max|dgrad|={dgrad:.2e}")
    assert dloss < 1e-4 and dgrad < 1e-3, "reversible parity FAILED"


def check_block_causal_mask():
    """Block-causal mask: within-block bidirectional, across-block causal."""
    cfg = FERNConfig(d_model=64, diff_block_size=4)
    attn = FERN(cfg).core.block.attn
    T = 12
    gidx = torch.zeros(1, cfg.event_global_topk, dtype=torch.long)  # anchors at pos 0
    m = attn._seq_mask(T, gidx, torch.device("cpu"), block_size=4)[0]  # [T,T]
    # token 1 (block 0) may attend token 2 (same block, FUTURE) -> bidirectional
    assert m[1, 2].item(), "within-block bidirectional attention missing"
    # token 1 (block 0) may NOT attend token 4 (block 1, future block)
    assert not m[1, 4].item(), "block-causal violated (attends future block)"
    # token 8 (block 2) may attend token 0 (earlier block, anchor)
    assert m[8, 0].item(), "cross-block causal attention missing"
    print("  block-causal mask: within-block bidir + across-block causal OK")


def check_bpe_tokenizer():
    if not _have_tokenizers():
        print("  [skip] `tokenizers` not installed — BPE check skipped")
        return
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    corpus = [
        "def add(a, b):\n    return a + b\n",
        "for i in range(10):\n    print(i)\n",
        "class Foo:\n    def bar(self):\n        return 42\n",
    ] * 50
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=500, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer=trainer)
    path = os.path.join(tempfile.gettempdir(), "fern2_smoke_tok.json")
    tok.save(path)
    total = tok.get_vocab_size() + FERNConfig.N_SPECIAL
    cfg = FERNConfig.bpe(path, vocab_size=total)
    bpe = make_tokenizer(cfg)
    s = "def add(a, b):\n    return a + b\n"
    ids = bpe.encode(s, add_bos=True, add_eos=True)
    assert ids[0] == cfg.bos_id and ids[-1] == cfg.eos_id
    assert bpe.decode(ids) == s, "BPE round-trip FAILED"
    print(f"  BPE round-trip OK (vocab {cfg.vocab_size}, specials at top, "
          f"{len(s)} bytes -> {len(ids)} ids)")


def _have_tokenizers():
    try:
        import tokenizers  # noqa: F401
        return True
    except Exception:
        return False


def main():
    torch.manual_seed(0)

    print("== FERN-2 additions ==")
    check_moe_vectorization()
    check_block_causal_mask()
    check_bpe_tokenizer()

    print("== FERN-2.5 eco additions ==")
    check_reversible_parity()

    cfg = FERNConfig(d_model=128, max_seq_len=64, max_fractal_depth=3,
                     n_heads=4, diff_block_size=8, reasoning_mode="ponder")
    model = FERN(cfg)
    tok = ByteTokenizer(cfg)

    print("\n== param report ==")
    for k, v in model.param_report().items():
        print(f"  {k}: {v}")

    B, T = 2, 32
    x = torch.randint(0, cfg.n_base_tokens, (B, T))
    y = torch.randint(0, cfg.n_base_tokens, (B, T))

    # ---- AR forward/backward ----
    out = model(x, targets=y, write_memory=True)
    out["loss"].backward()
    grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in model.parameters())
    print("\n== AR forward/backward ==")
    print("  loss:", round(out["loss"].item(), 4),
          "| ce:", round(out["ce"].item(), 4),
          "| ponder:", round(out["ponder"].item(), 4),
          "| world_model:", round(out["world_model"].item(), 4))
    print("  avg fractal depth:", round(out["info"]["avg_steps"], 3))
    print("  gradients flow:", grad_ok)
    assert grad_ok and math.isfinite(out["loss"].item())

    # ---- Block-diffusion forward/backward ----
    model.zero_grad()
    o = diffusion_loss(model, x, cfg)
    o["loss"].backward()
    dgrad_ok = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    print("\n== block-diffusion forward/backward ==")
    print("  loss:", round(o["loss"].item(), 4),
          "| ce:", round(o["ce"].item(), 4),
          "| mask_rate:", round(o["mask_rate"].item(), 3))
    print("  gradients flow:", dgrad_ok)
    assert dgrad_ok and math.isfinite(o["loss"].item()), "diffusion grads bad"

    # ---- recurrent latent carry-over (AR) ----
    out2 = model(x, latent=out["latent"].detach(), targets=y)
    print("\n  second-segment AR loss:", round(out2["loss"].item(), 4))

    # ---- world-model simulation ----
    traj = model.simulate(out["latent"].detach(), steps=3)
    print("  latent trajectory shape:", tuple(traj.shape))

    # ---- maintenance ----
    print("\n== maintenance ==")
    prec = assign_precision_by_usage(model)
    print("  assigned precisions:", sorted(set(prec.values()), reverse=True))
    print("  footprint:", {k: round(v, 3) if isinstance(v, float) else v
                           for k, v in model_memory_bytes(model).items()})
    print("  evolve:", evolve_experts(model))

    # ---- generation: AR + block diffusion ----
    print("\n== generation ==")
    prompt = torch.tensor([tok.encode("the")], dtype=torch.long)

    cfg.gen_mode = "ar"
    gen = model.generate(prompt, max_new_tokens=20)
    print("  AR generated ids:", tuple(gen.shape),
          "->", repr(tok.decode(gen[0].tolist()).encode("ascii", "backslashreplace").decode()))

    cfg.gen_mode = "diffusion"
    gen_d = model.generate(prompt, max_new_tokens=24, steps=4)
    assert gen_d.shape[1] > prompt.shape[1], "diffusion produced no tokens"
    assert torch.isfinite(gen_d.float()).all()
    print("  diffusion generated ids:", tuple(gen_d.shape),
          "->", repr(tok.decode(gen_d[0].tolist()).encode("ascii", "backslashreplace").decode()))

    print("\nALL SUBSYSTEMS OK")


if __name__ == "__main__":
    main()
