"""(FERN-2 Phase 1) Block diffusion — the generation paradigm.

BD3-LM-style: **autoregressive between fixed-size blocks** (so we keep arbitrary
length + cross-block KV-cache), **absorbing-state masked diffusion within a
block** (parallel decode + bidirectional local context -> native infilling and
editing, which makes the FERN-1 FIM hack redundant).

Two halves live here; the model backbone (fractal core, MoE, gates, memory) is
untouched — block diffusion only changes (a) the attention mask, which
`model.forward(block_size=...)` already handles, and (b) the training objective
and decode loop, which are this file.

Objective (absorbing-state / MDLM):
  * Corrupt each token of the clean sequence to `<mask>` independently with prob
    `t`, where `t` is sampled per example from a **clipped** range
    `[diff_mask_min, diff_mask_max]`. Clipping the rate is the BD3-LM trick that
    sharply lowers gradient variance — the single thing that makes from-scratch
    block diffusion trainable on consumer compute.
  * The model (block-causal attention) predicts the original token at every
    position; cross-entropy is taken ONLY on corrupted positions, weighted by
    `1/t` (the continuous-time ELBO weight; bounded because `t >= diff_mask_min`).

Sampling (MaskGIT-style, per block, left-to-right across blocks):
  * Start a block fully masked; over `diff_steps` iterations, predict all masked
    positions, reveal the highest-confidence ones on a cosine schedule, re-mask
    the rest, conditioned on already-clean previous blocks.
"""

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Training objective
# ---------------------------------------------------------------------------
def sample_mask_rate(batch: int, config, device, generator=None) -> torch.Tensor:
    """Per-example corruption rate t ~ U(diff_mask_min, diff_mask_max)  [B,1]."""
    lo, hi = config.diff_mask_min, config.diff_mask_max
    t = torch.rand(batch, 1, device=device, generator=generator)
    return lo + (hi - lo) * t


def corrupt(x_clean: torch.Tensor, rate: torch.Tensor, config, generator=None):
    """Absorbing-state corruption. Returns (x_noised, mask_bool). pad positions
    are never corrupted (they carry no signal and are excluded from the loss)."""
    B, T = x_clean.shape
    rand = torch.rand(B, T, device=x_clean.device, generator=generator)
    valid = x_clean != config.pad_id
    mask = (rand < rate) & valid                       # [B,T] True = corrupted
    x_noised = torch.where(mask, torch.full_like(x_clean, config.mask_id), x_clean)
    return x_noised, mask


def diffusion_loss(model, x_clean: torch.Tensor, config, generator=None) -> dict:
    """Block-diffusion training loss for one batch of clean sequences [B,T].

    Mirrors model.forward's AR loss bookkeeping (folds in MoE load-balance aux,
    ponder loss, and the world-model auxiliary) so train.py treats both
    paradigms uniformly."""
    B, T = x_clean.shape
    L = config.diff_block_size

    rate = sample_mask_rate(B, config, x_clean.device, generator)   # [B,1]
    x_noised, mask = corrupt(x_clean, rate, config, generator)

    out = model(x_noised, block_size=L)                 # block-causal, no targets
    logits = out["logits"]                              # [B,T,V]
    V = logits.size(-1)

    ce_tok = F.cross_entropy(
        logits.reshape(-1, V), x_clean.reshape(-1),
        reduction="none").view(B, T)                    # [B,T]

    # MDLM weight 1/t on corrupted positions, normalised by #corrupted tokens
    w = (1.0 / rate).expand(B, T)                       # bounded by 1/diff_mask_min
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    diff_ce = (ce_tok * m * w).sum() / denom

    # (10) world-model auxiliary (same as AR path), on the clean hidden states
    h = out["hidden"]
    if T >= 2:
        wm = F.smooth_l1_loss(model.world_model(h[:, :-1]), h[:, 1:].detach())
    else:
        wm = h.new_zeros(())

    loss = diff_ce + out["aux"] + out["ponder_loss"] + config.world_model_weight * wm
    return {
        "loss": loss,
        "ce": diff_ce.detach(),
        "aux": out["aux"].detach(),
        "ponder": out["ponder_loss"].detach(),
        "world_model": wm.detach(),
        "mask_rate": rate.mean().detach(),
        "info": out["info"],
    }


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def _denoise_block(model, ids: torch.Tensor, L: int, steps: int,
                   temperature: float, top_k: int, config) -> torch.Tensor:
    """Append one fully-masked block to `ids` and MaskGIT-denoise it in place,
    conditioned (block-causally) on the clean tokens already in `ids`.
    Returns the [B, L] denoised block."""
    B = ids.shape[0]
    device = ids.device
    keep_ctx = (config.max_seq_len // L) * L            # crop to a block multiple
    filled = torch.full((B, L), config.mask_id, dtype=ids.dtype, device=device)

    for s in range(steps):
        cur = torch.cat([ids, filled], dim=1)[:, -keep_ctx:]
        out = model(cur, block_size=L)
        blk_logits = out["logits"][:, -L:, :] / max(temperature, 1e-5)  # [B,L,V]
        if top_k:
            v, _ = torch.topk(blk_logits, min(top_k, blk_logits.size(-1)), dim=-1)
            blk_logits = blk_logits.masked_fill(blk_logits < v[..., [-1]], -1e9)
        probs = F.softmax(blk_logits, dim=-1)                           # [B,L,V]
        sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(B, L)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)      # [B,L]

        is_masked = filled == config.mask_id
        # already-revealed positions stay revealed (give them +inf confidence)
        conf_eff = torch.where(is_masked, conf, torch.full_like(conf, float("inf")))

        # cosine schedule: tokens still masked AFTER this step
        n_mask_next = int(math.floor(L * math.cos(math.pi / 2 * (s + 1) / steps)))
        n_reveal = L - n_mask_next                                      # total revealed
        if n_reveal >= L:
            keep_revealed = torch.ones_like(is_masked)
        else:
            thresh = conf_eff.topk(n_reveal, dim=-1).values[:, -1:]     # [B,1]
            keep_revealed = conf_eff >= thresh
        new_tok = torch.where(is_masked, sampled, filled)
        filled = torch.where(keep_revealed, new_tok,
                             torch.full_like(filled, config.mask_id))
    return filled


@torch.no_grad()
def generate_diffusion(model, prompt_ids: torch.Tensor, max_new_tokens=64,
                       temperature=1.0, top_k=40, steps=None, stop_ids=None):
    """Block-by-block diffusion decode. Generates ceil(max_new_tokens/L) blocks,
    stopping early if a stop token appears. Signature is AR-compatible so chat.py
    works unchanged."""
    model.eval()
    cfg = model.config
    L = cfg.diff_block_size
    steps = steps or cfg.diff_steps
    stop = set(stop_ids or [cfg.eos_id, cfg.end_id])
    device = prompt_ids.device

    ids = prompt_ids
    # pad the prompt up to a block boundary so generated blocks stay aligned;
    # the pad tokens are clean context and are stripped at decode time.
    pad_n = (-ids.shape[1]) % L
    if pad_n:
        ids = F.pad(ids, (0, pad_n), value=cfg.pad_id)

    n_blocks = math.ceil(max_new_tokens / L)
    for _ in range(n_blocks):
        block = _denoise_block(model, ids, L, steps, temperature, top_k, cfg)
        ids = torch.cat([ids, block], dim=1)
        if any(int(tok) in stop for tok in block[0].tolist()):
            break
    return ids
