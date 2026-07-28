"""Central configuration for the FERN architecture.

FERN = Fractal Event-driven Retrieval Network.

Every radical idea in the design is toggle-able and sized from here so you can
scale the model from a laptop-CPU toy up to something serious without touching
the module code. Defaults are deliberately tiny so the whole thing trains on a
CPU in seconds for smoke-testing.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FERNConfig:
    # ---- tokenizer / IO -------------------------------------------------
    # FERN-2 promotes the tokenizer to a code-aware BPE (ROADMAP: the #1 build —
    # byte-level wasted ~3-5x the steps to coherence). The 12 special tokens
    # always occupy the TOP of the vocab so the same id layout works for both
    # the byte fallback (base=256 -> vocab 268) and BPE (base=vocab_size-12).
    # The special ids below are DERIVED in __post_init__ from vocab_size; don't
    # set them by hand. Order is fixed in tokenizer.SPECIAL_ORDER.
    tokenizer_kind: str = "byte"      # "byte" | "bpe"
    tokenizer_path: str | None = None # path to a trained BPE tokenizer.json
    vocab_size: int = 268             # 256 bytes + 12 specials (byte fallback)
    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    unk_id: int = 259
    # FIM (fill-in-the-middle) special tokens. Under block diffusion FIM becomes
    # redundant (infilling is native), but the ids stay for back-compat / mixing.
    fim_prefix_id: int = 260
    fim_suffix_id: int = 261
    fim_middle_id: int = 262
    # Chat role tokens — baked in now so pretrain + SFT share one vocab.
    sys_id: int = 263
    user_id: int = 264
    asst_id: int = 265
    end_id: int = 266       # closes a turn; the assistant learns to emit it to stop
    mask_id: int = 267      # absorbing-state token for block diffusion
    max_seq_len: int = 512

    # ---- core dimensions ------------------------------------------------
    d_model: int = 256
    n_heads: int = 4

    # ---- (4) Fractal core: ONE block, applied recursively ---------------
    # Instead of N distinct layers we share weights and recurse.
    max_fractal_depth: int = 6     # max recursion steps
    ponder_epsilon: float = 0.05   # halting reserve
    inference_halt_threshold: float = 0.9  # early-stop cumulative prob at infer

    # ---- reasoning mode -------------------------------------------------
    # "ponder"      : PonderNet adaptive-depth recursion (original, robust)
    # "equilibrium" : Deep-Equilibrium latent reasoning — loop the SAME block
    #                 to a fixed point ("think in latent space until the thought
    #                 converges"). Unbounded effective depth at constant memory,
    #                 Jacobian-free backprop. Highest reasoning ceiling.
    reasoning_mode: str = "equilibrium"
    deq_max_iters: int = 20        # cap on fixed-point iterations
    deq_tol: float = 1e-3          # convergence tolerance on the residual
    deq_grad_steps: int = 2        # backprop through last N steps (JFB)
    deq_damping: float = 0.5       # Picard damping z<-(1-d)z+d*Block(z); aids convergence

    # ---- (NEW) test-time learning memory (DeltaNet fast-weights) --------
    # A memory whose weights UPDATE during inference via the delta rule — the
    # model literally learns the current context as it reads it (Titans/TTT/
    # DeltaNet family). For a coding model: adapts to the repo/style in-session.
    use_test_time_memory: bool = True
    ttm_heads: int = 4

    # ---- (NEW) neuromodulator -------------------------------------------
    # One global learned "brain-state" vector that jointly gates reasoning
    # depth, memory read/write, and routing temperature — a cheap controller
    # that puts the network into a fast/shallow or focused/deep cognitive mode.
    use_neuromodulator: bool = True
    neuromod_dim: int = 32

    # ---- (NEW) FIM training ---------------------------------------------
    fim_rate: float = 0.5          # fraction of training docs transformed to FIM

    # ---- (FERN-2) generation paradigm: AR or Block Diffusion ------------
    # "ar"        : token-by-token autoregressive (FERN-1 default; kept for A/B).
    # "diffusion" : BD3-LM-style block diffusion — autoregressive BETWEEN fixed
    #               blocks (KV-cache + arbitrary length), absorbing-state masked
    #               diffusion WITHIN a block (parallel decode + bidirectional
    #               local context -> native infilling/editing). See fern/diffusion.py.
    gen_mode: str = "ar"
    diff_block_size: int = 16      # tokens per diffusion block
    # Denoising steps per block at inference. Measured: at 8 steps the sampler
    # commits up to 3 tokens at once from near-identical marginals, which is
    # what makes generation repeat itself. Setting this to diff_block_size
    # commits one token per step and markedly improves coherence; it costs one
    # forward per token, so lower it if you want speed over quality.
    diff_steps: int = 16           # = diff_block_size -> one token per step
    # Clipped masking-rate schedule (BD3-LM): sampling the mask prob from a
    # CLIPPED range instead of full (0,1) sharply lowers gradient variance, the
    # key trick that makes from-scratch block diffusion trainable.
    diff_mask_min: float = 0.15
    diff_mask_max: float = 1.0

    # ---- (NEW) expert offloading ----------------------------------------
    # Keep ALL experts in CPU RAM; stream only the active top-k to the GPU per
    # forward. This is what makes "huge total / tiny active" trainable on a
    # small-VRAM card: total params scale with system RAM, VRAM holds only the
    # always-on core + the handful of active experts + activations.
    offload_experts: bool = False
    offload_device: str = "cpu"
    pin_expert_memory: bool = True   # faster async CPU->GPU copies

    # ---- (1)(6) Sparse MoE cognitive regions ----------------------------
    # Experts are grouped into named "cognitive regions". Routing is top-k
    # over experts, so only a tiny slice of the network fires per token.
    region_names: List[str] = field(default_factory=lambda: [
        "language", "math", "code", "spatial", "planning", "social",
    ])
    experts_per_region: int = 4
    moe_top_k: int = 2             # experts activated per token
    expert_hidden_mult: int = 2    # expert MLP hidden = mult * d_model
    moe_capacity_factor: float = 1.5
    load_balance_weight: float = 0.01
    # Vectorized MoE runs experts in chunks of this many at a time. Bounds the
    # transient weight-stack copy (≈10 GB at `large` if unchunked) without
    # changing the result. 64 keeps `small` (48 experts) a single pass while
    # chunking `base`/`large`. Raise for a touch more speed if VRAM allows.
    moe_stack_chunk: int = 64

    # ---- (1) Sparse attention -------------------------------------------
    attn_local_window: int = 64    # each token sees this many local neighbours
    attn_n_global: int = 8         # event-selected global tokens

    # ---- (7) Event-driven attention -------------------------------------
    # A cheap salience head decides which tokens deserve expensive compute.
    event_salience_dim: int = 32
    event_global_topk: int = 8     # most-salient tokens become global anchors

    # ---- (8) Predictive state compression -------------------------------
    # Context is summarised into a handful of latent "concept" tokens that
    # persist across segments (recurrent latent memory).
    n_latent_tokens: int = 16

    # ---- (11) Continuous-time latent dynamics ---------------------------
    use_continuous_time: bool = True
    ct_dt: float = 0.5             # Euler step size for the latent ODE

    # ---- (12) Hierarchical memory ---------------------------------------
    # Measured inert during pretraining: nothing writes to it (write_memory is
    # False in train.py and in generate()), so `memory.read()` returns exact
    # zeros every forward while still costing parameters, optimizer state, and
    # compute. Off for `eco`; turn on only if you actually write to it.
    use_hier_memory: bool = True
    mem_tiers: List[str] = field(default_factory=lambda: [
        "working", "short", "long", "archive",
    ])
    mem_tier_capacity: List[int] = field(default_factory=lambda: [
        32, 128, 512, 2048,
    ])
    mem_tier_decay: List[float] = field(default_factory=lambda: [
        0.0, 0.01, 0.001, 0.0001,
    ])
    mem_read_topk: int = 4

    # ---- (2) External knowledge store -----------------------------------
    knowledge_dim: int = 256       # dim of stored fact vectors
    knowledge_topk: int = 4
    use_knowledge: bool = True

    # ---- (9) Neural cache -----------------------------------------------
    use_neural_cache: bool = True
    cache_signature_bits: int = 12  # quantisation granularity of cache keys
    cache_max_entries: int = 100_000

    # ---- (3) Variable-precision neurons ---------------------------------
    # Bit-widths an expert can be assigned based on usage frequency.
    precision_buckets: List[int] = field(default_factory=lambda: [16, 8, 4])

    # ---- (10) World-model auxiliary objective ---------------------------
    # The model predicts its own next latent state (an internal simulation
    # signal), trained with a small auxiliary loss.
    world_model_weight: float = 0.1

    # ---- (FERN-2.5 eco) reversible reasoning block ----------------------
    # RevNet-style coupling so the fractal core's activations are RECOMPUTED on
    # the backward pass instead of stored. Peak activation VRAM then stops
    # scaling with max_fractal_depth / deq iterations — the whole point of an
    # "eco" model that iterates depth cheaply on a 12GB card. Numerically
    # equivalent to the plain block to ~1e-3 (verified in smoke_test). OFF by
    # default so the baseline path is untouched; turn on with --reversible.
    reversible: bool = False

    # ---- training defaults ----------------------------------------------
    dropout: float = 0.0
    tie_embeddings: bool = True

    # number of reserved special tokens, pinned to the TOP of the vocab
    N_SPECIAL: int = 12

    def __post_init__(self):
        """Place the 12 special tokens at the top of the vocab and derive their
        ids from vocab_size, so byte (base=256) and BPE (base=vocab_size-12)
        share one layout and the model code never hard-codes an id."""
        base = self.vocab_size - self.N_SPECIAL
        if base < 1:
            raise ValueError(f"vocab_size {self.vocab_size} too small for "
                             f"{self.N_SPECIAL} special tokens")
        (self.pad_id, self.bos_id, self.eos_id, self.unk_id,
         self.fim_prefix_id, self.fim_suffix_id, self.fim_middle_id,
         self.sys_id, self.user_id, self.asst_id, self.end_id,
         self.mask_id) = range(base, base + self.N_SPECIAL)

    @property
    def n_base_tokens(self) -> int:
        """Non-special tokens: byte values (256) or BPE pieces (vocab-12)."""
        return self.vocab_size - self.N_SPECIAL

    @property
    def n_experts(self) -> int:
        return len(self.region_names) * self.experts_per_region

    @property
    def expert_hidden(self) -> int:
        return self.d_model * self.expert_hidden_mult

    def region_of_expert(self, expert_idx: int) -> str:
        return self.region_names[expert_idx // self.experts_per_region]

    # ---- scale presets --------------------------------------------------
    @classmethod
    def preset(cls, name: str = "base", **overrides) -> "FERNConfig":
        """Sized configs (measured, not guessed). Total params scale with
        #experts (held in CPU RAM under offload); active params/token stay tiny
        because only top-k=2 experts fire — so these are *very* sparse, in the
        spirit of the original "huge brain, tiny activation" goal:

          eco   : d=256,  24 experts  -> ~15M total / ~2.5M active (eco, 5070)
          small : d=512,  48 experts  -> ~105M total / ~9M active  (8% active)
          base  : d=768,  192 experts -> ~915M total / ~19M active (2% active)
          large : d=1024, 288 experts -> ~2.4B total / ~33M active (1.4% active)

        small fits a 12GB GPU with no offload. base/large need `--offload`
        (experts in your 64GB RAM). Optimizer states (AdamW fp32) are ~2x the
        param bytes and also live in RAM under offload — large ≈ 2.4B*12B ≈ 29GB
        RAM, comfortable on 64GB.
        """
        cfgs = {
            # eco : the FERN-2.5 "train-fast-on-a-5070" size. A BLOCK-DIFFUSION
            #       model by default (gen_mode="diffusion", 16-token blocks) —
            #       parallel within-block decode + native infilling. ~1/5 of
            #       `small`, reversible-friendly, tuned to reach single-digit ce
            #       in minutes-to-an-hour on a 12GB Blackwell card.
            #       d=256, 24 experts -> ~8M total (byte) / ~15M (16k BPE),
            #       ~2.5M active/token (top-2 fire). No offload needed; pair with
            #       --reversible --galore --amp.
            # attn_local_window is the REAL context bound: the block-causal mask
            # gives each token its own block + `window` tokens back + 8 global
            # anchors, and attention is a dense masked matmul, so a large
            # --block with a 64-wide window computes a huge score matrix and
            # discards >90% of it. 256 makes the window the thing you actually
            # pay for; pair it with --block 512 (measured: ~4x cheaper attention
            # than --block 1024 while ~3.5x more context per token).
            # moe_capacity_factor=1.5 (the global default) measured 34% of expert
            # assignments DROPPED on a trained eco router — dropped tokens get no
            # expert edit at all, only the residual, so a third of the MoE's work
            # was discarded every step. Root cause is router imbalance (busiest
            # expert 21% vs 4.2% uniform), so raise the balancing pressure AND
            # give capacity headroom: 4.0 -> ~4% dropped. Buffers are
            # [E, capacity, D] and scale with batch*block, so lower the factor
            # (or the batch) if VRAM gets tight.
            "eco":   dict(d_model=256, n_heads=4, experts_per_region=4,
                          expert_hidden_mult=2, max_seq_len=1024,
                          max_fractal_depth=4, reversible=True,
                          gen_mode="diffusion", diff_block_size=16,
                          attn_local_window=256,
                          moe_capacity_factor=4.0, load_balance_weight=0.05,
                          # Measured dead during block-diffusion pretraining
                          # (zero gradient, exact-zero output) — they cost
                          # params, optimizer state and per-forward compute
                          # while contributing nothing. TTM is *already*
                          # skipped under diffusion; knowledge/hier-memory
                          # return zeros because nothing populates them.
                          use_test_time_memory=False, use_knowledge=False,
                          use_hier_memory=False),
            "small": dict(d_model=512, n_heads=8, experts_per_region=8,
                          expert_hidden_mult=4, max_seq_len=2048),
            "base":  dict(d_model=768, n_heads=12, experts_per_region=32,
                          expert_hidden_mult=4, max_seq_len=2048),
            "large": dict(d_model=1024, n_heads=16, experts_per_region=48,
                          expert_hidden_mult=4, max_seq_len=4096),
        }
        if name not in cfgs:
            raise ValueError(f"unknown preset {name!r}; choose {list(cfgs)}")
        params = {**cfgs[name], **overrides}
        return cls(**params)

    @classmethod
    def bpe(cls, tokenizer_path: str, vocab_size: int, preset: str | None = None,
            **overrides) -> "FERNConfig":
        """Build a config wired to a trained code-aware BPE tokenizer. The 12
        special tokens are auto-placed at the top of the given vocab_size."""
        kw = dict(tokenizer_kind="bpe", tokenizer_path=tokenizer_path,
                  vocab_size=vocab_size, **overrides)
        if preset:
            return cls.preset(preset, **kw)
        return cls(**kw)


def backfill_config(cfg: "FERNConfig") -> "FERNConfig":
    """Make a FERN-1 (pre-FERN-2) pickled config usable here.

    Unpickling restores a dataclass's __dict__ directly (no __init__/__post_init__),
    so configs saved before the FERN-2 fields existed are simply missing them.
    Add sensible defaults so old checkpoints (e.g. the FERN-1 chat model used as
    the baseline) load and evaluate without error. FERN-1 was always byte-level
    AR, so default to that."""
    defaults = dict(tokenizer_kind="byte", tokenizer_path=None,
                    gen_mode="ar", diff_block_size=16, diff_steps=8,
                    diff_mask_min=0.15, diff_mask_max=1.0, moe_stack_chunk=64)
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    if not hasattr(cfg, "N_SPECIAL"):
        # FERN-1 byte vocab = 256 base + its specials; keep n_base_tokens == 256
        cfg.N_SPECIAL = max(1, cfg.vocab_size - 256)
    if not hasattr(cfg, "mask_id"):
        cfg.mask_id = cfg.vocab_size - 1   # unused under gen_mode="ar"
    return cfg
