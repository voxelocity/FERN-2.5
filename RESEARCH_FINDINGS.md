# FERN-1 — Research Findings

Empirical log from building and training **FERN-1** (Fractal Event-driven Retrieval
Network), a from-scratch, byte-level, sparse, brain-inspired LLM. This documents what
was actually built, what worked, what broke, the bugs found, the measured results, and
the conclusions — written honestly, to inform FERN-2.

See `README.md` for the architecture, `ROADMAP.md` for FERN-2.

---

## 1. Setup

- **Model:** `small` preset — **~108M total params / ~11M active per token** (top-2 of
  48 experts fire), **~90% sparse**. d_model=512, 6 cognitive regions × 8 experts,
  expert_hidden_mult=4. Byte-level, vocab 267.
- **Reasoning:** `ponder` mode, `max_fractal_depth=2`, test-time-memory off (`--no_ttm`),
  neuromodulator on.
- **Data:** ~1 GB of permissive Python (`codeparrot/codeparrot-clean`), tokenized to a
  memmapped uint16 `.bin`.
- **Pretraining:** 150k steps, block 512, batch 24, bf16 (`--amp`), LR 2e-4 + warmup,
  ~50k tokens/sec on an RTX 5070 (12 GB). ≈ 1.8 epochs over the corpus.
- **SFT:** instruction-tuned on `ise-uiuc/Magicoder-Evol-Instruct-110K` with a chat
  template and assistant-only loss masking.
- **Hardware:** RTX 5070 (12 GB), 64 GB RAM, i9-10900K. All data/checkpoints/cache on
  drive E:.

---

## 2. Quantitative results

| Metric | Value | Note |
|--------|-------|------|
| Total params | ~108M | 107,589,754 |
| Active params / token | ~11M | top-2 of 48 experts |
| Sparsity | ~90% | active/total |
| Final pretrain perplexity | **~1.7–2.0** | ce ≈ 0.52–0.69 nats (~0.75–1.0 bits/byte) |
| Teacher-forced argmax accuracy | **0.78** | next-byte, in-distribution code |
| Throughput | ~50k tok/s | small model, bf16, batch 24 |
| GPU utilization | ~59% | dispatch-bound (see §4) |
| VRAM used | ~6 / 12 GB | model is small; lots of headroom |
| Time to train 150k | ~10 h | real (NaN-free) speed |

**Capability summary:** correct on memorized-common tasks (e.g. "add two numbers"),
incoherent/incorrect on novel composition (e.g. "reverse a string"). Learns *answer
structure* (conversational wrapper, Markdown code blocks, `def`, keywords) reliably;
cannot reliably *compose correct novel logic*.

---

## 3. Bugs found & fixed (the most valuable findings)

### 3.1 Ponder-loss NaN at low depth (silent, catastrophic)
The PonderNet geometric prior used `pg = 1/max(1, N/2)`. At `max_fractal_depth ≤ 2`,
`pg = 1.0`, which puts **zero** probability mass on a halting step → `log(0) = -inf` →
**infinite loss → NaN weights from ~step 1**.
- **Impact:** an entire 36k-step run trained on NaN and was lost (the single overwritten
  checkpoint was corrupted; 38/385 tensors NaN). **tok/s and ETA print normally under
  NaN** — only `ppl`/`ce` reveal it.
- **Fix:** `pg = min(0.9, 1/max(1, N/2))` and `prior.clamp_min(1e-6)` before `.log()`.
- **Lesson:** watch `ppl`, not throughput. Verify loss finite at all depths.

### 3.2 Dynamic expert grow/prune corrupts the optimizer
The "living model" maintenance (precision reassignment + expert growth/prune) swapped
parameter objects out from under AdamW, leaving stale optimizer state and un-optimized
new params. Over many cycles this destabilized training.
- **Fix:** maintenance **default OFF** during training (`--maintain_every 0`).
- **FERN-2:** function-preserving growth + optimizer-state migration before re-enabling.

### 3.3 `generate()` train/inference mismatch (the big one)
**Symptom:** after a healthy 150k-step run (ppl 1.7), generation produced pure garbage
(`o o o o o`). **Diagnosis** (see `diag2.py`):
- Teacher-forced argmax accuracy = **0.78** → the model *had* learned.
- Greedy decode with a **fresh latent + `infer=False`** (training-matched) →
  **coherent Python**: `def add(a, b):\n    """Adds..."""`.
- The shipped `generate()` instead **carried the evolving latent across steps**, used
  `infer=True` (ponder early-halt), and wrote hierarchical memory — all states the model
  **never saw in training** (training always uses a fresh latent, `infer=False`, no mem
  write). The OOD state collapsed generation.
- **Fix:** `generate()` now uses `latent=None` each step, `infer=False`,
  `write_memory=False`, `use_cache=False`.
- **Lesson (critical for FERN-2):** **generation must replicate training's statefulness
  exactly.** A model can have excellent teacher-forced ppl and still emit garbage if the
  decoder feeds it an unfamiliar internal state. Always diagnose with the
  teacher-forced-accuracy vs fresh-latent-greedy vs shipped-decoder comparison.

---

## 4. Performance / efficiency findings

- **The MoE is dispatch-bound.** The Python `for`-loop over 48 experts fires many tiny
  GPU kernels with CPU gaps between them. GPU utilization capped at **~59%** even after
  enlarging the batch; **CPU at ~8%, VRAM ~6/12 GB** — i.e. nothing is saturated, the
  GPU is *starved*, not busy. → **Vectorizing the MoE is the #1 perf fix.**
- **Bigger batch helps** (raised util 37%→59%, ETA fell) by amortizing dispatch — but
  only partially; the real fix is removing the loop.
- **Depth dominates step time.** Cutting `max_fractal_depth` 6→2 gave a ~7× speedup
  (CPU bench: 1.0→0.137 s/step). The test-time-memory sequential scan added ~1.4×.
- **A NaN run looks fast.** An early "6.6 h ETA" was a mirage — NaN weights
  short-circuited real expert computation. Real NaN-free speed was higher (~10 h). Fast
  + cheap can mean *broken*.
- **Expert offloading works but is situational.** CPU-RAM offload via
  `torch.func.functional_call` keeps gradients flowing to the CPU master weights
  (training-correct), but per-step CPU↔GPU streaming made the *small* model **much
  slower** (one bad run hit ~26 s/step). Offload is for big-total models that don't fit
  VRAM, not for `small`.
- **Checkpoint hygiene matters.** Overwriting a single file lost everything to one NaN
  blowup. Now: finite-guarded saves + immutable step-tagged backups + a NaN-guard that
  skips non-finite steps.

---

## 5. Capability findings

- **The model genuinely learned code** (0.78 teacher-forced next-byte accuracy; coherent
  greedy continuations of real code).
- **Memorization vs composition:** it reproduces ultra-common snippets ("add two
  numbers") correctly, but **cannot compose correct novel logic** — output for less
  common tasks is structurally valid but semantically wrong/incoherent.
- **Byte-level drift is the main incoherence source.** Predicting one fragile byte at a
  time lets small errors compound; words dissolve into nonsense mid-sentence. A
  **subword/BPE tokenizer would directly attack this** (each step = a whole word-piece).
- **Format learned, stopping not.** SFT taught the conversational wrapper + Markdown code
  blocks well, but the model **doesn't reliably emit `<end>`**, so it rambles past the
  answer. More SFT (or output trimming) addresses this; it's cosmetic.
- **~1.8 epochs on 1 GB → near the memorization edge.** Very low ppl partly reflects
  fitting the small corpus, not broad generalization. More/varied data would help.

---

## 6. Conclusions

1. **FERN-1 is a successful proof-of-concept.** A novel 12-principle sparse architecture
   + v2 extensions was built, trained from scratch on a single consumer GPU, and — after
   fixing the decoder — produces structured, conversational, occasionally-correct Python.
   The full pipeline (pretrain → SFT → chat) works end to end.
2. **It is not a usable coding assistant**, and that was the *expected* ceiling for 108M /
   byte-level / ~1 GB. The honest result is "memorized snippets + structural mimicry +
   byte-level drift," not reliable code generation.
3. **The three bugs were the real research value:** the ponder-NaN, the optimizer-
   corrupting growth, and especially the **decoder-must-match-training** finding —
   generalizable lessons, not FERN-specific quirks.
4. **The ceiling is composition, and the levers are known**, in order of impact:
   **(1) BPE/code-aware tokenizer** (coherence — kills byte drift), (2) more/cleaner
   data (generalization), (3) bigger model via rented compute (capacity), (4) RL on
   execution feedback (correctness). More SFT on *this* model does not fix composition.

---

## 7. Implications for FERN-2 (see ROADMAP.md / E:\FERN\BUILD_FERN.md)

- **BPE tokenizer is now the #1 build** — it's the single change most likely to convert
  "memorized snippets + drift" into "coherent small-scale code."
- **Vectorize the MoE** before renting any cloud GPU — a faster card would idle even
  harder on the Python expert loop (~20–30% util on an H100). Efficiency first, then rent.
- **Bake in all safety guards from the start:** finite loss at all depths, NaN-guard,
  immutable backups, maintenance off, and a decoder that mirrors training statefulness.
- **The architecture is validated as trainable and sparse-efficient.** The open question
  FERN-2 must answer with measurement: does it *beat a same-size vanilla transformer on
  code*? Build the eval harness first and find out.
