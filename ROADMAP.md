# FERN‑2 Roadmap

---

## Background: what FERN‑1 is, and where everything lives

**FERN = Fractal Event‑driven Retrieval Network.** FERN‑1 is a from‑scratch,
ultra‑lightweight, brain‑inspired LLM (byte‑level, vocab 267). Its defining trait:
instead of a fixed stack of ~100 layers, it has **one reasoning block applied
recursively** (depth chosen per token), with most of the network dormant per token —
sparse Mixture‑of‑Experts "cognitive regions", sparse attention, an external knowledge
store (facts live outside the weights), a compressed recurrent latent state,
hierarchical memory, a neuromodulator, and a DeltaNet test‑time‑learning memory. It
implements all 12 of the original design principles; see `README.md` for the full
principle→module table. FERN‑2 builds on this exact codebase — keep the backbone, swap
the generation paradigm and bolt on the intelligence/training methods below.

### Repo layout (code — on the **C:** drive)
Project root: **`C:\Users\shark\Downloads\FERN`**

```
FERN\
  fern\                     the importable package (`from fern import ...`)
    config.py               FERNConfig dataclass + scale presets (small/base/large)
    tokenizer.py            ByteTokenizer (byte-level, vocab 267, FIM + chat tokens)
    model.py                FERN — wires every subsystem into one forward pass
    fractal_core.py         (4)(11) recursive block; ponder + equilibrium modes
    sparse_moe.py           (1)(6) top-k MoE cognitive regions + expert OFFLOAD path
    sparse_attention.py     (1) local-window + global-anchor sparse attention
    precision.py            (3) variable-precision (fake-quant) linear
    event_gate.py           (7) salience / event-driven gating
    state_compress.py       (8)(11) latent concept tokens + continuous-time update
    hierarchical_memory.py  (12) working/short/long/archive memory tiers
    knowledge_store.py      (2) external vector knowledge store + retriever
    neural_cache.py         (9) reasoning-trace cache
    neuromodulator.py       global "brain-state" gate (depth/memory/routing)
    test_time_memory.py     DeltaNet fast-weights that learn during inference
    maintenance.py          (3)(5) precision assignment + dynamic grow/prune ops
    data.py                 FIM transform, chat template (render_chat), make_batch,
                            make_sft_batch, load_bin (memmap), load_code_dir
  train.py                  pretraining loop (AR next-byte); --bin/--preset/--offload/--amp
  sft.py                    supervised fine-tune into a chat assistant (loss-masked)
  chat.py                   interactive chat REPL (talk to a checkpoint)
  prepare_data.py           stream a HF dataset / local folder -> tokenized uint16 .bin
  smoke_test.py             exercises every subsystem + checks gradients
  examples\load_knowledge.py   demo of the external knowledge store
  README.md   ROADMAP.md   requirements.txt
```

### Data, checkpoints, caches (large files — on the **E:** drive, `hotstorage_02`)
Everything big goes on **E:** (3.6 TB free). Keep this convention for FERN‑2.

```
E:\fern_data\        tokenized corpora (.bin) + raw datasets   e.g. E:\fern_data\py.bin
E:\fern\             model checkpoints + step-tagged backups   e.g. E:\fern\fern_base.pt
E:\hf_cache\         HuggingFace cache (HF_HOME is set to this in the scripts)
```

### Environment
- Python **3.13** at `C:\Users\shark\AppData\Local\Programs\Python\Python313`.
- **PyTorch CUDA build (cu128)** for the RTX 5070 (Blackwell) — *not* the CPU wheel.
- `datasets` (HF) installed for `prepare_data.py` / `sft.py`.
- Hardware: RTX 5070 (12 GB VRAM), 64 GB RAM, i9‑10900K, 256 GB headroom on E.

### The FERN‑1 pipeline (the loop FERN‑2 inherits)
```
prepare_data.py  →  E:\fern_data\*.bin          (tokenize corpus, memmap)
train.py         →  E:\fern\fern_base.pt        (pretrain on code, next-byte)
sft.py           →  E:\fern\fern_chat.pt        (instruction tune into a chatbot)
chat.py          →  talk to it
```

### Hard‑won settings & gotchas (carry into FERN‑2)
- **Default `reasoning_mode = ponder`** (robust). `equilibrium` is the high‑ceiling
  but harder‑to‑train mode — opt‑in only.
- **Maintenance (precision + grow/prune) is OFF by default** (`--maintain_every 0`).
  Dynamic growth corrupts the AdamW optimizer state and destabilizes long runs — this
  is fixed properly in Phase 0b before re‑enabling.
- **NaN guard + step‑tagged immutable backups are always on.** A FERN‑1 run silently
  trained on NaN for 36k steps (a depth‑2 ponder‑loss `inf` bug, now fixed) and the
  single overwritten checkpoint was lost. Watch **`ppl`**, not tok/s — tok/s prints
  fine even under NaN.
- **Throughput is dispatch‑bound** (Python MoE loop → GPU ~59% on the 5070). Bigger
  batch helps now; **vectorizing the MoE (Phase 0a) is the real fix.**
- Pretrain config that worked: `--preset small --no_ttm --depth 2 --block 512
  --batch 24 --amp`, ponder mode, on `E:\fern_data\py.bin` (codeparrot Python).

---

**Mission:** make FERN‑2 *theoretically the best coding model of its scale* — not by
out‑scaling anyone, but by stacking experimental architecture and training methods
that raise **capability per parameter**. The thesis: a small, sparse, deeply‑reasoning,
self‑improving model with execution feedback can punch far above its weight class on code.

This is a **research bet**, stated honestly: most novel architectures lose to a tuned
transformer until heavily refined, and several ideas here could fail. The plan is
designed to *maximize the odds* and, critically, to **fail loudly and early** so we
spend effort only on what measurably works.

---

## Operating principles (read first)

1. **One variable at a time.** Never enable two experimental pieces in the same run.
   If quality moves, you must be able to attribute it. Five-at-once is an
   unattributable, unstable mess.
2. **Every phase has a measurement gate.** A change ships only if it beats the
   previous best on the eval harness. No "it feels smarter."
3. **Baselines are sacred.** Always keep a same‑parameter **vanilla transformer**
   trained on the same data as the control. "Best of its scale" only means something
   measured against that.
4. **Proven backbone, speculative garnish.** The proven methods (RL on execution
   feedback, test‑time compute, SFT) carry the weight. The exotic architecture is
   *one* high‑synergy gamble at a time layered on top — never the foundation.
5. **Checkpoint immutably.** Step‑tagged backups + NaN guards always on. (We already
   lost a 36k run to a silent NaN; never again.)

---

## Phase −1: Evaluation harness (build this BEFORE anything else)

You cannot claim "best of its scale" without a ruler. Build first:

- **Code benchmarks:** HumanEval, MBPP, and a held‑out internal set of problems with
  unit tests. Metric: **pass@1** and **pass@10** (sample k, run tests, count solves).
- **Reasoning probe:** a small set of multi‑step problems where you can watch whether
  "thinking longer" actually helps.
- **Efficiency metrics:** active params/token, tokens/sec, decode latency, VRAM. "Best
  of its scale" is a *quality‑per‑cost* claim — track the cost axis too.
- **A sandbox** to execute generated code safely (subprocess + timeouts + resource
  caps, or a container). This is load‑bearing infrastructure for Phases 2–3.

**Gate:** harness runs end‑to‑end on the FERN‑1 chat checkpoint and produces numbers.

---

## Phase 0: Foundation fixes (carry‑overs from FERN‑1)

These unblock everything else. None are research — just engineering done right.

### 0a. Vectorize the MoE  *(highest priority)*
The per‑expert Python loop is dispatch‑bound (GPU sat at ~59% on a 5070). Replace with
a batched grouped‑GEMM / token‑sort dispatch so all experts run as a few big kernels.
**Unlocks:** affordable deep fractal recursion + test‑time memory, and full GPU
saturation. *Without this, every experimental reasoning method is too slow to iterate on.*

### 0b. Optimizer‑safe dynamic growth (#5)
The naive grow/prune swapped parameter objects out from under AdamW and corrupted
training. Fix with:
- **Function‑preserving growth** (new expert ≈ copy of a busy one + small noise, so
  adding it doesn't shock the loss — net2net / MoE‑upcycling style).
- **Optimizer migration**: rebuild the optimizer on growth, copying Adam moments for
  surviving params and zero‑init for new ones.
- Grow/prune **on a schedule between steps**, not mid‑step.

### 0c. Post‑training variable precision (#3)
Quantizing experts to 4‑bit *while training through them* destabilized FERN‑1. Do it
the proven way:
- Train full precision, **assign 16/8/4‑bit by usage after training** (PTQ).
- Optional later: QAT with a gentle anneal (8‑bit → 4‑bit) and per‑group scales.
- The payoff is **serving footprint**, captured fully by PTQ — no training risk.

**Gate:** vectorized MoE matches FERN‑1 quality at ≥2× tokens/sec; growth runs 10k
steps with no NaN; PTQ shrinks footprint with <1% pass@1 loss.

---

## Phase 1: Generation paradigm — Block Diffusion

Swap token‑by‑token AR for **block diffusion** (BD3‑LM style): autoregressive *between*
fixed blocks (keeps KV‑cache + arbitrary length), diffusion *within* a block (parallel
decode + bidirectional context).

- **Why for code:** native **infilling/editing** (better than the FIM hack — FIM
  becomes redundant), parallel within‑block decode (lower latency), bidirectional
  local context (bracket/type coherence).
- **Build:** masked/absorbing‑diffusion objective within blocks; block size ~16–32;
  a few denoising steps; block‑causal attention with KV‑cache across blocks.
- **Keep the AR path** alongside it so you can A/B them on the harness.
- **Honest cost:** new training objective + decode loop; from‑scratch diffusion is
  harder to get to quality. The FERN backbone (fractal core, MoE, gates) is untouched.
- **Honest scope:** this is mainly a **speed + editing + flexibility** win, *not* an
  intelligence raise. Treat it as the generation substrate the smart stuff rides on.

**Gate:** block‑diffusion variant matches AR pass@1 within noise while delivering
faster decode and better infilling pass@1. If it can't match AR quality, keep AR and
move on — don't sink the project into making diffusion work.

---

## Phase 2: Intelligence core (the proven backbone — this is where "smart" comes from)

> The single highest ceiling‑raiser for a *coding* model that doesn't require more
> data is **RL on verifiable rewards**. This is how o1 / R1‑class reasoning emerged.

### 2a. Strong SFT
Fine‑tune on high‑quality code‑instruction data (Magicoder, Evol‑Instruct, etc.) with
chat template + assistant‑only loss masking. This is the floor.

### 2b. RL from execution feedback  *(the big lever)*
- Model generates solutions **with long chain‑of‑thought**; run them in the Phase‑(−1)
  sandbox against tests; **reward = tests passed**.
- Optimize with **GRPO** (group‑relative, no value net — simplest stable choice at this
  scale) or PPO.
- The reward comes from a **code interpreter**, not human labels or new scraped data —
  this is "smarter from the same data."
- **Synergy:** this is what finally gives FERN's adaptive‑depth / latent reasoning
  something concrete to optimize — it learns to *think before it codes*.

### 2c. Test‑time compute (free intelligence at inference)
- **Self‑consistency:** sample N, run all, keep the one passing the most tests.
- **Self‑refinement:** generate → execute → feed the error back → fix (a REPL loop;
  huge effective‑capability boost on its own).
- **Search:** tree/MCTS over reasoning steps scored by the verifier.

**Gate:** RL + test‑time compute beats the SFT‑only model by a wide margin on pass@1
*and* pass@10. This phase alone should already make FERN‑2 strong for its size.

---

## Phase 3: Experimental architecture (the speculative bets — in dependency order)

Layer these on the Phase‑2 core **one at a time**, each behind a gate. Dependency
structure (the critic/verifier is the keystone):

```
                 ┌─ Self‑questioning curriculum ─┐
                 │   (generates hard problems)    │ ← needs a judge or it
   Critic / verifier  ◄─────────────────────────┘    trains on its own wrong answers
   (judges "correct?")
        ▲
        │ signal for
  Latent reasoning        Sleep/consolidation, trace‑cache
  (the thinking engine)   (memory & efficiency — defer)
```

### 3a. Critic / verifier  *(keystone — do first)*
A scorer for "is this solution correct?" For code it's largely **grounded in
execution** (tests pass), with a learned critic head as a soft signal for cases without
tests. Enables safe self‑questioning and reranking. Low risk because it's
execution‑grounded.

### 3b. Latent reasoning  *(the one architecture gamble worth taking)*
Reason in continuous latent vectors **before emitting tokens** (Coconut‑style),
decoupling thinking from output length. **Best fit for FERN** — the equilibrium /
adaptive‑depth core is already half of this. Finicky to train; that's why it goes after
the critic exists to measure whether it actually reasons better.

### 3c. Self‑questioning curriculum
The model generates problems at the edge of its ability and trains on solving them
(self‑play for reasoning). **Powerful but dangerous without 3a** — unjudged, it trains
on confident‑but‑wrong outputs and collapses. Only enable once the verifier is trusted.

### 3d. Deferred to FERN‑2.x (memory & efficiency, not intelligence drivers)
- **Sleep / consolidation:** periodically distill test‑time memory + episodic
  experience back into weights (wake‑sleep). Risk: catastrophic forgetting.
- **Reasoning‑trace cache/retrieval:** store solved reasoning graphs, retrieve for
  similar problems (FERN's neural‑cache stub). An efficiency/reuse layer.

**Gate (each):** the piece must beat the current best on the harness. If it doesn't,
cut it. The recommended *combo* if all pass is **latent reasoning + critic +
self‑questioning** as a closed self‑improvement loop — but only assembled after each
proves itself solo.

---

## Suggested execution order (one validated step at a time)

1. Phase −1 eval harness + sandbox
2. 0a vectorize MoE → 0b growth → 0c precision
3. Phase 2a SFT → **2b RL on execution feedback** → 2c test‑time compute
4. Phase 1 block diffusion (can run in parallel as a separate track once 0a lands)
5. Phase 3a critic → 3b latent reasoning → 3c self‑questioning
6. Phase 3d memory/consolidation (FERN‑2.x)

> Note the ordering: **RL (2b) before the exotic architecture (Phase 3).** The proven
> lever first establishes a strong, measurable baseline; the gambles are judged against
> *that*, not against the weak SFT‑only model.

---

## Risk register (honest failure modes)

- **Reward hacking** (RL): model games the tests (hard‑codes outputs, exploits the
  harness). Mitigate with held‑out tests, diverse cases, output checks.
- **Latent reasoning won't train** from scratch on consumer compute — real possibility;
  keep the token‑reasoning path as fallback.
- **Block diffusion underperforms AR** — keep AR; don't let diffusion become a tar pit.
- **Self‑questioning collapse** without a solid verifier — gate hard on 3a.
- **Compute wall:** RL rollouts + test‑time search are *inference‑heavy*. On one 5070
  this is the real bottleneck — budget for it, keep models small (the whole point).
- **Catastrophic forgetting** from consolidation — why it's deferred.

---

## Scale & honest expectations

- Target scale: FERN presets (~0.1B–2.4B total, ~10–35M active). "Best of its scale"
  means **vs same‑parameter baselines**, not vs frontier models.
- Architecture buys **capability‑per‑compute**, not a higher absolute ceiling than a
  data‑center model. The realistic, *achievable* win: a small model that, via reasoning
  + execution feedback + sparsity, **beats a vanilla transformer of equal size by a
  wide margin on code** and serves cheaply. That is a genuinely strong, defensible goal.
- If even half of Phases 0–2 land, FERN‑2 is a real result. Phase 3 is the upside swing.

---

## Compute & hardware strategy (local dev + cloud rental)

The binding constraint on a *bigger* FERN is **compute**, not architecture — we can
already *define* a 2.4B model (`large` preset); we just can't afford to train it on one
consumer GPU. The plan is a two‑tier compute model.

### Tier 1 — local dev (free): RTX 5070, 12 GB
Iterate here: code, debug, smoke tests, and `small` (~108M) training runs. Free, always
available, fast enough for the `small` preset. **All architecture/method development
happens here before spending a cent.**

### Tier 2 — cloud rental (paid): rent only for the big runs
Pay only for the hours of an actual large training run, then shut it down.

| GPU | VRAM | ~$/hr | Speed | Use for |
|-----|------|-------|-------|---------|
| RTX A6000 | 48 GB | $0.50–1 | Ampere (slow‑ish) | mid runs if H100 unavailable |
| **H100 SXM** | **80 GB** | **~$1.50** | **Hopper, ~3–6× A6000** | **the serious big‑model run** |

**The H100 is the better deal despite 2× the hourly rate** — 3–6× the throughput means
it's *cheaper per unit of training and far faster*. A job that's ~$80 / 100 h on an
A6000 is ~$35 / 24 h on an H100. 80 GB VRAM trains `large` (2.4B) with no offloading and
big batches. $1.50/hr is excellent (normal H100 is $2–4/hr). A realistic big run is
**$30–100 total.** Renting turns "bigger model" from a hard wall into a small budget line.

### THE CRITICAL RULE: fix efficiency *before* you rent
> Renting before fixing efficiency = **paying frontier‑GPU prices to watch a GPU idle.**

The faster the GPU, the *worse* our current bottlenecks hurt:
- The **un‑vectorized MoE** is dispatch‑bound — it starved a 5070 to ~59% util. On an
  H100 it would sit at **~20–30% util**: you'd pay $1.50/hr for a Ferrari in traffic.
- **Byte‑level** tokenization wastes ~3–5× the steps a BPE tokenizer would need to reach
  the same quality — i.e. ~3–5× the rental bill for the same result.
- A 108M model is far too small to use an H100 at all — the H100 only pays off on a
  **vectorized 1–2.4B model**.

### The two efficiency fixes that unlock cheap cloud training (do these FIRST)
1. **Vectorize the MoE** *(= Phase 0a)* — replace the Python per‑expert loop with a
   token‑sort + grouped/batched GEMM (or a Triton kernel). Removes the dispatch
   bottleneck; lets a fast GPU actually saturate. **This is the single change that makes
   rented compute worth the money.**
2. **BPE / code‑aware tokenizer** *(new efficiency item — promote from "limitations")* —
   replace byte‑level with a code‑aware BPE vocab. ~3–5× fewer steps to coherence =
   directly proportional cut to both local training time and the cloud bill. Keep the
   `vocab_size` / special‑token interface so the model code is unchanged. (Note: this
   changes the embedding table → a retrain, so do it as part of FERN‑2 from the start.)

### Renting logistics (when the time comes)
- Get the corpus onto the box (download the dataset there, or sync the `.bin`).
- Use a provider with **persistent storage**, or **download checkpoints periodically**.
- **Spot instances can be preempted** — fine, we checkpoint every N steps with immutable
  step‑tagged backups (already standard).
- Provider stability: Runpod / Lambda are steadier; Vast.ai is cheapest but flakier.
- Use a CUDA Docker image / template so env setup is one step.

**Sequencing:** vectorized MoE + BPE tokenizer (free, local) → then rent the H100 for one
big run. In that order the rented hours do 3–5× more work each.

## Definition of done

FERN‑2 v1 ships when, at a fixed parameter budget, it **beats the same‑size vanilla
transformer baseline on HumanEval/MBPP pass@1 by a clear margin**, with RL + test‑time
compute on, at competitive tokens/sec. Every experimental piece that's enabled has a
gate result proving it helped. Anything that didn't help is documented and cut.
