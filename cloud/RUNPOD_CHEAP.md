# Training FERN-2.5 `small` on RunPod without breaking the bank

Target: the **`small`** preset (~110M total / ~16M active per token) — roughly 10x
`eco`, which is the jump that buys coherent multi-line code. Budget **$10–20**.

The golden rule: **generate the data ON the box.** Your home upload is slow; the
datacenter downloads codeparrot in minutes. You move ~200 KB of code up and one
checkpoint down.

---

## 1. Rent the pod (~3 min)

1. runpod.io → add **$20** credit (enough for this whole run).
2. **Deploy → Pods → GPU**: pick **RTX 4090 24GB** on **Community Cloud**
   (~$0.35–0.45/hr — the best price/performance for a model this size).
   A5000 / A40 also fine. You do *not* need an H100 for 110M params.
3. Template: any **PyTorch 2.x CUDA** image ("RunPod PyTorch").
4. **Attach a Network Volume, 60 GB** — mounts at `/workspace` and survives the
   pod being destroyed. Everything valuable goes there.
5. Deploy → open the **Web Terminal**.

**Before anything else**, confirm you got what you're paying for:

```bash
nvidia-smi
```

Wrong GPU or no CUDA? Destroy and redeploy. Don't debug on the clock.

---

## 2. Get the code + verify the environment (~2 min)

The repo is public, so no auth needed:

```bash
cd /workspace && git clone https://github.com/voxelocity/FERN-2.5.git && cd FERN-2.5
```

```bash
bash cloud/setup_pod.sh
```

That installs deps, checks the GPU, and runs the smoke test — catching environment
problems for a few cents instead of mid-run.

---

## 3. Tokenizer + data (~40 min, mostly CPU)

This part doesn't use the GPU, so keep it short. 1B tokens is plenty for `small`.

```bash
export WORK=/workspace
export HF_HOME=$WORK/hf_cache
```

```bash
python train_tokenizer.py --dataset codeparrot/codeparrot-clean --column content --max_docs 200000 --vocab_size 16384 --out $WORK/tok_code.json
```

```bash
python prepare_data.py --dataset codeparrot/codeparrot-clean --column content --tokenizer $WORK/tok_code.json --max_mb 1000 --out $WORK/py_bpe.bin
```

Note: `--max_mb` actually caps the **token count**, so `1000` ≈ 1B tokens (~2 GB
on disk). Raise it only if you plan a longer run.

---

## 4. Train (in tmux, so a dropped connection doesn't kill it)

```bash
tmux new -s train
```

```bash
python train.py --preset small --reversible --galore --amp --tokenizer $WORK/tok_code.json --bin $WORK/py_bpe.bin --block 1024 --batch 24 --grad_accum 1 --save_every 1000 --steps 20000 --out $WORK/fern25_small.pt
```

Detach with **Ctrl-b then d**; reattach with `tmux attach -t train`.

### Size the run from real numbers, not guesses

At around **step 100**, read the `tok/s` and `ETA` the trainer prints, then decide:

```
cost ≈ ETA_hours × hourly_rate
```

- ETA over ~30h? Lower `--steps` (restart; nothing is lost this early). 15000 steps
  still gives a solid model.
- Lots of spare VRAM (`nvidia-smi`)? Raise `--batch` to 32 or 48 — more tokens per
  step is free throughput.
- OOM? `--batch 12 --grad_accum 2`, or `--capacity_factor 3.0`.

**Stop when the metric flattens, not when the step counter ends.** From another
terminal (`tmux new -s eval`):

```bash
python sample.py --model /workspace/fern25_small.pt --infill_test --bin /workspace/py_bpe.bin
```

Watch the infill accuracy across checkpoints. While it's climbing, keep paying;
when it flattens for two checkpoints, you're done. (For reference, `eco` reached
44.4% at 11.6M params — `small` should beat that comfortably.)

---

## 5. Get the checkpoint home BEFORE you destroy the pod

On the pod:

```bash
runpodctl send /workspace/fern25_small.pt
```

It prints a one-time code. On your PC:

```bash
runpodctl receive <code>
```

Also grab `tok_code.json` — **a checkpoint is useless without its tokenizer**:

```bash
runpodctl send /workspace/tok_code.json
```

---

## 6. Don't leak money

- **Stop the pod the moment training ends.** Billing is per second and runs while
  it idles.
- A **stopped** pod still bills for its volume (pennies/day). Destroy the pod and
  keep only the network volume if you plan a second run.
- Set a **spend limit** in RunPod billing settings as a backstop.
- Checkpoints land on `/workspace`, so a crashed pod loses nothing — resume with
  `--resume /workspace/fern25_small.pt`.

---

## Rough cost expectation

| item | estimate |
|---|---|
| setup + tokenizer + data | ~1h → ~$0.45 |
| training 15–20k steps | ~15–30h → ~$7–14 |
| **total** | **~$8–15** |

Timing depends on the GPU you actually get, so treat the trainer's own ETA at
step 100 as the real number and adjust `--steps` to fit your budget.
