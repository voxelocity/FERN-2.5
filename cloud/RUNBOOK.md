# Renting an H100 and training FERN-2 — the no-fumbling runbook

The golden rule: **generate the data ON the box, never upload it from home.** Your
home upload is slow; the datacenter downloads codeparrot in minutes. The only
things you move are ~100 KB of code and, at the end, one checkpoint.

Recommended provider for a first run: **RunPod** (web terminal + Jupyter +
persistent volumes, least friction). Lambda is comparable; Vast.ai is cheaper but
flakier. All steps below are RunPod-flavored.

---

## 0. Before you rent (do this at home, free)
- Run the local validation so you rent to SCALE a proven model, not to debug:
  `python validate_local.py --gen_mode ar --steps 4000 --tokenizer <tok.json>`
- Push FERN-2 to a **private GitHub repo** (easiest way to get code on the box).
  It's not a git repo yet:
  ```
  cd C:\Users\shark\Downloads\FERN-2
  git init && git add . && git commit -m "FERN-2"
  # create an empty private repo on github.com, then:
  git remote add origin https://github.com/<you>/FERN-2.git
  git push -u origin main
  ```
  (No GitHub? Skip it and use `runpodctl send` in step 3 instead.)
- `.gitignore` already excludes checkpoints/data/venv so you don't push gigabytes.

## 1. Rent the pod (~2 min)
1. runpod.io → sign up → **add $30–50 credit**.
2. **Deploy → Pods → GPU**: pick **H100 SXM 80GB** (Community Cloud is cheapest).
3. Template: any **PyTorch 2.x CUDA** image (e.g. "RunPod PyTorch").
4. **Attach a Network Volume** (~100 GB) — it mounts at `/workspace` and PERSISTS
   after the pod dies. Everything valuable goes here.
5. Deploy. Open the **Web Terminal** (or SSH / Jupyter) from the pod page.

## 2. Sanity-check the box you're paying for (~1 min, do this FIRST)
```
nvidia-smi          # confirms an H100 is actually attached
```
If it's not an H100 or CUDA is missing, destroy and redeploy — don't proceed.

## 3. Get the code onto the pod
```
cd /workspace
git clone https://github.com/<you>/FERN-2.git && cd FERN-2
```
No GitHub? From your PC run `runpodctl send C:\Users\shark\Downloads\FERN-2`
(it prints a one-time code), then on the pod `runpodctl receive <code>`.

## 4. One-shot setup + the commands to run
```
bash cloud/setup_pod.sh
```
It installs deps, verifies the GPU, runs the smoke test, and prints the exact
tokenizer → data → train commands (all writing to `/workspace`). Run those in a
`tmux` session so training survives a dropped connection:
```
tmux new -s train      # detach with Ctrl-b then d ; reattach: tmux attach -t train
```

## 5. While it trains
- Watch **`ppl`**, not tok/s (a NaN run still prints tok/s — the FERN-1 trap).
  It should fall from ~hundreds into low single digits.
- Checkpoints land at `/workspace/fern2_base.pt` + step-tagged backups every
  `--save_every`. On a **spot** instance that's your preemption insurance.
- `watch -n5 nvidia-smi` to confirm the GPU is actually busy (util should be high
  now that the MoE is vectorized — that's the whole point of the rewrite).

## 6. Get your model home, then STOP PAYING
```
# on the pod:
runpodctl send /workspace/fern2_base.pt
# on your PC:
runpodctl receive <code>
```
Then in the RunPod UI: **Stop** the pod (billing stops). The Network Volume keeps
your data for a later run for a small storage fee; **Terminate** the pod +
volume when you're fully done.

---

## The five time-wasters this avoids
1. **Uploading the corpus from home** — generate it on the box (steps in setup_pod.sh).
2. **Env mismatch** — the smoke test catches it before the long run.
3. **Losing work to a disconnect / preemption** — tmux + `--save_every` + step-tagged backups.
4. **`.bin` / tokenizer mismatch** — `prepare_data.py --tokenizer` uses the SAME BPE as training.
5. **Forgetting to stop the pod** — the meter runs until you hit Stop.

## Training `large` (2.4B) on an A100 80GB
`large` = 2.444B total params, 43.2M active/token (98.2% sparse). It fits an
80 GB A100 with no offloading. Verified memory budget (bf16 training, AdamW):

| item | GB |
|---|---|
| params (fp32) | 9.8 |
| AdamW moments | 19.6 |
| gradients (fp32) | 9.8 |
| MoE weight-stack (chunked) | ~9.7 retained |
| **fixed subtotal** | **~49** |
| activations @ block 1024, batch 4 | ~6–10 |
| **peak** | **~55–60 GB** ✅ on 80 GB |

Recipe (what `setup_pod.sh` prints):
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # fewer fragmentation OOMs
python train.py --preset large --amp --gen_mode ar \
    --tokenizer $WORK/tok_code.json --bin $WORK/py.bin \
    --block 1024 --batch 4 --grad_accum 8 --depth 2 --no_ttm \
    --steps 60000 --save_every 500 --out $WORK/fern2_large.pt
```
Knobs if needed:
- **OOM** → `--batch 2` (halves activations). Still stuck → `--block 512`.
- **Lots of free VRAM** (check `nvidia-smi`) → `--batch 6` or `8` for throughput.
- Keep `--depth 2` for pretrain (higher depth multiplies both compute and
  activation memory) and `--no_ttm` (the sequential scan is slow).
- The MoE runs experts in chunks of `moe_stack_chunk` (default 64) to keep the
  weight-stack copy from spiking VRAM — no need to touch it.

## Cost/time sanity
A100 PCIe is ~$1.39/hr. `large` runs slower per step than a smaller model, so a
solid run is roughly **1–2 days (~$30–70)**. Start AR to validate the pipeline
cheaply, then do the block-diffusion run once AR's `ppl` is falling. If you just
want a first result fast/cheap, `--preset base` (~0.9B) on the same box trains in
a fraction of the time — do that first, then scale to `large`.
