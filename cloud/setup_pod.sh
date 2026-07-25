#!/usr/bin/env bash
# FERN-2 one-shot pod bootstrap. Run from the FERN-2 repo directory on a rented
# Linux GPU box (RunPod/Lambda/Vast). It installs deps, verifies the GPU, runs
# the smoke test (catch env problems for $0), and prints the exact next commands.
#
#   bash cloud/setup_pod.sh
#
set -euo pipefail

WORK="${WORK:-/workspace}"            # RunPod persistent volume mounts here
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
# fewer fragmentation OOMs on the big `large` model (harmless everywhere else)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "$WORK" "$HF_HOME"

echo "== 1/3 deps (torch is already in the CUDA image) =="
pip install -q --upgrade tokenizers datasets numpy

echo "== 2/3 GPU visible? =="
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda", ok,
      "|", torch.cuda.get_device_name(0) if ok else "NO GPU — wrong image/instance")
assert ok, "No CUDA GPU — fix the instance before spending money."
PY

echo "== 3/3 smoke test (proves the code runs on this box) =="
python smoke_test.py | tail -2

cat <<EOF

============================================================
Environment OK.  WORK=$WORK   HF_HOME=$HF_HOME
Everything below writes to the PERSISTENT volume ($WORK).

# 1) train the BPE tokenizer (downloads happen on the box — fast datacenter net)
python train_tokenizer.py --dataset codeparrot/codeparrot-clean --column content \\
    --max_docs 400000 --vocab_size 16384 --out $WORK/tok_code.json

# 2) tokenize the corpus to a memmapped .bin (MUST use the same tokenizer).
#    'large' wants lots of tokens — bump --max_mb as high as your volume allows.
python prepare_data.py --dataset codeparrot/codeparrot-clean --column content \\
    --tokenizer $WORK/tok_code.json --max_mb 12000 --out $WORK/py.bin

# 3) train LARGE (~2.4B) on the A100 80GB, inside tmux. Validate with AR first.
tmux new -s train
python train.py --preset large --amp --gen_mode ar \\
    --tokenizer $WORK/tok_code.json --bin $WORK/py.bin \\
    --block 1024 --batch 4 --grad_accum 8 --depth 2 --no_ttm \\
    --steps 60000 --save_every 500 --out $WORK/fern2_large.pt
#   detach: Ctrl-b then d   |   reattach: tmux attach -t train
#   ~55-60 GB peak on 80 GB. OOM? --batch 2. Headroom? --batch 6-8.
#   Switch to block diffusion with --gen_mode diffusion once AR ppl is falling.

# 4) pull the checkpoint home before you kill the pod (from your PC):
#    runpodctl receive <code>        (after 'runpodctl send $WORK/fern2_base.pt' on the pod)
============================================================
EOF
