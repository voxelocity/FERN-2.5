"""Diagnostic: can the BASE model greedily continue real code?

Greedy (top_k=1, no cache, no chat template) continuation of an in-distribution
prompt. If this produces coherent code -> the model CAN generate, and the chat
gibberish is a sampling/SFT/format issue. If this is ALSO garbage -> it's the
fundamental byte-level/overfitting wall (or a generation-state bug).
"""
import sys, torch
from fern import FERN, ByteTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(r"E:\fern\fern_base.pt", map_location=dev, weights_only=False)
cfg = ck["config"]
m = FERN(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
tok = ByteTokenizer(cfg)

for prompt in ["def add(a, b):\n    ", "import numpy as np\n", "for i in range("]:
    ids = torch.tensor([tok.encode(prompt, add_bos=True)], device=dev)
    out = m.generate(ids, max_new_tokens=80, temperature=1.0, top_k=1,
                     use_cache=False, write_memory=False)
    txt = tok.decode(out[0].tolist())
    sys.stdout.buffer.write((">>> PROMPT: " + repr(prompt) + "\n").encode("ascii", "replace"))
    sys.stdout.buffer.write((txt + "\n\n").encode("ascii", "replace"))
