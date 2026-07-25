"""Decisive diagnostic: is the model fine but generation broken?

Test 1: teacher-forced argmax accuracy on a real code line (should be HIGH if
        the model learned — this is what ppl 1.7 implies).
Test 2: manual greedy decode with a FRESH latent each step and infer=False
        (i.e. matching TRAINING conditions), vs the carried-latent path.

If Test 1 is high but generate() gives garbage, the bug is the carried latent /
infer=True at inference, not the model.
"""
import sys, torch
from fern import FERN, ByteTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(r"E:\fern\fern_base.pt", map_location=dev, weights_only=False)
cfg = ck["config"]
m = FERN(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
tok = ByteTokenizer(cfg)

def show(label, s):
    sys.stdout.buffer.write((label + s + "\n").encode("ascii", "replace"))

# ---- Test 1: teacher-forced next-byte accuracy ----
text = "def add(a, b):\n    return a + b\n"
ids = tok.encode(text, add_bos=True)
x = torch.tensor([ids], device=dev)
with torch.no_grad():
    out = m(x[:, :-1])
pred = out["logits"].argmax(-1)[0]
tgt = x[0, 1:]
acc = (pred == tgt).float().mean().item()
show("Test1 teacher-forced argmax accuracy: ", f"{acc:.3f}")
show("Test1 predicted (teacher-forced): ", repr(tok.decode(pred.tolist())))

# ---- Test 2a: greedy, FRESH latent each step, infer=False (training-like) ----
def greedy(latent_carry, infer):
    cur = torch.tensor([tok.encode("def add(a, b):\n    ", add_bos=True)], device=dev)
    latent = None
    for _ in range(50):
        with torch.no_grad():
            o = m(cur[:, -cfg.max_seq_len:], latent=latent, infer=infer)
        if latent_carry:
            latent = o["latent"]
        nxt = o["logits"][:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], 1)
    return tok.decode(cur[0].tolist())

show("Test2a greedy FRESH latent, infer=False: ", repr(greedy(False, False)))
show("Test2b greedy CARRIED latent, infer=True (current generate): ", repr(greedy(True, True)))
