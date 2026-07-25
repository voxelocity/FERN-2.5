"""(Phase -1) The ruler. Evaluate a FERN checkpoint on code benchmarks and
report pass@k + efficiency — the numbers every later phase is gated on.

You cannot claim "best of its scale" without this. Run it on the FERN-1 chat
checkpoint to get a baseline, then on every FERN-2 variant and compare.

Examples
--------
Offline Gate check (zero downloads, the built-in mini set):
    python eval_harness.py --model E:\\fern\\fern_chat.pt --benchmark mini

HumanEval pass@1 (greedy-ish) and pass@10 (sampled):
    python eval_harness.py --model E:\\fern\\fern2_chat.pt --benchmark humaneval \\
        --k 1 --n 1 --temperature 0.2
    python eval_harness.py --model E:\\fern\\fern2_chat.pt --benchmark humaneval \\
        --k 10 --n 20 --temperature 0.8
"""

import os
os.environ.setdefault("HF_HOME", r"E:\fern\hf_cache")

import argparse
import json
import math
import time

import torch

from fern import FERN, make_tokenizer
from fern.config import backfill_config
from fern.benchmarks import get_benchmark
from fern.sandbox import check_humaneval

# stop the completion when the model leaves the function / starts rambling
STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@",
             "\nassert ", "\n>>>", "\n\n\n"]


def truncate_completion(text: str) -> str:
    cut = len(text)
    for s in STOP_SEQS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def strip_code_fence(text: str) -> str:
    """For chat-mode replies wrapped in ```...``` markdown."""
    if "```" not in text:
        return text
    parts = text.split("```")
    block = parts[1] if len(parts) >= 2 else text
    if "\n" in block:  # drop a leading language tag like "python"
        first, rest = block.split("\n", 1)
        if first.strip().isalpha():
            block = rest
    return block


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021): 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


@torch.no_grad()
def sample_completions(model, tok, cfg, prompt_text, n, max_new_tokens,
                       temperature, top_k, device, chat):
    if chat:
        from fern.data import render_chat
        msg = [{"role": "user",
                "content": "Complete this Python function:\n\n" + prompt_text}]
        ids, _ = render_chat(msg, cfg, add_generation_prompt=True)
        stop = [cfg.end_id, cfg.eos_id]
    else:
        ids = tok.encode(prompt_text, add_bos=True, add_eos=False)
        stop = [cfg.eos_id]
    prompt = torch.tensor([ids], dtype=torch.long, device=device)

    comps = []
    for _ in range(n):
        out = model.generate(prompt, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=top_k, stop_ids=stop)
        new = out[0, prompt.shape[1]:].tolist()
        text = tok.decode(new)
        comps.append(strip_code_fence(text) if chat else truncate_completion(text))
    return comps


@torch.no_grad()
def measure_efficiency(model, tok, cfg, device, max_new_tokens=64):
    rep = model.param_report()
    prompt = torch.tensor([tok.encode("def solve(x):\n", add_bos=True)],
                          dtype=torch.long, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    out = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8,
                         top_k=40)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    gen = out.shape[1] - prompt.shape[1]
    return {
        "total_params": rep["total_params"],
        "active_params_per_token": rep["active_params_per_token"],
        "sparsity": round(rep["sparsity"], 4),
        "decode_tok_per_s": round(gen / max(dt, 1e-6), 1),
        "decode_latency_ms_per_tok": round(dt / max(gen, 1) * 1000, 2),
        "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                         if device == "cuda" else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="FERN checkpoint (.pt)")
    ap.add_argument("--benchmark", default="mini",
                    choices=["mini", "humaneval", "mbpp"])
    ap.add_argument("--limit", type=int, default=None, help="cap #problems")
    ap.add_argument("--k", type=int, default=1, help="pass@k")
    ap.add_argument("--n", type=int, default=1, help="samples per problem (>=k)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--chat", action="store_true",
                    help="prompt as a chat instruction (for SFT checkpoints)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.n < args.k:
        args.n = args.k

    ck = torch.load(args.model, map_location=args.device, weights_only=False)
    cfg = backfill_config(ck["config"])
    model = FERN(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    if getattr(cfg, "offload_experts", False):
        model.enable_offload("cpu")
    tok = make_tokenizer(cfg)

    problems = get_benchmark(args.benchmark, args.limit)
    print(f"model      : {args.model} (step {ck.get('step','?')})")
    print(f"paradigm   : {cfg.gen_mode} | reasoning {cfg.reasoning_mode} | "
          f"tokenizer {cfg.tokenizer_kind} (vocab {cfg.vocab_size})")
    print(f"benchmark  : {args.benchmark} ({len(problems)} problems) | "
          f"pass@{args.k} over n={args.n} @ T={args.temperature}\n")

    per_problem = []
    solved = 0
    t_start = time.time()
    for pi, prob in enumerate(problems):
        comps = sample_completions(model, tok, cfg, prob["prompt"], args.n,
                                   args.max_new_tokens, args.temperature,
                                   args.top_k, args.device, args.chat)
        c = 0
        first_err = ""
        for comp in comps:
            res = check_humaneval(prob["prompt"], comp, prob["test"],
                                  prob["entry_point"], timeout=args.timeout)
            if res["passed"]:
                c += 1
            elif not first_err:
                first_err = (res["stderr"] or "").strip().split("\n")[-1][:120]
        pk = pass_at_k(args.n, c, args.k)
        solved += pk
        per_problem.append({"id": prob["id"], "c": c, "n": args.n,
                            f"pass@{args.k}": round(pk, 3)})
        flag = "OK " if c > 0 else "   "
        print(f"  [{flag}] {prob['id']:<24} {c}/{args.n} pass"
              + (f"   | {first_err}" if (args.verbose and c == 0) else ""))

    elapsed = time.time() - t_start
    score = solved / max(len(problems), 1)
    eff = measure_efficiency(model, tok, cfg, args.device)

    print(f"\n=== RESULTS ({args.benchmark}) ===")
    print(f"pass@{args.k}              : {score*100:.1f}%  "
          f"({solved:.2f}/{len(problems)})")
    print(f"active params/token  : {eff['active_params_per_token']:,} "
          f"of {eff['total_params']:,} ({eff['sparsity']*100:.1f}% sparse)")
    print(f"decode throughput    : {eff['decode_tok_per_s']} tok/s "
          f"({eff['decode_latency_ms_per_tok']} ms/tok)")
    if eff["peak_vram_mb"] is not None:
        print(f"peak VRAM            : {eff['peak_vram_mb']:.0f} MB")
    print(f"eval wall-clock      : {elapsed:.1f}s")

    report = {
        "model": args.model, "step": ck.get("step"),
        "benchmark": args.benchmark, "k": args.k, "n": args.n,
        "temperature": args.temperature, "gen_mode": cfg.gen_mode,
        f"pass@{args.k}": score, "efficiency": eff,
        "per_problem": per_problem,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
