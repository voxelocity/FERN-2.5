"""Talk to your FERN coding chatbot.

    python chat.py --model E:\fern\fern_chat.pt

Maintains a multi-turn conversation, renders it with the chat template, and
generates the assistant reply until the model emits <end>. Commands: /reset to
clear history, /quit to exit.
"""

import argparse
import torch

from fern import FERN, ByteTokenizer
from fern.data import render_chat

SYSTEM = "You are FERN, a helpful coding assistant. Answer with correct, concise code."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="chat checkpoint from sft.py")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    args = ap.parse_args()

    ck = torch.load(args.model, map_location=args.device, weights_only=False)
    config = ck["config"]
    model = FERN(config).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    if config.offload_experts:
        model.enable_offload("cpu")
    tok = ByteTokenizer(config)

    print(f"FERN coding chatbot ({config.reasoning_mode} mode). "
          f"/reset to clear, /quit to exit.\n")
    history = []
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = []
            print("(history cleared)\n")
            continue

        history.append({"role": "user", "content": user})
        ids, _ = render_chat(history, config, system=SYSTEM,
                             add_generation_prompt=True)
        prompt = torch.tensor([ids], dtype=torch.long, device=args.device)
        out = model.generate(prompt, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             write_memory=False,
                             stop_ids=[config.end_id, config.eos_id])
        reply_ids = out[0, prompt.shape[1]:].tolist()
        reply = tok.decode(reply_ids)
        history.append({"role": "assistant", "content": reply})
        print(f"\nfern> {reply}\n")


if __name__ == "__main__":
    main()
