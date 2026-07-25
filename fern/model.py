"""FERN — the full model, wiring all 12 principles into one forward pass.

Pipeline per segment:
  tokens
    -> embed (+ position)
    -> (12) read hierarchical memory  + (2) retrieve external knowledge   [additive, gated]
    -> (7) event gate: salience + global anchors
    -> (4)(11) fractal core: recursive, adaptive-depth reasoning with
               (1)(6) sparse MoE cognitive regions and (1) sparse attention,
               attending to the (8) compressed latent state as global memory
    -> (8) update compressed latent  -> (12) write to hierarchical memory
    -> (10) world-model auxiliary objective (predict the future in latent space)
    -> language head
  (9) neural cache wraps generation to skip recomputation.
  (3) variable precision + (5) dynamic growth act on the MoE out-of-band.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FERNConfig
from .event_gate import EventGate
from .fractal_core import FractalCore
from .state_compress import StateCompressor
from .hierarchical_memory import HierarchicalMemory
from .knowledge_store import KnowledgeStore, KnowledgeRetriever
from .neural_cache import NeuralCache
from .neuromodulator import Neuromodulator
from .test_time_memory import TestTimeMemory


class FERN(nn.Module):
    def __init__(self, config: FERNConfig, knowledge_store: KnowledgeStore | None = None):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        self.event_gate = EventGate(config)
        self.compressor = StateCompressor(config)
        self.memory = HierarchicalMemory(config)
        self.store = knowledge_store or KnowledgeStore(config.knowledge_dim)
        self.retriever = KnowledgeRetriever(config, self.store)
        self.core = FractalCore(config)

        # NEW radical subsystems
        self.neuromod = Neuromodulator(config) if config.use_neuromodulator else None
        self.ttm = TestTimeMemory(config) if config.use_test_time_memory else None

        self.world_model = nn.Linear(config.d_model, config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self.cache = NeuralCache(config)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, input_ids, latent=None, targets=None, mod_state=None,
                ttm_state=None, write_memory: bool = False, infer: bool = False,
                block_size: int | None = None):
        """`block_size` set => block-causal attention (block-diffusion mode):
        bidirectional within a block, causal across blocks. Left None => the
        FERN-1 strictly-causal AR behaviour. The reasoning core is identical in
        both; only the attention mask changes. When `targets` is given the model
        returns the AR next-token loss; the block-diffusion objective lives in
        fern/diffusion.py and calls this with targets=None to read raw logits
        (it folds in the returned raw `aux`/`ponder_loss`/`hidden` itself)."""
        B, T = input_ids.shape
        device = input_ids.device
        pos = torch.arange(T, device=device).clamp_max(self.config.max_seq_len - 1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None]

        # (12)(2) condition on memory + external knowledge (no-ops when empty)
        x = x + self.memory.read(x) + self.retriever(x)

        # (7) event-driven salience + global anchors
        salience, global_idx = self.event_gate(x)

        # (NEW) neuromodulator: one global brain-state -> gates for everything
        gates, mod_state = (None, None)
        think_gate = route_temp = None
        if self.neuromod is not None:
            gates, mod_state = self.neuromod(x, salience, mod_state)
            think_gate = gates["think"]
            route_temp = gates["route_temp"]

        # (NEW) test-time learning memory: fast-weights updated as we read.
        # Skipped under block diffusion: its causal delta-rule scan assumes
        # left-to-right reading, which conflicts with within-block bidirectional
        # attention. (FERN-1 pretraining ran with --no_ttm anyway.)
        if self.ttm is not None and block_size is None:
            wg = gates["mem_write"] if gates is not None else None
            rg = gates["mem_read"].view(B, 1, 1) if gates is not None else 1.0
            ttm_out, ttm_state = self.ttm(x, write_gate=wg, S=ttm_state)
            x = x + rg * ttm_out

        # (8) recurrent compressed latent state, used as global attn memory
        if latent is None:
            latent = self.compressor.initial_state(B, device)

        # (4)(11)(1)(6) fractal reasoning core (equilibrium or ponder mode)
        h, aux, ponder_loss, info = self.core(
            x, global_idx, memory=latent, salience=salience, infer=infer,
            think_gate=think_gate, route_temp=route_temp, block_size=block_size)

        # (8) update compressed state; (12) optionally persist it
        latent_new = self.compressor(latent, h)
        if write_memory:
            self.memory.write(latent_new)

        h = self.final_norm(h)
        logits = self.lm_head(h)

        # raw `aux`/`ponder_loss`/`hidden` are exposed (with grad) so the
        # diffusion loss in fern/diffusion.py can fold them into its objective.
        out = {"logits": logits, "latent": latent_new, "info": info,
               "salience": salience, "mod_state": mod_state,
               "ttm_state": ttm_state, "hidden": h,
               "aux": aux, "ponder_loss": ponder_loss}

        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=self.config.pad_id)
            # (10) world model: predict next-token hidden state in latent space
            if T >= 2:
                pred = self.world_model(h[:, :-1])
                tgt = h[:, 1:].detach()
                wm = F.smooth_l1_loss(pred, tgt)
            else:
                wm = h.new_zeros(())
            loss = ce + aux + ponder_loss + self.config.world_model_weight * wm
            out.update(loss=loss, ce=ce.detach(), aux=aux.detach(),
                       ponder=ponder_loss.detach(), world_model=wm.detach())
        return out

    # ---- (10) internal simulation: roll the latent forward, no tokens ------
    @torch.no_grad()
    def simulate(self, latent: torch.Tensor, steps: int) -> torch.Tensor:
        """Imagine `steps` future latent states without emitting language."""
        traj = [latent]
        cur = latent
        for _ in range(steps):
            cur = cur + self.world_model(cur) * self.config.ct_dt
            traj.append(cur)
        return torch.stack(traj, dim=1)  # [B, steps+1, M, D]

    # ---- generation: dispatch on the configured paradigm ------------------
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0,
                 top_k=40, use_cache=False, write_memory=False, stop_ids=None,
                 steps=None):
        """AR decode (FERN-1) or block-diffusion decode, per `config.gen_mode`."""
        if self.config.gen_mode == "diffusion":
            from .diffusion import generate_diffusion
            return generate_diffusion(
                self, input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, steps=steps,
                stop_ids=stop_ids)
        return self._generate_ar(input_ids, max_new_tokens, temperature, top_k,
                                 use_cache, write_memory, stop_ids)

    # ---- (9) cached autoregressive generation -----------------------------
    @torch.no_grad()
    def _generate_ar(self, input_ids, max_new_tokens=64, temperature=1.0,
                     top_k=40, use_cache=False, write_memory=False, stop_ids=None):
        self.eval()
        ids = input_ids
        stop = set(stop_ids or [self.config.eos_id])
        # IMPORTANT: training always ran each forward with a FRESH latent
        # (latent=None) and infer=False and no memory write. Generation MUST
        # match those conditions — carrying the evolving latent / writing memory
        # / early-halting feeds the model states it never saw in training and
        # collapses output into garbage. So: fresh latent every step, infer=False.
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.config.max_seq_len:]
            out = self.forward(ids_cond, latent=None, infer=False,
                               write_memory=write_memory)
            h_last = out["logits"][:, -1]                      # [B, V]

            cached = self.cache.get(h_last[0]) if (use_cache and self.config.use_neural_cache) else None
            logits = cached if cached is not None else h_last
            if cached is None and use_cache and self.config.use_neural_cache:
                self.cache.put(h_last[0], h_last[0])

            logits = logits / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[..., [-1]], -1e9)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            ids = torch.cat([ids, nxt.view(ids.size(0), 1)], dim=1)
            if int(nxt.view(-1)[0]) in stop:
                break
        return ids

    # ---- expert offloading -------------------------------------------------
    def enable_offload(self, offload_device="cpu"):
        """Park experts in CPU RAM and stream the active ones to the GPU. Call
        AFTER `.to('cuda')`: that moves everything to GPU, then this pulls the
        experts back to CPU. Lets total params scale with RAM, not VRAM."""
        self.core.block.moe.enable_offload(
            offload_device=offload_device, pin=self.config.pin_expert_memory)
        return self

    # ---- reporting ---------------------------------------------------------
    def vram_estimate(self) -> dict:
        """Rough resident-on-GPU param bytes (fp16) with vs without offload."""
        moe = self.core.block.moe
        expert_p = sum(p.numel() for p in moe.experts.parameters())
        total_p = sum(p.numel() for p in self.parameters())
        per_expert = sum(p.numel() for p in moe.experts[0].parameters())
        resident = total_p - expert_p + per_expert * self.config.moe_top_k
        return {
            "all_on_gpu_MB": total_p * 2 / 1e6,
            "offloaded_gpu_MB": resident * 2 / 1e6,   # core + active experts only
            "cpu_ram_experts_MB": expert_p * 2 / 1e6,
        }

    def param_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        moe = self.core.block.moe
        per_expert = sum(p.numel() for p in moe.experts[0].parameters())
        # params that actually fire for one token: shared core + top-k experts
        shared = total - sum(p.numel() for p in moe.experts.parameters())
        active = shared + per_expert * self.config.moe_top_k
        return {
            "total_params": total,
            "active_params_per_token": active,
            "sparsity": 1.0 - active / total,
            "experts": len(moe.experts),
            "active_experts_per_token": self.config.moe_top_k,
        }
