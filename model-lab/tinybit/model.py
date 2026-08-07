#!/usr/bin/env python3
"""tinybit model.py — a llama-family ternary (BitNet-style QAT) decoder in plain
torch, built to match the A.L.I.C.E. Rust inference engine (aegis-core) EXACTLY.

Engine-parity contract (verified against aegis-core src on 2026-07-17):

  * RMSNorm:   y = x / sqrt(mean(x^2) + eps) * w        (rmsnorm_scalar in ops.rs)
  * RoPE:      HF-Llama rotate_half, freq = theta^(-2d/head_dim)
               (RopeCache + apply_rope in attention.rs; half-split, NOT interleaved)
  * SiLU:      x * sigmoid(x)                             (silu in ops.rs)
  * SwiGLU:    down( silu(gate(x)) * up(x) )
  * GQA:       kv_head = q_head // (n_heads / n_kv_heads)
  * SubLN:     attn_sub_norm on the concatenated attention output BEFORE o_proj;
               ffn_sub_norm on (silu(gate)*up) BEFORE down_proj  (BitNet b1.58)
  * BitLinear (all 7 projections q,k,v,o,gate,up,down), QAT fake-quant:
      weight (per-tensor):   gamma = w.abs().mean()
                             w_q   = round(w/gamma).clamp(-1, 1)
                             w_eff = w + (w_q*gamma - w).detach()          (STE)
      activation (per-token absmax int8, mirrors ops.rs quantize_activations_int8):
                             s     = 127 / x.abs().amax(-1,keepdim).clamp(min=1e-5)
                             x_eff = x + ((x*s).round().clamp(-127,127)/s - x).detach()
  * Embeddings / norms / LM head stay fp32 here (the engine reads them as BF16 —
    a deliberate, documented drift the round-trip gate bounds at <=3%).
  * NO biases anywhere. Embeddings tied to the LM head.

The activation quant lives INSIDE BitLinear, applied to the linear's input — which
is exactly where the engine applies it (rmsnorm -> quant_act -> ternary matmul).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class TinyBitConfig:
    vocab_size: int = 8192
    hidden_size: int = 256
    intermediate_size: int = 640
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    hidden_act: str = "silu"           # engine kernel: silu (llama) or relu2 (bitnet)
    tie_word_embeddings: bool = True
    use_subln: bool = True             # attn_sub_norm + ffn_sub_norm (default ON)
    linear: str = "bitlinear"          # "bitlinear" (ternary QAT) | "fp" (fp32 nn.Linear)

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.num_key_value_heads

    def validate_engine_constraints(self) -> None:
        """The repacker packs 4 ternary weights per byte along the input dim, so
        every projection's input dim must be divisible by 4. hidden feeds q/k/v/o
        and gate/up; intermediate feeds down; kv_dim is an output dim (must be a
        multiple of head_dim, already guaranteed)."""
        for name, v in (("hidden_size", self.hidden_size),
                        ("intermediate_size", self.intermediate_size),
                        ("kv_dim", self.kv_dim)):
            if v % 4 != 0:
                raise ValueError(f"{name}={v} must be divisible by 4 for 2-bit packing")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.hidden_act not in ("silu", "relu2"):
            raise ValueError(f"hidden_act {self.hidden_act!r} has no engine kernel")
        if self.linear not in ("bitlinear", "fp"):
            raise ValueError(f"linear={self.linear!r} must be 'bitlinear' or 'fp'")

    def to_hf_config(self) -> dict:
        return {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_word_embeddings,
            "hidden_act": self.hidden_act,
            "torch_dtype": "float32",
        }

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# QAT fake-quant primitives (STE)
# ---------------------------------------------------------------------------
def weight_ternary_gamma(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor gamma = mean(|w|). Kept in a helper so training, the QAT
    forward and export all compute the SAME value from the SAME latent w."""
    return w.abs().mean().clamp(min=1e-8)


def snap_ternary(w: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """round(w/gamma).clamp(-1,1) -> values in {-1, 0, +1}. torch.round is
    round-half-to-even, matching numpy.rint used by repack_ternary.py."""
    return torch.round(w / gamma).clamp_(-1.0, 1.0)


def weight_fake_quant(w: torch.Tensor) -> torch.Tensor:
    gamma = weight_ternary_gamma(w)
    w_q = snap_ternary(w, gamma)
    return w + (w_q * gamma - w).detach()


def act_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token (last-dim) absmax int8, quantize->dequantize round trip.
    Mirrors ops.rs quantize_activations_int8: s = 127/absmax, clamp [-127,127]."""
    absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    s = 127.0 / absmax
    x_q = torch.round(x * s).clamp_(-127.0, 127.0) / s
    return x + (x_q - x).detach()


class BitLinear(nn.Module):
    """Bias-free linear with BitNet QAT fake-quant. weight is [out, in], so
    F.linear(x, w) == x @ w.T, matching the engine's row-major projection."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = act_fake_quant(x)
        w_eff = weight_fake_quant(self.weight)
        return F.linear(x_q, w_eff)

    def export_ternary(self):
        """Return (w_q_float32 [out,in] in {-1,0,1}, scale float32 = 1/gamma).

        The engine multiplies the ternary dot by its stored scale; repack_ternary
        writes 1/source_scale, so source_scale must be 1/gamma for the engine to
        recover gamma. Round-trip identity: 1/scale == gamma == mean(|w|)."""
        with torch.no_grad():
            w = self.weight.detach().float()
            gamma = weight_ternary_gamma(w)
            w_q = snap_ternary(w, gamma)
            scale = (1.0 / gamma).reshape(1)
        return w_q.contiguous(), scale.contiguous(), float(gamma)


def make_linear(cfg: "TinyBitConfig", in_features: int, out_features: int) -> nn.Module:
    """Build one projection according to cfg.linear.

    SINGLE-VARIABLE RULE (M7/M7a twin experiment): the ONLY thing this switch
    changes is the weight-precision path of the 7 projections —
      * "bitlinear": BitLinear (ternary QAT weights + per-token int8 act quant)
      * "fp":        plain fp32 nn.Linear, NO bias, NO quantization of any kind
    Everything else (RMSNorm, RoPE, GQA, SwiGLU, SubLN, tied embedding head,
    init distribution normal(0, 0.02)) is byte-for-byte the same code path, so a
    ternary arm and an fp arm differ in exactly one variable: weight precision.
    Keep it that way — do not add fp-only or ternary-only features here."""
    if cfg.linear == "fp":
        lin = nn.Linear(in_features, out_features, bias=False)
        nn.init.normal_(lin.weight, mean=0.0, std=0.02)   # same init as BitLinear
        return lin
    return BitLinear(in_features, out_features)


# ---------------------------------------------------------------------------
# RMSNorm (fp32 weight, engine-exact formula)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(dt)


# ---------------------------------------------------------------------------
# RoPE — HF Llama rotate_half convention (matches attention.rs apply_rope)
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32):
    half = head_dim // 2
    idx = torch.arange(half, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** ((2.0 * idx) / head_dim))          # [half]
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)  # [seq]
    freqs = torch.outer(pos, inv_freq)                            # [seq, half]
    emb = torch.cat([freqs, freqs], dim=-1)                       # [seq, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, T, n_heads, head_dim]; cos,sin: [T, head_dim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


# ---------------------------------------------------------------------------
# attention + MLP
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: TinyBitConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = make_linear(cfg, cfg.hidden_size, cfg.hidden_size)
        self.k_proj = make_linear(cfg, cfg.hidden_size, cfg.kv_dim)
        self.v_proj = make_linear(cfg, cfg.hidden_size, cfg.kv_dim)
        self.o_proj = make_linear(cfg, cfg.hidden_size, cfg.hidden_size)
        self.attn_sub_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps) if cfg.use_subln else None

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim)

        q, k = apply_rope(q, k, cos, sin)

        # to [B, n_heads, T, head_dim]; expand kv heads for GQA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        rep = self.n_heads // self.n_kv
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        scores = scores + causal
        probs = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        out = torch.matmul(probs, v)                       # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # concat heads

        if self.attn_sub_norm is not None:
            out = self.attn_sub_norm(out)                  # SubLN before o_proj
        return self.o_proj(out)                            # BitLinear quantizes input


class MLP(nn.Module):
    def __init__(self, cfg: TinyBitConfig):
        super().__init__()
        self.gate_proj = make_linear(cfg, cfg.hidden_size, cfg.intermediate_size)
        self.up_proj = make_linear(cfg, cfg.hidden_size, cfg.intermediate_size)
        self.down_proj = make_linear(cfg, cfg.intermediate_size, cfg.hidden_size)
        self.act = cfg.hidden_act
        self.ffn_sub_norm = RMSNorm(cfg.intermediate_size, cfg.rms_norm_eps) if cfg.use_subln else None

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.act == "silu":
            gate = F.silu(gate)
        else:  # relu2
            gate = torch.relu(gate).pow(2)
        h = gate * up
        if self.ffn_sub_norm is not None:
            h = self.ffn_sub_norm(h)                       # SubLN before down_proj
        return self.down_proj(h)                           # BitLinear quantizes input


class Block(nn.Module):
    def __init__(self, cfg: TinyBitConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyBitModel(nn.Module):
    def __init__(self, cfg: TinyBitConfig):
        super().__init__()
        cfg.validate_engine_constraints()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # tied LM head: logits = hidden @ embed_tokens.weight.T
        self._rope = {}

    def _rope_cache(self, T, device):
        key = (T, device)
        if key not in self._rope:
            self._rope[key] = build_rope_cache(
                T, self.cfg.head_dim, self.cfg.rope_theta, device)
        return self._rope[key]

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        cos, sin = self._rope_cache(T, idx.device)
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = F.linear(x, self.embed_tokens.weight)     # tied head, fp32
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# teacher-forced perplexity — engine convention (aegis-core calculate_perplexity):
# predict positions 1..N-1 from their prefixes, average NLL over N-1 terms.
# ---------------------------------------------------------------------------
@torch.no_grad()
def teacher_forced_ppl(model: TinyBitModel, ids: torch.Tensor) -> float:
    """ids: 1-D LongTensor of length N (N <= max_position_embeddings). Uses the
    model's own forward: QAT (fake-quant) for linear="bitlinear" — exactly what
    export snapshots into the engine — or the plain fp32 pass for linear="fp"."""
    model.eval()
    ids = ids.view(1, -1)
    N = ids.shape[1]
    if N < 2:
        return float("nan")
    logits = model(ids).float()                 # [1, N, V]
    logp = torch.log_softmax(logits[0, :-1], dim=-1)   # predict 1..N-1
    tgt = ids[0, 1:]
    nll = -logp[torch.arange(N - 1), tgt]
    return float(torch.exp(nll.mean()).item())
