#!/usr/bin/env python3
"""export_hf.py — export a tinybit checkpoint to an HF-style directory that
aegis-forge/repack_ternary.py accepts in `--source-packing unpacked` mode.

Contract satisfied (read repack_ternary.py before editing this):
  * model.safetensors with HF llama tensor names.
  * Each of the 7 projections stored as SNAPPED ternary {-1,0,+1} (F32) PLUS a
    companion {proj}.weight_scale (F32, shape [1]) = 1/gamma = 1/mean(|w|).
    The repacker writes the reciprocal; the engine multiplies by gamma. Round-
    trip asserted here: 1/stored_scale ≈ w.abs().mean().
  * RMSNorm weights (input/post_attn/model.norm) + optional attn/ffn sub_norm
    stored as F32 (the repacker converts them to BF16).
  * tie_word_embeddings=true -> NO lm_head.weight; embeddings only.
  * config.json (LlamaForCausalLM / llama, all required fields, float32).
  * tokenizer.json copied in.
  * Divisibility (hidden, intermediate, kv_dim all % 4 == 0) asserted.

Usage (library):  export_checkpoint(model, cfg, tokenizer_json, out_dir)
Usage (CLI):      python3 export_hf.py CKPT.pt OUT_DIR [--tokenizer tokenizer.json]
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

from model import TinyBitModel, TinyBitConfig


def export_checkpoint(model: TinyBitModel, cfg: TinyBitConfig,
                      tokenizer_json: str, out_dir: str, verbose: bool = True):
    if getattr(cfg, "linear", "bitlinear") != "bitlinear":
        raise SystemExit(
            "[export] REFUSED: this checkpoint was trained with linear="
            f"{cfg.linear!r} (full-precision nn.Linear). Ternary packing "
            "(snapped {-1,0,+1} + weight_scale=1/gamma) is meaningless for fp "
            "weights and the aegis-forge repack path is ternary-only. "
            "fp twins are torch-side reference arms; nothing was written.")

    # ---- AUTHORITATIVE CHECK: interrogate the MODEL, not the caller's config ----
    # The config check above is advisory only. It reads a caller-supplied object,
    # and BitLinear vs nn.Linear produce state_dicts whose keys and shapes are
    # IDENTICAL at identical geometry (verified 2026-07-29: zero keys unique to
    # the fp arm). So a caller that hands us an fp model together with a config
    # saying "bitlinear" would sail past it and we would write an fp32 checkpoint
    # out as a ternary artifact — and every downstream gate would pass, because
    # the round-trip gate only ever compares torch against the engine, and both
    # would be consistently wrong. That failure mode ships a full-precision model
    # believing it is 2-bit.
    #
    # This matters most for the same-size ternary/fp ablation, where the two arms
    # have byte-identical geometry by construction and are therefore
    # indistinguishable by shape alone.
    from model import BitLinear  # local import: keeps module import order unchanged
    offenders = [n for n, m in model.named_modules()
                 if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"))
                 and not isinstance(m, BitLinear)]
    if offenders:
        raise SystemExit(
            f"[export] REFUSED: {len(offenders)} projection(s) are not BitLinear "
            f"— e.g. {offenders[:3]} (type {type(dict(model.named_modules())[offenders[0]]).__name__}). "
            f"The supplied config claims linear={getattr(cfg, 'linear', '?')!r}, so the "
            "config and the model DISAGREE. Refusing rather than exporting "
            "full-precision weights as a ternary artifact. Nothing was written.")
    cfg.validate_engine_constraints()
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    tensors: dict[str, torch.Tensor] = {}

    # embeddings (fp32, tied -> also the LM head)
    tensors["model.embed_tokens.weight"] = model.embed_tokens.weight.detach().float().contiguous()

    max_scale_err = 0.0
    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = layer.input_layernorm.weight.detach().float().contiguous()
        tensors[f"{p}.post_attention_layernorm.weight"] = layer.post_attention_layernorm.weight.detach().float().contiguous()
        if layer.self_attn.attn_sub_norm is not None:
            tensors[f"{p}.self_attn.attn_sub_norm.weight"] = layer.self_attn.attn_sub_norm.weight.detach().float().contiguous()
        if layer.mlp.ffn_sub_norm is not None:
            tensors[f"{p}.mlp.ffn_sub_norm.weight"] = layer.mlp.ffn_sub_norm.weight.detach().float().contiguous()

        proj_map = {
            "self_attn.q_proj": layer.self_attn.q_proj,
            "self_attn.k_proj": layer.self_attn.k_proj,
            "self_attn.v_proj": layer.self_attn.v_proj,
            "self_attn.o_proj": layer.self_attn.o_proj,
            "mlp.gate_proj": layer.mlp.gate_proj,
            "mlp.up_proj": layer.mlp.up_proj,
            "mlp.down_proj": layer.mlp.down_proj,
        }
        for name, bl in proj_map.items():
            w_q, scale, gamma = bl.export_ternary()
            # round-trip check: engine recovers gamma = 1/stored_scale
            recovered = 1.0 / float(scale.item())
            true_gamma = float(bl.weight.detach().float().abs().mean().item())
            max_scale_err = max(max_scale_err, abs(recovered - true_gamma) / max(true_gamma, 1e-8))
            # sanity: snapped values must be exactly ternary (repacker refuses otherwise)
            uniq = torch.unique(w_q)
            assert bool(((w_q == -1) | (w_q == 0) | (w_q == 1)).all()), \
                f"{p}.{name}: non-ternary snapped values {uniq.tolist()}"
            tensors[f"{p}.{name}.weight"] = w_q.float().contiguous()
            tensors[f"{p}.{name}.weight_scale"] = scale.float().contiguous()

    tensors["model.norm.weight"] = model.norm.weight.detach().float().contiguous()

    if max_scale_err > 1e-5:
        raise AssertionError(f"scale round-trip error {max_scale_err:.2e} too large "
                             "(1/stored_scale must equal mean|w|)")

    save_file(tensors, os.path.join(out_dir, "model.safetensors"))

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_hf_config(), f, indent=2)

    if not os.path.exists(tokenizer_json):
        sys.exit(f"tokenizer.json not found: {tokenizer_json}")
    shutil.copy(tokenizer_json, os.path.join(out_dir, "tokenizer.json"))

    if verbose:
        print(f"[export] {len(tensors)} tensors -> {out_dir}/model.safetensors")
        print(f"[export] scale round-trip max rel err {max_scale_err:.2e} (need <1e-5)")
        print(f"[export] tie_word_embeddings=True -> no lm_head.weight written")
    return out_dir


def _load_model_from_ckpt(ckpt_path: str) -> tuple[TinyBitModel, TinyBitConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TinyBitConfig(**ckpt["config"])
    model = TinyBitModel(cfg)
    model.load_state_dict(ckpt["model"])
    return model, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--tokenizer", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tokenizer.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)
    model, cfg = _load_model_from_ckpt(args.ckpt)
    export_checkpoint(model, cfg, args.tokenizer, args.out_dir)


if __name__ == "__main__":
    main()
