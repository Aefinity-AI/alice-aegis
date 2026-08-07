#!/usr/bin/env python3
"""param_count.py — build the model a train.py config JSON describes and print
its EXACT parameter count at runtime (no formula estimates).

Usage: python3 param_count.py configs/m7_ternary.json [configs/m7a_twin.json ...]
"""
import json
import sys

import torch.nn as nn

from model import TinyBitConfig, TinyBitModel, BitLinear

# train.py argparse dest -> TinyBitConfig field
KEYMAP = {
    "vocab_size": "vocab_size", "hidden": "hidden_size",
    "inter": "intermediate_size", "layers": "num_hidden_layers",
    "heads": "num_attention_heads", "kv_heads": "num_key_value_heads",
    "ctx": "max_position_embeddings", "rope_theta": "rope_theta",
    "rms_eps": "rms_norm_eps", "linear": "linear",
}


def report(path: str) -> int:
    with open(path) as f:
        raw = json.load(f)
    cfg = TinyBitConfig(**{KEYMAP[k]: v for k, v in raw.items() if k in KEYMAP})
    model = TinyBitModel(cfg)

    total = model.num_params()
    embed = model.embed_tokens.weight.numel()
    per_layer = sum(p.numel() for p in model.layers[0].parameters())
    n_bit = sum(1 for m in model.modules() if isinstance(m, BitLinear))
    n_fp = sum(1 for m in model.modules()
               if isinstance(m, nn.Linear) and not isinstance(m, BitLinear))

    print(f"{path}")
    print(f"  linear={cfg.linear}  hidden={cfg.hidden_size} inter={cfg.intermediate_size} "
          f"layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"kv={cfg.num_key_value_heads} vocab={cfg.vocab_size} ctx={cfg.max_position_embeddings}")
    print(f"  BitLinear modules: {n_bit} | plain nn.Linear modules: {n_fp}")
    print(f"  embedding (tied head): {embed:,}")
    print(f"  per-layer:             {per_layer:,}  x {cfg.num_hidden_layers}")
    print(f"  TOTAL params:          {total:,}  ({total/1e6:.3f}M)")
    return total


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    totals = [report(p) for p in sys.argv[1:]]
    if len(totals) == 2:
        print(f"ratio: {totals[0]/totals[1]:.3f}x  ({totals[0]:,} / {totals[1]:,})")


if __name__ == "__main__":
    main()
