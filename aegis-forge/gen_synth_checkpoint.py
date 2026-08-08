#!/usr/bin/env python3
"""Generate a synthetic HF-style ternary checkpoint and repack it into engine
artifacts. No real weights involved: this exists so the loader, the forge, and
the QEMU boot path can be exercised anywhere (see scripts/qemu_synth_gauntlet.sh
and aegis-core/tests/forge_artifacts.rs).

The model is silu / no-SubLN / untied-head — deliberately the NEW graph shape,
so the gate fails if any of those paths regress. Its vocab is the byte-level
alphabet (mirroring aegis-core tokenizer.rs byte_to_unicode), so any English
prompt tokenizes and the qemu-test generation loop can run. Output text is
gibberish by construction; the gates assert machinery, never quality.

Usage: python3 gen_synth_checkpoint.py CKPT_DIR OUT_DIR
"""
import json, os, struct, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repack_ternary import write_safetensors  # one safetensors writer, shared

HIDDEN, HEADS, KV_HEADS, FFN, LAYERS, VOCAB = 32, 4, 2, 64, 2, 257
KV_DIM = (HIDDEN // HEADS) * KV_HEADS
FORGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repack_ternary.py")
if len(sys.argv) != 3:
    sys.exit("usage: gen_synth_checkpoint.py CKPT_DIR OUT_DIR")
CKPT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(CKPT, exist_ok=True)

config = {
    "num_hidden_layers": LAYERS, "hidden_size": HIDDEN,
    "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
    "intermediate_size": FFN, "vocab_size": VOCAB,
    "max_position_embeddings": 32768,
    "hidden_act": "silu", "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
}
json.dump(config, open(os.path.join(CKPT, "config.json"), "w"))

# Byte-level alphabet (mirrors aegis-core tokenizer.rs byte_to_unicode) so
# any English prompt tokenizes — required for the QEMU generation smoke test.
def byte_to_unicode(b):
    if 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255:
        return chr(b)
    if b <= 32:
        return chr(256 + b)
    if 127 <= b <= 160:
        return chr(256 + 33 + (b - 127))
    return chr(323)  # 173

alphabet = []
seen = set()
for b in range(256):
    c = byte_to_unicode(b)
    if c not in seen:
        seen.add(c)
        alphabet.append(c)
assert len(alphabet) == VOCAB - 1, f"alphabet {len(alphabet)} != {VOCAB - 1}"
vocab_map = {c: i for i, c in enumerate(alphabet)}
tokenizer = {"model": {"vocab": vocab_map, "merges": []},
             "added_tokens": [{"id": VOCAB - 1, "content": "<|end|>"}]}
json.dump(tokenizer, open(os.path.join(CKPT, "tokenizer.json"), "w"))

def tern(n, seed):
    return [((seed + i) % 3) - 1 for i in range(n)]

tensors = {}
for i in range(LAYERS):
    p = f"model.layers.{i}"
    ones = struct.pack(f"<{HIDDEN}f", *([1.0] * HIDDEN))
    tensors[f"{p}.input_layernorm.weight"] = ("F32", [HIDDEN], ones)
    tensors[f"{p}.post_attention_layernorm.weight"] = ("F32", [HIDDEN], ones)
    for proj, do, di in [("self_attn.q_proj", HIDDEN, HIDDEN), ("self_attn.k_proj", KV_DIM, HIDDEN),
                         ("self_attn.v_proj", KV_DIM, HIDDEN), ("self_attn.o_proj", HIDDEN, HIDDEN),
                         ("mlp.gate_proj", FFN, HIDDEN), ("mlp.up_proj", FFN, HIDDEN),
                         ("mlp.down_proj", HIDDEN, FFN)]:
        vals = tern(do * di, i + do)
        raw = bytes((v + 256 if v < 0 else v) for v in vals)
        tensors[f"{p}.{proj}.weight"] = ("I8", [do, di], raw)
        tensors[f"{p}.{proj}.weight_scale"] = ("F32", [1], struct.pack("<f", 0.05))
tensors["model.norm.weight"] = ("F32", [HIDDEN], struct.pack(f"<{HIDDEN}f", *([1.0] * HIDDEN)))
tensors["model.embed_tokens.weight"] = ("F32", [VOCAB, HIDDEN],
    struct.pack(f"<{VOCAB*HIDDEN}f", *[0.01 * (k % 61) - 0.3 for k in range(VOCAB * HIDDEN)]))
tensors["lm_head.weight"] = ("F16", [VOCAB, HIDDEN],
    struct.pack(f"<{VOCAB*HIDDEN}e", *[0.02 * (k % 37) - 0.35 for k in range(VOCAB * HIDDEN)]))

write_safetensors(
    os.path.join(CKPT, "model.safetensors"),
    [(name, dtype, shape, raw) for name, (dtype, shape, raw) in tensors.items()],
)

sys.exit(subprocess.run([sys.executable, FORGE, CKPT, OUT, "--source-packing", "unpacked"]).returncode)
