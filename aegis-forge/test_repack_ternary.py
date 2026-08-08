#!/usr/bin/env python3
"""Self-test for repack_ternary.py (T3a round-trip + T3b artifact shape).

Runs stdlib-only on synthetic tensors:

    python3 aegis-forge/test_repack_ternary.py    -> exit 0 on pass

The real-weight run of the same checks happens locally where the
checkpoints live; this pins the packing math and the artifact writer.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import repack_ternary as rt  # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# --- T3a: pack -> unpack -> pack is byte-identical -------------------------
seq = [-1, 0, 1, 1, 0, -1, -1, 0, 1, 0, 0, 0, -1, -1, 1, 1]
packed = rt.pack_ternary(seq)
check("pack length", len(packed) == len(seq) // 4)
check("unpack inverts pack", rt.unpack_ternary(packed) == seq)
check("pack(unpack(x)) byte-identical", rt.pack_ternary(rt.unpack_ternary(packed)) == packed)

# exhaustive over every 4-weight block (all 81 ternary combinations)
all_blocks = []
for a in (-1, 0, 1):
    for b in (-1, 0, 1):
        for c in (-1, 0, 1):
            for d in (-1, 0, 1):
                all_blocks += [a, b, c, d]
packed_all = rt.pack_ternary(all_blocks)
check("exhaustive round-trip", rt.unpack_ternary(packed_all) == all_blocks)
check("code 3 never emitted", all(((byte >> s) & 3) != 3 for byte in packed_all for s in (0, 2, 4, 6)))

# engine encoding pinned to build_unpack_lut: 00=0, 01=+1, 10=-1, LSB-first
check("engine code order", rt.pack_ternary([1, -1, 0, 0]) == bytes([0b00_00_10_01]))

# non-ternary input refused (the anti-transmutation guard)
try:
    rt.pack_ternary([0, 1, 2, 0])
    check("non-ternary refused", False, "accepted a 2")
except ValueError:
    check("non-ternary refused", True)

# --- hf1bitllm layout: [out/4, in] packed ALONG DIM 0 in row blocks ---------
# (pinned by correct_transmute.py unpack_bitnet — the converter that produced
# the working BitNet-2B MODEL.SAF. A same-position byte relabel is WRONG.)
OUT, IN = 8, 4
W = [[((r * IN + c + r) % 3) - 1 for c in range(IN)] for r in range(OUT)]
M = OUT // 4
raw = bytearray(M * IN)
for r in range(M):
    for c in range(IN):
        b = 0
        for i in range(4):
            b |= (W[i * M + r][c] + 1) << (2 * i)  # field 2i = output row i*M + r
        raw[r * IN + c] = b
flat = [W[r][c] for r in range(OUT) for c in range(IN)]
check("hf1bitllm dim0-block unpack", list(rt.unpack_hf1bitllm(bytes(raw), OUT, IN)) == flat)
check("hf1bitllm -> engine packing round-trip",
      rt.unpack_ternary(rt.pack_ternary(rt.unpack_hf1bitllm(bytes(raw), OUT, IN))) == flat)
try:
    rt.unpack_hf1bitllm(bytes([0b11000000] * (M * IN)), OUT, IN)  # code-3 field
    check("hf1bitllm code-3 refused", False, "accepted code 3")
except ValueError:
    check("hf1bitllm code-3 refused", True)
try:
    rt.unpack_hf1bitllm(bytes(M * IN + 1), OUT, IN)
    check("hf1bitllm size mismatch refused", False, "accepted wrong size")
except ValueError:
    check("hf1bitllm size mismatch refused", True)

# --- dtype conversions ------------------------------------------------------
f32 = struct.pack("<f", 1.0) + struct.pack("<f", -2.5)
check("f32->bf16 truncation", rt.raw_to_bf16(f32, "F32") == bytes([0x80, 0x3F, 0x20, 0xC0]))
check("bf16 passthrough", rt.raw_to_bf16(b"\x80\x3f", "BF16") == b"\x80\x3f")
f16 = struct.pack("<e", 1.0)
check("f16->bf16 via f32", rt.raw_to_bf16(f16, "F16") == bytes([0x80, 0x3F]))

# --- numpy fast paths must agree with the pure-python paths -----------------
# The accelerated branches only engage above a size threshold, so the tiny
# cases above never touch them — real checkpoints always do. Compute the
# expected bytes through the pure path (per-block calls stay under the
# threshold) and compare one big accelerated call against it.
if rt._np is not None:
    n = 8192
    vals = [((i * 7) % 3) - 1 for i in range(n)]
    pure = b"".join(rt.pack_ternary(vals[i:i + 4]) for i in range(0, n, 4))
    check("numpy pack_ternary == pure", rt.pack_ternary(vals) == pure)

    floats = [0.001 * i - 4.0 for i in range(n)]
    pure = b"".join(rt.f32_list_to_bf16(floats[i:i + 2]) for i in range(0, n, 2))
    check("numpy f32->bf16 == pure", rt.f32_list_to_bf16(floats) == pure)

    raw32 = struct.pack(f"<{n}f", *floats)
    pure = b"".join(rt.raw_to_bf16(raw32[i:i + 4], "F32") for i in range(0, len(raw32), 4))
    check("numpy raw F32->bf16 == pure", rt.raw_to_bf16(raw32, "F32") == pure)

    raw16 = struct.pack(f"<{n}e", *[0.01 * (i % 500) - 2.5 for i in range(n)])
    pure = b"".join(rt.raw_to_bf16(raw16[i:i + 2], "F16") for i in range(0, len(raw16), 2))
    check("numpy raw F16->bf16 == pure", rt.raw_to_bf16(raw16, "F16") == pure)

    def valid_hf_byte(k):
        f = [(k + j) % 3 for j in range(4)]
        return f[0] | (f[1] << 2) | (f[2] << 4) | (f[3] << 6)

    big = bytes(valid_hf_byte(i) for i in range(4096))  # [64/4=16, 256]
    res_np = rt.unpack_hf1bitllm(big, 64, 256)
    saved_np, rt._np = rt._np, None
    try:
        res_pure = rt.unpack_hf1bitllm(big, 64, 256)
    finally:
        rt._np = saved_np
    check("numpy unpack_hf1bitllm == pure", list(res_np) == res_pure)
else:
    print("  skip  numpy fast-path equivalence (numpy not installed)")

# --- T3b: end-to-end repack of a synthetic checkpoint -----------------------
HIDDEN, HEADS, KV_HEADS, FFN, LAYERS, VOCAB = 8, 2, 1, 16, 2, 6
KV_DIM = (HIDDEN // HEADS) * KV_HEADS

with tempfile.TemporaryDirectory() as td:
    ckpt = os.path.join(td, "ckpt")
    out = os.path.join(td, "out")
    os.makedirs(ckpt)

    config = {
        "num_hidden_layers": LAYERS, "hidden_size": HIDDEN,
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
        "intermediate_size": FFN, "vocab_size": VOCAB,
        "max_position_embeddings": 32768,  # must be clamped to --max-seq
        "hidden_act": "silu", "rope_theta": 1000042.0,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
    }
    with open(os.path.join(ckpt, "config.json"), "w") as f:
        json.dump(config, f)

    vocab_map = {f"t{i}": i for i in range(VOCAB - 1)}
    tokenizer = {"model": {"vocab": vocab_map, "merges": ["t1 t2"]},
                 "added_tokens": [{"id": VOCAB - 1, "content": "<|end|>"}]}
    # make the merge target resolvable: "t1t2" must be a token
    vocab_map["t1t2"] = 3
    del vocab_map["t3"]
    with open(os.path.join(ckpt, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f)

    def tern(n, seed):
        return [((seed + i) % 3) - 1 for i in range(n)]

    tensors = {}
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = ("F32", [HIDDEN], struct.pack(f"<{HIDDEN}f", *([1.0] * HIDDEN)))
        tensors[f"{p}.post_attention_layernorm.weight"] = ("F32", [HIDDEN], struct.pack(f"<{HIDDEN}f", *([1.0] * HIDDEN)))
        for proj, do, di in [("self_attn.q_proj", HIDDEN, HIDDEN), ("self_attn.k_proj", KV_DIM, HIDDEN),
                             ("self_attn.v_proj", KV_DIM, HIDDEN), ("self_attn.o_proj", HIDDEN, HIDDEN),
                             ("mlp.gate_proj", FFN, HIDDEN), ("mlp.up_proj", FFN, HIDDEN),
                             ("mlp.down_proj", HIDDEN, FFN)]:
            vals = tern(do * di, i)
            raw = bytes((v + 256 if v < 0 else v) for v in vals)
            tensors[f"{p}.{proj}.weight"] = ("I8", [do, di], raw)
            tensors[f"{p}.{proj}.weight_scale"] = ("F32", [1], struct.pack("<f", 0.25))
    tensors["model.norm.weight"] = ("F32", [HIDDEN], struct.pack(f"<{HIDDEN}f", *([1.0] * HIDDEN)))
    emb = struct.pack(f"<{VOCAB * HIDDEN}f", *[0.01 * k for k in range(VOCAB * HIDDEN)])
    tensors["model.embed_tokens.weight"] = ("F32", [VOCAB, HIDDEN], emb)
    tensors["lm_head.weight"] = ("F16", [VOCAB, HIDDEN],
                                 struct.pack(f"<{VOCAB * HIDDEN}e", *[0.02 * k for k in range(VOCAB * HIDDEN)]))

    rt.write_safetensors(
        os.path.join(ckpt, "model.safetensors"),
        [(name, dtype, shape, raw) for name, (dtype, shape, raw) in tensors.items()],
    )

    proc = subprocess.run([sys.executable, os.path.join(HERE, "repack_ternary.py"),
                           ckpt, out, "--source-packing", "unpacked"],
                          capture_output=True, text=True)
    check("repack exits 0", proc.returncode == 0, proc.stderr[-400:])

    if proc.returncode == 0:
        with open(os.path.join(out, "MODEL.SAF"), "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            base = 8 + hlen
            data = f.read()

        meta = json.loads(hdr["__metadata__"]["aegis_config"])
        check("metadata config vocab", meta["vocab_size"] == VOCAB)
        check("metadata config act", meta["hidden_act"] == "silu")
        check("metadata max_seq clamped", meta["max_position_embeddings"] == 2048)
        check("metadata untied", meta["tie_word_embeddings"] is False)

        q = hdr["model.layers.0.self_attn.q_proj.weight"]
        check("packed q shape", q["shape"] == [HIDDEN, HIDDEN // 4])
        s, e = q["data_offsets"]
        check("packed q round-trip",
              rt.unpack_ternary(data[s:e]) == tern(HIDDEN * HIDDEN, 0))

        sc = hdr["model.layers.0.self_attn.q_proj.weight_scale"]
        s, e = sc["data_offsets"]
        # Source stores the transformers-BitLinear QUANTIZATION scale
        # (dequant divides); the engine multiplies, so MODEL.SAF carries
        # the reciprocal: 1/0.25 = 4.0. See scalar_f32_scale's docstring.
        check("scalar F32 scale is engine-convention reciprocal",
              struct.unpack("<f", data[s:e])[0] == 4.0)

        check("no sub_norm tensors emitted",
              not any("sub_norm" in k for k in hdr))

        head = hdr["lm_head.weight"]
        s, e = head["data_offsets"]
        check("lm_head BF16 size", e - s == VOCAB * HIDDEN * 2)

        embed_size = os.path.getsize(os.path.join(out, "EMBED.BIN"))
        check("EMBED.BIN BF16 size", embed_size == VOCAB * HIDDEN * 2)

        with open(os.path.join(out, "VOCAB.BIN"), "rb") as f:
            magic, n = struct.unpack("<II", f.read(8))
        check("VOCAB.BIN magic+count", magic == rt.MAGIC and n == VOCAB)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("all repack self-tests passed")
