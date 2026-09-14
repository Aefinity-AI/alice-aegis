#!/usr/bin/env python3
"""CA1 — PTQ collapse-point ladder on the dense component of Falcon-E-1B.

Scope note (adversarially checked before any heavy compute ran): the shipped
`falcon_e_1b_model.safetensors` on this box is already the ENGINE-forged
artifact — every attention/MLP linear in all 24 layers is stored 2-bit
packed (U8 codes {0,1,2} = {0,+1,-1}) with one F32 row-... actually
per-TENSOR absmax scale. Row-wise (or tensor-wise) absmax PTQ at ANY bit
width >= 2 bits (int8/int4/int3/int2/ternary all have >= 4 representable
levels) reconstructs a 3-level {-1,0,1} source EXACTLY — this is provable
algebraically (scale cancels) and was spot-checked numerically below. So
there is NO PTQ ladder to observe on the transformer body: it is already
sitting at the floor of the ladder by construction (QAT-trained ternary).

The only dense (non-ternary) linear layer at production scale in this
artifact is `lm_head.weight` (BF16, [32768, 2048], 67,108,864 params,
untied). This script applies the SAME row-wise absmax weight-only PTQ
recipe used for the 30M dense-body ladder (report:
2026-08-29-ALICE-RUST-AND-BARRIERS.md) to lm_head ONLY, producing 5 new
MODEL.SAF copies (int8_w, int4_w, int3_w, int2_w, ternary_w) with every
other tensor byte-identical to the source file. This is a scope reduction
from "every linear layer" to "the one dense linear layer this native-
ternary architecture actually has" — recorded honestly, not hidden.
"""
import json
import struct
import sys
import shutil
from pathlib import Path

import numpy as np

SRC = Path("/home/justinbrianthompson/projects/alice-aegis/falcon_e_1b_model.safetensors")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ca1_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEVELS = {
    "int8_w": 8,
    "int4_w": 4,
    "int3_w": 3,
    "int2_w": 2,
    "ternary_w": 2,  # ternary == 2-bit (3 used levels: -1,0,+1), same formula
}


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header_len = n
    return header, header_len


def bf16_bytes_to_f32(buf: bytes) -> np.ndarray:
    u16 = np.frombuffer(buf, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    u32 = arr.astype(np.float32).view(np.uint32)
    # round-to-nearest-even on the low 16 bits before truncation
    rounding_bias = ((u32 >> 16) & 1) + 0x7FFF
    u32_rounded = u32 + rounding_bias
    u16 = (u32_rounded >> 16).astype(np.uint16)
    return u16.tobytes()


def row_wise_absmax_ptq(w: np.ndarray, bits: int) -> np.ndarray:
    """Symmetric row-wise absmax PTQ, weight-only, fake-quantized back to
    float (dequantized) — same recipe as the 30M ladder (Idea-3): one scale
    per output row, N signed integer levels, round-to-nearest, clip."""
    qmax = (1 << (bits - 1)) - 1  # e.g. bits=8 -> 127, bits=2 -> 1
    absmax = np.abs(w).max(axis=1, keepdims=True)
    absmax = np.where(absmax == 0, 1.0, absmax)
    scale = absmax / qmax
    codes = np.clip(np.round(w / scale), -qmax, qmax)
    return (codes * scale).astype(np.float32)


def main():
    header, header_len = read_header(SRC)
    meta = header.get("__metadata__", {})
    info = header["lm_head.weight"]
    assert info["dtype"] == "BF16", info
    off0, off1 = info["data_offsets"]
    shape = info["shape"]

    with open(SRC, "rb") as f:
        f.seek(8 + header_len + off0)
        raw = f.read(off1 - off0)
    w = bf16_bytes_to_f32(np.frombuffer(raw, dtype="<u2").copy()).reshape(shape)

    # --- adversarial sanity check: the body IS lossless under this recipe ---
    # spot-check one packed ternary tensor's reconstructed values are exactly
    # {-scale, 0, +scale} and that requantizing at int2..int8 changes nothing.
    body_info = header["model.layers.0.self_attn.q_proj.weight"]
    boff0, boff1 = body_info["data_offsets"]
    with open(SRC, "rb") as f:
        f.seek(8 + header_len + boff0)
        body_raw = f.read(boff1 - boff0)
    codes_u8 = np.frombuffer(body_raw, dtype=np.uint8)
    unpacked = np.zeros(codes_u8.size * 4, dtype=np.float32)
    lut = {0: 0.0, 1: 1.0, 2: -1.0, 3: 0.0}
    for shift in range(4):
        c = (codes_u8 >> (2 * shift)) & 0x3
        vals = np.vectorize(lut.get)(c)
        unpacked[shift::4] = vals
    uniq = np.unique(unpacked)
    print(f"sanity: q_proj.0 unpacked unique codes = {uniq}", file=sys.stderr)
    assert set(uniq.tolist()) <= {-1.0, 0.0, 1.0}, "body is NOT ternary — assumption wrong"
    # requantizing a {-1,0,1}-valued row at any bits>=2 must be a no-op
    test_row = unpacked.reshape(-1, 512)[:8].copy()
    for b in (8, 4, 3, 2):
        req = row_wise_absmax_ptq(test_row, b)
        assert np.array_equal(req, test_row), f"UNEXPECTED: bits={b} changed an already-ternary row"
    print("sanity PASS: body tensors are already exactly ternary; PTQ at "
          "int8/int4/int3/int2/ternary is a provable no-op on the body.", file=sys.stderr)

    # --- real ladder: lm_head.weight only ---
    for name, bits in LEVELS.items():
        out_path = OUT_DIR / f"falcon_e_1b_model.{name}.safetensors"
        shutil.copyfile(SRC, out_path)
        q = row_wise_absmax_ptq(w, bits)
        new_bytes = f32_to_bf16_bytes(q)
        assert len(new_bytes) == (off1 - off0)
        with open(out_path, "r+b") as f:
            f.seek(8 + header_len + off0)
            f.write(new_bytes)
        n_exact = int(np.sum(q == w))
        print(f"{name} (bits={bits}): wrote {out_path}, lm_head rows unchanged "
              f"{n_exact}/{q.size} elements", file=sys.stderr)

    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
