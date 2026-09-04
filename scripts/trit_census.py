#!/usr/bin/env python3
"""E-S1: trit census of real ternary weights (lossless; what is achievable
today). See state/reports/2026-09-04-SUBBIT-TERNARY-PLAN.md (claudius-maximus)
section 1.

Packing format (found by reading aegis-core/src/ops.rs::build_unpack_lut and
aegis-forge's upstream repacker, correct_transmute.py/local_transmute.py — NOT
by assumption): each U8 byte holds 4 trits, 2 bits each, LSB-first:
    code 00 = 0, code 01 = +1, code 10 = -1, code 11 = undefined (maps to 0.0
    in the engine, so a corrupt byte degrades gracefully). Packing runs along
    the INPUT dimension: a `[out_features, in_features]` weight is stored as
    `[out_features, in_features // 4]` U8, row-major, i.e. `packed[:, j]`
    holds trits `4*j .. 4*j+3` of input feature index for every output row.
    A single F32 `<name>.weight_scale` (absmean, shape [1]) accompanies each
    packed tensor and is applied post-matmul, so it plays NO role in the trit
    statistics themselves. This format is ALSO what Falcon-E-1B's checkpoint
    uses (verified: identical dtype/shape convention), i.e. it has already
    been through this same forge pipeline.

Reference-vs-actual packing: the plan's CSV asks for a `packed_bits_per_w`
column at the literal constant 1.6 (5 trits/byte, the theoretical best fixed
packing — see bytes_per_token.py). The format actually shipped by
aegis-core is 2.0 bits/weight (4 trits/byte, 2-bit codes) — noticeably looser
than the 1.6 reference and further still from log2(3)=1.585 (max possible
ternary entropy). Both numbers are reported; do not conflate them.

Runs streaming, one tensor at a time, via mmap — the file is never loaded
whole into RAM (RAM rule: penguin has ~2GB to work with under systemd-run).
"""

from __future__ import annotations

import argparse
import csv
import json
import lzma
import mmap
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bytes_per_token import (  # noqa: E402
    ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT,
    REFERENCE_5TRIT_PACKING_BITS_PER_WEIGHT,
    h_ternary,
)
import rans_codec  # noqa: E402

try:
    import zstandard  # type: ignore

    _HAVE_ZSTD_LIB = True
except ImportError:
    _HAVE_ZSTD_LIB = False

# symbol codes used throughout this file for the *unpacked* trit array:
# 0 -> weight 0, 1 -> weight +1, 2 -> weight -1, 3 -> undefined/corrupt (0b11)
CODE_TO_WEIGHT = np.array([0, 1, -1, 0], dtype=np.int8)

LAYER_TYPE_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_o",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def layer_type_of(name: str) -> str:
    for k, v in LAYER_TYPE_MAP.items():
        if k in name:
            return v
    return "other"


def read_safetensors_header(path: str):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    return header, data_start


def unpack_4trit_bytes(packed: np.ndarray) -> np.ndarray:
    """packed: 1-D uint8 array. Returns 1-D int8 array of trits, 4x longer,
    LSB-first (matches aegis-core's build_unpack_lut bit order exactly)."""
    b = packed.astype(np.uint8)
    codes = np.empty((b.size, 4), dtype=np.uint8)
    codes[:, 0] = b & 0x03
    codes[:, 1] = (b >> 2) & 0x03
    codes[:, 2] = (b >> 4) & 0x03
    codes[:, 3] = (b >> 6) & 0x03
    return codes.reshape(-1)  # still "codes" (0..3), caller maps to weights


def zstd_compress(data: bytes, level: int = 19) -> bytes:
    if _HAVE_ZSTD_LIB:
        return zstandard.ZstdCompressor(level=level).compress(data)
    raise RuntimeError("zstandard python package not available")


def zstd_decompress(data: bytes) -> bytes:
    if _HAVE_ZSTD_LIB:
        return zstandard.ZstdDecompressor().decompress(data)
    raise RuntimeError("zstandard python package not available")


def xz_compress(data: bytes, preset: int = 9) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_XZ, preset=preset)


def xz_decompress(data: bytes) -> bytes:
    return lzma.decompress(data)


def conditional_entropy(ctx: np.ndarray, sym: np.ndarray, alphabet: int = 3) -> float:
    """H(sym | ctx) in bits, from empirical joint counts."""
    joint = np.zeros((alphabet, alphabet), dtype=np.int64)
    np.add.at(joint, (ctx, sym), 1)
    total = joint.sum()
    if total == 0:
        return 0.0
    p_joint = joint / total
    p_ctx = p_joint.sum(axis=1, keepdims=True)
    h_joint = -np.sum(np.where(p_joint > 0, p_joint * np.log2(p_joint), 0.0))
    h_ctx = -np.sum(np.where(p_ctx > 0, p_ctx * np.log2(p_ctx), 0.0))
    return float(h_joint - h_ctx)


class HistAccum:
    """Fixed-bin histogram accumulated across many calls (memory-frugal:
    never stores the underlying per-row/per-column values, just bin counts)."""

    def __init__(self, nbins: int = 20):
        self.edges = np.linspace(0.0, 1.0, nbins + 1)
        self.counts = np.zeros(nbins, dtype=np.int64)

    def add(self, values: np.ndarray):
        h, _ = np.histogram(values, bins=self.edges)
        self.counts += h

    def to_rows(self):
        rows = []
        for i in range(len(self.counts)):
            rows.append((f"[{self.edges[i]:.2f},{self.edges[i+1]:.2f})", int(self.counts[i])))
        return rows


# BitNet-b1.58-2B-4T linear geometry: (out_features, in_features). Packed U8 rows hold
# in_features/4 bytes. Used only to re-shape FLAT U8 tensors written by the forge.
_BITNET_2B_GEOMETRY = {
    "q_proj": (2560, 2560), "o_proj": (2560, 2560),
    "k_proj": (640, 2560), "v_proj": (640, 2560),
    "gate_proj": (6912, 2560), "up_proj": (6912, 2560),
    "down_proj": (2560, 6912),
}


def infer_flat_shape(name: str, n_bytes: int):
    """Return (out_features, in_features_packed) for a flat packed tensor, or None."""
    for key, (out_f, in_f) in _BITNET_2B_GEOMETRY.items():
        if key in name and out_f * in_f == n_bytes * 4:
            return (out_f, in_f // 4)
    return None


def census_tensor(name: str, mm: mmap.mmap, entry: dict, data_start: int,
                   row_hist: HistAccum, col_hist: HistAccum,
                   rans_lanes: int | None):
    start, end = entry["data_offsets"]
    out_features, in_features_packed = entry["shape"]
    in_features = in_features_packed * 4
    n = out_features * in_features

    raw = np.frombuffer(mm, dtype=np.uint8, count=end - start, offset=data_start + start)
    codes = unpack_4trit_bytes(raw)  # length n, values 0..3
    n_anomaly = int(np.count_nonzero(codes == 3))
    weights = CODE_TO_WEIGHT[codes]  # int8, -1/0/+1 (anomalies -> 0)

    n_minus = int(np.count_nonzero(weights == -1))
    n_zero = int(np.count_nonzero(weights == 0))
    n_plus = int(np.count_nonzero(weights == 1))
    p_minus, p_zero, p_plus = n_minus / n, n_zero / n, n_plus / n
    h0 = h_ternary(p_minus, p_zero, p_plus)

    # symbol alphabet for the coder/H1 calcs: 0=zero,1=+1,2=-1 (drop the
    # anomaly code — real weight bytes never contain 0b11; if they did,
    # n_anomaly > 0 below is the loud signal, not a silent remap).
    sym = np.where(weights == 0, 0, np.where(weights == 1, 1, 2)).astype(np.uint8)
    grid = sym.reshape(out_features, in_features)

    # H1, previous trit in the same row (row-major "left neighbor")
    ctx_row = grid[:, :-1].reshape(-1)
    cur_row = grid[:, 1:].reshape(-1)
    h1_row = conditional_entropy(ctx_row, cur_row)

    # H1, same column, previous row ("up neighbor")
    ctx_col = grid[:-1, :].reshape(-1)
    cur_col = grid[1:, :].reshape(-1)
    h1_col = conditional_entropy(ctx_col, cur_col)

    row_zero_frac = np.mean(grid == 0, axis=1)
    col_zero_frac = np.mean(grid == 0, axis=0)
    row_hist.add(row_zero_frac)
    col_hist.add(col_zero_frac)

    # --- real coders, lossless round trip asserted ---
    b0, mism0 = rans_codec.roundtrip_order0(sym, alphabet=3, K=rans_lanes)
    b1, mism1 = rans_codec.roundtrip_order1(sym, alphabet=3, K=rans_lanes)
    rans0_bits_per_w = rans_codec.coded_size_bytes(b0) * 8 / n
    rans1_bits_per_w = rans_codec.coded_size_bytes(b1) * 8 / n

    packed_bytes = raw.tobytes()
    zstd_mismatch = 0
    z = None
    if _HAVE_ZSTD_LIB:
        z = zstd_compress(packed_bytes, level=19)
        zstd_bits_per_w = len(z) * 8 / n
        if zstd_decompress(z) != packed_bytes:
            zstd_mismatch = 1
    else:
        zstd_bits_per_w = float("nan")

    x = xz_compress(packed_bytes, preset=9)
    xz_bits_per_w = len(x) * 8 / n
    xz_mismatch = 0 if xz_decompress(x) == packed_bytes else 1

    roundtrip_mismatches = mism0 + mism1 + zstd_mismatch + xz_mismatch + n_anomaly

    row = dict(
        name=name,
        layer_type=layer_type_of(name),
        out_features=out_features,
        in_features=in_features,
        n=n,
        p_minus=p_minus,
        p_zero=p_zero,
        p_plus=p_plus,
        H0=h0,
        H1_row_prev=h1_row,
        H1_col_prev=h1_col,
        packed_bits_per_w_ref1p6=REFERENCE_5TRIT_PACKING_BITS_PER_WEIGHT,
        actual_aegis_packed_bits_per_w=ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT,
        rans0_bits_per_w=rans0_bits_per_w,
        rans1_bits_per_w=rans1_bits_per_w,
        zstd19_bits_per_w=zstd_bits_per_w,
        xz9_bits_per_w=xz_bits_per_w,
        bytes_packed_ref=n * REFERENCE_5TRIT_PACKING_BITS_PER_WEIGHT / 8,
        bytes_packed_actual=len(packed_bytes),
        bytes_rans0=rans_codec.coded_size_bytes(b0),
        bytes_rans1=rans_codec.coded_size_bytes(b1),
        bytes_zstd19=(len(z) if z is not None else -1),
        bytes_xz9=len(x),
        n_anomaly_code3=n_anomaly,
        roundtrip_mismatches=roundtrip_mismatches,
    )
    return row


CSV_FIELDS = [
    "name", "layer_type", "out_features", "in_features", "n",
    "p_minus", "p_zero", "p_plus", "H0", "H1_row_prev", "H1_col_prev",
    "packed_bits_per_w_ref1p6", "actual_aegis_packed_bits_per_w",
    "rans0_bits_per_w", "rans1_bits_per_w", "zstd19_bits_per_w", "xz9_bits_per_w",
    "bytes_packed_ref", "bytes_packed_actual", "bytes_rans0", "bytes_rans1",
    "bytes_zstd19", "bytes_xz9", "n_anomaly_code3", "roundtrip_mismatches",
]


def run_census(model_path: str, out_dir: str, max_tensors: int | None, rans_lanes: int | None,
                label: str):
    header, data_start = read_safetensors_header(model_path)
    tensor_names = sorted(
        k for k, v in header.items() if k != "__metadata__" and k.endswith(".weight") and v["dtype"] == "U8"
    )
    if max_tensors:
        tensor_names = tensor_names[:max_tensors]

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{label}_census.csv")
    row_hist = HistAccum()
    col_hist = HistAccum()

    total_n = 0
    total_bytes_packed_ref = 0.0
    total_bytes_packed_actual = 0
    total_bytes_rans0 = 0.0
    total_bytes_rans1 = 0.0
    total_bytes_zstd19 = 0.0
    total_bytes_xz9 = 0.0
    total_mismatches = 0
    by_layer_type = {}

    t0 = time.time()
    skipped = []
    with open(model_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        with open(csv_path, "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for i, name in enumerate(tensor_names):
                entry = header[name]
                if len(entry["shape"]) != 2:
                    # The forge stores packed U8 weights FLAT (shape [n_bytes]). Recover the
                    # (out_features, in_features/4) matrix shape from the layer name using the
                    # BitNet-b1.58-2B-4T geometry (hidden 2560, intermediate 6912, 20 heads x 128,
                    # 5 KV heads x 128 = 640) and check the byte count matches exactly.
                    n_bytes = entry["shape"][0] if entry["shape"] else 0
                    shape2 = infer_flat_shape(name, n_bytes)
                    if shape2 is None:
                        print(f"skip {name}: U8 tensor with shape {entry['shape']} — no known 2-D geometry", flush=True)
                        skipped.append((name, entry["shape"]))
                        continue
                    entry = dict(entry, shape=list(shape2))
                    print(f"shape {name}: flat {n_bytes} bytes -> {shape2[0]}x{shape2[1]*4} trits", flush=True)
                row = census_tensor(name, mm, entry, data_start, row_hist, col_hist, rans_lanes)
                writer.writerow(row)
                csvf.flush()

                total_n += row["n"]
                total_bytes_packed_ref += row["bytes_packed_ref"]
                total_bytes_packed_actual += row["bytes_packed_actual"]
                total_bytes_rans0 += row["bytes_rans0"]
                total_bytes_rans1 += row["bytes_rans1"]
                if row["bytes_zstd19"] >= 0:
                    total_bytes_zstd19 += row["bytes_zstd19"]
                total_bytes_xz9 += row["bytes_xz9"]
                total_mismatches += row["roundtrip_mismatches"]

                lt = row["layer_type"]
                agg = by_layer_type.setdefault(lt, {"n": 0, "n_zero": 0})
                agg["n"] += row["n"]
                agg["n_zero"] += int(round(row["p_zero"] * row["n"]))

                print(
                    f"  [{i+1}/{len(tensor_names)}] {name}: n={row['n']} "
                    f"p0={row['p_zero']:.4f} H0={row['H0']:.4f} "
                    f"rans0={row['rans0_bits_per_w']:.4f} rans1={row['rans1_bits_per_w']:.4f} "
                    f"mism={row['roundtrip_mismatches']}",
                    flush=True,
                )

    elapsed = time.time() - t0

    # write histograms
    with open(os.path.join(out_dir, f"{label}_row_zero_frac_hist.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "count"])
        w.writerows(row_hist.to_rows())
    with open(os.path.join(out_dir, f"{label}_col_zero_frac_hist.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "count"])
        w.writerows(col_hist.to_rows())

    summary = {

        "skipped_non2d": [f"{n} {tuple(sh)}" for n, sh in skipped],
        "model_path": model_path,
        "label": label,
        "n_tensors": len(tensor_names),
        "total_weights": total_n,
        "overall_bits_per_weight": {
            "packed_reference_5trit": REFERENCE_5TRIT_PACKING_BITS_PER_WEIGHT,
            "actual_aegis_packing": ACTUAL_AEGIS_PACKING_BITS_PER_WEIGHT,
            "rans0_achieved": total_bytes_rans0 * 8 / total_n if total_n else None,
            "rans1_achieved": total_bytes_rans1 * 8 / total_n if total_n else None,
            "zstd19_achieved": total_bytes_zstd19 * 8 / total_n if total_n else None,
            "xz9_achieved": total_bytes_xz9 * 8 / total_n if total_n else None,
        },
        "total_bytes": {
            "packed_reference_5trit": total_bytes_packed_ref,
            "actual_aegis_packing": total_bytes_packed_actual,
            "rans0": total_bytes_rans0,
            "rans1": total_bytes_rans1,
            "zstd19": total_bytes_zstd19,
            "xz9": total_bytes_xz9,
        },
        "total_roundtrip_mismatches": total_mismatches,
        "by_layer_type_zero_fraction": {
            k: (v["n_zero"] / v["n"] if v["n"] else None) for k, v in by_layer_type.items()
        },
        "elapsed_seconds": elapsed,
        "csv_path": csv_path,
    }
    with open(os.path.join(out_dir, f"{label}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_path")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--max-tensors", type=int, default=None)
    ap.add_argument("--rans-lanes", type=int, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    label = args.label or Path(args.model_path).stem
    summary = run_census(args.model_path, args.out_dir, args.max_tensors, args.rans_lanes, label)

    print("\n=== SUMMARY:", label, "===")
    print(json.dumps(summary, indent=2))
    if summary["total_roundtrip_mismatches"] != 0:
        print("LOSSLESSNESS ASSERTION FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
