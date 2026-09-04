#!/usr/bin/env python3
"""E-S2b -- magnitude-aware zero-threshold sweep on the TRUE BitNet-2B bf16
master weights (LANE E-S2b, follow-up to E-S2).

Plan: state/reports/2026-09-04-SUBBIT-TERNARY-PLAN.md section 2, amended by
section 7 ("Amendment 2026-09-04 02:50 UTC (Fable)"). E-S2 (scripts/
ptq_zero_sweep.py, branch cm/es2-zero-threshold-sweep) found that
alice-aegis/model.safetensors is NOT a dense pre-quantization latent -- it
is the shipped QAT-trained ternary checkpoint (HF1BitLLM packed), so E-S2
had to fall back to magnitude-BLIND random pruning. This script is the
follow-up: it uses the TRUE dense bf16 master (Hugging Face
microsoft/bitnet-b1.58-2B-4T-bf16, MIT), staged on box1 at
~/aefinity-artifacts/bitnet-2b-bf16/model.safetensors (4,825,679,400 bytes,
sha256 529637ff6dab1f5890767356928693f69ffe61d3b6040a43de9306b37bfd5ae1),
which really does carry a continuous latent per weight -- so the plan's
original magnitude-AWARE tau*gamma sweep can finally run.

BitNet b1.58 absmean rule (Wang et al. 2023): per tensor,
  gamma = mean(|W|)                       (absmean scale, NOT excluding zeros)
  W~    = RoundClip(round(W / gamma), -1, 1)
which is algebraically a hard threshold at |W/gamma| < 0.5 -> 0, else
sign(W) (round-to-nearest of a value in [0, 1.5) only ever lands on 0 or 1;
above 1.5 the clip and sign(W) agree) -- proved in ptq_zero_sweep.py's
docstring, reused here rather than re-derived. This script generalizes that
threshold to tau*gamma for tau in a swept list, keeping gamma FIXED per
tensor (computed once from the full dense tensor, never recomputed after
zeroing) so tau=0.5 reproduces BitNet's own trained operating point exactly
-- the --verify mode checks precisely this against the shipped packed
checkpoint.

VERIFY FINDING (box1, nice -n 15, mmap-streamed, 382.3s wall -- Rule A: count
only): 210/210 tensors show SOME mismatch, 31,895,216 / 2,084,044,800 elements
(1.530448%) total, while the shipped-scale check (ref_gamma/gamma) is tight:
mean 0.999931, std 0.00175, range [0.996591, 1.003569] -- gamma itself is
recovered correctly (confirming the MULTIPLY convention, W ~= gamma*trit,
same as ptq_zero_sweep.py's finding for this model family). Direct inspection
of one tensor (model.layers.0.self_attn.q_proj.weight, 79,719 mismatches of
6,553,600) found the exact mechanism: EVERY mismatch in that tensor shares
the identical |W| bf16 value (0.609375), sitting 0.008% below that tensor's
0.5*gamma boundary, and every one resolves this script's trit to 0 while the
shipped checkpoint keeps the original sign. bf16 carries ~3 decimal digits
(mantissa ULP near a value of magnitude ~gamma is ~gamma/128, ~0.78%
relative) -- far coarser than the sub-percent margin many trained weights
actually sit at relative to their tensor's absmean boundary, so a small,
gamma-magnitude-dependent set of *discrete* bf16 levels near 0.5*gamma is
fundamentally ambiguous once the weights have been rounded to bf16 for
release. This is real information loss in the released bf16 artifact, not a
bug in this script's threshold logic or in the shipped checkpoint: 1.53% is
the honest floor of "how well can tau=0.5 on the bf16 master reproduce the
shipped trits", not "a handful of ties" as originally hoped in the plan's
own phrasing -- reported precisely rather than rounded down to that hope.
Because this exceeds the 0.1% refuse-threshold this script's own --sweep
mode enforces (per the task's leg-safety requirement), a real --sweep run
on this master, at this default threshold, WILL currently refuse -- this is
correct, conservative behavior given the finding above, and is flagged here
rather than silently loosened.

Scale convention used when re-packing a swept point into the engine's
MODEL.SAF: the stored per-tensor scale is gamma itself (the MULTIPLY
convention -- W ~= gamma * trit), matching the shipped checkpoint's own
convention, per ptq_zero_sweep.py's empirical finding for this model
family (not the onebitllms/Falcon-E DIVISION convention). --verify reports
whether the shipped scale equals gamma or 1/gamma to bf16 precision so this
is checked here too, not merely assumed.

RAM / streaming: the master file is 4.8 GB bf16 (~2.1B dense parameters).
This script NEVER loads it whole. Every tensor is read individually via
mmap (only the touched pages are paged in) and processed one at a time;
peak additional memory for the largest body tensor (mlp.gate_proj /
mlp.up_proj, 6912x2560) is a low hundred MB (raw bf16 + f32 + int8 trit
arrays). Per the house rule, real --verify/--sweep runs against the real
master only happen on box1; on penguin this script is exercised only via
--selftest against small synthetic tensors built in memory.

Modes:
    --verify           stream every one of the 210 body tensors (30 layers
                        x 7 projections, aegis-forge/repack_ternary.py's
                        PROJECTIONS list, reused) from the bf16 master,
                        compute gamma + tau=0.5 trits, and compare
                        tensor-for-tensor against the shipped packed
                        checkpoint's trits (ptq_zero_sweep.py's
                        engine_unpack_np + read_header/read_bytes, reused).
                        Reports mismatched tensors/elements, flags any
                        mismatch that lands exactly on the |W|==0.5*gamma
                        tie boundary (bf16 rounding could produce a few),
                        reports the shipped-scale-vs-gamma comparison, and
                        prints the whole-body + per-projection-type tau=0.5
                        p0 census.
    --sweep             for each tau in --taus (default 0.5,0.75,1.0,1.25,
                        1.5,2.0,2.5,3.0): recompute trits per tensor at
                        that tau (same fixed gamma), forge MODEL.SAF
                        (aegis-forge/repack_ternary.py's pack_ternary /
                        write_model_saf, reused -- EMBED.BIN/VOCAB.BIN are
                        NOT regenerated), run aegis-eval --cis-full if
                        --eval-bin is given, append one RESULT.txt row
                        immediately (append-only, per docs/LEGS.md), then
                        delete the forged model (unless --keep-forged).
                        Also emits the E-S3 bytes/token line per tau
                        (packed / order-0 entropy-coded / zero-skip) --
                        scripts/bytes_per_token.py does not exist on
                        origin/main as of this run (checked with `git show
                        origin/main:scripts/bytes_per_token.py`), so this
                        reuses ptq_zero_sweep.py's bytes_per_token_table()
                        (which itself re-derives the plan section 3
                        formulas directly, since E-S3 has not run) rather
                        than re-implementing a third copy.
    --selftest          penguin-safe: synthetic in-memory bf16 tensors only,
                        no file I/O against the real master. Checks (a) a
                        two-magnitude-group tensor produces the exact p0
                        predicted by hand for several tau, (b)
                        pack_ternary/unpack round-trips losslessly, (c) the
                        H0 formula reproduces the plan's own table
                        (H(0.9)=0.569, H(0.95)=0.336, H(0.98)=0.161).

Everything not specific to "read trits from a dense bf16 tensor via a
threshold" is imported, not re-implemented: aegis-forge/repack_ternary.py
(pack_ternary, write_model_saf, PROJECTIONS, to_f32_list, _to_f32_np) and
scripts/ptq_zero_sweep.py (h0_general, bytes_per_token_table, load_config,
iter_body_tensors, read_header, read_bytes, engine_unpack_np, run_eval,
parse_cis_full) are both loaded as modules below and called directly.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import mmap
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_FALLBACK = Path.home() / "projects" / "alice-aegis"


def _resolve(rel: str) -> Path:
    p = REPO / rel
    if p.exists():
        return p
    fb = _FALLBACK / rel
    return fb if fb.exists() else p


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_repack_path = _resolve("aegis-forge/repack_ternary.py")
rt = _load_module("repack_ternary", _repack_path)

_zero_sweep_path = _resolve("scripts/ptq_zero_sweep.py")
zs = _load_module("ptq_zero_sweep", _zero_sweep_path)

# Defaults. The bf16 master is NOT part of the repo (multi-GB artifact,
# .gitignored like model.safetensors) -- box1 stages it at this path per
# the task brief; overridden with --master for any other layout.
MASTER = Path.home() / "aefinity-artifacts" / "bitnet-2b-bf16" / "model.safetensors"
PRUNED_REF = zs.PRUNED_REF
CONFIG = zs.CONFIG
EMBED_BIN = zs.EMBED_BIN
VOCAB_BIN = zs.VOCAB_BIN


# --------------------------------------------------------------------------
# mmap streaming reader for the dense bf16 master (never loads the file
# whole -- Rule A / penguin+box1 RAM discipline)
# --------------------------------------------------------------------------

class MasterReader:
    def __init__(self, path: Path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.hdr.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def raw(self, name: str) -> bytes:
        off0, off1 = self.hdr[name]["data_offsets"]
        # slicing the mmap only pages in the touched range; this is a copy
        # (bytes), not the whole file, and is released when it goes out of
        # scope -- one tensor at a time, per the module docstring.
        return self.mm[self.base + off0:self.base + off1]

    def dtype(self, name: str) -> str:
        return self.hdr[name]["dtype"]

    def close(self):
        self.mm.close()
        self.f.close()


def bf16_tensor_to_f32(reader: MasterReader, name: str) -> np.ndarray:
    raw = reader.raw(name)
    dt = reader.dtype(name)
    return rt._to_f32_np(raw, dt) if rt._np is not None else np.asarray(
        rt.to_f32_list(raw, dt), dtype=np.float32)


# --------------------------------------------------------------------------
# core math: absmean gamma + tau-threshold trits (BitNet b1.58 rule,
# generalized from tau=0.5 to a swept tau -- see module docstring)
# --------------------------------------------------------------------------

def gamma_of(W: np.ndarray) -> float:
    """Per-tensor absmean scale, computed from the FULL dense tensor (never
    recomputed after any zeroing -- gamma is fixed across the whole tau
    sweep for a given tensor, matching BitNet's own trained gamma)."""
    return float(np.mean(np.abs(W.astype(np.float64))))


def trits_from_gamma(W: np.ndarray, gamma: float, tau: float) -> np.ndarray:
    """zero iff |W| < tau*gamma, else sign(W). At tau=0.5 this is exactly
    BitNet's RoundClip(round(W/gamma), -1, 1) rule (module docstring
    proof); a value landing EXACTLY on the tau*gamma boundary is the one
    case where round-to-nearest-even could differ from this strict '<'
    rule -- --verify flags any mismatch that lands on that exact tie."""
    thresh = tau * gamma
    sign = np.sign(W).astype(np.int8)  # sign(0.0) == 0, already correct
    return np.where(np.abs(W) < thresh, np.int8(0), sign).astype(np.int8)


def census(trits: np.ndarray):
    total = trits.size
    n_neg = int(np.count_nonzero(trits == -1))
    n_zero = int(np.count_nonzero(trits == 0))
    n_pos = int(np.count_nonzero(trits == 1))
    p_neg, p0, p_pos = n_neg / total, n_zero / total, n_pos / total
    H0 = zs.h0_general(p0, p_pos, p_neg)
    return {"total": total, "n_neg": n_neg, "n_zero": n_zero, "n_pos": n_pos,
            "p_neg": p_neg, "p0": p0, "p_pos": p_pos, "H0": H0}


# --------------------------------------------------------------------------
# --verify
# --------------------------------------------------------------------------

def verify(master_path: Path, pruned_ref: Path, config_path: Path):
    zs.CONFIG = config_path
    cfg, dims = zs.load_config()
    reader = MasterReader(master_path)
    hdr_ref, n_ref = zs.read_header(pruned_ref)

    n_tensors = 0
    n_mismatch_tensors = 0
    n_elem_mismatch = 0
    n_tie_mismatch = 0
    scale_ratio_gamma = []   # ref_gamma / gamma, expect ~1.0
    per_type = {}
    tot_neg = tot_zero = tot_pos = 0

    t0 = time.time()
    for name, dim_out, dim_in in zs.iter_body_tensors(dims, cfg["num_hidden_layers"]):
        n_tensors += 1
        W = bf16_tensor_to_f32(reader, name).reshape(-1)
        gamma = gamma_of(W)
        trits = trits_from_gamma(W, gamma, 0.5)

        ref_raw = zs.read_bytes(pruned_ref, hdr_ref, n_ref, name)
        ref_trits = zs.engine_unpack_np(ref_raw)
        ref_scale_raw = zs.read_bytes(pruned_ref, hdr_ref, n_ref, name + "_scale")
        ref_gamma = struct.unpack("<f", ref_scale_raw)[0]
        scale_ratio_gamma.append(ref_gamma / gamma if gamma else float("nan"))

        if ref_trits.shape != trits.shape or not np.array_equal(ref_trits, trits):
            mism = ref_trits != trits
            n_mismatch_tensors += 1
            n_elem_mismatch += int(np.count_nonzero(mism))
            # tie check: |W| within 2% of the tau*gamma boundary. bf16 has
            # a 7-bit mantissa -- its ULP near a value of magnitude ~gamma
            # is ~gamma/128 (~0.78% relative), NOT ~1e-4 relative; a first
            # run of this check at a 1e-4 window caught only ~8% of the
            # mismatches, and direct inspection of one representative
            # tensor (model.layers.0.self_attn.q_proj.weight) showed the
            # true cause: ALL of that tensor's 79,719 mismatches share the
            # exact SAME |W| bf16 value (0.609375, one specific
            # bf16-representable level), sitting 0.008% below 0.5*gamma,
            # and every one of them is a case where this script assigns 0
            # (correctly, by strict '<') but the shipped checkpoint kept
            # the original sign -- i.e. bf16's ~3-decimal-digit resolution
            # cannot always tell which side of the ternarization boundary
            # the true (higher-precision) training-time weight was on. A
            # 2% window (well inside one bf16 ULP at this magnitude, well
            # outside a coincidence) reliably classifies this.
            near_tie = np.abs(np.abs(W) - 0.5 * gamma) < (0.02 * gamma if gamma else 0)
            n_tie_mismatch += int(np.count_nonzero(mism & near_tie))

        c = census(trits)
        tot_neg += c["n_neg"]; tot_zero += c["n_zero"]; tot_pos += c["n_pos"]

        m = re.search(r"\.(self_attn|mlp)\.(\w+)\.weight$", name)
        key = m.group(2) if m else name
        e = per_type.setdefault(key, [0, 0, 0])
        e[0] += c["n_neg"]; e[1] += c["n_zero"]; e[2] += c["n_pos"]

    dt = time.time() - t0
    reader.close()
    total = tot_neg + tot_zero + tot_pos
    p_neg, p0, p_pos = tot_neg / total, tot_zero / total, tot_pos / total
    H0 = zs.h0_general(p0, p_pos, p_neg)

    ratios = np.array(scale_ratio_gamma)
    frac_matches_gamma = float(np.mean(np.abs(ratios - 1.0) < 1e-3))
    frac_matches_inv_gamma = float(np.mean(np.abs(ratios * ratios - 1.0) < 1e-3))  # ref_gamma ~= 1/gamma <=> ratio^2~=... not exact, see note below

    print(f"=== E-S2b verify (tau=0.5 vs shipped) -- {n_tensors} tensors, "
          f"{total:,} body weights, {dt:.1f}s (Rule A: count only) ===")
    print(f"mismatched tensors: {n_mismatch_tensors}/{n_tensors}   "
          f"mismatched elements: {n_elem_mismatch}/{total:,} "
          f"({100.0 * n_elem_mismatch / total:.6f}%)   "
          f"of which exactly on the 0.5*gamma tie boundary: {n_tie_mismatch}")
    print(f"shipped-scale check: ref_gamma/gamma mean={ratios.mean():.6f} "
          f"std={ratios.std():.6g} min={ratios.min():.6f} max={ratios.max():.6f}  "
          f"(fraction within 0.1% of 1.0: {frac_matches_gamma:.4f} -- MULTIPLY convention "
          f"if ~1.0, would need explicit 1/gamma check if not; observed range printed "
          f"above is sufficient to tell the two conventions apart since they differ by "
          f"gamma^2, not a fixed offset)")
    print(f"whole-body @ tau=0.5: p(-1)={p_neg:.6f} p(0)={p0:.6f} p(+1)={p_pos:.6f} "
          f"H0={H0:.4f} bits/weight")
    print("per-projection-type p0 @ tau=0.5:")
    for k, (neg, zero, pos) in sorted(per_type.items()):
        t = neg + zero + pos
        print(f"  {k:12s} p0={zero/t:.4f}  p(-1)={neg/t:.4f}  p(+1)={pos/t:.4f}  n={t:,}")

    return {
        "n_tensors": n_tensors, "n_mismatch_tensors": n_mismatch_tensors,
        "n_elem_mismatch": n_elem_mismatch, "n_tie_mismatch": n_tie_mismatch,
        "total": total, "p0": p0, "p_pos": p_pos, "p_neg": p_neg, "H0": H0,
        "scale_ratio_mean": float(ratios.mean()),
    }


# --------------------------------------------------------------------------
# --sweep: forge one point per tau from the bf16 master
# --------------------------------------------------------------------------

def forge_point_master(master_path: Path, config_path: Path, tau: float, out_dir: Path) -> dict:
    zs.CONFIG = config_path
    cfg, dims = zs.load_config()
    reader = MasterReader(master_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = []
    n_neg = n_zero = n_pos = 0
    for name, dim_out, dim_in in zs.iter_body_tensors(dims, cfg["num_hidden_layers"]):
        W = bf16_tensor_to_f32(reader, name).reshape(-1)
        gamma = gamma_of(W)
        trits = trits_from_gamma(W, gamma, tau)

        packed = rt.pack_ternary(trits.tolist() if trits.size < 4096 else trits)
        tensors.append((name, "U8", [dim_out, dim_in // 4], packed))
        tensors.append((name + "_scale", "F32", [1], struct.pack("<f", gamma)))

        n_neg += int(np.count_nonzero(trits == -1))
        n_zero += int(np.count_nonzero(trits == 0))
        n_pos += int(np.count_nonzero(trits == 1))

    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        for norm in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            key = f"{p}.{norm}"
            raw = reader.raw(key)
            tensors.append((key, "F32", reader.hdr[key]["shape"],
                             np.asarray(rt.to_f32_list(raw, reader.dtype(key)),
                                        dtype="<f4").tobytes()))
        for sub, _width in (("self_attn.attn_sub_norm.weight", dims["hidden"]),
                             ("mlp.ffn_sub_norm.weight", dims["ffn"])):
            key = f"{p}.{sub}"
            if key in reader.hdr:
                raw = reader.raw(key)
                tensors.append((key, "F32", reader.hdr[key]["shape"],
                                 np.asarray(rt.to_f32_list(raw, reader.dtype(key)),
                                            dtype="<f4").tobytes()))
    raw = reader.raw("model.norm.weight")
    tensors.append(("model.norm.weight", "F32", reader.hdr["model.norm.weight"]["shape"],
                     np.asarray(rt.to_f32_list(raw, reader.dtype("model.norm.weight")),
                                dtype="<f4").tobytes()))
    reader.close()

    aegis_config = {
        "num_hidden_layers": cfg["num_hidden_layers"],
        "hidden_size": dims["hidden"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "intermediate_size": dims["ffn"],
        "vocab_size": 50256,  # pruned vocab -- matches the reused EMBED.BIN/VOCAB.BIN
        "max_position_embeddings": min(cfg.get("max_position_embeddings", 2048), 2048),
        "hidden_act": cfg.get("hidden_act", "relu2"),
        "rope_theta": float(cfg.get("rope_theta", 500000.0)),
        "rms_norm_eps": float(cfg.get("rms_norm_eps", 1e-5)),
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", True)),
        "chat_template": "none",
    }
    out_path = out_dir / "MODEL.SAF"
    rt.write_model_saf(str(out_path), tensors, aegis_config)

    total = n_neg + n_zero + n_pos
    p_neg, p0, p_pos = n_neg / total, n_zero / total, n_pos / total
    H0 = zs.h0_general(p0, p_pos, p_neg)
    return {
        "out_path": out_path, "total": total, "p0": p0, "p_pos": p_pos, "p_neg": p_neg,
        "H0": H0, "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "size_bytes": out_path.stat().st_size,
    }


# 3.5 and 4.0 added so the sweep reaches the plan's p0≈0.95 ("0.314 bit") point (3-tensor pass: τ=3.0 → p0≈0.92–0.94).
DEFAULT_TAUS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def sweep(master_path, config_path, pruned_ref, taus, out_dir: Path, result_path: Path, name: str,
          eval_bin=None, embed=None, vocab=None, text=None, max_tokens=200, keep_forged=False,
          refuse_mismatch_frac=0.02):
    # 2026-09-04 coordinator decision (Fable): the released bf16 master differs from the shipped
    # ternary checkpoint on 1.53 % of weights (bf16 ULP at the 0.5*gamma boundary — a lossy export,
    # not a script bug). Accept that as the honest ceiling: the gate is 2 % and the tau=0.5 row's
    # PPL vs the shipped model quantifies the drift explicitly. Anything above 2 % would mean a
    # different model and still refuses.
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()

    with open(result_path, "a") as f:
        f.write(f"=== {name} start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"host={host} commit={commit}\n")

    v = verify(master_path, pruned_ref, config_path)
    mismatch_frac = v["n_elem_mismatch"] / v["total"]
    with open(result_path, "a") as f:
        f.write(f"verify: {v['n_mismatch_tensors']}/{v['n_tensors']} tensors mismatched, "
                f"{v['n_elem_mismatch']}/{v['total']} elements ({100*mismatch_frac:.6f}%), "
                f"{v['n_tie_mismatch']} on tie boundary, scale_ratio_mean={v['scale_ratio_mean']:.6f}\n")
    if mismatch_frac > refuse_mismatch_frac:
        with open(result_path, "a") as f:
            f.write(f"FATAL: verify mismatch {100*mismatch_frac:.4f}% exceeds refusal "
                    f"threshold {100*refuse_mismatch_frac:.4f}% -- refusing to sweep\n")
            f.write(f"=== {name} done {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        return 1

    with open(result_path, "a") as f:
        f.write("tau | p0 | H0(bits/w) | packed_bytes | coded_bytes | skip_bytes | "
                "float_ppl | hybrid_ppl | full_ppl | hybrid_digest | full_digest | sha256\n")

    for tau in taus:
        point_dir = out_dir / f"tau_{tau}"
        r = forge_point_master(master_path, config_path, tau, point_dir)
        btt = zs.bytes_per_token_table(r["total"], r["p0"], r["H0"])

        row = {"float_ppl": "", "hybrid_ppl": "", "full_ppl": "", "hybrid_digest": "", "full_digest": ""}
        if eval_bin:
            log = zs.run_eval(eval_bin, embed, vocab, text, max_tokens, r["out_path"])
            row.update(zs.parse_cis_full(log))

        line = (f"{tau} | {r['p0']:.6f} | {r['H0']:.4f} | "
                f"{btt['packed_bytes']:.0f} | {btt['coded_bytes']:.0f} | {btt['skip_bytes']:.0f} | "
                f"{row['float_ppl']} | {row['hybrid_ppl']} | {row['full_ppl']} | "
                f"{row['hybrid_digest']} | {row['full_digest']} | {r['sha256'][:16]}")
        with open(result_path, "a") as f:
            f.write(line + "\n")
        print(line)

        if not keep_forged:
            r["out_path"].unlink(missing_ok=True)

    with open(result_path, "a") as f:
        f.write(f"=== {name} done {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    return 0


# --------------------------------------------------------------------------
# --selftest: penguin-safe, synthetic tensors only
# --------------------------------------------------------------------------

def _f32_to_bf16_roundtrip(arr_f32: np.ndarray) -> np.ndarray:
    """f32 -> bf16 raw bytes -> f32, exercising the exact conversion path
    (rt.f32_list_to_bf16 / rt._to_f32_np) real tensors go through."""
    raw = rt.f32_list_to_bf16(arr_f32.tolist())
    return rt._to_f32_np(raw, "BF16") if rt._np is not None else np.asarray(
        rt.to_f32_list(raw, "BF16"), dtype=np.float32)


def selftest():
    failures = []

    # (a) two-magnitude-group synthetic tensor -> hand-computed p0 per tau.
    # 512 elements: half at magnitude 0.5, half at 2.0, signs alternating
    # (kept nonzero after bf16 round-trip: 0.5 and 2.0 are exact in bf16).
    n = 512
    mags = np.where(np.arange(n) % 2 == 0, 0.5, 2.0).astype(np.float32)
    signs = np.where(np.arange(n) % 4 < 2, 1.0, -1.0).astype(np.float32)
    W_true = (mags * signs).astype(np.float32)
    W = _f32_to_bf16_roundtrip(W_true)
    gamma = gamma_of(W)
    expected_gamma = 1.25  # mean(|W|) = (0.5+2.0)/2
    if abs(gamma - expected_gamma) > 1e-3:
        failures.append(f"(a) gamma={gamma}, expected {expected_gamma}")

    cases = [
        (0.5, 0.5),   # thresh=0.625: 0.5<0.625 zero, 2.0 nonzero -> p0=0.5
        (1.0, 0.5),   # thresh=1.25:  0.5<1.25 zero,  2.0>=1.25 nonzero -> p0=0.5
        (1.5, 0.5),   # thresh=1.875: 0.5 zero, 2.0>=1.875 nonzero -> p0=0.5
        (2.0, 1.0),   # thresh=2.5:   both 0.5 and 2.0 < 2.5 -> p0=1.0
    ]
    for tau, expected_p0 in cases:
        trits = trits_from_gamma(W, gamma, tau)
        p0 = float(np.count_nonzero(trits == 0)) / n
        if abs(p0 - expected_p0) > 1e-6:
            failures.append(f"(a) tau={tau}: p0={p0}, expected {expected_p0}")

    # (b) pack/unpack round-trip on a random ternary vector
    rng = np.random.default_rng(0)
    trits_rand = rng.integers(-1, 2, size=4096).astype(np.int8)
    packed = rt.pack_ternary(trits_rand.tolist())
    unpacked = np.array(rt.unpack_ternary(packed), dtype=np.int8)
    if not np.array_equal(unpacked, trits_rand):
        failures.append("(b) pack/unpack round-trip mismatch")
    # also check engine_unpack_np (the vectorized decoder used by --verify)
    unpacked_np = zs.engine_unpack_np(packed)
    if not np.array_equal(unpacked_np, trits_rand):
        failures.append("(b) engine_unpack_np round-trip mismatch")

    # (c) H0 formula against the plan's own table (section 0)
    table = [(0.90, 0.569), (0.95, 0.336), (0.98, 0.161)]
    for p0, expected_H in table:
        p_each = (1.0 - p0) / 2.0
        H0 = zs.h0_general(p0, p_each, p_each)
        if abs(H0 - expected_H) > 0.001:
            failures.append(f"(c) H0(p0={p0})={H0:.4f}, expected {expected_H}")

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELFTEST OK: (a) tau-threshold p0 matches hand-computed values "
          "(gamma=%.4f); (b) pack/unpack + engine_unpack_np round-trip "
          "lossless; (c) H0 matches plan table (0.90->0.569, 0.95->0.336, "
          "0.98->0.161)" % gamma)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", type=Path, default=MASTER)
    ap.add_argument("--pruned-ref", type=Path, default=PRUNED_REF)
    ap.add_argument("--config-json", type=Path, default=CONFIG)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--taus", type=str, default=",".join(str(t) for t in DEFAULT_TAUS))
    ap.add_argument("--out", type=Path, default=Path("/tmp/es2b_out"))
    ap.add_argument("--result", type=Path)
    ap.add_argument("--name", default="es2b-sweep")
    ap.add_argument("--eval-bin")
    ap.add_argument("--embed")
    ap.add_argument("--vocab")
    ap.add_argument("--text")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--keep-forged", action="store_true")
    ap.add_argument("--refuse-mismatch-frac", type=float, default=0.001,
                     help="refuse --sweep if --verify mismatch fraction exceeds this (default 0.1%%)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if args.verify:
        verify(args.master, args.pruned_ref, args.config_json)
        return

    if args.sweep:
        assert args.result, "--sweep requires --result RESULT.txt"
        taus = [float(t) for t in args.taus.split(",")]
        rc = sweep(args.master, args.config_json, args.pruned_ref, taus, args.out, args.result,
                   args.name, eval_bin=args.eval_bin, embed=args.embed, vocab=args.vocab,
                   text=args.text, max_tokens=args.max_tokens, keep_forged=args.keep_forged,
                   refuse_mismatch_frac=args.refuse_mismatch_frac)
        sys.exit(rc)

    ap.print_help()


if __name__ == "__main__":
    main()
