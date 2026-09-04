#!/usr/bin/env python3
"""E-S2 -- post-hoc zero-threshold sweep on BitNet-2B (LANE E-S2).

Plan: state/reports/2026-09-04-SUBBIT-TERNARY-PLAN.md section 2.

SCOPE-REDUCTION FINDING (found before any heavy compute ran -- mirroring
CA1's finding for Falcon-E-1B, docs/2026-09-02-CA1-ptq-collapse-falcon-e-1b.md):
`model.safetensors` (1.18 GB, on penguin, the file the plan calls the "BitNet
bf16 master weights") is NOT a dense pre-quantization checkpoint. It is the
shipped, already-QAT-trained BitNet-b1.58-2B-4T checkpoint in HF1BitLLM
2-bit-packed layout: U8 codes {0,1,2}={-1,0,+1} packed along dim 0 in out/4
row-blocks, one BF16 absmean scale per tensor (config.json:
quantization_config.quant_method=bitnet, quantization_mode=offline).

Verified two ways by `--verify` below, over ALL 210 body tensors (30 layers
x 7 projections), not a spot check:
  1. every unpacked 2-bit code is in {0,1,2}, never 3 -- exactly ternary,
     no residual continuous information survives.
  2. re-encoding these SAME trits into the engine's packing reproduces
     `aegis_pruned_model.safetensors` (already on penguin) tensor-for-tensor,
     0 mismatched elements out of ~2.1B, AND the per-tensor scale matches
     exactly (both files use the engine's MULTIPLY convention here -- this
     checkpoint is not the onebitllms/Falcon-E DIVISION convention that bit
     repack_ternary.py's scalar_f32_scale() docstring warns about; verified
     empirically, not assumed).
So this file and the shipped engine model encode IDENTICAL weights, just
different containers. There is no latent to threshold-sweep -- this is the
plan's own caveat ("on BitNet-2B, per tensor, rank |latent| is unavailable")
turning out to apply to the file the plan named as the workaround too.

BitNet's default rule (Wang et al. 2023, "BitNet b1.58"): absmean
quantization gamma = mean(|W|) (per tensor), W~ = RoundClip(W/gamma, -1, 1).
For the SIGN/ZERO decision alone (all a ternary weight retains), this is
algebraically identical to a hard threshold at 0.5*gamma: zero iff
|W/gamma| < 0.5, else sign(W) -- round-to-nearest of a value in [0, 1.5) can
only land on 0 or 1, and the clip only matters above 1.5, where sign(W) is
already what rounding-to-1 gives. So tau=0.5 in this script's tau*gamma
threshold parameterization IS BitNet's default, and it is EXACTLY what
produced the shipped body's p0 (measured below) -- confirmed by the exact
tensor-match above, not a resemblance.

ADAPTED METHOD for points beyond the shipped baseline: since no continuous
latent survives quantization, "raising the threshold" past tau=0.5 cannot be
computed on this artifact -- there is nothing left to re-threshold. This
script instead runs the honest substitute for what the plan's own section-0
math table is actually about: magnitude-blind (uniform-random, independent
per weight, seeded for reproducibility) pruning of the shipped nonzero trits
down to target zero-fractions p0 lifted directly from that table, including
its two headline numbers: p0=0.9535 (0.314 bit/weight) and p0=0.9803
(0.158 bit/weight). Magnitude-blind pruning cannot protect "important"
weights, so its PPL cost at a given p0 is a pessimistic (worst-case) upper
bound relative to any magnitude- or loss-aware pruning scheme -- flagged
wherever these numbers are used.

Modes:
    --verify         run the exact-match check against aegis_pruned_model.safetensors
                      and print the whole-body p0/p_pos/p_neg/H0 census (no
                      forging, no eval; this is the "census" step).
    --forge P0 OUT    prune the body to target zero-fraction P0 (magnitude-
                      blind Bernoulli, seed derived from P0) and write
                      OUT/MODEL.SAF (reuses aegis-forge/repack_ternary.py's
                      pack_ternary/write_safetensors, EMBED.BIN/VOCAB.BIN are
                      NOT regenerated -- reuse the existing
                      aegis-forge/embed.bin + aegis-forge/vocab.bin, which
                      this prune never touches).
    --sweep OUT_DIR RESULT_TXT [--eval EVAL_BIN EMBED VOCAB TEXT [MAX_TOKENS]]
                      the full 8-point sweep: census baseline + 7 pruned
                      points, appending one RESULT.txt row after EACH point
                      (append-only, per docs/LEGS.md). Forges every point;
                      runs aegis-eval only if --eval is given (box1 has the
                      AVX2 + 5.7 GB this needs; penguin does not -- see
                      Rule A/box RAM notes in the leg script).

Every point also prints the E-S3 bytes/token table (packed / entropy-coded /
zero-skip run-length) inline -- state/reports plan section 3; bytes_per_token.py
does not exist yet on this branch (E-S3 has not run), so the three formulas
are re-implemented here directly from the plan's own definitions rather than
imported.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# The large artifacts (model.safetensors, aegis_pruned_model.safetensors,
# aegis-forge/{embed,vocab}.bin) are .gitignored (skill rule: never commit
# regenerable weights) and so are NOT present in a fresh worktree checkout --
# only in whichever working copy actually staged them. Resolve against the
# worktree first, then fall back to the main checkout (penguin dev/smoke
# path); a box leg overrides all of these via --model/--pruned-ref/--embed/
# --vocab/--config since box1 stages the master at
# ~/aefinity-artifacts/bitnet_2b_master.safetensors, a different tree entirely.
_FALLBACK = Path.home() / "projects" / "alice-aegis"


def _resolve(rel: str) -> Path:
    p = REPO / rel
    if p.exists():
        return p
    fb = _FALLBACK / rel
    return fb if fb.exists() else p  # keep REPO path so errors name the intended location


SRC = _resolve("model.safetensors")
PRUNED_REF = _resolve("aegis_pruned_model.safetensors")
CONFIG = _resolve("config.json")
EMBED_BIN = _resolve("aegis-forge/embed.bin")
VOCAB_BIN = _resolve("aegis-forge/vocab.bin")

# --- reuse the proven forge helpers instead of reinventing packing ---
_repack_ternary_path = REPO / "aegis-forge" / "repack_ternary.py"
if not _repack_ternary_path.exists():
    _repack_ternary_path = _FALLBACK / "aegis-forge" / "repack_ternary.py"
_spec = importlib.util.spec_from_file_location("repack_ternary", _repack_ternary_path)
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def read_header(path: Path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, n


def read_bytes(path: Path, hdr, header_len: int, name: str) -> bytes:
    off0, off1 = hdr[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + off0)
        return f.read(off1 - off0)


def engine_unpack_np(raw: bytes) -> np.ndarray:
    """Engine 2-bit packing (4 values/byte, LSB-first) -> int8 {-1,0,1},
    vectorized. Matches repack_ternary.py's ENGINE_VALUE = {0:0,1:1,2:-1}
    (code 3 undefined, never emitted); repack_ternary.py only ships a
    pure-python unpack_ternary(), too slow for a 2B-parameter model."""
    b = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(b.size * 4, dtype=np.int8)
    lut = np.array([0, 1, -1, 0], dtype=np.int8)
    for shift in range(4):
        out[shift::4] = lut[(b >> (2 * shift)) & 3]
    return out


def load_config():
    with open(CONFIG) as f:
        cfg = json.load(f)
    hidden = cfg["hidden_size"]
    kv_dim = (hidden // cfg["num_attention_heads"]) * cfg["num_key_value_heads"]
    dims = {"hidden": hidden, "kv": kv_dim, "ffn": cfg["intermediate_size"]}
    return cfg, dims


def h0_general(p0: float, p_pos: float, p_neg: float) -> float:
    """-sum p*log2(p), 0*log2(0):=0. General form (task instruction): use
    this always, not the even-split h(p0)+(1-p0) shortcut, since the
    magnitude-blind prune preserves the +/- ratio but real hardware/roundoff
    can still leave it slightly uneven."""
    tot = 0.0
    for p in (p0, p_pos, p_neg):
        if p > 0.0:
            tot -= p * math.log2(p)
    return tot


def iter_body_tensors(dims, num_layers):
    """Yields (tensor_name, dim_out, dim_in) for all 30*7 body projections,
    in the same order/definition as aegis-forge/repack_ternary.py's
    PROJECTIONS list (reused, not re-typed)."""
    for i in range(num_layers):
        p = f"model.layers.{i}"
        for proj, dk_out, dk_in in rt.PROJECTIONS:
            yield f"{p}.{proj}.weight", dims[dk_out], dims[dk_in]


def verify_and_census():
    """--verify: exact tensor-for-tensor match against the shipped engine
    model (the real "compare trit histograms to aegis_pruned_model" check,
    done at full element-wise fidelity, not just histograms) + the
    whole-body p0/p_pos/p_neg/H0 census that stands in for E-S1 (not yet
    pushed) H0 on this artifact."""
    cfg, dims = load_config()
    hdr_src, n_src = read_header(SRC)
    hdr_ref, n_ref = read_header(PRUNED_REF)

    n_neg = n_zero = n_pos = 0
    n_mismatch_tensors = 0
    n_elem_mismatch = 0
    scale_mismatch = 0
    per_type = {}

    t0 = time.time()
    for name, dim_out, dim_in in iter_body_tensors(dims, cfg["num_hidden_layers"]):
        raw = read_bytes(SRC, hdr_src, n_src, name)
        trits = rt.unpack_hf1bitllm(raw, dim_out, dim_in)  # numpy int8, row-major flat

        scale_raw = read_bytes(SRC, hdr_src, n_src, name + "_scale")
        gamma = rt.to_f32_list(scale_raw, hdr_src[name + "_scale"]["dtype"])[0]

        # cross-check vs the shipped engine artifact (same values, engine packing)
        ref_raw = read_bytes(PRUNED_REF, hdr_ref, n_ref, name)
        ref_trits = engine_unpack_np(ref_raw)
        if ref_trits.shape != trits.shape or not np.array_equal(ref_trits, trits):
            n_mismatch_tensors += 1
            n_elem_mismatch += int(np.count_nonzero(ref_trits != trits))
        ref_scale_raw = read_bytes(PRUNED_REF, hdr_ref, n_ref, name + "_scale")
        ref_gamma = struct.unpack("<f", ref_scale_raw)[0]
        if abs(ref_gamma - gamma) > 1e-6 * max(1.0, abs(gamma)):
            scale_mismatch += 1

        neg = int(np.count_nonzero(trits == -1))
        zero = int(np.count_nonzero(trits == 0))
        pos = int(np.count_nonzero(trits == 1))
        n_neg += neg
        n_zero += zero
        n_pos += pos

        m = re.search(r"\.(self_attn|mlp)\.(\w+)\.weight$", name)
        key = m.group(2) if m else name
        e = per_type.setdefault(key, [0, 0, 0])
        e[0] += neg
        e[1] += zero
        e[2] += pos

    dt = time.time() - t0
    total = n_neg + n_zero + n_pos
    p_neg, p0, p_pos = n_neg / total, n_zero / total, n_pos / total
    H0 = h0_general(p0, p_pos, p_neg)

    print(f"=== E-S2 verify+census ({total:,} body weights, {dt:.1f}s on penguin -- "
          f"count only, Rule A: no timing figure reported) ===")
    print(f"exact-match vs {PRUNED_REF.name}: "
          f"{n_mismatch_tensors} mismatched tensors, {n_elem_mismatch} mismatched elements "
          f"(out of {total:,}), {scale_mismatch} scale mismatches")
    print(f"whole-body: p(-1)={p_neg:.6f} p(0)={p0:.6f} p(+1)={p_pos:.6f}  H0={H0:.4f} bits/weight")
    print("per-projection-type p0:")
    for k, (neg, zero, pos) in sorted(per_type.items()):
        t = neg + zero + pos
        print(f"  {k:12s} p0={zero/t:.4f}  p(-1)={neg/t:.4f}  p(+1)={pos/t:.4f}  n={t:,}")
    return {
        "total": total, "p_neg": p_neg, "p0": p0, "p_pos": p_pos, "H0": H0,
        "n_mismatch_tensors": n_mismatch_tensors, "n_elem_mismatch": n_elem_mismatch,
        "scale_mismatch": scale_mismatch,
    }


def bytes_per_token_table(total_trits: int, p0: float, H0: float) -> dict:
    """E-S3 (state/reports plan section 3), re-implemented directly from its
    formulas since bytes_per_token.py has not been pushed (E-S3 unrun):
    a decode step reads every body weight matrix once (batch=1 matvec).
      packed:   total_trits/4 bytes (fixed, independent of p0 -- the point
                of the comparison)
      coded:    total_trits*H0/8 bytes (order-0 Shannon estimate; an actual
                rANS/arithmetic decode would also need to touch these bytes
                per E-S1's remit, not re-run here)
      skip:     only nonzero positions read; bits/nonzero ~= log2(1/(1-p0))+1
                (run-length index cost + 1 sign bit), per the plan's own
                formula
    """
    packed = total_trits / 4
    coded = total_trits * H0 / 8
    n_nonzero = total_trits * (1.0 - p0)
    if p0 < 1.0:
        bits_per_nonzero = math.log2(1.0 / (1.0 - p0)) + 1.0
    else:
        bits_per_nonzero = 0.0
    skip = n_nonzero * bits_per_nonzero / 8
    return {"packed_bytes": packed, "coded_bytes": coded, "skip_bytes": skip,
            "bits_per_nonzero": bits_per_nonzero}


def forge_point(p0_target, out_dir: Path, seed_base: int = 20260904) -> dict:
    """Magnitude-blind Bernoulli prune of the shipped nonzero trits down to
    p0_target, then forge MODEL.SAF (engine packing + config metadata,
    reusing EMBED.BIN/VOCAB.BIN untouched). p0_target=None means the
    baseline point: pass every tensor's trits through UNCHANGED (q=0
    always) -- passing the global average p0 here instead would prune any
    tensor whose OWN p0 sits below that average, silently turning the
    "reproduce the shipped baseline" point into a real prune. Otherwise
    p0_target below a given tensor's own p0 is a no-op for that tensor
    (this script only ADDS zeros, per the plan's question "if we force MORE
    zeros...", and per-tensor -- not a global cross-tensor zero budget)."""
    cfg, dims = load_config()
    hdr_src, n_src = read_header(SRC)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_key = 0 if p0_target is None else int(round(p0_target * 1_000_000))
    rng = np.random.default_rng(seed_base + seed_key)

    tensors = []
    n_neg = n_zero = n_pos = 0
    for name, dim_out, dim_in in iter_body_tensors(dims, cfg["num_hidden_layers"]):
        raw = read_bytes(SRC, hdr_src, n_src, name)
        trits = rt.unpack_hf1bitllm(raw, dim_out, dim_in).copy()
        scale_raw = read_bytes(SRC, hdr_src, n_src, name + "_scale")
        gamma = rt.to_f32_list(scale_raw, hdr_src[name + "_scale"]["dtype"])[0]

        if p0_target is not None:
            p0_here = float(np.count_nonzero(trits == 0)) / trits.size
            if p0_target > p0_here:
                q = (p0_target - p0_here) / (1.0 - p0_here)  # prune prob among current nonzeros
                nz_mask = trits != 0
                drop = nz_mask & (rng.random(trits.size) < q)
                trits[drop] = 0
            # else: p0_target <= this tensor's own p0 -- nothing to add here

        packed = rt.pack_ternary(trits.tolist() if trits.size < 4096 else trits)
        tensors.append((name, "U8", [dim_out, dim_in // 4], packed))
        tensors.append((name + "_scale", "F32", [1], struct.pack("<f", float(gamma))))

        n_neg += int(np.count_nonzero(trits == -1))
        n_zero += int(np.count_nonzero(trits == 0))
        n_pos += int(np.count_nonzero(trits == 1))

    # norms + (tied embeddings -> no lm_head.weight tensor needed, matches
    # aegis_pruned_model.safetensors's own tensor set exactly)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        for norm in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            raw = read_bytes(SRC, hdr_src, n_src, f"{p}.{norm}")
            tensors.append((f"{p}.{norm}", "F32",
                             hdr_src[f"{p}.{norm}"]["shape"],
                             np.asarray(rt.to_f32_list(raw, hdr_src[f"{p}.{norm}"]["dtype"]),
                                        dtype="<f4").tobytes()))
        for sub, width in (("self_attn.attn_sub_norm.weight", dims["hidden"]),
                           ("mlp.ffn_sub_norm.weight", dims["ffn"])):
            key = f"{p}.{sub}"
            if key in hdr_src:
                raw = read_bytes(SRC, hdr_src, n_src, key)
                tensors.append((key, "F32", hdr_src[key]["shape"],
                                 np.asarray(rt.to_f32_list(raw, hdr_src[key]["dtype"]),
                                            dtype="<f4").tobytes()))
    raw = read_bytes(SRC, hdr_src, n_src, "model.norm.weight")
    tensors.append(("model.norm.weight", "F32", hdr_src["model.norm.weight"]["shape"],
                     np.asarray(rt.to_f32_list(raw, hdr_src["model.norm.weight"]["dtype"]),
                                dtype="<f4").tobytes()))

    aegis_config = {
        "num_hidden_layers": cfg["num_hidden_layers"],
        "hidden_size": dims["hidden"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "intermediate_size": dims["ffn"],
        "vocab_size": 50256,  # pruned vocab, matches aegis-forge/{embed,vocab}.bin on disk
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
    H0 = h0_general(p0, p_pos, p_neg)
    return {
        "out_path": out_path, "total": total, "p0": p0, "p_pos": p_pos, "p_neg": p_neg,
        "H0": H0, "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "size_bytes": out_path.stat().st_size,
    }


def run_eval(eval_bin, embed, vocab, text, max_tokens, model_path) -> str:
    out = subprocess.run(
        [eval_bin, str(model_path), embed, vocab, text, str(max_tokens), "--cis-full"],
        capture_output=True, text=True, timeout=7200,  # 200-token --cis-full on a 2B model ≈ 50 min single-thread on box1
    )
    return out.stdout + out.stderr


def parse_cis_full(log: str) -> dict:
    def grab(pat):
        m = re.search(pat, log)
        return m.group(1) if m else None
    return {
        "float_ppl": grab(r"float\s+PPL[^:]*:\s*([0-9.]+)"),
        "hybrid_ppl": grab(r"hybrid-int PPL[^:]*:\s*([0-9.]+)"),
        "full_ppl": grab(r"full-int\s+PPL[^:]*:\s*([0-9.]+)"),
        "hybrid_digest": grab(r"hybrid-int argmax digest[^:]*:\s*(0x[0-9A-Fa-f]+)"),
        "full_digest": grab(r"full-int\s+argmax digest[^:]*:\s*(0x[0-9A-Fa-f]+)"),
    }


# The plan's own H(p0) table (section 0), reused verbatim as the sweep's
# target points instead of the un-executable tau*gamma latent threshold --
# see module docstring "ADAPTED METHOD".
SWEEP_P0_TARGETS = [None, 0.70, 0.80, 0.90, 0.95, 0.9535, 0.97, 0.9803]
# None = point 1 = the shipped baseline (p0=0.4962, tau=0.5), no pruning.


def main():
    global SRC, PRUNED_REF, CONFIG, EMBED_BIN, VOCAB_BIN
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help=f"override SRC (default {SRC})")
    ap.add_argument("--pruned-ref", type=Path, help=f"override PRUNED_REF (default {PRUNED_REF})")
    ap.add_argument("--config-json", type=Path, help=f"override CONFIG (default {CONFIG})")
    ap.add_argument("--embed-bin", type=Path, help=f"override EMBED_BIN (default {EMBED_BIN})")
    ap.add_argument("--vocab-bin", type=Path, help=f"override VOCAB_BIN (default {VOCAB_BIN})")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--forge", type=float, metavar="P0")
    ap.add_argument("--out", type=Path, default=Path("/tmp/es2_out"))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--result", type=Path, help="RESULT.txt to append to (leg contract)")
    ap.add_argument("--name", default="es2-sweep")
    ap.add_argument("--eval-bin")
    ap.add_argument("--embed")
    ap.add_argument("--vocab")
    ap.add_argument("--text")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--keep-forged", action="store_true",
                     help="don't delete MODEL.SAF after eval (default: delete, per plan tidiness note)")
    args = ap.parse_args()

    if args.model:
        SRC = args.model
    if args.pruned_ref:
        PRUNED_REF = args.pruned_ref
    if args.config_json:
        CONFIG = args.config_json
    if args.embed_bin:
        EMBED_BIN = args.embed_bin
    if args.vocab_bin:
        VOCAB_BIN = args.vocab_bin

    if args.verify:
        verify_and_census()
        return

    if args.forge is not None:
        r = forge_point(args.forge, args.out)
        print(json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}, indent=2))
        return

    if args.sweep:
        assert args.result, "--sweep requires --result RESULT.txt"
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
        with open(args.result, "a") as f:
            f.write(f"=== {args.name} start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                    f"host={host} commit={commit}\n")
            f.write("tau_equiv | p0_target | p0_achieved | H0(bits/w) | packed_bytes | coded_bytes | "
                    "skip_bytes | float_ppl | hybrid_ppl | full_ppl | hybrid_digest | full_digest | sha256\n")

        base_census = verify_and_census()

        for idx, p0_target in enumerate(SWEEP_P0_TARGETS):
            point_dir = args.out / f"point_{idx}"
            if p0_target is None:
                r = forge_point(None, point_dir)  # pass-through, reproduces the shipped baseline exactly
                tau_label = "0.5(baseline)"
            else:
                r = forge_point(p0_target, point_dir)
                tau_label = "n/a(magnitude-blind)"

            btt = bytes_per_token_table(r["total"], r["p0"], r["H0"])
            row = {
                "float_ppl": "", "hybrid_ppl": "", "full_ppl": "",
                "hybrid_digest": "", "full_digest": "",
            }
            if args.eval_bin:
                log = run_eval(args.eval_bin, args.embed, args.vocab, args.text,
                                args.max_tokens, r["out_path"])
                row.update(parse_cis_full(log))

            line = (f"{tau_label} | {p0_target} | {r['p0']:.6f} | {r['H0']:.4f} | "
                    f"{btt['packed_bytes']:.0f} | {btt['coded_bytes']:.0f} | {btt['skip_bytes']:.0f} | "
                    f"{row['float_ppl']} | {row['hybrid_ppl']} | {row['full_ppl']} | "
                    f"{row['hybrid_digest']} | {row['full_digest']} | {r['sha256'][:16]}")
            with open(args.result, "a") as f:
                f.write(line + "\n")
            print(line)

            if not args.keep_forged:
                r["out_path"].unlink(missing_ok=True)

        with open(args.result, "a") as f:
            f.write(f"=== {args.name} done {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
