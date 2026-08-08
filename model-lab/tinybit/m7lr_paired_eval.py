#!/usr/bin/env python3
"""m7lr_paired_eval.py — paired multi-window PPL/NLL eval for the M7 LR/WD-cooldown
confound ablation.

NEW FILE. Modifies nothing. Imports only READ-ONLY from model.py (teacher_forced_ppl,
TinyBitConfig, TinyBitModel). train.py, roundtrip_gate.py and m7_final_roundtrip.py are
untouched, so the reproducible round-trip gate is unaffected by construction.

WHY THIS EXISTS
  The published M7 headline (twin 5.513 vs ternary 5.140) is a SINGLE 512-token window
  -- train.py:303-304 caps val_tokens at ctx and build_val_ids() takes one fixed slice at
  offset_frac=0.9. The twin's own consecutive val checks swing ~3% on that same fixed
  window at fixed weights. The claimed effect is ~2x the jitter of the instrument. This
  script replaces n=1 with n=NW disjoint windows and does a PAIRED analysis.

METHOD (fixed before running; see the pre-registration in the run log)
  * Unit of analysis  : the WINDOW (not the token). Tokens inside a window share a prefix
                        and are heavily autocorrelated -- treating 261k tokens as 261k
                        independent observations is pseudo-replication.
  * Statistic         : per-window mean NLL in nats/token.
  * Exactness         : teacher_forced_ppl (model.py:349) returns exp(mean NLL) over
                        exactly N-1 terms. Every window here has the SAME length T=512
                        => 511 predictions each => log(ppl_w) IS the exact per-window mean
                        NLL, and the token-weighted corpus NLL is exactly the unweighted
                        mean of those logs. No Jensen error, no new model code.
  * Windows           : deterministic stride grid, NO RNG. window w = data[w*stride : +T]
                        with stride = len(data)//NW. Asserted disjoint (stride > T).
                        IDENTICAL window list for every arm, scored in ONE process.
  * Tests             : paired sign-flip (randomization) test -- exact under the null of
                        exchangeable signs, no normality assumption. Plus a percentile
                        bootstrap over WINDOWS. scipy is not installed on this box; both
                        are implemented in numpy.

THE SILENT-FAILURE TRAP THIS SCRIPT GUARDS AGAINST
  BitLinear.weight and nn.Linear(bias=False).weight have IDENTICAL names and shapes, so
  loading the fp twin into a bitlinear-defaulted TinyBitConfig returns "<All keys matched
  successfully>" and silently scores the fp model THROUGH ternary fake-quant (measured:
  PPL 4.13 -> 4900.05, a 1200x error with a green load). Therefore this script builds
  every model with TinyBitConfig(**ckpt["config"]) and hard-asserts cfg.linear against the
  checkpoint's stored value. Never construct the config from a configs/*.json here.

Usage:
  m7lr_paired_eval.py --arm LABEL=/path/to.pt [--arm ...] --windows 512 \
                      --out results.json --log run.log
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import TinyBitConfig, TinyBitModel, teacher_forced_ppl  # noqa: E402

T_WIN = 512          # == max_position_embeddings of both checkpoints; do not exceed
BOOT = 10000         # bootstrap resamples over windows
FLIPS = 20000        # sign-flip randomization draws


class Tee:
    """Log to stdout and to a durable file. Same contract as roundtrip_gate.py:41."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_arm(path, log):
    if not os.path.exists(path):
        sys.exit(f"[eval] FATAL: checkpoint not found: {path}")
    t0 = time.time()
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    stored = dict(ck["config"])
    cfg = TinyBitConfig(**stored)
    # Guard the 1200x silent-failure trap described in the docstring.
    assert cfg.linear == stored["linear"], (
        f"linear mismatch: cfg={cfg.linear} stored={stored['linear']}")
    assert cfg.max_position_embeddings >= T_WIN, (
        f"{path}: max_position_embeddings {cfg.max_position_embeddings} < T_WIN {T_WIN}")
    model = TinyBitModel(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=True), None
    model.eval()
    nparam = model.num_params()
    log(f"[eval] loaded {os.path.basename(path)} | step {ck['step']} "
        f"| linear={cfg.linear} | layers {cfg.num_hidden_layers} hidden {cfg.hidden_size} "
        f"inter {cfg.intermediate_size} heads {cfg.num_attention_heads} "
        f"| params {nparam} ({nparam/1e6:.2f}M) | val_hist[-1] "
        f"{ck.get('val_hist', [float('nan')])[-1]:.4f} | {time.time()-t0:.2f}s")
    del ck
    return model, cfg, nparam


# ---------------------------------------------------------------------------
# statistics (numpy only -- scipy is not installed on this box)
# ---------------------------------------------------------------------------
def signflip_p(d, rng, draws=FLIPS):
    """Two-sided paired randomization test. Null: the sign of each per-window
    difference is exchangeable. Exact-in-the-limit, no distributional assumption."""
    d = np.asarray(d, dtype=np.float64)
    obs = abs(d.mean())
    n = len(d)
    hits = 0
    block = 2000
    done = 0
    while done < draws:
        b = min(block, draws - done)
        s = rng.choice(np.array([-1.0, 1.0]), size=(b, n))
        hits += int((np.abs((s * d).mean(axis=1)) >= obs - 1e-15).sum())
        done += b
    return (hits + 1) / (draws + 1)


def boot_idx(n, rng, reps=BOOT):
    return rng.integers(0, n, size=(reps, n))


def ci_from(vals, lo=2.5, hi=97.5):
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def holm(pairs):
    """pairs: list of (name, p). Returns {name: adjusted_p} (Holm-Bonferroni)."""
    order = sorted(pairs, key=lambda kv: kv[1])
    m = len(order)
    out, run = {}, 0.0
    for i, (name, p) in enumerate(order):
        adj = min(1.0, (m - i) * p)
        run = max(run, adj)           # enforce monotonicity
        out[name] = run
    return out


def describe(log, name, d, rng, bidx):
    d = np.asarray(d, dtype=np.float64)
    mean = float(d.mean())
    wins = int((d > 0).sum())
    bmeans = d[bidx].mean(axis=1)
    lo, hi = ci_from(bmeans)
    p = signflip_p(d, rng)
    log(f"  {name:22s} mean dNLL {mean:+.5f} nats/tok | 95% CI [{lo:+.5f}, {hi:+.5f}] "
        f"| favours-2nd {wins}/{len(d)} | sign-flip p {p:.5f}")
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "wins": wins,
            "n": int(len(d)), "p_signflip": p}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="LABEL=PATH", help="repeatable; e.g. --arm A=checkpoints/x.pt")
    ap.add_argument("--data", default=os.path.join(HERE, "valid_8k.bin"))
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260729, help="stats RNG only; windows are deterministic")
    ap.add_argument("--out", required=True, help="JSON sidecar with the full NLL matrix")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    log = Tee(args.log)
    torch.set_num_threads(args.threads)
    try:
        os.nice(10)
    except Exception:
        pass

    arms = []
    for spec in args.arm:
        if "=" not in spec:
            sys.exit(f"[eval] --arm needs LABEL=PATH, got {spec!r}")
        lbl, path = spec.split("=", 1)
        arms.append((lbl, os.path.abspath(path)))
    labels = [a[0] for a in arms]
    if len(set(labels)) != len(labels):
        sys.exit("[eval] duplicate arm labels")

    log("=" * 78)
    log(f"[eval] m7lr_paired_eval  UTC {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log(f"[eval] torch {torch.__version__} | numpy {np.__version__} | threads {args.threads}")
    log(f"[eval] argv: {' '.join(sys.argv)}")

    data = np.memmap(args.data, dtype=np.uint16, mode="r")
    NW = args.windows
    stride = len(data) // NW
    assert stride > T_WIN, (
        f"stride {stride} <= T_WIN {T_WIN}: windows would OVERLAP at NW={NW}")
    assert (NW - 1) * stride + T_WIN <= len(data), "last window runs off the end"
    wins = [torch.from_numpy(np.asarray(data[w * stride:w * stride + T_WIN],
                                        dtype=np.int64)) for w in range(NW)]

    # The in-training tripwire window both published runs printed, for the record.
    anchor = int(len(data) * 0.9)
    overlap = [w for w in range(NW)
               if not (w * stride + T_WIN <= anchor or w * stride >= anchor + T_WIN)]
    log(f"[eval] data {args.data} | {len(data)} tokens | NW {NW} | stride {stride} "
        f"| T {T_WIN} | scored positions {NW * (T_WIN - 1)}")
    log(f"[eval] deterministic stride grid, NO RNG in window selection; disjoint (stride > T)")
    log(f"[eval] in-training anchor window = tokens [{anchor}:{anchor+T_WIN}] "
        f"(build_val_ids offset_frac=0.9); eval windows overlapping it: {overlap or 'NONE'}")

    # ---- score every arm over the IDENTICAL window list, in one process ----
    nll = {}
    meta = {}
    for lbl, path in arms:
        model, cfg, nparam = load_arm(path, log)
        t0 = time.time()
        v = np.empty(NW, dtype=np.float64)
        for w in range(NW):
            v[w] = math.log(teacher_forced_ppl(model, wins[w]))
        dt = time.time() - t0
        nll[lbl] = v
        corpus = float(np.exp(v.mean()))
        meta[lbl] = {"path": path, "linear": cfg.linear, "params": nparam,
                     "corpus_ppl": corpus, "mean_nll": float(v.mean()),
                     "sec_per_window": dt / NW}
        log(f"[eval] arm {lbl:10s} corpus PPL {corpus:.4f} | mean NLL {v.mean():.5f} "
            f"| {dt:.1f}s ({dt/NW:.3f} s/window)")
        del model

    log("")
    log("[eval] --- corpus-level (exp of the token-weighted mean NLL over all windows) ---")
    for lbl in labels:
        log(f"  {lbl:10s} PPL {meta[lbl]['corpus_ppl']:.4f}  "
            f"NLL {meta[lbl]['mean_nll']:.5f}  linear={meta[lbl]['linear']}  "
            f"params={meta[lbl]['params']}")

    # ---- paired contrasts -------------------------------------------------
    rng = np.random.default_rng(args.seed)
    bidx = boot_idx(NW, rng)
    have = set(labels)
    contrasts = []
    # (name, left, right): d = NLL[left] - NLL[right]; positive => RIGHT arm is better
    for name, l, r in (("d_gap  (A-T)", "A", "T"),
                       ("d_tokens (A-H)", "A", "H"),
                       ("d_cool (H-K)", "H", "K"),
                       ("d_cool_raw (A-K)", "A", "K"),
                       ("d_resid (K-T)", "K", "T")):
        if l in have and r in have:
            contrasts.append((name, l, r))

    log("")
    log("[eval] --- paired per-window contrasts (positive dNLL => the SECOND arm is better) ---")
    stats = {}
    for name, l, r in contrasts:
        stats[name] = describe(log, name, nll[l] - nll[r], rng, bidx)
        stats[name]["left"], stats[name]["right"] = l, r

    ph = [(n, s["p_signflip"]) for n, s in stats.items()]
    adj = holm(ph)
    log("")
    log(f"[eval] Holm-adjusted p across the {len(ph)} contrasts:")
    for n in stats:
        stats[n]["p_holm"] = adj[n]
        log(f"  {n:22s} p_raw {stats[n]['p_signflip']:.5f} -> p_holm {adj[n]:.5f}")

    # ---- the headline ratio ----------------------------------------------
    ratios = {}
    if {"A", "H", "K", "T"} <= have:
        d_gap = nll["A"] - nll["T"]
        d_cool = nll["H"] - nll["K"]
        d_cool_raw = nll["A"] - nll["K"]
        d_tok = nll["A"] - nll["H"]
        log("")
        log("[eval] --- HEADLINE RATIO: fraction of the published gap explained ---")
        for rn, num in (("R_cooldown_token_controlled", d_cool),
                        ("R_cooldown_plus_tokens", d_cool_raw),
                        ("R_extra_tokens_alone", d_tok)):
            bn = num[bidx].mean(axis=1)
            bd = d_gap[bidx].mean(axis=1)
            denom_lo, denom_hi = ci_from(bd)
            point = float(num.mean() / d_gap.mean()) if d_gap.mean() != 0 else float("nan")
            if denom_lo <= 0.0 <= denom_hi:
                log(f"  {rn:30s} = {point:+.4f}  [RATIO CI SUPPRESSED: the d_gap "
                    f"bootstrap CI {denom_lo:+.5f}..{denom_hi:+.5f} spans zero, so the "
                    f"ratio is not interpretable]")
                ratios[rn] = {"point": point, "ci_lo": None, "ci_hi": None,
                              "interpretable": False}
            else:
                rlo, rhi = ci_from(bn / bd)
                log(f"  {rn:30s} = {point:+.4f}  95% CI [{rlo:+.4f}, {rhi:+.4f}]")
                ratios[rn] = {"point": point, "ci_lo": rlo, "ci_hi": rhi,
                              "interpretable": True}

    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv, "windows": NW, "stride": stride, "T": T_WIN,
        "data": args.data, "data_tokens": int(len(data)),
        "anchor_overlap_windows": overlap,
        "threads": args.threads, "torch": torch.__version__,
        "arms": meta,
        "per_window_nll": {k: v.tolist() for k, v in nll.items()},
        "contrasts": stats,
        "ratios": ratios,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    log("")
    log(f"[eval] full {NW}-window NLL matrix + stats written to {args.out}")
    log(f"[eval] re-runnable without re-scoring: the per_window_nll arrays are in that JSON")
    log("[eval] done")


if __name__ == "__main__":
    main()
