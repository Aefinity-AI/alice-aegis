#!/usr/bin/env python3
"""m7_paired_eval.py — paired multi-window evaluation of tinybit checkpoints.

WHY THIS EXISTS
---------------
Every val_ppl number in the M7/M7a runs came from a SINGLE FIXED 512-token
passage (train.py:303-304 -> build_val_ids(..., min(val_tokens=512, ctx=512))),
which scores 511 predictions — about 0.01% of valid_8k.bin. The headline
"ternary is 6.8% better than the twin" rests on comparing two such points.
That is far too thin to carry a company-direction decision.

This script scores each checkpoint over MANY disjoint windows spanning the
whole validation file, on the SAME windows in the SAME order for every model,
and reports:

  * token-weighted corpus perplexity   <- the headline number (correct pooling)
  * per-window perplexities            <- for paired statistics
  * paired mean NLL difference + t     <- is the gap bigger than window noise?

Models with DIFFERENT architectures are handled correctly: each checkpoint is
rebuilt from its own stored config, so the twin (6 layers / 256 hidden) and the
ternary (7 layers / 384 hidden) are both loaded faithfully. The comparison is
paired because both see identical token windows, not because they share shape.

USAGE
    python3 m7_paired_eval.py \
        --ckpt twin=checkpoints/m7a_twin.pt \
        --ckpt twin_cooled=checkpoints/m7a_twin_cooled.pt \
        --ckpt ternary=checkpoints/m7_ternary.pt \
        --windows 96 --baseline twin --compare ternary
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

from model import TinyBitModel, TinyBitConfig  # noqa: E402


def load_model(path):
    """Rebuild a model from the architecture stored inside its own checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TinyBitConfig(**ckpt["config"])
    model = TinyBitModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {
        "path": path,
        "step": ckpt.get("step"),
        "params": model.num_params(),
        "linear": cfg.linear,
        "layers": cfg.num_hidden_layers,
        "hidden": cfg.hidden_size,
        "inter": cfg.intermediate_size,
        "heads": cfg.num_attention_heads,
        "vocab": cfg.vocab_size,
    }
    del ckpt
    return model, cfg, meta


def build_windows(bin_path, n_windows, ctx, skip_frac=0.0):
    """n_windows DISJOINT (ctx+1)-token windows spread evenly over the file.

    Evenly spaced rather than contiguous-from-zero so the sample covers the
    whole distribution of the validation corpus, not just its head.
    """
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total = len(data)
    start_at = int(total * skip_frac)
    usable = total - start_at - (ctx + 1)
    if usable <= 0:
        sys.exit(f"[eval] FATAL: {bin_path} too small for {n_windows} windows of {ctx+1}")
    stride = usable // n_windows
    if stride < ctx + 1:
        sys.exit(f"[eval] FATAL: {n_windows} windows of {ctx+1} tokens would OVERLAP "
                 f"(stride {stride}). Reduce --windows or use a larger file.")
    out = []
    for k in range(n_windows):
        s = start_at + k * stride
        out.append(np.asarray(data[s:s + ctx + 1], dtype=np.int64))
    return out, {"file_tokens": int(total), "stride": int(stride),
                 "scored_per_window": ctx, "total_scored": int(n_windows * ctx)}


@torch.no_grad()
def score(model, vocab_size, windows):
    """Return (per_window_ppl[], sum_nll, n_tokens). Token-weighted pooling."""
    per_window, sum_nll, n_tok = [], 0.0, 0
    for w in windows:
        ids = torch.from_numpy(w)
        x = ids[:-1].unsqueeze(0)
        y = ids[1:].unsqueeze(0)
        logits = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1), reduction="sum")
        n = y.numel()
        per_window.append(math.exp(float(nll) / n))
        sum_nll += float(nll)
        n_tok += n
    return per_window, sum_nll, n_tok


def sign_flip_p(d, draws=20000, seed=1337):
    """Paired sign-flip permutation test on per-window differences.

    Under the null (no systematic difference) the sign of each window's
    difference is exchangeable. Two-sided p by the usual (b+1)/(m+1) rule.
    """
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(d)))
    n = len(d)
    signs = rng.choice((-1.0, 1.0), size=(draws, n))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (draws + 1))


def bootstrap_ci(d, draws=10000, seed=1337, alpha=0.05):
    """Percentile bootstrap CI over windows for the paired mean difference."""
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(draws, n))
    means = d[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def paired_stats(a_nll, b_nll):
    """Paired comparison of per-window mean NLL. a = baseline, b = challenger.

    Positive mean_nll_diff means the challenger has LOWER NLL (is better).
    Reports the locked protocol: sign-flip permutation p (20,000 draws) and a
    percentile bootstrap CI (10,000 resamples), per the pre-registration.
    """
    d = np.asarray(a_nll, dtype=float) - np.asarray(b_nll, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se > 0 else float("inf")
    wins = int((d > 0).sum())
    boot_lo, boot_hi = bootstrap_ci(d) if n > 1 else (mean, mean)
    return {"n_windows": n, "mean_nll_diff": mean, "sd": sd, "se": se,
            "t": t, "challenger_wins": wins,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
            "boot_ci_low": boot_lo, "boot_ci_high": boot_hi,
            "sign_flip_p": sign_flip_p(d) if n > 1 else 1.0,
            "_d": d}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="name=path (repeatable)")
    ap.add_argument("--val-data", default=os.path.join(HERE, "valid_8k.bin"))
    ap.add_argument("--windows", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--baseline", default=None, help="name to use as baseline in pairings")
    ap.add_argument("--compare", action="append", default=None,
                    help="name(s) to compare against --baseline (repeatable)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    specs = []
    for item in args.ckpt:
        if "=" not in item:
            sys.exit(f"[eval] --ckpt must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        if not os.path.isabs(path):
            path = os.path.join(HERE, path)
        if not os.path.exists(path):
            sys.exit(f"[eval] FATAL: missing checkpoint for {name!r}: {path}")
        specs.append((name, path))

    windows, winfo = build_windows(args.val_data, args.windows, args.ctx)
    print("=" * 74, flush=True)
    print("M7 PAIRED MULTI-WINDOW EVAL", flush=True)
    print("=" * 74, flush=True)
    print(f"  val file        : {args.val_data}", flush=True)
    print(f"  file tokens     : {winfo['file_tokens']:,}", flush=True)
    print(f"  windows         : {args.windows} disjoint, stride {winfo['stride']:,}", flush=True)
    print(f"  scored tokens   : {winfo['total_scored']:,}  "
          f"({winfo['total_scored']/511:.0f}x the original 511-token probe)", flush=True)
    print(f"  threads         : {args.threads}", flush=True)
    print("", flush=True)

    results = {}
    for name, path in specs:
        model, cfg, meta = load_model(path)
        t0 = time.time()
        per_w, sum_nll, n_tok = score(model, cfg.vocab_size, windows)
        dt = time.time() - t0
        corpus_ppl = math.exp(sum_nll / n_tok)
        results[name] = {
            "meta": meta,
            "corpus_ppl": corpus_ppl,
            "mean_window_ppl": float(np.mean(per_w)),
            "median_window_ppl": float(np.median(per_w)),
            "min_window_ppl": float(np.min(per_w)),
            "max_window_ppl": float(np.max(per_w)),
            "per_window_ppl": per_w,
            "per_window_nll": [math.log(p) for p in per_w],
            "sum_nll": sum_nll,
            "n_tokens": n_tok,
            "eval_seconds": dt,
        }
        print(f"  [{name}]", flush=True)
        print(f"     {meta['params']:,} params | linear={meta['linear']} | "
              f"L{meta['layers']} h{meta['hidden']} i{meta['inter']} "
              f"heads{meta['heads']} | step {meta['step']}", flush=True)
        print(f"     corpus PPL (token-weighted) : {corpus_ppl:.4f}", flush=True)
        print(f"     per-window PPL mean/median  : {np.mean(per_w):.4f} / {np.median(per_w):.4f}",
              flush=True)
        print(f"     per-window PPL min/max      : {np.min(per_w):.4f} / {np.max(per_w):.4f}",
              flush=True)
        print(f"     eval wall                   : {dt:.1f}s", flush=True)
        print("", flush=True)
        del model

    pairings = []
    if args.baseline and args.compare:
        base = args.baseline
        if base not in results:
            sys.exit(f"[eval] baseline {base!r} not among {list(results)}")
        print("=" * 74, flush=True)
        print(f"PAIRED COMPARISONS (baseline = {base})", flush=True)
        print("=" * 74, flush=True)
        for challenger in args.compare:
            if challenger not in results:
                print(f"  skip {challenger!r}: not evaluated", flush=True)
                continue
            st = paired_stats(results[base]["per_window_nll"],
                              results[challenger]["per_window_nll"])
            bp = results[base]["corpus_ppl"]
            cp = results[challenger]["corpus_ppl"]
            st.update({"baseline": base, "challenger": challenger,
                       "baseline_corpus_ppl": bp, "challenger_corpus_ppl": cp,
                       "pct_better": 100.0 * (bp - cp) / bp})
            pairings.append(st)

        # Holm correction across every contrast in this run (locked protocol)
        adj = holm([p["sign_flip_p"] for p in pairings]) if pairings else []
        for st, a in zip(pairings, adj):
            st["holm_p"] = a

        for st in pairings:
            print(f"  {st['challenger']} vs {st['baseline']}", flush=True)
            print(f"     corpus PPL       : {st['challenger_corpus_ppl']:.4f} vs "
                  f"{st['baseline_corpus_ppl']:.4f}  => {st['pct_better']:+.2f}% "
                  f"({'better' if st['pct_better'] > 0 else 'WORSE'})", flush=True)
            print(f"     paired mean ΔNLL : {st['mean_nll_diff']:+.6f} nats/token", flush=True)
            print(f"     bootstrap 95% CI : {st['boot_ci_low']:+.6f} .. "
                  f"{st['boot_ci_high']:+.6f}  (10,000 resamples)", flush=True)
            print(f"     sign-flip p      : {st['sign_flip_p']:.2e}  "
                  f"(Holm-adjusted {st['holm_p']:.2e})", flush=True)
            print(f"     windows won      : {st['challenger_wins']}/{st['n_windows']}", flush=True)
            gate = "PASSES" if st["holm_p"] <= 0.05 else "FAILS (not distinguishable from zero)"
            print(f"     significance gate: {gate} at Holm p<=0.05", flush=True)
            print("", flush=True)

        for st in pairings:
            st.pop("_d", None)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"windows_info": winfo,
                       "results": {k: {kk: vv for kk, vv in v.items()
                                       if kk != "per_window_nll"}
                                   for k, v in results.items()},
                       "pairings": pairings}, f, indent=2)
        print(f"  wrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
