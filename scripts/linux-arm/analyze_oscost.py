#!/usr/bin/env python3
"""analyze_oscost.py — evaluate the paired OS-cost measurement against the
PREREGISTERED protocol (docs/hardware_logs/oscost_PREREGISTRATION_2026-07-30.md).

Usage:
    analyze_oscost.py <U_bootlog> <L_bootlog>

Implements exactly the locked estimands — nothing here may be changed after
the real logs exist except bug fixes that do not alter an estimand, and any
such fix must be noted in the output it produces.

  primary   : pooled + per-prompt median of (r_L - r_U)/r_U over boot pairs,
              r = decode ticks/token, prompts = the three locked prompts
  gates     : output identity (token-identical text per prompt across arms),
              clock parity (|ratio_U - ratio_L| <= 2% per adjacent pair)
  secondary : prefill cycles/token delta; in-process CV (L2) summary
  bands     : |d|<2% null / 2-10% modest / >=10% investigate / sign reported
"""
import re
import sys
import random

PROMPTS = ["hello alice", "how are you today?", "continue"]
NBOOT = 10_000  # bootstrap resamples, locked


def split_boots(text, marker):
    """Split a log into per-boot chunks by its boot banner."""
    idx = [m.start() for m in re.finditer(re.escape(marker), text)]
    return [text[a:b] for a, b in zip(idx, idx[1:] + [len(text)])]


def parse_u(text):
    """Unikernel BOOTLOG.TXT: PROMPT/RESPONSE blocks with
    '(N tokens, T ticks, t ticks/token, ... clock C% of nominal)'."""
    boots = []
    for chunk in split_boots(text, "==== A.L.I.C.E. BOOT ===="):
        runs = {}
        for m in re.finditer(
            r'PROMPT: "([^"]+)"\s*\nRESPONSE:(.*?)\n\s*\((\d+) tokens, (\d+) ticks, '
            r"(\d+) ticks/token.*?clock (\d+)% of nominal\)",
            chunk,
            re.S,
        ):
            prompt, resp, ntok, _ticks, tpt, clk = m.groups()
            runs[prompt] = {
                "text": resp.strip(),
                "ntok": int(ntok),
                "ticks_per_tok": int(tpt),
                "clock_pct": int(clk),
            }
        if runs:
            boots.append(runs)
    return boots


def parse_l(text):
    """Linux BOOTLOG_LINUX_ARM.txt: L3 sections with engine PERFORMANCE lines."""
    boots = []
    for chunk in split_boots(text, "==== AEGIS MINIMAL-LINUX ARM ===="):
        runs = {}
        cur = re.search(r"cur_freq: (\d+)", chunk)
        for m in re.finditer(
            r'--- L3 prompt: "([^"]+)"(.*?)(?=--- L3 prompt:|==== GAUNTLET DONE)', chunk, re.S
        ):
            prompt, body = m.groups()
            perf = re.search(r"\[PERFORMANCE\] Average Cycles/Token: (\d+)", body)
            resp = re.search(r"Final Full Response:\s*(.*?)\nGenerated (\d+) tokens", body, re.S)
            if perf and resp:
                runs[prompt] = {
                    "text": resp.group(1).strip(),
                    "ntok": int(resp.group(2)),
                    "ticks_per_tok": int(perf.group(1)),
                    "cur_freq_khz": int(cur.group(1)) if cur else None,
                }
        # L2 in-process block: per-iteration decode cycles/token CSV lines
        l2 = re.findall(r"^\d+,(\d+),\d+,\d+,\d+,[0-9a-fx]+", chunk, re.M)
        if runs:
            boots.append({"runs": runs, "l2": [int(x) for x in l2]})
    return boots


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def bootstrap_ci(deltas, n=NBOOT, seed=1337):
    rng = random.Random(seed)
    meds = []
    for _ in range(n):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        meds.append(median(sample))
    meds.sort()
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    u_boots = parse_u(open(sys.argv[1], encoding="utf-8", errors="replace").read())
    l_boots = parse_l(open(sys.argv[2], encoding="utf-8", errors="replace").read())
    print(f"parsed: {len(u_boots)} U boots, {len(l_boots)} L boots")
    npairs = min(len(u_boots), len(l_boots))
    if npairs < 3:
        sys.exit("FATAL: <3 boot pairs; protocol requires 5/arm (report, do not analyze)")
    if npairs < 5:
        print(f"WARNING: only {npairs} pairs (<5) — report as under-powered, bands still apply")

    # ---- gate 1: output identity ------------------------------------------
    print("\n== GATE: output identity (bit-exactness across OS) ==")
    identical = True
    for p in PROMPTS:
        texts_u = {b[p]["text"] for b in u_boots if p in b}
        texts_l = {b["runs"][p]["text"] for b in l_boots if p in b["runs"]}
        same = texts_u == texts_l and len(texts_u) == 1
        identical &= same
        print(f'  "{p}": U variants={len(texts_u)} L variants={len(texts_l)} identical={same}')
    if not identical:
        print("  GATE FAILED — work-identity broken; timing comparison INVALID until explained.")

    # ---- gate 2: clock parity ---------------------------------------------
    print("\n== GATE: clock parity ==")
    for i in range(npairs):
        u_clk = {b["clock_pct"] for b in [u_boots[i][p] for p in PROMPTS if p in u_boots[i]]}
        l_khz = next(
            (l_boots[i]["runs"][p].get("cur_freq_khz") for p in PROMPTS if p in l_boots[i]["runs"]),
            None,
        )
        print(f"  pair {i+1}: U clock%={sorted(u_clk)} L cur_freq_khz={l_khz}")
    print("  (2% exclusion rule applies per prereg §4; excluded pairs must be listed)")

    # ---- primary estimand --------------------------------------------------
    print("\n== PRIMARY: decode ticks/token, (L-U)/U ==")
    pooled = []
    for p in PROMPTS:
        deltas = []
        for i in range(npairs):
            if p in u_boots[i] and p in l_boots[i]["runs"]:
                ru = u_boots[i][p]["ticks_per_tok"]
                rl = l_boots[i]["runs"][p]["ticks_per_tok"]
                deltas.append((rl - ru) / ru)
        if deltas:
            lo, hi = bootstrap_ci(deltas)
            print(
                f'  "{p}": median {median(deltas)*100:+.3f}%  '
                f"CI [{lo*100:+.3f}%, {hi*100:+.3f}%]  n={len(deltas)}"
            )
            pooled += deltas
    if not pooled:
        sys.exit("FATAL: no paired prompt runs found")
    lo, hi = bootstrap_ci(pooled)
    d = median(pooled)
    print(f"\n  POOLED Δ_OS: median {d*100:+.3f}%  CI [{lo*100:+.3f}%, {hi*100:+.3f}%]  n={len(pooled)}")

    # ---- bands (locked) ----------------------------------------------------
    if lo <= 0 <= hi:
        verdict = f"not distinguishable from zero at N={npairs}/arm (CI spans 0)"
    elif abs(d) < 0.02:
        verdict = "BAND 1 — null: OS cost below 2%"
    elif abs(d) < 0.10:
        verdict = "BAND 2 — modest: report with mechanism hypotheses"
    else:
        verdict = "BAND 3 — large: investigate mechanism BEFORE publishing"
    if d < 0 and not (lo <= 0 <= hi):
        verdict += "  [SIGN: minimal Linux FASTER than no-OS — report as-is]"
    print(f"  VERDICT: {verdict}")

    # ---- secondary: L2 in-process variance --------------------------------
    print("\n== SECONDARY: L2 in-process decode cycles/token (per boot) ==")
    for i, b in enumerate(l_boots[:npairs]):
        xs = b["l2"]
        if len(xs) >= 3:
            m = sum(xs) / len(xs)
            sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
            print(f"  L boot {i+1}: n={len(xs)} mean={m:,.0f} CV={sd/m*100:.3f}%")
    print("\nRemember prereg §7: this is MINIMAL Linux, single-thread, M7-class model.")
    print("Banned sentences list applies to whatever gets written up.")


if __name__ == "__main__":
    main()
