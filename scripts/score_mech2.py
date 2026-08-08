#!/usr/bin/env python3
"""score_mech2.py — executable derivation for the MECH v2 paired medians (A22).

Rule B parent type 2 (derivation from instrument-backed numbers, formula
written down): reads the raw MECHV2/MECHV2L lines from the boot logs, computes
per-prompt medians and both preregistered delta forms, and prints them. Its
captured output under docs/hardware_logs/ is the substantiation target for the
median figures; every input number is in the cited logs verbatim.

Formulae: U tok/s = TSC_HZ / median(ticks_per_token, n=10) with
TSC_HZ = 2.1975e9 (A12 calibration); L tok/s = median(decode-only tok/s as
printed by the harness, n=10); throughput delta = rate_U/rate_L - 1;
prereg-form delta = (t_L - t_U)/t_L with t = seconds/token.
"""
import re, statistics, sys

U_LOG = "docs/hardware_logs/mech2_U_BOOTLOG_2026-08-01.txt"
L_LOG = "docs/hardware_logs/mech2colskip_L_dell_BOOTLOG_2026-08-01.txt"
TSC_HZ = 2.1975e9
PROMPTS = ["hello alice", "how are you today?", "continue"]

def u_medians():
    txt = open(U_LOG).read()
    out = {}
    for p in PROMPTS:
        tpts = [int(m.group(1)) for m in re.finditer(
            rf'MECHV2 "{re.escape(p)}" run \d+/10: \d+ tokens, \d+ ticks, (\d+) ticks/token', txt)]
        assert len(tpts) == 10, (p, len(tpts))
        med = statistics.median(tpts)
        out[p] = (TSC_HZ / med, med, (max(tpts)-min(tpts))/min(tpts)*100)
    return out

def l_medians():
    txt = open(L_LOG).read()
    out = {}
    cur = None; acc = {p: [] for p in PROMPTS}
    for line in txt.splitlines():
        m = re.match(r'MECHV2L "([^"]+)" run \d+/(\d+)', line)
        if m: cur = m.group(1); continue
        m = re.search(r'Decode \d+ tokens in [\d.]+ s \(([\d.]+) tok/s decode-only\)', line)
        if m and cur in acc: acc[cur].append(float(m.group(1)))
    for p in PROMPTS:
        v = acc[p]; assert len(v) == 10, (p, len(v))
        out[p] = statistics.median(v)
    return out

u, l = u_medians(), l_medians()
print(f"MECH v2 scoring (derivation; inputs: {U_LOG} + {L_LOG}; TSC {TSC_HZ/1e9} GHz)")
for p in PROMPTS:
    ru, med_ticks, spread = u[p]
    rl = l[p]
    thr = (ru/rl - 1) * 100
    prereg = (1/rl - 1/ru) / (1/rl) * 100
    print(f'  "{p}": U median {ru:.1f} tok/s (median {med_ticks:,.0f} ticks/token, spread {spread:.2f}%) '
          f'vs L median {rl:.3f} tok/s decode-only -> U advantage {thr:+.1f}% throughput / {prereg:+.1f}% prereg-form')
wins = sum((u[p][0] > l[p]) for p in PROMPTS)
print(f"  verdict: U wins {wins}/3 per-prompt medians")
