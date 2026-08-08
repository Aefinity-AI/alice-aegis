#!/usr/bin/env python3
"""thread_sweep_parse.py — turn the sweep's rows into the four numbers the
ledger is allowed to carry, and refuse to produce them when the run does not
support them.

Reads the CSV the sweep wrote (or the tee'd log; it ignores lines it cannot
parse), prints a summary INTO THE SAME LOG, and emits a JSON block for
`ev metric`. It is deliberately a separate file so the statistic can be
re-derived from the log later by a third party — the property this program's
retracted numbers all lacked.

Three refusals, each aimed at a specific defect in the record:

  * WORK IDENTITY. Every arm must produce the same out_hash. If thread count
    changes the output, the arms did different work and no speedup statement is
    meaningful. (The engine's row-parallel matvec, ops.rs:653-695, partitions
    disjoint output rows with a fixed per-row column order, so the output SHOULD
    be bit-identical across thread counts — TECHNICAL_REPORT.md:36 blames a
    10.394-vs-10.348 delta on "floating-point summation order across thread
    counts", which the source contradicts. This check settles that empirically,
    in the log, forever.)

  * SPREAD BEFORE SIGNAL. It reports the within-arm IQR next to the between-arm
    difference and marks any ratio whose arms overlap as NOT RESOLVED. "8 threads
    beats 4 by ~5%" is unsupportable when the within-arm spread is 6%.

  * NO SINGLE-ROUND HEADLINES. With ROUNDS<3 it prints the medians and then
    states that no ratio may be published. A drift control needs repeats.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path


def smt_identifiable() -> tuple[bool, str]:
    """Can this machine answer an SMT question AT ALL?

    MEASURED on the dev box 2026-07-29: crosvm presents the i5-10210U as
    8 sockets x 1 core x 1 thread ('siblings: 1', 'cpu cores: 1', a distinct
    'physical id' per processor; lscpu agrees: Thread(s) per core = 1).
    The guest therefore CANNOT KNOW which vCPUs are SMT siblings, and the host
    scheduler chooses placement per run.

    Consequence, and it is the resolution of ledger row A4 rather than a caveat
    on it: '8 threads beats 4, so SMT is worth ~5%' is NOT AN IDENTIFIABLE
    CLAIM on this box, and never was. Both the 154f00a sweep (4t > 8t) and the
    254ba43 'clean re-measure' (8t > 4t) ran here. They do not contradict each
    other about SMT; they measured host placement luck under a flattened
    topology and disagreed, which is the expected behaviour of a
    non-identifiable experiment. Re-running the sweep on this machine more
    carefully cannot fix that.

    A 4t-vs-8t comparison here is still a legitimate OVERSUBSCRIPTION
    measurement (does asking for 8 workers on 8 vCPUs beat 4?) and that is what
    this harness reports it as. An SMT claim needs the UEFI gauntlet on the
    Dell/HP (real topology, no OS, CPUID visible) or host-side pinning outside
    the guest.
    """
    try:
        txt = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False, "cannot read /proc/cpuinfo"
    sib = set(re.findall(r"^siblings\s*:\s*(\d+)", txt, re.M))
    cores = set(re.findall(r"^cpu cores\s*:\s*(\d+)", txt, re.M))
    nproc = len(re.findall(r"^processor\s*:", txt, re.M))
    if sib == {"1"} and cores == {"1"} and nproc > 1:
        return False, (f"topology FLATTENED: {nproc} processors each reporting "
                       f"siblings=1 cpu-cores=1 (crosvm/KVM guest). SMT siblings are "
                       f"invisible here; a 4t-vs-8t delta measures host placement, "
                       f"not SMT.")
    return True, f"topology visible: siblings={sorted(sib)} cpu_cores={sorted(cores)}"


def read_rows(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 8 or parts[0] in ("round", "iter") or line.startswith("#"):
            continue
        try:
            rows.append({"round": int(parts[0]), "threads": int(parts[1]),
                         "iter": int(parts[2]), "cpt": int(parts[3]),
                         "total": int(parts[4]), "steps": int(parts[5]),
                         "prefill": int(parts[6]), "hash": parts[7].strip()})
        except ValueError:
            continue
    return rows


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: thread_sweep_parse.py <csv-or-log>", file=sys.stderr)
        return 2
    rows = read_rows(sys.argv[1])
    if not rows:
        print("NO PARSEABLE ROWS — nothing measured. No metric may be recorded.")
        return 1

    arms = sorted({r["threads"] for r in rows})
    rounds = len({r["round"] for r in rows})
    hashes = {r["hash"] for r in rows}
    out: dict = {"arms": {}, "rounds": rounds, "work_identical": len(hashes) == 1,
                 "out_hashes": sorted(hashes), "ratios": {}, "publishable": True,
                 "refusals": []}

    print(f"rounds={rounds}  arms={arms}  samples={len(rows)}")
    if len(hashes) != 1:
        out["publishable"] = False
        out["refusals"].append(
            "arms produced DIFFERENT outputs (out_hash differs) — the arms did not do "
            "the same work, so no speedup ratio is meaningful")
        print("\n*** WORK IDENTITY FAILED — out_hash differs across arms ***")
        for r in rows:
            print(f"    t={r['threads']} round={r['round']} iter={r['iter']} {r['hash']}")
    else:
        print(f"work identity: OK, all {len(rows)} samples produced out_hash "
              f"{hashes.pop()} (bit-identical output across thread counts)")

    print(f"\n{'thr':>4} {'n':>3} {'median cyc/tok':>15} {'IQR':>10} {'IQR%':>7} "
          f"{'min':>14} {'max':>14}")
    for t in arms:
        c = [r["cpt"] for r in rows if r["threads"] == t]
        med = st.median(c)
        iqr = q(c, 0.75) - q(c, 0.25)
        out["arms"][str(t)] = {"n": len(c), "median_cyc_per_tok": med,
                              "iqr": iqr, "iqr_pct": round(100 * iqr / med, 2) if med else None,
                              "min": min(c), "max": max(c),
                              "all": c}
        print(f"{t:>4} {len(c):>3} {med:>15,.0f} {iqr:>10,.0f} "
              f"{100*iqr/med if med else 0:>6.2f}% {min(c):>14,} {max(c):>14,}")

    base = arms[0]
    bmed = out["arms"][str(base)]["median_cyc_per_tok"]
    print(f"\nspeedup vs {base} thread(s), from cycles/token (NOT from tok/s):")
    for t in arms:
        a = out["arms"][str(t)]
        ratio = bmed / a["median_cyc_per_tok"] if a["median_cyc_per_tok"] else None
        # Resolved only if the arms' [min,max] ranges do not overlap. Crude on
        # purpose: it is a refusal criterion, not an inference.
        b = out["arms"][str(base)]
        resolved = t == base or a["max"] < b["min"] or a["min"] > b["max"]
        out["ratios"][f"r_{t}t_over_{base}t"] = {
            "ratio": round(ratio, 4) if ratio else None, "resolved": bool(resolved)}
        print(f"  {base}t -> {t}t : {ratio:>6.3f}x   "
              f"{'resolved' if resolved else '*** NOT RESOLVED (arm ranges overlap) ***'}")

    # The specific question the report got wrong, asked explicitly — and refused
    # outright when the hardware cannot answer it.
    ident, why = smt_identifiable()
    out["smt_identifiable"] = ident
    out["topology"] = why
    if 4 in arms and 8 in arms:
        a4, a8 = out["arms"]["4"], out["arms"]["8"]
        delta = 100 * (a4["median_cyc_per_tok"] - a8["median_cyc_per_tok"]) / a4["median_cyc_per_tok"]
        overlap = not (a8["max"] < a4["min"] or a8["min"] > a4["max"])
        label = "oversubscription (8 workers vs 4 on 8 vCPUs)" if not ident else "SMT"
        out["four_vs_eight"] = {
            "pct_8t_faster_than_4t": round(delta, 2), "ranges_overlap": overlap,
            "measures": label,
            "verdict": ("NOT_IDENTIFIABLE_ON_THIS_HOST" if not ident else
                        "UNRESOLVED" if overlap else
                        ("SMT HELPS" if delta > 0 else "SMT COSTS"))}
        print(f"\n4t vs 8t — the question TECHNICAL_REPORT.md:207 answers with 'SMT ~5%':")
        print(f"  topology: {why}")
        print(f"  8t is {delta:+.2f}% faster than 4t; arm ranges "
              f"{'OVERLAP' if overlap else 'are disjoint'}")
        print(f"  this measures: {label}")
        print(f"  verdict: {out['four_vs_eight']['verdict']}")
        if not ident:
            out["refusals"].append(
                "NO SMT CLAIM MAY BE MADE FROM THIS HOST: " + why +
                " Report the 4t/8t delta as oversubscription only. An SMT claim "
                "requires the UEFI gauntlet on bare metal or host-side pinning.")
        elif overlap:
            out["refusals"].append(
                "4t/8t ranges overlap: no SMT claim is supportable from this run")

    if rounds < 3:
        out["publishable"] = False
        out["refusals"].append(
            f"only {rounds} round(s); a drift control needs >= 3. Medians shown, no "
            f"ratio may be published.")

    print("\n--- METRICS-JSON ---")
    print(json.dumps(out, indent=2))
    if out["refusals"]:
        print("\nREFUSALS (these block `ev claim add`):")
        for r in out["refusals"]:
            print(f"  - {r}")
    return 0 if out["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
