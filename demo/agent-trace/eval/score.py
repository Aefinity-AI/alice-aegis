#!/usr/bin/env python3
"""score.py — read <outdir>/summary.tsv and print per-bucket and overall
rates from the eval plan section 4, with Wilson 95% score intervals
(stdlib only). Works on a partial summary (fewer rows than the suite).

Usage: score.py <summary.tsv>
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return (max(0.0, lo), min(1.0, hi))


def fmt_rate(successes: int, n: int) -> str:
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson_interval(successes, n)
    return f"{successes}/{n} = {successes / n:.2%}  (95% CI {lo:.2%}-{hi:.2%})"


def load_summary(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def is_well_formed(row: dict) -> bool:
    # A call is well-formed if the observed tool is not NONE/none for a
    # tool-expected item, i.e. the scanner accepted *some* call (whether
    # or not it's the *right* tool). "none" for every step means no call
    # parsed at all.
    observed = row["tool_observed"]
    return any(t.strip().upper() != "NONE" for t in observed.split(","))


def is_call_emitted(row: dict) -> bool:
    return is_well_formed(row)


def is_correct_tool(row: dict) -> bool:
    return row["tool_observed"].strip().upper() == row["tool_expected"].strip().upper()


def score(rows: list[dict]) -> None:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    red_flags = []

    print(f"{'bucket':<18}{'n':>4}  metric: value")
    print("-" * 72)

    overall = {
        "tool_expected_n": 0,
        "call": 0,
        "well_formed": 0,
        "correct_tool": 0,
        "correct_arg": 0,
        "correct_output": 0,
        "distractor_n": 0,
        "distractor_no_call": 0,
    }

    for bucket in sorted(by_bucket):
        brows = by_bucket[bucket]
        n = len(brows)
        is_distractor_bucket = all(r["tool_expected"].strip().upper() == "NONE" for r in brows)

        print(f"\n[{bucket}] n={n}")

        if is_distractor_bucket:
            no_call = sum(1 for r in brows if not is_call_emitted(r))
            print(f"  no-tool precision:  {fmt_rate(no_call, n)}")
            overall["distractor_n"] += n
            overall["distractor_no_call"] += no_call
            lo, _ = wilson_interval(no_call, n)
            if lo < 0.5:
                red_flags.append(
                    f"bucket '{bucket}': no-tool precision Wilson lower bound "
                    f"{lo:.2%} < 50%"
                )
            continue

        call = sum(1 for r in brows if is_call_emitted(r))
        well_formed = sum(1 for r in brows if is_well_formed(r))
        correct_tool = sum(1 for r in brows if is_correct_tool(r))
        correct_arg = sum(1 for r in brows if r["arg_match"].strip().lower() == "true")
        correct_out = sum(1 for r in brows if r["output_match"].strip().lower() == "true")

        print(f"  call rate:          {fmt_rate(call, n)}")
        print(f"  well-formed rate:   {fmt_rate(well_formed, n)}")
        print(f"  correct-tool rate:  {fmt_rate(correct_tool, n)}")
        print(f"  correct-arg rate:   {fmt_rate(correct_arg, n)}")
        print(f"  exact-output rate:  {fmt_rate(correct_out, n)}")

        overall["tool_expected_n"] += n
        overall["call"] += call
        overall["well_formed"] += well_formed
        overall["correct_tool"] += correct_tool
        overall["correct_arg"] += correct_arg
        overall["correct_output"] += correct_out

        lo, _ = wilson_interval(correct_arg, n)
        if lo < 0.5:
            red_flags.append(
                f"bucket '{bucket}': correct-argument Wilson lower bound {lo:.2%} < 50%"
            )

    print("\n" + "=" * 72)
    print(f"OVERALL (tool-expected items) n={overall['tool_expected_n']}")
    print(f"  call rate:          {fmt_rate(overall['call'], overall['tool_expected_n'])}")
    print(f"  well-formed rate:   {fmt_rate(overall['well_formed'], overall['tool_expected_n'])}")
    print(f"  correct-tool rate:  {fmt_rate(overall['correct_tool'], overall['tool_expected_n'])}")
    print(f"  correct-arg rate:   {fmt_rate(overall['correct_arg'], overall['tool_expected_n'])}")
    print(f"  exact-output rate:  {fmt_rate(overall['correct_output'], overall['tool_expected_n'])}")
    print(f"\nOVERALL (distractor items) n={overall['distractor_n']}")
    print(
        "  no-tool precision:  "
        f"{fmt_rate(overall['distractor_no_call'], overall['distractor_n'])}"
    )

    print("\n" + "=" * 72)
    if red_flags:
        print("PRE-REGISTERED RED FLAGS TRIPPED:")
        for f in red_flags:
            print(f"  - {f}")
    else:
        print("No pre-registered red flags tripped (on the data present).")

    n_total_rows = len(rows)
    print(f"\n(scored {n_total_rows} rows from summary.tsv; partial summaries are expected mid-run)")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: score.py <summary.tsv>", file=sys.stderr)
        return 2
    rows = load_summary(sys.argv[1])
    if not rows:
        print("summary.tsv has no data rows yet.")
        return 0
    score(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
