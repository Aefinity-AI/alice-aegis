#!/usr/bin/env python3
"""check_verbatim.py — receipt-only detector for shot-copied / key-snapped tool
arguments in AEGIS-TRACE receipts.

Rule: the argument inside a step-0 tool call (the text between the parentheses
of `CALC(...)` or `LOOKUP(...)`) must appear verbatim in the last `Q:` line of
the receipt's own prompt. Nothing outside the receipt is consulted, so a
verifier can apply the rule with no access to the suite or the model.

Why: on EVAL-60 T1 (2B, box2, 2026-09-06) the model answered three lookup
near-miss items by snapping the key to a real one (`p-100` -> `LOOKUP(P-100)`)
and one distractor by copying a shot (`two + two` -> `CALC(2 + 2)`). All four
gave a plausible-looking, wrong result. This rule flagged exactly those four
and none of the 47 correct calls.

Limits: only step 0 is checked. For K>1 episodes the receipt records the
initial prompt only, so later steps' queries are not available to the rule.

Usage:
    check_verbatim.py <receipts-dir> [summary.tsv]
Prints one row per receipt (item, tool, argument, query, verbatim ok/FLAG) and
a summary line. With summary.tsv it also cross-tabulates against the scorer's
arg_match column. Exit status 0 always; this is a report, not a gate.
"""
import csv
import os
import re
import sys

STEP0_RE = re.compile(r"^step 0: .*?tool=(\S+) in=([0-9a-f]*)", re.M)
PROMPT_RE = re.compile(r"^prompt-hex ([0-9a-f]+)", re.M)
ARG_RE = re.compile(r"^(CALC|LOOKUP)\((.*)\)$")


def last_query(prompt: str) -> str:
    qs = [l[2:].strip() for l in prompt.splitlines() if l.startswith("Q:")]
    return qs[-1] if qs else ""


def check_receipt_text(text: str):
    """Return (tool, argument, last_query, verdict) for one receipt body.
    verdict is 'ok', 'FLAG', or '-' (no tool call at step 0)."""
    m = PROMPT_RE.search(text)
    prompt = bytes.fromhex(m.group(1)).decode("utf-8", "replace") if m else ""
    q = last_query(prompt)
    st = STEP0_RE.search(text)
    if not st:
        return ("?", "", q, "-")
    tool = st.group(1)
    raw = bytes.fromhex(st.group(2)).decode("utf-8", "replace")
    am = ARG_RE.match(raw)
    arg = am.group(2) if am else raw
    if tool == "no-tool":
        return (tool, "", q, "-")
    verdict = "ok" if arg and arg in q else "FLAG"
    return (tool, arg, q, verdict)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = argv[1]
    scorer = {}
    if len(argv) > 2:
        with open(argv[2], newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                scorer[r["item_id"]] = r.get("arg_match", "")
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".txt") or "." in f[:-4]:
            continue  # skip .gen.err / .attest.out etc.
        with open(os.path.join(d, f), errors="replace") as fh:
            tool, arg, q, verdict = check_receipt_text(fh.read())
        rows.append((f[:-4], tool, arg, q, verdict, scorer.get(f[:-4], "")))
    print("item\ttool\targ\tquery\tverbatim\targ_match")
    for r in rows:
        print("\t".join(r))
    called = [r for r in rows if r[4] != "-"]
    flagged = [r for r in called if r[4] == "FLAG"]
    print(f"\nreceipts={len(rows)} step0-calls={len(called)} flagged={len(flagged)}")
    for r in flagged:
        print(f"  {r[0]}: arg {r[2]!r} not in query {r[3]!r}")
    if scorer:
        tp = sum(1 for r in flagged if r[5] == "false")
        fp = sum(1 for r in flagged if r[5] == "true")
        missed = [r[0] for r in called if r[4] == "ok" and r[5] == "false"]
        print(f"vs scorer: flagged-and-wrong={tp} flagged-but-right={fp} wrong-not-flagged={missed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
