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

K>1: every step is checked, but the rule necessarily weakens after step 0.
At step 0 the current query is known exactly (the last `Q:` line of the
prompt), so the argument must appear there. At step k>0 the receipt records
only the initial prompt -- the text the model generated in between is not
recoverable from the receipt without the vocabulary -- so the current query is
unknown. The rule therefore accepts an argument that appears verbatim in ANY
`Q:` line of the prompt (verdict `ok`) or in the decoded `out=` of an earlier
step (verdict `ok-tool`, the legitimate tool-chaining case), and flags one that
appears in neither, since such an argument was invented by the model rather
than copied from anything the receipt records. This is deliberately permissive:
a later step that snaps a key to a value present in an earlier shot will not be
caught. It is strictly more coverage than step 0 alone, not a complete rule.

Limits: a FLAG is a review signal, not a verdict of incorrectness, and this
script is a report rather than a gate (it always exits 0).

Usage:
    check_verbatim.py <receipts-dir> [summary.tsv]
Prints one row per step (item, step, tool, argument, source query, verdict)
and a summary line. With summary.tsv it also cross-tabulates against the scorer's
arg_match column. Exit status 0 always; this is a report, not a gate.
"""
import csv
import os
import re
import sys

STEP_RE = re.compile(r"^step (\d+): .*?tool=(\S+) in=([0-9a-f]*) out=([0-9a-f]*)", re.M)
PROMPT_RE = re.compile(r"^prompt-hex ([0-9a-f]+)", re.M)
ARG_RE = re.compile(r"^(CALC|LOOKUP)\((.*)\)$")


def last_query(prompt: str) -> str:
    qs = queries(prompt)
    return qs[-1] if qs else ""


def queries(prompt: str) -> list:
    """Every `Q:` line of the prompt, in order, stripped of the marker."""
    return [l[2:].strip() for l in prompt.splitlines() if l.startswith("Q:")]


def _unhex(h: str) -> str:
    return bytes.fromhex(h).decode("utf-8", "replace") if h else ""


def _argument(raw: str) -> str:
    m = ARG_RE.match(raw)
    return m.group(2) if m else raw


def check_receipt_steps(text: str):
    """Return one (step, tool, argument, source, verdict) per step line.

    verdict is 'ok' (argument copied from a query), 'ok-tool' (copied from an
    earlier step's tool result), 'FLAG' (found in neither) or '-' (no call).
    `source` is the query the argument was checked against: the last `Q:` line
    at step 0, and a short description of the accepted source afterwards.
    """
    m = PROMPT_RE.search(text)
    prompt = _unhex(m.group(1)) if m else ""
    qs = queries(prompt)
    last_q = qs[-1] if qs else ""
    rows = []
    outs = []  # decoded tool results of the steps already seen
    for sm in STEP_RE.finditer(text):
        idx, tool = int(sm.group(1)), sm.group(2)
        arg = _argument(_unhex(sm.group(3)))
        out = _unhex(sm.group(4))
        if tool == "no-tool" or not arg:
            verdict = "-" if tool == "no-tool" else "FLAG"
            rows.append((idx, tool, arg, last_q if idx == 0 else "", verdict))
            outs.append(out)
            continue
        if idx == 0:
            # The current query is known exactly, so the strict rule applies.
            verdict, source = ("ok", last_q) if arg in last_q else ("FLAG", last_q)
        elif any(arg in q for q in qs):
            verdict, source = "ok", next(q for q in qs if arg in q)
        elif any(arg in o for o in outs if o):
            verdict, source = "ok-tool", next(o for o in outs if o and arg in o)
        else:
            verdict, source = "FLAG", ""
        rows.append((idx, tool, arg, source, verdict))
        outs.append(out)
    return rows


def check_receipt_text(text: str):
    """Return (tool, argument, last_query, verdict) for step 0 of one receipt.

    Kept for callers that only care about the first step; check_receipt_steps
    is the full-episode form.
    """
    rows = check_receipt_steps(text)
    if not rows:
        m = PROMPT_RE.search(text)
        return ("?", "", last_query(_unhex(m.group(1)) if m else ""), "-")
    _, tool, arg, source, verdict = rows[0]
    return (tool, arg, source, verdict)


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
            steps = check_receipt_steps(fh.read())
        for idx, tool, arg, source, verdict in steps:
            rows.append((f[:-4], str(idx), tool, arg, source, verdict))
    print("item\tstep\ttool\targ\tsource\tverbatim")
    for r in rows:
        print("\t".join(r))
    called = [r for r in rows if r[5] != "-"]
    flagged = [r for r in called if r[5] == "FLAG"]
    chained = [r for r in called if r[5] == "ok-tool"]
    later = [r for r in called if r[1] != "0"]
    receipts = {r[0] for r in rows}
    print(
        f"\nreceipts={len(receipts)} tool-calls={len(called)} "
        f"(step0={len(called)-len(later)} later-steps={len(later)}) "
        f"from-tool-result={len(chained)} flagged={len(flagged)}"
    )
    for r in flagged:
        where = f"step {r[1]}"
        if r[1] == "0":
            print(f"  {r[0]} {where}: arg {r[3]!r} not in query {r[4]!r}")
        else:
            print(f"  {r[0]} {where}: arg {r[3]!r} in no prompt query and no earlier tool result")
    if scorer:
        # The scorer grades an item, not a step, so fold each receipt to its
        # worst verdict before comparing.
        worst = {}
        for r in called:
            if worst.get(r[0]) != "FLAG":
                worst[r[0]] = r[5]
        fl = [i for i, v in worst.items() if v == "FLAG"]
        tp = sum(1 for i in fl if scorer.get(i) == "false")
        fp = sum(1 for i in fl if scorer.get(i) == "true")
        missed = [i for i, v in worst.items() if v != "FLAG" and scorer.get(i) == "false"]
        print(f"vs scorer: flagged-and-wrong={tp} flagged-but-right={fp} wrong-not-flagged={missed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
