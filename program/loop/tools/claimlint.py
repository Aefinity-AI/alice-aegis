#!/usr/bin/env python3
"""claimlint.py — scan a DOCUMENT for numbers and hold each one to the ledger.

ARIS's evidence_check runs claim -> source: "the claim cites eval.json:73.2, is
73.2 in eval.json?" That direction cannot catch this program's actual disease,
because nobody would ever have written a claims.json entry for 8.25 tok/s. The
number did not enter through a claim. It entered through a paragraph.

So claimlint runs the other direction — document -> ledger:

    every number in this document, is it accounted for?

That direction is what would have caught 8.25 tok/s on day one, and it is the
direction that catches it AGAIN every time it tries to come back. It is also the
only direction that finds the copy nobody audited: ops.rs:621, where commit
254ba43's unlogged sweep changed the engine's default worker count to LOGICAL
processors and wrote "SMT is a small win" into a doc comment. 8.25 was retracted
in a commit message; that source comment and that default were never touched.
Document-level audits cannot see source comments. This tool reads .rs too.

TWO TIERS, because a linter that cries wolf gets deleted (the same reason
guard.py permits `dd` to removable media):

  tier 1  --dead-only   (DEFAULT; safe to make blocking today)
          Fail if the document contains a number the ledger records as
          retracted / superseded / unlogged / commit-only.
          Near-zero false positives. A line that ADMITS the number is dead
          (contains 'retracted', 'superseded', 'no log', 'unlogged', '⚠', ...)
          is allowed — so program/RESEARCH_LEDGER.md passes unannotated, while
          docs/TECHNICAL_REPORT.md:32 does not.

  tier 2  --strict      (for a document going outside the house)
          Every number inside a **bold** span or a | markdown table cell |
          must match a `live` claim of kind measured/derived, or sit in the
          structural allowlist. Scoped to bold+tables ON PURPOSE: that is
          empirically where the headline defects were (§2 rows 3/3a/4/6/7,
          §7.1's table, the bolded 2.27x / 0.7 tok/s / 40.8% / 12.801). Prose
          asides are NOT covered and the tool says so rather than pretending.

Exit 0 clean, 1 findings, 2 usage. Pure stdlib, no model, no network.

  claimlint.py docs/TECHNICAL_REPORT.md
  claimlint.py --strict docs/TECHNICAL_REPORT.md
  claimlint.py --dead-only aegis-core/src/ops.rs aegis-uefi/src/main.rs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_check import _NUM_TOKEN_RE, _dec  # noqa: E402  (same matcher, both directions)
import ledger  # noqa: E402

REPO = Path(os.environ.get("ALICE_REPO", Path.home())).resolve()

# A line that names the number's death may state the number. This is what lets
# the honest retraction rows in RESEARCH_LEDGER.md and TECHNICAL_REPORT.md §2
# coexist with a blocking gate — and it is exactly the distinction the report
# failed to maintain when row 7 declared 3.31 J/token superseded and §7.1 used
# it eleven lines later.
DEATH_WORDS = re.compile(
    r"retract|supersed|withdraw|unlogged|no log|never logged|refuted|REFUTED"
    r"|not reproducible|unreproducible|contradicted|⚠|❌|do not (?:quote|cite|use)"
    r"|banned|deprecated|WRONG|historical|for the record",
    re.IGNORECASE)

BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
ORDERED_LI = re.compile(r"^\s*\d+[.)]\s")
RUST_DOC = re.compile(r"^\s*(///|//!|\s*\*|//)")

# Structural numbers that are never measurements.
YEAR = re.compile(r"^(19|20|21)\d\d$")

# TRANSPARENT CHARACTERS — the one place claimlint must differ from
# evidence_check's boundary policy, and the reason is empirical.
#
# evidence_check's allow-list is tuned for LOG FILES, where a number is flanked
# by whitespace or plain punctuation. Documents are not log files. Run against
# docs/TECHNICAL_REPORT.md the unmodified matcher found 7 dead-number uses and
# MISSED the most prominent ones — `**8.25 tok/s**` at line 32, `**14.488 →
# 12.801**` at 150, `**40.8%**` at 266, `**17.3 GB/s**` at 185, `**9.86
# GMAC/s**` at 229 — because a markdown emphasis asterisk is not a safe
# boundary, so every BOLDED headline number failed closed. In a document, the
# bolded numbers are exactly the ones that matter. A gate that is blind to the
# headline is not a gate.
#
# So these characters are mapped to spaces before scanning, OFFSET-PRESERVING
# (one char in, one char out, so reported columns stay true). Each is safe
# because none of them can ever be part of a number's value:
#   * _ ` ~   markdown emphasis / code / strikethrough
#   ×         multiplication sign — a unit suffix ("2.27×"), never a separator
#   → ⟶       an arrow between two values ("14.488 → 12.801")
#   † ‡ §     footnote markers ("42.21%*" is handled by the * rule)
#   |         a markdown table cell wall
# NOT transparent, deliberately: - – — : / . , ' and any digit. Those really can
# be part of a compound (dates, times, versions, ranges, locale grouping), and
# for those the fail-closed behaviour is correct.
_TRANSPARENT = "*_`~×→⟶†‡§|"
_TRANS_TABLE = str.maketrans({c: " " for c in _TRANSPARENT})


def _scan_text(line: str) -> str:
    """Offset-preserving normalization so bolded numbers are matchable."""
    return line.translate(_TRANS_TABLE)


def _is_structural(tok: str, line: str, col: int) -> bool:
    """True for numbers that cannot be measurements: years, small counts,
    ordered-list markers, markdown table separators, percentages of 100."""
    if TABLE_SEP.match(line):
        return True
    if ORDERED_LI.match(line) and col <= len(line) - len(line.lstrip()) + len(tok) + 1:
        return True
    if YEAR.match(tok.replace(",", "")):
        return True
    try:
        v = _dec(tok)
    except InvalidOperation:
        return True
    # Bare small integers are counts ("4 threads", "6 layers", "2 bits"), not
    # results. Anything with a decimal point, or >= 13, is a candidate result.
    if v == v.to_integral_value() and abs(v) < 13:
        return True
    if v in (Decimal(100), Decimal(0)):
        return True
    return False


def _spans_of_interest(lines: list[str], is_rust: bool) -> list[tuple[int, int, str, str]]:
    """(lineno, col, token, context) for tier-2. Bold spans and table cells only
    (Rust: doc comments only). Fenced code is skipped in markdown."""
    out, in_fence = [], False
    for i, line in enumerate(lines, 1):
        if not is_rust and FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if is_rust:
            if not RUST_DOC.match(line):
                continue
            regions = [(0, line)]
        else:
            regions = []
            if TABLE_ROW.match(line) and not TABLE_SEP.match(line):
                regions.append((0, line))
            for m in BOLD.finditer(line):
                regions.append((m.start(1), m.group(1)))
        for base, text in regions:
            for m in _NUM_TOKEN_RE.finditer(_scan_text(text)):
                out.append((i, base + m.start(1), m.group(1) + (m.group(2) or ""), line))
    return out


def _all_tokens(lines: list[str], is_rust: bool) -> list[tuple[int, int, str, str]]:
    """(lineno, col, token, line) for every safely-bounded number token."""
    out, in_fence = [], False
    for i, line in enumerate(lines, 1):
        if not is_rust and FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in _NUM_TOKEN_RE.finditer(_scan_text(line)):
            out.append((i, m.start(1), m.group(1) + (m.group(2) or ""), line))
    return out


def _num_eq(a: str, b: str) -> bool:
    try:
        return _dec(a.rstrip("%")) == _dec(b.rstrip("%")) and a.endswith("%") == b.endswith("%")
    except InvalidOperation:
        return False


def lint(path: Path, strict: bool, dead_only: bool, waivers: set[str],
         exempt: bool = True) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    is_rust = path.suffix == ".rs"
    cur = ledger.current()

    dead = [r for r in cur.values()
            if r["status"] in ledger.DEAD or r.get("kind") in ("unlogged", "commit-only")]
    live = [r for r in cur.values()
            if r["status"] == "live" and r.get("kind") in ledger.EXTERNAL_OK]

    findings: list[dict] = []

    # ---- tier 1: dead numbers, anywhere in the file ------------------------
    for lineno, col, tok, line in _all_tokens(lines, is_rust):
        if TABLE_SEP.match(line) or f"{path.name}:{lineno}" in waivers or tok in waivers:
            continue
        for r in dead:
            if not r.get("value") or not _num_eq(tok, r["value"]):
                continue
            # Prose WRAPS: TECHNICAL_REPORT.md:247 states 2.14x and the word
            # 'superseded' lands on 248. A one-line window each side is the
            # honest reading of a wrapped sentence; a whole-paragraph window
            # would let a distant disclaimer launder a nearby assertion.
            window = " ".join(lines[max(0, lineno - 2):lineno + 1])
            if exempt and DEATH_WORDS.search(window):
                continue          # the passage admits it; that is honest usage
            findings.append({
                "severity": "BLOCK", "file": str(path), "line": lineno, "value": tok,
                "claim": r["id"], "status": f"{r['status']}/{r.get('kind','')}",
                "why": (r.get("reason") or r.get("statement") or "").strip()[:160],
                "fix": (f"cite {r['superseded_by']} instead" if r.get("superseded_by")
                        else "delete the number, or state on this line that it is "
                             "retracted/superseded"),
                "text": line.strip()[:150],
            })
            break

    if dead_only:
        return findings

    # ---- tier 2: bold + table numbers must map to a live claim -------------
    if strict:
        for lineno, col, tok, line in _spans_of_interest(lines, is_rust):
            if _is_structural(tok, line, col) or tok in waivers:
                continue
            if f"{path.name}:{lineno}" in waivers:
                continue
            if any(_num_eq(tok, r["value"]) for r in live if r.get("value")):
                continue
            if any(_num_eq(tok, r["value"]) for r in dead if r.get("value")):
                continue          # already reported by tier 1
            findings.append({
                "severity": "UNACCOUNTED", "file": str(path), "line": lineno,
                "value": tok, "claim": "", "status": "not-in-ledger",
                "why": "a headline number (bold or table cell) with no ledger entry",
                "fix": "run the measurement through a runner so it gets a runcard, then "
                       "`ev claim add`; or move it out of bold/table if it is not a result",
                "text": line.strip()[:150],
            })
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--strict", action="store_true",
                    help="tier 2: bold/table numbers must map to a live claim")
    ap.add_argument("--dead-only", action="store_true", default=False,
                    help="tier 1 only (the default when --strict is absent)")
    ap.add_argument("--waive", action="append", default=[], metavar="VALUE|FILE:LINE",
                    help="explicit exemption; repeatable")
    ap.add_argument("--no-exempt", action="store_true",
                    help="submission grade: do NOT let retraction vocabulary near a dead "
                         "number excuse it. Catches the inversion where a passage says "
                         "'superseded' and then names the dead number as the replacement.")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    dead_only = not a.strict
    waivers = set(a.waive)

    allf: list[dict] = []
    for f in a.files:
        p = Path(f)
        if not p.is_file():
            print(f"claimlint: no such file: {f}", file=sys.stderr)
            return 2
        allf += lint(p, a.strict, dead_only, waivers, exempt=not a.no_exempt)

    if a.json:
        print(json.dumps(allf, indent=2))
    else:
        for x in allf:
            print(f"{x['file']}:{x['line']}: {x['severity']} {x['value']} "
                  f"[{x['claim'] or '-'} {x['status']}]")
            print(f"    why: {x['why']}")
            print(f"    fix: {x['fix']}")
            print(f"    >>> {x['text']}")
        blocks = sum(1 for x in allf if x["severity"] == "BLOCK")
        unacc = len(allf) - blocks
        print(f"\n{blocks} dead-number use(s), {unacc} unaccounted headline number(s).")
        if not a.strict:
            print("tier 1 only. Prose asides are NOT covered — run --strict before "
                  "anything leaves the house, and read the document yourself anyway.")
    return 1 if allf else 0


if __name__ == "__main__":
    raise SystemExit(main())
