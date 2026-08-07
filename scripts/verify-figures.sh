#!/usr/bin/env bash
# verify-figures.sh — every published figure must resolve to primary evidence.
#
# CLAUDE.md Rule B: no number enters program/RESEARCH_LEDGER.md without a matching
# raw log. This script is that gate. Exit 0 only if every figure is substantiated.
#
# WHY THIS IS NOT A GREP. The first version of this script grepped each figure as
# a literal string across the logs. Tested on the real corpus it returned 9 of 11
# figures OK, of which roughly 8 were coincidental substrings:
#     5.140  matched  nll=[25.140014, ...]        (an eval NLL value)
#     6.6    matched  PPL 26.655                  (a perplexity chunk)
#     92%    matched  "  0%|   | 0/100"           (a JSON-escaped tqdm bar)
#     5.513  matched  a .md findings file         (prose, not an instrument)
# A gate that manufactures agreement is worse than no gate, because it converts
# "nobody checked" into "verified".
#
# Three rules make it real:
#
#   1. EVIDENCE IS TIERED. Only instrument output (.log/.tsv/.csv/.jsonl, and
#      .txt/.out under a logs directory) can SUBSTANTIATE a figure. Prose — .md,
#      .rs, .py, commit messages, READMEs — can only CORROBORATE, reported
#      separately and never sufficient. This stops a number bootstrapping itself
#      into truth by being quoted, which is exactly how 8.25 tok/s survived three
#      weeks and reached the public README.
#
#   2. MATCHING IS NUMERIC. Printed precision sets the tolerance, so a log
#      reading 3.988 substantiates a published 3.99, while 40.8 is NOT
#      substantiated by 42.21. Rounding is legitimate; drift is not.
#
#   3. THE QUANTITY MUST BE NAMED. A hit only counts if the declared unit sits
#      adjacent to the number and a declared keyword appears on the same line or
#      in the filename. See scripts/figures.tsv.
#
# Statuses:
#   SUBSTANTIATED   numeric match in instrument output, right unit, right quantity
#   PROSE_ONLY      only found in prose — an assertion, not a measurement
#   ANTI_EVIDENCE   the only matches sit in text that DISOWNS the number
#   UNSUBSTANTIATED no qualifying match anywhere
#
# Usage:  scripts/verify-figures.sh [-v] [--manifest FILE]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

VERBOSE=0
MANIFEST="scripts/figures.tsv"
while [ $# -gt 0 ]; do
    case "$1" in
        -v|--verbose) VERBOSE=1 ;;
        --manifest) MANIFEST="${2:?--manifest needs a path}"; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

[ -f "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST" >&2; exit 1; }

VERBOSE="$VERBOSE" MANIFEST="$MANIFEST" python3 - <<'PYEOF'
import os, re, sys
from pathlib import Path

VERBOSE  = os.environ.get("VERBOSE") == "1"
MANIFEST = Path(os.environ["MANIFEST"])

# Evidence roots. hardware_logs/ is carried at docs/hardware_logs/ in this repo;
# both spellings are honoured so relocating it does not silently empty the gate.
CANDIDATE_ROOTS = ["hardware_logs", "docs/hardware_logs", "artifacts",
                   "aegis-uefi/matrix_logs", "model-lab"]
INSTRUMENT_SUFFIX = {".log", ".tsv", ".csv", ".jsonl"}
PROSE_SUFFIX      = {".md", ".rs", ".py", ".sh", ".json", ".toml", ".yaml", ".yml"}
AMBIGUOUS_SUFFIX  = {".txt", ".out"}
LOGDIR_HINTS      = ("hardware_logs", "matrix_logs", "logs")

# Trailing lookahead rejects only a digit/dot, NOT any word char: instrument
# output writes "6.59x CTZ", "2.8MB", "14.17M". Rejecting a following letter
# made every multiplier figure unmatchable — it is why a freshly captured
# 6.59x log still reported UNSUBSTANTIATED. The leading lookbehind still
# prevents matching digits inside an identifier.
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d.])")
NEG = re.compile(r"\b(?:no log|not the|never logged|unlogged|superseded|retract\w*|"
                 r"deliberately not|instead of|rather than|discarded|invalid|withdrawn|"
                 r"fabricat\w*|do not (?:quote|cite)|unreproducible|deleted|has NO LOG)\b",
                 re.IGNORECASE)

UNIT_ALIAS = {
    "tok/s": ("tok/s", "tokens/s", "token/s"), "x": ("x", "×"), "×": ("x", "×"),
    "mb": ("mb", "mib"), "%": ("%",), "m": ("m",),
}

def tier(p: Path) -> str:
    s = p.suffix.lower()
    if s in INSTRUMENT_SUFFIX:
        return "INSTRUMENT"
    if s in AMBIGUOUS_SUFFIX:
        return "INSTRUMENT" if any(h in str(p) for h in LOGDIR_HINTS) else "PROSE"
    return "PROSE"

def tolerance(raw: str) -> float:
    plain = raw.replace(",", "").lstrip("~")
    return 0.5 * 10 ** (-len(plain.split(".")[1])) if "." in plain else 0.5

# Scale suffixes. A figure published as "14.17M params" is substantiated by a log
# that prints the exact integer 14,171,392 — the instrument does not round for us.
# Without this the gate reports a well-logged number as missing, which is the
# false-NEGATIVE failure mode: it trains you to ignore the gate.
SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9}


def targets_for(f: dict):
    """(value, tol, require_adjacent_unit) candidates that would substantiate f."""
    out = [(f["value"], f["tol"], True)]
    mult = SCALE.get(f["unit"].lower())
    if mult:
        # The scaled form appears as a raw integer with no unit attached.
        out.append((f["value"] * mult, f["tol"] * mult, False))
    return out

def unit_ok(line: str, end: int, want: str) -> bool:
    if want == "-":
        return True
    tail = line[end:end + 14].strip().lower()
    wants = UNIT_ALIAS.get(want.lower(), (want.lower(),))
    return any(tail.startswith(w) for w in wants)

# ---- load manifest -------------------------------------------------------
figures = []
for ln in MANIFEST.read_text(encoding="utf-8").splitlines():
    if not ln.strip() or ln.lstrip().startswith("#"):
        continue
    parts = ln.split("\t")
    if len(parts) < 5 or parts[0] == "id":
        continue
    fid, val, unit, kw, desc = (p.strip() for p in parts[:5])
    approx = val.startswith("~")
    try:
        v = float(val.lstrip("~").replace(",", ""))
    except ValueError:
        continue
    tol = tolerance(val)
    if approx:
        # "~500 tok/s" is a claim about a neighbourhood, not a point. Give it 2%
        # and say so in the output, rather than failing a figure that is logged
        # three times as 499.28 / 507.12 / 492.12.
        tol = max(tol, abs(v) * 0.02)
    figures.append({"id": fid, "raw": val, "value": v, "tol": tol, "approx": approx,
                    "unit": unit, "kw": [k for k in kw.lower().split("|") if k],
                    "desc": desc})

roots   = [r for r in CANDIDATE_ROOTS if Path(r).is_dir()]
missing = [r for r in CANDIDATE_ROOTS if not Path(r).is_dir()]

print("=" * 78)
print(" verify-figures.sh — figure substantiation against primary evidence")
print("=" * 78)
if not roots:
    print(f" FATAL: no evidence roots exist. Looked for: {CANDIDATE_ROOTS}", file=sys.stderr)
    sys.exit(1)
print(f" manifest       : {MANIFEST}  ({len(figures)} figures)")
print(f" evidence roots : {' '.join(roots)}")
if missing:
    print(f" not present    : {' '.join(missing)}  (skipped)")

# ---- index the corpus ONCE, line-granular -------------------------------
# One pass over every evidence file, recording each number with its line. Keeps
# the whole gate O(corpus + figures) instead of O(corpus x figures), which is
# what made the naive version time out at 120 s.
index = []          # (value, line_text_lower, match_end, path, lineno, tier)
n_files = n_inst = 0
for root in roots:
    for p in Path(root).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (INSTRUMENT_SUFFIX | PROSE_SUFFIX | AMBIGUOUS_SUFFIX):
            continue
        try:
            if p.stat().st_size > 64 * 1024 * 1024:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        t = tier(p)
        n_files += 1
        n_inst += (t == "INSTRUMENT")
        for i, line in enumerate(text.splitlines(), 1):
            if len(line) > 20000 or not any(c.isdigit() for c in line):
                continue
            low = line.lower()
            for m in NUM.finditer(line):
                try:
                    index.append((float(m.group(1).replace(",", "")), low,
                                  m.end(), p, i, t))
                except ValueError:
                    pass
print(f" indexed        : {n_files} files ({n_inst} INSTRUMENT, {n_files-n_inst} PROSE), "
      f"{len(index)} numbers")
print()

# ---- resolve each figure -------------------------------------------------
rows, fails = [], 0
for f in figures:
    sub = prose = anti = None
    cands = targets_for(f)
    for value, low, end, path, lineno, t in index:
        hit = False
        for tv, tt, need_unit in cands:
            if abs(value - tv) > tt:
                continue
            if need_unit and not unit_ok(low, end, f["unit"]):
                continue
            hit = True
            break
        if not hit:
            continue
        hay = low + " " + path.name.lower()
        if f["kw"] and not any(k in hay for k in f["kw"]):
            continue
        ref = f"{path}:{lineno}"
        if NEG.search(low):
            anti = anti or ref
        elif t == "INSTRUMENT":
            sub = sub or (ref, low.strip()[:110])
        else:
            prose = prose or ref

    if sub:
        status, ref, extra = "SUBSTANTIATED", sub[0], sub[1]
    elif anti and not prose:
        status, ref, extra = "ANTI_EVIDENCE", anti, "source disowns this number"
        fails += 1
    elif prose:
        status, ref, extra = "PROSE_ONLY", prose, "assertion, not a measurement"
        fails += 1
    else:
        status, ref, extra = "UNSUBSTANTIATED", "-", "no qualifying match in any instrument log"
        fails += 1
    rows.append((status, f, ref, extra))

W = max(len(f["id"]) for f in figures) + 1
for status, f, ref, extra in rows:
    mark = "ok " if status == "SUBSTANTIATED" else "!! "
    approx = "  [~2% tol]" if f.get("approx") else ""
    print(f"  {mark}{status:<16} {f['id']:<{W}} {f['raw']:>8} {f['unit']:<6} {f['desc']}{approx}")
    if status == "SUBSTANTIATED":
        print(f"        -> {ref}")
        if VERBOSE:
            print(f"           {extra}")
    else:
        print(f"        -> {extra}" + (f"  ({ref})" if ref != "-" else ""))

good = len(rows) - fails
print()
print("-" * 78)
print(f"  {good} substantiated / {fails} NOT substantiated  (of {len(rows)})")
if fails:
    print()
    print("  A figure that is not SUBSTANTIATED must not appear in")
    print("  program/RESEARCH_LEDGER.md, docs/TECHNICAL_REPORT.md, README.md, or")
    print("  anything leaving this repo. Re-measure it or delete it — do not soften it.")
    print("-" * 78)
    sys.exit(1)
print("  PASS")
print("-" * 78)
PYEOF
