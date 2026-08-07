#!/usr/bin/env bash
# preflight-outbound.sh — gate for any document that leaves this repository.
#
# verify-figures.sh answers "is every figure in the LEDGER substantiated?".
# This answers a different question: "is this DOCUMENT safe to send?" — where
# "send" means a funder, a reviewer, a journalist, a partner, or the public web.
#
# It exists because on 2026-08-05 an audit of the published tree found three
# unsupported claims sitting in funder-facing documents, and the worst of them
# —— a trusted-code-base size understated by ~4.7x —— was in
# docs/DARPA_CAPABILITY_STATEMENT.md, a file verify-figures.sh does not scan.
# The claim had been public for three days. Nothing mechanical was watching.
#
#   ./scripts/preflight-outbound.sh                 # check the default set
#   ./scripts/preflight-outbound.sh path/to/doc.md  # check specific files
#
# Exit 0 = safe to send. Non-zero = do not send.
#
# Checks, in order of how badly each one burns you:
#   1. RETRACTED FIGURES — a number this repo has already withdrawn, reappearing.
#      Sourced from program/loop/claims.jsonl, so retracting a claim once
#      protects every future document automatically.
#   2. PERSONAL CONTACT DETAILS — home address / phone reaching a public artifact.
#   3. EMULATION-DERIVED PERFORMANCE — Rule A. A tok/s or speedup figure in the
#      same breath as QEMU/TCG/OVMF.
#   4. TICK-DERIVED RATIOS — the RDTSC corollary. MAC/cycle or %-of-peak without
#      the measured effective/nominal clock ratio nearby.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

DEFAULT_TARGETS=(
  README.md
  docs/DARPA_CAPABILITY_STATEMENT.md      # was unscanned; the 2,400-line claim lived here
  docs/TECHNICAL_REPORT.md
  docs/RFI_SN_26_97_ALICE_DRAFT_V1.md
  docs/RFI_SN_26_97_ALICE_DRAFT_V2.md
  program/ROADMAP.md
  program/RESEARCH_LEDGER.md
)

if [ "$#" -gt 0 ]; then TARGETS=("$@"); else TARGETS=();
  for t in "${DEFAULT_TARGETS[@]}"; do [ -f "$t" ] && TARGETS+=("$t"); done
fi

echo "preflight-outbound — ${#TARGETS[@]} document(s)"
echo

CLAIMS="program/loop/claims.jsonl"
TARGETS_JOINED="$(printf '%s\n' "${TARGETS[@]}")"

TARGETS="$TARGETS_JOINED" CLAIMS="$CLAIMS" python3 - <<'PYEOF'
import json, os, re, sys

targets = [t for t in os.environ["TARGETS"].splitlines() if t.strip()]
claims_path = os.environ["CLAIMS"]

# ── Load the retraction dead-list ────────────────────────────────────────────
dead = {}   # value-string -> (id, unit, why)
if os.path.exists(claims_path):
    for line in open(claims_path):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        kind, status = d.get("kind"), d.get("status")
        # retracted outright, or live-but-banned-from-external-documents
        banned = status == "retracted" or kind == "unlogged" or (
            kind == "commit-only" and "external" in (d.get("ceiling") or "").lower()
        )
        if not banned:
            continue
        v = str(d.get("value") or "").strip()
        if not v or v in ("0", ""):
            continue
        why = (d.get("reason") or d.get("ceiling") or "withdrawn").strip()
        dead.setdefault(v, (d.get("id"), d.get("unit") or "", why[:110]))

print(f"  dead-list: {len(dead)} withdrawn/banned values loaded from {claims_path}")
print()

PII = re.compile(r'[0-9]{3,5}\s+County\s+Road\s+[0-9]+|\(?409\)?\s?[-. ]?656[-. ]2416', re.I)
PERF_UNIT = re.compile(r'[0-9][0-9,.]*\s*(tok/s|tokens?/s|GB/s|J/tok|J/token|cycles?/token|ticks?/token)\b', re.I)
# Rule A is about EMULATION (TCG), not about QEMU as such: QEMU/KVM runs on real
# silicon and its timings are not automatically void. Flag TCG and explicit
# emulation only, and never when KVM accel is named on the same line.
EMU = re.compile(r'\b(TCG|emulat\w*)\b', re.I)
KVM = re.compile(r'\bKVM\b', re.I)
TICK = re.compile(r'\b(MAC/cycle|%\s*of\s*peak|percent\s+of\s+peak|IPC)\b', re.I)
RATIO_NEARBY = re.compile(r'effective/nominal|nominal\s+ratio|1\.53\s*[x×]|clock\s+ratio', re.I)

# A withdrawn number is allowed to APPEAR — this repo publishes its own
# retractions on purpose, and burying them would be the actual dishonesty. What
# must never happen is a withdrawn number ASSERTED as a live result. These
# markers distinguish the two; if one is present near the number, the document
# is disclosing the retraction rather than making the claim.
DISCLOSED = re.compile(
    r'withdraw\w*|retract\w*|supersed\w*|no log|unlogged|not substantiated|'
    r'commit-only|commit message only|banned|dead|⚠|CONTRADICT\w*|'
    r'never logged|has no log|do not cite|must not appear|'
    # narrating a figure's own history is disclosure, not assertion
    r'an earlier|earlier \w+ sample|previously quoted|the old design|'
    r'sample figure|once evidenced|conclusion is unaffected', re.I)

findings = []
for path in targets:
    if not os.path.exists(path):
        print(f"  ?  missing: {path}")
        continue
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    for i, ln in enumerate(lines, 1):
        # 1. retracted figures — only when asserted, not when disclosed
        window = " ".join(lines[max(0, i - 3): i + 2])
        for v, (cid, unit, why) in dead.items():
            # word-boundary match so 8.25 does not match 18.253
            if re.search(r'(?<![0-9.])' + re.escape(v) + r'(?![0-9])', ln):
                if DISCLOSED.search(window):
                    continue   # documented retraction — the doctrine working
                findings.append(("RETRACTED", path, i,
                                 f"{v} {unit} — withdrawn as {cid}: {why}", ln.strip()[:110]))
        # 2. PII
        if PII.search(ln):
            findings.append(("PII", path, i, "personal contact details", "<redacted>"))
        # 3. emulation-derived performance (Rule A)
        if PERF_UNIT.search(ln) and EMU.search(ln) and not KVM.search(ln):
            if not re.search(r'correctness[- ]only|not a performance|no performance|'
                             r'do not cite|not comparable', ln, re.I):
                findings.append(("RULE-A", path, i,
                                 "performance figure attributed to emulation, unqualified",
                                 ln.strip()[:110]))
        # 4. tick-derived ratio without the clock ratio (RDTSC corollary)
        if TICK.search(ln):
            window = " ".join(lines[max(0, i - 4): i + 3])
            if not RATIO_NEARBY.search(window):
                findings.append(("RDTSC", path, i,
                                 "tick-derived ratio without effective/nominal clock ratio nearby",
                                 ln.strip()[:110]))

if not findings:
    print("  ✓ clean — nothing withdrawn, no contact details, no emulation-derived")
    print("    performance, no unqualified tick-derived ratios.")
    sys.exit(0)

order = {"PII": 0, "RETRACTED": 1, "RULE-A": 2, "RDTSC": 3}
findings.sort(key=lambda f: (order.get(f[0], 9), f[1], f[2]))
for kind, path, line, what, ctx in findings:
    print(f"  ✗ {kind:9s} {path}:{line}")
    print(f"      {what}")
    if ctx and ctx != "<redacted>":
        print(f"      | {ctx}")
print()
print(f"  {len(findings)} finding(s). Do not send these documents until each is")
print("  resolved — re-measure, cite a log, or delete the claim. Do not soften it.")
sys.exit(1)
PYEOF
