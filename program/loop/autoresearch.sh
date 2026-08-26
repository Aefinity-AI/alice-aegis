#!/usr/bin/env bash
# autoresearch.sh — the unattended half of the research loop.
#
# DESIGN RULE: this script never produces a number. It runs only checks that are
# safe without a human watching, and it QUEUES anything that needs a quiet box or
# a judgment call. That split is deliberate:
#
#   - Rule A forbids performance figures from emulation.
#   - Measurements require a quiet box (`ev run` refuses otherwise), and this may
#     run while a session or another job holds the CPU.
#   - Wording and interpretation are Layer 4 (JUDGE) and need review.
#
# So an unattended run can only ever: verify evidence, detect drift, and report.
# If it ever starts banking claims on its own, that is a bug, not a feature.
#
# Usage:  bash program/loop/autoresearch.sh [--report-dir DIR]
set -uo pipefail
export ALICE_REPO="${ALICE_REPO:-$HOME/projects/alice-aegis}"
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$ALICE_REPO" || exit 1

STAMP=$(date -u +%Y-%m-%dT%H%MZ)
DAY=$(date -u +%Y-%m-%d)
OUT_DIR="${2:-$ALICE_REPO/program/loop/state/autoresearch}"
mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/autoresearch_${DAY}.md"
: > "$REPORT"   # truncate: a re-run replaces the day's report, never appends to it
FINDINGS=0

say() { printf '%s\n' "$*" >> "$REPORT"; }
flag() { FINDINGS=$((FINDINGS+1)); say "$@"; }

say "# autoresearch — $STAMP"
say ""
say "Unattended integrity sweep. Produces NO measurements by design; see header."
say ""

# ---------- 1. evidence chain -------------------------------------------------
say "## Evidence chain"
if VERIFY=$(ev claim verify 2>&1); then
  FAILED=$(printf '%s' "$VERIFY" | grep -cE '^(fail|FAIL)' || true)
  say '```'; printf '%s\n' "$VERIFY" | tail -20 >> "$REPORT"; say '```'
  [ "${FAILED:-0}" -gt 0 ] && flag "**$FAILED live claim(s) failed evidence_check.**"
else
  flag "**\`ev claim verify\` itself failed** — the ledger tooling is broken, fix before trusting anything below."
fi

# ---------- 2. dead numbers in documents --------------------------------------
say ""
say "## Dead-number sweep (ev lint)"
say '| file | dead-number uses |'
say '|---|---|'
for f in docs/TECHNICAL_REPORT.md program/RESEARCH_LEDGER.md program/ROADMAP.md \
         aegis-core/src/ops.rs README.md; do
  [ -f "$f" ] || continue
  n=$(ev lint "$f" 2>/dev/null | grep -oE '^[0-9]+ dead-number' | grep -oE '^[0-9]+' | head -1)
  n=${n:-?}
  say "| \`$f\` | $n |"
  [ "$n" != "0" ] && [ "$n" != "?" ] && FINDINGS=$((FINDINGS+1))
done

# ---------- 3. runcard receipts still valid -----------------------------------
say ""
say "## Runcard receipts"
CARDS=$(find docs/hardware_logs/runcards -name '*.json' 2>/dev/null | wc -l)
say "- $CARDS runcard(s) on disk"
if [ "$CARDS" -gt 0 ]; then
  BAD=$(ev runcard validate docs/hardware_logs/runcards/*.json 2>&1 | grep -cvE '^ok' || true)
  [ "${BAD:-0}" -gt 0 ] && flag "- **$BAD runcard(s) no longer validate** against the filesystem."
  [ "${BAD:-0}" -eq 0 ] && say "- all validate"
fi

# ---------- 4. evidence that is NOT versioned ---------------------------------
# The reset proved local-only evidence dies. A log backing a live claim MUST be tracked.
say ""
say "## Unversioned evidence (would not survive a wipe)"
UNTRACKED=0
while IFS= read -r log; do
  [ -z "$log" ] && continue
  if ! git ls-files --error-unmatch "$log" >/dev/null 2>&1; then
    flag "- **untracked evidence:** \`$log\`"
    UNTRACKED=$((UNTRACKED+1))
  fi
done < <(find docs/hardware_logs -maxdepth 1 -type f \( -name '*.log' -o -name '*.txt' \) 2>/dev/null)
[ "$UNTRACKED" -eq 0 ] && say "- none — every hardware log is tracked"

# ---------- 5. roadmap drift --------------------------------------------------
say ""
say "## Roadmap drift"
BLOCKED=$(grep -c '⛔' program/ROADMAP.md 2>/dev/null || echo 0)
INFLIGHT=$(grep -c '🔶' program/ROADMAP.md 2>/dev/null || echo 0)
TODO=$(grep -c '⬜' program/ROADMAP.md 2>/dev/null || echo 0)
say "- ⛔ blocked: $BLOCKED   🔶 in-flight: $INFLIGHT   ⬜ todo: $TODO"
LASTMOD=$(git log -1 --format=%cs -- program/ROADMAP.md 2>/dev/null)
say "- ROADMAP.md last touched: ${LASTMOD:-unknown}"
if [ -n "$LASTMOD" ]; then
  AGE=$(( ( $(date -u +%s) - $(date -u -d "$LASTMOD" +%s) ) / 86400 ))
  [ "$AGE" -gt 14 ] && flag "- **ROADMAP.md is $AGE days stale** while work has continued; status markers are probably lying."
fi

# ---------- 6. what is QUEUED for a human -------------------------------------
say ""
say "## Queued for a human (cannot be done unattended)"
BUSY=$(ps -eo pcpu,comm --sort=-pcpu | awk 'NR==2{print $1}')
say "- box busiest process: ${BUSY:-?}% of a core"
say "- measurements (\`ev run …\`) need a quiet box AND no resident agent session."
if [ ! -r "$ALICE_REPO/aegis_pruned_model.safetensors" ]; then
  say "- ⛔ \`ev run thread_sweep\` BLOCKED: model weights absent (gitignored; lost in the 2026-08-24 reset)."
fi
QUEUE=$(ev review-queue 2>/dev/null | tail -10)
[ -n "$QUEUE" ] && { say ""; say '```'; printf '%s\n' "$QUEUE" >> "$REPORT"; say '```'; }

# ---------- summary -----------------------------------------------------------
say ""
say "---"
say "**$FINDINGS finding(s) needing attention.**"
say ""
say "_No number in this report was produced by this run. Anything requiring measurement is queued above._"

echo "$REPORT"
[ "$FINDINGS" -gt 0 ] && exit 1 || exit 0
