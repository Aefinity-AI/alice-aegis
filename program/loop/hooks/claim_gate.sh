#!/usr/bin/env bash
# PostToolUse claim gate — the deterministic layer for NUMBERS IN PROSE.
#
# The existing integrity_gate.sh asks of an edited .rs file: "does it PRINT a
# number it did not compute?" That is the right question and it stays.
# This asks the question it cannot: "does this file ASSERT a number the ledger
# has already killed?"
#
# The gap is not hypothetical. aegis-core/src/ops.rs:1240 carries `~2 GB/s
# against a 17.3 GB/s ceiling` in a doc comment — a figure with NO source
# anywhere in the repository — and ops.rs:621 carries "8 workers decode ~5%
# faster than 4", written by the same unlogged commit (254ba43) that produced
# the retracted 8.25 tok/s, still shipping as the engine's default worker count.
# Neither is printed at runtime, so integrity_gate.sh is silent on both. Neither
# is in a document, so every document-level audit is silent on both.
#
# Wire it in .claude/settings.json alongside the existing hook (see
# program/loop/hooks/settings.snippet.json). Exit 2 returns stderr to Claude as
# a correction it must act on.
#
# Scoped to the EDITED FILE, never the repo — for the same reason
# integrity_gate.sh is: a gate that punishes unrelated work gets removed.
INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))
except Exception: print('')
" 2>/dev/null)

case "$FILE" in
  *.md|*.rs|*.txt) ;;
  *) exit 0 ;;
esac
[ -f "$FILE" ] || exit 0

# Never gate the primary record. Logs and runcards are evidence: a log
# containing a number that was later retracted is the whole point of keeping it.
case "$FILE" in
  */docs/hardware_logs/*|*/runcards/*|*/state/*|*claims.jsonl) exit 0 ;;
esac

LINT="$HOME/program/loop/tools/claimlint.py"
[ -x "$LINT" ] || exit 0

# Tier 1 only in the hook. Tier 2 (--strict) is a submission-time gate run by
# hand via `ev gate`; making it blocking on every edit would demand a fully
# populated ledger before a single sentence could be written, and a gate you
# cannot satisfy is a gate you disable.
if ! OUT=$(python3 "$LINT" "$FILE" 2>&1); then
    {
        echo "CLAIM GATE: $FILE states a number this program has retracted or superseded."
        echo
        echo "$OUT"
        echo
        echo "Options, in order of preference:"
        echo "  1. Cite the live replacement the finding names."
        echo "  2. Delete the number."
        echo "  3. Keep it AND say on the same line (or the one before/after) that it is"
        echo "     retracted/superseded — the honest-retraction form the ledger uses."
        echo "  4. If the number is genuinely live and the ledger is wrong, fix the LEDGER"
        echo "     first (ev claim add), with the log path. Not the sentence."
        echo
        echo "Do not reword the span to slip past the matcher. That is the failure mode"
        echo "this gate exists to prevent, and it is checked again at submission."
    } >&2
    exit 2
fi
exit 0
