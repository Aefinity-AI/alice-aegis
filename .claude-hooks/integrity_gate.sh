#!/usr/bin/env bash
# PostToolUse gate: after any edit to a .rs file, that file must not print a
# number it did not compute.
#
# Scoped to the edited FILE, not the repo, so a dead crate's historical sins
# (antigravity-aegis) cannot block work on the live engine. A gate that punishes
# unrelated work is a gate that gets removed.
#
# Exit 2 sends stderr back to Claude as a correction it must act on.
INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | python3 -c "
import json,sys
try: print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))
except Exception: print('')
" 2>/dev/null)

case "$FILE" in
  *.rs) ;;
  *) exit 0 ;;
esac
[ -f "$FILE" ] || exit 0

CHECK="$HOME/scripts/integrity_check.py"
[ -x "$CHECK" ] || exit 0

if ! OUT=$(python3 "$CHECK" "$FILE" 2>&1); then
    {
        echo "INTEGRITY GATE: $FILE prints a value it did not compute."
        echo
        echo "$OUT"
        echo
        echo "Every number printed must be computed in the same run. This project"
        echo "spent fifteen months shipping hardcoded benchmarks. Delete the code"
        echo "path, or make it measure."
    } >&2
    exit 2
fi
exit 0
