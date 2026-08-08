#!/bin/bash
# Pull the transcript off the boot stick after a real-hardware run, verbatim.
#
#   ./collect_hardware_run.sh [device]     (default: /dev/sda)
#
# Read-only. Copies BOOTLOG.TXT into docs/hardware_logs/ with a timestamp,
# prints it, and extracts the numbers so nothing gets transcribed by hand.
# A number typed from memory is a number that can drift.
set -u
DEV="${1:-/dev/sda}"
OUT=~/docs/hardware_logs
mkdir -p "$OUT"

if ! mdir -i "$DEV" :: >/dev/null 2>&1; then
    echo "ERROR: $DEV is not readable as a FAT volume. Is the stick inserted?"
    exit 1
fi

STAMP=$(date +%Y-%m-%d_%H%M%S)
DEST="$OUT/dell_run_$STAMP.txt"

if ! mtype -i "$DEV" ::/BOOTLOG.TXT > "$DEST" 2>/dev/null || [ ! -s "$DEST" ]; then
    echo "ERROR: no BOOTLOG.TXT on $DEV — the run wrote nothing."
    rm -f "$DEST"
    exit 1
fi

echo "════════════════ VERBATIM TRANSCRIPT ════════════════"
cat "$DEST"
echo "═════════════════════════════════════════════════════"
echo "saved: $DEST"
echo

echo "── extracted ──"
grep -a "SIMD=" "$DEST" | tail -1 | sed 's/^/  /'
LAST=$(grep -ac "^STAGE" "$DEST"); echo "  stages logged: $LAST"

if grep -qa "^RESPONSE:" "$DEST"; then
    echo "  ✅ a generation was recorded"
    grep -a "^PROMPT:"   "$DEST" | tail -1 | sed 's/^/  /'
    grep -a "^RESPONSE:" "$DEST" | tail -1 | sed 's/^/  /'
    grep -aE "^  \([0-9]+ tokens" "$DEST" | tail -1 | sed 's/^/  /'
    echo
    if grep -aiq "paris" "$DEST"; then
        echo "  P2 (emits \"Paris\"): SUPPORTED by the transcript"
    else
        echo "  P2 (emits \"Paris\"): NOT SUPPORTED — the model said something else."
        echo "     Record it as a falsification. Do not re-run until it is explained."
    fi
else
    echo "  ⚠ no RESPONSE line — either the run used the old build, or no prompt was entered."
fi

if grep -qa "^BENCHMARK:" "$DEST"; then
    echo
    grep -a "^BENCHMARK:" "$DEST" | tail -1 | sed 's/^/  /'
    echo "  NOTE: process_intent() timing includes prefill. This is NOT a decode rate."
fi
echo
echo "Next: paste this into docs/PREREGISTERED_HARDWARE_TEST.md under 'Result',"
echo "whatever it says, and commit it."
