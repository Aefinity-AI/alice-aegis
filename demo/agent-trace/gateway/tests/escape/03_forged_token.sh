#!/usr/bin/env bash
# SAFE-5 escape test 3 (design doc #3):
# call a tool shim directly (bypassing the gateway entirely) with (i) no
# token and (ii) a forged/random token -- both must be refused.
#
# Run: bash tests/escape/03_forged_token.sh   (builds the shim if needed)
set -euo pipefail
cd "$(dirname "$0")/../.."   # demo/agent-trace/gateway

cargo build --offline --quiet
SHIM=target/debug/tool_shim_shell
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
KEY="$WORK/cap.key"
echo -n "escape-test-3-cap-key-not-for-prod" > "$KEY"
CONSUMED="$WORK/consumed.txt"
touch "$CONSUMED"

echo "== escape test 3: forged/missing capability token refused =="

echo "-- (i) no token --"
set +e
OUT1=$("$SHIM" "$KEY" "$CONSUMED" - echo pwned 2>&1)
RC1=$?
set -e
echo "$OUT1"
echo "exit=$RC1"

echo
echo "-- (ii) forged/random token --"
FORGED="idx=1/cap=$(printf 'ab%.0s' {1..32})/exp=9999999999"
set +e
OUT2=$("$SHIM" "$KEY" "$CONSUMED" "$FORGED" echo pwned 2>&1)
RC2=$?
set -e
echo "$OUT2"
echo "exit=$RC2"

if [ "$RC1" -ne 0 ] && echo "$OUT1" | grep -q "REFUSE: no capability token provided" \
   && [ "$RC2" -ne 0 ] && echo "$OUT2" | grep -q "REFUSE: capability token signature invalid"; then
    echo
    echo "PASS (fails closed): both no-token and forged-token calls refused, command never executed"
    exit 0
else
    echo
    echo "FAIL: expected both calls to be refused with the specific messages above"
    exit 1
fi
