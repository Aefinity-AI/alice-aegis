#!/usr/bin/env bash
# SAFE-5 escape test 3 (design doc #3):
# call a tool shim directly (bypassing the gateway entirely) with (i) no
# token and (ii) a forged/random token -- both must be refused.
#
# NOTE: updated 2026-09-14 for this branch's actual tool_shim_shell CLI
# (--idx/--cap/--exp/--action-hash/--key-file/--consumed-file --
# <command...>), which differs from box1's original pre-merge shim
# signature. See tests/escape/run.sh's tests 3-4 for the same coverage.
#
# Run: bash tests/escape/03_forged_token.sh   (builds the shim if needed)
set -euo pipefail
cd "$(dirname "$0")/../.."   # demo/agent-trace/gateway

cargo build --offline --quiet 2>/dev/null || cargo build --quiet
SHIM=target/debug/tool_shim_shell
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
KEY="$WORK/cap.key"
head -c 32 /dev/urandom > "$KEY"
CONSUMED="$WORK/consumed.txt"
touch "$CONSUMED"
CMD="echo pwned"
AHASH=$(printf '%s' "$CMD" | sha256sum | awk '{print $1}')

echo "== escape test 3: forged/missing capability token refused =="

echo "-- (i) no token at all --"
set +e
OUT1=$("$SHIM" --key-file "$KEY" --consumed-file "$CONSUMED" -- $CMD 2>&1)
RC1=$?
set -e
echo "$OUT1"
echo "exit=$RC1"

echo
echo "-- (ii) forged/random token (correct action-hash, garbage MAC) --"
FORGED=$(printf 'ab%.0s' {1..32})
set +e
OUT2=$("$SHIM" --idx 1 --cap "$FORGED" --exp 9999999999 --action-hash "$AHASH" \
  --key-file "$KEY" --consumed-file "$CONSUMED" -- $CMD 2>&1)
RC2=$?
set -e
echo "$OUT2"
echo "exit=$RC2"

if [ "$RC1" -ne 0 ] && echo "$OUT1" | grep -q "SHIM REFUSE: no token" \
   && [ "$RC2" -ne 0 ] && echo "$OUT2" | grep -q "SHIM REFUSE: bad mac"; then
    echo
    echo "PASS (fails closed): both no-token and forged-token calls refused, command never executed"
    exit 0
else
    echo
    echo "FAIL: expected both calls to be refused with the specific messages above"
    exit 1
fi
