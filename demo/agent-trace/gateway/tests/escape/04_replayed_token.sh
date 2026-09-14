#!/usr/bin/env bash
# SAFE-5 escape test 4 (design doc #4):
# call a tool shim directly with a previously consumed, otherwise VALID
# token -- the second call must be refused even though the token's HMAC
# and expiry are both fine.
#
# Uses cap_issue_for_test to mint a genuinely valid token via the exact
# same gateway::capability::issue code path the daemon uses (see that
# binary's own doc comment for why this is not a parallel/fake
# implementation of the token format).
#
# NOTE: updated 2026-09-14 for this branch's actual tool_shim_shell CLI
# (--idx/--cap/--exp/--action-hash/--key-file/--consumed-file --
# <command...>), which differs from box1's original pre-merge shim
# signature.
#
# Run: bash tests/escape/04_replayed_token.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # demo/agent-trace/gateway

cargo build --offline --quiet 2>/dev/null || cargo build --quiet
SHIM=target/debug/tool_shim_shell
ISSUE=target/debug/cap_issue_for_test
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
KEY="$WORK/cap.key"
head -c 32 /dev/urandom > "$KEY"
CONSUMED="$WORK/consumed.txt"
touch "$CONSUMED"

EXPIRY=$(( $(date +%s) + 300 ))
IDX=7
CMD="echo escape-test-4-marker"
AHASH=$(printf '%s' "$CMD" | sha256sum | awk '{print $1}')
TOKEN=$("$ISSUE" "$KEY" "$IDX" "$EXPIRY" shell $CMD)
echo "== escape test 4: replay of a consumed, otherwise-valid token refused =="
echo "minted token: $TOKEN"

echo
echo "-- first use: must ALLOW and actually run --"
set +e
OUT1=$("$SHIM" --idx "$IDX" --cap "$TOKEN" --exp "$EXPIRY" --action-hash "$AHASH" \
  --key-file "$KEY" --consumed-file "$CONSUMED" -- $CMD 2>&1)
RC1=$?
set -e
echo "$OUT1"
echo "exit=$RC1"

echo
echo "-- second use of the SAME token: must be refused (single-use) --"
set +e
OUT2=$("$SHIM" --idx "$IDX" --cap "$TOKEN" --exp "$EXPIRY" --action-hash "$AHASH" \
  --key-file "$KEY" --consumed-file "$CONSUMED" -- $CMD 2>&1)
RC2=$?
set -e
echo "$OUT2"
echo "exit=$RC2"

if [ "$RC1" -eq 0 ] && echo "$OUT1" | grep -q "escape-test-4-marker" \
   && [ "$RC2" -ne 0 ] && echo "$OUT2" | grep -q "SHIM REFUSE: already consumed"; then
    echo
    echo "PASS (fails closed): first use ALLOWed and ran, replay of the same token refused"
    exit 0
else
    echo
    echo "FAIL: expected first use to succeed and the replay to be refused as already-consumed"
    exit 1
fi
