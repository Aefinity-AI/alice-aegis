#!/usr/bin/env bash
# SAFE-5c relay-fix regression test: proves gateway-relay.sh's "MODE
# GATEWAY" header line actually reaches the gateway-dispatch branch
# instead of falling through to the catch-all DENY.
#
# Background bug (see state/reports/2026-09-14-safe5c-escape5-box1.md,
# and claudius-maximus's follow-up report): the original script did
# `read -r header rest`, which word-splits "MODE GATEWAY" on IFS
# whitespace into header="MODE" rest="GATEWAY". The case statement below
# it matches the full two-word literal "MODE GATEWAY" against $header,
# which is only ever "MODE" -- so it NEVER matches, and every request
# (legitimate or not) falls through to the `*) echo "DENY unrecognized
# MODE"` branch. This is fail-closed (not a security hole) but it means
# the relay could not dispatch ANY legitimate request either.
#
# This test does not need a live gateway socket: it stubs `socat` with a
# fake that just proves it was invoked (writes a sentinel line and
# exits), and points XDG_RUNTIME_DIR at a scratch dir so the script's
# $SOCK path resolution succeeds without a real unix socket existing.
#
# Run: bash tests/inter_box_relay_parsing.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # demo/agent-trace/gateway

RELAY="deploy/inter-box/gateway-relay.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Fake `socat` on PATH: real gateway-relay.sh does
#   exec socat - "UNIX-CONNECT:${SOCK}"
# We don't need a real socket -- just need to observe this branch was
# actually reached (as opposed to the DENY catch-all).
mkdir -p "$TMP/bin"
cat > "$TMP/bin/socat" <<'EOF'
#!/bin/sh
echo "GATEWAY_DISPATCH_REACHED"
EOF
chmod +x "$TMP/bin/socat"

export PATH="$TMP/bin:$PATH"
export XDG_RUNTIME_DIR="$TMP/run"
mkdir -p "$XDG_RUNTIME_DIR"

echo "== inter-box relay parsing: MODE GATEWAY should reach gateway dispatch, not DENY =="

set +e
OUT="$(printf 'MODE GATEWAY\nsome request body\n' | "$RELAY" 2>&1)"
RC=$?
set -e

echo "$OUT"
echo "exit code: $RC"

if echo "$OUT" | grep -q "GATEWAY_DISPATCH_REACHED"; then
    echo
    echo "PASS: MODE GATEWAY header correctly dispatched to the gateway socket relay"
elif echo "$OUT" | grep -q "DENY"; then
    echo
    echo "FAIL: legitimate 'MODE GATEWAY' request hit the DENY catch-all (header/rest split bug)"
    exit 1
else
    echo
    echo "FAIL: unexpected output, neither dispatch nor DENY observed"
    exit 1
fi

echo
echo "== sanity: unrecognized MODE still denies (fail-closed preserved) =="
set +e
OUT2="$(printf 'MODE BOGUS\n' | "$RELAY" 2>&1)"
RC2=$?
set -e
echo "$OUT2"
echo "exit code: $RC2"

if [ "$RC2" -ne 0 ] && echo "$OUT2" | grep -q "DENY unrecognized MODE"; then
    echo "PASS: unrecognized MODE still denied"
else
    echo "FAIL: unrecognized MODE did not deny as expected"
    exit 1
fi
