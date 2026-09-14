#!/usr/bin/env bash
# SAFE-5 escape test 1 (design doc "the 5 escape tests" #1):
# curl / AF_INET from inside the sandboxed agent unit must fail, and it
# must fail because the socket family is refused (EAFNOSUPPORT), not
# because of an incidental DNS/PATH failure inside an empty network
# namespace (that would prove nothing about the actual enforcement).
#
# Requires: passwordless (or interactive) sudo for `systemd-run` with the
# same RestrictAddressFamilies=AF_UNIX / PrivateNetwork=yes properties
# agent-sandbox.service applies; strace to inspect the real syscall/errno.
#
# Run: sudo bash tests/escape/01_no_af_inet.sh
set -euo pipefail

echo "== escape test 1: AF_INET refused from inside the sandbox =="

echo "-- curl to an IP literal (bypasses DNS so failure can't be blamed on resolution) --"
set +e
OUT=$(sudo systemd-run --pipe --wait --quiet \
    --property=PrivateNetwork=yes \
    --property=RestrictAddressFamilies=AF_UNIX \
    -- curl -sv -m 5 http://93.184.216.34/ 2>&1)
RC=$?
set -e
echo "$OUT"
echo "curl exit code: $RC"

echo
echo "-- strace confirms the actual kernel-level refusal (not DNS/PATH) --"
STRACE_OUT=$(sudo systemd-run --pipe --wait --quiet \
    --property=RestrictAddressFamilies=AF_UNIX \
    -- strace -f -e trace=network curl -s -m 5 http://93.184.216.34/ 2>&1 | grep -E "socket\(AF_INET|EAFNOSUPPORT" || true)
echo "$STRACE_OUT"

if echo "$STRACE_OUT" | grep -q "EAFNOSUPPORT"; then
    echo
    echo "PASS (fails closed): AF_INET socket() denied with EAFNOSUPPORT under RestrictAddressFamilies=AF_UNIX"
    exit 0
else
    echo
    echo "FAIL: expected an EAFNOSUPPORT on socket(AF_INET,...) -- enforcement did not fail closed as expected"
    exit 1
fi
