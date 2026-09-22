#!/usr/bin/env bash
# SAFE-5 escape test 2 (design doc #2):
# exec of a forbidden tool binary from inside the unit must fail, and it
# must fail because AppArmor denies the exec (LSM-level), not because the
# binary is simply missing from PATH.
#
# Requires: passwordless (or interactive) sudo; the `apparmor` package
# with a kernel that has AppArmor enabled (/sys/module/apparmor/parameters/enabled
# == "Y"); the agent-sandbox profile loaded
# (`sudo apparmor_parser -r systemd/agent-sandbox.apparmor`).
#
# Run: sudo bash tests/escape/02_no_exec_outside_shims.sh
set -uo pipefail

echo "== escape test 2: exec of a forbidden binary denied by AppArmor =="

if [ "$(cat /sys/module/apparmor/parameters/enabled 2>/dev/null)" != "Y" ]; then
    echo "SKIP: AppArmor not enabled on this kernel"
    exit 77
fi

if ! sudo aa-status 2>/dev/null | grep -q '^   agent-sandbox$'; then
    echo "profile not loaded, loading it now"
    sudo apparmor_parser -r "$(dirname "$0")/../../systemd/agent-sandbox.apparmor"
fi

echo "-- sanity: /bin/bash actually exists (so a denial can't be a missing-binary artifact) --"
ls -l /bin/bash

echo "-- confined exec of /bin/bash (not on the allow-list) --"
OUT=$(sudo aa-exec -p agent-sandbox -- /bin/sh -c '/bin/bash --version; echo shim_test_exit=$?' 2>&1)
echo "$OUT"

echo
echo "-- confined exec of an allowed path (sh itself, via the profile's own Cx rule) works, to show this isn't a global sh failure --"
OUT2=$(sudo aa-exec -p agent-sandbox -- /bin/sh -c 'echo sh-still-works' 2>&1)
echo "$OUT2"

if echo "$OUT" | grep -q "Permission denied" && echo "$OUT" | grep -q "shim_test_exit=126" && echo "$OUT2" | grep -q "sh-still-works"; then
    echo
    echo "PASS (fails closed): /bin/bash exec denied (Permission denied, exit 126) under the agent-sandbox AppArmor profile, while the profile's own allowed /bin/sh still runs"
    exit 0
else
    echo
    echo "FAIL: expected /bin/bash exec to be denied by AppArmor while /bin/sh still runs"
    exit 1
fi
