#!/usr/bin/env bash
# SAFE-5: escape tests 1-4 (test 5, the two-box split, is out of scope for
# this branch — see ENFORCEMENT.md and the report this script's output is
# pasted into). Run from demo/agent-trace/gateway/ after `cargo build
# --release`. Writes PASS/FAIL + the exact refusal/error line for each
# test to stdout.
set -u
cd "$(dirname "$0")/../.."
GW=target/release
mkdir -p /tmp/cm-gateway-escape
KEY=/tmp/cm-gateway-escape/cap.key
head -c 32 /dev/urandom > "$KEY"

pass=0
fail=0

section() { echo; echo "=== $1 ==="; }

# ---------------------------------------------------------------------
# Test 1: curl / AF_INET from inside a sandboxed unit must fail.
# ---------------------------------------------------------------------
section "1. curl from inside sandbox (AF_INET denied)"
if command -v systemd-run >/dev/null && systemctl --user status >/dev/null 2>&1; then
  out=$(systemd-run --user --pty --collect \
    -p RestrictAddressFamilies=AF_UNIX -p PrivateNetwork=yes \
    -p NoNewPrivileges=yes -p ProtectSystem=strict -p ProtectHome=read-only \
    -p CapabilityBoundingSet= \
    /bin/sh -c 'curl -m 5 -v http://example.com 2>&1; echo RC=$?' 2>&1)
  echo "$out"
  if echo "$out" | grep -q "RC=7" && echo "$out" | grep -qi "after 0 ms"; then
    echo "RESULT: PASS (curl failed instantly -> syscall-level denial, not a timeout)"
    pass=$((pass+1))
  else
    echo "RESULT: FAIL (curl did not fail the expected way)"
    fail=$((fail+1))
  fi
else
  echo "RESULT: SKIPPED (systemd --user manager not available on this box)"
fi

# ---------------------------------------------------------------------
# Test 2: exec of a forbidden tool binary from inside the unit.
# HONEST NOTE: plain systemd sandboxing (CapabilityBoundingSet=,
# NoNewPrivileges=, ProtectSystem=strict) does NOT stop exec() of an
# arbitrary binary already reachable on PATH at the unit's own uid --
# that requires AppArmor/SELinux exec allow-listing (apparmor/cm-agent in
# this repo), which is NOT installed on penguin (`apparmor_status`: "not
# present"). What systemd DOES still enforce across that exec is the
# network/capability restriction, which we demonstrate here: the exec
# succeeds, but the exec'd process still cannot reach the network.
# ---------------------------------------------------------------------
section "2. exec /bin/bash from inside sandbox"
if command -v systemd-run >/dev/null && systemctl --user status >/dev/null 2>&1; then
  out=$(systemd-run --user --pty --collect \
    -p RestrictAddressFamilies=AF_UNIX -p PrivateNetwork=yes \
    -p NoNewPrivileges=yes -p ProtectSystem=strict -p ProtectHome=read-only \
    -p CapabilityBoundingSet= \
    /bin/sh -c 'exec /bin/bash -c "id; curl -m5 http://example.com; echo inner_rc=\$?"' 2>&1)
  echo "$out"
  if echo "$out" | grep -q "^uid=" && echo "$out" | grep -qi "inner_rc=7"; then
    echo "RESULT: PARTIAL — exec itself is NOT blocked by systemd alone (id printed = plain uid, no AppArmor on this box); network access from the exec'd process IS still denied (curl inner_rc=7). Full exec allow-listing requires apparmor/cm-agent on a box with AppArmor active (see ENFORCEMENT.md); this must be re-run on box1/box2 once AppArmor is installed there to get a true PASS on exec-blocking specifically."
    fail=$((fail+1))
  else
    echo "RESULT: FAIL (unexpected output)"
    fail=$((fail+1))
  fi
else
  echo "RESULT: SKIPPED (systemd --user manager not available on this box)"
fi

# ---------------------------------------------------------------------
# Test 3: tool shim called directly with a forged/unsigned token.
# ---------------------------------------------------------------------
section "3. shim with forged token -> SHIM REFUSE: bad mac"
CONSUMED=/tmp/cm-gateway-escape/shim3.consumed
rm -f "$CONSUMED"
CMD="echo escape-test-3"
AHASH=$(printf '%s' "$CMD" | sha256sum | awk '{print $1}')
out=$("$GW/tool_shim_shell" --idx 1 \
  --cap "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef" \
  --exp 9999999999 --action-hash "$AHASH" --key-file "$KEY" \
  --consumed-file "$CONSUMED" -- $CMD 2>&1)
echo "$out"
if echo "$out" | grep -q "^SHIM REFUSE: bad mac$"; then
  echo "RESULT: PASS"
  pass=$((pass+1))
else
  echo "RESULT: FAIL"
  fail=$((fail+1))
fi

# ---------------------------------------------------------------------
# Test 4: tool shim called with a previously consumed valid token.
# ---------------------------------------------------------------------
section "4. shim with replayed token -> SHIM REFUSE: already consumed"
CONSUMED=/tmp/cm-gateway-escape/shim4.consumed
rm -f "$CONSUMED"
CMD="echo escape-test-4"
AHASH=$(printf '%s' "$CMD" | sha256sum | awk '{print $1}')
EXP=9999999999
IDX=1
TOK=$(python3 - "$KEY" "$IDX" "$AHASH" "$EXP" <<'PYEOF'
import hmac, hashlib, struct, sys
keyfile, idx, ahash_hex, exp = sys.argv[1:5]
key = open(keyfile, 'rb').read()
idx = int(idx); exp = int(exp)
ahash = bytes.fromhex(ahash_hex)
msg = struct.pack(">Q", idx) + b"\x00" + ahash + b"\x00" + struct.pack(">Q", exp)
print(hmac.new(key, msg, hashlib.sha256).hexdigest())
PYEOF
)
echo "issued token: $TOK"
first=$("$GW/tool_shim_shell" --idx "$IDX" --cap "$TOK" --exp "$EXP" --action-hash "$AHASH" \
  --key-file "$KEY" --consumed-file "$CONSUMED" -- $CMD 2>&1)
echo "first call: $first"
second=$("$GW/tool_shim_shell" --idx "$IDX" --cap "$TOK" --exp "$EXP" --action-hash "$AHASH" \
  --key-file "$KEY" --consumed-file "$CONSUMED" -- $CMD 2>&1)
echo "second call (replay): $second"
if echo "$first" | grep -q "^escape-test-4$" && echo "$second" | grep -q "^SHIM REFUSE: already consumed$"; then
  echo "RESULT: PASS"
  pass=$((pass+1))
else
  echo "RESULT: FAIL"
  fail=$((fail+1))
fi

section "SUMMARY"
echo "pass=$pass fail=$fail (test 2 counted as fail: exec-blocking needs AppArmor, not present on this box)"
echo "test 5 (box1 reaching box2's credential store outside the forced channel) is OUT OF SCOPE here — no inter-box forced channel exists yet; run on box1+box2 once (d) is built."
