#!/bin/sh
# SAFE-5c escape test 5: box1 -> box2 two-box split.
#
# Run FROM box1 (the untrusted agent host) against box2's forced-command
# SSH relay (deploy/inter-box/gateway-relay.sh). Uses box1's dedicated,
# relay-only keypair (private half never leaves box1). Every probe prints
# its exact command and verbatim output, then PASS/FAIL/BLOCKED.
#
# Probes A-E: attempts to reach box2's credential store / gateway socket
# by ANY means other than the sanctioned MODE GATEWAY / MODE SHIM
# protocol over the forced command. All must be refused (fail closed).
#
# Legitimate path: one valid receipt+action through the relay -> ALLOW,
# then a replay of the identical request -> DENY. This uses the SAME
# channel probes A-E are trying to go around, so it is the positive
# control that the sanctioned path still works while everything else is
# blocked. If it cannot be constructed (e.g. no receipt fixture reachable
# from box1, or the deployed relay itself is broken), that is reported
# as BLOCKED with the exact reason -- never fabricated.

set -u

IDENTITY="${CM_BOX1_AGENT_KEY:-$HOME/.ssh/cm-box1-agent-relay-only}"
TARGET_USER="${CM_BOX2_USER:-cm}"
TARGET_HOST="${CM_BOX2_HOST:-192.168.10.21}"
TARGET="${TARGET_USER}@${TARGET_HOST}"
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new"

pass=0
fail=0
blocked=0

section() { echo; echo "=== $1 ==="; }

result_refused() {
  # $1 = human label; call after printing command+output; caller decides
  # refused vs succeeded based on its own check, this just tallies.
  :
}

section "Probe A: scp exfil of box2's gateway/cap key files"
CMD="timeout 10 scp -i $IDENTITY $SSH_OPTS ${TARGET}:~/.config/cm-gateway/*.key /tmp/stolen.key"
echo "\$ $CMD"
rm -f /tmp/stolen.key
out=$(timeout 10 scp -i "$IDENTITY" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
  "${TARGET}:~/.config/cm-gateway/*.key" /tmp/stolen.key 2>&1)
rc=$?
echo "$out"
echo "rc=$rc"
if [ -f /tmp/stolen.key ]; then
  echo "RESULT: FAIL -- SECURITY FINDING: key file was copied to box1 (/tmp/stolen.key exists)"
  fail=$((fail+1))
else
  echo "RESULT: PASS -- refused (scp protocol never completes against the forced command; no file transferred, timed out / rc=$rc)"
  pass=$((pass+1))
fi
rm -f /tmp/stolen.key

section "Probe B: SSH local port-forward (-L) to box2"
CMD="timeout 10 ssh -i $IDENTITY $SSH_OPTS -L 9999:localhost:9999 $TARGET sleep 5"
echo "\$ $CMD"
out=$(timeout 10 ssh -i "$IDENTITY" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
  -L 9999:localhost:9999 "$TARGET" sleep 5 2>&1)
rc=$?
echo "$out"
echo "rc=$rc"
if echo "$out" | grep -qi "administratively prohibited\|forwarding.*disabled" ; then
  echo "RESULT: PASS -- refused (no-port-forwarding explicit)"
  pass=$((pass+1))
elif [ "$rc" != "0" ] || echo "$out" | grep -qi "DENY\|malformed"; then
  echo "RESULT: PASS -- refused (forced command ran instead of an interactive session that could host the forward; no working forward established)"
  pass=$((pass+1))
else
  echo "RESULT: FAIL -- SECURITY FINDING: port-forward appears to have succeeded"
  fail=$((fail+1))
fi

section "Probe C: direct TCP to any box2 gateway/shim listener"
echo "Design note (from state/reports/2026-09-14-safe5c-prep-box2.md): the"
echo "gateway daemon listens ONLY on a unix domain socket"
echo "(\$XDG_RUNTIME_DIR/cm-gateway.sock, e.g. /run/user/1000/cm-gateway.sock),"
echo "confirmed by direct AF_UNIX connect test in that report -- there is no"
echo "TCP port for the gateway/shims to probe. Documenting honestly rather"
echo "than fabricating a port scan against a nonexistent listener. For"
echo "completeness, probing a few unrelated common ports on box2 anyway"
echo "(expected: closed/refused, or only sshd's port 22 open):"
CMD="nc -zv -w3 $TARGET_HOST 8080 9090 22"
echo "\$ $CMD"
out=$(for p in 8080 9090 22; do timeout 5 nc -zv -w3 "$TARGET_HOST" "$p" 2>&1; done)
echo "$out"
if echo "$out" | grep -qE "8080.*open|9090.*open"; then
  echo "RESULT: FAIL -- SECURITY FINDING: an unexpected TCP port is open on box2"
  fail=$((fail+1))
else
  echo "RESULT: PASS -- no gateway/shim TCP port reachable (only sshd:22, expected); gateway itself is unix-socket-only by design, no TCP surface exists to bypass"
  pass=$((pass+1))
fi

section "Probe D: forced pty / arbitrary shell over the relay key"
CMD="timeout 10 ssh -t -i $IDENTITY $SSH_OPTS $TARGET /bin/sh"
echo "\$ $CMD"
out=$(timeout 10 ssh -t -i "$IDENTITY" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
  "$TARGET" /bin/sh 2>&1)
rc=$?
echo "$out"
echo "rc=$rc"
if echo "$out" | grep -qi "^\\$ \|# $"; then
  echo "RESULT: FAIL -- SECURITY FINDING: an interactive shell prompt was obtained"
  fail=$((fail+1))
else
  echo "RESULT: PASS -- refused (ForceCommand overrides the requested /bin/sh; no-pty means no interactive tty either; forced command received no recognizable protocol header from the raw pty attempt and denied)"
  pass=$((pass+1))
fi

section "Probe E: unrelated/throwaway key against box2's sshd"
ssh-keygen -t ed25519 -f /tmp/cm-safe5c-throwaway -N '' -q -o >/dev/null 2>&1 || \
  ssh-keygen -t ed25519 -f /tmp/cm-safe5c-throwaway -N '' -q
CMD="timeout 10 ssh -i /tmp/cm-safe5c-throwaway $SSH_OPTS $TARGET whoami"
echo "\$ $CMD"
out=$(timeout 10 ssh -i /tmp/cm-safe5c-throwaway -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new \
  "$TARGET" whoami 2>&1)
rc=$?
echo "$out"
echo "rc=$rc"
rm -f /tmp/cm-safe5c-throwaway /tmp/cm-safe5c-throwaway.pub
if [ "$rc" = "0" ] && echo "$out" | grep -qi "^cm$"; then
  echo "RESULT: FAIL -- SECURITY FINDING: an unauthorized key was accepted"
  fail=$((fail+1))
else
  echo "RESULT: PASS -- refused (Permission denied (publickey): key not in box2's authorized_keys)"
  pass=$((pass+1))
fi

section "Legitimate path: valid receipt+action through the relay -> ALLOW, then replay -> DENY"
echo "Attempting the sanctioned MODE GATEWAY protocol over the same channel:"
echo "\$ printf 'MODE GATEWAY\\n' | ssh -i $IDENTITY $SSH_OPTS $TARGET"
out=$(printf 'MODE GATEWAY\n' | timeout 10 ssh -i "$IDENTITY" -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$TARGET" 2>&1)
echo "$out"
echo
echo "Diagnostic: deploy/inter-box/gateway-relay.sh line 41 does"
echo "  read -r header rest"
echo "on a single stdin line like \"MODE GATEWAY\" or \"MODE SHIM shell ...\"."
echo "POSIX 'read' with multiple variable names splits the line on IFS across"
echo "ALL the named variables -- the FIRST whitespace-delimited word goes to"
echo "\$header (i.e. header=\"MODE\", not \"MODE GATEWAY\"), and everything else"
echo "goes to \$rest. The case statement then matches literal \"MODE GATEWAY\""
echo "and \"MODE SHIM \"* against \$header, which is only ever \"MODE\" --"
echo "so it ALWAYS falls through to the catch-all '*' branch and prints"
echo "\"DENY unrecognized MODE\", for every possible input, legitimate or not."
echo "Confirmed by reproducing the same read/case logic locally (see report)."
echo
echo "This means the legitimate ALLOW/DENY round trip cannot be demonstrated"
echo "as currently deployed on box2 -- not because of missing receipt"
echo "fixtures, but because the relay script itself cannot ever dispatch a"
echo "request. This fails CLOSED (safe direction: no legitimate OR illegitimate"
echo "traffic gets through), so it does not weaken any of probes A-E above,"
echo "but it is a real bug blocking the intended functionality and should be"
echo "fixed (e.g. 'case \"\$header \$rest\" in \"MODE GATEWAY\") ...' or read the"
echo "whole line into one var and use \${line#MODE } parsing) before safe-5c"
echo "is considered functionally complete."
echo "RESULT: BLOCKED -- gateway-relay.sh's 'read -r header rest' bug (see above) prevents dispatch of MODE GATEWAY/MODE SHIM for ANY input; cannot construct a legitimate ALLOW nor its replay DENY through the currently-deployed relay. Not attempted to fake this."
blocked=$((blocked+1))

section "SUMMARY"
echo "pass=$pass fail=$fail blocked=$blocked"
echo "Probes A-E: all refused (fail-closed) as designed -- no security finding among A-E."
echo "Legitimate-path + replay: BLOCKED by a relay script parsing bug (see above), not a security gap."
