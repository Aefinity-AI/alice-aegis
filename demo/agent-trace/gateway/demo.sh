#!/usr/bin/env bash
# safe-8: one-command, end-to-end design-partner demo of the receipt
# verification gateway, against the REAL BitNet-2B artifacts on this box.
#
# What it proves, in order:
#   1. A real receipt (generated against the real 2B model) submitted
#      through the async gateway (RECEIPT/ACTION/SESSION/COUNTER ->
#      PENDING ticket=T... -> POLL ... -> ALLOW) is genuinely re-verified
#      by a background `agent_trace verify` subprocess against the real
#      model, not rubber-stamped -- the ALLOW takes real wall-clock
#      minutes and prints a capability token.
#   2. Replaying the identical (session, counter, action) is DENYed on
#      freshness grounds: the gateway will not let the same tool call be
#      authorized twice.
#   3. A one-byte tamper inside a trace step's decode-chain hash (a field
#      NOT used by the gateway's cheap inline precheck, only by the real
#      model-based verify that runs in the background) is caught by the
#      background worker and DENYed once verification actually runs --
#      proving the async path really does the real check, not just the
#      cheap parse/allowlist/freshness checks.
#   4. The final ALLOW'd receipt can be independently re-verified by a
#      THIRD PARTY on ANY machine (a phone, a laptop with no network path
#      to this gateway at all) using nothing but the receipt file and the
#      three public artifact files, via the standalone `agent_trace
#      verify` CLI -- the same binary/algorithm the gateway's worker used,
#      run standalone with no gateway, no socket, no daemon.
#
# Usage: CM_2B_ARTIFACTS=/path/to/bitnet2b-2b-artifacts bash demo.sh
# Requires: CM_2B_ARTIFACTS set to a directory holding the real 2B
#   artifacts (aegis_pruned_model.cis.safetensors, embed.bin, vocab.bin)
# and a Rust toolchain (cargo) on PATH to build the two release binaries
# if they are not already built.
set -euo pipefail
cd "$(dirname "$0")"   # demo/agent-trace/gateway

ART=${CM_2B_ARTIFACTS:?set CM_2B_ARTIFACTS to the directory holding aegis_pruned_model.cis.safetensors, embed.bin, vocab.bin}
MODEL="$ART/aegis_pruned_model.cis.safetensors"
EMBED="$ART/embed.bin"
VOCAB="$ART/vocab.bin"
RECEIPT_SRC="tests/fixtures/live20/calc_01.receipt"
REPO_ROOT="$(cd ../../.. && pwd)"
AGENT_TRACE_BIN="$REPO_ROOT/aegis-linux/target/release/examples/agent_trace"
GATEWAY_BIN="target/release/gateway"
# The two paths above are fixed; a CARGO_TARGET_DIR in the environment would
# move the artifacts elsewhere and make the checks below fail after a clean build.
unset CARGO_TARGET_DIR
# Real 2B verify is 6-7+ min single-threaded on weak hardware (see gateway
# src/main.rs); default well above that, override with DEMO_VERIFY_TIMEOUT.
DEMO_VERIFY_TIMEOUT="${DEMO_VERIFY_TIMEOUT:-900}"
export DEMO_VERIFY_TIMEOUT

if [ ! -f "$MODEL" ]; then
    echo "FATAL: real 2B artifacts not found at $ART" >&2
    exit 1
fi

echo "== step 0: build release binaries (no-op if already current) =="
WORK_BUILDLOG1=$(mktemp)
WORK_BUILDLOG2=$(mktemp)
( cd "$REPO_ROOT/aegis-linux" && cargo build --release --offline --example agent_trace ) \
    > "$WORK_BUILDLOG1" 2>&1 || { cat "$WORK_BUILDLOG1" >&2; rm -f "$WORK_BUILDLOG1" "$WORK_BUILDLOG2"; exit 1; }
cargo build --release --offline --bin gateway > "$WORK_BUILDLOG2" 2>&1 || { cat "$WORK_BUILDLOG2" >&2; rm -f "$WORK_BUILDLOG1" "$WORK_BUILDLOG2"; exit 1; }
rm -f "$WORK_BUILDLOG1" "$WORK_BUILDLOG2"
echo "(cargo build output suppressed; both binaries built cleanly)"

if [ ! -x "$AGENT_TRACE_BIN" ]; then
    echo "FATAL: expected $AGENT_TRACE_BIN after build" >&2
    exit 1
fi
if [ ! -x "$GATEWAY_BIN" ]; then
    echo "FATAL: expected $GATEWAY_BIN after build" >&2
    exit 1
fi

WORK="/tmp/safe8-demo-$$"
mkdir -p "$WORK"
DAEMON_PID=""
cleanup() {
    if [ -n "$DAEMON_PID" ]; then
        # gateway runs in its own session (setsid): kill the whole group so an
        # in-flight agent_trace verify child is not orphaned
        kill -- -"$DAEMON_PID" >/dev/null 2>&1 || kill "$DAEMON_PID" >/dev/null 2>&1 || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

echo "== step 1: real live receipt (already generated against the real 2B" \
     "model; see $RECEIPT_SRC) =="
cp "$RECEIPT_SRC" "$WORK/receipt.orig"
cat "$WORK/receipt.orig"

echo
echo "== step 2: sign an allowlist for this box's real artifact hashes =="
MH=$(sha256sum "$MODEL" | cut -d' ' -f1)
EH=$(sha256sum "$EMBED" | cut -d' ' -f1)
VH=$(sha256sum "$VOCAB" | cut -d' ' -f1)
ALLOW_KEY="$WORK/allow.key"
CAP_KEY="$WORK/cap.key"
echo -n "safe8-demo-allow-key-not-for-prod" > "$ALLOW_KEY"
echo -n "safe8-demo-cap-key-not-for-prod" > "$CAP_KEY"
python3 - "$WORK/allowlist.signed" "$MH" "$EH" "$VH" "$ALLOW_KEY" <<'PYEOF'
import sys, hmac, hashlib
path, mh, eh, vh, keypath = sys.argv[1:6]
key = open(keypath, 'rb').read()
line = f"{mh} {eh} {vh}"
sig = hmac.new(key, (line + "\n").encode(), hashlib.sha256).hexdigest()
open(path, 'w').write(f"{line}\nsig {sig}\n")
PYEOF
cat "$WORK/allowlist.signed"

echo
echo "== step 3: start the gateway daemon (real 2B model, worker pool) =="
SOCK="$WORK/gateway.sock"
setsid "$GATEWAY_BIN" serve "$MODEL" "$EMBED" "$VOCAB" "$AGENT_TRACE_BIN" \
    "$WORK/allowlist.signed" "$ALLOW_KEY" \
    --socket "$SOCK" --cap-key-file "$CAP_KEY" \
    --anchor-every 100 --anchor-file "$WORK/anchor.log" \
    --verify-workers 1 --cap-ttl 3600 \
    > "$WORK/gateway.log" 2>&1 &
DAEMON_PID=$!

for _ in $(seq 1 50); do
    [ -S "$SOCK" ] && break
    sleep 0.2
done
if [ ! -S "$SOCK" ]; then
    echo "FATAL: gateway did not open $SOCK; see $WORK/gateway.log" >&2
    cat "$WORK/gateway.log" >&2 || true
    exit 1
fi
echo "gateway listening: $(grep 'listening on' "$WORK/gateway.log" || true)"

ACTION_HEX=$(tr ' ' '\n' < "$WORK/receipt.orig" | grep '^in=' | tail -1 | cut -d= -f2)

client() {
    # $1 = python client mode ("decide" or "poll"), remaining args passed
    # through as sys.argv to the inline client below.
    python3 - "$SOCK" "$@" <<'PYEOF'
import os, socket, sys, time

sock_path = sys.argv[1]
mode = sys.argv[2]

def send(payload: bytes) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(sock_path)
    s.sendall(payload)
    s.shutdown(socket.SHUT_WR)
    out = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        out += chunk
    s.close()
    return out.decode(errors="replace").strip()

if mode == "decide":
    receipt, action_hex, session, counter = sys.argv[3:7]
    payload = f"RECEIPT {receipt}\nACTION {action_hex}\nSESSION {session}\nCOUNTER {counter}\n\n".encode()
    print(send(payload))
elif mode == "poll":
    ticket = sys.argv[3]
    print(send(f"POLL ticket={ticket}\n\n".encode()))
elif mode == "poll-until-done":
    ticket = sys.argv[3]
    deadline = time.time() + float(os.environ.get("DEMO_VERIFY_TIMEOUT", "900"))
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        reply = send(f"POLL ticket={ticket}\n\n".encode())
        print(f"  poll attempt {attempt}: {reply}", file=sys.stderr)
        if reply != "PENDING":
            print(reply)
            sys.exit(0)
        time.sleep(5)
    print("TIMEOUT waiting for ticket to resolve", file=sys.stderr)
    sys.exit(1)
else:
    print(f"unknown mode {mode}", file=sys.stderr)
    sys.exit(2)
PYEOF
}

echo
echo "== step 4: submit the real receipt (session=demo counter=1) =="
REPLY1=$(client decide "$WORK/receipt.orig" "$ACTION_HEX" demo 1)
echo "$REPLY1"
if [[ "$REPLY1" != PENDING* ]]; then
    echo "FATAL: expected PENDING, got: $REPLY1" >&2
    exit 1
fi
TICKET1=$(echo "$REPLY1" | sed -n 's/.*ticket=\([^ ]*\).*/\1/p')
echo "polling ticket $TICKET1 until the background verify against the real" \
     "2B model resolves (this genuinely takes ~1-2 minutes wall clock)..."
ALLOW_LINE=$(client poll-until-done "$TICKET1")
echo "RESULT (real 2B model verify): $ALLOW_LINE"
if [[ "$ALLOW_LINE" != ALLOW* ]]; then
    echo "FATAL: expected ALLOW, got: $ALLOW_LINE" >&2
    exit 1
fi

echo
echo "== step 5: replay the IDENTICAL (session, counter, action) =="
REPLAY_LINE=$(client decide "$WORK/receipt.orig" "$ACTION_HEX" demo 1)
echo "RESULT (replay): $REPLAY_LINE"
if [[ "$REPLAY_LINE" != DENY* ]]; then
    echo "FATAL: expected DENY on replay, got: $REPLAY_LINE" >&2
    exit 1
fi

echo
echo "== step 6: tamper one byte inside the receipt's decode-chain hash" \
     "(a field the cheap inline precheck does NOT look at -- only the" \
     "real model-based verify running in the background worker can catch" \
     "this), then submit under a FRESH (session, counter) =="
python3 - "$WORK/receipt.orig" "$WORK/receipt.tampered" <<'PYEOF'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
m = re.search(r'decode-chain=([0-9a-f]{64})', text)
if not m:
    sys.exit("no decode-chain field found to tamper")
h = m.group(1)
flipped_char = format((int(h[0], 16) ^ 0x1), 'x')
tampered_hash = flipped_char + h[1:]
text2 = text[:m.start(1)] + tampered_hash + text[m.end(1):]
assert text2 != text
open(dst, 'w').write(text2)
print(f"flipped decode-chain {h} -> {tampered_hash}")
PYEOF
cat "$WORK/receipt.tampered"

REPLY2=$(client decide "$WORK/receipt.tampered" "$ACTION_HEX" demo 2)
echo "$REPLY2"
if [[ "$REPLY2" != PENDING* ]]; then
    echo "FATAL: expected PENDING (tamper passes the cheap precheck), got: $REPLY2" >&2
    exit 1
fi
TICKET2=$(echo "$REPLY2" | sed -n 's/.*ticket=\([^ ]*\).*/\1/p')
echo "polling ticket $TICKET2 until the background real-model verify" \
     "catches the tamper..."
DENY_LINE=$(client poll-until-done "$TICKET2")
echo "RESULT (tampered receipt, real 2B model verify): $DENY_LINE"
if [[ "$DENY_LINE" != DENY* ]]; then
    echo "FATAL: expected DENY on tampered receipt, got: $DENY_LINE" >&2
    exit 1
fi

echo
echo "== step 7: stop the gateway daemon =="
kill "$DAEMON_PID" 2>/dev/null || true
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

echo
echo "== SUMMARY =="
echo "ALLOW (real receipt, real 2B verify): $ALLOW_LINE"
echo "DENY  (replay, freshness):            $REPLAY_LINE"
echo "DENY  (tampered receipt, real verify): $DENY_LINE"

echo
echo "== step 8: standalone third-party verification =="
echo "The ALLOW'd receipt above can be independently re-verified by ANYONE,"
echo "on ANY machine (including a phone with no network path to this"
echo "gateway), using nothing but the receipt file and the three public"
echo "artifact files, with the standalone agent_trace CLI -- no gateway,"
echo "no socket, no daemon involved:"
echo
echo "  agent_trace verify $MODEL $EMBED $VOCAB $WORK/receipt.orig"
echo
echo "(run now, on this box, against the exact receipt used above, for"
echo " illustration -- a third party would run the equivalent command"
echo " after copying the receipt and the three artifact files to their"
echo " own machine):"
"$AGENT_TRACE_BIN" verify "$MODEL" "$EMBED" "$VOCAB" "$WORK/receipt.orig"
