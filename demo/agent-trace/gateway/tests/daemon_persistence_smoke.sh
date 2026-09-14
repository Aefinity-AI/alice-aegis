#!/usr/bin/env bash
# SAFE-5b sanity check (not an escape test): proves gateway_daemon actually
# fixes the gap named in the design doc -- freshness state surviving
# across separate calls, not just within one process's in-memory Gateway
# for the lifetime of a single exec. Starts a real daemon, sends the SAME
# (receipt, action, session, counter) over TWO SEPARATE unix-socket
# connections, and asserts request 1 ALLOWs while request 2 DENYs on
# freshness grounds -- which is only possible if the daemon's Gateway
# instance is the same object across both connections.
#
# Requires the real 2B artifacts at /home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
# (same ones live20 integration tests use) and python3. Builds
# aegis_trace release + gateway debug binaries if not already built.
#
# Run: bash tests/daemon_persistence_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # demo/agent-trace/gateway

BIN=/home/cm/projects/alice-aegis-cm-safe5b/aegis-linux/target/release/examples/agent_trace
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
RECEIPT=tests/fixtures/live20/calc_01.receipt

if [ ! -f "$ART/aegis_pruned_model.cis.safetensors" ]; then
    echo "SKIP: real 2B artifacts not present at $ART"
    exit 77
fi
if [ ! -x "$BIN" ]; then
    (cd /home/cm/projects/alice-aegis-cm-safe5b/aegis-linux && cargo build --release --offline --example agent_trace)
fi

WORK=$(mktemp -d)
trap 'kill "${DAEMON_PID:-0}" 2>/dev/null || true; rm -rf "$WORK"' EXIT

MH=$(sha256sum "$ART/aegis_pruned_model.cis.safetensors" | cut -d' ' -f1)
EH=$(sha256sum "$ART/embed.bin" | cut -d' ' -f1)
VH=$(sha256sum "$ART/vocab.bin" | cut -d' ' -f1)
ALLOW_KEY="$WORK/allow.key"; echo -n "daemon-smoke-allow-key-not-for-prod" > "$ALLOW_KEY"
CAP_KEY="$WORK/cap.key"; echo -n "daemon-smoke-cap-key-not-for-prod" > "$CAP_KEY"

python3 - "$WORK/allowlist.signed" "$MH" "$EH" "$VH" "$ALLOW_KEY" <<'PYEOF'
import sys, hmac, hashlib
path, mh, eh, vh, keypath = sys.argv[1:6]
key = open(keypath, 'rb').read()
line = f"{mh} {eh} {vh}"
sig = hmac.new(key, (line + "\n").encode(), hashlib.sha256).hexdigest()
open(path, 'w').write(f"{line}\nsig {sig}\n")
PYEOF

SOCK="$WORK/gateway.sock"
cargo build --offline --quiet
target/debug/gateway_daemon "$ART/aegis_pruned_model.cis.safetensors" "$ART/embed.bin" "$ART/vocab.bin" "$BIN" \
  "$WORK/allowlist.signed" "$ALLOW_KEY" "$CAP_KEY" "$SOCK" \
  --anchor-every 100 --cap-ttl-secs 60 &
DAEMON_PID=$!
sleep 1

ACTION_HEX=$(tr ' ' '\n' < "$RECEIPT" | grep '^in=' | tail -1 | cut -d= -f2)
REQ="receipt=$PWD/$RECEIPT action=$ACTION_HEX session=daemon-smoke counter=1"

send() {
    python3 -c "
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect('$SOCK')
s.sendall(('$REQ'+chr(10)).encode())
print(s.recv(65536).decode().strip())
"
}

echo "== request 1 (fresh connection) =="
RESP1=$(send)
echo "$RESP1"
echo
echo "== request 2: SAME (session, counter, action) over a SECOND, separate connection =="
RESP2=$(send)
echo "$RESP2"

if echo "$RESP1" | grep -q "^ALLOW " && echo "$RESP2" | grep -q "freshness: (session, counter, action-hash) already seen"; then
    echo
    echo "PASS: persistent daemon remembered freshness state across two separate unix-socket connections"
    exit 0
else
    echo "FAIL: expected request 1 ALLOW and request 2 DENY(freshness)"
    exit 1
fi
