#!/usr/bin/env bash
# safe-9: submit all 60 EVAL-60 T1 receipts (already-generated, box1 replay
# copy at /home/cm/legs/eval-60-replay-box1/receipts/) through the real
# async receipt-verification gateway (PR #99, demo/agent-trace/gateway),
# using the calling pattern from safe-8's demo.sh (PENDING ticket=<id> ->
# poll -> ALLOW/DENY). Records one ALLOW/DENY verdict per item; scoring
# happens in score_allowed.py afterward, over the ALLOWed subset only.
set -euo pipefail

REPO=/home/cm/projects/alice-aegis-cm-safe9
GW_BIN="$REPO/demo/agent-trace/gateway/target/release/gateway"
AGENT_TRACE_BIN="$REPO/aegis-linux/target/release/examples/agent_trace"
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
MODEL="$ART/aegis_pruned_model.cis.safetensors"
EMBED="$ART/embed.bin"
VOCAB="$ART/vocab.bin"
TABLE="$REPO/demo/agent-trace/tables/demo.tsv"
RECEIPTS_DIR=/home/cm/legs/eval-60-replay-box1/receipts
OUT=/home/cm/legs/eval-60-gateway-box1
mkdir -p "$OUT"

for f in "$GW_BIN" "$AGENT_TRACE_BIN" "$MODEL" "$EMBED" "$VOCAB" "$TABLE"; do
    [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

WORK="$OUT/work"
mkdir -p "$WORK"
DAEMON_PID=""
cleanup() {
    if [ -n "$DAEMON_PID" ]; then
        kill "$DAEMON_PID" >/dev/null 2>&1 || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "== sign allowlist for this box's real artifact hashes =="
MH=$(sha256sum "$MODEL" | cut -d' ' -f1)
EH=$(sha256sum "$EMBED" | cut -d' ' -f1)
VH=$(sha256sum "$VOCAB" | cut -d' ' -f1)
ALLOW_KEY="$WORK/allow.key"
CAP_KEY="$WORK/cap.key"
echo -n "safe9-eval60-allow-key-not-for-prod" > "$ALLOW_KEY"
echo -n "safe9-eval60-cap-key-not-for-prod" > "$CAP_KEY"
python3 - "$WORK/allowlist.signed" "$MH" "$EH" "$VH" "$ALLOW_KEY" <<'PYEOF'
import sys, hmac, hashlib
path, mh, eh, vh, keypath = sys.argv[1:6]
key = open(keypath, 'rb').read()
line = f"{mh} {eh} {vh}"
sig = hmac.new(key, (line + "\n").encode(), hashlib.sha256).hexdigest()
open(path, 'w').write(f"{line}\nsig {sig}\n")
PYEOF

echo "== start gateway daemon (real 2B model, worker pool) =="
SOCK="$OUT/gateway.sock"
ANCHOR="$OUT/anchor.log"
rm -f "$ANCHOR"
"$GW_BIN" serve "$MODEL" "$EMBED" "$VOCAB" "$AGENT_TRACE_BIN" \
    "$WORK/allowlist.signed" "$ALLOW_KEY" \
    --socket "$SOCK" --cap-key-file "$CAP_KEY" \
    --table "$TABLE" \
    --anchor-every 5 --anchor-file "$ANCHOR" \
    --verify-workers 2 --cap-ttl 7200 \
    > "$OUT/gateway.log" 2>&1 &
DAEMON_PID=$!

for _ in $(seq 1 50); do
    [ -S "$SOCK" ] && break
    sleep 0.2
done
if [ ! -S "$SOCK" ]; then
    echo "FATAL: gateway did not open $SOCK; see $OUT/gateway.log" >&2
    cat "$OUT/gateway.log" >&2 || true
    exit 1
fi
echo "gateway listening: $(grep 'listening on' "$OUT/gateway.log" || true)"

client() {
    python3 - "$SOCK" "$@" <<'PYEOF'
import socket, sys, time

sock_path = sys.argv[1]
mode = sys.argv[2]

def send(payload: bytes) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(15)
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
elif mode == "poll-until-done":
    ticket = sys.argv[3]
    deadline = time.time() + 900
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

VERDICTS="$OUT/verdicts.tsv"
printf 'item_id\tverdict\tdetail\n' > "$VERDICTS"

i=0
for r in "$RECEIPTS_DIR"/*.txt; do
    i=$((i+1))
    id=$(basename "$r" .txt)
    ACTION_HEX=$(tr ' ' '\n' < "$r" | grep '^in=' | tail -1 | cut -d= -f2)
    if [ -z "$ACTION_HEX" ]; then
        printf '%s\tERROR\tno in= field found in receipt\n' "$id" >> "$VERDICTS"
        continue
    fi
    echo "[$i/60] submitting $id ..." >&2
    REPLY=$(client decide "$r" "$ACTION_HEX" "$id" 1)
    echo "  decide: $REPLY" >&2
    if [[ "$REPLY" == PENDING* ]]; then
        TICKET=$(echo "$REPLY" | sed -n 's/.*ticket=\([^ ]*\).*/\1/p')
        RESULT=$(client poll-until-done "$TICKET")
        echo "  result: $RESULT" >&2
        VERDICT=$(echo "$RESULT" | awk '{print $1}')
        DETAIL=$(echo "$RESULT" | cut -d' ' -f2- )
        printf '%s\t%s\t%s\n' "$id" "$VERDICT" "$DETAIL" >> "$VERDICTS"
    else
        VERDICT=$(echo "$REPLY" | awk '{print $1}')
        DETAIL=$(echo "$REPLY" | cut -d' ' -f2- )
        printf '%s\t%s\t%s\n' "$id" "$VERDICT" "$DETAIL" >> "$VERDICTS"
    fi
done

echo "== stop gateway daemon =="
kill "$DAEMON_PID" 2>/dev/null || true
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

echo "== anchor log ($ANCHOR) =="
cat "$ANCHOR" 2>/dev/null || echo "(no anchor entries written -- fewer than anchor-every decisions since last anchor)"

echo "== verdict counts =="
tail -n +2 "$VERDICTS" | cut -f2 | sort | uniq -c
