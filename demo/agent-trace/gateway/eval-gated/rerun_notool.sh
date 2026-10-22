#!/usr/bin/env bash
# Rerun the 10 items whose first pass mis-flagged an empty ACTION (tool=no-tool,
# in= present but empty) as ERROR. Submits them through a fresh gateway
# instance with ACTION set to the empty string, exactly matching their
# receipt's (empty) last_step_input, so the real gate makes an actual
# ALLOW/DENY decision for these too instead of skipping them.
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
WORK="$OUT/work2"
mkdir -p "$WORK"

ITEMS="calc_hard_02 calc_hard_08 calc_hard_13 calc_hard_14 calc_overflow_04 calc_overflow_05 calc_overflow_07 distractor_01 mixed_01 mixed_03"

DAEMON_PID=""
cleanup() {
    if [ -n "$DAEMON_PID" ]; then
        kill "$DAEMON_PID" >/dev/null 2>&1 || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

MH=$(sha256sum "$MODEL" | cut -d' ' -f1)
EH=$(sha256sum "$EMBED" | cut -d' ' -f1)
VH=$(sha256sum "$VOCAB" | cut -d' ' -f1)
ALLOW_KEY="$WORK/allow.key"
CAP_KEY="$WORK/cap.key"
echo -n "safe9-eval60-allow-key-not-for-prod-2" > "$ALLOW_KEY"
echo -n "safe9-eval60-cap-key-not-for-prod-2" > "$CAP_KEY"
python3 - "$WORK/allowlist.signed" "$MH" "$EH" "$VH" "$ALLOW_KEY" <<'PYEOF'
import sys, hmac, hashlib
path, mh, eh, vh, keypath = sys.argv[1:6]
key = open(keypath, 'rb').read()
line = f"{mh} {eh} {vh}"
sig = hmac.new(key, (line + "\n").encode(), hashlib.sha256).hexdigest()
open(path, 'w').write(f"{line}\nsig {sig}\n")
PYEOF

SOCK="$OUT/gateway2.sock"
ANCHOR="$OUT/anchor2.log"
rm -f "$ANCHOR" "$SOCK"
"$GW_BIN" serve "$MODEL" "$EMBED" "$VOCAB" "$AGENT_TRACE_BIN" \
    "$WORK/allowlist.signed" "$ALLOW_KEY" \
    --socket "$SOCK" --cap-key-file "$CAP_KEY" \
    --table "$TABLE" \
    --anchor-every 5 --anchor-file "$ANCHOR" \
    --verify-workers 2 --cap-ttl 7200 \
    > "$OUT/gateway2.log" 2>&1 &
DAEMON_PID=$!

for _ in $(seq 1 50); do
    [ -S "$SOCK" ] && break
    sleep 0.2
done
[ -S "$SOCK" ] || { echo "FATAL: gateway2 did not open socket" >&2; cat "$OUT/gateway2.log" >&2; exit 1; }
echo "gateway2 listening: $(grep 'listening on' "$OUT/gateway2.log")"

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

VERDICTS2="$OUT/verdicts_notool.tsv"
printf 'item_id\tverdict\tdetail\n' > "$VERDICTS2"

i=0
for id in $ITEMS; do
    i=$((i+1))
    r="$RECEIPTS_DIR/$id.txt"
    echo "[$i/10] submitting $id (empty action, no tool call) ..." >&2
    REPLY=$(client decide "$r" "" "$id" 1)
    echo "  decide: $REPLY" >&2
    if [[ "$REPLY" == PENDING* ]]; then
        TICKET=$(echo "$REPLY" | sed -n 's/.*ticket=\([^ ]*\).*/\1/p')
        RESULT=$(client poll-until-done "$TICKET")
        echo "  result: $RESULT" >&2
        VERDICT=$(echo "$RESULT" | awk '{print $1}')
        DETAIL=$(echo "$RESULT" | cut -d' ' -f2- )
        printf '%s\t%s\t%s\n' "$id" "$VERDICT" "$DETAIL" >> "$VERDICTS2"
    else
        VERDICT=$(echo "$REPLY" | awk '{print $1}')
        DETAIL=$(echo "$REPLY" | cut -d' ' -f2- )
        printf '%s\t%s\t%s\n' "$id" "$VERDICT" "$DETAIL" >> "$VERDICTS2"
    fi
done

kill "$DAEMON_PID" 2>/dev/null || true
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

echo "== anchor2 log =="
cat "$ANCHOR" 2>/dev/null || echo "(no anchor entries -- fewer than anchor-every decisions)"
echo "== verdict counts (notool subset) =="
tail -n +2 "$VERDICTS2" | cut -f2 | sort | uniq -c
