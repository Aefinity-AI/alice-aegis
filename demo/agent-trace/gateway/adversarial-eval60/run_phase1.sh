#!/usr/bin/env bash
# safe-9b phase 1: non-strict gateway daemon (table=demo.tsv, same as
# safe-9). Submits, in order:
#   1. the 60 clean EVAL-60 T1 receipts (session=item_id, counter=1) --
#      expect 60 ALLOW, exactly as safe-9.
#   2. 5 replay items (category a) -- resubmit an already-ALLOWed clean
#      item's IDENTICAL (session, counter, action) -- expect DENY freshness.
#   3. 5 tamper items (category b) -- one flipped output byte -- expect
#      DENY via the real agent_trace verify (VERIFY FAIL).
#   4. 3 allowlist-decoy items (category d) -- flipped model hash --
#      expect DENY allowlist (cheap, before verify).
#   5. 2 binding-mismatch items (category e) -- forwarded action bytes are
#      a DIFFERENT item's action -- expect DENY binding (cheap, before
#      verify).
set -euo pipefail

REPO=/home/cm/projects/alice-aegis-cm-safe9b
GW_BIN="$REPO/demo/agent-trace/gateway/target/release/gateway"
AGENT_TRACE_BIN="$REPO/aegis-linux/target/release/examples/agent_trace"
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
MODEL="$ART/aegis_pruned_model.cis.safetensors"
EMBED="$ART/embed.bin"
VOCAB="$ART/vocab.bin"
TABLE="$REPO/demo/agent-trace/tables/demo.tsv"
RECEIPTS_DIR=/home/cm/legs/eval-60-replay-box1/receipts
OUT=/home/cm/legs/eval-60-adversarial-box1
MANIFEST="$OUT/injected_manifest.tsv"
mkdir -p "$OUT"

for f in "$GW_BIN" "$AGENT_TRACE_BIN" "$MODEL" "$EMBED" "$VOCAB" "$TABLE" "$MANIFEST"; do
    [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

WORK="$OUT/work1"
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
echo -n "safe9b-adv-allow-key-not-for-prod" > "$ALLOW_KEY"
echo -n "safe9b-adv-cap-key-not-for-prod" > "$CAP_KEY"
python3 - "$WORK/allowlist.signed" "$MH" "$EH" "$VH" "$ALLOW_KEY" <<'PYEOF'
import sys, hmac, hashlib
path, mh, eh, vh, keypath = sys.argv[1:6]
key = open(keypath, 'rb').read()
line = f"{mh} {eh} {vh}"
sig = hmac.new(key, (line + "\n").encode(), hashlib.sha256).hexdigest()
open(path, 'w').write(f"{line}\nsig {sig}\n")
PYEOF

echo "== start gateway daemon (real 2B model, worker pool, non-strict) =="
SOCK="$OUT/gateway1.sock"
ANCHOR="$OUT/anchor1.log"
rm -f "$ANCHOR" "$SOCK"
"$GW_BIN" serve "$MODEL" "$EMBED" "$VOCAB" "$AGENT_TRACE_BIN" \
    "$WORK/allowlist.signed" "$ALLOW_KEY" \
    --socket "$SOCK" --cap-key-file "$CAP_KEY" \
    --table "$TABLE" \
    --anchor-every 5 --anchor-file "$ANCHOR" \
    --verify-workers 2 --cap-ttl 7200 \
    > "$OUT/gateway1.log" 2>&1 &
DAEMON_PID=$!

for _ in $(seq 1 50); do
    [ -S "$SOCK" ] && break
    sleep 0.2
done
if [ ! -S "$SOCK" ]; then
    echo "FATAL: gateway did not open $SOCK; see $OUT/gateway1.log" >&2
    cat "$OUT/gateway1.log" >&2 || true
    exit 1
fi
echo "gateway listening: $(grep 'listening on' "$OUT/gateway1.log" || true)"

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
        time.sleep(3)
    print("TIMEOUT waiting for ticket to resolve", file=sys.stderr)
    sys.exit(1)
else:
    print(f"unknown mode {mode}", file=sys.stderr)
    sys.exit(2)
PYEOF
}

submit_one() {
    # $1=id $2=receipt $3=action_hex $4=session $5=counter
    local id="$1" r="$2" action_hex="$3" session="$4" counter="$5"
    REPLY=$(client decide "$r" "$action_hex" "$session" "$counter")
    echo "  decide: $REPLY" >&2
    if [[ "$REPLY" == PENDING* ]]; then
        TICKET=$(echo "$REPLY" | sed -n 's/.*ticket=\([^ ]*\).*/\1/p')
        RESULT=$(client poll-until-done "$TICKET")
        echo "  result: $RESULT" >&2
    else
        RESULT="$REPLY"
    fi
    VERDICT=$(echo "$RESULT" | awk '{print $1}')
    DETAIL=$(echo "$RESULT" | cut -d' ' -f2-)
    printf '%s\t%s\t%s\t%s\n' "$id" "$VERDICT" "$DETAIL" "$session" >> "$VERDICTS"
}

VERDICTS="$OUT/verdicts_phase1.tsv"
printf 'item_id\tverdict\tdetail\tsession\n' > "$VERDICTS"

echo "== stage 1: 60 clean EVAL-60 T1 receipts =="
i=0
for r in "$RECEIPTS_DIR"/*.txt; do
    i=$((i+1))
    id=$(basename "$r" .txt)
    ACTION_HEX=$(tr ' ' '\n' < "$r" | grep '^in=' | tail -1 | cut -d= -f2)
    echo "[clean $i/60] submitting $id ..." >&2
    submit_one "$id" "$r" "$ACTION_HEX" "$id" 1
done

echo "== stage 2: 5 replay (category a) =="
tail -n +2 "$MANIFEST" | awk -F'\t' '$2=="a-freshness-replay"' | while IFS=$'\t' read -r inj_id category base_item_id receipt_path action_hex session counter expected table note; do
    echo "[replay] submitting $inj_id (replays $base_item_id's session/action) ..." >&2
    submit_one "$inj_id" "$receipt_path" "$action_hex" "$session" "$counter"
done

echo "== stage 3: 5 tamper (category b) =="
tail -n +2 "$MANIFEST" | awk -F'\t' '$2=="b-tamper-output-byte"' | while IFS=$'\t' read -r inj_id category base_item_id receipt_path action_hex session counter expected table note; do
    echo "[tamper] submitting $inj_id (base=$base_item_id) ..." >&2
    submit_one "$inj_id" "$receipt_path" "$action_hex" "$session" "$counter"
done

echo "== stage 4: 3 allowlist-decoy (category d) =="
tail -n +2 "$MANIFEST" | awk -F'\t' '$2=="d-allowlist-decoy-triple"' | while IFS=$'\t' read -r inj_id category base_item_id receipt_path action_hex session counter expected table note; do
    echo "[allowlist] submitting $inj_id (base=$base_item_id) ..." >&2
    submit_one "$inj_id" "$receipt_path" "$action_hex" "$session" "$counter"
done

echo "== stage 5: 2 binding-mismatch (category e) =="
tail -n +2 "$MANIFEST" | awk -F'\t' '$2=="e-verify-execute-binding-mismatch"' | while IFS=$'\t' read -r inj_id category base_item_id receipt_path action_hex session counter expected table note; do
    echo "[binding] submitting $inj_id (base=$base_item_id, wrong action) ..." >&2
    submit_one "$inj_id" "$receipt_path" "$action_hex" "$session" "$counter"
done

echo "== stop gateway daemon =="
kill "$DAEMON_PID" 2>/dev/null || true
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

echo "== anchor log ($ANCHOR) =="
cat "$ANCHOR" 2>/dev/null || echo "(no anchor entries)"

echo "== verdict counts =="
tail -n +2 "$VERDICTS" | cut -f2 | sort | uniq -c
echo "PHASE1 DONE"
