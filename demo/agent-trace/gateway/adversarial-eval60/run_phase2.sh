#!/usr/bin/env bash
# safe-9b phase 2: strict-grounding gateway daemons (category c, 5 items).
# The gateway's server-side strict-grounding switch is `--strict` (checked
# against `gateway serve --help` / src/main.rs -- there is no
# `--strict-grounding` flag on the *gateway* binary; that name belongs to
# the standalone `agent_trace verify --strict-grounding` flag instead. The
# gateway's `--strict` implements the equivalent server-side policy: DENY
# any receipt whose (always-run, unconditional) `agent_trace verify`
# replay stdout contains a `WARNING step` line, even if the replay itself
# PASSes. See src/main.rs `run_verify_strict`/`verify_pass_strict`. This is
# the "adapt faithfully, note the deviation" case from the task.
#
# Two sub-daemons are needed because the gateway takes a single --table
# path and grounding_03 (chain_fileread_02) needs chain.tsv while the
# other 4 grounding items need demo.tsv (same table as the 60 clean items).
set -euo pipefail

REPO=/home/cm/projects/alice-aegis-cm-safe9b
GW_BIN="$REPO/demo/agent-trace/gateway/target/release/gateway"
AGENT_TRACE_BIN="$REPO/aegis-linux/target/release/examples/agent_trace"
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
MODEL="$ART/aegis_pruned_model.cis.safetensors"
EMBED="$ART/embed.bin"
VOCAB="$ART/vocab.bin"
DEMO_TABLE="$REPO/demo/agent-trace/tables/demo.tsv"
CHAIN_TABLE="$REPO/demo/agent-trace/tables/chain.tsv"
OUT=/home/cm/legs/eval-60-adversarial-box1
MANIFEST="$OUT/injected_manifest.tsv"

WORK="$OUT/work2"
mkdir -p "$WORK"
ALLOW_KEY="$WORK/allow.key"
CAP_KEY="$WORK/cap.key"
echo -n "safe9b-adv-allow-key-not-for-prod" > "$ALLOW_KEY"
echo -n "safe9b-adv-cap-key-not-for-prod" > "$CAP_KEY"
MH=$(sha256sum "$MODEL" | cut -d' ' -f1)
EH=$(sha256sum "$EMBED" | cut -d' ' -f1)
VH=$(sha256sum "$VOCAB" | cut -d' ' -f1)
python3 - "$WORK/allowlist.signed" "$MH" "$EH" "$VH" "$ALLOW_KEY" <<'PYEOF'
import sys, hmac, hashlib
path, mh, eh, vh, keypath = sys.argv[1:6]
key = open(keypath, 'rb').read()
line = f"{mh} {eh} {vh}"
sig = hmac.new(key, (line + "\n").encode(), hashlib.sha256).hexdigest()
open(path, 'w').write(f"{line}\nsig {sig}\n")
PYEOF

client() {
    python3 - "$1" "$2" "${@:3}" <<'PYEOF'
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
        if not chunk: break
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
            print(reply); sys.exit(0)
        time.sleep(3)
    print("TIMEOUT", file=sys.stderr); sys.exit(1)
else:
    print(f"unknown mode {mode}", file=sys.stderr); sys.exit(2)
PYEOF
}

VERDICTS="$OUT/verdicts_phase2.tsv"
printf 'item_id\tverdict\tdetail\tsession\n' > "$VERDICTS"

run_daemon_and_submit() {
    local table="$1" sock="$2" anchor="$3" gwlog="$4"; shift 4
    rm -f "$sock" "$anchor"
    "$GW_BIN" serve "$MODEL" "$EMBED" "$VOCAB" "$AGENT_TRACE_BIN" \
        "$WORK/allowlist.signed" "$ALLOW_KEY" \
        --socket "$sock" --cap-key-file "$CAP_KEY" \
        --table "$table" --strict \
        --anchor-every 2 --anchor-file "$anchor" \
        --verify-workers 1 --cap-ttl 7200 \
        > "$gwlog" 2>&1 &
    local pid=$!
    for _ in $(seq 1 50); do [ -S "$sock" ] && break; sleep 0.2; done
    if [ ! -S "$sock" ]; then
        echo "FATAL: gateway ($table) did not open $sock; see $gwlog" >&2
        cat "$gwlog" >&2 || true
        kill "$pid" 2>/dev/null || true
        exit 1
    fi
    echo "gateway (strict, table=$table) listening: $(grep 'listening on' "$gwlog" || true)"
    for row in "$@"; do
        IFS='|' read -r inj_id receipt_path action_hex session counter <<< "$row"
        echo "[grounding] submitting $inj_id ..." >&2
        REPLY=$(client "$sock" decide "$receipt_path" "$action_hex" "$session" "$counter")
        echo "  decide: $REPLY" >&2
        if [[ "$REPLY" == PENDING* ]]; then
            TICKET=$(echo "$REPLY" | sed -n 's/.*ticket=\([^ ]*\).*/\1/p')
            RESULT=$(client "$sock" poll-until-done "$TICKET")
            echo "  result: $RESULT" >&2
        else
            RESULT="$REPLY"
        fi
        VERDICT=$(echo "$RESULT" | awk '{print $1}')
        DETAIL=$(echo "$RESULT" | cut -d' ' -f2-)
        printf '%s\t%s\t%s\t%s\n' "$inj_id" "$VERDICT" "$DETAIL" "$session" >> "$VERDICTS"
    done
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "== anchor log ($anchor) =="
    cat "$anchor" 2>/dev/null || echo "(no anchor entries)"
}

# demo.tsv items: grounding_01, grounding_02, grounding_04, grounding_05
DEMO_ROWS=()
CHAIN_ROWS=()
while IFS=$'\t' read -r inj_id category base_item_id receipt_path action_hex session counter expected table note; do
    row="$inj_id|$receipt_path|$action_hex|$session|$counter"
    if [ "$table" = "chain" ]; then
        CHAIN_ROWS+=("$row")
    else
        DEMO_ROWS+=("$row")
    fi
done < <(tail -n +2 "$MANIFEST" | awk -F'\t' '$2=="c-strict-grounding"')

echo "== demo.tsv strict daemon (${#DEMO_ROWS[@]} items) =="
run_daemon_and_submit "$DEMO_TABLE" "$OUT/gateway2a.sock" "$OUT/anchor2a.log" "$OUT/gateway2a.log" "${DEMO_ROWS[@]}"

echo "== chain.tsv strict daemon (${#CHAIN_ROWS[@]} items) =="
run_daemon_and_submit "$CHAIN_TABLE" "$OUT/gateway2b.sock" "$OUT/anchor2b.log" "$OUT/gateway2b.log" "${CHAIN_ROWS[@]}"

echo "== verdict counts =="
tail -n +2 "$VERDICTS" | cut -f2 | sort | uniq -c
echo "PHASE2 DONE"
