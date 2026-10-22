#!/usr/bin/env bash
# safe-9b fixup: the original manifest picked calc_hard_02 as binding_02's
# "wrong action" source, but calc_hard_02 is one of the 10 no-tool-call
# EVAL-60 items (empty in= field) -- so the submitted ACTION ended up
# empty rather than a genuinely different item's real action bytes, and a
# bash `read -r IFS=$'\t'` field-splitting quirk (tab is POSIX IFS
# whitespace, so consecutive tabs collapse even under a custom
# single-character tab IFS) shifted the session/counter columns in the
# logged verdict row. This script redoes binding_02 correctly: forwarded
# action = calc_hard_03's real action (non-empty, genuinely different from
# calc_easy_15's), with a fresh short-lived gateway using the SAME signed
# allowlist as phase 1.
set -euo pipefail
REPO=/home/cm/projects/alice-aegis-cm-safe9b
GW_BIN="$REPO/demo/agent-trace/gateway/target/release/gateway"
AGENT_TRACE_BIN="$REPO/aegis-linux/target/release/examples/agent_trace"
ART=/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts
MODEL="$ART/aegis_pruned_model.cis.safetensors"
EMBED="$ART/embed.bin"
VOCAB="$ART/vocab.bin"
TABLE="$REPO/demo/agent-trace/tables/demo.tsv"
OUT=/home/cm/legs/eval-60-adversarial-box1
WORK="$OUT/work1"  # reuse phase 1's allowlist/keys

SOCK="$OUT/gateway1b.sock"
ANCHOR="$OUT/anchor1b.log"
rm -f "$SOCK" "$ANCHOR"
"$GW_BIN" serve "$MODEL" "$EMBED" "$VOCAB" "$AGENT_TRACE_BIN" \
    "$WORK/allowlist.signed" "$WORK/allow.key" \
    --socket "$SOCK" --cap-key-file "$WORK/cap.key" \
    --table "$TABLE" \
    --anchor-every 2 --anchor-file "$ANCHOR" \
    --verify-workers 1 --cap-ttl 7200 \
    > "$OUT/gateway1b.log" 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.2; done
[ -S "$SOCK" ] || { echo FATAL >&2; cat "$OUT/gateway1b.log" >&2; exit 1; }

RECEIPT=/home/cm/legs/eval-60-replay-box1/receipts/calc_easy_15.txt
WRONG_ACTION_HEX=$(tr ' ' '\n' < /home/cm/legs/eval-60-replay-box1/receipts/calc_hard_03.txt | grep '^in=' | tail -1 | cut -d= -f2)

REPLY=$(python3 - "$SOCK" "$RECEIPT" "$WRONG_ACTION_HEX" "binding_02b" 1 <<'PYEOF'
import socket, sys
sock_path, receipt, action_hex, session, counter = sys.argv[1:6]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(15)
s.connect(sock_path)
payload = f"RECEIPT {receipt}\nACTION {action_hex}\nSESSION {session}\nCOUNTER {counter}\n\n".encode()
s.sendall(payload)
s.shutdown(socket.SHUT_WR)
out = b""
while True:
    chunk = s.recv(65536)
    if not chunk: break
    out += chunk
print(out.decode(errors="replace").strip())
PYEOF
)
echo "binding_02 (redo): $REPLY"
echo "$REPLY" > "$OUT/binding_02_redo_reply.txt"
