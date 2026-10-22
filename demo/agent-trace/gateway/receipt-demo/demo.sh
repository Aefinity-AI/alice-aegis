#!/usr/bin/env bash
# tool-2 (state/QUEUE.md, claudius-maximus repo): one-command end-to-end
# capability/receipt gateway demo, for an xb-3 auditor walkthrough.
#
# Loop this demonstrates:
#   1. An "agent" proposes a tool call (a toy file-write).
#   2. The REAL gateway capability code (gateway::capability::issue/verify,
#      demo/agent-trace/gateway/src/capability.rs -- from tool-1's #112
#      gateway work, not reimplemented here) mints/checks a capability
#      token before the call is allowed to execute. `tool_shim_file`
#      (also existing gateway code, unmodified) is fail-closed: any bad
#      capability -> SHIM REFUSE, nothing written.
#   3. Each attempt (allowed or refused) is appended as one JSON line to
#      receipts.log: what was called, with what args, under which
#      capability, and when.
#   4. `offline_verify_receipts` -- a separate process, run in this same
#      script but with NO shared in-memory state with steps 1-3, reading
#      only the key file and receipts.log off disk -- independently
#      re-verifies every entry using the real `capability::verify`
#      function. This simulates "another box" auditing the log after the
#      fact.
#
# Run: ./demo.sh   (from this directory, or anywhere -- it cd's to the
# gateway crate itself). Needs only `cargo`+`rustc` (offline, no network).
# Expected total runtime on modest hardware (tested target: a Celeron
# N4020C): well under 5 minutes, most of it a debug `cargo build`.
#
# Expected output shape: one PASS case (Case A) and two FAIL cases
# (Case B: gateway refuses a tampered capability before the write ever
# happens; Case C: the offline verifier -- not the gateway -- catches a
# receipt log edited after the fact). Look for the "==>" case headers and
# the final "==> RESULT" summary line.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_DIR="$(cd "$HERE/.." && pwd)"

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

KEY_FILE="$WORK/cap.key"
head -c 32 /dev/urandom >"$KEY_FILE"

LOG="$WORK/receipts.log"
OUT_DIR="$WORK/out"
mkdir -p "$OUT_DIR"
touch "$LOG"

echo "==> building gateway binaries (debug, offline, no network deps)..."
(cd "$GATEWAY_DIR" && cargo build --quiet --bin cap_issue_for_test --bin tool_shim_file --bin offline_verify_receipts)
BIN="$GATEWAY_DIR/target/debug"

sha256_hex() { printf '%s' "$1" | sha256sum | cut -d' ' -f1; }

append_log() {
  # $1 idx  $2 tool  $3 path  $4 data  $5 action_hash  $6 cap  $7 exp
  # $8 decision  $9 reason
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"ts":"%s","idx":%s,"tool":"%s","path":"%s","data":"%s","action_hash":"%s","cap":"%s","exp":%s,"decision":"%s","reason":"%s"}\n' \
    "$ts" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" >>"$LOG"
}

NOW="$(date +%s)"
EXP="$((NOW + 300))"

########################################################################
# Case A: PASS -- valid capability, agent's file-write is allowed.
########################################################################
echo
echo "==> Case A (PASS): agent proposes a file-write; gateway issues a valid capability"
IDX_A=1
DATA_A="hello from the tool-2 receipt demo, decision index ${IDX_A}"
PATH_A="$OUT_DIR/case-a.txt"
CAP_A="$("$BIN/cap_issue_for_test" "$KEY_FILE" "$IDX_A" "$EXP" file "$PATH_A" "$DATA_A")"
HASH_A="$(sha256_hex "$DATA_A")"

set +e
OUT_A="$("$BIN/tool_shim_file" --idx "$IDX_A" --cap "$CAP_A" --exp "$EXP" --action-hash "$HASH_A" \
  --key-file "$KEY_FILE" --consumed-file "$WORK/consumed-file" --path "$PATH_A" --data "$DATA_A" 2>&1)"
RC_A=$?
set -e
if [ "$RC_A" -eq 0 ] && [ -f "$PATH_A" ]; then
  echo "gateway/shim: ALLOW -- wrote $PATH_A"
  append_log "$IDX_A" file "$PATH_A" "$DATA_A" "$HASH_A" "$CAP_A" "$EXP" ALLOW ""
else
  echo "gateway/shim: unexpected refusal in Case A: $OUT_A"
  append_log "$IDX_A" file "$PATH_A" "$DATA_A" "$HASH_A" "$CAP_A" "$EXP" DENY "$OUT_A"
fi

########################################################################
# Case B: FAIL (fail-closed at the gateway) -- tampered capability token.
########################################################################
echo
echo "==> Case B (FAIL): agent proposes another file-write; capability token is tampered before use"
IDX_B=2
DATA_B="an attempted write using a forged/tampered capability"
PATH_B="$OUT_DIR/case-b.txt"
CAP_B_REAL="$("$BIN/cap_issue_for_test" "$KEY_FILE" "$IDX_B" "$EXP" file "$PATH_B" "$DATA_B")"
# Flip the last hex character -- simulates a bit-flip/forgery attempt.
LAST="${CAP_B_REAL: -1}"
case "$LAST" in
  0) NEWLAST=1 ;; *) NEWLAST=0 ;;
esac
CAP_B_TAMPERED="${CAP_B_REAL%?}${NEWLAST}"
HASH_B="$(sha256_hex "$DATA_B")"

set +e
OUT_B="$("$BIN/tool_shim_file" --idx "$IDX_B" --cap "$CAP_B_TAMPERED" --exp "$EXP" --action-hash "$HASH_B" \
  --key-file "$KEY_FILE" --consumed-file "$WORK/consumed-file" --path "$PATH_B" --data "$DATA_B" 2>&1)"
RC_B=$?
set -e
if [ "$RC_B" -ne 0 ] && [ ! -f "$PATH_B" ]; then
  echo "gateway/shim: REFUSE as expected ($OUT_B) -- no file written"
  append_log "$IDX_B" file "$PATH_B" "$DATA_B" "$HASH_B" "$CAP_B_TAMPERED" "$EXP" DENY "$OUT_B"
else
  echo "!! Case B did not fail closed as expected: rc=$RC_B out=$OUT_B"
  append_log "$IDX_B" file "$PATH_B" "$DATA_B" "$HASH_B" "$CAP_B_TAMPERED" "$EXP" ALLOW "$OUT_B"
fi

########################################################################
# Case C: FAIL (caught offline, not at the gateway) -- the log itself is
# edited after the fact, e.g. by whoever controls the log file on disk;
# the capability token is untouched, but the offline verifier recomputes
# the action hash from the (now-tampered) logged data and it no longer
# matches.
########################################################################
echo
echo "==> Case C (FAIL): receipts.log for Case A is edited after the fact (data changed, token left alone)"
TAMPERED_LOG="$WORK/receipts.tampered.log"
sed "s/decision index ${IDX_A}\", *\"action_hash/decision index ${IDX_A} -- EDITED BY ATTACKER\", \"action_hash/" \
  "$LOG" >"$TAMPERED_LOG"
echo "wrote tampered copy to $TAMPERED_LOG (capability token/idx/exp untouched)"

########################################################################
# Step 4: OFFLINE, independent re-verification -- a separate process
# ("another box"), no shared state with the steps above, reading only
# the key file + a receipts log off disk.
########################################################################
echo
echo "==> offline re-verification of the REAL receipts.log (separate process, disk-only state)"
set +e
"$BIN/offline_verify_receipts" "$KEY_FILE" "$LOG"
RC_REAL=$?
set -e

echo
echo "==> offline re-verification of the TAMPERED receipts.log copy (simulating a compromised/edited log on another box)"
set +e
"$BIN/offline_verify_receipts" "$KEY_FILE" "$TAMPERED_LOG"
RC_TAMPERED=$?
set -e

echo
echo "==> RESULT"
PASS=true
[ "$RC_A" -eq 0 ] || { echo "FAIL: Case A did not ALLOW"; PASS=false; }
[ "$RC_B" -ne 0 ] || { echo "FAIL: Case B did not REFUSE (fail-closed broken!)"; PASS=false; }
[ "$RC_REAL" -eq 0 ] || { echo "FAIL: offline verify of the real log reported a failure"; PASS=false; }
[ "$RC_TAMPERED" -ne 0 ] || { echo "FAIL: offline verify did NOT catch the tampered log (fail-closed broken!)"; PASS=false; }

if $PASS; then
  echo "PASS: Case A allowed+logged+offline-verified; Case B refused fail-closed at the gateway; Case C's log tampering was caught fail-closed by the offline verifier."
  echo "receipts.log (real, from this run):"
  cat "$LOG"
  exit 0
else
  echo "one or more expected properties did not hold -- see FAIL lines above"
  exit 1
fi
