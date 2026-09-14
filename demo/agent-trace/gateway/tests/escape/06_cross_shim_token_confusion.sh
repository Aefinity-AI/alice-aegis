#!/usr/bin/env bash
# SAFE-11 escape test 6: a capability token minted for ONE tool must be
# REFUSED at every OTHER tool's shim, even when the action bytes hash-
# match (e.g. a string that is simultaneously valid file content and a
# valid shell command). This is the exact cross-shim token confusion
# safe-10's adaptive-adversary pass found as a MAJOR real security break
# (see tests/adversarial/safe10/cross_shim_token_confusion_poc.sh) and
# SAFE-11 fixes by folding a tool_tag into the capability token's HMAC
# input (see src/capability.rs) and having each shim always verify
# against its OWN fixed tag (src/bin/tool_shim_*.rs).
#
# NOTE ON THIS TEST'S PRE-FIX/POST-FIX STATUS: this test is written
# AGAINST the post-SAFE-11 `cap_issue_for_test`/`capability::issue`/
# `capability::verify`/`check_and_consume` signatures (which now all take
# an explicit tool_tag). Checked out against the pre-SAFE-11 commit
# (9ccdbeb, tag-less signatures), this script does not even compile
# (cap_issue_for_test takes 3 CLI args pre-fix and this script would
# still just call it the same way -- the actual proof of the pre-fix
# vulnerability, i.e. that a same-tag-less token issued for "file" is
# ACCEPTED at tool_shim_shell, is tests/adversarial/safe10/
# cross_shim_token_confusion_poc.sh, which is unchanged by this commit
# and still runs against both the pre- and post-fix tree; see that
# script's own "RESULT: CONFIRMED" (pre-fix) vs the fail-closed refusal
# (post-fix) it now prints.
#
# Uses cap_issue_for_test (mints tokens via the exact same
# gateway::capability::issue the real daemon calls on ALLOW) so tokens
# here are indistinguishable from real daemon output.
#
# Run: bash tests/escape/06_cross_shim_token_confusion.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # demo/agent-trace/gateway

cargo build --offline --quiet 2>/dev/null || cargo build --quiet
ISSUE=target/debug/cap_issue_for_test
SHIM_SHELL=target/debug/tool_shim_shell
SHIM_FILE=target/debug/tool_shim_file
SHIM_HTTP=target/debug/tool_shim_http
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
KEY="$WORK/cap.key"
head -c 32 /dev/urandom > "$KEY"

EXPIRY=$(( $(date +%s) + 300 ))
IDX=101

echo "== escape test 6: a token minted for one tool is refused at every other tool's shim =="

pass=0
fail=0

check_refused() {
  # Accepts either "bad mac" (the tool-tag mismatch itself, when the
  # presented action-hash still matches what the token claims) or
  # "action hash mismatch" (when the two shims' action-byte encodings
  # differ enough that the claimed hash does not even match, e.g. http's
  # "METHOD URL DATA\n" framing vs. file's raw content bytes) -- both are
  # fail-closed refusals and either is an acceptable proof that the token
  # did not authorize execution at this shim.
  local label="$1" out="$2" rc="$3"
  if [ "$rc" -ne 0 ] && echo "$out" | grep -qE "^SHIM REFUSE: (bad mac|action hash mismatch)$"; then
    echo "  PASS: $label refused (exit=$rc): $out"
    pass=$((pass+1))
  else
    echo "  FAIL: $label expected a SHIM REFUSE with nonzero exit, got exit=$rc output: $out"
    fail=$((fail+1))
  fi
}

# ---------------------------------------------------------------------
# Case 1: a token minted for TOOL=file (content string also happens to be
# a valid shell command) must be refused at tool_shim_shell AND
# tool_shim_http.
# ---------------------------------------------------------------------
echo
echo "-- case 1: file-write token presented to shell and http shims --"
CONTENT="echo cross-shim-marker"
AHASH_FILE=$(printf '%s' "$CONTENT" | sha256sum | awk '{print $1}')
TOKEN_FILE=$("$ISSUE" "$KEY" "$IDX" "$EXPIRY" file "/tmp/harmless-output.txt" "$CONTENT")
echo "minted TOOL=file token: $TOKEN_FILE"

CONSUMED_SHELL="$WORK/consumed.shell.txt"; touch "$CONSUMED_SHELL"
set +e
OUT=$("$SHIM_SHELL" --idx "$IDX" --cap "$TOKEN_FILE" --exp "$EXPIRY" --action-hash "$AHASH_FILE" \
  --key-file "$KEY" --consumed-file "$CONSUMED_SHELL" -- $CONTENT 2>&1)
RC=$?
set -e
echo "$OUT"
check_refused "file-token @ tool_shim_shell" "$OUT" "$RC"

CONSUMED_HTTP="$WORK/consumed.http.txt"; touch "$CONSUMED_HTTP"
set +e
OUT=$("$SHIM_HTTP" --idx "$IDX" --cap "$TOKEN_FILE" --exp "$EXPIRY" --action-hash "$AHASH_FILE" \
  --key-file "$KEY" --consumed-file "$CONSUMED_HTTP" --method GET --url "http://example.invalid/" 2>&1)
RC=$?
set -e
echo "$OUT"
check_refused "file-token @ tool_shim_http (wrong shim AND wrong action encoding)" "$OUT" "$RC"

# ---------------------------------------------------------------------
# Case 2: a token minted for TOOL=shell must be refused at tool_shim_file
# for the exact same content bytes.
# ---------------------------------------------------------------------
echo
echo "-- case 2: shell-exec token presented to the file shim --"
IDX2=102
CMD2="cross-shim-file-content"
AHASH_SHELL=$(printf '%s' "$CMD2" | sha256sum | awk '{print $1}')
TOKEN_SHELL=$("$ISSUE" "$KEY" "$IDX2" "$EXPIRY" shell "$CMD2")
echo "minted TOOL=shell token: $TOKEN_SHELL"

CONSUMED_FILE="$WORK/consumed.file.txt"; touch "$CONSUMED_FILE"
OUT_PATH="$WORK/should-not-be-written.txt"
set +e
OUT=$("$SHIM_FILE" --idx "$IDX2" --cap "$TOKEN_SHELL" --exp "$EXPIRY" --action-hash "$AHASH_SHELL" \
  --key-file "$KEY" --consumed-file "$CONSUMED_FILE" --path "$OUT_PATH" --data "$CMD2" 2>&1)
RC=$?
set -e
echo "$OUT"
check_refused "shell-token @ tool_shim_file" "$OUT" "$RC"
if [ -e "$OUT_PATH" ]; then
  echo "  FAIL: SECURITY FINDING -- $OUT_PATH was written despite the refused token"
  fail=$((fail+1))
fi

# ---------------------------------------------------------------------
# Positive control: the SAME file-write token from case 1, presented to
# the shim it was ACTUALLY minted for (tool_shim_file), must still ALLOW
# -- proves the fix does not just refuse everything.
# ---------------------------------------------------------------------
echo
echo "-- positive control: file-write token presented to its OWN shim (tool_shim_file) must still ALLOW --"
CONSUMED_FILE_OK="$WORK/consumed.file.ok.txt"; touch "$CONSUMED_FILE_OK"
OUT_PATH_OK="$WORK/legit-output.txt"
set +e
OUT=$("$SHIM_FILE" --idx "$IDX" --cap "$TOKEN_FILE" --exp "$EXPIRY" --action-hash "$AHASH_FILE" \
  --key-file "$KEY" --consumed-file "$CONSUMED_FILE_OK" --path "$OUT_PATH_OK" --data "$CONTENT" 2>&1)
RC=$?
set -e
echo "$OUT"
if [ "$RC" -eq 0 ] && [ -f "$OUT_PATH_OK" ] && [ "$(cat "$OUT_PATH_OK")" = "$CONTENT" ]; then
  echo "  PASS: same token correctly ALLOWed at its own (file) shim, file written"
  pass=$((pass+1))
else
  echo "  FAIL: expected the file-write token to still work at tool_shim_file"
  fail=$((fail+1))
fi

echo
echo "=== SUMMARY: pass=$pass fail=$fail ==="
if [ "$fail" -eq 0 ]; then
  echo "PASS (fails closed): a token minted for one tool is refused at every other tool's shim, and still works at its own shim"
  exit 0
else
  echo "FAIL: cross-shim token confusion is NOT fully closed (see failures above)"
  exit 1
fi
