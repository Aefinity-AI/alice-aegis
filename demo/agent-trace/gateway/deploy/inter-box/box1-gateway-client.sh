#!/bin/sh
# SAFE-5c: box1-side client. Runs on box1 (the untrusted agent host).
# Sends a decision request or a shim invocation to box2 over the forced
# SSH channel; box2's sshd ignores whatever we ask for here and always
# runs gateway-relay.sh instead (see ../inter-box/sshd-authorized-keys-snippet).
#
# Usage:
#   box1-gateway-client.sh gateway <<'EOF'
#   RECEIPT /path/on/box1/or/shared/fs/foo.receipt
#   ACTION <hex>
#   SESSION s1
#   COUNTER 1
#
#   EOF
#
#   box1-gateway-client.sh shim shell --idx 3 --cap <hex> --exp 123 \
#       --action-hash <hex> -- <shell argv...>
#
# STATUS: drafted, NOT usable yet -- box2 has no key for box1 in
# ~/.ssh/known_hosts/authorized_keys and box1 has generated no dedicated
# agent keypair. See state/reports/2026-09-14-safe5c-box2-half.md for the
# precise unblock step.

set -eu
BOX2_HOST="${CM_BOX2_HOST:-192.168.10.21}"  # box2's fixed LAN alias, confirmed in state/BOXES.md
BOX2_USER="${CM_BOX2_USER:-cm-box1-agent}"
IDENTITY="${CM_BOX1_AGENT_KEY:-$HOME/.ssh/cm-box1-agent-relay-only}"

mode="${1:?usage: $0 gateway|shim ...}"
shift

case "$mode" in
  gateway)
    { echo "MODE GATEWAY"; cat; } | \
      ssh -i "$IDENTITY" -o BatchMode=yes -o StrictHostKeyChecking=yes \
          "${BOX2_USER}@${BOX2_HOST}"
    ;;
  shim)
    tool="${1:?usage: $0 shim <shell|file|http> [args...]}"
    shift
    { echo "MODE SHIM $tool $*"; } | \
      ssh -i "$IDENTITY" -o BatchMode=yes -o StrictHostKeyChecking=yes \
          "${BOX2_USER}@${BOX2_HOST}"
    ;;
  *)
    echo "usage: $0 gateway|shim ..." >&2
    exit 2
    ;;
esac
