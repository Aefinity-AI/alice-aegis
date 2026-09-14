#!/bin/sh
# SAFE-5c: box2-side forced command for the inter-box channel.
#
# This is the ONLY program box1's future SSH key is allowed to invoke on
# box2 (enforced by `command="..."` in box2's authorized_keys — see
# ../inter-box/sshd-authorized-keys-snippet — which overrides whatever
# command the box1 client actually asked for, plus
# no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty in the
# same authorized_keys line, plus the sshd_config restrictions in
# sshd_config.d-cm-box1.conf). box1 gets exactly two things through this
# channel, multiplexed by a one-line header on stdin, nothing else:
#
#   MODE GATEWAY
#     <the rest of stdin is relayed verbatim to the local gateway
#      decision socket ($XDG_RUNTIME_DIR/cm-gateway.sock); the socket's
#      one-line reply (ALLOW/DENY) is relayed verbatim back to box1>
#
#   MODE SHIM <shell|file|http> <idx> <cap> <exp> <action-hash-hex> [args...]
#     invokes the matching LOCAL tool_shim_* binary (built on box2, next
#     to this script) with that capability token; the shim independently
#     re-verifies the token before doing anything (src/capability.rs,
#     src/shim_common.rs) so a relay bug here is not itself a bypass —
#     it can at worst refuse to relay a legitimate call, never grant one
#     the shim wouldn't have granted anyway.
#
# Nothing else is reachable: this script never execs a shell with
# box1-controlled bytes, never `eval`s, never touches
# ~/.config/cm-gateway/*.key directly (only the gateway daemon process
# and the tool shim binaries read those, both already-built, fixed
# binaries -- this script only pipes bytes to them).
#
# STATUS: drafted, NOT installed. Needs box1's public key appended to
# box2's authorized_keys by someone with existing box1 access (Justin) --
# see ../../../../state/reports/2026-09-14-safe5c-box2-half.md.

set -eu

SOCK="${XDG_RUNTIME_DIR:?XDG_RUNTIME_DIR not set}/cm-gateway.sock"
SHIM_DIR="$(cd "$(dirname "$0")/../.." && pwd)/target/release"

read -r header rest || { echo "DENY malformed request (no header)"; exit 1; }

case "$header" in
  "MODE GATEWAY")
    # Relay everything else on stdin straight to the gateway's own unix
    # socket protocol (RECEIPT/ACTION/SESSION/COUNTER lines, blank line
    # or EOF ends the request) and relay its one-line reply back.
    # socat is the one extra binary this needs; ncat/nc -U would also
    # work if socat is not present on box2.
    exec socat - "UNIX-CONNECT:${SOCK}"
    ;;
  "MODE SHIM "*)
    tool="${header#MODE SHIM }"
    tool="${tool%% *}"
    argrest="${header#MODE SHIM "$tool" }"
    case "$tool" in
      shell) bin="$SHIM_DIR/tool_shim_shell" ;;
      file)  bin="$SHIM_DIR/tool_shim_file" ;;
      http)  bin="$SHIM_DIR/tool_shim_http" ;;
      *) echo "DENY unknown shim $tool"; exit 1 ;;
    esac
    # shellcheck disable=SC2086 -- argrest is a fixed, space-separated
    # --idx/--cap/--exp/--action-hash/payload sequence the shim itself
    # parses defensively (src/shim_common.rs); no shell metacharacter in
    # it is ever eval'd, only word-split into argv.
    exec "$bin" $argrest
    ;;
  *)
    echo "DENY unrecognized MODE"
    exit 1
    ;;
esac
