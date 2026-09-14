#!/usr/bin/env bash
# SAFE-5c escape test 5: from box1, try to reach box2's credential store
# (~/.config/cm-gateway/{allowlist,cap}.key) or tool unix socket
# ($XDG_RUNTIME_DIR/cm-gateway.sock) by any means OTHER than the sanctioned
# forced-command channel (../../deploy/inter-box/), and confirm refusal.
#
# STATUS: NOT RUNNABLE on this branch. Exits 2 immediately unless
# CM_SAFE5C_BOX1_READY=1 is set, which nothing sets today, because:
#   - box2 (this host) has no outbound SSH credential to box1 at all
#     (~/.ssh has only authorized_keys + known_hosts, no private key, no
#     box1 host entry) -- verified 2026-09-14.
#   - box1 has not generated the dedicated cm-box1-agent-relay-only
#     keypair described in ../../deploy/inter-box/box1-gateway-client.sh.
#   - box2's authorized_keys has no forced-command entry for box1 (see
#     ../../deploy/inter-box/sshd-authorized-keys-snippet — a template,
#     not installed).
# Unblock: someone with existing box1 access must (1) generate
# ~/.ssh/cm-box1-agent-relay-only on box1, (2) append its PUBLIC half to
# box2's ~/.ssh/authorized_keys using the exact command=/no-* line in
# sshd-authorized-keys-snippet (not a bare key), (3) confirm box1 can
# resolve/route to box2 (CM_BOX2_HOST). None of that can happen from box2
# alone. See state/reports/2026-09-14-safe5c-box2-half.md.
#
# Once ready, this script would attempt (from box1), each expected to
# FAIL/refuse:
#   1. scp/cat of ~/.config/cm-gateway/*.key over the SAME forced key
#      (the forced command ignores the requested command, including scp's
#      internal `scp -t` protocol invocation -- ForceCommand always wins).
#   2. `ssh -R`/`ssh -L` port-forward attempt using the forced key ->
#      must be refused at the sshd level (no-port-forwarding /
#      AllowTcpForwarding no).
#   3. Direct TCP connect from box1 to any port box2 might expose for the
#      gateway -- there should be none; the gateway socket is AF_UNIX
#      only and the gateway unit itself has PrivateNetwork=yes
#      (deploy/box2/cm-gateway.service), so this is N/A by construction,
#      not just by firewall -- worth confirming with an nmap from box1
#      once reachable.
#   4. Using the forced key's ssh session to request a pty / run an
#      arbitrary command string -> ForceCommand + no-pty must ignore it.
#   5. Presenting ANY other box1 key (not the dedicated relay-only one)
#      against box2's sshd -> must fail authentication entirely (it is
#      not in authorized_keys), confirming the credential store is not
#      reachable by a box1 identity that predates this feature.

set -u
if [ "${CM_SAFE5C_BOX1_READY:-0}" != "1" ]; then
  echo "SKIP: escape test 5 needs box1<->box2 key exchange, not present on this box (box2)." >&2
  echo "See state/reports/2026-09-14-safe5c-box2-half.md for the exact unblock step." >&2
  exit 2
fi

echo "not implemented: fill in once box1 access exists" >&2
exit 2
