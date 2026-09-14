#!/bin/sh
# SAFE-5c: generate the two gateway HMAC keys on box2 ONLY. Never copy these
# off box2, never commit them, never let box1 read them (that is the entire
# point of the two-box split — see ../../ENFORCEMENT.md and
# state/reports/2026-09-13-SAFE5-ENFORCEMENT-BOUNDARY-DESIGN.md section (d),
# claudius-maximus repo).
#
# Idempotent: will not overwrite an existing key (rotating a live key here
# would orphan every capability token issued under the old one).
set -eu

DEST="${CM_GATEWAY_KEYDIR:-$HOME/.config/cm-gateway}"
install -d -m 700 "$DEST"

for f in allowlist.key cap.key; do
  path="$DEST/$f"
  if [ -f "$path" ]; then
    echo "gen-keys: $path already exists, leaving it alone" >&2
  else
    head -c 32 /dev/urandom > "$path"
    chmod 600 "$path"
    echo "gen-keys: wrote $path" >&2
  fi
done

echo "gen-keys: done. Keys live under $DEST, mode 700/600, box2-local only." >&2
