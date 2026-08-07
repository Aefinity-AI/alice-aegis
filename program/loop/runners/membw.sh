#!/usr/bin/env bash
# Closes TECHNICAL_REPORT.md:185 — "Measured machine bandwidth ceiling (single
# thread) | 17.3 GB/s" — the one claim in the 2026-07-29 audit with NO SOURCE
# ANYWHERE. Reports three figures and refuses to collapse them.
set -uo pipefail
REPO="${ALICE_REPO:-$HOME}"
BIN="$REPO/aegis-linux/target/release/examples/membw"
MB="${MB:-512}"; PASSES="${PASSES:-5}"; THREADS="${THREADS:-$(nproc)}"
[ -x "$BIN" ] || { echo "FATAL: build it first:" >&2
  echo "  CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release -p aegis-linux --example membw" >&2
  exit 2; }
echo "# free before: $(free -m | awk '/^Mem:/{print $7" MB available"}')"
exec "$BIN" "$MB" "$PASSES" "$THREADS"
