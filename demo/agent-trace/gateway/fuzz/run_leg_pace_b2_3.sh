#!/usr/bin/env bash
# leg-pace-b2-3: SAFE-7c cargo-fuzz continuation (targets built for #106,
# c4993e4). Runs both targets sequentially, 3600s each, under the
# systemd unit's MemoryMax=2G. Raw logs land next to this script so a
# later tick's harvest can pull execs/crashes/corpus size out of them.
set -euo pipefail
cd "$(dirname "$0")/.."

LOGDIR="$(dirname "$0")/logs"
mkdir -p "$LOGDIR"

echo "leg-pace-b2-3: starting fuzz_request_parser at $(date -u +%FT%TZ)"
cargo +nightly fuzz run fuzz_request_parser -- \
  -max_total_time=3600 -rss_limit_mb=1700 \
  > "$LOGDIR/fuzz_request_parser.log" 2>&1 || true
echo "leg-pace-b2-3: fuzz_request_parser done at $(date -u +%FT%TZ)"

echo "leg-pace-b2-3: starting fuzz_token_verifier at $(date -u +%FT%TZ)"
cargo +nightly fuzz run fuzz_token_verifier -- \
  -max_total_time=3600 -rss_limit_mb=1700 \
  > "$LOGDIR/fuzz_token_verifier.log" 2>&1 || true
echo "leg-pace-b2-3: fuzz_token_verifier done at $(date -u +%FT%TZ)"

echo "leg-pace-b2-3: all targets finished at $(date -u +%FT%TZ)"
