#!/usr/bin/env bash
# capture-bench.sh — run a bench binary and emit a Rule-B-shaped instrument log.
#
# Rule B admits three parents for a number: an instrument log, a written-down
# derivation from instrument-backed numbers, or an external citation. A source
# comment is not one of them. This script produces the first kind.
#
# Every capture carries, in the log itself:
#   - the machine, CPU, core count and loadavg at capture time (Rule: name the
#     machine; benchmark hygiene needs the idle state on the record, not in
#     someone's memory)
#   - the git HEAD and branch the binary was built from
#   - the CLOCK STATE block from `clockstate` — Rule A's RDTSC corollary. Any
#     tick-derived figure is wrong by the effective/nominal ratio without it.
#   - the raw, unedited stdout+stderr of the bench
#
# The equivalent of this script existed only in a scratchpad and was lost to a
# crash, which cost a re-derivation. It is checked in now.
#
# Usage: scripts/capture-bench.sh <bench-bin-name> <log-basename> [title]
#   e.g. scripts/capture-bench.sh lut_mpgemm lut_mpgemm "LUT-mpGEMM pshufb vs f32 LUT+FMA"
#
# Writes docs/hardware_logs/<log-basename>_<UTC-date>.log and refuses to
# overwrite an existing one (hardware_logs is append-only, Rule C).

set -euo pipefail

BIN=${1:?usage: capture-bench.sh <bench-bin-name> <log-basename> [title]}
BASENAME=${2:?usage: capture-bench.sh <bench-bin-name> <log-basename> [title]}
TITLE=${3:-$BIN}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DATE_TAG=$(date -u +%Y-%m-%d)
OUT="$REPO/docs/hardware_logs/${BASENAME}_${DATE_TAG}.log"

if [ -e "$OUT" ]; then
    echo "REFUSING: $OUT already exists." >&2
    echo "hardware_logs/ is append-only (Rule C). Pick a new basename." >&2
    exit 1
fi

# Idle check. Benchmark hygiene (D3): a loaded machine silently poisons a
# capture, and that already cost this program a day of numbers.
LOADAVG=$(cut -d' ' -f1-3 /proc/loadavg)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
if awk "BEGIN{exit !($LOAD1 > 1.0)}"; then
    echo "WARNING: 1-min loadavg is $LOAD1 — this capture may be poisoned." >&2
    echo "         Waiting 30s for the machine to settle." >&2
    sleep 30
    LOADAVG=$(cut -d' ' -f1-3 /proc/loadavg)
fi

CPU=$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)
CORES=$(nproc)
GIT_HEAD=$(git -C "$REPO" rev-parse --short HEAD)
GIT_BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
MACHINE=${AEGIS_MACHINE:-"dev box, crosvm guest (Chromebook)"}

cd "$REPO/aegis-core"
cargo build --release --bin "$BIN" >/dev/null 2>&1

{
    echo "=== ${TITLE} — instrument capture ==="
    echo "date_utc          : ${DATE_UTC}"
    echo "machine           : ${MACHINE}"
    echo "cpu               : ${CPU}"
    echo "cores_online      : ${CORES}"
    echo "loadavg           : ${LOADAVG}"
    echo "binary            : aegis-core/target/release/${BIN} (release, default features)"
    echo "git_head          : ${GIT_HEAD} (${GIT_BRANCH})"
    echo ""
    (cd "$REPO/aegis-linux" && cargo run --release --example clockstate 2>/dev/null)
    echo ""
    echo "--- BENCH RUN 1 ---"
    "$REPO/aegis-core/target/release/${BIN}" 2>&1
} | tee "$OUT"

echo ""
echo "wrote $OUT"
