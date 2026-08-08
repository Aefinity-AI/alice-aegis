#!/usr/bin/env bash
# launch_m7a_twin.sh — guarded launcher for the FULL M7a fp-twin pretraining run.
#
#   arm:     configs/m7a_twin.json (fp32 nn.Linear twin, 6,529,920 params)
#   budget:  122,071 steps x (B=8 x T=512) = 500,002,816 tokens (~0.93 epoch of
#            train_8k.bin's 536,203,430 tokens)
#   threads: 8 (this run OWNS the box; do not start while aegis-ppl-rerun runs)
#   val:     PPL on a fixed 512-token slice of valid_8k.bin every 500 steps,
#            with train.py's existing 2-consecutive-rises tripwire
#   ckpt:    checkpoints/m7a_twin.pt, atomic write, resumable (--resume);
#            cadence set to ~30 min from the MEASURED 8-thread throughput of the
#            2026-07-18 smoke test (see SAVE_EVERY below)
#
# This script only STARTS the run under systemd; building it was the M7a lane
# deliverable. It refuses to run when preconditions fail.
set -euo pipefail

TB=/home/killboxincorporated/model-lab/tinybit
VENV_PY=/home/killboxincorporated/ranger-venv/bin/python3
LOG="$TB/logs/m7a_twin_full_2026-07-18.log"
CKPT="$TB/checkpoints/m7a_twin.pt"
UNIT=m7a-twin
# Thread count: 8 by default, overridable because the 2026-07-18 smoke measured
# contended 8T SLOWER than 4T; queue_twin_after_eval.sh re-benches on the idle
# box and passes the winner here.
THREADS="${TWIN_THREADS:-8}"

# --- guard 1: the long-running eval owns background capacity ---------------
if systemctl --user is-active --quiet aegis-ppl-rerun; then
    echo "REFUSING: user unit 'aegis-ppl-rerun' is still active. The twin run" >&2
    echo "wants 8 threads and must not contend with the eval. Try again after" >&2
    echo "it finishes (systemctl --user status aegis-ppl-rerun)." >&2
    exit 1
fi

# --- guard 2: DISK LAW, >= 8 GB free on / ----------------------------------
avail_kb=$(df --output=avail -k / | tail -1 | tr -d ' ')
if [ "$avail_kb" -lt $((8 * 1024 * 1024)) ]; then
    echo "REFUSING: only $((avail_kb / 1024 / 1024)) GB free on /, need >= 8 GB." >&2
    exit 1
fi

# --- guard 3: don't double-start -------------------------------------------
if systemctl --user is-active --quiet "$UNIT"; then
    echo "REFUSING: unit '$UNIT' is already running." >&2
    exit 1
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

# --- checkpoint cadence, derived from measured throughput ------------------
# Smoke test 2026-07-18 (logs/m7a_twin_smoke_2026-07-18.log) measured
# 619.7 tok/s in the 8-thread burst (NOTE: measured while aegis-ppl-rerun was
# still running — that burst was SLOWER than 4 threads (1174-1310 tok/s) due to
# HT oversubscription + swap; guard 1 ensures the real run starts uncontended,
# so 619.7 is a conservative floor). steps/30min = 1800 * 619.7 / 4096 = 272.3.
# If the uncontended run is faster, checkpoints simply land more often than
# every 30 min — the safe direction.
SAVE_EVERY=272

RESUME=()
if [ -f "$CKPT" ]; then
    RESUME=(--resume)
    echo "checkpoint exists -> resuming from $CKPT"
fi

mkdir -p "$TB/logs" "$TB/checkpoints"
echo "launching unit '$UNIT' ($THREADS threads, save every $SAVE_EVERY steps ~= 30 min," \
     "val every 500 steps); log: $LOG"

systemd-run --user --unit="$UNIT" \
    -p WorkingDirectory="$TB" \
    -p Environment=OMP_NUM_THREADS="$THREADS" \
    -p Environment=MKL_NUM_THREADS="$THREADS" \
    -p StandardOutput=append:"$LOG" \
    -p StandardError=append:"$LOG" \
    -p Nice=10 \
    "$VENV_PY" "$TB/train.py" \
        --config "$TB/configs/m7a_twin.json" \
        --threads "$THREADS" \
        --save-every "$SAVE_EVERY" \
        --eval-every 500 \
        "${RESUME[@]}"

echo "started. follow with: journalctl --user -u $UNIT -f   (or tail -f $LOG)"
