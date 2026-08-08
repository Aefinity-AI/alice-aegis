#!/usr/bin/env bash
# launch_m7lr_control.sh — guarded, durable launcher for the M7 LR/WD-cooldown ablation.
#
# NEW FILE. Modifies nothing that exists. Makes the two working COPIES of the twin
# checkpoint, runs the outside guards, then submits run_m7lr_control.sh as a transient
# systemd --user unit so it survives session close (Linger=yes is asserted below).
#
# NOTE ON THE EXISTING LAUNCHERS: do NOT use launch_m7a_twin.sh or launch_m7_ternary.sh
# for this experiment. Both auto-append --resume when the checkpoint exists and pass a
# config whose steps==122071, which now EQUALS the checkpoint step — so the training loop
# (train.py:167) would be empty and the unconditional final save (train.py:209-210) would
# silently rewrite the protected M7 headline checkpoint with zero training.
set -euo pipefail

TB=/home/killboxincorporated/model-lab/tinybit
LOGDIR=/home/killboxincorporated/docs/hardware_logs
STAMP=2026-07-29
LOG="$LOGDIR/m7lr_control_${STAMP}.log"
UNIT=m7-lr-control
THREADS=4

SRC="$TB/checkpoints/m7a_twin.pt"
TERN="$TB/checkpoints/m7_ternary.pt"
CK_H="$TB/checkpoints/m7lr_hold.pt"
CK_K="$TB/checkpoints/m7lr_cool.pt"

mkdir -p "$LOGDIR"

# --- guard 1: linger, or the unit dies with the session --------------------------
if ! loginctl show-user "$USER" -p Linger | grep -q 'Linger=yes'; then
    echo "REFUSING: Linger is not enabled; a --user unit would die at session close." >&2
    echo "  fix: loginctl enable-linger $USER" >&2
    exit 1
fi

# --- guard 2: DISK LAW, >= 8 GB free on / (verbatim from launch_m7a_twin.sh:36-41) -
avail_kb=$(df --output=avail -k / | tail -1 | tr -d ' ')
if [ "$avail_kb" -lt $((8 * 1024 * 1024)) ]; then
    echo "REFUSING: only $((avail_kb / 1024 / 1024)) GB free on /, need >= 8 GB." >&2
    exit 1
fi

# --- guard 3: don't double-start, and don't contend with anything real -------------
# (the existing launchers gate on 'aegis-ppl-rerun', a unit that no longer exists and
#  therefore always passes — a dead no-op guard. Gate on live units instead.)
for u in "$UNIT" m7a-twin m7-ternary tpu-retry full-backup-run; do
    if systemctl --user is-active --quiet "$u"; then
        echo "REFUSING: user unit '$u' is active; this run needs the box to itself." >&2
        echo "  check: systemctl --user status $u" >&2
        exit 1
    fi
done
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

# --- guard 4: the protected originals must exist and must NOT be the write targets --
for f in "$SRC" "$TERN"; do
    [ -f "$f" ] || { echo "REFUSING: missing protected checkpoint $f" >&2; exit 1; }
done

# --- make the working copies; refuse to clobber an existing arm ---------------------
for c in "$CK_H" "$CK_K"; do
    if [ -e "$c" ]; then
        echo "REFUSING: $c already exists. Move it aside or pick a new stamp; this" >&2
        echo "  launcher never overwrites an arm checkpoint." >&2
        exit 1
    fi
done
cp -n "$SRC" "$CK_H"
cp -n "$SRC" "$CK_K"
echo "copied $SRC ->"
echo "  $CK_H"
echo "  $CK_K"
md5sum "$SRC" "$CK_H" "$CK_K"

chmod +x "$TB/run_m7lr_control.sh"

systemd-run --user --unit="$UNIT" \
    -p WorkingDirectory="$TB" \
    -p Environment=OMP_NUM_THREADS="$THREADS" \
    -p Environment=MKL_NUM_THREADS="$THREADS" \
    -p StandardOutput=append:"$LOG" \
    -p StandardError=append:"$LOG" \
    -p Nice=10 \
    /bin/bash "$TB/run_m7lr_control.sh"

echo
echo "launched unit: $UNIT"
echo "NOTE: -p StandardOutput=append: means stdout NEVER reaches journald."
echo "      'journalctl --user -u $UNIT -f' will look EMPTY — that is not a dead run."
echo "  follow  : tail -f $LOG"
echo "  status  : systemctl --user status $UNIT"
echo "  done    : $TB/M7LR_CONTROL_DONE.txt appears"
echo "  stop    : systemctl --user stop $UNIT"
