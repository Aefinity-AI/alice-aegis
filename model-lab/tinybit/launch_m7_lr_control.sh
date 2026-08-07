#!/usr/bin/env bash
# launch_m7_lr_control.sh — guarded launcher for the M7 LR-floor control.
#
#   question: how much of the ternary arm's win over the fp32 twin is explained
#             by the end-of-training learning-rate floor rather than by ternary
#             quantization?
#   method:   resume the FINISHED twin checkpoint and give it the cooldown it
#             never had (lr 1.00e-04 -> 0 cosine, wd 0.1 -> 0), then re-score
#             with the paired multi-window eval.
#   arm:      m7a_twin.pt (6,529,920 params, fp32) -> m7a_twin_cooled.pt
#   budget:   4,000 steps x (B=8 x T=512) = 16,384,000 tokens
#   threads:  4 (MEASURED optimum on this box; 8T was slower — HT contention)
#   estimate: ~2.2h at the twin run's measured 2,110 tok/s
#
# The source checkpoint is copied before any write; the original is never
# modified. train.py is NOT edited — the round-trip gate depends on it.
#
# PRE-REGISTERED INTERPRETATION (written before the run, per SESSION_PROTOCOL):
#   Baseline, measured 2026-07-29 on 400 disjoint windows / 204,800 tokens:
#     twin corpus PPL 4.4816, ternary 4.1664 -> ternary 7.03% better,
#     paired t +40.69, ternary wins 396/400 windows.
#   After the twin is cooled, re-run the SAME eval and read the outcome as:
#     * ternary lead shrinks to  < 1.0%          -> the "win" was mostly the LR
#                                                   floor. Retract the ternary
#                                                   superiority claim.
#     * lead lands in 1.0% - 4.0%                -> the LR floor explains a
#                                                   large share. Restate as
#                                                   "comparable, not superior".
#     * lead stays > 4.0% and t > 2              -> the LR floor is NOT the
#                                                   explanation. Claim survives,
#                                                   but ONLY in its honest form
#                                                   (14.2M ternary vs 6.5M fp32,
#                                                   seven knobs differ).
#   No outcome permits the sentence "ternary beats fp32 at equal budget" —
#   the 2.17x parameter gap is untouched by this experiment and needs a
#   same-size pair (~130h) to settle.
set -euo pipefail

TB=/home/killboxincorporated/model-lab/tinybit
VENV_PY=/home/killboxincorporated/ranger-venv/bin/python3
LOG="$TB/logs/m7_lr_control_2026-07-29.log"
SRC="$TB/checkpoints/m7a_twin.pt"
OUT="$TB/checkpoints/m7a_twin_cooled.pt"
UNIT=m7-lr-control
THREADS="${CONTROL_THREADS:-4}"
STEPS="${CONTROL_STEPS:-4000}"

# --- guard 1: source checkpoint must exist and match its recorded hash -------
EXPECT_SHA=7179a67427873ca47e6dbbf766a7ec76a53d531b2a8836f6adf60b107204cc01
[ -f "$SRC" ] || { echo "REFUSING: source checkpoint missing: $SRC" >&2; exit 1; }
GOT_SHA=$(sha256sum "$SRC" | cut -d' ' -f1)
if [ "$GOT_SHA" != "$EXPECT_SHA" ]; then
    echo "REFUSING: $SRC hash mismatch." >&2
    echo "  expected $EXPECT_SHA" >&2
    echo "  got      $GOT_SHA" >&2
    exit 1
fi

# --- guard 2: no competing heavy tenant -------------------------------------
for u in aegis-ppl-rerun m7a-twin m7-ternary m3-mc-1b-full m3-mc-3b-full; do
    if systemctl --user is-active --quiet "$u" 2>/dev/null; then
        echo "REFUSING: user unit '$u' is active; this control must run uncontended." >&2
        exit 1
    fi
done

# --- guard 3: DISK LAW, >= 8 GB free on / -----------------------------------
avail_kb=$(df --output=avail -k / | tail -1 | tr -d ' ')
if [ "$avail_kb" -lt $((8 * 1024 * 1024)) ]; then
    echo "REFUSING: only $((avail_kb / 1024 / 1024)) GB free on /, need >= 8 GB." >&2
    exit 1
fi

# --- guard 4: don't double-start --------------------------------------------
if systemctl --user is-active --quiet "$UNIT"; then
    echo "REFUSING: unit '$UNIT' is already running." >&2
    exit 1
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null || true

mkdir -p "$TB/logs" "$TB/checkpoints"
echo "launching unit '$UNIT' ($THREADS threads, $STEPS steps); log: $LOG"

systemd-run --user --unit="$UNIT" \
    -p WorkingDirectory="$TB" \
    -p Environment=OMP_NUM_THREADS="$THREADS" \
    -p Environment=MKL_NUM_THREADS="$THREADS" \
    -p StandardOutput=append:"$LOG" \
    -p StandardError=append:"$LOG" \
    -p Nice=10 \
    "$VENV_PY" "$TB/m7_lr_cooldown.py" \
        --src-ckpt "$SRC" \
        --out-ckpt "$OUT" \
        --steps "$STEPS" \
        --lr-start 1.0e-4 \
        --threads "$THREADS" \
        --eval-every 250 \
        --save-every 500

echo "started. follow with: journalctl --user -u $UNIT -f   (or tail -f $LOG)"
