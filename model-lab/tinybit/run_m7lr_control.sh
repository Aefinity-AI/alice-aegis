#!/usr/bin/env bash
# run_m7lr_control.sh — serial two-arm LR/WD-cooldown ablation payload for M7.
#
# NEW FILE. Invoked by launch_m7lr_control.sh under systemd-run --user. Do not run
# this directly for the real experiment (you lose durability); the launcher does the
# outside guards and submits this as a transient unit.
#
# Modifies NOTHING that exists. train.py, model.py, roundtrip_gate.py,
# m7_final_roundtrip.py, both configs/*.json, m7a_twin.pt and m7_ternary.pt are all
# read-only here. The two arms write ONLY to fresh copies.
#
# ARM H (hold-control, the null) : +2000 steps at the twin's ORIGINAL recipe
#                                  (sched=cosine, lr pinned at its 1e-4 floor, wd 0.1).
#                                  Measures what 2000 extra tokens alone buy.
# ARM K (cooldown, the treatment): +2000 steps annealing lr 1.000e-04 -> 2.50e-11 with
#                                  wd forced to 0. Measures cooldown + the same 2000
#                                  extra tokens.
# H vs K is therefore TOKEN-MATCHED and isolates the LR/WD cooldown itself.
set -euo pipefail

TB=/home/killboxincorporated/model-lab/tinybit
VENV_PY=/home/killboxincorporated/ranger-venv/bin/python3
LOGDIR=/home/killboxincorporated/docs/hardware_logs
STAMP=2026-07-29
LOG="$LOGDIR/m7lr_control_${STAMP}.log"
EVAL_LOG="$LOGDIR/m7lr_paired_eval_${STAMP}.log"
EVAL_JSON="$LOGDIR/m7lr_paired_eval_${STAMP}.json"
SENTINEL="$TB/M7LR_CONTROL_DONE.txt"

SRC="$TB/checkpoints/m7a_twin.pt"
TERN="$TB/checkpoints/m7_ternary.pt"
CK_H="$TB/checkpoints/m7lr_hold.pt"
CK_K="$TB/checkpoints/m7lr_cool.pt"

START_STEP=122071
N=2000
TOTAL=$((START_STEP + N))          # 124071
# Back-solved so that lr_schedule()'s stage-2 cosine passes through EXACTLY 1.000e-04
# at step 122071 and lands at 2.50e-11 at step 124070. See the rationale banner below.
COOL_PEAK=0.390265
THREADS=4

mkdir -p "$LOGDIR"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

exec 2> >(tee -a "$LOG" >&2)

{
echo "================================================================================"
echo "[$(ts)] M7 LR/WD-COOLDOWN CONFOUND ABLATION — run_m7lr_control.sh"
echo "================================================================================"
echo "PURPOSE"
echo "  Bound how much of the published M7 'ternary beats fp32' gap is explained by the"
echo "  UNDISCLOSED end-of-training LR/WD cooldown rather than by ternary quantization."
echo "  Published arms differ in SEVEN variables. This controls exactly ONE of them."
echo ""
echo "ARMS (all resume from step $START_STEP; both write ONLY to fresh copies)"
echo "  A  m7a_twin.pt      published fp twin, step $START_STEP, untouched, read-only"
echo "  T  m7_ternary.pt    published ternary,  step $START_STEP, untouched, read-only"
echo "  H  m7lr_hold.pt     A + $N steps, sched=cosine lr~1.00e-04 flat, wd 0.1 held"
echo "                      => token-matched NULL control (extra tokens, original recipe)"
echo "  K  m7lr_cool.pt     A + $N steps, sched=two_stage lr 1.00e-04 -> 2.50e-11, wd 0"
echo "                      => the cooldown TREATMENT"
echo ""
echo "THE --lr $COOL_PEAK IS A SCHEDULE-SHAPE PARAMETER, NOT A LEARNING RATE."
echo "  train.py exposes no --min-lr-mult / --stage2-frac / --stage2-lr-mult flag; the"
echo "  floor/knee args of lr_schedule() (train.py:34) are pinned at the call site"
echo "  (train.py:172, 3 positional args). Only sched=two_stage can reach a true zero"
echo "  floor (min_lr_mult=0.0) and only it drops wd to 0 (wd_for_step, train.py:50)."
echo "  Resuming at step $START_STEP is already deep in stage 2 (stage2_start ="
echo "  int(0.5*$TOTAL) = $((TOTAL / 2))), so peak_lr is NEVER APPLIED — it acts purely as a"
echo "  scale knob on the stage-2 cosine. Solving lr_schedule($START_STEP,$TOTAL,2000,peak)"
echo "  = 1.0e-4 gives peak = $COOL_PEAK. Verified against train.py's own function:"
echo "    step 122071 -> 1.000000e-04   (continuous with the twin's final 1.0000e-04)"
echo "    step 122571 -> 5.627e-05    step 123071 -> 2.502e-05"
echo "    step 123571 -> 6.255e-06    step 124070 -> 2.502e-11"
echo "    wd = 0.000 at every step of the window."
echo "  This value is ONLY correct for --steps $TOTAL resuming at $START_STEP. Changing"
echo "  --steps without recomputing it silently starts the cooldown at the wrong lr."
echo ""
echo "DISCLOSED DEVIATIONS (state these in any writeup; do not let them be found later)"
echo "  1. Arms H and K each see $((N * 8 * 512)) tokens MORE than arms A and T"
echo "     (+$(awk "BEGIN{printf \"%.2f\", $N*8*512/500002816*100}")% on the 500,002,816-token budget). H exists precisely to price that."
echo "  2. The resumed data stream is NOT a continuation: train.py:156 builds a fresh"
echo "     np.random.default_rng(seed + start_step) = default_rng($((1337 + START_STEP))). Both H and K"
echo "     get the IDENTICAL batch sequence, so H/K is paired at the data level too."
echo "  3. load_ckpt (train.py:136) ignores ckpt['config']; architecture is protected"
echo "     only by strict state_dict loading. Asserted explicitly below."
echo "================================================================================"
} | tee -a "$LOG"

# --- preflight: the copies must exist, be the right thing, and not be the originals --
say "PREFLIGHT"
for f in "$SRC" "$TERN" "$CK_H" "$CK_K"; do
    [ -f "$f" ] || { say "FATAL: missing $f"; exit 1; }
done
if [ "$(readlink -f "$CK_H")" = "$(readlink -f "$SRC")" ] || \
   [ "$(readlink -f "$CK_K")" = "$(readlink -f "$SRC")" ]; then
    say "FATAL: an arm checkpoint path resolves to the protected original. Refusing."
    exit 1
fi
SRC_MD5=$(md5sum "$SRC" | cut -d' ' -f1)
TERN_MD5=$(md5sum "$TERN" | cut -d' ' -f1)
say "protected m7a_twin.pt  md5 $SRC_MD5"
say "protected m7_ternary.pt md5 $TERN_MD5"

"$VENV_PY" - "$CK_H" "$CK_K" "$START_STEP" <<'PY' | tee -a "$LOG"
import sys, torch
for p in sys.argv[1:3]:
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    step, cfg = ck["step"], ck["config"]
    assert step == int(sys.argv[3]), f"{p}: step {step} != {sys.argv[3]} (not a clean copy)"
    assert cfg["linear"] == "fp", f"{p}: linear={cfg['linear']}, expected fp"
    assert cfg["num_hidden_layers"] == 6 and cfg["hidden_size"] == 256, f"{p}: wrong arch"
    print(f"[preflight] {p}: step {step} linear={cfg['linear']} "
          f"L{cfg['num_hidden_layers']}/H{cfg['hidden_size']} OK")
print("[preflight] both arm copies verified")
PY

# guards the empty-loop-that-still-overwrites failure (train.py:167 + 209-210)
if [ "$TOTAL" -le "$START_STEP" ]; then
    say "FATAL: --steps $TOTAL <= checkpoint step $START_STEP; the loop would be empty"
    say "       and train.py:209-210 would still re-save. Refusing."
    exit 1
fi
say "steps guard OK: $TOTAL > $START_STEP ($N cooldown steps)"

run_arm() {
    local name="$1" ckpt="$2"; shift 2
    say "---------------- ARM $name START ----------------"
    say "argv: $VENV_PY $TB/train.py --config $TB/configs/m7a_twin.json --ckpt $ckpt --resume $* --steps $TOTAL --threads $THREADS"
    local t0=$SECONDS
    "$VENV_PY" "$TB/train.py" \
        --config "$TB/configs/m7a_twin.json" \
        --ckpt "$ckpt" \
        --resume \
        "$@" \
        --steps "$TOTAL" \
        --threads "$THREADS" \
        --save-every 500 \
        --eval-every 250 \
        --log-every 25 2>&1 | tee -a "$LOG"
    say "---------------- ARM $name DONE in $((SECONDS - t0))s ----------------"
}

# ARM H first (the null). If the box dies mid-experiment we would rather hold the
# control than hold a treatment with nothing to compare it to.
run_arm H "$CK_H" --sched cosine     --lr 1e-3      --wd 0.1
run_arm K "$CK_K" --sched two_stage  --lr "$COOL_PEAK" --wd 0.0

# --- the originals must be byte-identical to what we started with ------------------
say "POST-RUN INTEGRITY"
NEW_SRC=$(md5sum "$SRC" | cut -d' ' -f1)
NEW_TERN=$(md5sum "$TERN" | cut -d' ' -f1)
[ "$NEW_SRC" = "$SRC_MD5" ] || { say "FATAL: m7a_twin.pt CHANGED ($SRC_MD5 -> $NEW_SRC)"; exit 1; }
[ "$NEW_TERN" = "$TERN_MD5" ] || { say "FATAL: m7_ternary.pt CHANGED"; exit 1; }
say "both protected checkpoints byte-identical: OK"

# --- both arms must have actually finished, or the eval is scoring a partial anneal --
"$VENV_PY" - "$CK_H" "$CK_K" "$TOTAL" <<'PY' | tee -a "$LOG"
import sys, torch
for p in sys.argv[1:3]:
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    assert ck["step"] == int(sys.argv[3]), (
        f"{p}: step {ck['step']} != {sys.argv[3]} — arm did not complete; "
        f"scoring it would evaluate a PARTIAL anneal. Refusing to eval.")
    lr = ck["optim"]["param_groups"][0]["lr"]
    wd = ck["optim"]["param_groups"][0]["weight_decay"]
    print(f"[postflight] {p}: step {ck['step']} final lr {lr:.4e} wd {wd:.3f}")
PY

# --- paired multi-window eval, all four arms, one process, identical windows -------
say "PAIRED EVAL (4 arms x 512 disjoint 512-token windows)"
"$VENV_PY" "$TB/m7lr_paired_eval.py" \
    --arm "A=$SRC" \
    --arm "H=$CK_H" \
    --arm "K=$CK_K" \
    --arm "T=$TERN" \
    --windows 512 \
    --threads "$THREADS" \
    --out "$EVAL_JSON" \
    --log "$EVAL_LOG" 2>&1 | tee -a "$LOG"

{
echo "[$(ts)] ALL ARMS + EVAL COMPLETE"
echo "  train log : $LOG"
echo "  eval log  : $EVAL_LOG"
echo "  eval json : $EVAL_JSON"
echo "  arm H     : $CK_H"
echo "  arm K     : $CK_K"
echo "  protected : m7a_twin.pt md5 $SRC_MD5 (unchanged), m7_ternary.pt md5 $TERN_MD5 (unchanged)"
} | tee -a "$SENTINEL" | tee -a "$LOG"
say "sentinel written: $SENTINEL"
