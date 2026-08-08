#!/usr/bin/env bash
# run_m7lr_armh_verdict.sh — arm H (the null), then the decisive paired eval.
#
# Runs under systemd unit m7lr-armh-verdict. Waits for arm K (unit
# m7-lr-control) to finish, verifies integrity, then:
#   ARM H : the SAME +4000 steps as arm K but at the twin's ORIGINAL recipe
#           (sched=cosine, lr held ~1.0e-04, wd 0.1). This is the NULL: it
#           prices the extra tokens so that d_cool = NLL_H - NLL_K isolates
#           the LR/WD cooldown with the token confound differenced out.
#   EVAL  : 512 disjoint windows, all four arms, per the locked protocol in
#           docs/hardware_logs/m7lr_PREREGISTRATION_2026-07-29.md (git 1ee1f5c).
#
# H and K are exactly paired at the data level: train.py:156 and
# m7_lr_cooldown.py:167 both build np.random.default_rng(1337+122071).
#
# REFUSES to evaluate a partial anneal. Never writes the published checkpoints.
set -uo pipefail

TB=/home/killboxincorporated/model-lab/tinybit
HW=/home/killboxincorporated/docs/hardware_logs
VENV_PY=/home/killboxincorporated/ranger-venv/bin/python3
LOG="$HW/m7lr_armh_verdict_2026-07-29.log"

A="$TB/checkpoints/m7a_twin.pt"          # published twin  (protected)
T="$TB/checkpoints/m7_ternary.pt"        # published ternary (protected)
K="$TB/checkpoints/m7a_twin_cooled.pt"   # arm K, produced by m7-lr-control
H="$TB/checkpoints/m7lr_hold.pt"         # arm H, produced here

SHA_A=7179a67427873ca47e6dbbf766a7ec76a53d531b2a8836f6adf60b107204cc01
SHA_T=0f203221a2439593ce55882f48a81b5f3cca72fd5cfad5fbaec81a63207c7f63
STEPS_TOTAL=126071   # 122071 + 4000, matching arm K exactly

say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

say "=== arm H + verdict runner started ==="
say "pre-registration: $HW/m7lr_PREREGISTRATION_2026-07-29.md (git 1ee1f5c)"

# ---- 1. wait for arm K -----------------------------------------------------
say "waiting for unit m7-lr-control (arm K) to finish..."
while systemctl --user is-active --quiet m7-lr-control 2>/dev/null; do sleep 60; done
say "arm K unit is no longer active"

# ---- 2. integrity + completeness gates -------------------------------------
fail() { say "ABORT: $*"; exit 1; }

[ -f "$K" ] || fail "arm K checkpoint missing: $K"
got_a=$(sha256sum "$A" | cut -d' ' -f1)
got_t=$(sha256sum "$T" | cut -d' ' -f1)
[ "$got_a" = "$SHA_A" ] || fail "published twin was MODIFIED (sha $got_a). Experiment void."
[ "$got_t" = "$SHA_T" ] || fail "published ternary was MODIFIED (sha $got_t). Experiment void."
say "published checkpoints verified unmodified"

if ! grep -q '\[cooldown\] done' "$TB/logs/m7_lr_control_2026-07-29.log"; then
    fail "arm K did not reach completion ('[cooldown] done' absent). REFUSING to \
evaluate a partial anneal — a truncated cooldown understates d_cool and would \
bias the result toward exonerating the claim under audit."
fi
k_last=$(grep '^cool ' "$TB/logs/m7_lr_control_2026-07-29.log" | tail -1)
say "arm K final line: $k_last"

# sanity abort from the pre-registration: arm K must end at ~1.5e-11, wd 0.000
echo "$k_last" | grep -q 'wd 0.000' || fail "arm K did not run at wd 0 — misconfigured, no band applies"

# ---- 3. arm H --------------------------------------------------------------
if [ -f "$H" ]; then
    say "arm H checkpoint already exists; refusing to overwrite: $H"
else
    cp -p "$A" "$H"
    say "arm H: copied published twin -> $H (original untouched)"
    say "arm H: running $STEPS_TOTAL total steps at the ORIGINAL recipe (cosine, wd 0.1)"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 \
      "$VENV_PY" "$TB/train.py" \
        --config "$TB/configs/m7a_twin.json" \
        --ckpt "$H" \
        --steps "$STEPS_TOTAL" \
        --threads 4 \
        --save-every 500 \
        --eval-every 250 \
        --resume >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] || fail "arm H training exited rc=$rc"
fi

h_last=$(grep -E '^step +12[0-9]{4}/' "$LOG" | tail -1)
say "arm H final line: $h_last"
# sanity abort: arm H must have held lr ~1e-4 with wd 0.100
echo "$h_last" | grep -q 'wd 0.100' || fail "arm H did not hold wd 0.1 — misconfigured, no band applies"

# ---- 4. the decisive eval --------------------------------------------------
say "=== PAIRED EVAL — 512 disjoint windows, locked protocol ==="
cd "$TB" || fail "cannot cd $TB"

run_eval() {   # $1 = baseline, rest = challengers
    local base="$1"; shift
    local args=()
    for c in "$@"; do args+=(--compare "$c"); done
    OMP_NUM_THREADS=4 "$VENV_PY" "$TB/m7_paired_eval.py" \
        --ckpt twin="$A" --ckpt twin_hold="$H" \
        --ckpt twin_cooled="$K" --ckpt ternary="$T" \
        --windows 512 --baseline "$base" "${args[@]}" \
        --json-out "$HW/m7lr_verdict_${base}_2026-07-29.json" 2>&1 | tee -a "$LOG"
}

say "--- contrast set 1: baseline twin_hold (H)  ->  d_cool = H - K ---"
run_eval twin_hold twin_cooled ternary twin

say "--- contrast set 2: baseline twin (A)  ->  d_gap = A - T, d_tokens = A - H ---"
run_eval twin ternary twin_hold twin_cooled

say "--- contrast set 3: baseline twin_cooled (K)  ->  d_resid = K - T ---"
run_eval twin_cooled ternary

say "=== RUN COMPLETE ==="
say "Read the bands in $HW/m7lr_PREREGISTRATION_2026-07-29.md before interpreting."
say "REMINDER: d_cool is a LOWER BOUND (arm K anneals 3.3% of training; the"
say "ternary annealed 25%). No outcome licenses 'at equal budget' — the ternary"
say "arm has 2.17x the parameters."
touch "$TB/M7LR_VERDICT_DONE.txt"
