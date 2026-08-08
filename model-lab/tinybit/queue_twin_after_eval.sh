#!/usr/bin/env bash
# queue_twin_after_eval.sh — waits for the aegis-ppl-rerun eval to finish, then
# resolves the open 4T-vs-8T question from the 2026-07-18 smoke (contended 8T
# measured SLOWER than 4T: 619.7 vs 1174-1310 tok/s) with a 50-step re-bench on
# the idle box, and finally starts the full twin run via the guarded launcher.
#
# Cancel the queue any time:   systemctl --user stop m7a-twin-queue
# Cancel the twin run itself:  systemctl --user stop m7a-twin
set -uo pipefail

TB=/home/killboxincorporated/model-lab/tinybit
VENV_PY=/home/killboxincorporated/ranger-venv/bin/python3
BENCH_CKPT="$TB/checkpoints/bench_tmp.pt"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] queue armed: waiting for aegis-ppl-rerun AND any m3-mc-*-full runs to exit"
while systemctl --user is-active --quiet aegis-ppl-rerun \
   || systemctl --user is-active --quiet m3-mc-1b-full \
   || systemctl --user is-active --quiet m3-mc-3b-full; do
    sleep 300
done
echo "[$(ts)] eval + MC baselines all done; starting 4T-vs-8T re-bench (50 steps each)"

bench() {
    # diagnostics -> stderr (lands in the unit log); stdout = seconds only
    local n=$1 t0 t1
    rm -f "$BENCH_CKPT"
    t0=$(date +%s)
    OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" nice -n 10 "$VENV_PY" "$TB/train.py" \
        --config "$TB/configs/m7a_twin.json" \
        --steps 50 --warmup 5 --log-every 10 \
        --save-every 100000 --eval-every 100000 \
        --threads "$n" --ckpt "$BENCH_CKPT" >&2 \
        || { echo "[$(ts)] BENCH FAILED at $n threads" >&2; return 1; }
    t1=$(date +%s)
    echo "[$(ts)] bench ${n}T: $((t1 - t0)) s for 50 steps" >&2
    echo $((t1 - t0))
}

T4=$(bench 4) || exit 1
T8=$(bench 8) || exit 1
rm -f "$BENCH_CKPT"

if [ "$T8" -le "$T4" ]; then WINNER=8; else WINNER=4; fi
echo "[$(ts)] re-bench result: 4T=${T4}s 8T=${T8}s for 50 steps -> launching twin with $WINNER threads"

TWIN_THREADS="$WINNER" "$TB/launch_m7a_twin.sh"
rc=$?
echo "[$(ts)] launcher exited rc=$rc"
exit "$rc"
