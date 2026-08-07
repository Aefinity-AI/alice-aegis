#!/usr/bin/env bash
# tpu_retry.sh — daily off-peak retry of the A4 TPU smoke (ledger M23:
# Kaggle silently downgrades TPU requests to CPU when the pool is dry).
# Logs one line per attempt; stops itself once a TPU is actually granted.
set -uo pipefail
export KAGGLE_API_TOKEN=$(cat /home/killboxincorporated/.kaggle/token)
K=/home/killboxincorporated/ranger-venv/bin/kaggle
D=/home/killboxincorporated/model-lab/kaggle/a4-tpu-smoke
LOG=/home/killboxincorporated/model-lab/kaggle/tpu_retry.log
ts() { date -u +%FT%TZ; }

[ -f "$D/TPU_GRANTED" ] && exit 0
cd "$D" || exit 1
$K kernels push -p . >/dev/null 2>&1
for i in $(seq 1 20); do
    sleep 60
    st=$($K kernels status aefinityaiinc/a4-tpu-smoke 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z]*' || true)
    case "$st" in *COMPLETE|*ERROR|*CANCEL*) break;; esac
done
OUT=$(mktemp -d)
$K kernels output aefinityaiinc/a4-tpu-smoke -p "$OUT" >/dev/null 2>&1
if grep -q 'A4_SMOKE_JAX_PASS\|TpuDevice' "$OUT"/*.log 2>/dev/null; then
    echo "[$(ts)] TPU GRANTED — smoke results in kernel output" | tee -a "$LOG"
    cp "$OUT"/*.log "$D/tpu_granted_output.log" 2>/dev/null
    touch "$D/TPU_GRANTED"
else
    echo "[$(ts)] no TPU (downgraded again, status=$st)" >> "$LOG"
fi
rm -rf "$OUT"
