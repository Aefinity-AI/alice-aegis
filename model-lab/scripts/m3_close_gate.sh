#!/usr/bin/env bash
# m3_close_gate.sh — self-executing close of the M3 n=100 ARC-Easy parity gate.
# Waits for the detached engine run (PID passed as $1) to exit, then assembles
# the durable log and computes the formal verdict via mc_compare.py.
# Armed 2026-07-18 under systemd-run --user --unit=m3-gate-close because
# in-session watchers were repeatedly killed; this survives session close.
set -uo pipefail

ENGINE_PID="${1:?usage: m3_close_gate.sh <engine_pid>}"
HWLOG=/home/killboxincorporated/docs/hardware_logs/m3_mc_parity_arc_easy_2026-07-18.log
EVDIR=/home/killboxincorporated/model-lab/data/evals/arc_easy
REF_STDOUT=/tmp/claude-1000/-home-killboxincorporated/7809ff91-8a70-454f-96e2-5f1b3458f670/scratchpad/m3_ref_stdout.log
QBARS="$EVDIR/smoke3_evidence/m3_quality_bars.log"
PY=/home/killboxincorporated/ranger-venv/bin/python
COMPARE=/home/killboxincorporated/model-lab/scripts/mc_compare.py

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "[$(ts)] gate-close armed, waiting for engine PID $ENGINE_PID"
while kill -0 "$ENGINE_PID" 2>/dev/null; do sleep 60; done
sleep 10  # let the wrapper flush its rc line

# preserve the final reference stdout durably before the scratchpad vanishes
if [ -f "$REF_STDOUT" ]; then
    cp -f "$REF_STDOUT" "$EVDIR/smoke3_evidence/m3_ref_stdout.log"
    {
        echo ""
        echo "--- reference side (full stdout, appended by m3_close_gate.sh $(ts)) ---"
        cat "$REF_STDOUT"
    } >> "$HWLOG"
else
    echo "[$(ts)] WARNING: $REF_STDOUT missing; durable copy at smoke3_evidence/ is the record" >> "$HWLOG"
fi

{
    echo ""
    echo "--- quality bars (fmt/clippy, appended by m3_close_gate.sh $(ts)) ---"
    cat "$QBARS" 2>/dev/null || echo "quality bars log missing"
    echo ""
    echo "--- M3 PARITY VERDICT (mc_compare.py, run $(ts)) ---"
    echo "engine result rows: $(grep -c '^{' "$EVDIR/m3_engine_results_n100.jsonl" 2>/dev/null || echo 0)"
    echo "ref result rows:    $(grep -c '^{' "$EVDIR/m3_ref_results_n100.jsonl" 2>/dev/null || echo 0)"
} >> "$HWLOG"

"$PY" "$COMPARE" "$EVDIR/m3_engine_results_n100.jsonl" "$EVDIR/m3_ref_results_n100.jsonl" >> "$HWLOG" 2>&1
rc=$?
echo "mc_compare exit code: $rc ($([ $rc -eq 0 ] && echo PASS || echo 'FAIL or error'))" >> "$HWLOG"
echo "[$(ts)] gate closed, mc_compare rc=$rc" > "$EVDIR/M3_GATE_CLOSED.txt"
exit "$rc"
