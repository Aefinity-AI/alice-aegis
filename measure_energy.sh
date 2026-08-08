#!/bin/bash
# Measured joules-per-token for A.L.I.C.E. — battery-discharge method.
#
# PROCEDURE (must be run ON BATTERY — unplug AC first):
#   ./measure_energy.sh
#
# Method:
#   1. Sample battery discharge power for a long idle baseline.
#   2. Start a generation run. Wait until the harness prints its prefill banner
#      (model load finished), then sample power only while tokens are being
#      generated.
#   3. Incremental inference power = P_load - P_idle.
#      J/token = incremental watts / (tokens / generation-seconds).
#
# We attribute only the MARGINAL cost of inference: screen, chipset, and OS
# idle draw are subtracted out. Report it that way — also print the total
# system figure so the reader can see both.
#
# SENSOR CAVEAT (measured on this Chromebook, 2026-07-09): `current_now` is
# LATCHED — it holds a value for 15-30 s and lags load changes by several
# seconds. `charge_counter` moves in coarse 40 mAh steps. Neither can resolve a
# 60-second window. Both windows below are therefore long enough that the
# latch updates many times, and the coulomb counter is reported as an
# independent cross-check. If the two disagree badly, trust neither.
#
# Also: CLOSE OTHER WORKLOADS. A single busy core adds ~13 W here, which is
# larger than the entire inference signal.
set -u

IDLE_S=${IDLE_S:-150}   # idle baseline seconds
MIN_LOAD_S=180          # warn if the decode window is shorter than this

# Locate the battery node (varies: "battery" on Crostini, BAT0/BAT1 on laptops).
BAT=""
for cand in /sys/class/power_supply/*; do
    [ -f "$cand/type" ] || continue
    [ "$(cat "$cand/type")" = "Battery" ] || continue
    BAT="$cand"; break
done
[ -n "$BAT" ] || { echo "ERROR: no battery node found under /sys/class/power_supply."; exit 1; }
echo "Battery node: $BAT"

status=$(cat "$BAT/status" 2>/dev/null || echo unknown)
if [ "$status" != "Discharging" ]; then
    echo "ERROR: battery status is '$status'. Unplug AC power and re-run."
    echo "       (Power counters read zero while charging or on AC.)"
    exit 1
fi

# Prefer power_now (µW) when the platform provides it; else current×voltage.
read_watts() {
    if [ -f "$BAT/power_now" ] && [ "$(cat "$BAT/power_now")" != "0" ]; then
        awk '{printf "%.3f", ($1<0?-$1:$1)/1e6}' "$BAT/power_now"
    else
        local ua uv
        ua=$(cat "$BAT/current_now" 2>/dev/null || echo 0)
        uv=$(cat "$BAT/voltage_now" 2>/dev/null || echo 0)
        echo "$ua $uv" | awk '{printf "%.3f", ($1<0?-$1:$1)*$2/1e12}'
    fi
}

sample_avg() { # $1 = seconds
    local n=$1 sum=0 w
    for _ in $(seq "$n"); do
        w=$(read_watts)
        sum=$(echo "$sum $w" | awk '{print $1+$2}')
        sleep 1
    done
    echo "$sum $n" | awk '{printf "%.3f", $1/$2}'
}

LOG=/tmp/energy_run.log
MODEL=~/aegis_pruned_model.safetensors
EMBED=~/aegis-forge/embed.bin
VOCAB=~/aegis-forge/vocab.bin
NTOK=${NTOK:-1024}

# Refuse to measure while something else is eating a core.
BUSY=$(ps -eo pcpu,comm --sort=-pcpu | awk 'NR==2 && $1>25 {print $2" ("$1"%)"}')
if [ -n "$BUSY" ]; then
    echo "ERROR: $BUSY is using CPU. Its draw dwarfs the inference signal."
    echo "       Stop it and re-run."
    exit 1
fi

cc()   { cat "$BAT/charge_counter" 2>/dev/null || echo 0; }
volt() { awk '{printf "%.4f", $1/1e6}' "$BAT/voltage_now"; }

echo "[1/3] Idle baseline (${IDLE_S}s — do not touch the machine)..."
CC0=$(cc); T0=$(date +%s)
P_IDLE=$(sample_avg "$IDLE_S")
CC1=$(cc); T1=$(date +%s)
V=$(volt)
IDLE_COULOMB=$(echo "$CC0 $CC1 $T1 $T0 $V" | awk '{d=($1-$2)/1e6; h=($3-$4)/3600; if(h>0&&d>0) printf "%.2f", d*$5/h; else printf "n/a"}')
echo "      idle: ${P_IDLE} W  (coulomb cross-check: ${IDLE_COULOMB} W)"

echo "[2/3] Generation run (${NTOK} tokens)..."
: > "$LOG"
~/aegis-linux/target/release/aegis-linux "$MODEL" "$EMBED" "$VOCAB" \
    "$NTOK" "Write a detailed essay about the history of computing." > "$LOG" 2>&1 &
RUN_PID=$!

# Wait for model load to finish — the engine prints "[SYSTEM] Analyzing" when
# prefill starts. Don't guess with a fixed sleep: load time scales with disk.
while kill -0 $RUN_PID 2>/dev/null && ! grep -q "Analyzing" "$LOG"; do sleep 1; done

CC2=$(cc); T2=$(date +%s)
SAMPLES=0; SUM=0
while kill -0 $RUN_PID 2>/dev/null; do
    w=$(read_watts)
    SUM=$(echo "$SUM $w" | awk '{print $1+$2}')
    SAMPLES=$((SAMPLES+1))
    sleep 1
done
wait $RUN_PID 2>/dev/null
CC3=$(cc); T3=$(date +%s)

if [ "$SAMPLES" -lt 5 ]; then
    echo "ERROR: only ${SAMPLES}s of generation sampled — too short to attribute energy."
    echo "       Increase NTOK, or the run failed. See $LOG"
    exit 1
fi
if [ "$SAMPLES" -lt "$MIN_LOAD_S" ]; then
    echo "      WARNING: decode window ${SAMPLES}s < ${MIN_LOAD_S}s; the latched"
    echo "               current sensor may not have settled. Raise NTOK."
fi
P_LOAD=$(echo "$SUM $SAMPLES" | awk '{printf "%.3f", $1/$2}')
LOAD_COULOMB=$(echo "$CC2 $CC3 $T3 $T2 $V" | awk '{d=($1-$2)/1e6; h=($3-$4)/3600; if(h>0&&d>0) printf "%.2f", d*$5/h; else printf "n/a"}')
echo "      load: ${P_LOAD} W over ${SAMPLES}s  (coulomb cross-check: ${LOAD_COULOMB} W)"

# Real token count from the harness, not an assumption.
TOKENS=$(grep -oE 'Generated [0-9]+ tokens' "$LOG" | grep -oE '[0-9]+' | tail -1)
if [ -z "${TOKENS:-}" ] || [ "$TOKENS" -eq 0 ]; then
    # Fallback: if the engine hit the 2048-token context limit and panicked
    # (inference.rs:332 off-by-one, observed 2026-07-14) it never prints the
    # summary line, but the count is still determinate: 2048 - prompt tokens.
    PROMPT_TOK=$(grep -m1 -oE 'Analyzing [0-9]+ tokens' "$LOG" | grep -oE '[0-9]+')
    if grep -q 'index out of bounds.*2048' "$LOG" && [ -n "${PROMPT_TOK:-}" ]; then
        TOKENS=$((2048 - PROMPT_TOK))
        echo "      WARNING: engine panicked at the context limit; token count"
        echo "               derived as 2048 - ${PROMPT_TOK} prompt = ${TOKENS}."
        echo "               Keep prompt+NTOK below 2040 to avoid this."
    else
        echo "ERROR: could not read generated token count from $LOG"
        exit 1
    fi
fi

echo "      load: ${P_LOAD} W averaged over ${SAMPLES}s of generation"
echo "      tokens generated: ${TOKENS}"
echo "[3/3] Results"
echo "$P_IDLE $P_LOAD $SAMPLES $TOKENS" | awk '{
    idle=$1; load=$2; gsec=$3; tok=$4;
    dw = load-idle; tps = tok/gsec;
    printf "  idle power:            %.2f W\n", idle;
    printf "  power during decode:   %.2f W\n", load;
    printf "  incremental power:     %.2f W\n", dw;
    printf "  decode rate:           %.2f tok/s (generation window only)\n", tps;
    printf "\n";
    printf "  ENERGY/TOKEN (incremental, inference only): %.3f J\n", dw/tps;
    printf "  ENERGY/TOKEN (total system draw):           %.3f J\n", load/tps;
}'
echo
echo "Report the incremental figure with the caveat that it excludes idle"
echo "system draw; report both when comparing against published numbers."
echo "Log: $LOG"
