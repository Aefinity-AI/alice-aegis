#!/bin/bash
# amdahl-decode leg: per-phase Amdahl decomposition of greedy decode
# (aegis-linux/examples/amdahl_decode.rs, aegis-core/src/phase_timers.rs)
# run as a box leg.
#
# QUIET GUARD FIRST, ALWAYS. This box (iron / aefinity-box, i5-5200U 2c/4t
# AVX2+FMA) is a NAMED PHYSICAL BOX running nothing else when this leg is
# allowed to proceed — Rule A (alice-aegis/CLAUDE.md) permits timing numbers
# from exactly this situation. Every guard failure below writes its reason
# to RESULT.txt and exits 0 WITHOUT measuring; it never stops or restarts
# anything else on the box.
set -uo pipefail

NAME=amdahl-decode
OUT="${LEG_OUT:-$HOME/legs/$NAME}"
RESULT="$OUT/RESULT.txt"
RAW="$OUT/raw"
mkdir -p "$OUT" "$RAW"

log() { echo "$1" | tee -a "$RESULT"; }
refuse() {
    log "REFUSE: $1"
    log "=== $NAME done $(date -u +%Y-%m-%dT%H:%M:%SZ) — refused, no measurement taken"
    exit 0
}
fail() {
    log "FAIL: $1"
    log "=== $NAME done $(date -u +%Y-%m-%dT%H:%M:%SZ) — FAILED"
    exit 1
}

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------------------
# QUIET GUARD (order matters: cheapest/most-decisive checks first; nothing
# above this point runs any measurement, build, or state-changing command).
# ---------------------------------------------------------------------------

# 0. Not a VM. penguin (crosvm) and any other virtualized host must refuse —
#    Rule A: timing numbers are legal ONLY on a named physical box.
VIRT_REASON=""
PRODUCT_NAME="$(cat /sys/class/dmi/id/product_name 2>/dev/null || true)"
case "$PRODUCT_NAME" in
    *crosvm*|*QEMU*|*KVM*|*VirtualBox*|*VMware*) VIRT_REASON="dmi product_name='$PRODUCT_NAME'" ;;
esac
if [ -z "$VIRT_REASON" ] && command -v lscpu >/dev/null 2>&1; then
    HV="$(lscpu 2>/dev/null | awk -F: '/[Hh]ypervisor vendor/{gsub(/^[ \t]+/,"",$2); print $2}')"
    [ -n "$HV" ] && VIRT_REASON="lscpu Hypervisor vendor='$HV'"
fi
if [ -z "$VIRT_REASON" ] && grep -qw hypervisor /proc/cpuinfo 2>/dev/null; then
    VIRT_REASON="/proc/cpuinfo flags contain 'hypervisor'"
fi
if [ -n "$VIRT_REASON" ]; then
    log "=== $NAME start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) commit=novirt"
    refuse "running under a VM ($VIRT_REASON) — Rule A: no timing number may originate from a hypervisor guest"
fi

# From here on the box is (by the checks above) physical, so it's safe to
# name it in the start line before continuing the guard.
COMMIT_PROBE="$(git -C "${ALICE_AEGIS_MAIN:-$HOME/projects/alice-aegis}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "=== $NAME start $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) commit=$COMMIT_PROBE"
log "# host is a named physical box (no hypervisor signature found) — timing numbers below are Rule-A legal if every remaining guard also clears"

# 1. No other leg-* unit active (besides this one's own unit, leg-amdahl-decode).
if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    OTHER_LEGS="$(systemctl --user list-units 'leg-*' --state=active --no-legend 2>/dev/null | awk '{print $1}' | grep -v "^leg-${NAME}\.\(scope\|service\)\$" || true)"
    if [ -n "$OTHER_LEGS" ]; then
        refuse "another leg-* unit is active: $(echo "$OTHER_LEGS" | tr '\n' ' ')"
    fi
fi

# 2. 1-min load average > 0.5.
LOAD1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"
if awk -v l="$LOAD1" 'BEGIN{exit !(l>0.5)}'; then
    refuse "1-min load average $LOAD1 > 0.5"
fi

# 3. cm-tick.service not running.
if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
    if systemctl --user is-active --quiet cm-tick.service 2>/dev/null; then
        refuse "cm-tick.service is active"
    fi
fi

# 4. CPU must have avx2 (this leg targets aefinity-box, i5-5200U AVX2+FMA).
if ! grep -qw avx2 /proc/cpuinfo 2>/dev/null; then
    refuse "CPU has no avx2 flag ($(hostname) is not an AVX2 box)"
fi

log "# quiet guard PASSED: no other leg-* unit, load1=$LOAD1 <= 0.5, cm-tick.service not active, avx2 present, not a VM"

# ---------------------------------------------------------------------------
# Machine identity (captured once the guard has passed, before any build).
# ---------------------------------------------------------------------------
IDENT="$RAW/machine_identity.txt"
{
    echo "hostname: $(hostname)"
    echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "lscpu:"
    lscpu 2>/dev/null | sed 's/^/  /'
    echo "kernel: $(uname -r)"
    AVX2_COUNT="$(grep -c avx2 /proc/cpuinfo || true)"; AVX2_COUNT="${AVX2_COUNT:-0}"
    FMA_COUNT="$(grep -c '\bfma\b' /proc/cpuinfo || true)"; FMA_COUNT="${FMA_COUNT:-0}"
    echo "avx2 flag occurrences (per-logical-cpu /proc/cpuinfo lines): $AVX2_COUNT"
    echo "fma  flag occurrences (per-logical-cpu /proc/cpuinfo lines): $FMA_COUNT"
    echo "governor(s):"
    for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        [ -r "$f" ] && echo "  $f = $(cat "$f")"
    done
    echo "AC power:"
    ac_found=0
    for f in /sys/class/power_supply/*/online /sys/class/power_supply/*/type; do
        [ -r "$f" ] || continue
        ac_found=1
        echo "  $f = $(cat "$f")"
    done
    [ "$ac_found" = 0 ] && echo "  (no /sys/class/power_supply nodes — desktop or not exposed)"
    echo "cgroup cpu.max (this leg's own cgroup, if readable):"
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        echo "  /sys/fs/cgroup/cpu.max = $(cat /sys/fs/cgroup/cpu.max)"
    elif [ -r "/proc/self/cgroup" ]; then
        CG_PATH="$(awk -F: '$1==0{print $3}' /proc/self/cgroup 2>/dev/null)"
        if [ -n "$CG_PATH" ] && [ -r "/sys/fs/cgroup${CG_PATH}/cpu.max" ]; then
            echo "  /sys/fs/cgroup${CG_PATH}/cpu.max = $(cat "/sys/fs/cgroup${CG_PATH}/cpu.max")"
        else
            echo "  (cpu.max not readable — CPUQuota may still be applied by systemd; check the unit)"
        fi
    else
        echo "  (no cgroup v2 cpu.max node found)"
    fi
    echo "thermal BEFORE:"
    tb=0
    for f in /sys/class/thermal/thermal_zone*/temp; do
        [ -r "$f" ] || continue
        tb=1
        echo "  $f = $(cat "$f")"
    done
    [ "$tb" = 0 ] && echo "  (no /sys/class/thermal nodes)"
} > "$IDENT"
log "machine identity: $IDENT"

# ---------------------------------------------------------------------------
# Build (own worktree, own target dir — the main checkout at
# $HOME/projects/alice-aegis is NEVER branch-switched or built into).
# ---------------------------------------------------------------------------
MAIN_REPO="${ALICE_AEGIS_MAIN:-$HOME/projects/alice-aegis}"
WT="$HOME/legs-worktrees/alice-aegis-$NAME"
BRANCH="${AMDAHL_BRANCH:-cm/amdahl-phase-counters}"

if [ ! -d "$MAIN_REPO/.git" ]; then
    refuse "no alice-aegis checkout at $MAIN_REPO — cannot build"
fi

log "-- git fetch (main checkout, read-only fetch; no branch switch) + worktree sync to origin/$BRANCH"
git -C "$MAIN_REPO" fetch origin "$BRANCH" >>"$RAW/git.log" 2>&1
if [ -d "$WT/.git" ] || [ -f "$WT/.git" ]; then
    git -C "$WT" fetch origin "$BRANCH" >>"$RAW/git.log" 2>&1
    git -C "$WT" reset --hard "origin/$BRANCH" >>"$RAW/git.log" 2>&1
else
    mkdir -p "$(dirname "$WT")"
    git -C "$MAIN_REPO" worktree add "$WT" "origin/$BRANCH" --detach >>"$RAW/git.log" 2>&1
fi
if [ ! -d "$WT" ]; then
    fail "worktree $WT not created — see $RAW/git.log"
fi
WT_COMMIT="$(git -C "$WT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
log "worktree: $WT commit=$WT_COMMIT (detached from origin/$BRANCH; main checkout untouched)"

log "-- build amdahl_decode: CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --features parallel,phase-timers --example amdahl_decode"
( cd "$WT/aegis-linux" && CARGO_BUILD_JOBS=2 nice -n 10 cargo build --release --features parallel,phase-timers --example amdahl_decode ) >>"$RAW/build_amdahl_decode.log" 2>&1
BIN="$WT/aegis-linux/target/release/examples/amdahl_decode"
if [ ! -x "$BIN" ]; then
    fail "amdahl_decode build failed — see $RAW/build_amdahl_decode.log"
fi
log "built: $BIN"

# ---------------------------------------------------------------------------
# Artifacts (never downloaded by this leg — must already be present in the
# worktree, forged from the same source as every other timing leg on this
# repo).
# ---------------------------------------------------------------------------
ASSETS="${AMDAHL_ASSETS:-$WT/model-lab/tinybit/m7_final_gate_work/artifacts}"
for f in "$ASSETS/MODEL.SAF" "$ASSETS/EMBED.BIN" "$ASSETS/VOCAB.BIN"; do
    [ -f "$f" ] || fail "missing artifact: $f (this leg never downloads artifacts)"
done
log "artifacts: $ASSETS"

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
RUN_TS="$(date -u +%Y%m%d_%H%M%SZ)"
RUN_LOG="$RAW/amdahl_decode_$RUN_TS.log"
log "-- run: $BIN $ASSETS/MODEL.SAF $ASSETS/EMBED.BIN $ASSETS/VOCAB.BIN"
"$BIN" "$ASSETS/MODEL.SAF" "$ASSETS/EMBED.BIN" "$ASSETS/VOCAB.BIN" > "$RUN_LOG" 2>&1
RUN_RC=$?
cat "$RUN_LOG" >> "$RAW/amdahl_decode_stdout_combined.log"
if [ $RUN_RC -ne 0 ]; then
    log "amdahl_decode: exited $RUN_RC — raw log: $RUN_LOG (treat as FAILED, not a measurement)"
    log "=== $NAME done $(date -u +%Y-%m-%dT%H:%M:%SZ) — FAILED"
    exit 1
fi

log "-- AMDAHL lines (ticks are invariant-TSC ticks, NOT core cycles — see the NOTE line the binary itself prints):"
grep -E '^(#|AMDAHL) ' "$RUN_LOG" | while IFS= read -r line; do
    log "$line  raw_log=$RUN_LOG"
done

log "# NOTE: every ticks/total_ticks/overhead_ticks figure above comes from amdahl_decode's own"
log "# RDTSC/RDTSCP counters (aegis-core/src/phase_timers.rs) and its own runtime tsc_hz calibration"
log "# against CLOCK_MONOTONIC — never a fixed frequency assumption."
log "# CAVEAT: CPUQuota may have been applied to this leg's systemd unit/scope (see cgroup cpu.max in"
log "# $IDENT); a throttled quota changes wall-clock pacing but not the TSC tick counts themselves,"
log "# since the TSC is invariant and keeps advancing at its nominal rate regardless of throttling —"
log "# still, report the cpu.max value alongside these numbers so a reader can judge scheduling noise."

# Thermal AFTER, for the identity file.
{
    echo "thermal AFTER:"
    ta=0
    for f in /sys/class/thermal/thermal_zone*/temp; do
        [ -r "$f" ] || continue
        ta=1
        echo "  $f = $(cat "$f")"
    done
    [ "$ta" = 0 ] && echo "  (no /sys/class/thermal nodes)"
} >> "$IDENT"
log "thermal after appended to: $IDENT"

log "=== $NAME done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
