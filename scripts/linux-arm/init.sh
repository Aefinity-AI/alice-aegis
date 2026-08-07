#!/bin/sh
# /init for the minimal-Linux arm of the paired OS-cost measurement.
# Runs entirely from initramfs; mounts the boot USB's FAT partition only to
# append the gauntlet log; powers off when done.
#
# Mirrors the unikernel's evidence pattern: STAGE banners, TURBO_DIAG-style
# MSR readback, engine-printed clock ratio, everything into one BOOTLOG.

export PATH=/bin:/sbin

echo "==== AEGIS MINIMAL-LINUX ARM BOOT ===="
mount -t devtmpfs devtmpfs /dev 2>/dev/null
mount -t proc proc /proc
mount -t sysfs sysfs /sys
echo "STAGE 1: pseudo-filesystems mounted"

# --- modules (ordered list generated at build time from modules.dep) --------
while read -r m; do
    [ -n "$m" ] && insmod "/modules/$m" 2>/dev/null
done < /modules/insmod.order
echo "STAGE 2: modules loaded ($(wc -l < /modules/insmod.order) listed)"

# --- machine detection (one stick now boots two machines) -------------------
CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-)
CPU_MODEL=${CPU_MODEL# }
case "$CPU_MODEL" in
    *5200U*) AEGIS_MACHINE="Dell Inspiron 15 i5-5200U (Broadwell-U), minimal-Linux initramfs, bd-prochot cleared" ;;
    *N4020*) AEGIS_MACHINE="HP Stream Celeron N4020 (Gemini Lake), minimal-Linux initramfs" ;;
    *)       AEGIS_MACHINE="$CPU_MODEL (unrecognized), minimal-Linux initramfs" ;;
esac
export AEGIS_MACHINE
echo "MACHINE: $AEGIS_MACHINE"

# --- cmdline overrides (QEMUTEST UKI pins both to 1 so the correctness boot
# stays inside its timeout; both absent on the real-iron UKI) ----------------
BENCHREPS=
MECHN=10
for tok in $(cat /proc/cmdline); do
    case "$tok" in
        aegis_benchreps=*) BENCHREPS=${tok#aegis_benchreps=} ;;
        aegis_mechv2n=*)   MECHN=${tok#aegis_mechv2n=} ;;
    esac
done

# --- clock parity with the unikernel (STAGE 7 equivalent) -------------------
echo "TURBO_DIAG pre (MSR 0x1FC POWER_CTL):"
msrtool read 1fc || echo "  (msr read failed)"
if msrtool clearbit 1fc 0; then
    echo "STAGE 7-equiv bd-prochot: CLEARED (read-back confirms)"
else
    echo "STAGE 7-equiv bd-prochot: FAILED — arm is NOT clock-comparable"
fi
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [ -f "$g" ] && echo performance > "$g" 2>/dev/null
done
echo "governors: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo none-exposed)"
sleep 2

# --- find the boot USB FAT partition by marker file -------------------------
MNT=/mnt
mkdir -p $MNT
FOUND=
i=0
while [ $i -lt 30 ]; do
    for d in /dev/sd?? /dev/sd?; do
        [ -b "$d" ] || continue
        if mount -t vfat -o rw "$d" $MNT 2>/dev/null; then
            if [ -f "$MNT/AEGIS_LINUX_ARM.tag" ]; then FOUND=$d; break; fi
            umount $MNT
        fi
    done
    [ -n "$FOUND" ] && break
    sleep 1; i=$((i+1))
done
if [ -n "$FOUND" ]; then
    echo "STAGE 3: boot volume $FOUND mounted rw"
    LOG=$MNT/BOOTLOG_LINUX_ARM.txt
else
    echo "STAGE 3: NO BOOT VOLUME FOUND — logging to console only"
    LOG=/dev/null
fi

run() {
    # run and tee to log
    "$@" 2>&1 | tee -a "$LOG"
}

{
    echo ""
    echo "==== AEGIS MINIMAL-LINUX ARM ===="
    echo "boot_utc_unavailable_rtc: $(date -u 2>/dev/null)"
    echo "kernel : $(uname -srv)"
    echo "cmdline: $(cat /proc/cmdline)"
    echo "cpu    : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
    echo "machine: $AEGIS_MACHINE"
    echo "cpus_on: $(grep -c ^processor /proc/cpuinfo)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "cur_freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo n/a) kHz"
    echo "loadavg : $(cat /proc/loadavg)"
    msrtool read 1fc
    msrtool read 198 2>/dev/null   # IA32_PERF_STATUS: current P-state
} | tee -a "$LOG"

M=/aegis/MODEL.SAF; E=/aegis/EMBED.BIN; V=/aegis/VOCAB.BIN

echo "==== GAUNTLET (Linux arm) ====" | tee -a "$LOG"
# Arm L1: n=7 fresh-process runs, 64 new tokens, fixed prompt
# (pairs with unikernel PROMPT runs and the skeptic ARM-1 protocol)
n=1
while [ $n -le 7 ]; do
    echo "--- L1 fresh-process run $n/7" | tee -a "$LOG"
    run aegis-linux $M $E $V 64 "Once upon a time"
    n=$((n+1))
done

# Arm L2: in-process repetition (launch confound removed; pairs with the
# in-proc variance probe protocol): n=20 identical generations, one process
echo "--- L2 in-process n=20" | tee -a "$LOG"
run inproc_variance $M $E $V 64 "Once upon a time" 20

# Arm L3: the three prompts from the bare-metal 2026-07-29 session, for
# direct row-by-row pairing with m7_baremetal_prompts_postfix_2026-07-29.log
for p in "hello alice" "how are you today?" "continue"; do
    echo "--- L3 prompt: \"$p\"" | tee -a "$LOG"
    run aegis-linux $M $E $V 256 "$p"
done

echo "==== GAUNTLET DONE ====" | tee -a "$LOG"
echo "final cur_freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo n/a) kHz" | tee -a "$LOG"

# --- MECH v2 Linux arm (MECHV2L): paired OS-cost runs, console neutralized --
# stdout+stderr go to a tmpfs file DURING each run (console cost = one RAM
# write, matching the unikernel QUIET2 buffering); afterwards only the
# harness's measurement lines are appended to $LOG, plus the run-1 response
# text so answer quality stays auditable. n=10 default, aegis_mechv2n=N
# overrides (QEMUTEST pins 1).
mkdir -p /tmp
MTMP=/tmp/mechv2l.out
echo "==== MECHV2L (n=$MECHN per prompt; output buffered to tmpfs during runs) ====" | tee -a "$LOG"
for p in "hello alice" "how are you today?" "continue"; do
    i=1
    while [ $i -le $MECHN ]; do
        aegis-linux $M $E $V 256 "$p" > "$MTMP" 2>&1
        echo "MECHV2L \"$p\" run $i/$MECHN" | tee -a "$LOG"
        grep -E 'Prefill|Decode|Cycles' "$MTMP" >> "$LOG"
        if [ $i -eq 1 ]; then
            echo "MECHV2L RESPONSE \"$p\":$(grep 'Final Full Response:' "$MTMP" | cut -d: -f2-)" >> "$LOG"
        fi
        i=$((i+1))
    done
done
echo "==== MECHV2L DONE ====" | tee -a "$LOG"

# --- CIS selftest: cross-ISA identity digest --------------------------------
# Integer semantics only; the Dell (AVX2) and the HP (SSE2-class) MUST print
# the same "CIS_SELFTEST digest=..." line. That equality is the artifact.
echo "==== CIS_SELFTEST (integer semantics identity) ====" | tee -a "$LOG"
run cis_selftest

# --- witness v0 on iron: hash-chained generation transcript -----------------
# Identity/correctness artifact (chain hash), never a perf instrument.
echo "==== WITNESS v0 gen (hash-chain identity artifact) ====" | tee -a "$LOG"
run witness gen $M $E $V 32 "hello alice"

# --- membw probe (FABLE-0 gate G1): measured read bandwidth, three patterns -
# One invocation reports 1-thread sequential, 4-thread sequential, and the
# ternary weight-stream pattern. Under QEMUTEST (BENCHREPS set) the buffer is
# shrunk: TCG bandwidth numbers are meaningless (Rule A) and only completion
# is being tested there.
echo "==== MEMBW PROBE (FABLE-0 G1) ====" | tee -a "$LOG"
if [ -n "$BENCHREPS" ]; then
    run membw 64 1 2
else
    run membw 512 5 4
fi

# --- kernel-candidate A/B: fused dual/tri matvec + bitplane vs incumbent ----
# Same-binary interleaved benches; each run prints its own clock-state block
# (TSC nominal + effective/nominal ratio), so the raw log is a complete
# instrument record (Rule A corollary). 3 captures each = the minimum the
# lut_mpgemm findings deemed admissible for bimodal kernel A/Bs. On the HP
# (no AVX2) both benches print a skip message and exit cleanly.
echo "==== KERNEL A/B: fused_vs_sequential + bitplane_vs_lut (reps=${BENCHREPS:-default}) ====" | tee -a "$LOG"
n=1
while [ $n -le 3 ]; do
    echo "--- fused_vs_sequential capture $n/3" | tee -a "$LOG"
    run fused_vs_sequential $BENCHREPS
    n=$((n+1))
done
n=1
while [ $n -le 3 ]; do
    echo "--- bitplane_vs_lut capture $n/3" | tee -a "$LOG"
    run bitplane_vs_lut $BENCHREPS
    n=$((n+1))
done
# colskip vs incumbent (A15's justified candidate; ordered variant is
# byte-identical to the incumbent, so a win here is a zero-risk drop-in).
# Primary scenario = real captured BitNet-2B down_proj vectors. AVX2-only;
# prints a skip message and exits cleanly on the HP.
n=1
while [ $n -le 3 ]; do
    echo "--- colskip_vs_incumbent capture $n/3" | tee -a "$LOG"
    run colskip_vs_incumbent ${BENCHREPS:-5} /aegis/relu2_down_in.av1
    n=$((n+1))
done
# CIS-1 integer semantics cost: three arms (float/AVX2, float/scalar,
# int/scalar) interleaved in one binary. C/B is the answer — the cost of
# integer semantics with SIMD held constant; C/A is NOT, because it conflates
# scalar-vs-SIMD with int-vs-float. As of 2026-08-05 no hardware log contained
# any CIS-1 throughput figure, so both "near-zero overhead" and "6x slower"
# were unsupported. Runs on BOTH machines: on the HP (no AVX2) arm A falls back
# to the scalar path, so A/B collapses to ~1.0 and C/B still answers the
# question.
n=1
while [ $n -le 3 ]; do
    echo "--- cis_vs_float capture $n/3" | tee -a "$LOG"
    run cis_vs_float ${BENCHREPS:-5}
    n=$((n+1))
done
echo "==== KERNEL A/B DONE ====" | tee -a "$LOG"

sync
[ -n "$FOUND" ] && umount $MNT
echo "powering off in 3s"
sleep 3
poweroff -f
