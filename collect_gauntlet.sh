#!/bin/bash
# Harvest a /gauntlet run into the fleet dataset, with the ratios that ARE the
# experiment. Read-only against the stick.
#
#   ./collect_gauntlet.sh [device] [nickname]
set -u
DEV="${1:-/dev/sda}"
NICK="${2:-$(date +%H%M%S)}"
OUT=~/docs/hardware_logs
DATA="$OUT/gauntlet_dataset.tsv"
mkdir -p "$OUT"

mdir -i "$DEV" :: >/dev/null 2>&1 || { echo "ERROR: $DEV unreadable."; exit 1; }
RAW="$OUT/gauntlet_${NICK}_$(date +%Y-%m-%d_%H%M%S).txt"
mtype -i "$DEV" ::/BOOTLOG.TXT > "$RAW" 2>/dev/null
[ -s "$RAW" ] || { echo "ERROR: no BOOTLOG.TXT on $DEV"; rm -f "$RAW"; exit 1; }
grep -q "GAUNTLET DONE" "$RAW" || { echo "⚠ no completed /gauntlet in this log:"; grep -a GAUNTLET "$RAW"; exit 1; }

echo "════════════════ $NICK ════════════════"
grep -a "GAUNTLET" "$RAW"
echo "═══════════════════════════════════════"

python3 - "$RAW" "$NICK" "$DATA" <<'PY'
import re, sys, os
raw, nick, data = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(raw, errors="replace").read()
# BOOTLOG.TXT accumulates across boots (crash forensics). Parse only the last
# gauntlet block, or an old run's lines shadow the new ones.
txt = txt.split("==== GAUNTLET ====")[-1]

def tps(label):
    # decode-only tok/s (prefill excluded; see gbench in aegis-uefi/src/main.rs)
    m = re.search(rf"GAUNTLET {re.escape(label)}:.*?(\d+)\.(\d+) tok/s", txt)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None
def dticks(label):
    # decode TSC ticks/token — the high-precision basis for ALL speed ratios.
    # tok/s is printed at 2 decimals; deriving ratios from it injects up to
    # ~5% rounding error (audit 2026-07-14: published 5.08x vs tick-true 4.84x).
    m = re.search(rf"GAUNTLET {re.escape(label)}:.*?decode (\d+) tok (\d+) ticks/tok", txt)
    return int(m.group(2)) if m else None
def pfx_ticks(label):
    # prefill TSC ticks — high-precision basis for the batching ratio
    m = re.search(rf"GAUNTLET {re.escape(label)}:.*?prefill (\d+) tok (\d+)t", txt)
    return int(m.group(2)) if m else None
def clk(label):
    m = re.search(rf"GAUNTLET {re.escape(label)}:.*?clock (\d+)%", txt)
    return int(m.group(1)) if m else None

cpu = re.search(r"GAUNTLET CPU: (.*?) \|", txt)
cpu = cpu.group(1).strip() if cpu else "unknown"
simd = re.search(r"simd=([^ ]+(?: \([^)]*\))?)", txt)
simd = simd.group(1) if simd else "?"

seg = {k: tps(k) for k in [
    "SIMD_scalar","SIMD_native",
    "PSTATE_run1","PSTATE_run2_control","PSTATE_run3_turbo",
    "CTX_20","CTX_100","CTX_400"]}
seg["PREFILL_pertoken_ticks"] = pfx_ticks("PREFILL_pertoken")
seg["PREFILL_batched_ticks"]  = pfx_ticks("PREFILL_batched")
def ratio(a,b):
    return round(seg[a]/seg[b],3) if seg.get(a) and seg.get(b) and seg[b] else None
def tick_ratio(fast_label, slow_label):
    # speed ratio from decode ticks/token (fewer ticks = faster): slow/fast
    a, b = dticks(slow_label), dticks(fast_label)
    return round(a/b, 3) if a and b else None

# All speed ratios derive from decode ticks/token (fall back to tok/s only if
# the log predates tick output). r_batch already uses prefill ticks.
r_simd  = tick_ratio("SIMD_native","SIMD_scalar")   or ratio("SIMD_native","SIMD_scalar")
r_batch = ratio("PREFILL_pertoken_ticks","PREFILL_batched_ticks")  # ticks: >1 = batching wins
drift   = tick_ratio("PSTATE_run2_control","PSTATE_run1") or ratio("PSTATE_run2_control","PSTATE_run1")
r_turbo = tick_ratio("PSTATE_run3_turbo","PSTATE_run2_control") or ratio("PSTATE_run3_turbo","PSTATE_run2_control")
r_ctx   = tick_ratio("CTX_400","CTX_20")            or ratio("CTX_400","CTX_20")

print(f"CPU  : {cpu}")
print(f"SIMD : {simd}")
print()
print(f"  SIMD value (native/scalar)   : {r_simd}x   {'(=1 on pre-AVX2 chips)' if r_simd and r_simd<1.1 else ''}")
print(f"  Batching value (pt/batched ticks): {r_batch}x")
print(f"  P-state drift (run2/run1)    : {drift}x  <- noise floor")
print(f"  Turbo effect (run3/run2)     : {r_turbo}x {'*** real ***' if r_turbo and drift and (r_turbo-1)>3*abs(drift-1) and r_turbo>1.15 else '(within drift)'}")
print(f"  clock under load: run1={clk('PSTATE_run1')}%  run3={clk('PSTATE_run3_turbo')}%")
print(f"  Context slope (400tok/20tok) : {r_ctx}x")

hdr = "nickname\tcpu\tsimd\t" + "\t".join(seg.keys()) + "\tr_simd\tr_batch\tdrift\tr_turbo\tr_ctx\n"
if not os.path.exists(data): open(data,"w").write(hdr)
row = [nick, cpu, simd] + [str(seg[k]) for k in seg] + [str(r_simd),str(r_batch),str(drift),str(r_turbo),str(r_ctx)]
open(data,"a").write("\t".join(row)+"\n")
print(f"\nappended to {data}  ({sum(1 for _ in open(data))-1} machines)")
PY
