#!/bin/bash
# Harvest an /autotest run from a boot stick into the growing dataset.
#
#   ./collect_autotest.sh [device] [nickname]
#
# Read-only against the stick. Appends one row per machine to
# docs/hardware_logs/pstate_dataset.tsv and files the raw log verbatim.
#
# The analysis it prints is the whole point of the experiment:
#
#   drift  = RUN2 / RUN1   (nothing changed between them; this is the noise floor)
#   effect = RUN3 / RUN2   (only the P-state changed)
#
# An effect that does not clearly exceed the drift is not an effect.
set -u
DEV="${1:-/dev/sda}"
NICK="${2:-$(date +%H%M%S)}"
OUT=~/docs/hardware_logs
DATA="$OUT/pstate_dataset.tsv"
mkdir -p "$OUT"

mdir -i "$DEV" :: >/dev/null 2>&1 || { echo "ERROR: $DEV unreadable. Stick inserted?"; exit 1; }

RAW="$OUT/autotest_${NICK}_$(date +%Y-%m-%d_%H%M%S).txt"
mtype -i "$DEV" ::/BOOTLOG.TXT > "$RAW" 2>/dev/null
[ -s "$RAW" ] || { echo "ERROR: no BOOTLOG.TXT on $DEV"; rm -f "$RAW"; exit 1; }

grep -q "AUTOTEST DONE" "$RAW" || {
    echo "⚠ This log has no completed /autotest. Contents:"; cat "$RAW"; exit 1
}

echo "════════════════ $NICK ════════════════"
grep -a "AUTOTEST" "$RAW"
echo "═══════════════════════════════════════"
echo "raw: $RAW"
echo

python3 - "$RAW" "$NICK" "$DATA" <<'PY'
import re, sys, os
raw, nick, data = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(raw, errors="replace").read()

def tps(tag):
    m = re.search(rf"AUTOTEST {tag}:.*?([\d.]+)\.(\d+) tok/s", txt)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None

def clk(tag):
    m = re.search(rf"AUTOTEST {tag}:.*?clock (\d+)%", txt)
    return int(m.group(1)) if m else None

cpu = re.search(r"AUTOTEST CPU: (.*?) \|", txt)
cpu = cpu.group(1).strip() if cpu else "unknown"
hwp = "hwp=true" in txt
bare = "baremetal=true" in txt
idle = re.search(r"idle_clock=(\d+)%", txt)
idle = int(idle.group(1)) if idle else None
turbo_line = re.search(r"AUTOTEST TURBO[^\n]*", txt)
turbo = turbo_line.group(0).replace("AUTOTEST TURBO", "").strip(": ") if turbo_line else "?"

r1, r2, r3 = tps("RUN1_baseline"), tps("RUN2_control"), tps("RUN3_turbo")
c1, c2, c3 = clk("RUN1_baseline"), clk("RUN2_control"), clk("RUN3_turbo")

print(f"CPU        : {cpu}")
print(f"bare metal : {bare}   HWP: {hwp}")
print(f"idle clock : {idle if idle is not None else '?'}% of nominal")
print(f"turbo      : {turbo}")
print()
if None in (r1, r2, r3):
    print("⚠ incomplete run — cannot analyse"); sys.exit(0)

drift  = r2 / r1
effect = r3 / r2
print(f"RUN1 baseline : {r1:5.2f} tok/s   clock {c1 if c1 is not None else '?'}%")
print(f"RUN2 control  : {r2:5.2f} tok/s   clock {c2 if c2 is not None else '?'}%   drift  = {drift:.3f}x")
print(f"RUN3 turbo    : {r3:5.2f} tok/s   clock {c3 if c3 is not None else '?'}%   effect = {effect:.3f}x")
print()

noise = abs(drift - 1.0)
gain  = effect - 1.0
if gain > max(3 * noise, 0.15):
    verdict = "H1 SUPPORTED — raising the P-state materially speeds up inference"
elif abs(gain) <= max(2 * noise, 0.10):
    verdict = "H2 SUPPORTED — no headroom; the clock was already where it needed to be"
else:
    verdict = "AMBIGUOUS — effect comparable to drift; repeat the run"
print(f"  {verdict}")
if c1 is not None:
    print(f"  (clock under load before turbo: {c1}% of nominal — H1 predicts <60%, H2 predicts ~100%)")

hdr = "nickname\tcpu\tbaremetal\thwp\tidle_clock\trun1_tps\trun2_tps\trun3_tps\tclk1\tclk2\tclk3\tdrift\teffect\tturbo\n"
if not os.path.exists(data):
    open(data, "w").write(hdr)
with open(data, "a") as f:
    f.write(f"{nick}\t{cpu}\t{bare}\t{hwp}\t{idle}\t{r1}\t{r2}\t{r3}\t{c1}\t{c2}\t{c3}\t{drift:.4f}\t{effect:.4f}\t{turbo}\n")
print(f"\nappended to {data}  ({sum(1 for _ in open(data))-1} machines so far)")
PY
