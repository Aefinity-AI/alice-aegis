# edge-1 verify profile — 2026-09-14, box1 (aefinity-box), main @ 1da7351

Measurement only, no code changes. Repro:

```
cd aegis-linux
cargo build --release --features phase-timers --example agent_trace --target-dir target-phases-leg
cargo build --release --example agent_trace --target-dir target-release-leg

ART=~/aefinity-artifacts/bitnet2b-2b-artifacts   # facb3597.. / e32b99a2.. / 5bde1b03.. (see artifact-sha256.txt)
R=~/legs/eval-60-replay-box1/receipts/calc_easy_01.txt   # K=1, 91 forward tokens

# 1. phase-timer split (TSC-derived, clock-independent) -> phases-out.txt / time-out.txt
/usr/bin/time -v target-phases-leg/release/examples/agent_trace verify \
  $ART/aegis_pruned_model.cis.safetensors $ART/embed.bin $ART/vocab.bin "$R" --phases

# 2. full call-graph perf on the default (non-instrumented) binary -> perf-report-top.txt
sudo perf record -g --call-graph dwarf -F 499 -o perf.data -- \
  target-release-leg/release/examples/agent_trace verify \
  $ART/aegis_pruned_model.cis.safetensors $ART/embed.bin $ART/vocab.bin "$R"
perf report -i perf.data -f --stdio --no-children --percent-limit 0.5
```

**Caveat (bd_prochot):** `bdprochot-status.txt` shows the CPU was clamped to
`cur_ratio=5` (500 MHz) against `req_ratio=27` (2700 MHz) for this run — the
known box1 BD-PROCHOT MSR-clamp bug from the 09-12 report. Attempting the
usual fix (`cm-bdprochot.sh fix`, clears MSR 0x1FC bit 0) failed with
`PermissionError: Operation not permitted` even under `sudo` in this session
(no raw MSR write capability available). Consequence: **absolute wall-clock
numbers here (79 s single-receipt verify) are ~4x inflated vs the 09-12
report's full-clock ~21 s and are not comparable to it.** The percentage
breakdown between hashing/matmul/other is NOT affected by a uniform clock
scalar (TSC runs at a fixed nominal rate independent of core P-state, and the
`perf` cycle-sampling shares are ratios of the same clamped run to itself),
so the percentages below are trustworthy; the seconds are not. Load1 was
0.34-0.6 throughout (quiet), nothing else running.

See `../../../../state/reports/2026-09-14-edge1-verify-profile.md` (in the
claudius-maximus repo) for the full writeup.
