# MECH v2 PREREGISTRATION — paired OS-cost redo (written 2026-08-01, BEFORE the boots)

## Question

Does the ring-0 unikernel decode faster than minimal Linux on the same machine
once both arms are hardfloat, both buffer console output during timing, and
both reach the 1-core turbo bin? MECH v1.1's unpreregistered indicator said
yes (+5.4/+18.5/+24.4% on 3 prompts, n=1 each). This preregisters the paired
test with n=10.

## Design

Machine: Dell Inspiron 15 (i5-5200U, Broadwell-U). Model: M7 (14.17M), same
pinned artifacts both arms. Prompts (identical, greedy): "hello alice",
"how are you today?", "continue". Max 256 new tokens.

- **Arm U (unikernel):** hardfloat build, MECH v2 block — N=10 repeats per
  prompt AFTER AP-PARK (MWAIT-C6) with console buffered during the timed
  region (QUIET2 conditions). Log lines `MECHV2 ... run i/10: ... ticks,
  ticks/token, clock %`.
- **Arm L (minimal Linux):** same-day boot, performance governor,
  bd-PROCHOT cleared, N=10 repeats per prompt with stdout+stderr redirected
  to tmpfs during the run (console cost = RAM write, matching QUIET2
  buffering). Log lines `MECHV2L ...` + the harness Prefill/Decode lines.

## Metrics and units (declared)

- U per-run metric: total process_intent ticks / generated tokens
  (TSC 2.1975 GHz; **includes prefill amortized over generated tokens** —
  this biases AGAINST the unikernel and is accepted as conservative).
- L per-run metrics: decode-only tok/s AND total wall time as printed by the
  harness. Primary comparison uses **L decode-only** (again conservative
  against U). Secondary comparison uses L total/token.
- Per prompt: median over the 10 runs. Δ per prompt = (t_L − t_U)/t_L with
  t in seconds/token.

## Predictions

- P-V2-1: U median faster than L decode-only median on **3/3 prompts**
  (direction from v1.1 indicator).
- P-V2-2: U in-boot run-to-run spread (max/min ticks/token per prompt)
  < 3% — ring 0 with parked APs should be near-deterministic.
- P-V2-3: within-arm responses byte-identical across all 10 runs per prompt
  (greedy); U responses byte-identical to v1.1 QUIET2 responses.
- P-V2-4 (cross-ISA, if the HP boots the L stick): HP scalar-path responses
  byte-identical to Dell responses for the same prompts, and
  `CIS_SELFTEST digest` identical on both machines.

## Decision rules

- Publishable "no-OS ≥ OS on this workload" claim requires: P-V2-1 (3/3) AND
  P-V2-3 clean. 2/3 or mixed = report as mixed, no headline.
- If L wins pooled: Band-3's residual is real and unikernel-intrinsic;
  H4 investigation continues; report the negative.
- Declared residual differences that this design does NOT eliminate: code
  path (process_intent vs harness main), EOS-token count delta ±1, timing
  boundary (declared above). Any claim must carry these caveats.

## Analysis plan

Extract post-offset logs to new hardware_logs files; parse MECHV2/MECHV2L
lines; medians + spreads per prompt; score P-V2-1..4; ledger row with this
file as the prereg parent. No number leaves the logs without
scripts/verify-figures.sh.
