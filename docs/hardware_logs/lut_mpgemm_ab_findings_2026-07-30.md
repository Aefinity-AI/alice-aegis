# LUT-mpGEMM (T-MAC pshufb) same-binary A/B — findings and derivation
2026-07-30 · dev box, crosvm guest (Chromebook), i5-10210U · git_head a249b2c

## Why this was re-measured

Ledger row A7 ("16% SLOWER") cited `aegis-core/benches/lut_mpgemm.rs` — a
source file, which Rule B does not admit. Worse, the bench NEVER MEASURED the
f32 LUT+FMA arm it claimed to lose to: its own comment (pre-2026-07-30 revision,
lines 261-265) said "this run does not measure that kernel" and gave the
baseline as "~14-16 GMAC/s on this box (and 9.86 GMAC/s in the 2026-07-09
run)" — a 60% range, typed into a comment. The bench was extended so every arm
runs in ONE binary, interleaved in one loop, on the same packed weights.

## Instrument results (3 logged runs)

| run | log | pshufb eff GMAC/s | f32 matvec | f32/pshufb | GEMM (b=8, per tok) |
|---|---|---|---|---|---|
| 1 | lut_mpgemm_sameproc_ab_2026-07-30.log      |  8.24 | 8.94 | 1.09x | 20.68 |
| 2 | lut_mpgemm_sameproc_ab_run2_2026-07-30.log | 11.29 | 9.36 | 0.83x | 23.67 |
| 3 | lut_mpgemm_sameproc_ab_run3_2026-07-30.log | 11.12 | 9.30 | 0.84x | 21.03 |

(A fourth, unlogged spot-run between 1 and 2 matched the run-2/3 mode:
11.01 vs 9.18. Recorded here as an observation only; it substantiates nothing.)

## Findings

1. **"16% SLOWER" is NOT REPRODUCIBLE.** The pshufb arm is bimodal: 8.2 vs
   ~11.2 GMAC/s effective (36% swing) while f32 moves only 8.94→9.36 (4.7%)
   and GEMM is stable. In the modal state pshufb is 1.19-1.21x FASTER than the
   f32 matvec kernel-side; in run 1 it LOST at 1.09x. The ordering flips.
   Untested hypothesis for the bimodality: the 4-bit layout's 8 MB working set
   straddles the 6 MB LLC, the 2-bit layout's 4 MB fits. Not investigated
   further because finding 2 moots it.

2. **The rejection verdict STANDS, on memory-traffic grounds, not kernel
   throughput.** Derivation (sources: baremetal_speed_findings_2026-07-29.md:113
   traffic decomposition; :339 bandwidth crossover and roof):

       2-bit decode traffic/token = MODEL.SAF 522,831,576 B + EMBED.BIN 257,310,720 B
                                  = 780,142,296 B
       4-bit layout doubles the ternary portion:
       4-bit decode traffic/token = 780,142,296 + 522,831,576 = 1,302,973,872 B
       traffic ratio              = 1,302,973,872 / 780,142,296 = 1.670x

   Decode arithmetic intensity is 5.69 FLOP/byte with the bandwidth-bound
   crossover at 1.42-1.78 GHz and a roof of 10.3-12.8 tok/s on ~780 MB/token.
   Post-A12 (throttle fixed, full clocks) the path IS in that regime, so the
   roof under the 4-bit layout falls to 10.3/1.670-12.8/1.670 = 6.2-7.7 tok/s.
   pshufb's BEST measured kernel gain is 1.21x; 1.21/1.670 = 0.72x -> the real
   decode path gets ~28% SLOWER in the bandwidth-bound regime even granting
   the kernel its best day. At throttled (compute-bound) clocks the two arms
   are within each other's instability band. There is no operating point where
   the swap wins.

3. **Two standing subordinate facts, re-confirmed same-binary:** batched GEMM
   beats pshufb-effective 1.89-2.51x for prefill; and the benchmarked pshufb
   kernel is numerically incorrect (madd pairing artifact, flagged in-source) —
   repairing it costs widening instructions, eroding the modal 20% before the
   traffic penalty is even paid.

## Disposition

A7's conclusion (REJECT on this hardware) is CONFIRMED; its stated evidence
("16% slower", unmeasured baseline) is RETRACTED and replaced by the
measurements and derivation above. The VNNI (`vpdpbusd`) revisit note in the
bench header survives untouched — none of this measured a VNNI machine.
