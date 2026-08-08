# PAIRED OS-COST MEASUREMENT — ANALYSIS RECORD, 2026-07-31

Derivation from instrument logs (Rule B legal parent 2). Protocol and bands
were locked in advance: `oscost_PREREGISTRATION_2026-07-30.md`. Analyzer:
`scripts/linux-arm/analyze_oscost.py`, unmodified since 2026-07-30 12:57.

## Evidence

| file | md5 | content |
|---|---|---|
| oscost_U_BOOTLOG_2026-07-31.txt | 6906af8068ee8acf232787e7646666fe | full ALICE_M7 BOOTLOG.TXT, verbatim |
| oscost_L_BOOTLOG_2026-07-31.txt | 17c802798759f26990b456fd5cfd0959 | full BOOTLOG_LINUX_ARM.txt, verbatim |

Append-only integrity verified before analysis: first 14,429 B of the U log
md5 b302dc442e09197503a5ea8545523bcd (== the pre-test file, byte-identical to
archived m7_baremetal_prompts_postfix_2026-07-29.log); first 16,197 B of the
L log md5 6d0a4ac7df469fcc67e08d7521ca4315 (== the shakedown snapshot taken
before the protocol runs). Analysis input = bytes AFTER those offsets
(protocol content only). Machine: Dell Inspiron 15, i5-5200U, Broadwell-U.
Model artifacts on both arms md5-verified against the prereg pins before the
runs (MODEL.SAF 53235f59…, EMBED.BIN 29731589…, VOCAB.BIN 03301400…).

## Result (locked estimand)

6 boots per arm (exceeds N=5 minimum). r = decode ticks/token (invariant-TSC,
i.e. a WALL-TIME measure at the 2.2 GHz nominal tick rate).

Per-prompt median (L−U)/U, bootstrap 95% CI over boot pairs (10,000 resamples):

| prompt | tokens | median Δ | CI |
|---|---|---|---|
| "hello alice" | 85 | −9.616% | [−10.351%, −2.011%] |
| "how are you today?" | 257 | −39.326% | [−45.397%, −20.557%] |
| "continue" | 214 | −49.256% | [−52.816%, −43.072%] |

**Pooled Δ_OS: median −39.313%, CI [−45.397%, −9.972%], n=18.**
**Band 3 (large) with NEGATIVE sign: minimal Linux is FASTER than the no-OS
unikernel.** Per the prereg: the sign is reported as-is, and a Band-3 result
is NOT publishable until the mechanism is understood. This record therefore
states a measurement, not a conclusion about "what an OS costs".

## Gates

- **Output identity (bit-exactness): PASS.** Each of the three prompts
  produced exactly one response variant per arm, identical across arms
  (85/257/214 tokens). Work-identity holds at the token level.
- **Clock parity: PARTIALLY EVALUABLE — disclosed instrumentation gap.**
  U printed `clock 113% of nominal` (~2.49 GHz) on every response of every
  boot (TURBO_DIAG: req_ratio=27 granted cur_ratio=25). L logs only an idle
  banner freq and one post-gauntlet sample; those samples ranged 2.10–2.69
  GHz across boots (Linux cpufreq reached the full 2.7 GHz 1-core turbo in 3
  of 6 boots). Strict §4 per-pair exclusion is not evaluable from these
  samples. Two mitigations: (1) ticks are invariant-TSC wall time, so the
  headline is a wall-time result independent of core clock; (2) even the
  slowest-clocked L boot (≈2.10 GHz, 84% of U's effective clock) beat U on
  the long prompts (8.28M vs ≈12.5M ticks/token) — clock cannot flip the sign.

## Protocol deviations (disclosed)

1. **Blocks, not interleaved.** The prereg locked U,L,U,L…; the L RTC stamps
   (00:37:59→00:47:56 UTC, six boots ~2 min apart) show the L boots ran
   back-to-back, so the arms were run as two blocks. The thermal/time-drift
   control is therefore lost. Mitigation observed in-data: U is internally
   stable across its block (e.g. "hello alice" 6.59–6.82M ticks/token over 6
   boots), so uncontrolled drift within blocks is small relative to the effect.
2. **U boot 4 anomaly.** Its banner lacks GAUNTLET DONE, an extra malformed
   prompt ("\0/gauntlet") appears after the three locked prompts, and its
   "how are you today?" ticks/token (7.27M) is ~42% below the U median for
   that prompt — unexplained. Medians are robust to it; dropping boot 4
   entirely does not change the band or sign.
3. One shakedown L boot (2026-07-31 ~19:13 local) preceded the protocol and
   is excluded by byte offset, as planned before the runs.

## Secondary observations

- L2 in-process CV per L boot: 0.114%–9.05% (freq wander under cpufreq).
- **The U cost gradient is the mechanism clue.** U ticks/token: 6.6M @85 tok,
  12.6M @257, 13.9M @214. U's own CTX_20/100/400 gauntlet probes are FLAT
  (~91.0M ticks prefill each), so KV/attention growth does NOT explain it.
  The cost tracks the *volume of text emitted*, and the dialogue-heavy
  "continue" response (most newlines per token) is the worst case.
- L reached full 1-core turbo (2.7 GHz) under Linux cpufreq while the
  unikernel's SpeedStep request was granted only ratio 25 — the OS P-state
  path is genuinely better than our firmware path (~8% of the gap at most).

## Mechanism hypotheses for the Band-3 investigation (in test order)

- **H1 (prime): synchronous UEFI console output inside the timed decode
  loop.** Firmware SimpleTextOutput scrolls by full-screen redraw; cost
  scales with lines emitted; the Linux arm's stdout goes into a pipe
  (tee) and never blocks the decode loop on rendering. Test: rebuild
  aegis-uefi with per-token printing suppressed (or timestamp around
  generation excluding output), re-run one U boot, compare.
- H2: unikernel P-state ceiling (ratio 25 vs 27) — measured, small (~8%).
- H3: memory caching attributes of UEFI-allocated buffers — test only if H1
  does not close the gap.

## Banned sentences (prereg §8) — still in force

No "the unikernel is Nx slower/faster than Linux" without the minimal-Linux
qualifier and, until the H1 experiment lands, no publication of the pooled
number at all (Band 3 rule).
