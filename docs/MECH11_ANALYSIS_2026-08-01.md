# MECH v1.1 + kernel A/B — analysis (2026-08-01)

Machine: Dell Inspiron 15 (i5-5200U, Broadwell-U), two hands-off boots.
Raw logs (append-only, extraction offsets per `RUNCARD_MECH1.1_2026-08-01.md`):
- `docs/hardware_logs/mech11_U_BOOTLOG_2026-08-01.txt` (unikernel, hardfloat
  `322efa42`, on-stick md5 re-verified at extraction)
- `docs/hardware_logs/kernelab_L_BOOTLOG_2026-08-01.txt` (minimal-Linux UKI
  `8a70d08e`, same)

TSC conversion where used: ticks ÷ **2.1975 GHz** (the A12-pinned Dell TSC
rate). Effective/nominal clock is printed in-log per measurement.

## Preregistered predictions, scored

**P1 — MECH-A1 root cause (A14): CONFIRMED.**
`process_intent` ticks/token, LOUD arm: 6,479,475 / 7,225,902 / 7,594,839
(v1 soft-float: 1.65–1.67 × 10⁹). Recovery factor = 1.65e9/7.59e6 …
1.67e9/6.48e6 = **217–258×**. Predicted band was 6.6–14M; measured 6.48–7.59M
LOUD (QUIET2 5.72–6.45M, faster still because parked+turbo). The entire
MECH-A1 anomaly was the soft-float build. A14 ⬜ → ✅.

**P2 — H1 console share on the honest denominator: CONFIRMED.**
LOUD−QUIET per prompt: 332,269 / 301,815 / 874,052 ticks/token — the same
absolute band as v1 (0.31–0.84M), confirming console cost is decode-speed-
independent firmware work. As a share of LOUD: **5.1% / 4.2% / 11.5%**
(predicted 2–13%). H1 is dead as the Band-3 mechanism but is now a real,
worth-buffering cost: QUIET beats LOUD by 4–12%.

**P3 — H2 turbo bin: REPLICATED, with a quantified residual.**
QUIET2/QUIET ticks ratio: 0.93078 / 0.93177 / 0.93167 (v1: 0.9264;
25/27 bin arithmetic: 0.92593). TURBO_DIAG: cur_ratio 25 → 27 post-park;
clock 113% → 122% — identical to v1. The ~0.6-point excess over pure bin
arithmetic = the fraction of per-token time that does not scale with core
clock (memory-latency share) — small, as expected for the compute-bound
hardfloat workload.

**P4 — H3 MTRR/PAT audit: H3 IS FALSIFIED.**
First bare-iron dump: MTRR default UC with WB variable ranges covering
0–8 GB (var0/var1), UC holes only over MMIO windows; PAT is the power-on
default `[WB WT UC- UC WB WT UC- UC]`. Verdict per engine range — image,
MODEL.SAF, EMBED.BIN, VOCAB.BIN, heap0 (16 MB), heap1 (700 MB): **WB
(uniform), every one.** No UC/WT/WC engine buffer exists. The Band-3
residual is NOT a memory-attribute effect; H3 closes as a settled negative
and the residual needs a new hypothesis (H4).

**P5 — bit-exactness: DOUBLE PASS.**
Within-boot: LOUD == QUIET == QUIET2 byte-identical per prompt (verified
programmatically). Cross-codegen: **v1 (soft-float, zero vector
instructions) and v1.1 (hardfloat AVX2+FMA) responses are byte-identical on
all three prompts** — soft-float lowered `_mm256_fmadd_ps` to fused `fmaf`
libcalls, preserving single-rounding semantics. A 217–258×-slower binary
produced the same tokens bit-for-bit: the strongest cross-implementation
determinism evidence in the program; direct CIS-1 support.

## Kernel A/B verdicts (L stick, 3 captures each, interleaved, gates green)

**Fused dual/tri matvec: SETTLED NEGATIVE.** fused/sequential best-of time
ratios across captures 1/2/3:
- DUAL BitNet-2B SwiGLU (2×6912×2560): 1.726 / 1.611 / 1.733
- TRI BitNet-2B Q/K/V GQA (2560/640/640×2560): 1.437 / 1.442 / 1.434
- DUAL M7 (2×1024×384): 1.734 / 1.735 / 1.734
- TRI M7 (3×384×384): 1.913 / 1.914 / 1.912
Fused is **1.43–1.91× slower** everywhere; outputs byte-identical in every
scenario, every capture. Mechanism matches the disassembly prediction: M×4
row pointers exceed 16 GPRs → stack reloads in the inner loop outweigh the
shared-input-load saving. The recurring "dual matvec" idea is now CLOSED
with a measured answer. Do not re-propose without a register-budget-aware
redesign (≤2 matrices × 2 rows, or interleaved weight layout).

**Bitplane-dense matvec: SETTLED NEGATIVE.** vs incumbent 7.47 GMAC/s
(stable across all 3 captures): variant (i) ordered 0.39 / 0.39 / 0.40×,
variant (ii) dual-acc 0.41 / 0.41 / 0.41×. The armchair prior (0.85–1.0×)
was falsified downward — vpshufb/vpand/vpcmpeqd mask expansion per 8-column
group costs ~2.5× the incumbent's 256×4-f32 LUT unpack. Byte-identity gates
held pre- and post-timing in every capture. The format's one surviving
virtue — scalar mirror bit-identical to AVX2 across ISAs — is a CIS-1
verification idea, not a performance idea.

## Unpreregistered but instrumented (flag as such)

- **Band-3 directional indicator:** same-day, same-machine per-prompt decode:
  unikernel QUIET2 = 384.1 / 340.6 / 351.0 tok/s (ticks ÷ 2.1975 GHz) vs
  minimal-Linux L3 replicate = 364.4 / 287.5 / 282.2 tok/s decode-only →
  unikernel **+5.4% / +18.5% / +24.4%**. Arms are not protocol-paired (QUIET2
  buffers console, Linux harness prints; token counts differ by 1), so this
  does NOT overturn Band 3 by itself — it says the −39.3% pooled gap does not
  survive hardfloat + AP-PARK + buffered console. A preregistered paired redo
  is the publishable path (proposed MECH v2).
- **Batched prefill GEMM on iron:** 27-token prefill, per-token 102,570,215 →
  batched 65,728,001 ticks = **1.5605×** batched speedup, same boot, clock
  122% both sides. First bare-metal instrument behind the (dev-box) 1.63–1.75×
  claim.
- **SIMD ratio on iron:** scalar 529,483,549 vs AVX2 82,967,126 ticks for the
  same 33-token prefill = **6.38×** at matched 122% clock.
- **A12 durability datapoint:** boot-pre TURBO_DIAG shows `bdprochot_en=0`,
  "STAGE 7 bd-prochot: already disabled, no write issued" — the clear
  SURVIVED at least one full power cycle; prochot_log=1 records the historical
  assertion only.

## Consequences

1. A14 ⬜ → ✅ (P1). MECH-A1 closed end-to-end: cause, fix, gate, iron
   confirmation.
2. H3 closed as settled negative (all-WB). Band-3 residual → H4 hypothesis
   design; the honest starting point is that v1.1 conditions already erase
   the sign (see indicator above), so H4 may simply be "H1+H2+workload".
3. Fused (A16) and bitplane (A17) are settled negatives; ReLU² column-skip
   (A15, task #27) is now the only live kernel candidate.
4. CIS-1 gains the cross-codegen byte-identity exhibit (P5).
