# ARM ternary (BitNet b1.58) kernel research — next-step candidates

Source: mailbox research task #48 (Grok, assigned 2026-10-21, result recorded 2026-10-21 13:15Z,
Supabase `aefinity-mailbox.mailbox_tasks` id 48). Full raw result kept at
`state/reports/bible/research_grok_48.md` in claudius-maximus (not this repo) for provenance; this
file is the folded, repo-relevant summary.

Context given to Grok: ALICE runs BitNet b1.58 2B-4T on a Motorola Edge 2025 (4x Cortex-A78 + 4x A55,
ARMv8.2, dotprod/asimddp present, no i8mm, no SVE). Current kernel: 2-bit packed weights decoded via
shift/mask/TBL to int8, int8 per-token activations, SDOT into int32. Measured on-device: decode 4.5
tok/s, prefill ~0.5M timer cycles/token.

All numbers below are **other people's hardware**, not measured on this phone. Flags per item:
SOURCED (citable), ESTIMATE (our arithmetic), UNVERIFIED (claimed, no citation found).

## 1. bitnet.cpp (microsoft/BitNet) ARM kernels

- I2_S [SOURCED, arXiv 2410.16144 §2 Table 1; `src/ggml-bitnet-mad.cpp` L317-401]: same scheme as
  ALICE's current kernel — 2 bits/weight, unpack to int8, `vdotq_s32` (SDOT) when `__ARM_FEATURE_DOTPROD`
  is present, else `vmlal_s8`.
- TL1 [SOURCED, arXiv 2502.11880 §3.1.1, App. D Alg. 3, Table 5; `utils/codegen_tl1.py` L100-222]:
  lookup-table method, g=2 weights per 4-bit index (2 bpw). Builds int16 partial-sum tables per
  activation pair, splits into hi/lo byte tables via `vqtbl1q_s8`, reconstructs with `vzipq_s8`.
  TL1_1 is lossless; TL1_0 quantizes the table ("negligible loss" per the paper).
- No published tok/s for BitNet-2B-4T on any ARM phone. On Apple M2 Ultra 3.8B, TL1_0 (90.73 tok/s) is
  about equal to I2_S (91.65 tok/s) [SOURCED, arXiv 2502.11880 Table 7] — no evidence TL1 alone beats
  an SDOT kernel on ARM.
- Cobalt 100 (Neoverse N2, server ARM, DOTPROD), optimized vs original I2_S: tg speedup 1.49-1.67x,
  pp128 speedup 2.10-2.19x [SOURCED, but read by eye from a chart in the repo's own assets].

## 2. T-MAC (arXiv 2407.00088, EuroSys'25)

LUT-based, splits an n-bit weight into n 1-bit planes; BitNet b1.58 costs 2 planes (2 bpw). Reported:
BitNet-3B 30 tok/s (1 core) / 71 tok/s (8 cores) on M2 Ultra, 11 tok/s on Raspberry Pi 5 [SOURCED §5.3].
bitnet.cpp's own comparison: TL2_0 beats T-MAC by up to 1.19x on M2 Ultra [SOURCED, 2502.11880 §4.1.3].
Caveat: T-MAC's baselines are llama.cpp dequant kernels, not an SDOT ternary kernel like ours.

## 3. Other ternary-on-ARM kernels

- llama.cpp TQ1_0/TQ2_0 [SOURCED, PR 8151 @9bc6db28]: faster vec_dot throughput than Q4_K on
  Cortex-A72/Neoverse N1, but **not lossless for BitNet** — only 1.4% exact-match vs FP32, vs 100% for
  I2_S/TL1/TL2 [SOURCED, 2410.16144 Table 7]. Not usable if bit-exactness is required.
- TL2 (1.67 bpw, g=3+sign): not enabled for 2B-4T on ARM in the upstream repo.

## 4. Is phone decode memory-bandwidth bound?

- Weights at 2 bpw: ~521 MB [ESTIMATE from `config.json` arithmetic] (tech report gives 0.4 GB
  non-embedding). LM head: 657/328/269 MB at fp16/int8/Q6_K [ESTIMATE].
- Dimensity 7400 LPDDR "up to 6400 Mbps" [SOURCED, MediaTek product page]; theoretical peak ≈51.2 GB/s
  assuming a 4x16-bit bus [ESTIMATE, bus width not independently confirmed].
- Ceiling: ~98 tok/s (weights only) or ~43 tok/s (with fp16 head) at peak bandwidth; roughly half that
  if practical bandwidth is 50% of peak [ESTIMATE, not sourced].
- Current measured ~4.5 tok/s is ~2.3-5.3 GB/s effective — far below any plausible ceiling.
  **Conclusion: decode today is compute/latency-bound, not bandwidth-bound. A 2x gain is not blocked by
  DRAM.** [ESTIMATE from measured + above]

## 5. Recommendation

Port bitnet.cpp's I2_S GEMV/GEMM structure (row-blocked, activation-parallel SDOT, configurable tiling:
`include/gemm-config.h` ROW_BLOCK_SIZE 4 / COL_BLOCK_SIZE 128 / PARALLEL_SIZE 4) across the phone's 4
A78 threads, rather than TL1 or T-MAC:

- Stays bit-exact (same int32-accumulation, int8-activation math we already use).
- Only technique with a *published* gain over an SDOT ternary baseline on an ARM DOTPROD core (Cobalt
  100: tg 1.49-1.67x, pp 2.10-2.19x).
- TL1 showed no advantage over I2_S on the one ARM datapoint available (M2 Ultra).
- Risk: the Cobalt 100 gain is measured on a server Neoverse N2 (more bandwidth/cache) than our A78;
  untested on this exact core. If ALICE's kernel is currently single-threaded, most of the gain may
  just be from adding threads, not from the tiling itself — check this before crediting the technique.

**Before any kernel work**: confirm (a) whether ALICE decode is currently single- or multi-threaded,
(b) the tied LM head's dtype (fp16 vs int8/Q6_K — this could dominate bytes/token), (c) run a pinned
STREAM-style read-bandwidth test on the Edge 2025 to fix the real ceiling instead of the MediaTek
spec-page estimate above.

## Summary table

| Technique | Reported speedup | Hardware | Source |
|---|---|---|---|
| bitnet.cpp I2_S tiled/threaded vs original I2_S | tg 1.49-1.67x, pp 2.10-2.19x | Cobalt 100 (Neoverse N2), DOTPROD | microsoft/BitNet README + perf chart @0b341e58 |
| bitnet.cpp I2_S vs llama.cpp fp16 | 1.37-5.07x (2.42x at 2.5B) | Apple M2 Ultra | arXiv 2410.16144 Table 4 |
| TL1_0 vs I2_S | ~0.99x (90.73 vs 91.65 tok/s, 3.8B) | Apple M2 Ultra | arXiv 2502.11880 Table 7 |
| TL2_0 vs T-MAC | up to 1.19x | Apple M2 Ultra | arXiv 2502.11880 §4.1.3 |
| TL2_0 vs TQ1_0 | up to 1.65x | Apple M2 Ultra | arXiv 2502.11880 §4.1.3 |
| T-MAC vs llama.cpp e2e, BitNet-3B | 1.7x multi-thread; 5.8x single-thread (RPi5) | M2 Ultra / RPi 5 (A76) | arXiv 2407.00088 §5.3 |
| llama.cpp TQ2_0 vs Q4_K vec_dot | 1.45x (65.76 vs 45.40 GB/s) | Neoverse N1 DOTPROD | llama.cpp PR 8151 @9bc6db28 |

**Nothing above was measured on a Cortex-A78 or on this phone.** All speedups are from other authors'
hardware; treat as priors for what to try next, not as a promised result.
