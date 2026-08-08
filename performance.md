# Performance: kernel inventory, roadmap, honest benchmarking

## What's already good — keep it

The 4-row-unrolled, LUT-based AVX2 ternary matvec is genuinely strong: a 256-entry
compile-time LUT decodes 4 packed 2-bit ternary weights per byte into f32 lanes,
consumed with `_mm256_fmadd_ps` + horizontal sums. Don't rewrite it. Remaining
headroom there is weight-row prefetching and an 8-row unroll if register pressure
allows — measure before and after, on the target box.

The zero-allocation `WorkingMemoryArena` is the right pattern for OOM-free
bare-metal inference. Extend it; never allocate in the token loop.

## Where the time actually goes

Per decode token, the dominant cost is the **full-vocab F32 LM head**:
~50,189 × 2560 ≈ 128M MACs in f32, every token, memory-bandwidth bound. The
attention QK·/·V inner loops (head_dim = 128) are scalar and grow O(seq²·head_dim).
Everything below is ordered by expected win per unit of risk.

## Optimization roadmap (verify current state before starting any item —
changelog claims of completion have been wrong before)

1. **Fuse argmax into the LM-head dot.** Greedy decode needs only the argmax, not
   materialized logits: track running (max, index) inside the dot loop, drop the
   `logits.fill(-INF)` and the second pass. Same math, one pass, less bandwidth.
2. **Vectorize attention inner loops.** QK· and ·V loops over head_dim 128 use the
   exact `_mm256_fmadd_ps` + horizontal-sum pattern already proven in the ternary
   matvec. Straightforward, big at long context.
3. **BF16 embeddings + widen-on-load.** Halves EMBED.BIN (514MB → 257MB) and
   halves LM-head memory traffic — bandwidth is the limiter, so this is a real
   speedup, not just a size win. Widen with `_mm256_cvtepu16_epi32` + 16-bit left
   shift into f32 lanes. This *changes the artifact dtype*: forge, the UEFI
   loader's expectations, and every reader must flow through the single derived
   element size (prime directive 3). Do forge + engine in one change, with the
   init-time byte-length assert updated, and re-run the coherence parity check.
4. **LUT init once.** Move `init_unpack_lut` from `static mut` + per-call flag
   check to one-time init at engine construction.
5. **(Milestone, not a tweak) int8 activation path.** True BitNet W1.58A8:
   per-token activation quantization to int8, integer accumulation. This is where
   the order-of-magnitude lives, and it's a large, accuracy-affecting change —
   scope it as its own milestone with before/after perplexity, never as a drive-by.

Current f32 activations are a *deliberate* accuracy-over-speed choice (fidelity ≥
reference for retained tokens). Any pitch that says "fastest possible" must either
include milestone 5 or say cycles are being spent on fidelity — pick one honestly.

## Honest benchmarking (the only kind this project ships)

**Clock calibration.** `rdtsc` counts TSC ticks; TSC frequency ≠ core clock and is
not 2.5 GHz by assumption. Either read the TSC frequency from CPUID leaf 0x15
(crystal ratio) at startup and derive seconds, or report **cycles/token** and label
it as such. A fixed-constant conversion is a fabricated denominator.

**What a benchmark run must do.** Generate real tokens through the real engine in
the same process that prints the numbers; report token count, total cycles,
calibrated seconds (if available), tokens/sec, and the prompt used. Warmup pass
excluded from timing, stated as such. TTFT measured, not asserted. Memory numbers
measured (arena sizes are known at init; report them), not invented.

**Perplexity.** Real WikiText-2 test split, teacher-forced forward passes,
accumulate NLL over predicted tokens, report exp(mean NLL) with the exact eval
config (context length, stride, tokenizer). Two non-negotiables:
tokenize through the **pruned** mapping (specials remapped from 128000+ to
~50000+, see architecture.md) or the numbers are meaningless; and report the
baseline (unpruned/reference) measured the same way, including where the pruned
model loses. The pruning cost is a finding, not an embarrassment.

**Baselines.** Same hardware, same prompts, same token counts vs `bitnet.cpp` and
`llama.cpp`: tokens/sec, TTFT, peak RSS, energy if measurable. Publish the losing
comparisons too — a review board trusts a table with red cells far more than one
without.

**Regression discipline.** Every optimization lands with a before/after
measurement on the same box and, for anything touching math, an unchanged parity
check and (if feasible) unchanged greedy outputs on a fixed prompt set. "It should
be faster" is not a result.
