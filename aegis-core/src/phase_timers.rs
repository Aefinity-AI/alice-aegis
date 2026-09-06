//! Per-phase decode cycle counters (Amdahl decomposition of decode).
//!
//! Gated entirely behind the `phase-timers` cargo feature, default OFF: when
//! the feature is off this module does not exist, `TernaryInferenceEngine`
//! carries no extra field, and every call site that would otherwise touch a
//! counter is removed by `#[cfg]` before codegen — not branched around, not
//! folded away by the optimizer's discretion, actually absent from the AST
//! the compiler sees. Zero cost means zero cost.
//!
//! ## Which counter, and why
//!
//! x86_64 gives two free-running cycle counters: `RDTSC` and `RDTSCP`.
//! Neither one is a core-cycle counter (see `aegis-linux/examples/clockstate.rs`
//! for the invariant-TSC-vs-core-clock distinction — a ratio this module does
//! NOT attempt to correct for; every number here is in TSC ticks, and the
//! caller is responsible for converting via a runtime-calibrated `tsc_hz` if
//! it wants wall time). What they DO give, reliably, is elapsed ticks between
//! two points in the instruction stream, provided the boundary is fenced:
//!
//!   * `RDTSC` alone can retire out of program order relative to the code
//!     around it — the CPU is free to execute later instructions before the
//!     counter read commits, and earlier instructions after it. An
//!     unfenced RDTSC pair can therefore measure less (or more) than the
//!     code between them actually took.
//!   * `RDTSCP` waits for all prior instructions to retire before it reads
//!     the counter (it is partially serializing on the *read* side), but
//!     instructions after it can still start before its result is used.
//!
//! This module uses the sequence Intel's own benchmarking guidance
//! ("How to Benchmark Code Execution Times on Intel IA-32 and IA-64
//! Instruction Set Architectures", Paoloni 2010) recommends when `CPUID`
//! serialization is too expensive to pay on every sample (it is, here — a
//! `CPUID` is hundreds of cycles, dwarfing several of the phases we time):
//!
//!   start = LFENCE; RDTSC      (LFENCE drains the load/store and
//!                                out-of-order-issue queues so RDTSC cannot
//!                                be hoisted above prior instructions)
//!   end   = RDTSCP; LFENCE     (RDTSCP cannot be issued before prior
//!                                instructions retire; the trailing LFENCE
//!                                stops LATER instructions from being
//!                                dispatched before RDTSCP's result lands,
//!                                so the interval's end is not open on the
//!                                right either)
//!
//! This is not full serialization (that is what CPUID buys you) but it is a
//! documented, standard middle ground, and it is applied identically to the
//! overhead-calibration pairs (see `calibrate_overhead`) and every phase
//! span, which is the property that actually matters: the *same* systematic
//! bias is present in the overhead sample and every measurement it is
//! subtracted from, so it approximately cancels even though it is not zero.

use core::arch::x86_64::{__rdtscp, _mm_lfence, _rdtsc};

/// Start of a fenced interval. See module docs for why LFENCE precedes RDTSC.
#[inline(always)]
pub fn tick_start() -> u64 {
    // SAFETY: RDTSC/LFENCE are unprivileged, always available on x86_64, and
    // read-only with no observable side effects beyond the returned value.
    unsafe {
        _mm_lfence();
        _rdtsc()
    }
}

/// End of a fenced interval. See module docs for why RDTSCP precedes LFENCE.
#[inline(always)]
pub fn tick_end() -> u64 {
    // SAFETY: see tick_start. The TSC_AUX value RDTSCP also returns (core/
    // socket id on most kernels) is discarded — this module only ever wants
    // the tick count, never the processor id.
    unsafe {
        let mut aux: u32 = 0;
        let t = __rdtscp(&mut aux as *mut u32);
        _mm_lfence();
        t
    }
}

/// Which of the decode loop's named phases a span belongs to. `Other` is
/// deliberately NOT a variant here: per the task's own definition, "other" is
/// derived as `total - sum(phases)` at report time, never accumulated
/// directly, so a bug that forgets to time some code shows up as inflated
/// `other` rather than silently vanishing into the wrong bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Phase {
    /// Every ternary GEMV/matvec call: q/k/v/o/up/gate/down_proj, all layers.
    Gemv = 0,
    /// QK^T + softmax + AV, one span per layer (the span wraps the whole
    /// head loop, so `pairs[Attn]` counts layers, not heads). NOTE: this also
    /// contains the KV-cache *read* traffic — see `Phase::Kv` doc comment
    /// for why the read side could not be split out honestly.
    Attn = 1,
    /// KV-cache *write* only (the two `copy_from_slice` calls into the
    /// current position's K/V slot, once per layer per token). The read
    /// side lives inside `Attn`: it happens inside a `for t in 0..=seq_pos`
    /// loop whose body already does a dot product or a weighted add in the
    /// same iteration, so isolating the read would require one additional
    /// fenced tick pair per (head, t) — up to `num_heads * seq_pos` extra
    /// pairs per layer per token. At ctx=2048 that is thousands of pairs
    /// whose own ~tens-of-cycles overhead would exceed the few cycles a
    /// cache-line-resident slice index costs, corrupting exactly the number
    /// it was meant to reveal. The context-length dependence this project
    /// wants visible (KV cost grows with seq_pos) still shows up — as growth
    /// in `Attn`'s share across the three context lengths — just not
    /// separated from the O(1)-per-token compute in the same loop.
    Kv = 2,
    /// RMSNorm calls: input_layernorm, attn_sub_norm, post_attention_layernorm,
    /// ffn_sub_norm, and the final norm before the LM head.
    Norm = 3,
    /// The fused LM-head projection + argmax over the tied embedding table.
    LmHead = 4,
    /// Repetition-penalty pass over generated tokens + the conditional
    /// re-argmax it can trigger.
    Sample = 5,
}

pub const NUM_PHASES: usize = 6;

/// Fixed-size, zero-allocation phase accumulators. Owned by the engine (or
/// its caller); never heap-allocates, so it is safe to live inside the hot
/// decode loop's owning struct.
#[derive(Clone, Copy, Debug)]
pub struct PhaseCycles {
    /// Raw (uncorrected) accumulated tick count per phase.
    pub raw: [u64; NUM_PHASES],
    /// Number of fenced tick pairs contributing to each phase's `raw` entry
    /// — needed at report time to subtract `pairs * overhead_mean`.
    pub pairs: [u64; NUM_PHASES],
    /// Raw accumulated tick count for the whole-decode-step span (forward
    /// step + final norm + LM head + sampling), one pair per decode step.
    pub total_raw: u64,
    pub total_pairs: u64,
}

impl PhaseCycles {
    pub const fn zero() -> Self {
        Self {
            raw: [0u64; NUM_PHASES],
            pairs: [0u64; NUM_PHASES],
            total_raw: 0,
            total_pairs: 0,
        }
    }

    #[inline(always)]
    pub fn record(&mut self, phase: Phase, start: u64, end: u64) {
        let i = phase as usize;
        self.raw[i] = self.raw[i].wrapping_add(end.wrapping_sub(start));
        self.pairs[i] = self.pairs[i].wrapping_add(1);
    }

    #[inline(always)]
    pub fn record_total(&mut self, start: u64, end: u64) {
        self.total_raw = self.total_raw.wrapping_add(end.wrapping_sub(start));
        self.total_pairs = self.total_pairs.wrapping_add(1);
    }
}

/// Times `reps` empty fenced start/stop pairs and returns
/// `(sum_of_raw_ticks, mean_ticks_per_pair)`. Call this once at process
/// startup, before any decode, on the same core the decode loop will run on
/// (no core-pinning is done here — the caller controls scheduling; a
/// migration between calibration and use would invalidate the estimate, and
/// callers on shared/virtualized hardware should treat a wide spread across
/// repeated calibration runs as a reason to distrust the subtraction, not a
/// reason to hide it).
pub fn calibrate_overhead(reps: usize) -> (u64, f64) {
    let mut total: u64 = 0;
    for _ in 0..reps {
        let t0 = tick_start();
        let t1 = tick_end();
        total = total.wrapping_add(t1.wrapping_sub(t0));
    }
    let mean = if reps > 0 {
        total as f64 / reps as f64
    } else {
        0.0
    };
    (total, mean)
}

/// Subtracts `pairs * overhead_mean` from `raw`, floored at 0.0 — a phase
/// whose true cost is smaller than its own measurement overhead reports 0.0
/// rather than a nonsensical negative number.
#[inline]
pub fn corrected(raw: u64, pairs: u64, overhead_mean: f64) -> f64 {
    let c = raw as f64 - pairs as f64 * overhead_mean;
    if c > 0.0 {
        c
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_accumulates_and_counts_pairs() {
        let mut pc = PhaseCycles::zero();
        pc.record(Phase::Gemv, 100, 150);
        pc.record(Phase::Gemv, 200, 260);
        assert_eq!(pc.raw[Phase::Gemv as usize], 50 + 60);
        assert_eq!(pc.pairs[Phase::Gemv as usize], 2);
        assert_eq!(pc.raw[Phase::Attn as usize], 0);
        assert_eq!(pc.pairs[Phase::Attn as usize], 0);
    }

    #[test]
    fn record_total_independent_of_phase_slots() {
        let mut pc = PhaseCycles::zero();
        pc.record_total(1000, 5000);
        assert_eq!(pc.total_raw, 4000);
        assert_eq!(pc.total_pairs, 1);
    }

    #[test]
    fn corrected_subtracts_overhead_per_pair() {
        // 10 pairs, 5.0 ticks overhead each -> 50.0 subtracted.
        assert_eq!(corrected(1050, 10, 5.0), 1000.0);
    }

    #[test]
    fn corrected_floors_at_zero_never_negative() {
        // Overhead alone would exceed raw -> must report 0.0, not negative.
        assert_eq!(corrected(10, 10, 5.0), 0.0);
    }

    #[test]
    fn corrected_zero_pairs_is_zero() {
        assert_eq!(corrected(0, 0, 5.0), 0.0);
    }

    #[test]
    fn calibrate_overhead_zero_reps_is_zero_mean() {
        let (total, mean) = calibrate_overhead(0);
        assert_eq!(total, 0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn calibrate_overhead_runs_and_returns_nonzero_pairs_mean() {
        // Smoke test only (real hardware timing, not a fixed expectation):
        // the mean must be finite and non-negative. rdtscp is monotonic on a
        // single core with no migration inside this tight loop.
        let (total, mean) = calibrate_overhead(1000);
        assert!(mean.is_finite());
        assert!(mean >= 0.0);
        assert_eq!(total as f64, mean * 1000.0);
    }

    #[test]
    fn sum_check_matches_total_when_phases_cover_everything() {
        // sum_check = sum(corrected phases) / corrected(total). If every
        // cycle in a synthetic "total" span is accounted for by the phases,
        // sum_check must land at 1.0 (within float rounding).
        let overhead_mean = 3.0;
        let mut pc = PhaseCycles::zero();
        pc.record(Phase::Gemv, 0, 1003); // 1000 real + 3 overhead
        pc.record(Phase::Attn, 0, 2003); // 2000 real + 3 overhead
        pc.record_total(0, 3006); // 3000 real (=1000+2000) + 3 overhead
        let gemv = corrected(pc.raw[Phase::Gemv as usize], pc.pairs[Phase::Gemv as usize], overhead_mean);
        let attn = corrected(pc.raw[Phase::Attn as usize], pc.pairs[Phase::Attn as usize], overhead_mean);
        let total = corrected(pc.total_raw, pc.total_pairs, overhead_mean);
        let sum_check = (gemv + attn) / total;
        assert!((sum_check - 1.0).abs() < 0.02, "sum_check={sum_check}");
    }
}
