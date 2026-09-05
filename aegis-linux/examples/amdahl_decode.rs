//! amdahl_decode — Amdahl decomposition of greedy decode into named phases
//! (GEMV, attention, KV-cache write, RMSNorm, LM head, sampling), reported
//! as a share of total corrected TSC ticks at three context lengths.
//!
//! Timing/perf artifact (Rule A): built ONLY with the `phase-timers` cargo
//! feature (forwarded to aegis-core), which gates a zero-cost RDTSC/RDTSCP
//! counter block per phase — see aegis-core/src/phase_timers.rs for the
//! fencing rationale and why KV-cache reads live inside `Attn`, not `Kv`.
//!
//! RDTSC/RDTSCP tick counts are NOT core cycles: the TSC runs at its own
//! invariant nominal rate regardless of the core's actual clock, so every
//! number this binary prints is an elapsed-tick count, converted to wall
//! time only via a `tsc_hz` this process calibrates for itself against
//! CLOCK_MONOTONIC (the same technique aegis-linux/examples/clockstate.rs
//! uses) — never a hardcoded constant. See clockstate.rs's module docs for
//! the full core-clock-vs-invariant-TSC distinction.
//!
//! Method per context length N:
//!   1. Fresh engine, one warm-up decode pass of N tokens (discarded — this
//!      is the first-touch/cold-cache/ramp pass, not a measurement).
//!   2. Three measured passes, each on a FRESH engine (clean KV cache), each
//!      resetting the phase-cycle accumulators right before the decode loop
//!      it measures (`reset_phase_cycles`) so a pass's AMDAHL line reports
//!      only that pass's decode, not prefill or any earlier pass.
//!   3. The three measured passes' corrected phase ticks and corrected total
//!      ticks are each summed before computing percentages, i.e. the
//!      reported line is the N=3 aggregate, not a single run's noise.
//!
//! Run: amdahl_decode <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN>
//!
//! Build: cd aegis-linux && cargo build --release --features parallel,phase-timers --example amdahl_decode

#[cfg(not(feature = "phase-timers"))]
fn main() {
    eprintln!(
        "amdahl_decode requires the `phase-timers` feature: \
         cargo build --release --features parallel,phase-timers --example amdahl_decode"
    );
    std::process::exit(2);
}

#[cfg(feature = "phase-timers")]
fn main() {
    use aegis_core::inference::TernaryInferenceEngine;
    use aegis_core::phase_timers::{self, Phase, PhaseCycles};

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: amdahl_decode <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN>");
        std::process::exit(2);
    }
    let model = std::fs::read(&args[1]).expect("read MODEL.SAF");
    let emb = std::fs::read(&args[2]).expect("read EMBED.BIN");
    let vocab = std::fs::read(&args[3]).expect("read VOCAB.BIN");

    // A long, generic prompt so 2048-token context lengths can be reached by
    // decode alone without depending on prompt length; process_intent caps
    // the prompt to the model's window internally if this ever needs to grow.
    let prompt = "Describe, in as much technical detail as you can, how a modern \
        transformer decoder processes a single token: embedding lookup, \
        multi-head attention with a KV cache, RMSNorm, the feed-forward \
        block, and the final projection to logits. Keep going for as long \
        as you can, and cover matrix multiplication, ternary weight \
        quantization, rotary position embeddings, and why the attention \
        cost grows with sequence length.";

    // -------------------------------------------------------------------
    // Calibrate RDTSC/RDTSCP fenced-pair overhead (per phase_timers.rs doc:
    // once at process startup, same core the decode loop runs on).
    // -------------------------------------------------------------------
    let (overhead_total, overhead_mean) = phase_timers::calibrate_overhead(200_000);
    let overhead_ticks = overhead_total;

    // -------------------------------------------------------------------
    // Calibrate TSC ticks/second against CLOCK_MONOTONIC over >= 200 ms,
    // exactly as aegis-linux/examples/clockstate.rs does. Never a fixed
    // constant (scripts/integrity_check.py has a tripwire for that).
    // -------------------------------------------------------------------
    let tsc_hz = calibrate_tsc_hz();

    println!(
        "# amdahl_decode: RDTSC/RDTSCP invariant-TSC ticks, NOT core cycles \
         (see aegis-linux/examples/clockstate.rs); tsc_hz below is calibrated \
         against CLOCK_MONOTONIC in this process, not assumed."
    );
    println!(
        "# overhead calibration: {overhead_ticks} ticks / 200000 fenced pairs, \
         mean {overhead_mean:.3} ticks/pair"
    );

    for &ctx in &[256usize, 1024usize, 2048usize] {
        // Warm-up pass: fresh engine, discarded.
        {
            let mut engine =
                TernaryInferenceEngine::new(&emb, &model, &vocab).expect("engine init (warmup)");
            engine.reset_phase_cycles();
            let _ = engine.process_intent(prompt, ctx, |_| {});
        }

        // Three measured passes, aggregated.
        let mut agg = PhaseCycles::zero();
        let mut total_tokens: u64 = 0;
        for _ in 0..3 {
            let mut engine =
                TernaryInferenceEngine::new(&emb, &model, &vocab).expect("engine init (measured)");
            engine.reset_phase_cycles();
            let _ = engine.process_intent(prompt, ctx, |_| {});
            let pc = engine.phase_cycles;
            for i in 0..phase_timers::NUM_PHASES {
                agg.raw[i] = agg.raw[i].wrapping_add(pc.raw[i]);
                agg.pairs[i] = agg.pairs[i].wrapping_add(pc.pairs[i]);
            }
            agg.total_raw = agg.total_raw.wrapping_add(pc.total_raw);
            agg.total_pairs = agg.total_pairs.wrapping_add(pc.total_pairs);
            total_tokens += pc.total_pairs;
        }

        let total_ticks = phase_timers::corrected(agg.total_raw, agg.total_pairs, overhead_mean);

        let gemv = phase_timers::corrected(
            agg.raw[Phase::Gemv as usize],
            agg.pairs[Phase::Gemv as usize],
            overhead_mean,
        );
        let attn = phase_timers::corrected(
            agg.raw[Phase::Attn as usize],
            agg.pairs[Phase::Attn as usize],
            overhead_mean,
        );
        let kv = phase_timers::corrected(
            agg.raw[Phase::Kv as usize],
            agg.pairs[Phase::Kv as usize],
            overhead_mean,
        );
        let norm = phase_timers::corrected(
            agg.raw[Phase::Norm as usize],
            agg.pairs[Phase::Norm as usize],
            overhead_mean,
        );
        let lmhead = phase_timers::corrected(
            agg.raw[Phase::LmHead as usize],
            agg.pairs[Phase::LmHead as usize],
            overhead_mean,
        );
        let sample = phase_timers::corrected(
            agg.raw[Phase::Sample as usize],
            agg.pairs[Phase::Sample as usize],
            overhead_mean,
        );

        let named_sum = gemv + attn + kv + norm + lmhead + sample;
        let other = (total_ticks - named_sum).max(0.0);

        let pct = |x: f64| if total_ticks > 0.0 { 100.0 * x / total_ticks } else { 0.0 };

        // sum_check: named phases + other must reconstruct 100% of total by
        // construction (other is defined as the residual) — this is a
        // sanity check on the arithmetic above, not an independent
        // measurement, so drift here means a bug, not measurement noise.
        let sum_pct = pct(named_sum) + pct(other);
        let drift = (sum_pct - 100.0).abs();
        let sum_check = if drift < 0.05 {
            "ok".to_string()
        } else {
            format!("drift:{drift:.4}")
        };

        println!(
            "AMDAHL ctx={ctx} tokens={total_tokens} total_ticks={total_ticks:.0} \
             gemv={:.2} attn={:.2} kv={:.2} norm={:.2} lmhead={:.2} sample={:.2} other={:.2} \
             sum_check={sum_check} tsc_hz={tsc_hz:.0} overhead_ticks={overhead_ticks}",
            pct(gemv),
            pct(attn),
            pct(kv),
            pct(norm),
            pct(lmhead),
            pct(sample),
            pct(other),
        );
    }

    println!(
        "# NOTE: all figures above are invariant-TSC ticks (RDTSC/RDTSCP), not core \
         cycles — the TSC runs at its own nominal rate regardless of the core's \
         actual clock speed. Do not report these as cycles/token or divide by a \
         cycles-based peak-FLOPs figure without first rescaling by the core-GHz/ \
         nominal-GHz ratio (aegis-linux/examples/clockstate.rs)."
    );
}

/// Calibrate TSC ticks/second against CLOCK_MONOTONIC over a >= 200 ms
/// window. Mirrors aegis-linux/examples/clockstate.rs::calibrate_tsc_hz —
/// duplicated here (not imported) because clockstate.rs is a binary, not a
/// library, and the two probes must each be independently self-contained.
#[cfg(feature = "phase-timers")]
fn calibrate_tsc_hz() -> f64 {
    use std::arch::x86_64::_rdtsc;
    use std::time::Instant;
    // SAFETY: rdtsc is unprivileged and always available on x86_64; it reads
    // a counter and has no observable side effects.
    let t0 = Instant::now();
    let c0 = unsafe { _rdtsc() };
    while t0.elapsed().as_millis() < 200 {
        std::hint::spin_loop();
    }
    let c1 = unsafe { _rdtsc() };
    let secs = t0.elapsed().as_secs_f64();
    (c1 - c0) as f64 / secs
}
