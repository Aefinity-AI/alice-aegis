//! Same-binary interleaved A/B: incumbent AVX2 LUT+FMA ternary matvec vs the
//! bitplane-dense candidates (`ops_bitplane`, variants (i) order-preserving
//! and (ii) dual-accumulator). Methodology copied from the 2026-07-30
//! LUT-mpGEMM same-binary A/B (docs/hardware_logs/
//! lut_mpgemm_ab_findings_2026-07-30.md): kernel A/Bs on the dev box are
//! BIMODAL, so non-interleaved comparisons are inadmissible — every arm runs
//! in this ONE binary, alternating inside one rep loop, on the same packed
//! weights, and a verdict needs >= 3 logged runs.
//!
//! The bitplane family is NOT the settled CTZ negative (ledger A6): no
//! `trailing_zeros`, no bit scan, no per-nonzero branch — dense SIMD mask
//! consumption, sparsity-independent. And unlike the settled T-MAC pshufb
//! layout (A7), it stays at 2 bits/weight: decode memory traffic is identical
//! to the incumbent's, so this race is decided kernel-side.
//!
//! Armchair prior going in (recorded so the result can contradict it):
//! 0.85-1.0x of the incumbent on AVX2 — mask expansion (shuffle+and+cmpeq+
//! and+add/sub per plane) costs more vector-port uops than LUT unpack+FMA.
//! The bench exists to SETTLE the family either way.
//!
//! Rule A: numbers printed here are only admissible when captured on physical
//! hardware, machine named, with the clock-state block below bound to the
//! capture (RDTSC ticks are NOT core cycles; the effective/nominal ratio
//! rescales every tick-derived figure). Rule B: a figure enters the ledger
//! only via a raw log under docs/hardware_logs/ (use scripts/capture-bench.sh
//! or redirect the full stdout).
//!
//! Run: cargo run --release --bin bitplane_vs_lut [reps]   (default 5)

// `cargo test` auto-builds bin targets on every architecture so integration
// tests can exec them; this bench is x86-only, so non-x86 gets a stub main.
// It also needs the AVX2 kernels this bench races, which are cfg'd out under
// `scalar_only` — so that build gets the same stub.
#[cfg(any(not(target_arch = "x86_64"), feature = "scalar_only"))]
fn main() {
    eprintln!("x86_64 AVX2-only benchmark; nothing to run on this architecture/feature set");
}

#[cfg(all(target_arch = "x86_64", not(feature = "scalar_only")))]
mod x86 {

    use aegis_core::ops::ternary_matvec;
    use aegis_core::ops_bitplane::{
        bitplane_matvec_avx2, bitplane_matvec_avx2_dual, bitplane_matvec_scalar_dual,
        bitplane_words_per_row, pack_bitplanes,
    };
    use std::arch::x86_64::_rdtsc;
    use std::time::Instant;

    const DIM_OUT: usize = 2560;
    const DIM_IN: usize = 6912; // BitNet down_proj, the decode-dominant shape

    // --------------------------------------------------------------------------
    // Clock state (Rule A, RDTSC corollary) — ported from
    // aegis-linux/examples/clockstate.rs so this binary's stdout is a complete
    // instrument record even without the capture wrapper.
    // --------------------------------------------------------------------------

    /// Executes `n` iterations of an 8-deep dependent add chain. Each add has
    /// 1-cycle latency and depends on its predecessor, so the chain retires at
    /// exactly one op per core cycle: the op count is a direct core-cycle count.
    fn dependent_add_chain(n: u64) -> u64 {
        let mut x: u64 = 1;
        let mut i: u64 = n;
        // SAFETY: pure register arithmetic on locals. No memory operands, no
        // observable side effects; every register touched is declared inout so
        // the compiler cannot assume a value survives the block. `nostack` holds
        // because the block neither pushes nor calls.
        unsafe {
            std::arch::asm!(
                "2:",
                "add {x}, 1",
                "add {x}, 1",
                "add {x}, 1",
                "add {x}, 1",
                "add {x}, 1",
                "add {x}, 1",
                "add {x}, 1",
                "add {x}, 1",
                "dec {i}",
                "jnz 2b",
                x = inout(reg) x,
                i = inout(reg) i,
                options(nostack),
            );
        }
        x.wrapping_add(i)
    }

    /// Calibrate TSC ticks/second against CLOCK_MONOTONIC over a ~200 ms window.
    fn calibrate_tsc_hz() -> f64 {
        let t0 = Instant::now();
        // SAFETY: rdtsc is unprivileged and always available on x86_64; it reads
        // a counter and has no side effects.
        let c0 = unsafe { _rdtsc() };
        while t0.elapsed().as_millis() < 200 {
            std::hint::spin_loop();
        }
        // SAFETY: as above.
        let c1 = unsafe { _rdtsc() };
        let secs = t0.elapsed().as_secs_f64();
        (c1 - c0) as f64 / secs
    }

    fn print_clock_state() {
        let chain_len: u64 = 120_000_000;
        let reps = 5;
        let iters = chain_len / 8;
        let ops = iters * 8;

        // Warm up so the first rep is not measuring the ramp out of idle.
        let _ = dependent_add_chain(iters / 4);

        let tsc_hz = calibrate_tsc_hz();
        let tsc_ghz = tsc_hz / 1e9;

        println!(
            "--- CLOCK STATE (bound to this measurement; RDTSC ticks are NOT core cycles) ---"
        );
        println!("    TSC nominal rate        : {tsc_ghz:.4} GHz  (calibrated vs CLOCK_MONOTONIC)");
        println!("    chain length            : {ops} dependent adds x {reps} reps");
        println!("  ");
        println!("    rep       tsc_ticks              ops     core_GHz vs_nominal");

        let mut ghz = Vec::with_capacity(reps);
        for rep in 0..reps {
            // SAFETY: see calibrate_tsc_hz.
            let c0 = unsafe { _rdtsc() };
            let sink = dependent_add_chain(iters);
            // SAFETY: see calibrate_tsc_hz.
            let c1 = unsafe { _rdtsc() };
            std::hint::black_box(sink);

            let ticks = c1 - c0;
            let core_ghz = ops as f64 * tsc_ghz / ticks as f64;
            println!(
                "    {rep}    {ticks:>14}   {ops:>14}     {core_ghz:>6.4}     {:.3}x",
                core_ghz / tsc_ghz
            );
            ghz.push(core_ghz);
        }

        let mean = ghz.iter().sum::<f64>() / ghz.len() as f64;
        let min = ghz.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = ghz.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("  ");
        println!(
            "    mean {mean:.4} GHz | min {min:.4} | max {max:.4} | spread {:.3}%",
            (max - min) / mean * 100.0
        );
        println!(
            "    EFFECTIVE / NOMINAL = {:.3}  <-- the factor that rescales every derived figure",
            mean / tsc_ghz
        );
    }

    // --------------------------------------------------------------------------
    // Fixtures — same generation scheme as ctz_vs_simd.rs.
    // --------------------------------------------------------------------------

    /// Packed 2-bit ternary weights (00=0, 01=+1, 10=-1) at the measured real
    /// BitNet zero fraction (42.21%, ledger A6 full-model scan).
    fn make_packed(zero_frac: f64) -> Vec<u8> {
        let mut s = 0x853C49E6748FEA9Bu64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let zt = (zero_frac * 1000.0) as u64;
        let mut w = vec![0u8; DIM_OUT * DIM_IN / 4];
        for b in w.iter_mut() {
            let mut byte = 0u8;
            for lane in 0..4 {
                let r = next() % 1000;
                let code: u8 = if r < zt {
                    0
                } else if r % 2 == 0 {
                    1
                } else {
                    2
                };
                byte |= code << (2 * lane);
            }
            *b = byte;
        }
        w
    }

    fn bits_identical(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    pub fn run() {
        let reps: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        println!(
            "Bitplane-dense tri-matvec vs incumbent AVX2 LUT+FMA  ({DIM_OUT}x{DIM_IN} ternary)"
        );
        println!(
            "machine           : {}",
            std::env::var("AEGIS_MACHINE").unwrap_or_else(|_| {
                "UNNAMED — set AEGIS_MACHINE or use scripts/capture-bench.sh".into()
            })
        );
        println!("interleaved reps  : {reps} (best-of; >= 3 logged runs required for a verdict)");
        println!();
        print_clock_state();
        println!();

        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("AVX2+FMA not available: nothing to race on this machine");
            return;
        }

        let packed = make_packed(0.4221);
        let x: Vec<f32> = (0..DIM_IN)
            .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
            .collect();

        let words = bitplane_words_per_row(DIM_IN);
        let mut pos = vec![0u64; DIM_OUT * words];
        let mut neg = vec![0u64; DIM_OUT * words];
        let t = Instant::now();
        pack_bitplanes(&packed, DIM_OUT, DIM_IN, &mut pos, &mut neg);
        println!(
            "pack_bitplanes (one-time, at model load): {:.1} ms for this tensor",
            t.elapsed().as_secs_f64() * 1e3
        );
        println!(
            "  layout memory: {} bytes/plane x 2 = {} bytes (vs {} packed) -- both 2 bits/weight",
            DIM_OUT * words * 8,
            DIM_OUT * words * 16,
            packed.len()
        );
        println!();

        let macs = (DIM_OUT * DIM_IN) as f64;
        let mut out_inc = vec![0.0f32; DIM_OUT];
        let mut out_i = vec![0.0f32; DIM_OUT];
        let mut out_ii = vec![0.0f32; DIM_OUT];
        let mut out_ii_ref = vec![0.0f32; DIM_OUT];

        // Bit-exactness gates BEFORE any timing (Rule D: a wrong kernel's
        // throughput is not a result). The incumbent is reached through the
        // engine dispatcher `ternary_matvec` — single-threaded here (this bin
        // builds without the `parallel` feature), so it is exactly
        // `ternary_matvec_avx2` on this path.
        ternary_matvec(&mut out_inc, &x, &packed, DIM_OUT, DIM_IN, 1.0);
        // SAFETY: AVX2+FMA detected above; plane/input/output sizes match the
        // documented contracts.
        unsafe {
            bitplane_matvec_avx2(&mut out_i, &x, &pos, &neg, DIM_OUT, DIM_IN, 1.0);
            bitplane_matvec_avx2_dual(&mut out_ii, &x, &pos, &neg, DIM_OUT, DIM_IN, 1.0);
        }
        bitplane_matvec_scalar_dual(&mut out_ii_ref, &x, &pos, &neg, DIM_OUT, DIM_IN, 1.0);

        let gate_i = bits_identical(&out_i, &out_inc);
        let gate_ii = bits_identical(&out_ii, &out_ii_ref);
        println!("gate: variant(i) byte-identical to incumbent ........ {gate_i}");
        println!("gate: variant(ii) byte-identical to scalar-dual ref .. {gate_ii}");
        assert!(
            gate_i && gate_ii,
            "bit-exactness gate failed; timings would be meaningless"
        );
        println!();

        // Interleaved timing: all three arms alternate inside ONE loop so every
        // arm sees the same turbo/thermal state. Best-of-N (state drifts).
        let (mut t_inc, mut t_i, mut t_ii) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        for _ in 0..reps {
            let t = Instant::now();
            ternary_matvec(&mut out_inc, &x, &packed, DIM_OUT, DIM_IN, 1.0);
            t_inc = t_inc.min(t.elapsed().as_secs_f64());

            // SAFETY: as above.
            let t = Instant::now();
            unsafe { bitplane_matvec_avx2(&mut out_i, &x, &pos, &neg, DIM_OUT, DIM_IN, 1.0) };
            t_i = t_i.min(t.elapsed().as_secs_f64());

            // SAFETY: as above.
            let t = Instant::now();
            unsafe { bitplane_matvec_avx2_dual(&mut out_ii, &x, &pos, &neg, DIM_OUT, DIM_IN, 1.0) };
            t_ii = t_ii.min(t.elapsed().as_secs_f64());
        }
        std::hint::black_box((&out_inc, &out_i, &out_ii));

        println!(
            "incumbent LUT+FMA matvec : {:7.2} ms  ({:5.2} GMAC/s)  [engine decode kernel]",
            t_inc * 1e3,
            macs / t_inc / 1e9
        );
        println!(
            "bitplane (i) ordered     : {:7.2} ms  ({:5.2} GMAC/s)  = {:.2}x incumbent",
            t_i * 1e3,
            macs / t_i / 1e9,
            t_inc / t_i
        );
        println!(
            "bitplane (ii) dual-acc   : {:7.2} ms  ({:5.2} GMAC/s)  = {:.2}x incumbent",
            t_ii * 1e3,
            macs / t_ii / 1e9,
            t_inc / t_ii
        );

        // Re-assert AFTER timing so a torn/overwritten buffer can't hide a
        // miscompare that the pre-gate happened to miss.
        println!(
            "post-timing gates hold: {}",
            bits_identical(&out_i, &out_inc) && bits_identical(&out_ii, &out_ii_ref)
        );
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "scalar_only")))]
fn main() {
    x86::run()
}
