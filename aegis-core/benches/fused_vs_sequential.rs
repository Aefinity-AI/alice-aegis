//! Fused dual/tri matvec vs sequential incumbent calls — same-binary A/B.
//!
//! Question: SwiGLU's gate_proj+up_proj (dual) and attention's Q/K/V (tri)
//! share one input vector. Does fusing them — one pass interleaving 4-row
//! blocks from every matrix so each 8-lane input load feeds M*4 FMA chains —
//! beat back-to-back calls of the incumbent `ternary_matvec_avx2`?
//!
//! Honest prior: SMALL GAIN AT BEST. The input vector is KB-scale and
//! cache-resident; the weights dominate memory traffic and their traffic is
//! unchanged by construction. This bench exists so the recurring "dual
//! matvec" idea gets a measured answer instead of a shrug.
//!
//! Methodology (per docs/hardware_logs/lut_mpgemm_ab_findings_2026-07-30.md):
//! kernel A/Bs on this box are BIMODAL — non-interleaved comparisons are
//! inadmissible. Reps strictly alternate the two variants and swap which one
//! goes first on every rep; all per-rep numbers are printed, not just a
//! summary. A clock-state block (TSC nominal + effective/nominal core-clock
//! ratio) heads the output because RDTSC ticks are NOT core cycles (Rule A
//! corollary).
//!
//! RULES A/B REMINDER: numbers printed by an ad-hoc run of this binary are
//! NOT results. A result is a capture via scripts/capture-bench.sh on an
//! idle, NAMED physical machine, logged under docs/hardware_logs/.
//!
//! Run: cargo run --release --bin fused_vs_sequential [reps]   (default 7)

// `cargo test` auto-builds bin targets on every architecture so integration
// tests can exec them; this bench is x86-only, so non-x86 gets a stub main.
// It also needs the AVX2 kernels this bench races, which are cfg'd out under
// `scalar_only` — so that build gets the same stub.
#[cfg(any(not(target_arch = "x86_64"), feature = "scalar_only"))]
fn main() {
    eprintln!(
        "x86_64 AVX2-only benchmark; nothing to run on this architecture/feature set"
    );
}

#[cfg(all(target_arch = "x86_64", not(feature = "scalar_only")))]
mod x86 {

    use aegis_core::ops::{
        ternary_matvec_avx2, ternary_matvec_fused2_avx2, ternary_matvec_fused3_avx2,
    };
    use std::arch::x86_64::_rdtsc;
    use std::time::Instant;

    // ---------------------------------------------------------------------------
    // Clock state (methodology of aegis-linux/examples/clockstate.rs, embedded so
    // this binary is self-contained when run outside capture-bench.sh).
    // ---------------------------------------------------------------------------

    /// `n` iterations of an 8-deep dependent add chain: retires exactly one add
    /// per core cycle, so the retired-op count is a direct core-cycle count.
    fn dependent_add_chain(n: u64) -> u64 {
        let mut x: u64 = 1;
        let mut i: u64 = n;
        // SAFETY: pure register arithmetic on locals; no memory operands, no
        // stack use (`nostack`), and both registers are declared inout so the
        // compiler cannot assume their values survive the block.
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

    /// TSC ticks/second calibrated against CLOCK_MONOTONIC over ~200 ms.
    fn calibrate_tsc_hz() -> f64 {
        let t0 = Instant::now();
        // SAFETY: rdtsc is unprivileged on x86_64, reads a counter, no side
        // effects.
        let c0 = unsafe { _rdtsc() };
        while t0.elapsed().as_millis() < 200 {
            std::hint::spin_loop();
        }
        // SAFETY: as above.
        let c1 = unsafe { _rdtsc() };
        (c1 - c0) as f64 / t0.elapsed().as_secs_f64()
    }

    fn print_clock_state() {
        let iters: u64 = 120_000_000 / 8;
        let ops = iters * 8;
        let reps = 5;
        let _ = dependent_add_chain(iters / 4); // leave the idle P-state first

        let tsc_ghz = calibrate_tsc_hz() / 1e9;
        println!("--- CLOCK STATE (bound to this run; RDTSC ticks are NOT core cycles) ---");
        println!("    TSC nominal rate        : {tsc_ghz:.4} GHz  (calibrated vs CLOCK_MONOTONIC)");
        println!("    chain length            : {ops} dependent adds x {reps} reps");
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
        println!(
            "    mean {mean:.4} GHz | EFFECTIVE / NOMINAL = {:.3}  <-- rescales every tick-derived figure",
            mean / tsc_ghz
        );
        println!();
    }

    // ---------------------------------------------------------------------------
    // Data generation (42.21% zeros = the full-scan BitNet b1.58-2B weight
    // sparsity from the A6 ledger entry; value distribution is irrelevant to this
    // kernel's timing but kept representative anyway).
    // ---------------------------------------------------------------------------

    fn make_packed(dim_out: usize, dim_in: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut w = vec![0u8; dim_out * dim_in / 4];
        for b in w.iter_mut() {
            let mut byte = 0u8;
            for lane in 0..4 {
                let r = next() % 10000;
                let code: u8 = if r < 4221 {
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

    fn make_input(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 251) as f32 - 125.0) / 125.0).collect()
    }

    fn bits_identical(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    // ---------------------------------------------------------------------------
    // Scenarios
    // ---------------------------------------------------------------------------

    struct Matrices {
        dims_out: Vec<usize>,
        dim_in: usize,
        weights: Vec<Vec<u8>>,
        scales: Vec<f32>,
    }

    impl Matrices {
        fn new(dims_out: &[usize], dim_in: usize) -> Self {
            let weights = dims_out
                .iter()
                .enumerate()
                .map(|(i, &d)| make_packed(d, dim_in, 0x9E3779B97F4A7C15 ^ (i as u64) << 17))
                .collect();
            let scales = dims_out
                .iter()
                .enumerate()
                .map(|(i, _)| 0.0123 + 0.011 * i as f32)
                .collect();
            Matrices {
                dims_out: dims_out.to_vec(),
                dim_in,
                weights,
                scales,
            }
        }

        fn macs(&self) -> f64 {
            self.dims_out
                .iter()
                .map(|&d| (d * self.dim_in) as f64)
                .sum()
        }
    }

    /// # Safety
    /// Caller must have verified AVX2+FMA via `is_x86_feature_detected!`.
    unsafe fn run_sequential(m: &Matrices, input: &[f32], outs: &mut [Vec<f32>]) {
        for (i, out) in outs.iter_mut().enumerate() {
            // SAFETY: caller verified AVX2+FMA; buffers sized by construction.
            unsafe {
                ternary_matvec_avx2(
                    out,
                    input,
                    &m.weights[i],
                    m.dims_out[i],
                    m.dim_in,
                    m.scales[i],
                );
            }
        }
    }

    /// # Safety
    /// Caller must have verified AVX2+FMA via `is_x86_feature_detected!`.
    unsafe fn run_fused(m: &Matrices, input: &[f32], outs: &mut [Vec<f32>]) {
        match m.dims_out.len() {
            2 => {
                let (a, b) = outs.split_at_mut(1);
                // SAFETY: caller verified AVX2+FMA; buffers sized by construction.
                unsafe {
                    ternary_matvec_fused2_avx2(
                        &mut a[0],
                        &mut b[0],
                        input,
                        &m.weights[0],
                        &m.weights[1],
                        m.dims_out[0],
                        m.dims_out[1],
                        m.dim_in,
                        m.scales[0],
                        m.scales[1],
                    );
                }
            }
            3 => {
                let (a, rest) = outs.split_at_mut(1);
                let (b, c) = rest.split_at_mut(1);
                // SAFETY: caller verified AVX2+FMA; buffers sized by construction.
                unsafe {
                    ternary_matvec_fused3_avx2(
                        &mut a[0],
                        &mut b[0],
                        &mut c[0],
                        input,
                        &m.weights[0],
                        &m.weights[1],
                        &m.weights[2],
                        m.dims_out[0],
                        m.dims_out[1],
                        m.dims_out[2],
                        m.dim_in,
                        m.scales[0],
                        m.scales[1],
                        m.scales[2],
                    );
                }
            }
            _ => unreachable!("scenarios are dual or tri"),
        }
    }

    fn run_scenario(name: &str, dims_out: &[usize], dim_in: usize, reps: usize) {
        let m = Matrices::new(dims_out, dim_in);
        let input = make_input(dim_in);
        let macs = m.macs();

        println!(
            "--- {name}: outs {dims_out:?} x in {dim_in}  ({:.1} MMAC/pass)",
            macs / 1e6
        );

        let mut outs_seq: Vec<Vec<f32>> = dims_out.iter().map(|&d| vec![0.0; d]).collect();
        let mut outs_fus: Vec<Vec<f32>> = dims_out.iter().map(|&d| vec![0.0; d]).collect();

        // Correctness gate before any timing: byte-identity is the contract.
        // SAFETY: main() verified AVX2+FMA before calling run_scenario.
        unsafe {
            run_sequential(&m, &input, &mut outs_seq);
            run_fused(&m, &input, &mut outs_fus);
        }
        let identical = outs_seq
            .iter()
            .zip(outs_fus.iter())
            .all(|(a, b)| bits_identical(a, b));
        println!("  outputs byte-identical: {identical}");
        if !identical {
            println!("  ABORTING scenario: fused kernel is WRONG; timing it would be meaningless.");
            return;
        }

        // Interleaved A/B, alternating which variant runs first each rep.
        let mut t_seq = Vec::with_capacity(reps);
        let mut t_fus = Vec::with_capacity(reps);
        println!("  rep   order        sequential_ms   fused_ms   fused/seq");
        for rep in 0..reps {
            let (a, b);
            // SAFETY: main() verified AVX2+FMA.
            unsafe {
                if rep % 2 == 0 {
                    let t = Instant::now();
                    run_sequential(&m, &input, &mut outs_seq);
                    a = t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    run_fused(&m, &input, &mut outs_fus);
                    b = t.elapsed().as_secs_f64();
                } else {
                    let t = Instant::now();
                    run_fused(&m, &input, &mut outs_fus);
                    b = t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    run_sequential(&m, &input, &mut outs_seq);
                    a = t.elapsed().as_secs_f64();
                }
            }
            std::hint::black_box((&outs_seq, &outs_fus));
            t_seq.push(a);
            t_fus.push(b);
            println!(
                "  {rep:>3}   {}   {:>12.3}   {:>8.3}   {:>8.3}",
                if rep % 2 == 0 {
                    "seq-first "
                } else {
                    "fused-first"
                },
                a * 1e3,
                b * 1e3,
                b / a
            );
        }

        let min_seq = t_seq.iter().cloned().fold(f64::INFINITY, f64::min);
        let min_fus = t_fus.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "  best: sequential {:.3} ms ({:.2} GMAC/s) | fused {:.3} ms ({:.2} GMAC/s) | fused/seq {:.3}",
            min_seq * 1e3,
            macs / min_seq / 1e9,
            min_fus * 1e3,
            macs / min_fus / 1e9,
            min_fus / min_seq
        );
        println!();
    }

    pub fn run() {
        let reps: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);

        println!("Fused dual/tri matvec vs sequential incumbent — same-binary interleaved A/B");
        println!(
            "NOT A RESULT unless captured via scripts/capture-bench.sh on an idle, named machine."
        );
        println!(
            "machine: {}",
            std::env::var("AEGIS_MACHINE").unwrap_or_else(|_| {
                "UNNAMED — set AEGIS_MACHINE or use scripts/capture-bench.sh".into()
            })
        );
        println!();

        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            println!("AVX2+FMA not available: the fused kernels have no scalar variant to race.");
            return;
        }

        print_clock_state();

        // BitNet b1.58-2B shapes: hidden 2560, intermediate 6912, GQA K/V rows 640.
        run_scenario("DUAL BitNet-2B SwiGLU gate+up", &[6912, 6912], 2560, reps);
        run_scenario("TRI  BitNet-2B Q/K/V (GQA)", &[2560, 640, 640], 2560, reps);
        // M7 shapes: hidden 384, intermediate 1024.
        run_scenario("DUAL M7 SwiGLU gate+up", &[1024, 1024], 384, reps);
        run_scenario("TRI  M7 Q/K/V", &[384, 384, 384], 384, reps);

        println!("Reminder: single-thread kernel A/B; pool-split fusion is a separate question.");
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "scalar_only")))]
fn main() {
    x86::run()
}
