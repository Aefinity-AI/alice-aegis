//! Same-binary interleaved A/B: incumbent AVX2 LUT+FMA ternary matvec vs the
//! ReLU^2 activation column-skip candidates (`ops_colskip`, variants
//! "ordered" — byte-identical to the incumbent — and "chain" — pinned to its
//! scalar mirror). Methodology copied from the 2026-07-30 LUT-mpGEMM
//! same-binary A/B (docs/hardware_logs/lut_mpgemm_ab_findings_2026-07-30.md):
//! kernel A/Bs on the dev box are BIMODAL, so non-interleaved comparisons
//! are inadmissible — every arm runs in this ONE binary, alternating inside
//! one rep loop, on the same packed weights, and a verdict needs >= 3
//! logged runs.
//!
//! WHY the inputs matter more than usual: the candidate's entire mechanism
//! is skipping columns whose input element is exactly +/-0.0, so its time is
//! a direct function of the input's zero pattern. Synthetic uniform-random
//! zeros under-model the clustering of real ReLU^2 zeros. The PRIMARY
//! scenario therefore loads REAL down_proj input vectors captured from a
//! live BitNet-2B run (aegis-linux/examples/act_capture.rs, AEGISAV1 file);
//! a synthetic z-sweep (z in 0.0/0.5/0.789/0.9) is included only for the
//! SHAPE of the curve. z = 0.7888 is the measured pooled decode mean
//! (ledger A15, raw log docs/hardware_logs/
//! relu2_act_sparsity_bitnet2b_2026-08-01.log).
//!
//! This candidate is NOT a settled negative: not CTZ zero-mult (A6 — weight
//! sparsity, bit-scan inner loop), not pshufb LUT-mpGEMM (A7 — stays 2
//! bits/weight; traffic here is the incumbent's times (1 - z)), not fused
//! dual/tri (A16), not bitplane-dense (A17).
//!
//! GMAC/s figures use NOMINAL MACs (dim_out x dim_in) for every arm, so a
//! skip arm's higher GMAC/s is exactly its time ratio vs the incumbent —
//! skipped work is counted as done, never as extra throughput.
//!
//! Rule A: numbers printed here are only admissible when captured on
//! physical hardware, machine named, with the clock-state block below bound
//! to the capture (RDTSC ticks are NOT core cycles; the effective/nominal
//! ratio rescales every tick-derived figure). Rule B: a figure enters the
//! ledger only via a raw log under docs/hardware_logs/. Timings from the
//! contended dev box are smoke only.
//!
//! Run: cargo run --release --bin colskip_vs_incumbent [reps] [act_file] [max_real]
//!   reps      interleaved best-of reps           (default 5)
//!   act_file  AEGISAV1 capture, or "-" to skip   (default: artifacts path, see below)
//!   max_real  cap on real vectors, stride-sampled (default 96)

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

    use aegis_core::ops::ternary_matvec;
    use aegis_core::ops_colskip::{
        colskip_col_bytes, colskip_covered_cols, colskip_matvec_avx2_chain,
        colskip_matvec_avx2_ordered, colskip_matvec_scalar_chain, pack_colmajor,
    };
    use std::arch::x86_64::_rdtsc;
    use std::time::Instant;

    const DIM_OUT: usize = 2560;
    const DIM_IN: usize = 6912; // BitNet down_proj, the decode-dominant shape

    const DEFAULT_ACT_FILES: [&str; 3] = [
        "artifacts/relu2_down_in_bitnet2b_2026-08-01.av1",
        "../artifacts/relu2_down_in_bitnet2b_2026-08-01.av1",
        "../../artifacts/relu2_down_in_bitnet2b_2026-08-01.av1",
    ];

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
    // Fixtures — weight generation as in bitplane_vs_lut.rs; inputs are either
    // REAL captured vectors or synthetic z-controlled vectors.
    // --------------------------------------------------------------------------

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Packed 2-bit ternary weights (00=0, 01=+1, 10=-1) at the measured real
    /// BitNet zero fraction (42.21%, ledger A6 full-model scan). The WEIGHTS are
    /// synthetic (the repo ships no model tensors); the candidate's mechanism
    /// depends on the INPUT zero pattern, which is real in the primary scenario.
    fn make_packed(zero_frac: f64) -> Vec<u8> {
        let mut rng = Rng(0x853C_49E6_748F_EA9B);
        let zt = (zero_frac * 1000.0) as u64;
        let mut w = vec![0u8; DIM_OUT * DIM_IN / 4];
        for b in w.iter_mut() {
            let mut byte = 0u8;
            for lane in 0..4 {
                let r = rng.next() % 1000;
                let code: u8 = if r < zt {
                    0
                } else if r.is_multiple_of(2) {
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

    /// Synthetic input at exact-zero fraction `z`; alternate zeros are bitwise
    /// -0.0 (the A15 caveat), values on the same grid as the exactness tests.
    fn make_synthetic_input(z: f64, seed: u64) -> Vec<f32> {
        let mut rng = Rng(seed);
        let zt = (z * 10_000.0) as u64;
        let mut flip = false;
        (0..DIM_IN)
            .map(|_| {
                let r = rng.next();
                if r % 10_000 < zt {
                    flip = !flip;
                    if flip { -0.0 } else { 0.0 }
                } else {
                    ((r % 4001) as f32 - 2000.0) / 512.0
                }
            })
            .collect()
    }

    fn zero_frac(v: &[f32]) -> f64 {
        v.iter().filter(|x| **x == 0.0).count() as f64 / v.len() as f64
    }

    fn bits_identical(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Load an AEGISAV1 capture (see aegis-linux/examples/act_capture.rs),
    /// stride-sampled down to at most `max_vecs` vectors so every layer band of
    /// the capture stays represented.
    fn load_real_vectors(path: &str, max_vecs: usize) -> Result<Vec<Vec<f32>>, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if bytes.len() < 16 || &bytes[0..8] != b"AEGISAV1" {
            return Err(format!("{path}: not an AEGISAV1 capture"));
        }
        let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if dim != DIM_IN {
            return Err(format!("{path}: dim {dim} != bench DIM_IN {DIM_IN}"));
        }
        let rec_bytes = 8 + dim * 4;
        if bytes.len() < 16 + count * rec_bytes {
            return Err(format!("{path}: truncated ({} bytes)", bytes.len()));
        }
        let stride = count.div_ceil(max_vecs.max(1)).max(1);
        let mut out = Vec::new();
        let mut i = 0;
        while i < count && out.len() < max_vecs {
            let base = 16 + i * rec_bytes + 8; // skip layer + token_ordinal
            let v: Vec<f32> = (0..dim)
                .map(|j| {
                    let o = base + j * 4;
                    f32::from_bits(u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()))
                })
                .collect();
            out.push(v);
            i += stride;
        }
        Ok(out)
    }

    struct Arms<'a> {
        packed: &'a [u8],
        colmajor: &'a [u8],
        out_inc: Vec<f32>,
        out_ord: Vec<f32>,
        out_chn: Vec<f32>,
        scratch: Vec<f32>,
    }

    impl<'a> Arms<'a> {
        fn new(packed: &'a [u8], colmajor: &'a [u8]) -> Self {
            Self {
                packed,
                colmajor,
                out_inc: vec![0.0; DIM_OUT],
                out_ord: vec![0.0; DIM_OUT],
                out_chn: vec![0.0; DIM_OUT],
                scratch: vec![0.0; 8 * DIM_OUT],
            }
        }

        fn run_incumbent(&mut self, x: &[f32]) {
            // Engine dispatcher; this bin builds without `parallel`, so this is
            // exactly ternary_matvec_avx2 on this path.
            ternary_matvec(&mut self.out_inc, x, self.packed, DIM_OUT, DIM_IN, 1.0);
        }
        fn run_ordered(&mut self, x: &[f32]) {
            // SAFETY: AVX2+FMA detected in main before any Arms call; colmajor
            // holds covered*col_bytes bytes; outputs DIM_OUT; scratch 8*DIM_OUT.
            unsafe {
                colskip_matvec_avx2_ordered(
                    &mut self.out_ord,
                    x,
                    self.colmajor,
                    DIM_OUT,
                    DIM_IN,
                    1.0,
                    &mut self.scratch,
                );
            }
        }
        fn run_chain(&mut self, x: &[f32]) {
            // SAFETY: as run_ordered (chain needs no scratch).
            unsafe {
                colskip_matvec_avx2_chain(
                    &mut self.out_chn,
                    x,
                    self.colmajor,
                    DIM_OUT,
                    DIM_IN,
                    1.0,
                );
            }
        }

        /// Bit-exactness gates for one input vector (Rule D: a wrong kernel's
        /// throughput is not a result). Ordered must equal the incumbent;
        /// chain must equal its non-skipping scalar mirror.
        fn gate(&mut self, x: &[f32], what: &str) {
            self.run_incumbent(x);
            self.run_ordered(x);
            self.run_chain(x);
            let mut chain_ref = vec![0.0f32; DIM_OUT];
            colskip_matvec_scalar_chain(&mut chain_ref, x, self.colmajor, DIM_OUT, DIM_IN, 1.0);
            assert!(
                bits_identical(&self.out_ord, &self.out_inc),
                "{what}: ordered vs incumbent bit-exactness gate FAILED"
            );
            assert!(
                bits_identical(&self.out_chn, &chain_ref),
                "{what}: chain vs scalar-mirror bit-exactness gate FAILED"
            );
        }
    }

    fn report(label: &str, t_inc: f64, t_ord: f64, t_chn: f64, n: f64) {
        let macs = (DIM_OUT * DIM_IN) as f64 * n;
        println!(
            "  incumbent LUT+FMA        : {:9.3} ms  ({:5.2} GMAC/s nominal)  [engine decode kernel]",
            t_inc * 1e3,
            macs / t_inc / 1e9
        );
        println!(
            "  colskip ordered (=inc)   : {:9.3} ms  ({:5.2} GMAC/s nominal)  = {:.2}x incumbent",
            t_ord * 1e3,
            macs / t_ord / 1e9,
            t_inc / t_ord
        );
        println!(
            "  colskip chain (mirror-pin): {:8.3} ms  ({:5.2} GMAC/s nominal)  = {:.2}x incumbent",
            t_chn * 1e3,
            macs / t_chn / 1e9,
            t_inc / t_chn
        );
        let _ = label;
    }

    pub fn run() {
        let reps: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let act_file_arg = std::env::args().nth(2);
        let max_real: usize = std::env::args()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(96);

        println!(
            "ReLU^2 column-skip matvec vs incumbent AVX2 LUT+FMA  ({DIM_OUT}x{DIM_IN} ternary, down_proj shape)"
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
        let cb = colskip_col_bytes(DIM_OUT);
        let covered = colskip_covered_cols(DIM_IN);
        let mut colmajor = vec![0u8; covered * cb];
        let t = Instant::now();
        pack_colmajor(&packed, DIM_OUT, DIM_IN, &mut colmajor);
        println!(
            "pack_colmajor (one-time, at model load): {:.1} ms for this tensor",
            t.elapsed().as_secs_f64() * 1e3
        );
        println!(
            "  layout memory: {} bytes column-major (vs {} row-major) -- both 2 bits/weight;",
            colmajor.len(),
            packed.len()
        );
        println!("  a skipped column skips {cb} contiguous bytes of weight traffic");
        println!();

        let mut arms = Arms::new(&packed, &colmajor);

        // ----------------------------------------------------------------------
        // PRIMARY scenario: real captured down_proj input vectors.
        // ----------------------------------------------------------------------
        let real = match act_file_arg.as_deref() {
            Some("-") => Err("skipped by request ('-')".to_string()),
            Some(p) => load_real_vectors(p, max_real),
            None => {
                let mut r = Err("no capture file found".to_string());
                for p in DEFAULT_ACT_FILES {
                    r = load_real_vectors(p, max_real);
                    if r.is_ok() {
                        println!("real capture       : {p}");
                        break;
                    }
                }
                r
            }
        };

        match real {
            Ok(vectors) => {
                let mean_z =
                    vectors.iter().map(|v| zero_frac(v)).sum::<f64>() / vectors.len() as f64;
                println!(
                    "=== REAL vectors: {} captured down_proj inputs, mean z = {:.4} (A15 pooled mean was 0.7888) ===",
                    vectors.len(),
                    mean_z
                );

                // Pre-timing gates on EVERY vector that will be timed.
                for (i, v) in vectors.iter().enumerate() {
                    arms.gate(v, &format!("real vector {i}"));
                }
                println!(
                    "gate: ordered==incumbent and chain==mirror on all {} vectors .. true",
                    vectors.len()
                );

                // Interleaved timing: arms alternate per vector inside one sweep;
                // best-of-N on the sweep totals (state drifts between reps).
                let (mut t_inc, mut t_ord, mut t_chn) =
                    (f64::INFINITY, f64::INFINITY, f64::INFINITY);
                for _ in 0..reps {
                    let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
                    for v in &vectors {
                        let t = Instant::now();
                        arms.run_incumbent(v);
                        a += t.elapsed().as_secs_f64();
                        let t = Instant::now();
                        arms.run_ordered(v);
                        b += t.elapsed().as_secs_f64();
                        let t = Instant::now();
                        arms.run_chain(v);
                        c += t.elapsed().as_secs_f64();
                    }
                    t_inc = t_inc.min(a);
                    t_ord = t_ord.min(b);
                    t_chn = t_chn.min(c);
                }
                std::hint::black_box((&arms.out_inc, &arms.out_ord, &arms.out_chn));
                println!("  (totals over {} matvecs, best-of-{reps})", vectors.len());
                report("real", t_inc, t_ord, t_chn, vectors.len() as f64);

                // Post-timing gate on the last vector: a torn buffer can't hide.
                if let Some(last) = vectors.last() {
                    arms.gate(last, "real post-timing");
                    println!("  post-timing gates hold: true");
                }
                println!();
            }
            Err(e) => {
                println!("=== REAL vectors: SKIPPED ({e}) ===");
                println!("  capture one with: cargo run --release --features act_stats \\");
                println!(
                    "    --example act_capture -p aegis-linux -- <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <out.av1>"
                );
                println!(
                    "  A verdict without the real-vector scenario is NOT admissible for adoption."
                );
                println!();
            }
        }

        // ----------------------------------------------------------------------
        // Synthetic z sweep — curve shape only. Uniform-random zero placement
        // under-models real clustering; adoption rides the REAL scenario.
        // ----------------------------------------------------------------------
        for (si, &z) in [0.0f64, 0.5, 0.789, 0.9].iter().enumerate() {
            let x = make_synthetic_input(z, 0x1357_9BDF_2468_ACE0 ^ (si as u64) << 32);
            println!(
                "=== SYNTHETIC z = {z:.3} (actual {:.4}, uniform placement — curve shape only) ===",
                zero_frac(&x)
            );
            arms.gate(&x, "synthetic pre");
            println!("gate: ordered==incumbent and chain==mirror ........... true");

            let (mut t_inc, mut t_ord, mut t_chn) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
            for _ in 0..reps {
                let t = Instant::now();
                arms.run_incumbent(&x);
                t_inc = t_inc.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                arms.run_ordered(&x);
                t_ord = t_ord.min(t.elapsed().as_secs_f64());
                let t = Instant::now();
                arms.run_chain(&x);
                t_chn = t_chn.min(t.elapsed().as_secs_f64());
            }
            std::hint::black_box((&arms.out_inc, &arms.out_ord, &arms.out_chn));
            report("synthetic", t_inc, t_ord, t_chn, 1.0);
            arms.gate(&x, "synthetic post");
            println!("  post-timing gates hold: true");
            println!();
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "scalar_only")))]
fn main() {
    x86::run()
}
