//! Same-binary interleaved A/B: what does CIS-1's integer semantics actually
//! COST in throughput?
//!
//! WHY THIS EXISTS. As of 2026-08-05 the ledger carries CIS-1's *quality* cost
//! (A20 +0.0637%, A21 +0.7408% perplexity) and its *portability* proof (A25,
//! one digest across four codegen paths) — but **not one of the 97 files in
//! `docs/hardware_logs/` contains a CIS-1 throughput measurement**. The only
//! timing signal that exists anywhere is the bracketed wall-time in
//! `cis1_e2_bitnet2b_int_vs_float_ppl_i5-10210U_crosvm_2026-08-01.log`
//! (float 62.9s vs integer 370.1s), and line 3 of that very file disclaims it:
//! *"Rule A: no timing in this file is a result; bracketed wall-times are
//! incidental."* A strategy review then quoted those disclaimed numbers back as
//! a 5.9-6.6x slowdown. Both the claim "near-zero overhead" and the claim
//! "6x slower" are currently unsupported. This binary exists to replace both
//! with a measurement.
//!
//! THE TRAP THIS BENCH IS DESIGNED TO AVOID. `cis::ternary_matvec_i8` is a
//! plain scalar reference loop; `ops::ternary_matvec` is the hand-written AVX2
//! LUT+FMA kernel. Racing those two directly measures **scalar-vs-SIMD**, not
//! **integer-vs-float**, and would attribute the entire AVX2 speedup to the
//! integer semantics. That comparison is reported below, but it is NOT the
//! answer and is labelled accordingly.
//!
//! THE DESIGN. Three arms over identical packed weights and identical logical
//! input values, interleaved inside one rep loop:
//!
//!   A  float  AVX2    ops::ternary_matvec                    (incumbent)
//!   B  float  scalar  ops::ternary_matvec + set_force_scalar (SIMD held OFF)
//!   C  int    scalar  cis::ternary_matvec_i8                 (CIS-1 reference)
//!   D  int    AVX2    cis_avx2::ternary_matvec_i8_avx2       (the new kernel)
//!
//! D/A is now THE answer: determinism's cost with both paths vectorised. D is
//! asserted byte-identical to C every run; a mismatch voids the capture.
//!
//! Two ratios carry all the meaning:
//!
//!   C/B  THE ANSWER — cost of integer semantics with SIMD held constant.
//!        Both arms are scalar, same packing, same loop shape. Whatever this
//!        is, it is what determinism costs at the kernel today.
//!   A/B  the SIMD headroom on the float path — i.e. roughly what an AVX2
//!        integer kernel would have to recover to reach parity. It is NOT a
//!        CIS-1 result; it is the size of the prize for writing one.
//!
//! C/A is printed for completeness and is explicitly NOT the determinism cost.
//!
//! Methodology copied from the 2026-07-30 LUT-mpGEMM same-binary A/B
//! (`docs/hardware_logs/lut_mpgemm_ab_findings_2026-07-30.md`): kernel A/Bs on
//! the dev box are BIMODAL, so non-interleaved comparisons are inadmissible.
//! Every arm runs in this ONE binary, alternating inside one rep loop, on the
//! same packed weights; a verdict needs >= 3 logged runs.
//!
//! GMAC/s uses NOMINAL MACs (dim_out x dim_in) for every arm, so a ratio of
//! GMAC/s is exactly the inverse ratio of time. No arm can bank skipped work.
//!
//! Rule A: numbers printed here are admissible only when captured on physical
//! hardware, machine named, with the clock-state block below bound to the
//! capture (RDTSC ticks are NOT core cycles; the effective/nominal ratio
//! rescales every tick-derived figure). Timings from the contended crosvm dev
//! box are SMOKE ONLY and must not enter the ledger. Rule B: a figure enters
//! the ledger only via a raw log under `docs/hardware_logs/`.
//!
//! Rule D: this bench also asserts the integer arm is bit-exact across reps
//! (accumulator digest) and that it agrees with the float arm to within
//! quantization tolerance. A timing run whose digests disagree is void.
//!
//! Run: cargo run --release --bin cis_vs_float [reps]
//!   reps   interleaved best-of reps (default 5)

// `cargo test` auto-builds bin targets on every architecture so integration
// tests can exec them; this bench is x86-only, so non-x86 gets a stub main.
#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("x86_64-only benchmark; nothing to run on this architecture");
}

#[cfg(target_arch = "x86_64")]
mod x86 {

    use aegis_core::cis::ternary_matvec_i8;
    use aegis_core::cis_avx2::ternary_matvec_i8_avx2;
    use aegis_core::ops::{set_force_scalar, ternary_matvec};
    use std::arch::x86_64::_rdtsc;
    use std::hint::black_box;
    use std::time::Instant;

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
        // observable side effects; every register touched is declared inout so the
        // compiler cannot assume a value survives the block. `nostack` holds
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

    fn calibrate_tsc_hz() -> f64 {
        let t0 = Instant::now();
        // SAFETY: rdtsc is unprivileged and always available on x86_64; it reads a
        // counter and has no side effects.
        let c0 = unsafe { _rdtsc() };
        while t0.elapsed().as_millis() < 200 {
            std::hint::spin_loop();
        }
        // SAFETY: as above.
        let c1 = unsafe { _rdtsc() };
        let secs = t0.elapsed().as_secs_f64();
        (c1 - c0) as f64 / secs
    }

    fn print_clock_state() -> f64 {
        let chain_len: u64 = 120_000_000;
        let reps = 5;
        let iters = chain_len / 8;
        let ops = iters * 8;

        let _ = dependent_add_chain(iters / 4); // warm out of idle

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
            black_box(sink);

            let ticks = c1 - c0;
            let core_ghz = ops as f64 * tsc_ghz / ticks as f64;
            println!(
                "    {rep}    {ticks:>14}   {ops:>14}     {core_ghz:>6.4}     {:.3}x",
                core_ghz / tsc_ghz
            );
            ghz.push(core_ghz);
        }
        let med = {
            let mut v = ghz.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        println!("  ");
        println!(
            "    median core clock       : {med:.4} GHz   effective/nominal = {:.3}x",
            med / tsc_ghz
        );
        println!("    (every tick-derived figure below must carry this ratio)");
        println!();
        med / tsc_ghz
    }

    // --------------------------------------------------------------------------
    // Inputs
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

    /// Packed 2-bit ternary weights, matching BOTH kernels' documented layout:
    /// row-major, 4 weights per byte, low bit-pair first, 00=0 01=+1 10=-1.
    fn pack_weights(dim_out: usize, dim_in: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut out = vec![0u8; dim_out * dim_in / 4];
        for byte in out.iter_mut() {
            let mut b = 0u8;
            for k in 0..4 {
                // ~42% zeros, matching the measured BitNet-2B ternary density
                // (ledger A6: 42.21% of real weights are zero).
                let r = rng.next() % 100;
                let code: u8 = if r < 42 {
                    0b00
                } else if r < 71 {
                    0b01
                } else {
                    0b10
                };
                b |= code << (2 * k);
            }
            *byte = b;
        }
        out
    }

    /// One input vector in both representations. The i8 codes ARE the logical
    /// values; the f32 arm sees exactly the same numbers widened, so neither arm
    /// is handed an easier input distribution.
    fn make_input(dim_in: usize, seed: u64) -> (Vec<i8>, Vec<f32>) {
        let mut rng = Rng(seed);
        let mut qi = vec![0i8; dim_in];
        let mut qf = vec![0f32; dim_in];
        for i in 0..dim_in {
            let v = ((rng.next() % 255) as i32 - 127) as i8; // full i8 range
            qi[i] = v;
            qf[i] = v as f32;
        }
        (qi, qf)
    }

    fn fnv1a64(mut h: u64, bytes: &[u8]) -> u64 {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
        h
    }

    // --------------------------------------------------------------------------

    struct Shape {
        name: &'static str,
        dim_out: usize,
        dim_in: usize,
    }

    pub fn run() {
        let reps: usize = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        println!("=== CIS-1 INTEGER SEMANTICS: THROUGHPUT COST (same-binary interleaved A/B) ===");
        println!();
        println!("  QUESTION: what does CIS-1 cost in throughput? Never measured before this run.");
        println!("  ARMS:  A float/AVX2   B float/scalar   C int/scalar (ref)   D int/AVX2 (new)");
        println!(
            "  C/B is the answer (SIMD held constant). A/B is the prize for an AVX2 int kernel."
        );
        println!(
            "  C/A is NOT the determinism cost: it conflates scalar-vs-SIMD with int-vs-float."
        );
        println!("  reps (interleaved, best-of): {reps}");
        println!();

        let ratio = print_clock_state();
        if (ratio - 1.0).abs() > 0.10 {
            println!(
                "  !! effective/nominal is {ratio:.3}x — the box is not at nominal clock. Every"
            );
            println!(
                "     tick-derived figure below is scaled by it. Re-read Rule A before quoting."
            );
            println!();
        }

        // Real BitNet-2B shapes. down_proj is the decode-time hot path (A17/A23).
        let shapes = [
            Shape {
                name: "down_proj  2560x6912",
                dim_out: 2560,
                dim_in: 6912,
            },
            Shape {
                name: "gate/up    6912x2560",
                dim_out: 6912,
                dim_in: 2560,
            },
            Shape {
                name: "attn qkv   2560x2560",
                dim_out: 2560,
                dim_in: 2560,
            },
        ];

        println!("--- PER-SHAPE RESULTS ---");
        println!();

        let mut all_cb: Vec<f64> = Vec::new();
        let mut all_da: Vec<f64> = Vec::new();
        let mut all_da2: Vec<f64> = Vec::new();
        let mut all_oe: Vec<f64> = Vec::new();
        let mut all_dc: Vec<f64> = Vec::new();
        let mut all_ab: Vec<f64> = Vec::new();

        for sh in &shapes {
            let w = pack_weights(sh.dim_out, sh.dim_in, 0x5EED_1234);
            let (qi, qf) = make_input(sh.dim_in, 0xC15_0001);

            let mut out_a = vec![0f32; sh.dim_out];
            let mut out_b = vec![0f32; sh.dim_out];
            let mut out_c = vec![0i32; sh.dim_out];
            let mut out_d = vec![0i32; sh.dim_out];

            // Warm the LUT and the caches on every arm before timing.
            set_force_scalar(false);
            ternary_matvec(&mut out_a, &qf, &w, sh.dim_out, sh.dim_in, 1.0);
            set_force_scalar(true);
            ternary_matvec(&mut out_b, &qf, &w, sh.dim_out, sh.dim_in, 1.0);
            set_force_scalar(false);
            ternary_matvec_i8(&mut out_c, &qi, &w, sh.dim_out, sh.dim_in);

            // Rule D: the integer arm must be bit-exact run to run, and must agree
            // with the float arm. A timing run that fails either is void.
            let digest_c0 = {
                let bytes: Vec<u8> = out_c.iter().flat_map(|v| v.to_le_bytes()).collect();
                fnv1a64(0xcbf2_9ce4_8422_2325, &bytes)
            };
            let mut max_abs_diff = 0f64;
            for j in 0..sh.dim_out {
                let d = (out_a[j] as f64 - out_c[j] as f64).abs();
                if d > max_abs_diff {
                    max_abs_diff = d;
                }
            }

            let mut t_a = u64::MAX;
            let mut t_a2 = u64::MAX;
            let mut t_b = u64::MAX;
            let mut t_c = u64::MAX;
            let mut t_d = u64::MAX;

            for _ in 0..reps {
                // --- A: float AVX2 ---
                set_force_scalar(false);
                // SAFETY: see calibrate_tsc_hz.
                let c0 = unsafe { _rdtsc() };
                ternary_matvec(&mut out_a, &qf, &w, sh.dim_out, sh.dim_in, 1.0);
                // SAFETY: as above.
                let c1 = unsafe { _rdtsc() };
                black_box(&out_a);
                t_a = t_a.min(c1 - c0);

                // --- B: float scalar ---
                set_force_scalar(true);
                // SAFETY: as above.
                let c0 = unsafe { _rdtsc() };
                ternary_matvec(&mut out_b, &qf, &w, sh.dim_out, sh.dim_in, 1.0);
                // SAFETY: as above.
                let c1 = unsafe { _rdtsc() };
                black_box(&out_b);
                t_b = t_b.min(c1 - c0);
                set_force_scalar(false);

                // --- C: integer scalar (CIS-1) ---
                // SAFETY: as above.
                let c0 = unsafe { _rdtsc() };
                ternary_matvec_i8(&mut out_c, &qi, &w, sh.dim_out, sh.dim_in);
                // SAFETY: as above.
                let c1 = unsafe { _rdtsc() };
                black_box(&out_c);
                t_c = t_c.min(c1 - c0);

                // --- D: integer AVX2 (the new kernel) ---
                // SAFETY: as above.
                let c0 = unsafe { _rdtsc() };
                ternary_matvec_i8_avx2(&mut out_d, &qi, &w, sh.dim_out, sh.dim_in);
                // SAFETY: as above.
                let c1 = unsafe { _rdtsc() };
                black_box(&out_d);
                t_d = t_d.min(c1 - c0);

                // --- A' : arm A again, now AFTER D, as an ORDER CONTROL ---
                // On 2026-08-05 the first arm-D capture showed arm A running 23%
                // slower than the clock ratio explained, while B and C tracked it
                // to within 0.1%. That is the signature of one arm perturbing
                // another's cache state, and it would flatter D/A. A' is the same
                // kernel on the same data in the other position: if A' == A the
                // ordering is innocent, and if it is not, the gap IS the effect.
                set_force_scalar(false);
                // SAFETY: as above.
                let c0 = unsafe { _rdtsc() };
                ternary_matvec(&mut out_a, &qf, &w, sh.dim_out, sh.dim_in, 1.0);
                // SAFETY: as above.
                let c1 = unsafe { _rdtsc() };
                black_box(&out_a);
                t_a2 = t_a2.min(c1 - c0);
            }

            let digest_c1 = {
                let bytes: Vec<u8> = out_c.iter().flat_map(|v| v.to_le_bytes()).collect();
                fnv1a64(0xcbf2_9ce4_8422_2325, &bytes)
            };
            let exact = digest_c0 == digest_c1;

            let macs = (sh.dim_out * sh.dim_in) as f64;
            let g = |ticks: u64| macs / ticks as f64; // MACs/tick; scaled below
            let cb = t_c as f64 / t_b as f64;
            let da = t_d as f64 / t_a as f64;
            let da2 = t_d as f64 / t_a2 as f64;
            let order_effect = t_a2 as f64 / t_a as f64;
            let dc = t_d as f64 / t_c as f64;
            let d_exact = out_d == out_c;
            let ab = t_a as f64 / t_b as f64;
            let ca = t_c as f64 / t_a as f64;
            all_cb.push(cb);
            all_da.push(da);
            all_da2.push(da2);
            all_oe.push(order_effect);
            all_dc.push(dc);
            all_ab.push(ab);

            println!("  {}   ({} nominal MACs)", sh.name, macs as u64);
            println!(
                "    A float/AVX2    {t_a:>12} ticks   {:>7.3} MAC/tick",
                g(t_a)
            );
            println!(
                "    B float/scalar  {t_b:>12} ticks   {:>7.3} MAC/tick",
                g(t_b)
            );
            println!(
                "    C int/scalar    {t_c:>12} ticks   {:>7.3} MAC/tick",
                g(t_c)
            );
            println!(
                "    D int/AVX2      {t_d:>12} ticks   {:>7.3} MAC/tick",
                g(t_d)
            );
            println!(
                "    A' float/AVX2   {t_a2:>12} ticks   {:>7.3} MAC/tick   (same arm, after D)",
                g(t_a2)
            );
            println!("    D/A  = {da:.3}x   using A measured FIRST");
            println!("    D/A' = {da2:.3}x   using A measured LAST (after D)");
            println!(
                "    order effect A'/A = {order_effect:.3}x  {}",
                if (order_effect - 1.0).abs() <= 0.05 {
                    "-- within 5%, ordering is innocent; quote D/A"
                } else {
                    "-- ARMS PERTURB EACH OTHER; D/A is NOT clean, quote the range"
                }
            );
            println!("    D/C = {dc:.3}x      speedup of the new kernel over the CIS-1 reference");
            println!(
                "    D bit-identical to C: {}",
                if d_exact { "YES" } else { "NO  ** RUN VOID **" }
            );
            println!("    C/B = {cb:.3}x      cost of integer semantics, scalar-vs-scalar");
            println!(
                "    A/B = {ab:.3}x      SIMD headroom on float (the prize for an AVX2 int kernel)"
            );
            println!("    C/A = {ca:.3}x      NOT the determinism cost — conflates two variables");
            println!(
                "    bit-exact across reps: {}   |A-C| max = {max_abs_diff:.1} (exact-int vs f32 accum)",
                if exact { "YES" } else { "NO  ** RUN VOID **" }
            );
            println!();
        }

        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let cb_med = med(all_cb);
        let ab_med = med(all_ab);

        println!("--- VERDICT (this run; a ledger row needs >= 3 logged runs, Rule B) ---");
        println!();
        let da_med = med(all_da.clone());
        let dc_med = med(all_dc.clone());
        let da2_med = med(all_da2.clone());
        let oe_med = med(all_oe.clone());
        println!(
            "  ORDER CONTROL  A'/A median = {oe_med:.3}x  (1.000 = arms do not perturb each other)"
        );
        if (oe_med - 1.0).abs() > 0.05 {
            println!("  !! ARM A MOVES WITH ITS POSITION. D/A is not a clean ratio in this run;");
            println!("     the honest statement is the RANGE below, not either endpoint.");
        }
        println!("  DETERMINISM COST AT PARITY SIMD:  D/A = {da_med:.3}x .. D/A' = {da2_med:.3}x");
        println!(
            "  New AVX2 integer kernel vs CIS-1 scalar reference:    D/C median = {dc_med:.3}x"
        );
        println!(
            "  Cost of CIS-1 integer semantics, SIMD held constant:  C/B median = {cb_med:.3}x"
        );
        println!(
            "  SIMD headroom currently unrealised on the integer path: A/B median = {ab_med:.3}x"
        );
        println!();
        if da_med <= 1.05 {
            println!("  READ: with both paths vectorised, integer semantics cost essentially");
            println!("  nothing -- or win. Determinism is free at the matvec on this machine.");
        } else if da_med <= 1.5 {
            println!("  READ: with both paths vectorised, determinism carries a modest and");
            println!("  quotable cost at the matvec. This is the number to publish.");
        } else {
            println!("  READ: even vectorised, the integer path lags float. The gap is real and");
            println!("  belongs in the record as a cost of determinism, not as a to-do.");
        }
        println!();
        if cb_med <= 1.35 {
            println!(
                "  READ: integer semantics are CHEAP at the kernel. Any larger end-to-end gap"
            );
            println!("  measured elsewhere is therefore not attributable to determinism by this");
            println!(
                "  result alone. An AVX2 integer ternary kernel is the open work item, and the"
            );
            println!("  A/B figure above is its budget.");
        } else if cb_med <= 3.0 {
            println!("  READ: integer semantics carry a real but bounded kernel cost. An AVX2");
            println!("  integer kernel would have to beat this margin before determinism is free.");
        } else {
            println!("  READ: integer semantics are EXPENSIVE at the kernel even against scalar");
            println!("  float. This is a negative finding about CIS-1's cost and should be");
            println!("  recorded as one — see the working convention on negative results.");
        }
        println!();
        println!("  NOT ESTABLISHED by this bench: end-to-end decode tok/s, attention/softmax");
        println!("  integer cost (cis_attn), or any figure on a machine other than the one named");
        println!("  in this log's header. Kernel ratios are not token ratios.");
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86::run()
}
