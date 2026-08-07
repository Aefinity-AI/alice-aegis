//! Kernel race: the CTZ dual-bitmap "zero-multiplication" matvec (from the
//! Infinity OS era, `aegis-forge-v5/src/dual_matvec.rs`) against the current
//! AVX2 LUT+FMA kernels.
//!
//! The CTZ design is mathematically sound: for ternary weights {-1,0,+1},
//!     y[i] = sum(x[j] where w=+1) - sum(x[j] where w=-1)
//! so it stores two u64 bitmaps (positions of +1 and of -1) and walks only the
//! set bits with `trailing_zeros()` + the `w &= w-1` clear-lowest-bit idiom.
//! No multiplications at all — only adds, subtracts, and bit scans. Same
//! 2 bits/weight of memory as the current packed layout.
//!
//! The question this benchmark answers is not whether it avoids multiplies
//! (it does) but whether avoiding them makes it *faster* than a SIMD kernel
//! that performs them 8-wide.
//!
//! # RESULT: the SIMD kernel wins by 6.59x at the real weight sparsity, and CTZ
//! # cannot win at ANY realistic sparsity.
//!
//! These numbers are a SUMMARY. The figures of record live in the instrument
//! log — `docs/hardware_logs/ctz_vs_simd_2026-07-29.log` (dev box, i5-10210U
//! under crosvm, git_head a249b2c). A source comment is not a legal parent for
//! a number (Rule B); cite the log, not this block.
//!
//!   zeros    CTZ        AVX2 matvec        batched GEMM
//!   42.21%*  12.19 ms   1.85 ms  (6.59x)   1.04 ms  (11.74x)
//!   61.8%     8.73 ms   1.85 ms  (4.73x)   1.05 ms  ( 8.33x)
//!   90.0%     5.52 ms   1.83 ms  (3.02x)   1.01 ms  ( 5.49x)
//!   99.0%     2.02 ms   1.83 ms  (1.10x)   1.01 ms  ( 1.99x)
//!   100%      0.35 ms   1.83 ms  (0.19x)   0.99 ms  ( 0.36x)  <- CTZ only wins here
//!   * = zero fraction of the real BitNet b1.58-2B weights, from a FULL scan of
//!       all 210 ternary tensors (2.084 G weights). An earlier 6-tensor sample
//!       gave 40.8%; the full number is 42.21% and changes nothing.
//!
//! CTZ only ties the matvec at ~99% zeros, and the GEMM at ~99.5%. BitNet is
//! at 42.2%. Substituting CTZ into the engine today would cost the decode path
//! a factor of 6.59 on ternary_matvec.
//!
//! Why the "zero-multiplication" premise misleads: on this silicon a fused
//! multiply-add is ONE instruction, 8 lanes wide, fully pipelined — the same
//! cost as an add. The SIMD kernel therefore processes a zero weight at *zero
//! marginal cost*. Skipping zeros buys nothing, while the skip machinery costs
//! a serial loop-carried accumulator, one unpredictable branch per nonzero, and
//! one scalar load per nonzero. Ternary weights make multiplies free; they do
//! not make SIMD lanes free.
//!
//! This is not a criticism of the idea — CTZ zero-skip is the right algorithm
//! when sparsity is extreme and unstructured. It is the wrong algorithm at 41%.
//!
//! Run: cargo run --release --bin ctz_vs_simd

// `cargo test` auto-builds bin targets on every architecture so integration
// tests can exec them; this bench is x86-only, so non-x86 gets a stub main.
#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("x86_64-only benchmark; nothing to run on this architecture");
}

#[cfg(target_arch = "x86_64")]
mod x86 {

    use aegis_core::ops::{ternary_matmul, ternary_matvec};
    use std::time::Instant;

    const DIM_OUT: usize = 2560;
    const DIM_IN: usize = 6912; // BitNet down_proj

    /// Faithful port of the dual-bitmap CTZ kernel.
    /// `pos`/`neg` are row-major bitmaps, DIM_IN/64 u64 words per row.
    fn dual_matvec_ctz(out: &mut [f32], x: &[f32], pos: &[u64], neg: &[u64], scale: f32) {
        let words = DIM_IN / 64;
        for row in 0..DIM_OUT {
            let mut acc = 0.0f32;
            let base = row * words;

            for w in 0..words {
                let off = w * 64;

                let mut p = pos[base + w];
                while p != 0 {
                    acc += x[off + p.trailing_zeros() as usize];
                    p &= p - 1; // clear lowest set bit
                }

                let mut n = neg[base + w];
                while n != 0 {
                    acc -= x[off + n.trailing_zeros() as usize];
                    n &= n - 1;
                }
            }
            out[row] = acc * scale;
        }
    }

    /// Build packed 2-bit weights (00=0, 01=+1, 10=-1) at a chosen zero fraction.
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

    /// Convert packed 2-bit weights to the dual-bitmap layout.
    fn to_bitmaps(packed: &[u8]) -> (Vec<u64>, Vec<u64>) {
        let words = DIM_IN / 64;
        let pdi = DIM_IN / 4;
        let mut pos = vec![0u64; DIM_OUT * words];
        let mut neg = vec![0u64; DIM_OUT * words];
        for row in 0..DIM_OUT {
            for c in 0..pdi {
                let byte = packed[row * pdi + c];
                for lane in 0..4 {
                    let col = c * 4 + lane;
                    let bit = 1u64 << (col % 64);
                    let idx = row * words + col / 64;
                    match (byte >> (2 * lane)) & 3 {
                        1 => pos[idx] |= bit,
                        2 => neg[idx] |= bit,
                        _ => {}
                    }
                }
            }
        }
        (pos, neg)
    }

    fn checksum(v: &[f32]) -> f64 {
        v.iter().map(|&x| x as f64).sum()
    }

    pub fn run() {
        println!(
            "CTZ dual-bitmap vs AVX2 LUT+FMA  ({}x{} ternary matvec)\n",
            DIM_OUT, DIM_IN
        );
        let macs = (DIM_OUT * DIM_IN) as f64;
        let x: Vec<f32> = (0..DIM_IN)
            .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
            .collect();

        for (label, zf) in [
            (
                "real BitNet weights (42.21% zeros, full-model scan)",
                0.4221,
            ),
            ("Infinity OS target (61.8% zeros)", 0.618),
            ("extreme sparsity (90% zeros)", 0.900),
            ("99% zeros", 0.990),
            ("ALL ZEROS - pure loop floor", 1.000),
        ] {
            println!("--- {label}");
            let packed = make_packed(zf);
            let (pos, neg) = to_bitmaps(&packed);

            let mut out_ctz = vec![0.0f32; DIM_OUT];
            let mut out_simd = vec![0.0f32; DIM_OUT];

            // warm
            dual_matvec_ctz(&mut out_ctz, &x, &pos, &neg, 1.0);
            ternary_matvec(&mut out_simd, &x, &packed, DIM_OUT, DIM_IN, 1.0);

            // Interleave and take best-of-N: this machine's turbo state drifts.
            let (mut t_ctz, mut t_simd, mut t_gemm) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut out_gemm = vec![0.0f32; 8 * DIM_OUT];
            let x8: Vec<f32> = (0..8).flat_map(|_| x.iter().copied()).collect();

            for _ in 0..5 {
                let t = Instant::now();
                dual_matvec_ctz(&mut out_ctz, &x, &pos, &neg, 1.0);
                t_ctz = t_ctz.min(t.elapsed().as_secs_f64());

                let t = Instant::now();
                ternary_matvec(&mut out_simd, &x, &packed, DIM_OUT, DIM_IN, 1.0);
                t_simd = t_simd.min(t.elapsed().as_secs_f64());

                let t = Instant::now();
                ternary_matmul(&mut out_gemm, &x8, &packed, 8, DIM_OUT, DIM_IN, 1.0);
                t_gemm = t_gemm.min(t.elapsed().as_secs_f64() / 8.0); // per token
            }
            std::hint::black_box((&out_ctz, &out_simd, &out_gemm));

            // Both kernels must agree (float order differs, so compare loosely).
            let (a, b) = (checksum(&out_ctz), checksum(&out_simd));
            let agree = (a - b).abs() / a.abs().max(1.0) < 1e-3;

            println!(
                "  CTZ dual-bitmap : {:7.2} ms  ({:5.2} GMAC/s)",
                t_ctz * 1e3,
                macs / t_ctz / 1e9
            );
            println!(
                "  AVX2 LUT matvec : {:7.2} ms  ({:5.2} GMAC/s)  = {:.2}x CTZ",
                t_simd * 1e3,
                macs / t_simd / 1e9,
                t_ctz / t_simd
            );
            println!(
                "  AVX2 batched GEMM:{:7.2} ms  ({:5.2} GMAC/s)  = {:.2}x CTZ  (per token, batch 8)",
                t_gemm * 1e3,
                macs / t_gemm / 1e9,
                t_ctz / t_gemm
            );
            println!("  outputs agree: {agree}");
            println!(
                "  CTZ did {:.0}M add/sub ops, SIMD did {:.0}M FMA lanes\n",
                macs * (1.0 - zf) / 1e6,
                macs / 1e6
            );
        }

        println!("Memory: both layouts are 2 bits/weight (CTZ: 1 bit in each of two");
        println!("bitmaps; current: one 2-bit code). No footprint advantage either way.");
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86::run()
}
