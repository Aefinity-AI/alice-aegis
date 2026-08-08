//! Microbenchmark for a LUT-based mixed-precision GEMM (the technique
//! bitnet.cpp calls TL1, and T-MAC generalizes).
//!
//! # RESULT: REJECTED on this hardware — but on MEMORY-TRAFFIC grounds, not
//! # kernel throughput. The original "16% SLOWER" verdict is RETRACTED.
//!
//! The figures of record are the 2026-07-30 same-binary A/B logs
//! (`docs/hardware_logs/lut_mpgemm_sameproc_ab*_2026-07-30.log`, three runs)
//! and the derivation note (`lut_mpgemm_ab_findings_2026-07-30.md`). A source
//! comment is not a legal parent for a number (Rule B); cite those, not this.
//!
//! History: until 2026-07-30 this bench did not measure the f32 kernel it
//! claimed to lose to — the "16% SLOWER" header compared against a number
//! typed into a comment. Measured same-binary, the pshufb arm is bimodal
//! (~36% swing) and in its modal state ~20% FASTER than the f32 matvec
//! kernel-side. The rejection stands anyway: the 4-bit layout carries 1.67x
//! the decode traffic on a path that is bandwidth-bound at post-A12 clocks,
//! which converts pshufb's best kernel day into a ~28% net decode LOSS, and
//! the batched GEMM beats it ~2x for prefill regardless. See the findings
//! note for the full derivation.
//!
//! The idea: a packed weight NIBBLE encodes two ternary weights (9 valid states
//! of 16). For a fixed pair of activations (a0, a1), all 16 values of
//! w0*a0 + w1*a1 fit in a 16-byte table — one `_mm256_shuffle_epi8` operand.
//! A single pshufb then evaluates that pair for 32 output rows at once: 64 MACs
//! in one instruction, which is where the ~10x projection came from.
//!
//! On the widening cost (the mechanism the old header blamed): the pshufb
//! yields i8 partial sums, and widening those to i16 costs an `extracti128`
//! plus two `cvtepi8_epi16` plus two `add_epi16` — about 9 instructions per
//! 64 MACs, not 3, while the f32 kernel gets 8 MACs from a single
//! `vfmadd231ps`. Real, but same-binary measurement shows it is NOT decisive
//! at these dimensions. What is decisive is 4 bits/weight instead of 2,
//! doubling ternary weight traffic on a bandwidth-bound decode path.
//!
//! `_mm256_maddubs_epi16` would widen and pair-sum in one instruction, but it
//! sums ADJACENT lanes — which here are different output rows, not different
//! groups of the same row — so it cannot be used without another layout change
//! that then breaks the one-table-per-pshufb property.
//!
//! A real win needs `vpdpbusd` (AVX-512 VNNI / AVX-VNNI): 32 int8 MACs in one
//! instruction with i32 accumulation and no widening. Comet Lake lacks it.
//! On a VNNI machine this is worth revisiting; on this one, the f32 LUT+FMA
//! kernel is already the right answer.
//!
//! Run: cargo run --release --bin lut_mpgemm

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

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const DIM_OUT: usize = 2560;
    const DIM_IN: usize = 6912;
    const ROW_BLK: usize = 32; // rows evaluated per pshufb

    fn ternary(code: u8) -> i32 {
        match code {
            1 => 1,
            2 => -1,
            _ => 0,
        }
    }

    /// Reference: plain row-major 2-bit packed weights, scalar dot product.
    fn reference(out: &mut [i32], acts: &[i8], packed: &[u8]) {
        let pdi = DIM_IN / 4;
        for r in 0..DIM_OUT {
            let mut acc = 0i32;
            for c in 0..pdi {
                let b = packed[r * pdi + c];
                for k in 0..4 {
                    let w = ternary((b >> (2 * k)) & 3);
                    acc += w * acts[c * 4 + k] as i32;
                }
            }
            out[r] = acc;
        }
    }

    /// Repack row-major 2-bit weights into LUT layout:
    /// `[group][row_block][32 bytes: one nibble per row]`
    /// where group g covers input dims (2g, 2g+1).
    fn repack(packed: &[u8]) -> Vec<u8> {
        let pdi = DIM_IN / 4;
        let groups = DIM_IN / 2;
        let blocks = DIM_OUT / ROW_BLK;
        let mut out = vec![0u8; groups * blocks * ROW_BLK];
        for g in 0..groups {
            let c = (2 * g) / 4; // byte holding dims 2g,2g+1
            let k = (2 * g) % 4; // starting 2-bit slot within that byte
            for blk in 0..blocks {
                for r in 0..ROW_BLK {
                    let row = blk * ROW_BLK + r;
                    let b = packed[row * pdi + c];
                    let nib = (b >> (2 * k)) & 0x0F; // two codes = one nibble
                    out[(g * blocks + blk) * ROW_BLK + r] = nib;
                }
            }
        }
        out
    }

    /// Build the 16-entry table for activation pair (a0, a1): entry n encodes
    /// w0 = n&3, w1 = (n>>2)&3.
    fn build_table(a0: i8, a1: i8) -> [i8; 16] {
        let mut t = [0i8; 16];
        for n in 0..16u8 {
            let v = ternary(n & 3) * a0 as i32 + ternary((n >> 2) & 3) * a1 as i32;
            debug_assert!((-128..=127).contains(&v), "pair sum {v} overflows i8");
            t[n as usize] = v as i8;
        }
        t
    }

    /// Tables depend only on the activations, so they are built ONCE per matvec and
    /// reused across every row block. (Building them inside the block loop is a
    /// factor-of-`blocks` mistake — it made this kernel look 3x slower than the
    /// f32 one on the first run.)
    fn build_all_tables(acts: &[i8]) -> Vec<[i8; 16]> {
        (0..DIM_IN / 2)
            .map(|g| build_table(acts[2 * g], acts[2 * g + 1]))
            .collect()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn lut_gemm(out: &mut [i32], tables: &[[i8; 16]], repacked: &[u8]) {
        unsafe {
            let groups = DIM_IN / 2;
            let blocks = DIM_OUT / ROW_BLK;

            for blk in 0..blocks {
                // 32 rows -> accumulate in i16 (32 lanes across two vectors), widen
                // to i32 periodically before the i16 lanes can overflow.
                let mut acc_lo = _mm256_setzero_si256(); // rows 0..15  as i16
                let mut acc_hi = _mm256_setzero_si256(); // rows 16..31 as i16
                let mut sum_lo = _mm256_setzero_si256(); // i32
                let mut sum_hi = _mm256_setzero_si256();
                let mut since_widen = 0;

                for g in 0..groups {
                    let tbl = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                        tables[g].as_ptr() as *const __m128i
                    ));
                    let idx = _mm256_loadu_si256(
                        repacked.as_ptr().add((g * blocks + blk) * ROW_BLK) as *const __m256i,
                    );

                    // One pshufb: 32 rows x 2 MACs = 64 MACs.
                    let p = _mm256_shuffle_epi8(tbl, idx);

                    // widen i8 -> i16 and accumulate
                    let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p));
                    let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p, 1));
                    acc_lo = _mm256_add_epi16(acc_lo, lo);
                    acc_hi = _mm256_add_epi16(acc_hi, hi);

                    since_widen += 1;
                    if since_widen == 128 {
                        sum_lo = _mm256_add_epi32(
                            sum_lo,
                            _mm256_madd_epi16(acc_lo, _mm256_set1_epi16(1)),
                        );
                        sum_hi = _mm256_add_epi32(
                            sum_hi,
                            _mm256_madd_epi16(acc_hi, _mm256_set1_epi16(1)),
                        );
                        acc_lo = _mm256_setzero_si256();
                        acc_hi = _mm256_setzero_si256();
                        since_widen = 0;
                    }
                }
                sum_lo = _mm256_add_epi32(sum_lo, _mm256_madd_epi16(acc_lo, _mm256_set1_epi16(1)));
                sum_hi = _mm256_add_epi32(sum_hi, _mm256_madd_epi16(acc_hi, _mm256_set1_epi16(1)));

                // madd_epi16 with ones pairs adjacent i16 lanes -> results are for
                // row pairs; unpack them back to per-row sums.
                let mut tmp = [0i32; 8];
                _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum_lo);
                for (i, v) in tmp.iter().enumerate() {
                    out[blk * ROW_BLK + i * 2] += *v; // NOTE: pairing artifact, see below
                }
                _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, sum_hi);
                for (i, v) in tmp.iter().enumerate() {
                    out[blk * ROW_BLK + 16 + i * 2] += *v;
                }
            }
        }
    }

    pub fn run() {
        // NOTE ON CORRECTNESS: `_mm256_madd_epi16(x, ones)` sums ADJACENT i16 lanes,
        // which mixes two different rows. A production kernel must instead widen with
        // unpacklo/unpackhi (or keep 4 i32 accumulators per block). This benchmark
        // therefore measures THROUGHPUT of the pshufb pipeline faithfully, but its
        // numeric output is only correct for the timing loop's purposes. Flagged so
        // nobody mistakes it for a validated kernel.
        println!(
            "LUT mpGEMM throughput probe ({}x{} ternary)\n",
            DIM_OUT, DIM_IN
        );

        let mut s = 0x2545F4914F6CDD1Du64;
        let mut packed = vec![0u8; DIM_OUT * DIM_IN / 4];
        for b in packed.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mut byte = 0u8;
            for k in 0..4 {
                byte |= (((s >> (k * 8)) % 3) as u8) << (2 * k);
            }
            *b = byte;
        }
        // int7 activations so a pair sum stays inside i8
        let acts: Vec<i8> = (0..DIM_IN).map(|i| ((i % 127) as i32 - 63) as i8).collect();

        let t = Instant::now();
        let repacked = repack(&packed);
        println!(
            "repack (one-time, at model load): {:.1} ms for this tensor",
            t.elapsed().as_secs_f64() * 1e3
        );
        println!(
            "  layout memory: {} MB (vs {} MB packed) -- 4 bits/weight\n",
            repacked.len() / 1_000_000,
            packed.len() / 1_000_000
        );

        let macs = (DIM_OUT * DIM_IN) as f64;

        let mut out_ref = vec![0i32; DIM_OUT];
        let t = Instant::now();
        reference(&mut out_ref, &acts, &packed);
        let dt_ref = t.elapsed().as_secs_f64();
        std::hint::black_box(&out_ref); // or LLVM deletes the whole loop
        println!(
            "scalar reference : {:7.1} ms ({:.2} GMAC/s)",
            dt_ref * 1e3,
            macs / dt_ref / 1e9
        );

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut out = vec![0i32; DIM_OUT];

                // The engine's f32 LUT+FMA kernels, on the SAME packed weights.
                // Until 2026-07-30 this bench did not measure them — the "16%
                // SLOWER" header verdict compared against a number typed into a
                // comment. Every arm now runs in this one binary, interleaved in
                // one loop so all kernels see the same turbo state.
                let x: Vec<f32> = acts.iter().map(|&a| a as f32).collect();
                let x8: Vec<f32> = (0..8).flat_map(|_| x.iter().copied()).collect();
                let mut out_f32 = vec![0.0f32; DIM_OUT];
                let mut out_gemm = vec![0.0f32; 8 * DIM_OUT];

                // Table build is per-activation-vector, amortized over all rows.
                let t = Instant::now();
                let tables = build_all_tables(&acts);
                let dt_tbl = t.elapsed().as_secs_f64();

                // warm every arm
                unsafe { lut_gemm(&mut out, &tables, &repacked) };
                ternary_matvec(&mut out_f32, &x, &packed, DIM_OUT, DIM_IN, 1.0);
                ternary_matmul(&mut out_gemm, &x8, &packed, 8, DIM_OUT, DIM_IN, 1.0);

                let (mut best, mut t_f32, mut t_gemm) =
                    (f64::INFINITY, f64::INFINITY, f64::INFINITY);
                for _ in 0..5 {
                    out.iter_mut().for_each(|v| *v = 0);
                    let t = Instant::now();
                    unsafe { lut_gemm(&mut out, &tables, &repacked) };
                    best = best.min(t.elapsed().as_secs_f64());

                    let t = Instant::now();
                    ternary_matvec(&mut out_f32, &x, &packed, DIM_OUT, DIM_IN, 1.0);
                    t_f32 = t_f32.min(t.elapsed().as_secs_f64());

                    let t = Instant::now();
                    ternary_matmul(&mut out_gemm, &x8, &packed, 8, DIM_OUT, DIM_IN, 1.0);
                    t_gemm = t_gemm.min(t.elapsed().as_secs_f64() / 8.0); // per token
                }
                std::hint::black_box((&out, &out_f32, &out_gemm));

                // The f32 kernel must agree with the scalar reference. (The pshufb
                // arm's output is NOT checked — see the pairing-artifact note at
                // the top of main(); it is a throughput probe only.)
                let ref_sum: f64 = out_ref.iter().map(|&v| v as f64).sum();
                let f32_sum: f64 = out_f32.iter().map(|&v| v as f64).sum();
                let agree = (ref_sum - f32_sum).abs() / ref_sum.abs().max(1.0) < 1e-3;

                println!(
                    "LUT pshufb kernel: {:7.1} ms ({:.2} GMAC/s)  [best of 5]",
                    best * 1e3,
                    macs / best / 1e9
                );
                println!(
                    "  + table build  : {:7.3} ms once per activation vector ({:.1}% of kernel)",
                    dt_tbl * 1e3,
                    dt_tbl / best * 100.0
                );
                let eff = best + dt_tbl;
                println!(
                    "  effective      : {:.2} GMAC/s including table build",
                    macs / eff / 1e9
                );
                println!(
                    "f32 LUT+FMA matvec:{:7.1} ms ({:.2} GMAC/s)  = {:.2}x pshufb-effective  [engine decode kernel]",
                    t_f32 * 1e3,
                    macs / t_f32 / 1e9,
                    eff / t_f32
                );
                println!(
                    "f32 batched GEMM :{:7.1} ms ({:.2} GMAC/s)  = {:.2}x pshufb-effective  (per token, batch 8) [engine prefill kernel]",
                    t_gemm * 1e3,
                    macs / t_gemm / 1e9,
                    eff / t_gemm
                );
                println!("f32 matvec agrees with scalar reference: {agree}");
            } else {
                println!("AVX2 not available");
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86::run()
}
