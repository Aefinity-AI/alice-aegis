//! Microbenchmark: prefill GEMM throughput vs the per-token matvec baseline,
//! on real BitNet FFN dimensions. Run with:
//!   cargo run --release --bench gemm_tile
//! (plain binary, not libtest bench, so it works on stable)

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

    fn make_weights(dim_out: usize, dim_in: usize) -> Vec<u8> {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut w = vec![0u8; dim_out * dim_in / 4];
        for b in w.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mut byte = 0u8;
            for lane in 0..4 {
                byte |= (((s >> (lane * 8)) % 3) as u8) << (lane * 2);
            }
            *b = byte;
        }
        w
    }

    pub fn run() {
        // BitNet's largest projection: down_proj is 2560 x 6912.
        let (dim_out, dim_in) = (2560usize, 6912usize);
        let weights = make_weights(dim_out, dim_in);
        let macs_per_token = (dim_out * dim_in) as f64;

        // This machine's turbo/thermal state makes the first timed region in any
        // process look fastest. Interleave both kernels and take best-of-N so the
        // comparison isn't an artifact of measurement order.
        const REPS: usize = 5;

        for &batch in &[8usize, 32, 64] {
            let input: Vec<f32> = (0..batch * dim_in)
                .map(|i| ((i % 200) as f32 - 100.0) / 100.0)
                .collect();
            let mut out = vec![0.0f32; batch * dim_out];

            // Warm caches, branch predictors, and clock ramp.
            for _ in 0..2 {
                ternary_matmul(&mut out, &input, &weights, batch, dim_out, dim_in, 1.0);
            }

            let mut gemm = f64::INFINITY;
            let mut matvec = f64::INFINITY;
            for _ in 0..REPS {
                let t = Instant::now();
                ternary_matmul(&mut out, &input, &weights, batch, dim_out, dim_in, 1.0);
                gemm = gemm.min(t.elapsed().as_secs_f64());

                let t = Instant::now();
                for b in 0..batch {
                    ternary_matvec(
                        &mut out[b * dim_out..(b + 1) * dim_out],
                        &input[b * dim_in..(b + 1) * dim_in],
                        &weights,
                        dim_out,
                        dim_in,
                        1.0,
                    );
                }
                matvec = matvec.min(t.elapsed().as_secs_f64());
            }

            let total_macs = macs_per_token * batch as f64;
            println!(
                "batch {:3}:  gemm {:6.1} ms ({:5.2} GMAC/s)   matvec {:6.1} ms ({:5.2} GMAC/s)   speedup {:.2}x  [best of {}]",
                batch,
                gemm * 1e3,
                total_macs / gemm / 1e9,
                matvec * 1e3,
                total_macs / matvec / 1e9,
                matvec / gemm,
                REPS,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86::run()
}
