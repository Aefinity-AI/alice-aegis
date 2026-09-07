//! Perf-counter harness for the CIS-1 FullInt ternary kernels. Runs ONE kernel on ONE shape
//! for `reps` repetitions so `perf stat` sees a single kernel's counters.
//!   perf_i8_kernels tmv|tmm <dim_out> <dim_in> <reps>
//! Work per rep = 8 tokens (tmm: one 8-token tile call; tmv: 8 single-token calls).
use aegis_core::cis_avx2::{ternary_matmul_i8_avx2, ternary_matvec_i8_avx2};
use std::time::Instant;

const N_TOK: usize = 8;

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

fn make_inputs(n: usize) -> Vec<i8> {
    let mut s = 0x2545F4914F6CDD1Du64;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) % 201) as i32 - 100) as i8
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 5 {
        eprintln!("usage: perf_i8_kernels tmv|tmm <dim_out> <dim_in> <reps>");
        std::process::exit(2);
    }
    let mode = a[1].as_str();
    let dim_out: usize = a[2].parse().unwrap();
    let dim_in: usize = a[3].parse().unwrap();
    let reps: usize = a[4].parse().unwrap();
    let w = make_weights(dim_out, dim_in);
    let inp = make_inputs(N_TOK * dim_in);
    let mut out = vec![0i32; N_TOK * dim_out];
    let run = |out: &mut [i32]| match mode {
        "tmv" => {
            for t in 0..N_TOK {
                ternary_matvec_i8_avx2(
                    &mut out[t * dim_out..(t + 1) * dim_out],
                    &inp[t * dim_in..(t + 1) * dim_in],
                    &w,
                    dim_out,
                    dim_in,
                );
            }
        }
        "tmm" => ternary_matmul_i8_avx2(out, &inp, &w, dim_out, dim_in, N_TOK),
        _ => panic!("mode must be tmv or tmm"),
    };
    for _ in 0..3 {
        run(&mut out);
    }
    let t = Instant::now();
    for _ in 0..reps {
        run(&mut out);
    }
    let secs = t.elapsed().as_secs_f64();
    let mut h = 0xcbf29ce484222325u64;
    for &v in &out {
        h ^= v as u32 as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    let blocks = (dim_out * dim_in / 128) as f64 * N_TOK as f64 * reps as f64;
    let macs = (dim_out * dim_in) as f64 * N_TOK as f64 * reps as f64;
    println!(
        "RESULT mode={} dim_out={} dim_in={} reps={} tokens={} wall_s={:.3} ns_per_token_block={:.3} gmac_s={:.2} packed_MB_per_s={:.0} checksum={:016x}",
        mode,
        dim_out,
        dim_in,
        reps,
        N_TOK * reps,
        secs,
        secs * 1e9 / blocks,
        macs / secs / 1e9,
        (dim_out * dim_in / 4) as f64
            * (if mode == "tmv" { N_TOK } else { 1 }) as f64
            * reps as f64
            / secs
            / 1e6,
        h
    );
}
