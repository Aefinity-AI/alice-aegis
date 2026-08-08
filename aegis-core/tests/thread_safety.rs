//! The kernels must be callable from multiple threads.
//!
//! Before 2026-07-09 the unpack LUT and the SIMD-detection flag were
//! `static mut` with check-then-set initialization: two threads racing on them
//! was undefined behavior regardless of the values involved, which blocked
//! multicore decode. The LUT is now a compile-time `const` and the SIMD state
//! is an atomic. This test runs the kernels concurrently and checks every
//! thread computes the same answer as the single-threaded path.
//!
//! Run under `cargo +nightly test -Zsanitizer=thread` to check for races
//! directly; this test catches the observable consequences.

use aegis_core::ops::{avx2_active, ternary_matvec};
use std::sync::Arc;
use std::thread;

fn make_weights(dim_out: usize, dim_in: usize) -> Vec<u8> {
    let mut s = 0x243F6A8885A308D3u64;
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

#[test]
fn kernels_agree_across_threads() {
    let (dim_out, dim_in) = (512usize, 1024usize);
    let weights = Arc::new(make_weights(dim_out, dim_in));
    let input: Arc<Vec<f32>> = Arc::new(
        (0..dim_in)
            .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
            .collect(),
    );

    // Single-threaded reference.
    let mut want = vec![0.0f32; dim_out];
    ternary_matvec(&mut want, &input, &weights, dim_out, dim_in, 0.75);
    let want = Arc::new(want);

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let w = Arc::clone(&weights);
            let inp = Arc::clone(&input);
            let want = Arc::clone(&want);
            thread::spawn(move || {
                // Each thread independently triggers SIMD detection and LUT use.
                let _ = avx2_active();
                for _ in 0..20 {
                    let mut got = vec![0.0f32; dim_out];
                    ternary_matvec(&mut got, &inp, &w, dim_out, dim_in, 0.75);
                    for i in 0..dim_out {
                        assert_eq!(
                            got[i].to_bits(),
                            want[i].to_bits(),
                            "thread divergence at {i}"
                        );
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

#[test]
fn simd_detection_is_stable_under_concurrency() {
    let handles: Vec<_> = (0..16).map(|_| thread::spawn(avx2_active)).collect();
    let results: Vec<bool> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(
        results.windows(2).all(|w| w[0] == w[1]),
        "avx2_active() disagreed across threads: {results:?}"
    );
}

/// The row-parallel kernels must produce bit-identical results to the serial
/// path — including argmax tie-breaking, which must pick the lowest index just
/// as a single sequential scan would.
#[cfg(feature = "parallel")]
#[test]
fn parallel_kernels_match_serial() {
    use aegis_core::ops::f32_dot_argmax;

    // Big enough to cross PARALLEL_MIN_MACS so the threaded path actually runs.
    let (dim_out, dim_in) = (4096usize, 2560usize);
    let weights = make_weights(dim_out, dim_in);
    let input: Vec<f32> = (0..dim_in)
        .map(|i| ((i % 131) as f32 - 65.0) / 65.0)
        .collect();

    let mut par = vec![0.0f32; dim_out];
    ternary_matvec(&mut par, &input, &weights, dim_out, dim_in, 0.5);

    // Force the serial path by asking for one thread.
    unsafe { std::env::set_var("AEGIS_THREADS", "1") };
    // worker_threads() caches, so compare against an explicit chunk-by-chunk
    // serial computation instead of relying on the env var taking effect.
    let mut ser = vec![0.0f32; dim_out];
    for r in (0..dim_out).step_by(256) {
        let rows = 256.min(dim_out - r);
        let pdi = dim_in / 4;
        ternary_matvec(
            &mut ser[r..r + rows],
            &input,
            &weights[r * pdi..(r + rows) * pdi],
            rows,
            dim_in,
            0.5,
        );
    }
    for i in 0..dim_out {
        assert_eq!(par[i].to_bits(), ser[i].to_bits(), "row {i} differs");
    }

    // Argmax reduction: build an embedding table with a known unique maximum,
    // then a tie, and confirm index selection.
    let (vocab, emb_dim) = (2048usize, 2560usize);
    let mut emb = vec![0u8; vocab * emb_dim * 2];
    // BF16 1.0 = 0x3F80 -> little-endian bytes [0x80, 0x3F]
    for row in [7usize, 1500] {
        for c in 0..emb_dim {
            let o = (row * emb_dim + c) * 2;
            emb[o] = 0x80;
            emb[o + 1] = 0x3F;
        }
    }
    let ones = vec![1.0f32; emb_dim];
    let mut logits = vec![0.0f32; vocab];
    let idx = f32_dot_argmax(&mut logits, &ones, &emb, vocab, emb_dim);
    assert_eq!(
        idx, 7,
        "tie must resolve to the lowest index, as a serial scan would"
    );
    assert!((logits[7] - emb_dim as f32).abs() < 1.0);
    assert!((logits[1500] - emb_dim as f32).abs() < 1.0);
}
