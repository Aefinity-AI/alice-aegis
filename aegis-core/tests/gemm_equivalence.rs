//! The batched prefill GEMM must produce bit-identical results to the
//! per-token matvec path it replaces, for every batch size including the
//! partial trailing tile.
//!
//! Bit-identical is achievable here because both paths accumulate over the
//! same columns in the same order with the same FMA sequence; only the loop
//! nesting differs.

use aegis_core::ops::{ternary_matmul, ternary_matvec};
use std::sync::Mutex;

/// Serializes the tests in this file that are sensitive to the PROCESS-GLOBAL
/// force-scalar flag: the toggle test flips it, and the equivalence test's
/// bitwise asserts are only valid while the dispatch path is stable. Without
/// this lock the default parallel harness interleaves them and the equivalence
/// test flakes (~2-ulp "mismatches" that are really a mid-comparison path
/// switch). `unwrap_or_else(into_inner)` keeps the lock usable after a poisoned
/// (panicked) holder — the Reset guard below restores the flag either way.
static FORCE_SCALAR_LOCK: Mutex<()> = Mutex::new(());

fn make_weights(dim_out: usize, dim_in: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut w = vec![0u8; dim_out * dim_in / 4];
    for b in w.iter_mut() {
        // xorshift; keep only valid ternary codes 00/01/10 in each 2-bit lane
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let mut byte = 0u8;
        for lane in 0..4 {
            let code = ((s >> (lane * 8)) % 3) as u8; // 0,1,2 -> 0,+1,-1
            byte |= code << (lane * 2);
        }
        *b = byte;
    }
    w
}

fn make_input(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

#[test]
fn batched_gemm_matches_per_token_matvec() {
    let _serial = FORCE_SCALAR_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Dims mirror real BitNet shapes (multiples of 32) plus a non-multiple case.
    let cases = [(256usize, 128usize), (64, 2560), (128, 6912)];
    let scale = 0.0123_f32;

    for &(dim_out, dim_in) in &cases {
        let weights = make_weights(dim_out, dim_in, 0x9E3779B97F4A7C15);
        // Cover full tiles, partial tiles, and every remainder 1..=8.
        for batch in [1usize, 2, 3, 5, 7, 8, 9, 15, 16, 17] {
            let input = make_input(batch * dim_in, 0xDEADBEEFCAFEF00D);

            let mut got = vec![0.0f32; batch * dim_out];
            ternary_matmul(&mut got, &input, &weights, batch, dim_out, dim_in, scale);

            let mut want = vec![0.0f32; batch * dim_out];
            for b in 0..batch {
                ternary_matvec(
                    &mut want[b * dim_out..(b + 1) * dim_out],
                    &input[b * dim_in..(b + 1) * dim_in],
                    &weights,
                    dim_out,
                    dim_in,
                    scale,
                );
            }

            for i in 0..batch * dim_out {
                assert_eq!(
                    got[i].to_bits(),
                    want[i].to_bits(),
                    "mismatch dim_out={dim_out} dim_in={dim_in} batch={batch} idx={i}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }
}

#[test]
fn batched_gemm_rejects_undersized_buffers() {
    let (dim_out, dim_in, batch) = (64usize, 128usize, 4usize);
    let weights = make_weights(dim_out, dim_in, 1);
    let input = make_input(batch * dim_in, 2);
    let mut out = vec![0.0f32; batch * dim_out - 1]; // one short
    ternary_matmul(&mut out, &input, &weights, batch, dim_out, dim_in, 1.0);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "must no-op on undersized output"
    );
}

/// The runtime race toggles must (a) actually change the executed path and
/// (b) leave the math correct.
///
/// This test mutates PROCESS-GLOBAL state (the force-scalar flag). It holds
/// FORCE_SCALAR_LOCK for its whole body so the equivalence test above can never
/// observe a mid-flip path switch, regardless of harness parallelism (the old
/// contract — "the coherence gate runs --test-threads=1" — made plain
/// `cargo test` flaky). The `Reset` guard below restores the default even if an
/// assertion panics, so a failure here cannot leave the flag stuck for whatever
/// runs next.
///
/// The two paths do NOT agree bit-for-bit: the AVX2 kernel sums eight lanes and
/// horizontal-adds, the scalar kernel accumulates sequentially, so their
/// floating-point rounding differs. That divergence is real, documented, and
/// harmless (it can only flip an argmax on a near-exact tie). We assert
/// closeness, not identity.
#[test]
fn force_scalar_toggle_is_correct_and_reversible() {
    use aegis_core::ops::{active_path_name, set_force_scalar};

    let _serial = FORCE_SCALAR_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            set_force_scalar(false);
        }
    }
    let _guard = Reset;

    let (dim_out, dim_in) = (256usize, 512usize);
    let weights = make_weights(dim_out, dim_in, 0xABCDEF);
    let input = make_input(dim_in, 0x123456);

    set_force_scalar(false);
    let default_path = active_path_name();
    let mut a = vec![0.0f32; dim_out];
    ternary_matvec(&mut a, &input, &weights, dim_out, dim_in, 1.0);

    set_force_scalar(true);
    assert_eq!(
        active_path_name(),
        "scalar",
        "force_scalar must select the scalar path"
    );
    let mut s = vec![0.0f32; dim_out];
    ternary_matvec(&mut s, &input, &weights, dim_out, dim_in, 1.0);

    set_force_scalar(false);
    assert_eq!(
        active_path_name(),
        default_path,
        "toggle must be reversible"
    );

    // Close, not identical: relative error under 1e-4 per element.
    for i in 0..dim_out {
        let (x, y) = (a[i], s[i]);
        let denom = x.abs().max(1.0);
        assert!(
            (x - y).abs() / denom < 1e-4,
            "scalar vs avx2 diverged at row {i}: {x} vs {y}"
        );
    }
}
