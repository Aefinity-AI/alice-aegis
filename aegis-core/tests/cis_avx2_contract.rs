//! The AVX2 kernel must reject exactly what the reference rejects.
//!
//! `cis_avx2::ternary_matvec_i8_avx2` claims to be "byte-identical to
//! `cis::ternary_matvec_i8` for every input". Identical *arithmetic* is not
//! enough: if the reference panics on an illegal call and the AVX2 path
//! silently returns a number, then the observable behaviour of one binary
//! depends on which CPU it lands on — which is the exact failure mode CIS-1
//! exists to eliminate. These tests pin the rejection surface, not the values.

use aegis_core::cis::ternary_matvec_i8;
use aegis_core::cis_avx2::ternary_matvec_i8_avx2;

/// Did the call panic? Used to compare rejection behaviour, not values.
fn panicked(f: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {})); // keep the test output readable
    let r = std::panic::catch_unwind(f);
    std::panic::set_hook(prev);
    r.is_err()
}

#[test]
fn dim_in_not_multiple_of_four_is_rejected_by_both() {
    // 130 is not a multiple of 4. n_bytes = 32, so the AVX2 path is taken and
    // the reference's assert is never reached unless the wrapper re-checks.
    let (dim_out, dim_in) = (1usize, 130usize);
    let w = vec![0b01_01_01_01u8; dim_out * dim_in / 4 + 1];
    let inp = vec![1i8; dim_in];

    let s = panicked(|| {
        let mut o = vec![0i32; dim_out];
        ternary_matvec_i8(&mut o, &inp, &w, dim_out, dim_in);
    });
    let a = panicked(|| {
        let mut o = vec![0i32; dim_out];
        ternary_matvec_i8_avx2(&mut o, &inp, &w, dim_out, dim_in);
    });
    assert_eq!(
        s, a,
        "dim_in=130: reference panicked={s}, avx2 panicked={a} — the rejection \
         surface must not depend on the ISA"
    );
}

#[test]
fn same_violation_behaves_the_same_either_side_of_the_block_threshold() {
    // dim_in=126 -> n_bytes=31 -> fallback (assert reachable)
    // dim_in=130 -> n_bytes=32 -> SIMD     (assert bypassed unless re-checked)
    // Both violate the same precondition, so both must behave the same way.
    let mut behaviours = Vec::new();
    for dim_in in [126usize, 130usize] {
        let w = vec![0b01_01_01_01u8; dim_in / 4 + 2];
        let inp = vec![1i8; dim_in];
        behaviours.push(panicked(|| {
            let mut o = vec![0i32; 1];
            ternary_matvec_i8_avx2(&mut o, &inp, &w, 1, dim_in);
        }));
    }
    assert_eq!(
        behaviours[0], behaviours[1],
        "the same illegal dim_in behaves differently either side of the 32-byte \
         block threshold: n_bytes=31 panicked={}, n_bytes=32 panicked={}",
        behaviours[0], behaviours[1]
    );
}

#[test]
fn short_output_slice_is_rejected_by_both() {
    let (dim_out, dim_in) = (4usize, 512usize);
    let w = vec![0b01_01_01_01u8; dim_out * dim_in / 4];
    let inp = vec![1i8; dim_in];

    let s = panicked(|| {
        let mut o = vec![0i32; 2]; // shorter than dim_out
        ternary_matvec_i8(&mut o, &inp, &w, dim_out, dim_in);
    });
    let a = panicked(|| {
        let mut o = vec![0i32; 2];
        ternary_matvec_i8_avx2(&mut o, &inp, &w, dim_out, dim_in);
    });
    assert_eq!(
        s, a,
        "short output: reference panicked={s}, avx2 panicked={a}"
    );
}

#[test]
fn short_weights_are_rejected_before_anything_is_written() {
    // The reference writes nothing when it rejects. The AVX2 path must not
    // leave a partially-computed output behind either.
    let (dim_out, dim_in) = (4usize, 512usize);
    let w = vec![0b01_01_01_01u8; dim_out * dim_in / 4 - 8]; // too short
    let inp = vec![1i8; dim_in];

    let mut o = vec![-1i32; dim_out];
    let _ = panicked({
        let w = w.clone();
        let inp = inp.clone();
        let o_ptr = o.as_mut_ptr() as usize;
        move || {
            // SAFETY: single-threaded test; the slice outlives the closure and
            // is not aliased elsewhere while the closure runs.
            let o = unsafe { std::slice::from_raw_parts_mut(o_ptr as *mut i32, dim_out) };
            ternary_matvec_i8_avx2(o, &inp, &w, dim_out, dim_in);
        }
    });
    assert!(
        o.iter().all(|&v| v == -1),
        "a rejected call must not write to output; got {o:?}"
    );
}

#[test]
fn headroom_ceiling_is_enforced_by_both() {
    // cis.rs asserts dim_in <= i32::MAX/127 so the i32 dot stays exact. Past
    // it the accumulator wraps — silently, in the kernel whose whole purpose
    // is exactness.
    let ceiling = (i32::MAX / 127) as usize;
    let dim_in = (ceiling + 4) & !3usize; // just past, still a multiple of 4
    let dim_out = 1usize;
    let w = vec![0b01_01_01_01u8; dim_out * dim_in / 4];
    let inp = vec![127i8; dim_in];

    let s = panicked(|| {
        let mut o = vec![0i32; dim_out];
        ternary_matvec_i8(&mut o, &inp, &w, dim_out, dim_in);
    });
    let a = panicked(|| {
        let mut o = vec![0i32; dim_out];
        ternary_matvec_i8_avx2(&mut o, &inp, &w, dim_out, dim_in);
    });
    assert_eq!(
        s, a,
        "headroom ceiling: reference panicked={s}, avx2 panicked={a} — an \
         unenforced ceiling means a silent i32 wrap"
    );
}

#[test]
fn force_scalar_reaches_this_kernel_too() {
    // Every other dispatch site in the crate gates on ops::simd_on(), which
    // honours set_force_scalar. The in-boot same-binary A/B depends on that
    // toggle reaching every path. Verified positively: with the toggle on, the
    // kernel must still produce the reference's answer.
    let (dim_out, dim_in) = (3usize, 1024usize);
    let mut rng = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let w: Vec<u8> = (0..dim_out * dim_in / 4)
        .map(|_| (next() & 0xFF) as u8)
        .collect();
    let inp: Vec<i8> = (0..dim_in)
        .map(|_| ((next() % 255) as i32 - 127) as i8)
        .collect();

    let mut want = vec![0i32; dim_out];
    ternary_matvec_i8(&mut want, &inp, &w, dim_out, dim_in);

    for forced in [false, true] {
        aegis_core::ops::set_force_scalar(forced);
        let mut got = vec![0i32; dim_out];
        ternary_matvec_i8_avx2(&mut got, &inp, &w, dim_out, dim_in);
        assert_eq!(want, got, "force_scalar={forced}: output diverged");
    }
    aegis_core::ops::set_force_scalar(false);
}
