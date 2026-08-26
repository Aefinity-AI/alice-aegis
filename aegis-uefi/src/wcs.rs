//! `wcslen` for the hard-float UEFI target.
//!
//! WHY THIS EXISTS: `x86_64-uefi-hardfloat.json` enables SSE — that is the
//! entire point of the target (see `build_hardfloat.sh`). With SSE on, LLVM's
//! loop-idiom pass recognizes the `uefi` crate's UTF-16 scan loops
//! (`CStr16` length walks in `File::get_info`, `Directory::read_entry`,
//! `HiiConfigRouting::export`, …) and rewrites them into calls to the C
//! library function `wcslen`. The stock `x86_64-unknown-uefi` target is
//! `+soft-float`, which does not trigger the transform — which is why the
//! correctness build links and the production build did not.
//!
//! Nothing in a UEFI sysroot defines `wcslen`: it is absent from the
//! precompiled `compiler_builtins` on stable AND from the `-Zbuild-std`
//! sources, so `compiler-builtins-mem` does not supply it either. Without this
//! file the hard-float link fails with 15 undefined references to `wcslen`.
//!
//! `read_volatile` is LOAD-BEARING, not defensive. A plain `*s.add(n) != 0`
//! scan here is itself the wcslen idiom, so LLVM rewrites this function's body
//! into a call to this very function — an infinite recursion that links
//! cleanly and hangs the firmware. A volatile load cannot be elided or
//! reassociated, so the pass leaves it alone.

/// UEFI/Windows `wchar_t` is UTF-16, so the unit is `u16`.
///
/// # Safety
/// `s` must point to a NUL-terminated UTF-16 string that stays valid for the
/// whole scan. This is the C contract; every caller here is LLVM itself,
/// lowering a loop that already upheld it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wcslen(s: *const u16) -> usize {
    let mut n: usize = 0;
    // SAFETY: per the contract above, `s.add(n)` is in bounds for every n up to
    // and including the terminator.
    while unsafe { core::ptr::read_volatile(s.add(n)) } != 0 {
        n += 1;
    }
    n
}
