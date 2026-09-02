//! AEFINITY OS phase 5 — LAB: the job kinds beyond `PROMPT`/`BENCH`.
//!
//! Spec: `program/AEFINITY_OS.md` §9 (phase 5), build contract
//! `program/AEFINITY_OS_FLEET_DESIGN.md` §2 (`JOB.TXT` additions), §2.1
//! (`EVAL`), §3 (`RESULT.TXT` additions), §4 (this module).
//!
//! Phase 5 adds **no protocol**: the lab suites become `JOB.TXT` directives
//! and their answers become `job.N.*` blocks. Everything here is pure
//! computation over the resident artifacts plus files the file plane of phase
//! 4 already put on the volume, so `LAB` composes with `FILES` without either
//! knowing about the other.
//!
//! Rule A note: nothing in this module produces a rate. `MEMBW`'s bandwidth
//! number is gated exactly like `tps` — `n/a` unless `rate_valid=true`, which
//! is `false` on every `env=vm` record — while its *checksum* is comparable
//! across boxes and is always emitted. `EVAL`'s `nll_q16` is exact-integer by
//! construction (§2.1) and is therefore comparable across boxes; it is not a
//! performance figure and never becomes one.

use alloc::format;

use aegis_core::inference::TernaryInferenceEngine;
use uefi::proto::media::file::Directory;

// ---------------------------------------------------------------------------
// MECH (design §2: `MECH`, moved last by the dispatcher)
// ---------------------------------------------------------------------------

/// The OS-advantage mechanism experiment, lifted verbatim out of `main.rs`.
///
/// **Pure move.** Every `boot_log` line, every console write and their order
/// are what `main.rs` emitted before phase 5; the only changes are the paths
/// (`crate::`) that a submodule needs and the `root`/`engine` borrows that
/// were locals there and are parameters here. `cargo xtask boot-test` still
/// reporting its 33 checks is the evidence that the no-`JOB.TXT` boot path is
/// unchanged.
pub fn mech(root: &mut Directory, engine: &mut TernaryInferenceEngine<'_>) {
    // ---- MECH: OS-advantage mechanism experiment, hands-off, one boot ------
    // Decomposes the 2026-07-31 Band-3 result (minimal Linux faster than this
    // unikernel) into named mechanisms:
    //   H1 — per-token firmware console output inside the timed region
    //        (LOUD replicates the protocol path; QUIET buffers and prints
    //        after timing stops; LOUD-QUIET = the console's share).
    //   H2 — turbo-bin ceiling from hot-parked APs (AP-PARK sends them to
    //        MWAIT C6, then QUIET2 reruns; clock%% delta = cpuidle's share).
    // Greedy decode makes all three passes token-identical — every RESPONSE
    // line is its own bit-exactness gate.
    {
        crate::boot_log(
            root,
            "==== MECH v1 (2026-08-01): H1 console / H2 cpuidle turbo-bin ====",
        );
        if let Some(raw) = crate::cpu::turbo_ratio_limit_raw() {
            crate::boot_log(
                root,
                &format!(
                    "MECH MSR_TURBO_RATIO_LIMIT raw=0x{:016x} (byte0=1C bin, byte1=2C bin)",
                    raw
                ),
            );
        }
        #[cfg(feature = "qemu-test")]
        const MECH_MAX: usize = 8;
        #[cfg(not(feature = "qemu-test"))]
        const MECH_MAX: usize = 256;
        const MECH_PROMPTS: [&str; 3] = ["hello alice", "how are you today?", "continue"];

        macro_rules! mech_run {
            ($tag:expr, $prompt:expr, $loud:expr) => {{
                let mut ntok: u64 = 0;
                // Preallocated BEFORE the timed region so QUIET adds no
                // allocation cost that LOUD does not have.
                let mut quiet_buf = alloc::string::String::with_capacity(8192);
                let t0 = unsafe { core::arch::x86_64::_rdtsc() };
                let g0 = crate::cpu::perf_snapshot();
                let response = engine.process_intent($prompt, MECH_MAX, |token_str| {
                    if !token_str.starts_with("[SYSTEM]") && !token_str.contains("[PERFORMANCE]") {
                        ntok += 1;
                    }
                    if $loud {
                        let _ = crate::console::with_console(|st| {
                            let _ = st.write_str(token_str);
                            core::fmt::Result::Ok(())
                        });
                    } else {
                        quiet_buf.push_str(token_str);
                    }
                });
                let g1 = crate::cpu::perf_snapshot();
                let dt = unsafe { core::arch::x86_64::_rdtsc() } - t0;
                // QUIET output appears only after the timed region closes.
                if !$loud {
                    let _ = crate::console::with_console(|st| {
                        let _ = st.write_str(&quiet_buf);
                        let _ = st.write_str("\r\n");
                        core::fmt::Result::Ok(())
                    });
                }
                let clock = match (g0, g1) {
                    (Some(a), Some(b)) => crate::cpu::actual_pct_of_nominal(a, b),
                    _ => None,
                };
                crate::boot_log(
                    root,
                    &format!("MECH {} RESPONSE {:?}: {}", $tag, $prompt, response),
                );
                crate::boot_log(
                    root,
                    &format!(
                        "MECH {} {:?}: {} tokens, {} ticks, {} ticks/token, clock {}",
                        $tag,
                        $prompt,
                        ntok,
                        dt,
                        if ntok > 0 { dt / ntok } else { 0 },
                        match clock {
                            Some(p) => format!("{}%", p),
                            None => alloc::string::String::from("?"),
                        }
                    ),
                );
            }};
        }

        for p in MECH_PROMPTS {
            mech_run!("LOUD", p, true);
        }
        for p in MECH_PROMPTS {
            mech_run!("QUIET", p, false);
        }
        crate::park_aps_for_turbo(root);
        uefi::boot::stall(core::time::Duration::from_millis(200));
        crate::log_throttle_diag(root, "mech-postpark");
        for p in MECH_PROMPTS {
            mech_run!("QUIET2", p, false);
        }

        // ---- MECH v2: preregistered n=N repeats under QUIET2 conditions ----
        // APs are already parked (MWAIT C6) and the console is buffered during
        // the timed region exactly like QUIET/QUIET2. Greedy decode means run 1
        // defines the byte-exact reference; runs 2..N are compared against it,
        // so every repeat doubles as a determinism gate (Rule D).
        {
            #[cfg(feature = "qemu-test")]
            const MECHV2_N: usize = 2;
            #[cfg(not(feature = "qemu-test"))]
            const MECHV2_N: usize = 10;
            crate::boot_log(
                root,
                "==== MECH v2 (2026-08-01): paired repeats under QUIET2 (APs parked) ====",
            );
            // One buffer reused across all runs: clear() keeps the capacity, so
            // the timed region never allocates and heap churn stays at the v1
            // level (one ~1KB response String per run, dropped each iteration).
            let mut quiet_buf = alloc::string::String::with_capacity(8192);
            for p in MECH_PROMPTS {
                let mut reference: Option<alloc::string::String> = None;
                for i in 1..=MECHV2_N {
                    let mut ntok: u64 = 0;
                    quiet_buf.clear();
                    // SAFETY: RDTSC reads the timestamp counter; no memory is
                    // accessed and no CPU state is modified.
                    let t0 = unsafe { core::arch::x86_64::_rdtsc() };
                    let g0 = crate::cpu::perf_snapshot();
                    let response = engine.process_intent(p, MECH_MAX, |token_str| {
                        if !token_str.starts_with("[SYSTEM]")
                            && !token_str.contains("[PERFORMANCE]")
                        {
                            ntok += 1;
                        }
                        quiet_buf.push_str(token_str);
                    });
                    let g1 = crate::cpu::perf_snapshot();
                    // SAFETY: RDTSC reads the timestamp counter; no memory is
                    // accessed and no CPU state is modified.
                    let dt = unsafe { core::arch::x86_64::_rdtsc() } - t0;
                    // Buffered output is NOT echoed to the console here: the
                    // timed region above is identical to QUIET2, and the full
                    // response text is logged once per prompt below.
                    let clock = match (g0, g1) {
                        (Some(a), Some(b)) => crate::cpu::actual_pct_of_nominal(a, b),
                        _ => None,
                    };
                    crate::boot_log(
                        root,
                        &format!(
                            "MECHV2 {:?} run {}/{}: {} tokens, {} ticks, {} ticks/token, clock {}",
                            p,
                            i,
                            MECHV2_N,
                            ntok,
                            dt,
                            if ntok > 0 { dt / ntok } else { 0 },
                            match clock {
                                Some(pct) => format!("{}%", pct),
                                None => alloc::string::String::from("?"),
                            }
                        ),
                    );
                    match &reference {
                        None => {
                            crate::boot_log(
                                root,
                                &format!("MECHV2 RESPONSE {:?}: {}", p, response),
                            );
                            reference = Some(response);
                        }
                        Some(r) => {
                            crate::boot_log(
                                root,
                                &format!(
                                    "MECHV2 EXACT {:?} run {}: {}",
                                    p,
                                    i,
                                    r.as_bytes() == response.as_bytes()
                                ),
                            );
                        }
                    }
                }
            }
        }

        crate::boot_log(root, "==== MECH DONE ====");
    }
}
