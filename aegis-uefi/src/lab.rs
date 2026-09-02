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
use alloc::string::String;
use alloc::vec::Vec;

use core::arch::x86_64::{__cpuid, __cpuid_count};

use aegis_core::inference::TernaryInferenceEngine;
use uefi::proto::media::file::Directory;

use crate::files::FileErr;
use crate::job::StepResult;

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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Mark a step failed with a §1.3 slug and a human line, and stop.
///
/// `pass=false` and a slug together are what a scheduler reads: `pass` says
/// the step did not succeed, the slug says whether that is the job's fault
/// (`bad-corpus`, `bad-args`) or the box's (`io`, `fw-error`).
fn fail_step(mut sr: StepResult, slug: &str, why: &str) -> StepResult {
    sr.pass = Some(false);
    sr.err = Some(String::from(slug));
    sr.partial = Some(0);
    sr.detail = Some(String::from(why));
    sr
}

/// Read a whole file from the boot volume root through the phase-4 streaming
/// reader and its DMA-reachable bounce buffer, capped at `max` bytes.
///
/// The cap is checked against the file's own size **before** any bytes are
/// pulled, so an oversized file costs one `open` and no memory. `rearm` is
/// called once per chunk (design §8).
fn read_file(
    root: &mut Directory,
    name: &str,
    max: u64,
    rearm: &mut dyn FnMut(),
) -> Result<Vec<u8>, FileErr> {
    let mut bounce = crate::files::Bounce::new().ok_or(FileErr::Io)?;
    let mut reader = crate::files::Reader::open(root, name)?;
    if reader.size > max {
        reader.close();
        return Err(FileErr::BadLen);
    }
    let mut out = Vec::with_capacity(reader.size as usize);
    loop {
        rearm();
        match reader.next(bounce.buf()) {
            Ok(0) => break,
            Ok(n) => out.extend_from_slice(&bounce.buf()[..n]),
            Err(e) => {
                reader.close();
                return Err(e);
            }
        }
    }
    reader.close();
    Ok(out)
}

/// Decode 64 lowercase hex characters into the 32 bytes they name.
///
/// [`crate::reload::Digests`] stores the artifact hashes as hex because that
/// is what every record and every `HEALTH` reply quotes; `WitnessHeader` wants
/// the bytes. Re-hashing 1.83 GB to get them back would be absurd, so this
/// inverts the rendering instead. `None` on anything that is not 64 hex.
fn unhex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let b = s.as_bytes();
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        let hi = (b[i * 2] as char).to_digit(16)?;
        let lo = (b[i * 2 + 1] as char).to_digit(16)?;
        *o = (hi * 16 + lo) as u8;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// CPUID (design §2)
// ---------------------------------------------------------------------------

/// `CPUID` — identity and leaf dump, no measurement.
///
/// The cheapest directive there is, and the one a fleet needs first: a
/// `merge_key` deliberately excludes `cpu_brand` (§3.1) so that two *different*
/// CPUs can share one, and `AGREE` is defined as identical digests from
/// **different** `cpuid_sig`s (§5). Something has to carry the identity the
/// comparison is about, and this is it.
///
/// Everything below is a register read. `pass` is always `true`: CPUID is
/// architecturally present on every CPU this unikernel can boot on, so there
/// is no failure to report — an absent extended leaf is reported as absent,
/// which is a fact about the silicon, not an error.
pub fn cpuid(rate_valid: bool) -> StepResult {
    let mut sr = StepResult::lab("cpuid", rate_valid);
    let mut vendor_buf = [0u8; 12];
    let vendor = crate::cpu::vendor_string(&mut vendor_buf);
    let mut brand_buf = [0u8; 48];
    let brand = crate::cpu::brand_string(&mut brand_buf);
    let (family, model, stepping) = crate::cpu::family_model_stepping();
    let (avx2, fma, sse2) = crate::cpu::identity_feats();

    // Leaf 0 reports the highest basic leaf; leaf 0x8000_0000 the highest
    // extended one. Both are architecturally defined, and `__cpuid` is a
    // register read with no memory operand — safe on this target.
    let l0 = __cpuid(0);
    let l1 = __cpuid(1);
    let lx = __cpuid(0x8000_0000);
    let l7 = if l0.eax >= 7 {
        __cpuid_count(7, 0)
    } else {
        core::arch::x86_64::CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        }
    };
    let lx1 = if lx.eax >= 0x8000_0001 {
        __cpuid(0x8000_0001)
    } else {
        core::arch::x86_64::CpuidResult {
            eax: 0,
            ebx: 0,
            ecx: 0,
            edx: 0,
        }
    };

    let detail = format!(
        "vendor={vendor} brand=\"{brand}\" family={family} model={model} stepping={stepping} \
         env={env} sig={sig:08x} maxleaf={maxb:x}/{maxe:x} \
         l1.ebx={l1ebx:08x} l1.ecx={l1ecx:08x} l1.edx={l1edx:08x} \
         l7_0.ebx={l7ebx:08x} l7_0.ecx={l7ecx:08x} l7_0.edx={l7edx:08x} \
         l8000_0001.ecx={x1ecx:08x} l8000_0001.edx={x1edx:08x} \
         feats=<avx2:{a},fma:{f},sse2:{s}>",
        env = crate::sysinfo::env().as_str(),
        sig = l1.eax,
        maxb = l0.eax,
        maxe = lx.eax,
        l1ebx = l1.ebx,
        l1ecx = l1.ecx,
        l1edx = l1.edx,
        l7ebx = l7.ebx,
        l7ecx = l7.ecx,
        l7edx = l7.edx,
        x1ecx = lx1.ecx,
        x1edx = lx1.edx,
        a = avx2 as u8,
        f = fma as u8,
        s = sse2 as u8,
    );

    sr.pass = Some(true);
    sr.err = Some(String::from("none"));
    sr.items = Some(1);
    sr.partial = Some(0);
    // `escape_capped` at `DETAIL_MAX` happens in `ResultRecord::render`; the
    // string is stored raw so the cap is applied in exactly one place.
    sr.detail = Some(detail);
    // §3.1: `cpuid` has an empty `<step-input>` — the dump is an output, and
    // putting it in the key would mean two boxes could never share one, which
    // is the opposite of what the key is for.
    sr.merge_input = Some(String::new());
    sr
}

// ---------------------------------------------------------------------------
// VERIFY (design §2)
// ---------------------------------------------------------------------------

/// A receipt is a text file of a few KiB. This is the ceiling a `VERIFY` will
/// read; anything larger is not a witness v1 receipt and is refused before the
/// bytes are pulled off the volume.
const RECEIPT_MAX_BYTES: u64 = 1 << 20;

/// `VERIFY <NAME>` — replay a witness receipt through [`crate::verifier::run`].
///
/// The file arrives by `PUT` (phase 4) and the name goes through
/// [`crate::files::validate_name`], so `LAB` composes with `FILES` without
/// either knowing about the other. The verification itself is unchanged
/// machinery: the same `verifier::run` the boot-time `RECEIPT.TXT` path has
/// used since the Provable AI Kit, against the **resident** artifact slices —
/// the bytes the engine is actually holding, never a re-read from FAT.
pub fn verify(
    root: &mut Directory,
    slot: &crate::reload::EngineSlot<'_>,
    name: &str,
    rate_valid: bool,
    rearm: &mut dyn FnMut(),
) -> StepResult {
    let mut sr = StepResult::lab("verify", rate_valid);
    let name = match crate::files::validate_name(name) {
        Ok(n) => n,
        Err(e) => return fail_step(sr, e.slug(), "the receipt name is not a legal <NAME>"),
    };
    // Provisional `<step-input>`: §3.1's form needs the receipt's own digest,
    // which a step that could not read the file does not have. The name alone
    // is what it can honestly serialise.
    sr.merge_input = Some(name.clone());
    let bytes = match read_file(root, &name, RECEIPT_MAX_BYTES, rearm) {
        Ok(b) => b,
        Err(e) => return fail_step(sr, e.slug(), "could not read the receipt"),
    };
    // §3.1: `verify` → `<NAME>:<receipt file 64hex>`. The digest is over the
    // bytes actually replayed, so a controller cannot swap a receipt under a
    // name and have two records still look like replications.
    let receipt_sha = crate::files::hex64(&aegis_core::witness::sha256(&bytes));
    sr.merge_input = Some(format!("{name}:{receipt_sha}"));

    let (model, embed, vocab) = slot.slices();
    let verdict = crate::verifier::run(model, embed, vocab, &bytes);
    sr.pass = Some(verdict.pass);
    sr.items = Some(verdict.items);
    sr.partial = Some(0);
    // A `VERIFY` that ran and said FAIL is a *result*, not a fault of the box:
    // §1.3's slugs describe an unhealthy box, and this box is healthy. So the
    // slug stays `none` and the verdict lives in `pass`.
    sr.err = Some(String::from("none"));
    if let Some(d) = verdict.digest {
        sr.digest = d;
    }
    sr.detail = Some(verdict.detail);
    sr
}
