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

use aegis_core::cis::rne_div;
use aegis_core::cis_attn::{ExpLut, LOG2E_Q32, exp_neg_q31, log2_u64_q32};
use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, F, QScale64, SCORE_F};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::witness::{WitnessChain, WitnessHeader};

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
///
/// `rearm` covers the whole step, not just the file read: the receipt is a
/// few KiB and was never what could outlast a watchdog window — the replay
/// behind it is three artifact hashes over 1.83 GB followed by `maxtok`
/// decode steps, and that is where design §8 needs the arm.
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
    // Design §8: the replay is the long part — three artifact hashes over
    // 1.83 GB and then `maxtok` decode steps — so the same re-arm hook that
    // covered the receipt read goes into `verifier::run` too. Reading the
    // file was never what could outlast a watchdog window.
    let verdict = crate::verifier::run(model, embed, vocab, &bytes, rearm);
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

// ---------------------------------------------------------------------------
// EVAL (design §2.1) — the AEFCORP1 container and the exact-integer NLL
// ---------------------------------------------------------------------------

/// `AEFCORP1` — the corpus container magic (design §2.1).
pub const CORPUS_MAGIC: [u8; 8] = *b"AEFCORP1";
/// The fixed 64-byte header the magic opens.
pub const CORPUS_HEADER_BYTES: u64 = 64;
/// Tokens one `EVAL` will hold in RAM. `[lo, hi)` is materialised as `u32`s,
/// so this is a 4 MiB ceiling — enough for two thousand windows and small
/// enough that a job cannot exhaust the pool by asking for a slice.
const EVAL_SLICE_MAX: u64 = 1 << 20;

/// `EVAL <NAME> <lo>:<hi>` — integer NLL over the token slice `[lo, hi)` of an
/// `AEFCORP1` corpus on the boot volume (design §2.1).
///
/// **`nll_q16` is bit-exact by construction.** It does *not* call
/// `CisEngine::calculate_perplexity_int`, which computes its NLL with
/// `libm::exp`/`libm::log` on f64 and is not cross-box comparable. This reuses
/// that function's exact-integer half — `logits_int_exact`'s rational
/// `ActScale`, `QScale64::from_ratio`, `exp_neg_q31` — and replaces its float
/// half with `log2_u64_q32` over the i64 softmax denominator. No `f32` or
/// `f64` value exists anywhere in the accumulation, so two conforming boxes
/// produce the same integer or CIS-1 is falsified.
///
/// The seven steps of §2.1, in the code below:
///
/// 1. `logits_int_exact()` fills the exact i64 logits and yields the logit
///    unit as an exact rational at Q.`F`.
/// 2. `m = max(logits)` (exact i64 compare); gaps `d_j = m − L_j ≥ 0`.
/// 3. `g_j = qs.rescale(d_j)` onto the Q.24 score grid, where `qs` is built
///    from that rational by `QScale64::from_ratio` — exact long division with
///    round-to-nearest-even.
/// 4. `e_j = exp_neg_q31(g_j << 8, lut)` — literally SOFTMAX-I step 2; the max
///    element contributes exactly `2^31`.
/// 5. `S = Σ e_j` in i64, ascending vocabulary index (exact: `V · 2^31 < 2^51`).
/// 6. `nll_t = (g_t << 8) + rne_div((log2_u64_q32(S) − (31 << 32)) << 32,
///    LOG2E_Q32)` in Q.32 nats.
/// 7. `Σ nll_t` in `i128` at Q.32 in ascending (window, position) order, then
///    exactly **one** rounding: `nll_q16 = rne_div(total, 1 << 16)`.
///
/// Underflow is declared, not accidental: a logit more than ~21.49 nats below
/// the max contributes `e_j = 0` exactly (the `n ≥ 31` early return in
/// `exp_neg_q31`) — identically on every box — and since step 6 never divides
/// by `e_t`, a target in that tail still gets a finite, correct-by-definition
/// NLL.
#[allow(clippy::too_many_arguments)]
pub fn eval(
    root: &mut Directory,
    slot: &crate::reload::EngineSlot<'_>,
    name: &str,
    lo: u64,
    hi: u64,
    rate_valid: bool,
    over_budget: &dyn Fn() -> bool,
    rearm: &mut dyn FnMut(),
) -> StepResult {
    let mut sr = StepResult::lab("eval", rate_valid);
    // Only for the log line below: how much of the job's budget the setup —
    // the integer model conversion and the streamed corpus pass — spent
    // before the first window's deadline check could run. Nothing in the
    // record depends on it; it is what makes a budget stop at window 0
    // readable in `BOOTLOG.TXT` instead of a guess.
    let w0 = crate::wall_seconds();
    let name = match crate::files::validate_name(name) {
        Ok(n) => n,
        Err(e) => return fail_step(sr, e.slug(), "the corpus name is not a legal <NAME>"),
    };
    // A provisional `<step-input>`: §3.1's form needs the corpus payload
    // digest, and a step that never got to read its corpus has none. The
    // requested slice is what it can honestly serialise, and it is still
    // enough to keep two different failed `EVAL`s from sharing a `merge_key`.
    sr.merge_input = Some(format!("{name}:{lo}:{hi}"));
    if lo >= hi || hi - lo > EVAL_SLICE_MAX {
        return fail_step(
            sr,
            FileErr::BadArgs.slug(),
            "the slice is empty or over the per-step token ceiling",
        );
    }

    // ---- the engine this EVAL scores with -------------------------------
    // Built from the **resident** slices, exactly as `verifier::run` does, so
    // the number is about the bytes the box is holding and not about whatever
    // FAT would hand back on a re-read.
    // The tokenizer's bytes are not needed: a corpus is already token ids.
    let (model, embed, _vocab) = slot.slices();
    let tensors = match SafeTensors::deserialize(model) {
        Ok(t) => t,
        Err(_) => return fail_step(sr, FileErr::Io.slug(), "MODEL.SAF did not parse"),
    };
    let config = match tensors
        .metadata_field("aegis_config")
        .ok()
        .flatten()
        .and_then(|j| ModelConfig::from_json(&j).ok())
    {
        Some(c) => c,
        None => {
            return fail_step(
                sr,
                FileErr::Io.slug(),
                "MODEL.SAF carries no parseable aegis_config",
            );
        }
    };
    let pipeline = match FullBitNetPipeline::new(&tensors, embed, &config) {
        Ok(p) => p,
        Err(_) => return fail_step(sr, FileErr::Io.slug(), "pipeline build failed"),
    };
    let cis_model = match CisModel::new(&pipeline, &config) {
        Ok(m) => m,
        Err(_) => return fail_step(sr, FileErr::Io.slug(), "CIS model conversion failed"),
    };

    // ---- the corpus, validated in §2.1's exact order --------------------
    let corpus = match load_corpus(root, &name, &config, lo, hi, rearm) {
        Ok(c) => c,
        Err(CorpusErr::Bad(why)) => return fail_step(sr, "bad-corpus", why),
        Err(CorpusErr::Args(why)) => return fail_step(sr, FileErr::BadArgs.slug(), why),
        Err(CorpusErr::File(e)) => return fail_step(sr, e.slug(), "could not read the corpus"),
    };

    // §2.1: `W = min(EVAL_WINDOW, config.max_position_embeddings)`. Stride is
    // W — non-overlapping — because a sliding stride would score some
    // positions with more context than others and make the number depend on a
    // parameter nobody would remember.
    let w = crate::job::EVAL_WINDOW
        .min(config.max_position_embeddings)
        .max(2);
    // §3.1: `eval` → `<NAME>:<corpus payload 64hex>:<lo>:<hi>:<W>`.
    sr.merge_input = Some(format!("{name}:{}:{lo}:{hi}:{w}", corpus.payload_sha_hex));

    let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);
    let lut = ExpLut::new();

    // The witness genesis binds the digest to the artifacts, the slice length
    // and the corpus itself: `prompt` is the payload's own sha256, so a fold
    // over one corpus can never be presented as a fold over another.
    let (mh, eh, vh) = {
        let d = slot.digests();
        (
            unhex32(&d.model).unwrap_or([0u8; 32]),
            unhex32(&d.embed).unwrap_or([0u8; 32]),
            unhex32(&d.vocab).unwrap_or([0u8; 32]),
        )
    };
    let header = WitnessHeader {
        model_sha: &mh,
        embed_sha: &eh,
        vocab_sha: &vh,
        max_new: hi - lo,
        prompt: &corpus.payload_sha,
    };
    let mut chain = WitnessChain::from_header(&header);

    crate::boot_log(
        root,
        &format!(
            "JOB: eval setup ready in {} ms (W={w}, {} slice tokens)",
            crate::job::wall_ms_between(w0, crate::wall_seconds()),
            corpus.tokens.len()
        ),
    );

    let mut total_q32: i128 = 0;
    let mut scored: u64 = 0;
    let mut windows_done: u64 = 0;
    let mut partial: Option<u64> = None;

    for win in corpus.tokens.chunks(w) {
        // A final short window of fewer than 2 tokens is dropped: it has no
        // scored position (the first token of a window is context, never a
        // prediction), so it can contribute nothing.
        if win.len() < 2 {
            break;
        }
        // Re-armed per window (design §8): one window of 2048 positions can
        // outlast any single watchdog interval on a slow box.
        rearm();
        if over_budget() {
            partial = Some(windows_done);
            break;
        }
        // Each window resets the KV cache and runs positions `0..len`, so a
        // window's score does not depend on where in the corpus it fell.
        engine.reset_prefix(win.len());
        engine.forward_step_int(win[0], 0);
        for i in 0..win.len() - 1 {
            let target = win[i + 1] as usize;
            // ---- §2.1 step 1: exact rational, never the f64 --------------
            let s = engine.logits_int_exact();
            let logits = engine.logits();
            chain.fold_step(win[i + 1], logits);
            // ---- step 2 -------------------------------------------------
            let m = logits.iter().copied().max().unwrap_or(0);
            // ---- step 3: the Q.24 score grid, by exact long division -----
            // A logit unit is `num / (den · 2^F)` real; the score grid is
            // Q.SCORE_F, so one gap unit is `num · 2^SCORE_F / (den · 2^F)`.
            let qs = QScale64::from_ratio(s.num << SCORE_F, s.den << F);
            // ---- steps 4 and 5 ------------------------------------------
            let mut sum: i64 = 0;
            for &l in logits.iter() {
                // `m` is the maximum, so `m - l >= 0` and the rescale of a
                // non-negative by a non-negative scale is non-negative. The
                // clamp is belt: `exp_neg_q31` takes a u128 and a negative
                // cast would wrap into the underflow tail silently.
                let g = qs.rescale(m - l).max(0); // Q.24
                sum += exp_neg_q31((g as u128) << 8, &lut);
            }
            // The max element contributes exactly 2^31, so `sum ≥ 2^31` and
            // step 6's subtraction of `31 << 32` cannot go negative.
            let g_t = qs.rescale(m - logits[target]).max(0);
            // ---- step 6: Q.32 nats, no float anywhere -------------------
            let log2_s = log2_u64_q32(sum as u64) as i128;
            let ln_ratio = rne_div((log2_s - (31i128 << 32)) << 32, LOG2E_Q32 as i128);
            total_q32 += ((g_t as i128) << 8) + ln_ratio;
            scored += 1;

            engine.forward_step_int(win[i + 1], i + 1);
        }
        windows_done += 1;
    }

    // ---- step 7: exactly one rounding -----------------------------------
    sr.nll_q16 = Some(rne_div(total_q32, 1 << 16).max(0) as u64);
    sr.ntok = Some(scored);
    sr.items = Some(windows_done);
    sr.digest = crate::files::hex64(&chain.digest());
    match partial {
        Some(k) => {
            sr.partial = Some(k);
            sr.pass = Some(false);
            // `budget` is not a §1.3 wire slug — no verb can return it — but
            // it is what `job.N.err` has to say when a window boundary is
            // where the job ran out of time. The accumulation above is over
            // exactly the windows that finished, so the record is prefix
            // evidence and still deterministic (design §5).
            sr.err = Some(String::from(crate::job::BUDGET_ERR));
            sr.detail = Some(format!(
                "budget spent after {k} of {} windows; nll is over those {k}",
                corpus.windows(w)
            ));
        }
        None => {
            sr.partial = Some(0);
            sr.pass = Some(true);
            sr.err = Some(String::from("none"));
            sr.detail = Some(format!(
                "{scored} scored positions in {windows_done} window(s) of W={w}, \
                 corpus ntok={} vocab={}",
                corpus.ntok, config.vocab_size
            ));
        }
    }
    sr
}

/// A validated `AEFCORP1` corpus slice.
struct Corpus {
    /// Total tokens the container declares.
    ntok: u64,
    /// sha256 of the whole payload, and its hex rendering for `merge_key`.
    payload_sha: [u8; 32],
    payload_sha_hex: String,
    /// The `[lo, hi)` slice, materialised.
    tokens: Vec<u32>,
}

impl Corpus {
    /// Windows of `w` the slice would produce if nothing stopped it — the
    /// denominator `job.N.partial` is a numerator of.
    fn windows(&self, w: usize) -> usize {
        self.tokens.chunks(w).filter(|c| c.len() >= 2).count()
    }
}

/// Why a corpus was refused.
enum CorpusErr {
    /// Design §2.1: every container-validation failure is one slug,
    /// `bad-corpus`, whichever check caught it. The reason is in
    /// `job.N.detail`; the slug is what a scheduler branches on, and "the
    /// corpus is not usable" is one decision however it was reached.
    Bad(&'static str),
    /// The slice, not the container: `[lo, hi)` outside the corpus.
    Args(&'static str),
    /// The volume, not the file's content.
    File(FileErr),
}

impl From<FileErr> for CorpusErr {
    fn from(e: FileErr) -> CorpusErr {
        CorpusErr::File(e)
    }
}

/// Read and validate an `AEFCORP1` container, returning the `[lo, hi)` slice.
///
/// Validation order is design §2.1's, exactly: magic; `version == 1`;
/// `token_width == 4`; `file_size == 64 + 4·ntok`; `vocab_size ==
/// engine.config.vocab_size`; then payload sha256 **and** every id `< vocab`
/// in one streamed pass — all before any forward pass, and all reported as
/// `bad-corpus`.
fn load_corpus(
    root: &mut Directory,
    name: &str,
    config: &ModelConfig,
    lo: u64,
    hi: u64,
    rearm: &mut dyn FnMut(),
) -> Result<Corpus, CorpusErr> {
    let mut bounce = crate::files::Bounce::new().ok_or(FileErr::Io)?;
    let mut reader = crate::files::Reader::open(root, name)?;
    let size = reader.size;
    if size < CORPUS_HEADER_BYTES {
        reader.close();
        return Err(CorpusErr::Bad("file is shorter than the 64-byte header"));
    }
    let mut hdr = [0u8; 64];
    {
        let mut got = 0usize;
        while got < 64 {
            match reader.next(&mut hdr[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) => {
                    reader.close();
                    return Err(CorpusErr::File(e));
                }
            }
        }
        if got != 64 {
            reader.close();
            return Err(CorpusErr::Bad("short read on the header"));
        }
    }
    let u32at = |o: usize| u32::from_le_bytes([hdr[o], hdr[o + 1], hdr[o + 2], hdr[o + 3]]);
    let u64at = |o: usize| {
        u64::from_le_bytes([
            hdr[o],
            hdr[o + 1],
            hdr[o + 2],
            hdr[o + 3],
            hdr[o + 4],
            hdr[o + 5],
            hdr[o + 6],
            hdr[o + 7],
        ])
    };
    let bad = |reader: crate::files::Reader, why: &'static str| -> CorpusErr {
        reader.close();
        CorpusErr::Bad(why)
    };
    if hdr[..8] != CORPUS_MAGIC {
        return Err(bad(reader, "magic is not AEFCORP1"));
    }
    if u32at(8) != 1 {
        return Err(bad(reader, "container version is not 1"));
    }
    if u32at(12) != 4 {
        return Err(bad(reader, "token_width is not 4"));
    }
    let ntok = u64at(16);
    let want = CORPUS_HEADER_BYTES.saturating_add(ntok.saturating_mul(4));
    if size != want {
        return Err(bad(reader, "file_size is not 64 + 4*ntok"));
    }
    let vocab_size = u32at(24) as usize;
    if vocab_size != config.vocab_size {
        return Err(bad(reader, "vocab_size does not match the resident model"));
    }
    let mut want_sha = [0u8; 32];
    want_sha.copy_from_slice(&hdr[32..64]);

    if hi > ntok {
        reader.close();
        return Err(CorpusErr::Args("[lo, hi) runs past the corpus"));
    }

    // One streamed pass: sha256 over the payload, every id bounds-checked,
    // and the requested slice collected. `Reader::next` hands back whatever
    // the firmware gave, which need not be a multiple of 4, so ids are
    // assembled a byte at a time across chunk boundaries rather than assuming
    // an alignment the file protocol never promised.
    let mut h = aegis_core::witness::Sha256::new();
    let mut tokens: Vec<u32> = Vec::with_capacity((hi - lo) as usize);
    let mut pending = [0u8; 4];
    let mut have = 0usize;
    let mut index: u64 = 0;
    loop {
        rearm();
        let n = match reader.next(bounce.buf()) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                reader.close();
                return Err(CorpusErr::File(e));
            }
        };
        let chunk = &bounce.buf()[..n];
        h.update(chunk);
        for &byte in chunk {
            pending[have] = byte;
            have += 1;
            if have < 4 {
                continue;
            }
            have = 0;
            let id = u32::from_le_bytes(pending);
            if id as usize >= vocab_size {
                return Err(bad(reader, "a token id is >= vocab_size"));
            }
            if index >= lo && index < hi {
                tokens.push(id);
            }
            index += 1;
        }
    }
    reader.close();
    if have != 0 || index != ntok {
        return Err(CorpusErr::Bad("payload is not 4*ntok bytes"));
    }
    let got_sha = h.finalize();
    if got_sha != want_sha {
        return Err(CorpusErr::Bad("payload sha256 does not match the header"));
    }
    Ok(Corpus {
        ntok,
        payload_sha: got_sha,
        payload_sha_hex: crate::files::hex64(&got_sha),
        tokens,
    })
}

// ---------------------------------------------------------------------------
// MEMBW (design §2, §3)
// ---------------------------------------------------------------------------

/// Hard ceiling on `MEMBW <mib>`, independent of how much memory is free.
const MEMBW_MAX_MIB: u64 = 4096;
/// `u64` words in one MiB.
const WORDS_PER_MIB: u64 = (1 << 20) / 8;
/// The pattern's odd multiplier (the 64-bit golden ratio). Odd, so the map
/// `i -> i * K` is a bijection on `u64` and no two words share a value by
/// construction — a checksum over it therefore witnesses *every* word.
const MEMBW_K: u64 = 0x9E37_79B9_7F4A_7C15;

/// `MEMBW <mib>` — a deterministic touch pattern over `<mib>` MiB.
///
/// Two facts come out of this, and they are gated differently on purpose
/// (design §3):
///
/// * `job.N.digest` is the checksum of the touched pattern. It depends only
///   on `mib`, so it **is** comparable across boxes: two boxes that disagree
///   here disagree about arithmetic or about memory, and that is a finding.
///   It is always emitted.
/// * `job.N.membw_mibs` is a bandwidth number, and is gated exactly like
///   `tps`: `n/a` unless `rate_valid` — which is `false` on every `env=vm`
///   record. A MiB/s figure taken under TCG models no cache hierarchy and no
///   memory latency, and CLAUDE.md Rule A forbids recording it at all.
///
/// The pattern is a sequential write pass followed by a sequential read pass
/// that folds the checksum. Sequential is the honest shape for a bandwidth
/// probe — it is what a streaming kernel does — and the read pass consumes
/// the write pass, so neither can be optimised away without changing the
/// digest.
///
/// Both passes are cut into one-MiB chunks so that `rearm` and `over_budget`
/// are consulted at every chunk boundary (design §8), exactly as `EVAL` does
/// per window and as the file plane does per `XFER_CHUNK`. `MEMBW 4096` moves
/// 8 GiB of scalar traffic; on a slow box that is minutes of work, and
/// without the re-arm the firmware would reset a machine that is healthy and
/// making progress. A MiB is the natural granularity here because it is the
/// directive's own unit — `job.N.items` counts the same MiB.
///
/// A budget stop is filed the way `EVAL`'s is: `pass=false`, `err=budget`,
/// `partial` = the MiB actually folded, and the digest is the fold over
/// exactly that prefix — deterministic, and readable as prefix evidence
/// rather than as the full-run checksum, which only `partial=0` with
/// `err=none` claims. No `membw_mibs` is computed for a stopped run: a rate
/// over a truncated pass is not the rate the directive asked for.
pub fn membw(
    mib: u64,
    rate_valid: bool,
    over_budget: &dyn Fn() -> bool,
    rearm: &mut dyn FnMut(),
) -> StepResult {
    let mut sr = StepResult::lab("membw", rate_valid);
    // §3.1: `membw` → `<mib decimal>`.
    sr.merge_input = Some(format!("{mib}"));
    if mib == 0 || mib > MEMBW_MAX_MIB {
        return fail_step(sr, FileErr::BadArgs.slug(), "mib must be 1..=MEMBW_MAX_MIB");
    }
    let bytes = mib << 20;
    // Bounded by free memory, and by half of it rather than all: this box is
    // also holding a 1.83 GB engine and a listener, and a `MEMBW` that
    // succeeded by exhausting the pool would take the box out to answer one
    // directive.
    let free = crate::server::free_pool_bytes();
    if free != 0 && bytes > free / 2 {
        return fail_step(
            sr,
            FileErr::BadArgs.slug(),
            "mib is over half of the free conventional memory",
        );
    }
    let words = (mib * WORDS_PER_MIB) as usize;
    let mut buf: Vec<u64> = Vec::new();
    if buf.try_reserve_exact(words).is_err() {
        return fail_step(
            sr,
            FileErr::BadArgs.slug(),
            "the allocator refused a buffer that size",
        );
    }

    // One MiB of `u64`s: the chunk both passes step in, so the watchdog is
    // re-armed and the budget re-read `mib` times per pass (design §8).
    let chunk = WORDS_PER_MIB as usize;

    let w0 = crate::wall_seconds();
    let mut written: usize = 0;
    while written < words {
        rearm();
        if over_budget() {
            break;
        }
        let end = core::cmp::min(written + chunk, words);
        for i in written..end {
            buf.push((i as u64).wrapping_mul(MEMBW_K));
        }
        written = end;
    }
    // The read pass covers exactly what the write pass laid down, so a
    // stopped write does not fold uninitialised memory — there is none: the
    // buffer only ever holds the words that were pushed.
    let mut fold: u64 = aegis_core::cis_infer::FNV1A64_OFFSET;
    let mut folded: usize = 0;
    while folded < written {
        rearm();
        if over_budget() {
            break;
        }
        let end = core::cmp::min(folded + chunk, written);
        for &v in &buf[folded..end] {
            fold = aegis_core::cis_infer::fnv1a64(fold, &v.to_le_bytes());
        }
        folded = end;
    }
    let w1 = crate::wall_seconds();
    drop(buf);

    // What was actually touched, not what a benchmark would quote: the write
    // pass moved `written` words and the read pass `folded`. A run that
    // finished has `written == folded == words`, so this is `2 * bytes` and
    // the digest is the same 64-bit value it has always been for this `mib`.
    let complete = folded == words;
    let done_mib = folded as u64 / WORDS_PER_MIB;
    let moved = ((written + folded) as u64).saturating_mul(8);
    sr.digest = format!("{fold:016x}");
    sr.items = Some(done_mib);
    sr.wall_ms = crate::job::wall_ms_between(w0, w1);
    if complete {
        sr.partial = Some(0);
        sr.pass = Some(true);
        sr.err = Some(String::from("none"));
    } else {
        sr.partial = Some(done_mib);
        sr.pass = Some(false);
        sr.err = Some(String::from(crate::job::BUDGET_ERR));
    }
    // Computed only where it may be quoted, and only where there is something
    // to quote: on `env=vm` this stays `None`, on a run the budget stopped it
    // stays `None` (a rate over a truncated pass is not the answer to
    // `MEMBW <mib>`), and on iron with a firmware whose clock did not move it
    // stays `None` too, because a `0` there would be a bandwidth measurement
    // of zero rather than the absence of one.
    // `render` prints `n/a` for both, and prints the key on every `membw`
    // block either way.
    sr.membw_mibs = if rate_valid && complete && sr.wall_ms > 0 {
        Some(moved / (1 << 20) * 1000 / sr.wall_ms)
    } else {
        None
    };
    sr.detail = Some(if complete {
        format!("touched {mib} MiB twice (write then read+fold), {moved} bytes moved")
    } else {
        format!(
            "budget spent after {done_mib} of {mib} MiB folded; the digest is over that \
             prefix, {moved} bytes moved"
        )
    });
    sr
}
