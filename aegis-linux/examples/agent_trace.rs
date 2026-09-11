//! agent_trace — witness receipts for a small, deterministic AGENT EPISODE:
//! K rounds of {greedy CIS-1 FullInt decode, one scan-for-a-tool-call, run
//! the tool}, hash-chained end to end so a verifier can replay the whole
//! episode bit-for-bit on another machine and detect any altered step,
//! tool input, or tool output.
//!
//! This is the "did an agent DO the right things" receipt, one layer above
//! `cis_witness`'s "did a model SAY the right tokens" receipt. It reuses
//! `cis_witness`'s decode path and `aegis_core::witness`'s chain primitives
//! unchanged; the only new fold is the small per-episode "trace chain" that
//! links each step's already-chained decode digest to that step's tool
//! name/input/output.
//!
//! Tools: `calc`, grammar `CALC(<int> <op> <int>)` with op in {+ - * / %},
//! i64 checked arithmetic; and `lookup`, grammar `LOOKUP(<key>)` with
//! key matching `[A-Za-z0-9_.-]{1,64}`, resolved against a fixed table file
//! supplied with `--table` (a hit returns the table's value string, a miss
//! returns the literal `NONE`). `lookup` only exists when a table is given —
//! with no `--table`, `LOOKUP(...)` text is not scanned for at all and the
//! episode behaves exactly as it did before this tool existed. A step whose
//! decoded text contains no matching call of either kind is `tool=no-tool`.
//! A step whose `CALC` call parses but whose arithmetic overflows or
//! divides/mods by zero is `tool=calc-error` with a fixed error string as
//! output — itself a recorded, deterministic step outcome, not a crash.
//!
//! Scanner policy (same for one tool or two): find the earliest starting
//! occurrence of `CALC(` or `LOOKUP(` in the step's newly decoded text and
//! attempt to parse *only* that occurrence per its own grammar. Exception:
//! if the step's running prompt (trailing whitespace trimmed) itself ends
//! with an unclosed `CALC(` or `LOOKUP(` opener — e.g. a suite that primes
//! the prompt with `"A: CALC("` so the model transcribes only the
//! arguments — that opener is carried as a scan prefix and the newly
//! decoded text is scanned as `prefix + decoded_text`, so a continuation
//! like `"3 + 4)"` closes the call. A prompt with no trailing opener scans
//! exactly as before (prefix is empty). If it fails to parse, the step is
//! `no-tool` — the scanner never falls back to a later occurrence or the
//! other tool. Both tools may appear across one episode (different steps).
//!
//! Each step re-encodes its own growing prompt and decodes from position 0
//! with a fresh engine (no carried KV state across steps) — the simplest
//! thing that is unambiguously deterministic and cheap at this episode size
//! (K=3, N=16 by default).
//!
//!   agent_trace gen    <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <K> <N> ["prompt"] [--table <path>] [--suite-sha256 <64hex>] > receipt
//!   agent_trace verify <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> receipt1 [receipt2 ...] [--table <path>] [--suite-sha256 <64hex>] [--phases] [--fail-fast]
//!
//! Rule A: prints no timing, ever. Rule B: the receipt carries a commit hash
//! and hostname. The commit is captured at BUILD time (`aegis-linux/build.rs`
//! bakes `env!("AEGIS_GIT_COMMIT")` in from the repo the binary was built
//! from) rather than shelled out by `gen` at run time — a runtime `git
//! rev-parse HEAD` depends on the generating process's current working
//! directory and used to silently report either "unknown" (run from
//! outside any checkout) or a different repo's HEAD (run from inside one).
//! `commit_hash()` is now a pure, cwd-independent lookup; `unknown` means
//! the binary itself was built without a resolvable git commit (`gen`
//! refuses to run unless `AEGIS_ALLOW_UNKNOWN_COMMIT=1`, and `verify`
//! prints a WARNING, never a failure, for such a receipt). Through format 2
//! commit/host were informational only — pure decoration, editable in a
//! receipt that still reported VERIFY PASS, as E23 showed. From format 3
//! (`AEGIS-TRACE v2`) on they are folded into the trace genesis (see
//! `trace_genesis`), which closes that hole and has one consequence a
//! reader must not mistake for a failure:
//!
//!   TWO MACHINES RUNNING THE SAME EPISODE PRODUCE DIFFERENT `trace-chain`
//!   VALUES, because their `host` lines (and usually their `commit` lines)
//!   differ. That is correct and intended.
//!
//! Cross-machine bit-identity is therefore asserted over the per-step
//! `decode-chain`, `ctx` and `q` digests, which are provenance-free and
//! MUST match exactly; the `trace-chain` binds a receipt to the box and
//! build that issued it. `verify` re-derives genesis from the receipt's own
//! `commit`/`host` lines, so a receipt from another machine still verifies
//! bit-for-bit there — it is the receipt's own claimed provenance that is
//! now unforgeable, not a value shared between machines.
//!
//! Multi-receipt verify: the three artifacts (MODEL.SAF, EMBED.BIN,
//! VOCAB.BIN) are read and hashed ONCE, and the `SafeTensors` parse,
//! `FullBitNetPipeline`, and `CisModel` (including its one-time LM-head
//! plane pre-conversion) are built ONCE for the whole process — not once
//! per receipt — then every receipt is verified in order against that same
//! model state (`decode_step` still creates a FRESH `CisEngine` per
//! episode step; see its doc comment). With exactly one receipt argument,
//! output is byte-identical to single-receipt verify before this existed:
//! no `==` header, no `SUMMARY` line, exit 0/1 exactly as before. With
//! several receipts: a `== <receipt path>` line precedes each receipt's
//! own PASS/FAIL output, a FAIL does not stop the run (every receipt is
//! attempted), and a final `SUMMARY pass=<n> fail=<m>` line reports the
//! totals; the process exits 1 if any receipt failed, 0 otherwise.
//! `--phases`: with one receipt the table prints right after its PASS
//! line, unchanged from before; with several receipts one table
//! accumulated across every receipt's replay prints once, after
//! `SUMMARY`, regardless of the individual PASS/FAIL outcomes (it is a
//! performance report, not a verify verdict).
//!
//! `--fail-fast` (verify only, rejected for gen the same way `--phases`
//! is): diffs each step against the receipt's claimed step as soon as
//! that step is replayed, instead of after the full K-step replay
//! finishes. On the first divergent step it prints the same `step {i}
//! divergence: ...` line (and any ctx/q mismatch lines) full-mode verify
//! would print for that step, then `VERIFY FAIL — replay diverged from
//! the receipt (fail-fast after step {i})` and stops — the remaining
//! `K - i - 1` steps are never replayed, which is the whole point for a
//! receipt tampered early (see the `step {i} divergence` test). All
//! structural/artifact/table/suite checks that run before the replay are
//! unchanged. On a receipt that verifies clean, `--fail-fast` produces
//! output byte-identical to full mode (nothing diverges, so the
//! early-print path never fires). Without `--fail-fast`, behaviour is
//! byte-identical to before this flag existed.
//!
//! Table binding: when `--table` is given, the table file's sha256 and byte
//! length are folded into the trace genesis (see `trace_genesis`'s doc
//! comment for the exact fold order) and the receipt records a
//! `table-sha256 <64 hex>` header line. A table-less (v0) episode's genesis
//! fold is byte-identical to the pre-LOOKUP code path — no format bump.
//! `verify` recomputes the table's sha256 from the `--table` file it is
//! given and rejects a mismatch (or a missing `--table` when the receipt
//! declares one) as `VERIFY FAIL` before ever running the replay.
//!
//! Suite binding: `--suite-sha256 <64 lowercase hex>` folds an arbitrary
//! caller-supplied 32-byte digest (e.g. the sha256 of an eval suite TSV)
//! into the trace genesis, after the table slot, under its own `b"SUITE"`
//! domain tag. The pre-existing table fold (table_sha then table_len, no
//! tag) is left byte-for-byte unchanged — archived table-bound receipts
//! generated before this flag existed must keep verifying. `gen` validates
//! the hex and writes a `suite-sha256 <64 hex>` header line. `verify` folds
//! a suite-sha256 header exactly as `gen` did; if `--suite-sha256` is also
//! given on the command line and disagrees with the header, verify fails
//! before replay. A receipt with no `suite-sha256` header is unaffected —
//! fully backward compatible.
//!
//! Format 2 — per-step context/query binding: known gap in format 1 (the
//! `AEGIS-TRACE v0` receipts above): a receipt records only the INITIAL
//! prompt, so for a K>1 episode the actual text the model was asked at
//! step 1+ (initial prompt + every prior step's own generation and tool
//! result) is not in the receipt, and a verbatim-argument check — a tool
//! call's argument must appear somewhere in what the model was actually
//! shown — cannot be applied past step 0 (seen on EVAL-60 T1 `mixed_02`/
//! `mixed_03`; see `demo/agent-trace/eval/check_verbatim.py`, which is
//! limited to step 0 for exactly this reason). `gen` now always emits
//! header line `AEGIS-TRACE v1` and adds two fields to every step line:
//! `ctx=<64 hex>`, sha256 of the exact prompt text `decode_step` was
//! given for that step (the full accumulated context), and `q=<64 hex>`,
//! sha256 of that step's own "query text" — the initial prompt at step 0,
//! or the immediately preceding step's tool-result text at step 1+ (the
//! text newly appended to the prompt since the previous step). Grammar,
//! before -> after, one step line:
//!   step 0: toks=5,9,2 tool=calc in=4341...2029 out=32 decode-chain=<hex>
//!   step 0: toks=5,9,2 tool=calc in=4341...2029 out=32 decode-chain=<hex> ctx=<hex> q=<hex>
//! `verify` recomputes both digests from its own replay for every step of
//! a format-2 (`AEGIS-TRACE v1`) receipt and fails with `STEP n CTX
//! MISMATCH` / `STEP n QUERY MISMATCH` on a mismatch (in addition to, not
//! instead of, the pre-existing toks/tool/in/out/decode-chain checks); it
//! also prints one `WARNING step n: ...` line (never a failure) per step
//! whose tool-call argument does not appear verbatim in the externally
//! supplied text seen by that step (the initial prompt's last `Q:` line
//! plus every prior tool result; the model's own generated text is never
//! consulted, so a self-authored `Q:` line cannot satisfy the rule) — the
//! generalization, to every step, of the step-0-only rule
//! `check_verbatim.py` applies from outside the receipt. Neither `ctx=`/
//! `q=` nor the WARNING line are folded into the trace chain (see
//! `StepRecord::ctx_digest`'s doc comment for why); a format-1
//! (`AEGIS-TRACE v0`) receipt verifies exactly as it did before format 2
//! existed — same trace-genesis/trace-fold-step math, same step-line
//! fields expected, no `ctx=`/`q=` comparison attempted — and `verify`
//! prints one extra line, `NOTE: format-1 receipt, per-step query binding
//! not present`, so that omission is visible rather than silent.

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;
use aegis_core::witness::{Sha256, WitnessChain, WitnessHeader, hex_lower, sha256};

/// `--phases`: an Amdahl decomposition of `verify`'s FullInt CIS replay,
/// printed AFTER the normal VERIFY PASS/FAIL line and never changing it —
/// see the module doc comment's `--phases` entry. `phase-timers` feature
/// only; every item in this module is compiled out otherwise, and `main`
/// rejects `--phases` with a one-line error + exit(2) when the feature is
/// off (checked once, not per call site).
///
/// `decode_step` builds a FRESH `CisEngine` per episode step (see its own
/// doc comment), so the per-engine `phase_cycles` this module wants to
/// report on would otherwise be dropped with the engine. This module's
/// thread-local accumulates every step's `PhaseCycles` into one running
/// total for the whole `verify` invocation — single-threaded, single
/// process, so a thread-local is just a static with interior mutability,
/// not an actual concurrency primitive.
#[cfg(feature = "phase-timers")]
mod phases_report {
    use aegis_core::phase_timers::{self, Phase, PhaseCycles};
    use std::cell::RefCell;

    thread_local! {
        static ACC: RefCell<PhaseCycles> = RefCell::new(PhaseCycles::zero());
        /// Total tokens forward-passed across every `decode_step` call in
        /// this process (prefill + generated, every episode step) — tracked
        /// directly from the caller's known prompt/N lengths rather than
        /// derived from `PhaseCycles.total_pairs`, because that counter
        /// deliberately mixes two different call-site kinds (see
        /// `accumulate`'s doc comment).
        static TOKENS: RefCell<u64> = const { RefCell::new(0) };
        /// `(raw_ticks, pairs)` for `CisModel::new_with_options`'s one-time
        /// LM-head plane conversion (`headconv`) — see `accumulate_headconv`.
        /// Timed around the ONE `CisModel::new_with_options` call `main`
        /// makes for the whole process (previously timed once per
        /// `decode_step` call, i.e. K times per episode, before model state
        /// was hoisted out of `decode_step`; `pairs` is now 1 per process
        /// regardless of episode or receipt count). Deliberately NOT inside
        /// any `PhaseCycles`/`Phase::*` slot: that struct's fields are
        /// engine-owned, per decode step, and headconv happens once, before
        /// any engine exists.
        static HEADCONV: RefCell<(u64, u64)> = const { RefCell::new((0, 0)) };
    }

    /// Fold one `decode_step` call's engine-owned `PhaseCycles` into the
    /// process-wide running total, and record how many tokens that step
    /// forward-passed (prompt tokens + N generated).
    pub fn accumulate(pc: &PhaseCycles, tokens_forward: u64) {
        ACC.with(|acc| {
            let mut acc = acc.borrow_mut();
            for i in 0..phase_timers::NUM_PHASES {
                acc.raw[i] = acc.raw[i].wrapping_add(pc.raw[i]);
                acc.pairs[i] = acc.pairs[i].wrapping_add(pc.pairs[i]);
            }
            acc.total_raw = acc.total_raw.wrapping_add(pc.total_raw);
            acc.total_pairs = acc.total_pairs.wrapping_add(pc.total_pairs);
        });
        TOKENS.with(|t| *t.borrow_mut() += tokens_forward);
    }

    /// Fold the (now single, process-wide) `CisModel::new_with_options`
    /// call's LM-head plane conversion span into the `headconv` total.
    pub fn accumulate_headconv(ticks: u64) {
        HEADCONV.with(|c| {
            let mut c = c.borrow_mut();
            c.0 = c.0.wrapping_add(ticks);
            c.1 = c.1.wrapping_add(1);
        });
    }

    /// Calibrate TSC ticks/second against `CLOCK_MONOTONIC` over a >= 200 ms
    /// window. Byte-for-byte the same technique as
    /// `aegis-linux/examples/amdahl_decode.rs::calibrate_tsc_hz` and
    /// `clockstate.rs` before it — duplicated, not imported, because none
    /// of the three is a library and each must stay independently
    /// self-contained.
    fn calibrate_tsc_hz() -> f64 {
        use std::arch::x86_64::_rdtsc;
        use std::time::Instant;
        // SAFETY: rdtsc is unprivileged and always available on x86_64; it
        // reads a counter and has no observable side effects.
        let t0 = Instant::now();
        let c0 = unsafe { _rdtsc() };
        while t0.elapsed().as_millis() < 200 {
            std::hint::spin_loop();
        }
        let c1 = unsafe { _rdtsc() };
        let secs = t0.elapsed().as_secs_f64();
        (c1 - c0) as f64 / secs
    }

    /// Calibrate fenced-pair overhead once, at process startup, before any
    /// replay — same call site discipline `amdahl_decode` uses (see
    /// `phase_timers::calibrate_overhead`'s doc comment for why timing and
    /// use must share a core/scheduling context).
    pub fn calibrate() -> (u64, f64, f64) {
        let (overhead_total, overhead_mean) = phase_timers::calibrate_overhead(200_000);
        let tsc_hz = calibrate_tsc_hz();
        (overhead_total, overhead_mean, tsc_hz)
    }

    /// Print the phases table for whatever has been folded into `ACC` so
    /// far. Format mirrors `amdahl_decode.rs`'s single-line-per-context-length
    /// report so both can be quoted the same way in a report.
    pub fn print_table(overhead_total: u64, overhead_mean: f64, tsc_hz: f64) {
        let pc = ACC.with(|acc| *acc.borrow());
        let tokens_forward = TOKENS.with(|t| *t.borrow());
        let (headconv_raw, headconv_pairs) = HEADCONV.with(|c| *c.borrow());

        let named: [(&str, Phase); 7] = [
            ("gemv", Phase::Gemv),
            ("attn", Phase::Attn),
            ("kv", Phase::Kv),
            ("norm", Phase::Norm),
            ("act", Phase::Act),
            ("lmhead", Phase::LmHead),
            ("sample", Phase::Sample),
        ];

        let total_ticks = phase_timers::corrected(pc.total_raw, pc.total_pairs, overhead_mean);
        let mut named_sum = 0.0f64;
        let mut rows: Vec<(&str, f64, u64)> = Vec::with_capacity(named.len());
        for (label, phase) in named {
            let i = phase as usize;
            let ticks = phase_timers::corrected(pc.raw[i], pc.pairs[i], overhead_mean);
            named_sum += ticks;
            rows.push((label, ticks, pc.pairs[i]));
        }
        let other = (total_ticks - named_sum).max(0.0);

        let pct = |x: f64| {
            if total_ticks > 0.0 {
                100.0 * x / total_ticks
            } else {
                0.0
            }
        };

        println!(
            "# agent_trace --phases: RDTSC/RDTSCP invariant-TSC ticks, NOT core cycles \
             (see aegis-linux/examples/clockstate.rs); tsc_hz below is calibrated against \
             CLOCK_MONOTONIC in this process, not assumed."
        );
        println!(
            "# overhead calibration: {overhead_total} ticks / 200000 fenced pairs, \
             mean {overhead_mean:.3} ticks/pair"
        );
        println!(
            "# NOTE: total_ticks/total_pairs below fold TWO call-site kinds into one \
             counter (aegis_core::phase_timers::PhaseCycles.total_*): one record_total \
             pair per CisEngine::forward_step_int call (every prefill AND generated \
             token) and one per CisEngine::logits_int call (final norm + LM head, \
             generated tokens only). tokens_forward is tracked separately from the \
             caller's known prompt/N lengths, not derived from total_pairs."
        );
        println!("PHASE   TICKS_CORRECTED PAIRS   SHARE_PCT");
        for (label, ticks, pairs) in &rows {
            println!("{label:<7} {ticks:>15.0} {pairs:>7} {:>9.2}", pct(*ticks));
        }
        println!(
            "{:<7} {other:>15.0} {:>7} {:>9.2}",
            "other",
            "-",
            pct(other)
        );
        println!(
            "TOTAL total_ticks={total_ticks:.0} total_pairs={} tokens_forward={tokens_forward} \
             tsc_hz={tsc_hz:.0} overhead_ticks={overhead_total}",
            pc.total_pairs
        );

        // headconv: the one-time LM-head plane conversion inside
        // `CisModel::new_with_options`, timed around the single call `main`
        // makes for the whole process (per-process, not per `decode_step`
        // call — see the `HEADCONV` thread_local's doc comment) — NOT part
        // of `total_ticks`/`total_pairs` above, so it is printed as its own
        // line rather than folded into the PHASE table's rows or its
        // `other` bucket.
        let headconv_ticks = phase_timers::corrected(headconv_raw, headconv_pairs, overhead_mean);
        let headconv_ms = if tsc_hz > 0.0 {
            1000.0 * headconv_ticks / tsc_hz
        } else {
            0.0
        };
        println!(
            "headconv (per-process) ticks={headconv_ticks:.0} pairs={headconv_pairs} ms={headconv_ms:.1}"
        );
    }
}

/// `AEGIS_HEAD_PRECONVERT=0` disables the one-time LM-head i16 hi/lo plane
/// conversion (`CisModel::new_with_options`'s `preconvert_head` argument),
/// keeping the pre-existing on-the-fly BF16 path unconditionally — an A/B
/// switch and a tiny-RAM-host escape hatch (the plane representation costs
/// `+4` bytes per weight over the BF16 table already held in memory). Unset,
/// or set to anything other than `"0"`, leaves preconversion ON (the
/// default `CisModel::new` also uses). Read once per `decode_step` call
/// (cheap; `std::env::var` is not on any hot path here).
fn head_preconvert_enabled() -> bool {
    std::env::var("AEGIS_HEAD_PRECONVERT")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// `AEGIS_PREFILL_BATCH=0` forces `CisEngine::forward_prefill_int` back onto
/// its sequential per-token `forward_step_int` loop instead of the batched
/// ternary-GEMM path (`CisEngine::set_prefill_batch`) — an A/B switch only;
/// both paths are bit-identical by construction. Unset, or set to anything
/// other than `"0"`, leaves batching ON (the engine's own default). Read
/// once per `decode_step` call, same discipline as `head_preconvert_enabled`.
fn prefill_batch_enabled() -> bool {
    std::env::var("AEGIS_PREFILL_BATCH")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

/// Strict hex decode: even length, every byte pair a valid hex digit pair.
/// Used only on receipt-derived (hostile) fields in `verify`; `gen` never
/// parses hex, so its output path is unaffected.
/// First 16 chars of a string for messages; char-safe (never panics on
/// multi-byte input such as an adversarial header value).
fn short16(s: &str) -> String {
    s.chars().take(16).collect()
}

/// Validate a `--suite-sha256`/`suite-sha256` value: exactly 64 lowercase
/// hex characters. Returns the decoded 32 bytes, or an error string naming
/// what was wrong (used for both the CLI argument and the receipt header,
/// with distinct messages at each call site).
fn parse_suite_sha256(s: &str) -> Result<[u8; 32], String> {
    if s.len() != 64 || !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(format!(
            "malformed suite-sha256 (want 64 lowercase hex, got {:?})",
            short16(s)
        ));
    }
    let bytes = unhex(s).map_err(|()| "malformed suite-sha256 hex".to_string())?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// Format-2 verify message: this step's replayed context digest does not
/// match the receipt's claimed `ctx=` field. Pulled out as a pure function
/// (rather than inlined at its one `println!` call site) so its exact
/// wording is unit-testable without capturing stdout.
fn ctx_mismatch_msg(step: usize) -> String {
    format!("STEP {step} CTX MISMATCH")
}

/// Format-2 verify message: this step's replayed query digest does not
/// match the receipt's claimed `q=` field. See `ctx_mismatch_msg`.
fn query_mismatch_msg(step: usize) -> String {
    format!("STEP {step} QUERY MISMATCH")
}

/// Format-2 gen/verify message: this step's tool-call argument was not
/// found verbatim in the step's externally supplied text — a WARNING, never a failure
/// (see the module doc comment's format-2 entry).
fn verbatim_warning_msg(step: usize) -> String {
    format!("WARNING step {step}: tool argument not found verbatim in context")
}

/// Printed once by `verify` for a format-1 (`AEGIS-TRACE v0`) receipt, so
/// the pre-format-2 gap (no per-step query binding to check) is visible in
/// the output rather than silently absent.
const FORMAT1_NOTE: &str = "NOTE: format-1 receipt, per-step query binding not present";

/// Printed once by `verify` for a format-3+ receipt whose `commit` line
/// reads `unknown` — informational, never a failure (see `verify_one`'s
/// provenance block and `commit_hash`'s doc comment). Pulled out as a
/// const, same rationale as `ctx_mismatch_msg` above, so the exact wording
/// is unit-testable without capturing stdout.
const UNKNOWN_COMMIT_WARNING: &str =
    "WARNING: receipt commit is unknown (provenance not pinned to code)";

fn unhex(s: &str) -> Result<Vec<u8>, ()> {
    if !s.len().is_multiple_of(2) {
        return Err(());
    }
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| ()))
        .collect()
}

// ---------------------------------------------------------------------
// calc tool: grammar CALC(<int> <op> <int>), checked i64 arithmetic.
// ---------------------------------------------------------------------

/// The outcome of scanning one piece of decoded text for a tool call.
struct ToolOutcome {
    name: &'static str, // "calc" | "calc-error" | "no-tool"
    input: Vec<u8>,     // matched "CALC(...)" substring, or empty
    output: Vec<u8>,    // decimal result, or a fixed error string, or empty
}

/// Find the first `CALC(...)` call in `text` and parse it. Returns
/// `Some((matched_substring, a, op, b))` on a grammar match, `None` if no
/// `CALC(` occurs or the content after it does not parse as the grammar.
/// Only the FIRST `CALC(` occurrence is tried — this is a scan, not a
/// search over all occurrences.
fn find_calc(text: &str) -> Option<(&str, i64, u8, i64)> {
    let start = text.find("CALC(")?;
    let rest = &text[start + "CALC(".len()..];
    let close = rest.find(')')?;
    let body = &rest[..close];
    let full = &text[start..start + "CALC(".len() + close + 1];

    // Manual scan over bytes (grammar is ASCII-only): int, ws*, op, ws*, int,
    // with optional surrounding/interior whitespace at every boundary — so
    // both "3 + 4" and "3+4" parse, but stray trailing content does not.
    let b = body.as_bytes();
    let mut i = 0usize;
    let skip_ws = |b: &[u8], mut i: usize| {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        i
    };
    let parse_int = |b: &[u8], mut i: usize| -> Option<(i64, usize)> {
        let start = i;
        if i < b.len() && b[i] == b'-' {
            i += 1;
        }
        let digits_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == digits_start {
            return None;
        }
        let s = core::str::from_utf8(&b[start..i]).ok()?;
        let v: i64 = s.parse().ok()?;
        Some((v, i))
    };

    i = skip_ws(b, i);
    let (a, i2) = parse_int(b, i)?;
    i = skip_ws(b, i2);
    if i >= b.len() {
        return None;
    }
    let op = b[i];
    if !matches!(op, b'+' | b'-' | b'*' | b'/' | b'%') {
        return None;
    }
    i += 1;
    i = skip_ws(b, i);
    let (val_b, i3) = parse_int(b, i)?;
    i = skip_ws(b, i3);
    if i != b.len() {
        return None; // trailing garbage before the closing paren
    }
    Some((full, a, op, val_b))
}

/// Checked i64 arithmetic for the calc grammar's five ops. `Err` carries a
/// fixed, deterministic error string — never a crash.
fn eval_calc(a: i64, op: u8, b: i64) -> Result<i64, &'static str> {
    match op {
        b'+' => a.checked_add(b).ok_or("overflow"),
        b'-' => a.checked_sub(b).ok_or("overflow"),
        b'*' => a.checked_mul(b).ok_or("overflow"),
        b'/' => {
            if b == 0 {
                Err("div-by-zero")
            } else {
                a.checked_div(b).ok_or("overflow")
            }
        }
        b'%' => {
            if b == 0 {
                Err("div-by-zero")
            } else {
                a.checked_rem(b).ok_or("overflow")
            }
        }
        _ => Err("bad-op"),
    }
}

// ---------------------------------------------------------------------
// lookup tool: grammar LOOKUP(<key>), key in [A-Za-z0-9_.-]{1,64},
// resolved against a fixed table parsed from a `key<TAB>value` file.
// ---------------------------------------------------------------------

/// `key` grammar for both the `LOOKUP(...)` call and every table row:
/// 1 to 64 ASCII bytes, each alphanumeric, `_`, `.`, or `-`.
fn is_valid_key(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 64
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
}

/// A parsed lookup table: the key/value map plus the sha256 and byte length
/// of the exact file bytes it was parsed from (folded into trace genesis).
#[derive(Debug)]
struct LookupTable {
    map: std::collections::HashMap<String, String>,
    sha256: [u8; 32],
    len: u64,
}

/// Strictly parse a `key<TAB>value` table file: UTF-8, one row per line,
/// blank lines skipped, `key` per `is_valid_key`, `value` non-empty-file-line
/// text up to 256 bytes with no tab or newline (a newline can't occur within
/// one `str::lines()` line by construction; the tab check catches a row with
/// more than one tab, which would otherwise silently fold into value).
/// Duplicate keys and any malformed row are hard errors — no silent drop.
fn parse_table(bytes: &[u8]) -> Result<LookupTable, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "table is not valid UTF-8".to_string())?;
    let mut map = std::collections::HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let lineno = i + 1;
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, '\t');
        let key = parts.next().unwrap_or("");
        let value = match parts.next() {
            Some(v) => v,
            None => return Err(format!("table line {lineno}: no tab separator")),
        };
        if !is_valid_key(key) {
            return Err(format!("table line {lineno}: bad key {key:?}"));
        }
        if value.is_empty() {
            return Err(format!("table line {lineno}: empty value"));
        }
        if value.len() > 256 {
            return Err(format!(
                "table line {lineno}: value exceeds 256 bytes ({})",
                value.len()
            ));
        }
        if value.contains('\t') {
            return Err(format!("table line {lineno}: value contains a tab"));
        }
        if map.contains_key(key) {
            return Err(format!("table line {lineno}: duplicate key {key:?}"));
        }
        map.insert(key.to_string(), value.to_string());
    }
    Ok(LookupTable {
        map,
        sha256: sha256(bytes),
        len: bytes.len() as u64,
    })
}

/// Find the first `LOOKUP(...)` call in `text` and validate its key.
/// Returns `Some((matched_substring, key))` on a grammar match, `None` if no
/// `LOOKUP(` occurs or its body is not a valid key. Only the FIRST
/// `LOOKUP(` occurrence is tried, same scan-not-search policy as `find_calc`.
fn find_lookup(text: &str) -> Option<(&str, &str)> {
    let start = text.find("LOOKUP(")?;
    let rest = &text[start + "LOOKUP(".len()..];
    let close = rest.find(')')?;
    let key = &rest[..close];
    if !is_valid_key(key) {
        return None;
    }
    let full = &text[start..start + "LOOKUP(".len() + close + 1];
    Some((full, key))
}

/// One recognized tool call site in a step's decoded text, before it has
/// been run.
enum ToolCall<'a> {
    Calc(&'a str, i64, u8, i64),
    Lookup(&'a str, &'a str),
}

/// Scan `text` for the earliest-starting `CALC(` or `LOOKUP(` occurrence and
/// attempt to parse only that one. `LOOKUP(` is not scanned for at all when
/// `table_present` is false, so a table-less episode's scan is identical to
/// the pre-LOOKUP `find_calc`-only behavior. See the module doc comment for
/// the full scanner policy (earliest occurrence wins; no fallback).
fn find_tool_call(text: &str, table_present: bool) -> Option<ToolCall<'_>> {
    let calc_pos = text.find("CALC(");
    let lookup_pos = if table_present {
        text.find("LOOKUP(")
    } else {
        None
    };
    let calc_first = match (calc_pos, lookup_pos) {
        (Some(c), Some(l)) => c <= l,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => return None,
    };
    if calc_first {
        find_calc(text).map(|(m, a, op, b)| ToolCall::Calc(m, a, op, b))
    } else {
        find_lookup(text).map(|(m, k)| ToolCall::Lookup(m, k))
    }
}

/// If `prompt` (trailing whitespace trimmed) ends with an unclosed `CALC(`
/// or `LOOKUP(` opener, return that literal opener so the caller can carry
/// it as a scan prefix for the step's decoded text. Otherwise `""`, which
/// makes the caller's scan identical to scanning `decoded_text` alone.
fn prompt_scan_prefix(prompt: &str) -> &'static str {
    let trimmed = prompt.trim_end();
    if trimmed.ends_with("CALC(") {
        "CALC("
    } else if trimmed.ends_with("LOOKUP(") {
        "LOOKUP("
    } else {
        ""
    }
}

/// Run the tool scanner over one step's decoded text, optionally prefixed
/// by an opener carried over from the prompt (see `prompt_scan_prefix`).
/// Deterministic: same prefix, text, and table in, same `ToolOutcome` out,
/// always. `table` is `None` for a table-less episode, in which case
/// `LOOKUP(...)` is never recognized. `prefix` is `""` in the pre-existing
/// (no trailing opener) case, so the scan is byte-identical to scanning
/// `decoded_text` alone.
fn run_tool(prefix: &str, decoded_text: &str, table: Option<&LookupTable>) -> ToolOutcome {
    let scanned: std::borrow::Cow<str> = if prefix.is_empty() {
        std::borrow::Cow::Borrowed(decoded_text)
    } else {
        std::borrow::Cow::Owned(format!("{prefix}{decoded_text}"))
    };
    match find_tool_call(&scanned, table.is_some()) {
        None => ToolOutcome {
            name: "no-tool",
            input: Vec::new(),
            output: Vec::new(),
        },
        Some(ToolCall::Calc(matched, a, op, b)) => match eval_calc(a, op, b) {
            Ok(v) => ToolOutcome {
                name: "calc",
                input: matched.as_bytes().to_vec(),
                output: v.to_string().into_bytes(),
            },
            Err(msg) => ToolOutcome {
                name: "calc-error",
                input: matched.as_bytes().to_vec(),
                output: msg.as_bytes().to_vec(),
            },
        },
        Some(ToolCall::Lookup(matched, key)) => {
            // `table_present` gated the scan above, so this is always Some.
            let table = table.expect("LOOKUP scanned only when a table is present");
            let value = table.map.get(key).map(String::as_str).unwrap_or("NONE");
            ToolOutcome {
                name: "lookup",
                input: matched.as_bytes().to_vec(),
                output: value.as_bytes().to_vec(),
            }
        }
    }
}

/// Strip a matched `CALC(...)`/`LOOKUP(...)` call (as recorded in
/// `ToolOutcome::input`) down to the text between its parentheses — the
/// same extraction `demo/agent-trace/eval/check_verbatim.py`'s `ARG_RE`
/// does on the receipt's `in=` field, kept in sync deliberately: both
/// implementations answer "what did the model actually claim was the
/// argument" from the same matched-call bytes. Returns `None` if `input`
/// is not one well-formed call wrapper (never happens for `calc`/
/// `calc-error`/`lookup` outcomes, whose `input` is always a matched call;
/// only reachable defensively).
fn extract_call_arg(input: &[u8]) -> Option<&str> {
    let s = core::str::from_utf8(input).ok()?;
    for prefix in ["CALC(", "LOOKUP("] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return rest.strip_suffix(')');
        }
    }
    None
}

/// The last line starting with `Q:` in `text`, with the prefix stripped and
/// surrounding whitespace trimmed — `""` if there is none. Mirrors
/// `check_verbatim.py`'s `last_query`, generalized in this module to run
/// against any step's accumulated context (`ctx_digest`'s preimage), not
/// only the initial prompt.
fn last_q_line(text: &str) -> &str {
    text.lines()
        .filter_map(|l| l.strip_prefix("Q:"))
        .next_back()
        .map(str::trim)
        .unwrap_or("")
}

/// The verbatim-argument rule for one step. `external_text` is everything
/// the model was shown that did NOT come from the model itself: the initial
/// prompt's last `Q:` line (step 0's question; the few-shot examples above
/// it are deliberately excluded, as in `check_verbatim.py`) plus every tool
/// result appended so far. The model's own decoded text is never consulted:
/// EVAL-60 T1 `mixed_02` (2026-09-07, format 2 on box1) showed the model
/// writing its own `Q: 2 + 2` line at step 0 and then calling `CALC(2 + 2)`
/// at step 1 — a rule that scans the running prompt's last `Q:` line accepts
/// that self-authored question, which is exactly the case the rule exists to
/// flag. `true` for `no-tool` or a call with no parseable argument text.
///
/// The test is a **substring** test, so the rule is sound but not complete:
/// a `false` (which raises the step's WARNING) proves the argument does not
/// occur anywhere in externally supplied text and therefore came from the
/// model, but a `true` only proves the argument is *some contiguous fragment*
/// of that text. Given `Q: What is part 401?`, `LOOKUP(40)` draws no WARNING.
/// The looseness is the safe direction: tightening to a token match would
/// flag a model that correctly extracts `206` from an external `P-206`, so
/// the rule trades false negatives away from false positives on purpose.
/// `verbatim_rule_is_sound_but_not_complete` pins both directions; changing
/// the rule changes the WARNING set of every receipt already generated, so it
/// is a format change, not a bug fix.
fn verbatim_ok_for(external_text: &str, tool_input: &[u8]) -> bool {
    match extract_call_arg(tool_input) {
        Some(arg) if !arg.is_empty() => external_text.contains(arg),
        _ => true,
    }
}

/// Deterministic text appended to the running prompt after a tool runs.
/// `prompt_k = prompt_{k-1} + decoded_text + tool_result_text`.
fn tool_result_text(outcome: &ToolOutcome) -> String {
    format!(
        "\nTOOL[{}]={}\n",
        outcome.name,
        String::from_utf8_lossy(&outcome.output)
    )
}

// ---------------------------------------------------------------------
// Trace chain: one small fold on top of the reused decode-chain primitives.
// ---------------------------------------------------------------------

const TRACE_DOMAIN: &[u8] = b"AEGIS-TRACE v0\n";

/// Genesis value for the trace chain: binds artifact hashes, K, N, and the
/// initial prompt. Deliberately its own domain string, distinct from
/// `WITNESS_DOMAIN_V1`, so a trace-chain digest can never collide with a
/// plain decode-chain digest.
///
/// Fold order (exact, UNCHANGED for the table case from the pre-suite-hash
/// code — this is load-bearing: archived receipts that declare
/// table-sha256, generated before --suite-sha256 existed, must keep
/// verifying byte-for-byte): TRACE_DOMAIN, model_sha, embed_sha, vocab_sha,
/// k (BE u64), n (BE u64), prompt.len() (BE u64), prompt bytes, THEN — only
/// when `table` is `Some((table_sha, table_len))` — table_sha (32 raw
/// bytes) followed by table_len (BE u64), with no tag (exactly as before
/// this fold gained a suite hash), THEN — only when `suite` is
/// `Some(suite_sha)` — the tag `b"SUITE"` followed by suite_sha (32 raw
/// bytes). The `b"SUITE"` tag is the only new domain-separation: it sits
/// after the (untagged) table slot, so a suite hash can never be mistaken
/// for a table hash — the table slot's own position and its trailing
/// table_len already make it unambiguous on its own, and adding a tag
/// there would have changed every existing table-bound receipt's genesis,
/// which is exactly what must not happen. A call with neither `table` nor
/// `suite` folds none of that trailing material, so its digest is
/// byte-identical to the pre-LOOKUP genesis fold (this is what keeps
/// existing v0 receipts verifying unchanged).
fn trace_genesis(
    model_sha: &[u8; 32],
    embed_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    k: u64,
    n: u64,
    prompt: &[u8],
    table: Option<(&[u8; 32], u64)>,
    suite: Option<&[u8; 32]>,
    // `(commit, host)` for format 3 (`AEGIS-TRACE v2`) and later; `None`
    // for the earlier formats, whose genesis bytes must stay exactly as
    // they were so already-issued receipts keep verifying. Folding these
    // in is what stops the `commit`/`host` lines from being editable in a
    // receipt that still reports VERIFY PASS — E23 showed both were pure
    // decoration before.
    provenance: Option<(&[u8], &[u8])>,
) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(TRACE_DOMAIN);
    s.update(model_sha);
    s.update(embed_sha);
    s.update(vocab_sha);
    s.update(&k.to_be_bytes());
    s.update(&n.to_be_bytes());
    s.update(&(prompt.len() as u64).to_be_bytes());
    s.update(prompt);
    if let Some((table_sha, table_len)) = table {
        s.update(table_sha);
        s.update(&table_len.to_be_bytes());
    }
    if let Some(suite_sha) = suite {
        s.update(b"SUITE");
        s.update(suite_sha);
    }
    if let Some((commit, host)) = provenance {
        s.update(b"PROV");
        for field in [commit, host] {
            s.update(&(field.len() as u32).to_le_bytes());
            s.update(field);
        }
    }
    s.finalize()
}

/// Fold one step into the running trace chain. Fields are length-prefixed
/// (LE u32) per the design brief, distinguishing this fold's field
/// encoding from `WitnessHeader`'s (BE u64) — deliberate, not a typo: this
/// is the NEW fold, not a reuse of the header encoding.
fn trace_fold_step(
    chain: [u8; 32],
    step: u64,
    decode_chain_digest: &[u8; 32],
    tool_name: &[u8],
    tool_input: &[u8],
    tool_output: &[u8],
) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(&chain);
    s.update(b"TSTEP");
    s.update(&step.to_be_bytes());
    s.update(decode_chain_digest);
    for field in [tool_name, tool_input, tool_output] {
        s.update(&(field.len() as u32).to_le_bytes());
        s.update(field);
    }
    s.finalize()
}

// ---------------------------------------------------------------------
// Episode replay: the deterministic core both gen and verify run.
// ---------------------------------------------------------------------

struct StepRecord {
    toks: Vec<u32>,
    tool_name: &'static str,
    tool_input: Vec<u8>,
    tool_output: Vec<u8>,
    decode_chain: [u8; 32],
    /// sha256 of the exact prompt text (bytes) fed to `decode_step` for this
    /// step — format-2 only field. Deliberately NOT folded into
    /// `trace_fold_step`/`trace_chain`: `verify` recomputes it from its own
    /// independent replay and compares it directly against the receipt's
    /// claimed `ctx=` field (see `verify_one`), which is exactly as strong
    /// a binding without touching `trace_fold_step`'s inputs — required so
    /// a format-1 receipt's trace-chain math stays byte-for-byte unchanged
    /// (see the module doc comment's format-2 compatibility rule).
    ctx_digest: [u8; 32],
    /// sha256 of this step's "query text": the initial prompt for step 0,
    /// or the previous step's tool-result text for step >= 1 — i.e. the
    /// text newly appended to the running prompt since the previous step,
    /// as opposed to `ctx_digest`'s full accumulated prompt.
    query_digest: [u8; 32],
    /// `true` when `tool_name` is `"no-tool"` (nothing to check) or the
    /// tool call's argument (the text inside `CALC(...)`/`LOOKUP(...)`)
    /// appears verbatim in the externally supplied text the model had seen
    /// by this step: the initial prompt's last `Q:`-prefixed line plus every
    /// prior tool result, never the model's own generated text (see
    /// `verbatim_ok_for`). At step 0 this is exactly
    /// `demo/agent-trace/eval/check_verbatim.py`'s rule. `false` means a
    /// step-2+ verbatim-argument WARNING should print (format-2 only,
    /// never a failure — see the module doc comment's verbatim rule note).
    verbatim_ok: bool,
}

struct EpisodeReplay {
    steps: Vec<StepRecord>,
    trace_chain: [u8; 32],
}

/// Decode N greedy tokens from a FRESH engine seeded with `prompt`'s tokens
/// at position 0 (no KV state carried across steps), folding each token id
/// and its full i64 logit vector into a per-step `WitnessChain` exactly the
/// way `cis_witness` folds decode steps. Returns the decoded token ids and
/// that chain's digest.
///
/// Takes the already-built `&CisModel` — `main` parses `SafeTensors` and
/// builds the `FullBitNetPipeline`/`CisModel` (with its one-time LM-head
/// plane conversion) exactly ONCE per process now, not once per call to
/// this function. This function still creates only a FRESH `CisEngine`
/// per call: the engine (not the model) owns all per-step KV state, so
/// "fresh engine seeded at position 0, no KV carried across steps" is
/// completely unchanged from before model construction was hoisted out —
/// only the repeated (parse, pipeline, LM-head conversion) work is gone.
#[allow(clippy::too_many_arguments)]
fn decode_step(
    cis_model: &CisModel,
    tokenizer: &AegisTokenizer,
    model_sha: &[u8; 32],
    embed_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    prompt: &str,
    n: usize,
) -> (Vec<u32>, [u8; 32]) {
    let mut engine = CisEngine::new_with_mode(cis_model, CisMode::FullInt);
    engine.set_prefill_batch(prefill_batch_enabled());

    let prompt_ids = tokenizer.encode(prompt);
    assert!(!prompt_ids.is_empty(), "step prompt tokenized to nothing");
    assert!(
        prompt_ids.len() + n <= cis_model.config.max_position_embeddings,
        "step prompt ({}) + N ({}) exceeds max_position_embeddings ({})",
        prompt_ids.len(),
        n,
        cis_model.config.max_position_embeddings
    );

    let step_header = WitnessHeader {
        model_sha,
        embed_sha,
        vocab_sha,
        max_new: n as u64,
        prompt: prompt.as_bytes(),
    };
    let mut chain = WitnessChain::from_header(&step_header);

    // Batched prefill (ternary GEMM over token tiles): bit-identical to the
    // sequential `forward_step_int` loop it replaces — see
    // `CisEngine::forward_prefill_int`'s doc for why. `AEGIS_PREFILL_BATCH=0`
    // forces the old sequential loop for A/B.
    engine.forward_prefill_int(&prompt_ids, 0);
    let mut pos = prompt_ids.len();

    let mut generated = Vec::with_capacity(n);
    for _ in 0..n {
        let tok = {
            let logits = engine.decode_logits();
            let t = argmax_i64(logits);
            chain.fold_step(t, logits);
            t
        };
        generated.push(tok);
        engine.forward_step_int(tok, pos);
        pos += 1;
    }
    // Fold this step's engine-owned phase counters into the process-wide
    // `--phases` accumulator before `engine` (and its `phase_cycles`) is
    // dropped at the end of this function. Always runs when the feature is
    // compiled in, independent of whether `--phases` was passed — same
    // "feature gates the cost, the flag only gates printing" discipline
    // `amdahl_decode`/`TernaryInferenceEngine` already use elsewhere.
    // Zero effect on `generated`/`chain.digest()`, i.e. no receipt or
    // verify-decision byte this function returns is touched by it.
    #[cfg(feature = "phase-timers")]
    phases_report::accumulate(&engine.phase_cycles, (prompt_ids.len() + n) as u64);
    (generated, chain.digest())
}

/// Replay the whole K-step episode from the header inputs. Shared by gen
/// and verify — a verifier that calls this and gets the same
/// `EpisodeReplay` as the receipt claims has replayed the episode
/// bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn replay_episode(
    cis_model: &CisModel,
    tokenizer: &AegisTokenizer,
    model_sha: &[u8; 32],
    embed_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    initial_prompt: &str,
    k: usize,
    n: usize,
    table: Option<&LookupTable>,
    suite_sha: Option<&[u8; 32]>,
    provenance: Option<(&str, &str)>,
    // `--fail-fast` hook: called with (step_idx, this step's freshly
    // decoded `StepRecord`) right after the step is produced, before the
    // next step's decode starts. Returning `true` stops the replay after
    // this step (fewer than `k` steps land in the returned
    // `EpisodeReplay`); `gen` and every non-fail-fast `verify` call pass
    // `None`, in which case this is a no-op and the loop runs exactly as
    // it always has — see `verify_one`'s fail-fast branch for the only
    // caller that passes `Some`.
    mut on_step: Option<&mut dyn FnMut(usize, &StepRecord) -> bool>,
) -> EpisodeReplay {
    let mut prompt = initial_prompt.to_string();
    let mut trace_chain = trace_genesis(
        model_sha,
        embed_sha,
        vocab_sha,
        k as u64,
        n as u64,
        initial_prompt.as_bytes(),
        table.map(|t| (&t.sha256, t.len)),
        suite_sha,
        provenance.map(|(c, h)| (c.as_bytes(), h.as_bytes())),
    );
    let mut steps = Vec::with_capacity(k);
    // The text newly appended to the running prompt since the previous
    // step: `None` at step 0 (where the query is the initial prompt
    // itself), then `Some(previous step's tool_result_text)` from step 1
    // on — see `StepRecord::query_digest`'s doc comment.
    let mut prev_tool_result: Option<String> = None;
    // Externally supplied text only: the initial prompt's last `Q:` line
    // plus every tool result so far. Never the model's own decoded text —
    // see `verbatim_ok_for`.
    let mut external_text = last_q_line(initial_prompt).to_string();

    for step_idx in 0..k {
        let query_bytes: &[u8] = match &prev_tool_result {
            None => initial_prompt.as_bytes(),
            Some(tr) => tr.as_bytes(),
        };
        let ctx_digest = sha256(prompt.as_bytes());
        let query_digest = sha256(query_bytes);

        let (toks, decode_chain) = decode_step(
            cis_model, tokenizer, model_sha, embed_sha, vocab_sha, &prompt, n,
        );
        let decoded_text = tokenizer.decode(&toks);
        let prefix = prompt_scan_prefix(&prompt);
        let outcome = run_tool(prefix, &decoded_text, table);

        trace_chain = trace_fold_step(
            trace_chain,
            step_idx as u64,
            &decode_chain,
            outcome.name.as_bytes(),
            &outcome.input,
            &outcome.output,
        );

        let verbatim_ok = verbatim_ok_for(&external_text, &outcome.input);

        let tool_result = tool_result_text(&outcome);
        prompt = prompt + &decoded_text + &tool_result;
        external_text.push_str(&tool_result);
        prev_tool_result = Some(tool_result);

        let record = StepRecord {
            toks,
            tool_name: outcome.name,
            tool_input: outcome.input,
            tool_output: outcome.output,
            decode_chain,
            ctx_digest,
            query_digest,
            verbatim_ok,
        };
        let stop = on_step
            .as_mut()
            .map(|cb| cb(step_idx, &record))
            .unwrap_or(false);
        steps.push(record);
        if stop {
            break;
        }
    }

    EpisodeReplay { steps, trace_chain }
}

/// Pure bounds checks shared by `validate_receipt_header`: K, N, and the
/// prompt/N fit against `max_position_embeddings`. Split out from the
/// model/tokenizer parsing so it can be unit-tested with synthetic numbers
/// (no model fixture required) — in particular the "N too large" case.
fn check_header_bounds(
    k: usize,
    n: usize,
    prompt_tokens: usize,
    max_position_embeddings: usize,
) -> Result<(), String> {
    if k < 1 {
        return Err("K must be >= 1".to_string());
    }
    if n == 0 {
        return Err("N must be > 0".to_string());
    }
    if prompt_tokens == 0 {
        return Err("prompt tokenizes to zero tokens".to_string());
    }
    if prompt_tokens.saturating_add(n) > max_position_embeddings {
        return Err(format!(
            "prompt tokens ({prompt_tokens}) + N ({n}) exceeds max_position_embeddings ({max_position_embeddings})"
        ));
    }
    Ok(())
}

/// Pre-flight checks on receipt-derived (hostile) header fields before
/// `verify` calls `replay_episode`. `replay_episode`'s own asserts are for
/// `gen`'s trusted inputs and are left as-is; this function exists so a
/// malformed receipt fails cleanly instead of panicking. Takes the
/// already-parsed config and tokenizer (built once per process by `main`,
/// same trusted MODEL.SAF/VOCAB.BIN every receipt is checked against)
/// instead of re-parsing them per receipt.
fn validate_receipt_header(
    config: &ModelConfig,
    tokenizer: &AegisTokenizer,
    prompt: &str,
    k: usize,
    n: usize,
) -> Result<(), String> {
    let prompt_ids = tokenizer.encode(prompt);
    check_header_bounds(k, n, prompt_ids.len(), config.max_position_embeddings)
}

// ---------------------------------------------------------------------
// Receipt I/O.
// ---------------------------------------------------------------------

/// The commit this binary was BUILT from, baked in at compile time by
/// `aegis-linux/build.rs` (see its doc comment) — deliberately NOT a
/// runtime `git rev-parse HEAD` in the generating process's current
/// working directory, which used to silently report either "unknown" (run
/// from outside any checkout) or a different repo's HEAD (run from inside
/// one), and then got folded into the trace genesis and TPM-attested as if
/// it were trustworthy.
fn commit_hash() -> String {
    env!("AEGIS_GIT_COMMIT").to_string()
}

/// Validate a receipt "step N:" label against the step's actual position in
/// the file (0-based). `label` is the text before the colon, untrimmed.
fn check_step_label(label: &str, position: usize) -> Result<(), String> {
    match label.trim().parse::<usize>() {
        Ok(n) if n == position => Ok(()),
        Ok(n) => Err(format!("step label {n} at position {position}")),
        Err(_) => Err(format!(
            "step label {} at position {position}",
            label.trim()
        )),
    }
}

fn host_name() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Pull a `--flag value` pair out of `args` (in place) wherever it occurs,
/// leaving the remaining positional args untouched. Used for `--table
/// <path>`, which is optional and orthogonal to the positional gen/verify
/// argument lists.
fn extract_flag(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == flag)?;
    if pos + 1 >= args.len() {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    }
    let val = args.remove(pos + 1);
    args.remove(pos);
    Some(val)
}

/// Pull a bare `--flag` (no value) out of `args` in place, returning whether
/// it was present. Used for `--phases`, which is a switch, not a `--flag
/// value` pair like `extract_flag` handles.
fn extract_bool_flag(args: &mut Vec<String>, flag: &str) -> bool {
    match args.iter().position(|a| a == flag) {
        Some(pos) => {
            args.remove(pos);
            true
        }
        None => false,
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let table_path = extract_flag(&mut args, "--table");
    let suite_sha256_arg = extract_flag(&mut args, "--suite-sha256");
    let phases_flag = extract_bool_flag(&mut args, "--phases");
    let fail_fast_flag = extract_bool_flag(&mut args, "--fail-fast");
    if args.len() < 6 {
        eprintln!(
            "usage: agent_trace gen    <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <K> <N> [prompt] [--table <path>] [--suite-sha256 <64hex>]"
        );
        eprintln!(
            "       agent_trace verify <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> receipt1 [receipt2 ...] [--table <path>] [--suite-sha256 <64hex>] [--phases] [--fail-fast]"
        );
        std::process::exit(2);
    }
    if phases_flag && args[1] != "verify" {
        eprintln!("--phases is only supported by `agent_trace verify`");
        std::process::exit(2);
    }
    if fail_fast_flag && args[1] != "verify" {
        eprintln!("--fail-fast is only supported by `agent_trace verify`");
        std::process::exit(2);
    }
    #[cfg(not(feature = "phase-timers"))]
    if phases_flag {
        eprintln!(
            "--phases requires building with the `phase-timers` feature: \
             cargo build --release --features phase-timers --example agent_trace"
        );
        std::process::exit(2);
    }
    // Calibrate once, at startup, before any replay — required whenever
    // --phases will print a table, regardless of gen/verify (checked above:
    // only verify reaches here with phases_flag true). See
    // `phases_report::calibrate`'s doc comment for why this must happen
    // before the timed work, not after.
    #[cfg(feature = "phase-timers")]
    let phase_calib = phases_flag.then(phases_report::calibrate);
    let suite_sha256_arg = match suite_sha256_arg {
        Some(s) => match parse_suite_sha256(&s) {
            Ok(bytes) => Some(bytes),
            Err(reason) => {
                eprintln!("--suite-sha256: {reason}");
                std::process::exit(2);
            }
        },
        None => None,
    };
    let mode = args[1].as_str();
    let model_bytes = std::fs::read(&args[2]).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(&args[3]).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(&args[4]).expect("read VOCAB.BIN");
    let model_sha = sha256(&model_bytes);
    let embed_sha = sha256(&embed_bytes);
    let vocab_sha = sha256(&vocab_bytes);

    // Parse SafeTensors, build the config/tokenizer, and build the
    // FullBitNetPipeline + CisModel (with its one-time LM-head plane
    // pre-conversion) exactly ONCE per process — previously this whole
    // sequence repeated inside `decode_step`, K times per episode, once per
    // receipt. Every `gen`/`verify` receipt below shares this same model
    // state; `decode_step` still creates only a fresh `CisEngine` per
    // episode step (see its doc comment).
    let tensors = SafeTensors::deserialize(&model_bytes).expect("parse MODEL.SAF");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("read __metadata__")
        .expect("MODEL.SAF carries no aegis_config — repack in the forge");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
    let tokenizer = AegisTokenizer::new(&vocab_bytes).expect("parse VOCAB.BIN");
    let pipeline =
        FullBitNetPipeline::new(&tensors, &embed_bytes, &config).expect("build pipeline");
    #[cfg(feature = "phase-timers")]
    let __headconv_start = aegis_core::phase_timers::tick_start();
    let cis_model = CisModel::new_with_options(&pipeline, &config, head_preconvert_enabled())
        .expect("CIS model conversion");
    #[cfg(feature = "phase-timers")]
    {
        let __headconv_end = aegis_core::phase_timers::tick_end();
        phases_report::accumulate_headconv(__headconv_end.wrapping_sub(__headconv_start));
    }

    match mode {
        "gen" => {
            let k: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
            let n: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(16);
            let prompt = args
                .get(7)
                .map(String::as_str)
                .unwrap_or("Once upon a time");

            let table = table_path.as_ref().map(|p| {
                let bytes = std::fs::read(p).expect("read --table file");
                parse_table(&bytes).expect("parse --table file")
            });

            // Bound into the trace genesis from format 3 on, so a
            // receipt cannot be relabelled with a different commit or host
            // and still verify.
            let commit = commit_hash();
            if commit == "unknown"
                && std::env::var("AEGIS_ALLOW_UNKNOWN_COMMIT").as_deref() != Ok("1")
            {
                eprintln!(
                    "ERROR: this binary was built without a git commit (AEGIS_GIT_COMMIT); set AEGIS_ALLOW_UNKNOWN_COMMIT=1 to generate an unpinned receipt"
                );
                std::process::exit(2);
            }
            let host = host_name();
            let r = replay_episode(
                &cis_model,
                &tokenizer,
                &model_sha,
                &embed_sha,
                &vocab_sha,
                prompt,
                k,
                n,
                table.as_ref(),
                suite_sha256_arg.as_ref(),
                Some((commit.as_str(), host.as_str())),
                None,
            );

            println!("AEGIS-TRACE v2");
            println!("model {}", hex(&model_sha));
            println!("embed {}", hex(&embed_sha));
            println!("vocab {}", hex(&vocab_sha));
            println!("K {k}");
            println!("N {n}");
            if let Some(t) = &table {
                println!("table-sha256 {}", hex(&t.sha256));
            }
            if let Some(s) = &suite_sha256_arg {
                println!("suite-sha256 {}", hex(s));
            }
            println!("prompt-hex {}", hex(prompt.as_bytes()));
            println!("commit {commit}");
            println!("host {host}");
            for (i, s) in r.steps.iter().enumerate() {
                let ids: Vec<String> = s.toks.iter().map(|t| t.to_string()).collect();
                println!(
                    "step {i}: toks={} tool={} in={} out={} decode-chain={} ctx={} q={}",
                    ids.join(","),
                    s.tool_name,
                    hex(&s.tool_input),
                    hex(&s.tool_output),
                    hex(&s.decode_chain),
                    hex(&s.ctx_digest),
                    hex(&s.query_digest)
                );
                if !s.verbatim_ok {
                    println!("{}", verbatim_warning_msg(i));
                }
            }
            println!("trace-chain {}", hex(&r.trace_chain));
        }
        "verify" => {
            let receipt_paths = &args[5..];
            if receipt_paths.len() == 1 {
                let pass = verify_one(
                    &receipt_paths[0],
                    &cis_model,
                    &tokenizer,
                    &model_sha,
                    &embed_sha,
                    &vocab_sha,
                    table_path.as_ref(),
                    suite_sha256_arg,
                    fail_fast_flag,
                );
                if !pass {
                    std::process::exit(1);
                }
                // --phases: printed only after a PASS (a FAIL already
                // exited above), never altering the PASS line or exit code
                // above it — unchanged from before multi-receipt verify
                // existed.
                #[cfg(feature = "phase-timers")]
                if let Some((overhead_total, overhead_mean, tsc_hz)) = phase_calib {
                    phases_report::print_table(overhead_total, overhead_mean, tsc_hz);
                }
            } else {
                let mut n_pass = 0usize;
                let mut n_fail = 0usize;
                for path in receipt_paths {
                    println!("== {path}");
                    let pass = verify_one(
                        path,
                        &cis_model,
                        &tokenizer,
                        &model_sha,
                        &embed_sha,
                        &vocab_sha,
                        table_path.as_ref(),
                        suite_sha256_arg,
                        fail_fast_flag,
                    );
                    if pass {
                        n_pass += 1;
                    } else {
                        n_fail += 1;
                    }
                }
                println!("SUMMARY pass={n_pass} fail={n_fail}");
                // One accumulated --phases table across every receipt this
                // process verified, printed once at the end regardless of
                // individual PASS/FAIL outcomes (a performance report, not
                // a verify verdict) — see the module doc comment's
                // multi-receipt verify entry.
                #[cfg(feature = "phase-timers")]
                if let Some((overhead_total, overhead_mean, tsc_hz)) = phase_calib {
                    phases_report::print_table(overhead_total, overhead_mean, tsc_hz);
                }
                if n_fail > 0 {
                    std::process::exit(1);
                }
            }
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}

/// Diff one replayed step against the receipt's claimed step, printing
/// exactly the lines `verify_one`'s per-step loop has always printed for a
/// mismatching step (the `toks-match=.../decode-chain-match=...` line and,
/// for format >= 2, the ctx/query mismatch lines and the verbatim-argument
/// WARNING). Returns whether this step is a mismatch (the WARNING line is
/// not one — it prints regardless). Factored out so full-mode's
/// after-the-fact loop and `--fail-fast`'s per-step callback (see
/// `verify_one`) call the identical logic and therefore print
/// byte-identical lines for the same divergent step, whichever mode finds
/// it.
#[allow(clippy::type_complexity)]
fn step_diff(
    i: usize,
    local: &StepRecord,
    w_step: &(
        Vec<u32>,
        String,
        String,
        String,
        String,
        Option<String>,
        Option<String>,
    ),
    w_format: u8,
) -> bool {
    let (w_toks, w_tool, w_in, w_out, w_dchain, w_ctx, w_query) = w_step;
    let mut mismatch = false;

    let local_toks_match = &local.toks == w_toks;
    let local_tool_match = local.tool_name == *w_tool;
    let local_in_match = hex(&local.tool_input) == *w_in;
    let local_out_match = hex(&local.tool_output) == *w_out;
    let local_dchain_match = hex(&local.decode_chain) == *w_dchain;
    if !(local_toks_match
        && local_tool_match
        && local_in_match
        && local_out_match
        && local_dchain_match)
    {
        println!(
            "step {i} divergence: toks-match={local_toks_match} tool-match={local_tool_match} in-match={local_in_match} out-match={local_out_match} decode-chain-match={local_dchain_match}"
        );
        mismatch = true;
    }

    // Format-2 only: per-step context/query binding. A format-1 receipt
    // carries no ctx=/q= fields (w_ctx/w_query are `None`) and is
    // otherwise verified exactly as above — see the module doc comment's
    // format-2 entry and the `FORMAT1_NOTE` line already printed above.
    if w_format >= 2 {
        let local_ctx = hex(&local.ctx_digest);
        match w_ctx {
            Some(claimed) if *claimed == local_ctx => {}
            _ => {
                println!("{}", ctx_mismatch_msg(i));
                mismatch = true;
            }
        }
        let local_query = hex(&local.query_digest);
        match w_query {
            Some(claimed) if *claimed == local_query => {}
            _ => {
                println!("{}", query_mismatch_msg(i));
                mismatch = true;
            }
        }
        if !local.verbatim_ok {
            println!("{}", verbatim_warning_msg(i));
        }
    }

    mismatch
}

/// Verify one receipt against the already-hashed artifacts and the
/// process-wide `CisModel`/tokenizer `main` built once (see the module doc
/// comment's multi-receipt verify entry). Prints exactly the lines
/// `agent_trace verify` printed for a single receipt before multi-receipt
/// support existed; returns `true` on `VERIFY PASS`, `false` on any FAIL
/// (structural or replay divergence) instead of exiting the process, so
/// `main` can continue to the next receipt when several were given.
/// `--phases` printing is the caller's responsibility (see `main`), not
/// this function's — the single- and multi-receipt CLI shapes print the
/// table at different points, but neither ever prints it from inside here.
/// `fail_fast`: see the module doc comment's `--fail-fast` entry. `false`
/// reproduces this function's pre-`--fail-fast` behaviour exactly.
#[allow(clippy::too_many_arguments)]
fn verify_one(
    receipt_path: &str,
    cis_model: &CisModel,
    tokenizer: &AegisTokenizer,
    model_sha: &[u8; 32],
    embed_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    table_path: Option<&String>,
    suite_sha256_arg: Option<[u8; 32]>,
    fail_fast: bool,
) -> bool {
    let wtext = std::fs::read_to_string(receipt_path).expect("read receipt");
    let mut w_model = String::new();
    let mut w_embed = String::new();
    let mut w_vocab = String::new();
    let mut w_k = 0usize;
    let mut w_n = 0usize;
    let mut w_prompt = String::new();
    let mut w_steps: Vec<(
        Vec<u32>,
        String,
        String,
        String,
        String,
        Option<String>,
        Option<String>,
    )> = Vec::new();
    let mut w_trace_chain = String::new();
    let mut w_format: u8 = 1;
    // Whether an `AEGIS-TRACE <ver>` magic line was actually seen, and on
    // which line. Without this, an unrecognised or absent header fell
    // through the `_ => {}` arm below and left w_format at its default of
    // 1, silently downgrading a format-2 receipt to format-1 rules — which
    // skip the per-step ctx=/q= binding checks entirely. An attacker who
    // could edit a step line could also edit line 1, so that downgrade was
    // a complete bypass of the format-2 binding. See the
    // `header_downgrade_*` tests.
    let mut w_format_line: Option<usize> = None;
    let mut w_table_sha: Option<String> = None;
    let mut w_suite_sha: Option<String> = None;
    let mut w_commit: Option<String> = None;
    let mut w_host: Option<String> = None;
    let mut w_warn_steps: Vec<usize> = Vec::new();
    let mut seen_keys: Vec<&str> = Vec::new();

    for (line_no, line) in wtext.lines().enumerate() {
        if let Some(rest) = line.strip_prefix("step ") {
            // "IDX: toks=.. tool=.. in=.. out=.. decode-chain=.."
            let (label, body) = rest.split_once(':').unwrap_or((rest, ""));
            let position = w_steps.len();
            if let Err(reason) = check_step_label(label, position) {
                println!("FAIL structure: {reason}");
                return false;
            }
            let rest = body.trim();
            let mut toks: Option<Vec<u32>> = None;
            let mut tool: Option<String> = None;
            let mut input: Option<String> = None;
            let mut output: Option<String> = None;
            let mut dchain: Option<String> = None;
            let mut ctx: Option<String> = None;
            let mut query: Option<String> = None;
            // Every whitespace token on a step line must be a recognised
            // `key=value` field, and each key may appear at most once.
            // This loop previously had no else-branch, so an unrecognised
            // token was silently dropped: `step 0:X toks=...` verified
            // unchanged, and so did any attacker-chosen text appended
            // anywhere on the line. The receipt was therefore not
            // canonical — many distinct byte strings shared one PASSing
            // trace-chain, which is exactly what a receipt must not allow.
            // Found by E23's tamper matrix on cm-box2; see the
            // `step_line_*` tests.
            for field in rest.split_whitespace() {
                let (name, value) = match field.split_once('=') {
                    Some(kv) => kv,
                    None => {
                        println!("FAIL structure: step {position}: stray token {field:?}");
                        return false;
                    }
                };
                let duplicate = match name {
                    "toks" => {
                        let mut ids = Vec::new();
                        for t in value.split(',').filter(|t| !t.is_empty()) {
                            match t.parse::<u32>() {
                                Ok(id) => ids.push(id),
                                Err(_) => {
                                    println!("FAIL structure: step {position}: bad token id {t:?}");
                                    return false;
                                }
                            }
                        }
                        toks.replace(ids).is_some()
                    }
                    "tool" => tool.replace(value.to_string()).is_some(),
                    "in" => input.replace(value.to_string()).is_some(),
                    "out" => output.replace(value.to_string()).is_some(),
                    "decode-chain" => dchain.replace(value.to_string()).is_some(),
                    "ctx" => ctx.replace(value.to_string()).is_some(),
                    "q" => query.replace(value.to_string()).is_some(),
                    other => {
                        println!("FAIL structure: step {position}: unknown field {other:?}");
                        return false;
                    }
                };
                if duplicate {
                    println!("FAIL structure: step {position}: duplicate field {name:?}");
                    return false;
                }
            }
            let (toks, tool, input, output, dchain) = match (toks, tool, input, output, dchain) {
                (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, c, d, e),
                _ => {
                    println!(
                        "FAIL structure: step {position}: missing one of toks= tool= in= out= decode-chain="
                    );
                    return false;
                }
            };
            w_steps.push((toks, tool, input, output, dchain, ctx, query));
            continue;
        }
        let mut it = line.splitn(2, ' ');
        let (key, v) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
        if key.is_empty() {
            continue;
        }
        // Unknown keys used to fall through the `_ => {}` arm below, which
        // made every lead field deletable simply by renaming it — `commit`
        // to `commitX` still verified — and let arbitrary attacker-chosen
        // lines ride inside a receipt that printed VERIFY PASS. E23 found
        // both. The allowlist below is the receipt's complete lead-line
        // vocabulary; anything else is a structural failure.
        const LEAD_KEYS: [&str; 12] = [
            "AEGIS-TRACE",
            "model",
            "embed",
            "vocab",
            "K",
            "N",
            "prompt-hex",
            "trace-chain",
            "table-sha256",
            "suite-sha256",
            "commit",
            "host",
        ];
        if key != "WARNING" && !LEAD_KEYS.contains(&key) {
            println!("FAIL structure: unknown line key {key:?} on line {line_no}");
            return false;
        }
        if key != "WARNING" {
            if seen_keys.contains(&key) {
                println!("FAIL structure: duplicate {key} line");
                return false;
            }
            seen_keys.push(key);
        }
        match key {
            "AEGIS-TRACE" => {
                if w_format_line.is_some() {
                    println!("FAIL structure: duplicate AEGIS-TRACE header line");
                    return false;
                }
                w_format_line = Some(line_no);
                w_format = match v {
                    "v0" => 1,
                    "v1" => 2,
                    "v2" => 3,
                    other => {
                        println!("FAIL structure: unknown AEGIS-TRACE format {other:?}");
                        return false;
                    }
                };
            }
            "model" => w_model = v.into(),
            "embed" => w_embed = v.into(),
            "vocab" => w_vocab = v.into(),
            "K" => match v.parse() {
                Ok(x) => w_k = x,
                Err(_) => {
                    println!("FAIL structure: malformed K {v:?}");
                    return false;
                }
            },
            "N" => match v.parse() {
                Ok(x) => w_n = x,
                Err(_) => {
                    println!("FAIL structure: malformed N {v:?}");
                    return false;
                }
            },
            "prompt-hex" => {
                let bytes = match unhex(v) {
                    Ok(b) => b,
                    Err(()) => {
                        println!("FAIL structure: malformed hex in prompt-hex");
                        return false;
                    }
                };
                w_prompt = match String::from_utf8(bytes) {
                    Ok(s) => s,
                    Err(_) => {
                        println!("FAIL structure: malformed hex in prompt-hex");
                        return false;
                    }
                };
            }
            "trace-chain" => w_trace_chain = v.into(),
            "table-sha256" => {
                if v.len() != 64 || !v.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
                    println!("FAIL structure: malformed table-sha256 (want 64 lowercase hex)");
                    return false;
                }
                w_table_sha = Some(v.into());
            }
            "suite-sha256" => {
                if v.len() != 64 || !v.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
                    println!("FAIL structure: malformed suite-sha256 (want 64 lowercase hex)");
                    return false;
                }
                w_suite_sha = Some(v.into());
            }
            "commit" => w_commit = Some(v.into()),
            "host" => w_host = Some(v.into()),
            // A WARNING line must be exactly what `verbatim_warning_msg`
            // emits for some step; free text on a WARNING line was another
            // way to smuggle attacker-chosen content into a PASSing
            // receipt. The set of warned steps is compared against the
            // replay's own below.
            "WARNING" => {
                let idx = line
                    .strip_prefix("WARNING step ")
                    .and_then(|r| r.split_once(':'))
                    .and_then(|(n, tail)| {
                        n.parse::<usize>()
                            .ok()
                            .filter(|_| tail == " tool argument not found verbatim in context")
                    });
                match idx {
                    Some(i) => w_warn_steps.push(i),
                    None => {
                        println!("FAIL structure: malformed WARNING line on line {line_no}");
                        return false;
                    }
                }
            }
            _ => {}
        }
    }

    // The magic line is mandatory and must come first: anything else is a
    // format downgrade, not a legacy receipt.
    match w_format_line {
        None => {
            println!("FAIL structure: missing AEGIS-TRACE header line");
            return false;
        }
        Some(n) if n != 0 => {
            println!("FAIL structure: AEGIS-TRACE header line at position {n}, must be first");
            return false;
        }
        Some(_) => {}
    }
    // Belt and braces: a receipt that declares format 1 must not carry the
    // format-2 per-step binding fields. If it does, the header was altered.
    if w_format == 1 {
        if let Some(i) = w_steps
            .iter()
            .position(|(_, _, _, _, _, ctx, q)| ctx.is_some() || q.is_some())
        {
            println!(
                "FAIL structure: receipt declares format 1 but step {i} carries ctx=/q= (format-2 downgrade)"
            );
            return false;
        }
    }

    // Format 3 folds `commit`/`host` into the trace genesis, so both lines
    // are mandatory there. A downgrade to `v1` does not help an attacker:
    // verify would then rebuild genesis without the provenance bytes and
    // the trace-chain would not match.
    let provenance: Option<(String, String)> = if w_format >= 3 {
        match (w_commit.clone(), w_host.clone()) {
            (Some(c), Some(h)) => {
                // Informational only — does not change PASS/FAIL below.
                // `unknown` means the generating binary was built without
                // a resolvable git commit (AEGIS_ALLOW_UNKNOWN_COMMIT=1 at
                // gen time), so this receipt's code provenance is not
                // pinned even though it is still cryptographically bound
                // into the trace genesis (tamper-evident, not
                // tamper-informative).
                if c == "unknown" {
                    println!("{UNKNOWN_COMMIT_WARNING}");
                }
                Some((c, h))
            }
            _ => {
                println!("FAIL structure: format-3 receipt is missing its commit or host line");
                return false;
            }
        }
    } else {
        None
    };

    let mut fail = false;
    for (name, local, claimed) in [
        ("MODEL", hex(model_sha), &w_model),
        ("EMBED", hex(embed_sha), &w_embed),
        ("VOCAB", hex(vocab_sha), &w_vocab),
    ] {
        if &local != claimed {
            println!(
                "FAIL artifact: {name} hash mismatch (receipt {} vs local {})",
                short16(claimed),
                short16(&local)
            );
            fail = true;
        }
    }
    if fail {
        return false;
    }

    if w_steps.len() != w_k {
        println!(
            "FAIL structure: receipt claims K={} but has {} step lines",
            w_k,
            w_steps.len()
        );
        return false;
    }

    if let Err(reason) = validate_receipt_header(&cis_model.config, tokenizer, &w_prompt, w_k, w_n)
    {
        println!("FAIL structure: {reason}");
        return false;
    }

    if w_format == 1 {
        println!("{FORMAT1_NOTE}");
    }

    // Table resolution: only when the receipt declares a
    // table-sha256 does verify require and use a --table. A --table
    // given for a receipt with no table-sha256 line is ignored
    // (the episode it describes never consulted one).
    let table: Option<LookupTable> = match &w_table_sha {
        Some(claimed) => {
            let path = match &table_path {
                Some(p) => p,
                None => {
                    println!(
                        "VERIFY FAIL — receipt declares table-sha256 {} but no --table was given",
                        short16(claimed)
                    );
                    return false;
                }
            };
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    println!("VERIFY FAIL — could not read --table {path}: {e}");
                    return false;
                }
            };
            let local_sha = hex(&sha256(&bytes));
            if &local_sha != claimed {
                println!(
                    "FAIL artifact: TABLE hash mismatch (receipt {} vs local {})",
                    short16(claimed),
                    short16(&local_sha)
                );
                return false;
            }
            match parse_table(&bytes) {
                Ok(t) => Some(t),
                Err(reason) => {
                    println!("FAIL structure: bad --table: {reason}");
                    return false;
                }
            }
        }
        None => {
            if table_path.is_some() {
                eprintln!("note: --table given but the receipt has no table-sha256 line; ignored");
            }
            None
        }
    };

    // Suite resolution: a receipt's suite-sha256 header (if any) is
    // authoritative and must be folded into genesis exactly as gen
    // did. A --suite-sha256 argument is only ever a cross-check
    // against that header — it is never folded on its own, and a
    // receipt with no header is unaffected regardless of whether
    // --suite-sha256 was passed.
    let suite_sha: Option<[u8; 32]> = match &w_suite_sha {
        Some(header_hex) => {
            let header_bytes = match parse_suite_sha256(header_hex) {
                Ok(b) => b,
                Err(reason) => {
                    println!("FAIL structure: {reason}");
                    return false;
                }
            };
            if let Some(arg_bytes) = suite_sha256_arg {
                let arg_hex = hex(&arg_bytes);
                if &arg_hex != header_hex {
                    println!(
                        "VERIFY FAIL — suite-sha256 mismatch: receipt {header_hex}, argument {arg_hex}"
                    );
                    return false;
                }
            }
            Some(header_bytes)
        }
        None => None,
    };

    // `--fail-fast`: diff each step against the receipt's claimed step as
    // soon as `replay_episode` produces it, via `step_diff` (the same
    // function full mode's loop below calls), instead of waiting for the
    // whole K-step replay to finish. `fail_fast_step` records the first
    // divergent step's index; the callback returning `true` stops
    // `replay_episode` right after that step. See the module doc
    // comment's `--fail-fast` entry.
    let mut fail_fast_step: Option<usize> = None;
    let r = if fail_fast {
        let mut on_step = |i: usize, local: &StepRecord| -> bool {
            if i >= w_steps.len() {
                // Malformed receipt (fewer claimed steps than K): let the
                // replay run to completion so the length check below
                // fires exactly as it would in full mode, instead of a
                // fail-fast message that full mode would never print for
                // this case.
                return false;
            }
            if step_diff(i, local, &w_steps[i], w_format) {
                fail_fast_step = Some(i);
                true
            } else {
                false
            }
        };
        replay_episode(
            cis_model,
            tokenizer,
            model_sha,
            embed_sha,
            vocab_sha,
            &w_prompt,
            w_k,
            w_n,
            table.as_ref(),
            suite_sha.as_ref(),
            provenance.as_ref().map(|(c, h)| (c.as_str(), h.as_str())),
            Some(&mut on_step),
        )
    } else {
        replay_episode(
            cis_model,
            tokenizer,
            model_sha,
            embed_sha,
            vocab_sha,
            &w_prompt,
            w_k,
            w_n,
            table.as_ref(),
            suite_sha.as_ref(),
            provenance.as_ref().map(|(c, h)| (c.as_str(), h.as_str())),
            None,
        )
    };

    if let Some(i) = fail_fast_step {
        println!("VERIFY FAIL — replay diverged from the receipt (fail-fast after step {i})");
        return false;
    }

    let local_trace_chain = hex(&r.trace_chain);
    println!("receipt trace-chain {}", short16(&w_trace_chain));
    println!("local   trace-chain {}", short16(&local_trace_chain));

    if r.steps.len() != w_steps.len() {
        println!(
            "VERIFY FAIL — replay produced {} steps, receipt has {}",
            r.steps.len(),
            w_steps.len()
        );
        return false;
    }

    // `--fail-fast` already diffed every step (via the same `step_diff`
    // this loop calls) as `replay_episode` produced it, above — and
    // returned early on the first divergence, so reaching here in
    // fail-fast mode means every step already diffed clean. Re-running
    // the diff would only reprint identical (empty, since nothing
    // diverged) output a second time, so it is skipped; `mismatch` stays
    // `false` exactly as the loop below would have left it. Without
    // `--fail-fast`, this is the original after-the-fact loop, unchanged.
    let mut mismatch = false;
    if !fail_fast {
        for (i, (local, w_step)) in r.steps.iter().zip(w_steps.iter()).enumerate() {
            if step_diff(i, local, w_step, w_format) {
                mismatch = true;
            }
        }
    }

    // The receipt's WARNING lines are part of what a reader is shown, so
    // they must match what this replay independently derives — otherwise a
    // warning could be deleted from, or invented in, a PASSing receipt.
    // Gated to format 2 and later: pre-format-2 receipts predate this
    // module's warning emission and their WARNING lines are not evidence.
    if w_format >= 2 {
        let mut claimed = w_warn_steps.clone();
        claimed.sort_unstable();
        claimed.dedup();
        let local_warns: Vec<usize> = r
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.verbatim_ok)
            .map(|(i, _)| i)
            .collect();
        if claimed != local_warns {
            println!(
                "VERIFY FAIL — WARNING lines claim steps {claimed:?}, replay derives {local_warns:?}"
            );
            mismatch = true;
        }
    }

    if !mismatch && local_trace_chain == w_trace_chain {
        println!(
            "VERIFY PASS — replay reproduced {} steps and the full trace chain bit-for-bit",
            r.steps.len()
        );
        true
    } else {
        println!("VERIFY FAIL — replay diverged from the receipt");
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- calc parser: accept cases ---

    #[test]
    fn calc_parses_basic_add() {
        let (m, a, op, b) = find_calc("here is CALC(3 + 4) done").unwrap();
        assert_eq!(m, "CALC(3 + 4)");
        assert_eq!((a, op, b), (3, b'+', 4));
    }

    #[test]
    fn calc_parses_tight_spacing() {
        let (m, a, op, b) = find_calc("CALC(3+4)").unwrap();
        assert_eq!(m, "CALC(3+4)");
        assert_eq!((a, op, b), (3, b'+', 4));
    }

    #[test]
    fn calc_parses_negative_operands() {
        let (_, a, op, b) = find_calc("CALC(-5 * 2)").unwrap();
        assert_eq!((a, op, b), (-5, b'*', 2));
    }

    #[test]
    fn calc_parses_extra_whitespace() {
        let (_, a, op, b) = find_calc("CALC(  7   %   3  )").unwrap();
        assert_eq!((a, op, b), (7, b'%', 3));
    }

    #[test]
    fn calc_finds_first_of_two() {
        let (m, ..) = find_calc("x CALC(1 + 1) y CALC(2 + 2)").unwrap();
        assert_eq!(m, "CALC(1 + 1)");
    }

    // --- calc parser: reject cases ---

    #[test]
    fn calc_rejects_missing_call() {
        assert!(find_calc("no tool call here").is_none());
    }

    #[test]
    fn calc_rejects_bad_op() {
        assert!(find_calc("CALC(3 & 4)").is_none());
    }

    #[test]
    fn calc_rejects_missing_operand() {
        assert!(find_calc("CALC(3 + )").is_none());
    }

    #[test]
    fn calc_rejects_non_integer_operand() {
        assert!(find_calc("CALC(x + 4)").is_none());
    }

    #[test]
    fn calc_rejects_unclosed_call() {
        assert!(find_calc("CALC(3 + 4").is_none());
    }

    // --- prompt-side opener prefix (T3: prompt ends "A: CALC(") ---

    #[test]
    fn verbatim_rule_rejects_argument_only_present_in_model_generated_text() {
        // EVAL-60 T1 mixed_02: step-0 question "part P-206", model output
        // included its own "Q: 2 + 2" line, step 1 called CALC(2 + 2).
        let external = "part P-206\nTOOL[lookup]=Bolt, hex head, 1/4-20 x 3/4 in.\n";
        assert!(!verbatim_ok_for(external, b"CALC(2 + 2)"));
        assert!(verbatim_ok_for(external, b"LOOKUP(P-206)"));
    }

    #[test]
    fn verbatim_rule_accepts_argument_from_a_tool_result_or_no_tool() {
        let external = "part P-100\nTOOL[lookup]=see part P-4023\n";
        assert!(verbatim_ok_for(external, b"LOOKUP(P-4023)"));
        assert!(verbatim_ok_for(external, b"no-tool"));
        assert!(verbatim_ok_for(external, b"CALC()"));
    }

    #[test]
    fn verbatim_rule_is_sound_but_not_complete() {
        // Sound: an argument absent from external text always warns. There is
        // no external text a WARNING can be raised against falsely, because
        // `contains` is exact.
        let external = "What is part 401?";
        assert!(!verbatim_ok_for(external, b"LOOKUP(403)"));
        assert!(!verbatim_ok_for(external, b"LOOKUP(4010)"));

        // Not complete: a fragment of external text passes. `40` never
        // appeared as a part number, but it is a substring of `401`, so this
        // step is not flagged. E34's WARNING counts are therefore a lower
        // bound on ungrounded tool calls, never an upper one.
        assert!(verbatim_ok_for(external, b"LOOKUP(40)"));
        assert!(verbatim_ok_for(external, b"LOOKUP(4)"));

        // And a step that calls no tool is never flagged at all, so the
        // WARNING census says nothing about the groundedness of model prose.
        assert!(verbatim_ok_for("", b"no-tool"));
    }

    #[test]
    fn verbatim_rule_step0_ignores_few_shot_examples() {
        // last_q_line drops the shots, so a shot-copied argument is flagged.
        let prompt = "Q: 2 + 2\nA: CALC(2 + 2).\nQ: two + two\nA:";
        assert!(!verbatim_ok_for(last_q_line(prompt), b"CALC(2 + 2)"));
    }

    #[test]
    fn prompt_scan_prefix_detects_calc_opener() {
        assert_eq!(prompt_scan_prefix("Q: 758 + 927\nA: CALC("), "CALC(");
    }

    #[test]
    fn prompt_scan_prefix_detects_lookup_opener() {
        assert_eq!(prompt_scan_prefix("Q: P-100\nA: LOOKUP("), "LOOKUP(");
    }

    #[test]
    fn prompt_scan_prefix_ignores_trailing_whitespace() {
        assert_eq!(prompt_scan_prefix("A: CALC(   \n  "), "CALC(");
    }

    #[test]
    fn prompt_scan_prefix_empty_when_no_trailing_opener() {
        assert_eq!(prompt_scan_prefix("Q: 3 + 4\nA:"), "");
        assert_eq!(prompt_scan_prefix("CALC(1 + 1) already closed"), "");
    }

    #[test]
    fn run_tool_carries_calc_prefix_from_prompt() {
        // Prompt ended "A: CALC(", model only transcribes the continuation.
        let o = run_tool("CALC(", "758 + 927). ", None);
        assert_eq!(o.name, "calc");
        assert_eq!(o.input, b"CALC(758 + 927)");
        assert_eq!(o.output, b"1685");
    }

    #[test]
    fn run_tool_carries_lookup_prefix_from_prompt() {
        let mut map = std::collections::HashMap::new();
        map.insert("P-100".to_string(), "widget".to_string());
        let t = LookupTable {
            map,
            sha256: [0u8; 32],
            len: 0,
        };
        let o = run_tool("LOOKUP(", "P-100) done", Some(&t));
        assert_eq!(o.name, "lookup");
        assert_eq!(o.input, b"LOOKUP(P-100)");
        assert_eq!(o.output, b"widget");
    }

    #[test]
    fn run_tool_no_prefix_is_unchanged() {
        // Empty prefix must behave byte-identically to the pre-existing
        // (no trailing opener) scan.
        let with_prefix = run_tool("", "prefix CALC(2 + 2) suffix", None);
        assert_eq!(with_prefix.name, "calc");
        assert_eq!(with_prefix.input, b"CALC(2 + 2)");
        assert_eq!(with_prefix.output, b"4");
    }

    // --- calc eval: checked arithmetic ---

    #[test]
    fn eval_calc_basic_ops() {
        assert_eq!(eval_calc(3, b'+', 4), Ok(7));
        assert_eq!(eval_calc(3, b'-', 4), Ok(-1));
        assert_eq!(eval_calc(3, b'*', 4), Ok(12));
        assert_eq!(eval_calc(7, b'/', 2), Ok(3));
        assert_eq!(eval_calc(7, b'%', 2), Ok(1));
    }

    #[test]
    fn eval_calc_div_by_zero() {
        assert_eq!(eval_calc(1, b'/', 0), Err("div-by-zero"));
        assert_eq!(eval_calc(1, b'%', 0), Err("div-by-zero"));
    }

    #[test]
    fn eval_calc_overflow() {
        assert_eq!(eval_calc(i64::MAX, b'+', 1), Err("overflow"));
        assert_eq!(eval_calc(i64::MIN, b'-', 1), Err("overflow"));
        assert_eq!(eval_calc(i64::MIN, b'/', -1), Err("overflow"));
    }

    // --- run_tool wiring ---

    #[test]
    fn run_tool_no_match_is_no_tool() {
        let o = run_tool("", "plain text", None);
        assert_eq!(o.name, "no-tool");
        assert!(o.input.is_empty());
        assert!(o.output.is_empty());
    }

    #[test]
    fn run_tool_success_is_calc() {
        let o = run_tool("", "prefix CALC(2 + 2) suffix", None);
        assert_eq!(o.name, "calc");
        assert_eq!(o.input, b"CALC(2 + 2)");
        assert_eq!(o.output, b"4");
    }

    #[test]
    fn run_tool_error_is_calc_error() {
        let o = run_tool("", "CALC(9 / 0)", None);
        assert_eq!(o.name, "calc-error");
        assert_eq!(o.output, b"div-by-zero");
    }

    // --- trace chain: deterministic, sensitive to every folded field ---

    fn base_digest() -> [u8; 32] {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2")
    }

    #[test]
    fn trace_fold_is_deterministic() {
        assert_eq!(base_digest(), base_digest());
    }

    #[test]
    fn trace_fold_changes_with_step_index() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 1, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_decode_chain() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[8u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_tool_name() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[9u8; 32], b"no-tool", b"CALC(1 + 1)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_tool_input() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 2)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_tool_output() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"3");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_genesis_changes_with_prompt_or_k_n() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g0 = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let g1 = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hellp", None, None, None,
        );
        let g2 = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 4, 16, b"hello", None, None, None,
        );
        let g3 = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 17, b"hello", None, None, None,
        );
        assert_ne!(g0, g1);
        assert_ne!(g0, g2);
        assert_ne!(g0, g3);
    }

    // --- hardening: verify-path structural validation ---

    #[test]
    fn check_step_label_accepts_matching_position() {
        assert!(check_step_label("2", 2).is_ok());
        assert!(check_step_label(" 0 ", 0).is_ok());
    }

    #[test]
    fn check_step_label_rejects_mismatched_position() {
        let err = check_step_label("5", 2).unwrap_err();
        assert_eq!(err, "step label 5 at position 2");
    }

    #[test]
    fn check_step_label_rejects_non_numeric_label() {
        assert!(check_step_label("x", 0).is_err());
    }

    #[test]
    fn unhex_accepts_valid_hex() {
        assert_eq!(unhex("48656c6c6f").unwrap(), b"Hello".to_vec());
        assert_eq!(unhex("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn unhex_rejects_odd_length() {
        assert!(unhex("abc").is_err());
    }

    #[test]
    fn unhex_rejects_non_hex_chars() {
        assert!(unhex("zz").is_err());
        assert!(unhex("4g").is_err());
    }

    #[test]
    fn check_header_bounds_rejects_oversized_n_without_panic() {
        // N so large that prompt_tokens + N overflows a naive sum on some
        // platforms if not handled carefully — this must return Err, not
        // panic, and must not touch a model.
        let err = check_header_bounds(1, usize::MAX, 4, 2048).unwrap_err();
        assert!(err.contains("exceeds max_position_embeddings"));
    }

    #[test]
    fn check_header_bounds_rejects_zero_k_or_n() {
        assert!(check_header_bounds(0, 16, 4, 2048).is_err());
        assert!(check_header_bounds(1, 0, 4, 2048).is_err());
    }

    #[test]
    fn check_header_bounds_rejects_empty_prompt() {
        assert!(check_header_bounds(1, 16, 0, 2048).is_err());
    }

    #[test]
    fn check_header_bounds_accepts_in_range() {
        assert!(check_header_bounds(3, 16, 4, 2048).is_ok());
    }

    #[test]
    fn tool_result_text_is_deterministic_and_reflects_outcome() {
        let o = run_tool("", "CALC(2 + 2)", None);
        assert_eq!(tool_result_text(&o), "\nTOOL[calc]=4\n");
        let o2 = run_tool("", "no call", None);
        assert_eq!(tool_result_text(&o2), "\nTOOL[no-tool]=\n");
    }

    // --- lookup table: parse (good, dup key, bad char, oversize value) ---

    fn demo_table_bytes() -> Vec<u8> {
        b"P-100\tGasket, O-ring, fuel line\nP-205\tBolt, 3/8-16 hex head\n".to_vec()
    }

    #[test]
    fn parse_table_accepts_well_formed_rows() {
        let t = parse_table(&demo_table_bytes()).unwrap();
        assert_eq!(t.map.get("P-100").unwrap(), "Gasket, O-ring, fuel line");
        assert_eq!(t.map.get("P-205").unwrap(), "Bolt, 3/8-16 hex head");
        assert_eq!(t.len, demo_table_bytes().len() as u64);
        assert_eq!(t.sha256, sha256(&demo_table_bytes()));
    }

    #[test]
    fn parse_table_skips_blank_lines() {
        let bytes = b"P-100\tGasket\n\nP-205\tBolt\n".to_vec();
        let t = parse_table(&bytes).unwrap();
        assert_eq!(t.map.len(), 2);
    }

    #[test]
    fn parse_table_rejects_duplicate_key() {
        let bytes = b"P-100\tGasket\nP-100\tOther\n".to_vec();
        let err = parse_table(&bytes).unwrap_err();
        assert!(err.contains("duplicate key"), "{err}");
    }

    #[test]
    fn parse_table_rejects_bad_key_char() {
        let bytes = b"P 100\tGasket\n".to_vec();
        let err = parse_table(&bytes).unwrap_err();
        assert!(err.contains("bad key"), "{err}");
    }

    #[test]
    fn parse_table_rejects_oversize_value() {
        let long_value = "x".repeat(257);
        let bytes = format!("P-100\t{long_value}\n").into_bytes();
        let err = parse_table(&bytes).unwrap_err();
        assert!(err.contains("exceeds 256 bytes"), "{err}");
    }

    #[test]
    fn parse_table_rejects_missing_tab() {
        let bytes = b"P-100 Gasket\n".to_vec();
        let err = parse_table(&bytes).unwrap_err();
        assert!(err.contains("no tab separator"), "{err}");
    }

    #[test]
    fn parse_table_rejects_value_with_extra_tab() {
        let bytes = b"P-100\tGasket\tExtra\n".to_vec();
        let err = parse_table(&bytes).unwrap_err();
        assert!(err.contains("contains a tab"), "{err}");
    }

    #[test]
    fn parse_table_rejects_non_utf8() {
        let bytes = vec![0x50, 0xFF, 0xFE, b'\t', b'v'];
        assert!(parse_table(&bytes).is_err());
    }

    // --- lookup tool: hit/miss ---

    #[test]
    fn lookup_hit_returns_table_value() {
        let t = parse_table(&demo_table_bytes()).unwrap();
        let o = run_tool("", "please LOOKUP(P-100) now", Some(&t));
        assert_eq!(o.name, "lookup");
        assert_eq!(o.input, b"LOOKUP(P-100)");
        assert_eq!(o.output, b"Gasket, O-ring, fuel line");
    }

    #[test]
    fn lookup_miss_returns_none_literal() {
        let t = parse_table(&demo_table_bytes()).unwrap();
        let o = run_tool("", "LOOKUP(P-999)", Some(&t));
        assert_eq!(o.name, "lookup");
        assert_eq!(o.output, b"NONE");
    }

    #[test]
    fn lookup_without_table_is_not_scanned() {
        // No table present: LOOKUP( text is inert, exactly like any other
        // plain text — this is what keeps a table-less episode identical to
        // the pre-LOOKUP behavior.
        let o = run_tool("", "LOOKUP(P-100)", None);
        assert_eq!(o.name, "no-tool");
    }

    #[test]
    fn lookup_rejects_bad_key_char() {
        assert!(find_lookup("LOOKUP(P 100)").is_none());
    }

    #[test]
    fn lookup_rejects_oversize_key() {
        let key = "x".repeat(65);
        let text = format!("LOOKUP({key})");
        assert!(find_lookup(&text).is_none());
    }

    // --- scanner: both tools in one text, earliest occurrence wins ---

    #[test]
    fn scanner_picks_earliest_of_calc_and_lookup() {
        let t = parse_table(&demo_table_bytes()).unwrap();
        let o = run_tool("", "first LOOKUP(P-100) then CALC(1 + 1)", Some(&t));
        assert_eq!(o.name, "lookup");
        let o2 = run_tool("", "first CALC(1 + 1) then LOOKUP(P-100)", Some(&t));
        assert_eq!(o2.name, "calc");
    }

    #[test]
    fn scanner_does_not_fall_back_when_earliest_call_fails_to_parse() {
        let t = parse_table(&demo_table_bytes()).unwrap();
        // Earliest is an invalid LOOKUP( — scanner must not fall back to the
        // later, valid CALC(...).
        let o = run_tool("", "LOOKUP(bad key) then CALC(1 + 1)", Some(&t));
        assert_eq!(o.name, "no-tool");
    }

    // --- genesis changes with the table ---

    #[test]
    fn trace_genesis_changes_with_table() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g_none = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let table_sha_a = [7u8; 32];
        let table_sha_b = [8u8; 32];
        let g_a = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            Some((&table_sha_a, 42)),
            None,
            None,
        );
        let g_b = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            Some((&table_sha_b, 42)),
            None,
            None,
        );
        let g_len = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            Some((&table_sha_a, 43)),
            None,
            None,
        );
        assert_ne!(
            g_none, g_a,
            "table-less genesis must differ from table-bound genesis"
        );
        assert_ne!(g_a, g_b, "genesis must be sensitive to table sha256");
        assert_ne!(g_a, g_len, "genesis must be sensitive to table length");
    }

    // --- genesis changes with the suite hash; domain-separated from table ---

    #[test]
    fn trace_genesis_changes_with_suite() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g_none = trace_genesis(
            &model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None, None, None,
        );
        let suite_a = [7u8; 32];
        let suite_b = [8u8; 32];
        let g_a = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            None,
            Some(&suite_a),
            None,
        );
        let g_b = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            None,
            Some(&suite_b),
            None,
        );
        assert_ne!(
            g_none, g_a,
            "suite-less genesis must differ from suite-bound genesis"
        );
        assert_ne!(g_a, g_b, "genesis must be sensitive to suite sha256");
    }

    #[test]
    fn trace_genesis_differs_between_two_hosts_running_the_same_episode() {
        // Pins the consequence of folding provenance in, so nobody reads a
        // cross-machine `trace-chain` mismatch as a reproducibility failure:
        // the same episode on two boxes MUST give two different chains.
        // Measured on real receipts (BitNet-2B, K=2 N=12, identical calc
        // episode): every per-step decode-chain/ctx/q matched exactly across
        // aefinity-box and aefinity-box2, while the trace-chains differed
        // (44ff6db6.. vs 2dc19eba..) because the host lines differed.
        // Cross-machine bit-identity is asserted over the per-step digests.
        let (m, e, v) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        let g = |commit: &str, host: &str| {
            trace_genesis(
                &m,
                &e,
                &v,
                3,
                16,
                b"hello",
                None,
                None,
                Some((commit.as_bytes(), host.as_bytes())),
            )
        };
        let box1 = g("aa0f99df", "aefinity-box");
        let box2 = g("aa0f99df", "aefinity-box2");
        assert_ne!(box1, box2, "a different host must give a different genesis");
        assert_ne!(
            box1,
            g("unknown", "aefinity-box"),
            "a different commit must give a different genesis"
        );
        assert_eq!(
            box1,
            g("aa0f99df", "aefinity-box"),
            "genesis must be a pure function of its inputs"
        );
        // And the provenance fold is length-prefixed, not concatenated, so
        // no (commit, host) split can be shifted to collide with another.
        assert_ne!(
            g("ab", "cd"),
            g("a", "bcd"),
            "provenance fields must be length-prefixed, not concatenated"
        );
    }

    #[test]
    fn trace_genesis_table_fold_is_unchanged_from_pre_suite_hash_code() {
        // Load-bearing: this exact digest was computed independently
        // (sha256 of the documented byte sequence: TRACE_DOMAIN + 3x32
        // artifact shas + k/n/prompt-len BE u64 + prompt + table_sha(32) +
        // table_len BE u64, no tag) BEFORE --suite-sha256 existed. Archived
        // receipts that declare table-sha256 depend on this fold never
        // changing. If this test ever needs to change, an archived
        // table-bound receipt has just been broken.
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let table_sha = [9u8; 32];
        let g = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            Some((&table_sha, 32)),
            None,
            None,
        );
        let expected: [u8; 32] = [
            0x65, 0x0f, 0x2b, 0x11, 0x05, 0x73, 0x60, 0x3f, 0x20, 0x9a, 0x27, 0x43, 0x34, 0xea,
            0x5c, 0xee, 0x71, 0x23, 0x7e, 0xff, 0xb4, 0xd8, 0xa8, 0x25, 0xb7, 0x59, 0x84, 0x39,
            0x91, 0xf5, 0x18, 0x01,
        ];
        assert_eq!(g, expected, "table-bound genesis fold must not change");
    }

    #[test]
    fn trace_genesis_suite_hash_differs_from_table_hash_of_same_bytes() {
        // Same 32 bytes used as a table hash vs a suite hash fold to
        // different genesis digests. This is NOT because the table slot is
        // tagged (it is deliberately untagged, unchanged from before) but
        // because a suite hash is folded in an additional, later position
        // (after the table slot, tagged b"SUITE") — a table-only fold and a
        // suite-only fold of the same bytes cover different byte ranges by
        // construction and can never collide.
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let same_bytes = [9u8; 32];
        let g_table = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            Some((&same_bytes, 32)),
            None,
            None,
        );
        let g_suite = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            None,
            Some(&same_bytes),
            None,
        );
        assert_ne!(
            g_table, g_suite,
            "a table-bound genesis and a suite-bound genesis of the same bytes must differ"
        );
    }

    #[test]
    fn parse_suite_sha256_accepts_valid_hex() {
        let hex64 = "a".repeat(64);
        assert_eq!(parse_suite_sha256(&hex64).unwrap(), [0xaa_u8; 32]);
    }

    #[test]
    fn parse_suite_sha256_rejects_wrong_length() {
        assert!(parse_suite_sha256(&"a".repeat(63)).is_err());
        assert!(parse_suite_sha256(&"a".repeat(65)).is_err());
        assert!(parse_suite_sha256("").is_err());
    }

    #[test]
    fn parse_suite_sha256_rejects_uppercase_and_non_hex() {
        assert!(parse_suite_sha256(&"A".repeat(64)).is_err());
        let mut bad = "a".repeat(63);
        bad.push('z');
        assert!(parse_suite_sha256(&bad).is_err());
    }

    // --- format 2: per-step context/query binding — pure-function pieces ---

    #[test]
    fn extract_call_arg_strips_calc_wrapper() {
        assert_eq!(extract_call_arg(b"CALC(2 + 2)"), Some("2 + 2"));
    }

    #[test]
    fn extract_call_arg_strips_lookup_wrapper() {
        assert_eq!(extract_call_arg(b"LOOKUP(P-100)"), Some("P-100"));
    }

    #[test]
    fn extract_call_arg_rejects_unwrapped_text() {
        assert_eq!(extract_call_arg(b"not a call"), None);
    }

    #[test]
    fn last_q_line_picks_the_final_q_line() {
        let ctx = "Q: part P-100\nA: LOOKUP(P-100).\nQ: part P-206\nA:";
        assert_eq!(last_q_line(ctx), "part P-206");
    }

    #[test]
    fn last_q_line_empty_when_no_q_line() {
        assert_eq!(last_q_line("no questions here"), "");
    }

    #[test]
    fn mismatch_messages_match_the_brief_wording() {
        assert_eq!(ctx_mismatch_msg(2), "STEP 2 CTX MISMATCH");
        assert_eq!(query_mismatch_msg(2), "STEP 2 QUERY MISMATCH");
        assert_eq!(
            verbatim_warning_msg(0),
            "WARNING step 0: tool argument not found verbatim in context"
        );
    }

    // --- format 2: end-to-end round trip + tamper on the checked-in M7
    // tinybit model (small: MODEL.SAF ~2.7 MB, EMBED.BIN ~6 MB, VOCAB.BIN
    // ~160 KB — a K=2, N in {8,24} episode over it is cheap, so these tests
    // load the real model rather than a canned fixture; see the module doc
    // comment's format-2 entry for the grammar these receipts carry). ---

    /// Loads the checked-in M7 tinybit model and leaks its byte buffers to
    /// `'static` (test-only; a handful of small one-time leaks per test
    /// binary run, never in `gen`/`verify`'s own real paths) so the
    /// borrowed `SafeTensors`/`FullBitNetPipeline`/`CisModel`/
    /// `AegisTokenizer` chain can be returned by value instead of pinning
    /// this whole test module's helpers to one caller-supplied lifetime.
    fn load_m7_model() -> (
        CisModel<'static>,
        AegisTokenizer<'static>,
        [u8; 32],
        [u8; 32],
        [u8; 32],
    ) {
        let root = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../model-lab/tinybit/m7_final_gate_work/artifacts"
        );
        let model_bytes: &'static [u8] = Box::leak(
            std::fs::read(format!("{root}/MODEL.SAF"))
                .expect("read MODEL.SAF fixture")
                .into_boxed_slice(),
        );
        let embed_bytes: &'static [u8] = Box::leak(
            std::fs::read(format!("{root}/EMBED.BIN"))
                .expect("read EMBED.BIN fixture")
                .into_boxed_slice(),
        );
        let vocab_bytes: &'static [u8] = Box::leak(
            std::fs::read(format!("{root}/VOCAB.BIN"))
                .expect("read VOCAB.BIN fixture")
                .into_boxed_slice(),
        );
        let model_sha = sha256(model_bytes);
        let embed_sha = sha256(embed_bytes);
        let vocab_sha = sha256(vocab_bytes);
        let tensors: &'static SafeTensors = Box::leak(Box::new(
            SafeTensors::deserialize(model_bytes).expect("parse MODEL.SAF"),
        ));
        let cfg_json = tensors
            .metadata_field("aegis_config")
            .expect("read __metadata__")
            .expect("MODEL.SAF carries no aegis_config");
        let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
        let tokenizer = AegisTokenizer::new(vocab_bytes).expect("parse VOCAB.BIN");
        let pipeline: &'static FullBitNetPipeline<'static> = Box::leak(Box::new(
            FullBitNetPipeline::new(tensors, embed_bytes, &config).expect("build pipeline"),
        ));
        let cis_model = CisModel::new_with_options(pipeline, &config, head_preconvert_enabled())
            .expect("CIS model conversion");
        (cis_model, tokenizer, model_sha, embed_sha, vocab_sha)
    }

    /// Render an `EpisodeReplay` as receipt text, `format` 1 (`AEGIS-TRACE
    /// v0`, no `ctx=`/`q=` fields) or 2 (`AEGIS-TRACE v1`, with them) —
    /// deliberately independent of `main`'s own printing so a bug shared by
    /// both would not go unnoticed by these tests.
    fn render_receipt(
        format: u8,
        model_sha: &[u8; 32],
        embed_sha: &[u8; 32],
        vocab_sha: &[u8; 32],
        prompt: &str,
        k: usize,
        n: usize,
        r: &EpisodeReplay,
    ) -> String {
        let mut text = String::new();
        text.push_str(match format {
            1 => "AEGIS-TRACE v0\n",
            2 => "AEGIS-TRACE v1\n",
            _ => "AEGIS-TRACE v2\n",
        });
        text.push_str(&format!("model {}\n", hex(model_sha)));
        text.push_str(&format!("embed {}\n", hex(embed_sha)));
        text.push_str(&format!("vocab {}\n", hex(vocab_sha)));
        text.push_str(&format!("K {k}\n"));
        text.push_str(&format!("N {n}\n"));
        text.push_str(&format!("prompt-hex {}\n", hex(prompt.as_bytes())));
        text.push_str("commit test\n");
        text.push_str("host test\n");
        for (i, s) in r.steps.iter().enumerate() {
            let ids: Vec<String> = s.toks.iter().map(|t| t.to_string()).collect();
            if format == 1 {
                text.push_str(&format!(
                    "step {i}: toks={} tool={} in={} out={} decode-chain={}\n",
                    ids.join(","),
                    s.tool_name,
                    hex(&s.tool_input),
                    hex(&s.tool_output),
                    hex(&s.decode_chain)
                ));
            } else {
                text.push_str(&format!(
                    "step {i}: toks={} tool={} in={} out={} decode-chain={} ctx={} q={}\n",
                    ids.join(","),
                    s.tool_name,
                    hex(&s.tool_input),
                    hex(&s.tool_output),
                    hex(&s.decode_chain),
                    hex(&s.ctx_digest),
                    hex(&s.query_digest)
                ));
            }
        }
        text.push_str(&format!("trace-chain {}\n", hex(&r.trace_chain)));
        text
    }

    /// Flip one hex nibble of `field=` (e.g. `"q="`) on the line starting
    /// with `step_prefix` (e.g. `"step 1:"`) — the smallest tamper that
    /// changes the field's value without touching its length or any other
    /// field on the line.
    fn flip_hex_field(text: &str, step_prefix: &str, field: &str) -> String {
        let mut out = String::new();
        for line in text.lines() {
            if line.starts_with(step_prefix) {
                if let Some(pos) = line.find(field) {
                    let val_start = pos + field.len();
                    let rest = &line[val_start..];
                    let val_len = rest.find(' ').unwrap_or(rest.len());
                    let mut chars: Vec<char> = rest[..val_len].chars().collect();
                    chars[0] = if chars[0] == '0' { '1' } else { '0' };
                    let newval: String = chars.into_iter().collect();
                    out.push_str(&line[..val_start]);
                    out.push_str(&newval);
                    out.push_str(&rest[val_len..]);
                    out.push('\n');
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        out
    }

    fn write_temp_receipt(name: &str, text: &str) -> std::path::PathBuf {
        let path =
            std::env::temp_dir().join(format!("agent_trace_test_{}_{name}", std::process::id()));
        std::fs::write(&path, text).expect("write temp receipt");
        path
    }

    #[test]
    fn format2_round_trip_gen_then_verify_pass_with_ctx_q_fields() {
        // Tool-call binding (a tampered `q=`/tool-call argument breaking
        // verify) is covered separately by `format2_query_tamper_fails_verify`
        // and by the demo's tamper 4 — this test asserts only what the
        // generator/verifier guarantee for every format-2 episode regardless
        // of what the tiny M7 test model happens to decode: ctx/q fields of
        // the documented length are present and correct on every step, and
        // the whole receipt round-trips PASS.
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let k = 2usize;
        let n = 16usize;
        let prompt = "Once upon a time";
        let r = replay_episode(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, None, None,
            None, None,
        );
        assert_eq!(r.steps.len(), k);
        for s in &r.steps {
            assert_eq!(s.ctx_digest.len(), 32);
            assert_eq!(s.query_digest.len(), 32);
        }
        assert_ne!(
            r.steps[0].ctx_digest, r.steps[1].ctx_digest,
            "step 1's context (initial prompt + step 0's generation/tool result) \
             must differ from step 0's context"
        );
        assert_eq!(
            r.steps[0].query_digest,
            sha256(prompt.as_bytes()),
            "step 0's q= must be sha256 of the initial prompt bytes, exactly as \
             replay_episode defines the step-0 query"
        );

        let text = render_receipt(2, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        let path = write_temp_receipt("roundtrip.txt", &text);
        let pass = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        );
        let _ = std::fs::remove_file(&path);
        assert!(
            pass,
            "format-2 receipt with ctx/q fields should round-trip PASS"
        );
    }

    #[test]
    fn format2_query_tamper_fails_verify() {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let k = 2usize;
        let n = 16usize;
        let prompt = "Once upon a time";
        let r = replay_episode(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, None, None,
            None, None,
        );
        let good_text = render_receipt(2, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        let tampered = flip_hex_field(&good_text, "step 1:", "q=");
        assert_ne!(
            good_text, tampered,
            "tamper helper must actually change the receipt"
        );

        let path = write_temp_receipt("query-tamper.txt", &tampered);
        let pass = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        );
        let _ = std::fs::remove_file(&path);
        assert!(
            !pass,
            "a flipped q= field must make verify FAIL (STEP 1 QUERY MISMATCH)"
        );
    }

    // --- --fail-fast ---

    /// `replay_episode`'s per-step hook is the whole mechanism `--fail-fast`
    /// relies on to avoid replaying steps after the first divergence: this
    /// tests the hook directly (returning `true` from `on_step` on the
    /// very first step) and asserts the replay produces exactly one step,
    /// not `k`. `verify_one`'s `fail_fast` tests below cover the
    /// end-to-end behaviour built on top of this hook.
    #[test]
    fn replay_episode_on_step_hook_stops_the_replay() {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let k = 3usize;
        let n = 16usize;
        let prompt = "Once upon a time";
        let mut calls = 0usize;
        let mut cb = |_i: usize, _rec: &StepRecord| -> bool {
            calls += 1;
            true
        };
        let r = replay_episode(
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            prompt,
            k,
            n,
            None,
            None,
            None,
            Some(&mut cb),
        );
        assert_eq!(
            calls, 1,
            "on_step must not be called again once it has returned true"
        );
        assert_eq!(
            r.steps.len(),
            1,
            "a true return from on_step must stop the replay after that step, \
             not run the remaining k-1 steps"
        );
    }

    /// Build a clean (untampered) format-2 receipt, then flip a step-0 hex
    /// field so the receipt disagrees with replay starting at step 0 —
    /// the tamper both `fail_fast_verify_fails_at_step_0_tamper` and
    /// `full_mode_still_fails_on_step_0_tamper` share, so both tests are
    /// exercising the identical divergence.
    fn step0_tampered_receipt(
        cis_model: &CisModel,
        tokenizer: &AegisTokenizer,
        model_sha: &[u8; 32],
        embed_sha: &[u8; 32],
        vocab_sha: &[u8; 32],
        prompt: &str,
        k: usize,
        n: usize,
    ) -> String {
        let r = replay_episode(
            cis_model, tokenizer, model_sha, embed_sha, vocab_sha, prompt, k, n, None, None,
            None, None,
        );
        let good_text = render_receipt(2, model_sha, embed_sha, vocab_sha, prompt, k, n, &r);
        let tampered = flip_hex_field(&good_text, "step 0:", "decode-chain=");
        assert_ne!(
            good_text, tampered,
            "tamper helper must actually change the receipt"
        );
        tampered
    }

    /// Requirement (a): `--fail-fast` on a receipt tampered at step 0
    /// reports the divergence at step 0 and fails verify (the fail-fast
    /// message itself — `VERIFY FAIL — replay diverged from the receipt
    /// (fail-fast after step {i})` — is only observable on stdout, which
    /// this test-module style does not capture; see the module doc
    /// comment's `--fail-fast` entry and `verify_one`'s fail-fast branch
    /// for where `i` is pinned to the first divergent step index).
    #[test]
    fn fail_fast_verify_fails_at_step_0_tamper() {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let k = 3usize;
        let n = 16usize;
        let prompt = "Once upon a time";
        let tampered = step0_tampered_receipt(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n,
        );
        let path = write_temp_receipt("fail-fast-step0.txt", &tampered);
        let pass = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            true, // --fail-fast
        );
        let _ = std::fs::remove_file(&path);
        assert!(
            !pass,
            "a step-0 decode-chain tamper must fail --fail-fast verify"
        );
    }

    /// Requirement (c): full-mode verify (`fail_fast=false`) on the exact
    /// same step-0 tamper `fail_fast_verify_fails_at_step_0_tamper` uses
    /// must also FAIL — both modes reach the same verdict for the same
    /// divergence, only fail-fast stops the replay early. Full mode's
    /// per-step FAIL wording for a non-step-0 tamper is already covered by
    /// `format2_query_tamper_fails_verify` above.
    #[test]
    fn full_mode_still_fails_on_step_0_tamper() {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let k = 3usize;
        let n = 16usize;
        let prompt = "Once upon a time";
        let tampered = step0_tampered_receipt(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n,
        );
        let path = write_temp_receipt("full-mode-step0.txt", &tampered);
        let pass = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        );
        let _ = std::fs::remove_file(&path);
        assert!(
            !pass,
            "a step-0 decode-chain tamper must fail full-mode verify too"
        );
    }

    /// Requirement (b): on an untampered receipt, `--fail-fast` still
    /// PASSes, agreeing with full mode on the same receipt.
    #[test]
    fn fail_fast_pass_matches_full_mode_on_clean_receipt() {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let k = 2usize;
        let n = 16usize;
        let prompt = "Once upon a time";
        let r = replay_episode(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, None, None,
            None, None,
        );
        let text = render_receipt(2, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        let path = write_temp_receipt("fail-fast-clean.txt", &text);
        let pass_full = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        );
        let pass_fail_fast = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            true,
        );
        let _ = std::fs::remove_file(&path);
        assert!(pass_full, "clean receipt must PASS full-mode verify");
        assert!(
            pass_fail_fast,
            "clean receipt must PASS --fail-fast verify too, identically to full mode"
        );
    }

    #[test]
    fn format1_receipt_still_verifies_pass_with_no_ctx_q_fields() {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let k = 1usize;
        let n = 8usize;
        let prompt = "Once upon a time";
        let r = replay_episode(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, None, None,
            None, None,
        );
        let text = render_receipt(1, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        assert!(
            !text.contains("ctx=") && !text.contains(" q="),
            "a format-1 fixture must carry no ctx=/q= fields"
        );

        let path = write_temp_receipt("format1.txt", &text);
        let pass = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        );
        let _ = std::fs::remove_file(&path);
        assert!(pass, "a format-1 receipt must still verify PASS unchanged");
    }

    /// Build a format-2 receipt whose ctx= binding has been tampered with,
    /// then apply `mangle` to its header line. Returns whether verify passed.
    ///
    /// Before the header was made mandatory, every one of these mangles
    /// downgraded the receipt to format-1 rules, which do not check ctx=/q=
    /// at all, so a forged context verified PASS.
    fn header_downgrade_attempt(mangle: impl Fn(&str) -> String) -> bool {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let (k, n, prompt) = (2usize, 8usize, "Once upon a time");
        let r = replay_episode(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, None, None,
            None, None,
        );
        let text = render_receipt(2, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        // Tamper the last step's ctx= so format-2 rules would reject it.
        let mut lines: Vec<String> = text.lines().map(|l| l.to_string()).collect();
        let step = lines
            .iter()
            .rposition(|l| l.starts_with("step "))
            .expect("a step line");
        let i = lines[step].find("ctx=").expect("a ctx= field") + 4;
        let c = lines[step].as_bytes()[i];
        let flipped = if c == b'0' { '1' } else { '0' };
        lines[step].replace_range(i..i + 1, &flipped.to_string());
        lines[0] = mangle(&lines[0]);
        let text = lines.join("\n") + "\n";

        let path = write_temp_receipt("header-downgrade.txt", &text);
        let pass = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        );
        let _ = std::fs::remove_file(&path);
        pass
    }

    #[test]
    fn header_downgrade_corrupted_magic_is_rejected() {
        assert!(
            !header_downgrade_attempt(|_| "AEGIS-TRACEX v1".to_string()),
            "a corrupted magic word must not downgrade a format-2 receipt to format-1"
        );
    }

    #[test]
    fn header_downgrade_missing_header_is_rejected() {
        assert!(
            !header_downgrade_attempt(|_| "# no header here".to_string()),
            "a missing AEGIS-TRACE header must be rejected, not defaulted to format 1"
        );
    }

    #[test]
    fn header_downgrade_lowercase_magic_is_rejected() {
        assert!(
            !header_downgrade_attempt(|_| "aegis-trace v1".to_string()),
            "the magic word is case-sensitive and must not fall through to format 1"
        );
    }

    // --- E23 (cm-box2, 2026-09-09): the tamper matrix ran 321 mutants of
    // four episodes against the format-2 verifier and 28 of them still
    // reported VERIFY PASS. They fell into three families, each closed by
    // one of the tests below: trailing junk on a step line, a lead key
    // renamed out of the parser's vocabulary, and an edit to the `commit`
    // or `host` value. ---

    /// Build a real format-3 receipt over the M7 fixture, apply `mangle` to
    /// its text, and return whether `verify` accepted the result.
    fn format3_receipt_survives(mangle: impl Fn(&str) -> String) -> bool {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let prompt = "Q: What is 2 + 2?\nA:";
        let (k, n) = (2usize, 8usize);
        let r = replay_episode(
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            prompt,
            k,
            n,
            None,
            None,
            Some(("test", "test")),
            None,
        );
        let text = render_receipt(3, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        // Unique per call: these tests run in parallel in one process and
        // `write_temp_receipt` names the file after the pid.
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = write_temp_receipt(&format!("format3-{seq}.txt"), &mangle(&text));
        verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        )
    }

    /// Same as `format3_receipt_survives`, but the receipt's `commit` line
    /// (and the genesis it is folded into) is built from `commit` instead
    /// of the fixed `"test"` — used to exercise the `unknown`-commit
    /// WARNING path, which must still verify unchanged (see
    /// `unknown_commit_receipt_still_verifies_pass`).
    fn format3_receipt_survives_with_commit(commit: &str, mangle: impl Fn(&str) -> String) -> bool {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let prompt = "Q: What is 2 + 2?\nA:";
        let (k, n) = (2usize, 8usize);
        let r = replay_episode(
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            prompt,
            k,
            n,
            None,
            None,
            Some((commit, "test")),
            None,
        );
        let mut text = render_receipt(3, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        text = text.replace("commit test\n", &format!("commit {commit}\n"));
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = write_temp_receipt(&format!("format3-commit-{seq}.txt"), &mangle(&text));
        verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        )
    }

    #[test]
    fn unknown_commit_receipt_still_verifies_pass() {
        // An "unknown"-commit format-3 receipt is unpinned provenance, not
        // a tamper: `verify` must PASS it exactly as it would a receipt
        // with a real commit, and print `UNKNOWN_COMMIT_WARNING` alongside
        // (see `verify_one`'s provenance block) rather than change
        // PASS/FAIL semantics.
        assert!(
            format3_receipt_survives_with_commit("unknown", |t| t.to_string()),
            "a format-3 receipt with commit=unknown must still verify PASS \
             structurally exactly as any other commit value would"
        );
    }

    #[test]
    fn unknown_commit_warning_text_is_exact() {
        assert_eq!(
            UNKNOWN_COMMIT_WARNING,
            "WARNING: receipt commit is unknown (provenance not pinned to code)"
        );
    }

    #[test]
    fn commit_hash_is_forty_hex_or_unknown() {
        let c = commit_hash();
        let is_forty_hex =
            c.len() == 40 && c.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'));
        assert!(
            is_forty_hex || c == "unknown",
            "commit_hash() must be 40 lowercase hex or the literal \"unknown\", got {c:?}"
        );
        // This crate's own manifest dir is inside a git checkout (the repo
        // this test itself was built from), so build.rs must have resolved
        // a real commit here, never "unknown".
        if std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.git")).exists() {
            assert_ne!(
                c,
                "unknown",
                "built inside a git checkout ({}/../.git exists); commit_hash() must not be \
                 \"unknown\" — check build.rs's git invocation",
                env!("CARGO_MANIFEST_DIR")
            );
        }
    }

    #[test]
    fn format3_round_trip_verifies() {
        assert!(
            format3_receipt_survives(|t| t.to_string()),
            "an untampered format-3 receipt must verify — without this positive \
             control every test below could pass for the wrong reason"
        );
    }

    #[test]
    fn format3_commit_value_tamper_fails_verify() {
        assert!(
            !format3_receipt_survives(|t| t.replace("commit test", "commit tesX")),
            "commit is folded into the trace genesis from format 3 on"
        );
    }

    #[test]
    fn format3_host_value_tamper_fails_verify() {
        assert!(
            !format3_receipt_survives(|t| t.replace("host test", "host tesX")),
            "host is folded into the trace genesis from format 3 on"
        );
    }

    #[test]
    fn format3_downgrade_to_v1_header_fails_verify() {
        // Relabelling a format-3 receipt as format 2 does not unbind the
        // provenance: verify then rebuilds genesis without those bytes and
        // the trace-chain no longer matches.
        assert!(!format3_receipt_survives(
            |t| t.replace("AEGIS-TRACE v2", "AEGIS-TRACE v1")
        ));
    }

    #[test]
    fn step_line_rejects_trailing_junk_after_the_index() {
        // E23 F.l9.t1: `step 0:X ...` verified unchanged on every episode.
        assert!(!format3_receipt_survives(
            |t| t.replace("step 0:", "step 0:X")
        ));
    }

    #[test]
    fn step_line_rejects_unknown_field() {
        assert!(!format3_receipt_survives(
            |t| t.replace(" tool=", " smuggled=anything tool=")
        ));
    }

    #[test]
    fn step_line_rejects_duplicate_field() {
        assert!(!format3_receipt_survives(
            |t| t.replace(" tool=", " tool=x tool=")
        ));
    }

    #[test]
    fn lead_line_rejects_unknown_key() {
        // E23 F.l7.t0/F.l8.t0: renaming a lead key deleted the field.
        assert!(!format3_receipt_survives(
            |t| t.replace("commit test", "commitX test")
        ));
        assert!(!format3_receipt_survives(
            |t| t.replace("host test", "hostX test")
        ));
    }

    #[test]
    fn lead_line_rejects_duplicate_key() {
        assert!(!format3_receipt_survives(
            |t| t.replace("commit test", "commit test\nK 99")
        ));
    }

    #[test]
    fn invented_warning_line_fails_verify() {
        assert!(!format3_receipt_survives(|t| t.replace(
            "commit test",
            "WARNING step 0: tool argument not found verbatim in context\ncommit test"
        )));
    }

    #[test]
    fn malformed_warning_line_is_rejected() {
        assert!(!format3_receipt_survives(|t| t.replace(
            "commit test",
            "WARNING step 0: the agent did nothing wrong\ncommit test"
        )));
    }

    #[test]
    fn malformed_k_is_rejected_without_panic() {
        assert!(!format3_receipt_survives(
            |t| t.replace("\nK 2\n", "\nK two\n")
        ));
    }

    #[test]
    fn header_must_be_the_first_line() {
        let (cis_model, tokenizer, model_sha, embed_sha, vocab_sha) = load_m7_model();
        let (k, n, prompt) = (1usize, 8usize, "Once upon a time");
        let r = replay_episode(
            &cis_model, &tokenizer, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, None, None,
            None, None,
        );
        let text = render_receipt(2, &model_sha, &embed_sha, &vocab_sha, prompt, k, n, &r);
        let moved = format!("model {}\nAEGIS-TRACE v1\n{}", hex(&model_sha), text);

        let path = write_temp_receipt("header-not-first.txt", &moved);
        let pass = verify_one(
            path.to_str().unwrap(),
            &cis_model,
            &tokenizer,
            &model_sha,
            &embed_sha,
            &vocab_sha,
            None,
            None,
            false,
        );
        let _ = std::fs::remove_file(&path);
        assert!(
            !pass,
            "the AEGIS-TRACE header must be required to come first"
        );
    }
}
