//! AEFINITY OS phase 0 — `JOB.TXT` autorun and the `RESULT.TXT` record.
//!
//! Spec: `program/AEFINITY_OS.md` §2 (input), §3 (output), §5 (this file).
//!
//! A stick that carries a `JOB.TXT` turns the unikernel from an interactive
//! console into a lab worker: parse the directives, run them in file order,
//! write one machine-readable `RESULT.TXT` to the boot volume, and reset (or
//! halt). A stick with no `JOB.TXT` never enters this module and boot
//! behaviour is exactly what it was before — that invariant is what
//! `cargo xtask boot-test` still exiting 33 checks.
//!
//! Rule A note: `tps` is a *structural* field. Every record carries
//! `env=iron|vm` from [`crate::sysinfo::env`], and under QEMU/TCG that reads
//! `vm`, which is the machine-readable statement that the number in it is not
//! a performance measurement and may not be quoted as one. `tsc_per_tok` is
//! RDTSC ticks, not cycles (CLAUDE.md Rule A corollary); the key is named for
//! what it actually holds.
//!
//! Nothing here calls `unwrap` on a firmware operation. Every failure is
//! logged to `BOOTLOG.TXT` and degraded past, because a box running headless
//! in a rack has no other way to tell anyone what went wrong.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use uefi::proto::media::file::Directory;

/// Whole-job wall budget when `JOB.TXT` does not say (spec §2).
pub const DEFAULT_BUDGET_S: u64 = 300;
/// Seconds added to the budget when arming the firmware watchdog (spec §2).
pub const WATCHDOG_MARGIN_S: u64 = 60;
/// Default `TOKENS` for a `PROMPT` (spec §2).
pub const DEFAULT_TOKENS: usize = 64;
/// Hard cap on `TOKENS`, whatever the file asks for (spec §2).
pub const MAX_TOKENS: usize = 1024;
/// Default resident-mode listener port (spec §2; phase 2 uses it).
pub const DEFAULT_LISTEN: u16 = 4242;
/// `job.N.response` is truncated to this many bytes *after* escaping (spec §3).
pub const RESPONSE_MAX: usize = 256;
/// Each leg of a `NETCHECK` — connect, send, wait for the peer to close — is
/// given this long. Phase 1a's gate (spec §6) states the wait as 5 s; the same
/// bound is used for the connect and the write so a directive can never cost
/// the job more than a few seconds whatever the peer does.
pub const NETCHECK_TIMEOUT_MS: u64 = 5_000;
/// The four-way close afterwards. Short: the exchange is already over, and a
/// peer that will not finish the close must not hold the job open.
const NETCHECK_CLOSE_MS: u64 = 2_000;
/// Bound for the whole `REPORT` POST — connect, write, read **and** close —
/// phase 1b (spec §5). [`crate::net::http::post`] holds one deadline across
/// all four phases, so this is the real ceiling on the exchange and not a
/// per-phase timeout that four stalled phases multiply by four.
///
/// The arithmetic it has to satisfy. The watchdog is armed at
/// `budget_s + WATCHDOG_MARGIN_S` before `run_job`, and `run_job` is itself
/// bounded by `budget_s`, so in the worst case exactly `WATCHDOG_MARGIN_S`
/// (60 s) of margin is left when the job's own window closes. Everything
/// after it must fit inside that:
///
/// | after the job | worst case |
/// |---|---|
/// | `write_result_txt` + the WIP delete + BOOTLOG lines | firmware-bound, small |
/// | `REPORT` POST (this constant) | 30 s |
/// | [`settle_volume`]: a 3 s stall, the read-back, a 1 s stall | 4 s |
/// | **total** | **~34 s of the 60 s** |
///
/// That leaves ~26 s of slack for the FAT writes and the console echo before
/// `AFTER` runs, which is the point: a collector that will not answer must
/// never turn a completed job into a firmware cold reset. Raising this
/// constant without raising [`WATCHDOG_MARGIN_S`] spends that slack.
const REPORT_TIMEOUT_MS: u64 = 30_000;
/// Job bodies are capped (spec §4); the same cap guards the on-disk file.
const JOB_MAX_BYTES: usize = 64 * 1024;
/// A single `PROMPT` line is capped (spec §4).
const PROMPT_MAX_BYTES: usize = 4 * 1024;
/// Watchdog code passed to `set_watchdog_timer`; anything >= 0x1_0000 is the
/// UEFI-spec range reserved for the caller, so it cannot collide with a
/// firmware-defined code in a post-mortem.
const WATCHDOG_CODE: u64 = 0x1_0000;

/// `RUNID <id>` is `[A-Za-z0-9._-]{1,64}` (design §1.2).
pub const RUNID_MAX_BYTES: usize = 64;
/// `TAG <text>` is free-form; this is the byte cap that keeps one directive
/// from filling a record. Not a protocol constant — a bound.
pub const TAG_MAX_BYTES: usize = 128;

/// `EVAL_WINDOW` (design §1.3): the token window `EVAL` scores in, capped
/// again by the model's own `max_position_embeddings` (§2.1).
pub const EVAL_WINDOW: usize = 2048;
/// `SHARD <i>/<n>`: `1 <= i <= n <= SHARD_MAX` (design §2).
pub const SHARD_MAX: u32 = 4096;
/// `job.N.detail` is truncated to this many bytes *after* escaping (design §3).
pub const DETAIL_MAX: usize = 1024;

/// The prompt `BENCH n` generates from. Fixed on purpose: a bench whose
/// prompt varies between boxes is not a cross-machine comparison, and the
/// `digest` of its output is only a fleet witness if every box conditioned on
/// the identical bytes.
pub const BENCH_PROMPT: &str = "The quick brown fox jumps over the lazy dog.";

/// `MODE` (spec §2). Phase 0 implements `oneshot`; `resident` is phase 2 and
/// is parsed and recorded here so a phase-2 job file is not a parse error on
/// a phase-0 box.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Oneshot,
    Resident,
}

/// `AFTER` (spec §2): what to do once the directive list is done.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum After {
    /// `ResetSystem(COLD)`. Under QEMU with `-no-reboot` this exits QEMU 0.
    Reset,
    /// Return to the caller, which falls through to whatever this build does
    /// next (the interactive console in a production build).
    Halt,
}

/// `NET` (spec §2). Parsed and stored in phase 0; brought up in phase 1a.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetCfg {
    Dhcp,
    Static { cidr: String, gateway: String },
}

/// One unit of work, in file order (spec §2: "directives run in file order").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// `PROMPT <text>` with the `TOKENS` value in force where it appeared.
    Prompt { text: String, tokens: usize },
    /// `BENCH <n>`: generate n tokens from [`BENCH_PROMPT`].
    Bench { tokens: usize },
    /// `NETCHECK host:port` — phase 1a. Recorded now, executed there.
    NetCheck { target: String },
    /// `CPUID` — identity/leaf dump. No measurement (design §2).
    Cpuid,
    /// `VERIFY <NAME>` — replay a witness receipt through
    /// [`crate::verifier::run`] (design §2).
    Verify { name: String },
    /// `MECH` — the diagnostic block of [`crate::lab::mech`], as a directive.
    ///
    /// **Moved last by the dispatcher regardless of file order** (design §2):
    /// it runs ~24 minutes under TCG and must never starve real work.
    Mech,
}

/// A parsed `JOB.TXT`.
#[derive(Clone, Debug)]
pub struct Job {
    pub budget_s: u64,
    pub mode: Mode,
    pub net: NetCfg,
    /// `REPORT <url>` — phase 1b posts `RESULT.TXT` here.
    pub report: Option<String>,
    /// `TOKEN <value>` — the shared secret a collector may require. Sent as
    /// `X-Aefinity-Token` on the `REPORT` POST and on nothing else: a
    /// collector reachable from a LAN is an unauthenticated write endpoint
    /// without it (`docs/UEFI-REMOTE-LANE.md`, "Exposing the collector").
    /// Not in spec §2's table; an addition, recorded in
    /// `docs/AEFINITY_OS_STATUS.md`.
    pub token: Option<String>,
    pub listen: u16,
    /// `RUNID <id>` (design §2) — the idempotency key, same grammar as the
    /// `RUNID` verb. It names the record: `run_id=` in `RESULT.TXT` is this
    /// when the file carries one, and the `RUN.ID` file otherwise, so the id a
    /// controller dedupes on and the id in the record it stores are one string
    /// and the v0.1 `run_id=` key keeps its v0.1 position.
    pub run_id: Option<String>,
    /// `TAG <text>` (design §2) — free-form, echoed into `RESULT.TXT` and the
    /// ledger. Deliberately **outside** `merge_key` (§3.1): a tag is a label
    /// for humans, not an input two boxes have to agree on.
    pub tag: Option<String>,
    /// `SHARD <i>/<n>` (design §2), `1 <= i <= n <= SHARD_MAX`.
    ///
    /// Informational to the box and load-bearing to the collector: the box
    /// never computes its own slice — the controller writes concrete work into
    /// the body — so a record can be attributed without trusting the
    /// collector's bookkeeping alone. Outside `merge_key` for the same reason
    /// as `tag`.
    pub shard: Option<(u32, u32)>,
    /// `SEED <u64>` (design §2) — inside `merge_key`, because two records that
    /// used different seeds are not replications of each other.
    pub seed: Option<u64>,
    /// `STRICT on` (design §2): the first failing step stops the job with
    /// `verdict=FAIL <step-kind>` and the steps behind it are not run.
    pub strict: bool,
    /// The `TOKENS` value left in force at end of file.
    pub tokens: usize,
    pub after: After,
    pub steps: Vec<Step>,
    /// How many directive lines were understood (for the BOOTLOG line).
    pub directives: usize,
    /// Keys that were not recognised. Logged and ignored (spec §2).
    pub unknown: Vec<String>,
}

impl Default for Job {
    fn default() -> Self {
        Job {
            budget_s: DEFAULT_BUDGET_S,
            mode: Mode::Oneshot,
            net: NetCfg::Dhcp,
            report: None,
            token: None,
            listen: DEFAULT_LISTEN,
            run_id: None,
            tag: None,
            shard: None,
            seed: None,
            strict: false,
            tokens: DEFAULT_TOKENS,
            after: After::Reset,
            steps: Vec::new(),
            directives: 0,
            unknown: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing (spec §2)
// ---------------------------------------------------------------------------

/// Parse a `JOB.TXT` body. Accepts CRLF or LF, `#` starts a comment, blank
/// lines are skipped, keys are case-insensitive. A malformed value leaves the
/// default in place and is recorded in `unknown` so it reaches `BOOTLOG.TXT`
/// — a job that silently ran with a different budget than the file asked for
/// would be worse than one that says so.
pub fn parse(body: &str) -> Job {
    let mut job = Job::default();

    for raw in body.lines() {
        let line = match raw.find('#') {
            Some(i) => &raw[..i],
            None => raw,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (key, rest) = match line.find(char::is_whitespace) {
            Some(i) => (&line[..i], line[i..].trim()),
            None => (line, ""),
        };

        let mut key_buf = [0u8; 16];
        let key_uc = upper_ascii(key, &mut key_buf);
        let mut understood = true;

        match key_uc {
            "BUDGET" => match rest.parse::<u64>() {
                Ok(v) if v > 0 => job.budget_s = v,
                _ => {
                    understood = false;
                    job.unknown
                        .push(format!("BUDGET {rest:?} (not a positive integer)"));
                }
            },
            "MODE" => {
                if eq_ascii_ci(rest, "oneshot") {
                    job.mode = Mode::Oneshot;
                } else if eq_ascii_ci(rest, "resident") {
                    job.mode = Mode::Resident;
                } else {
                    understood = false;
                    job.unknown
                        .push(format!("MODE {rest:?} (want oneshot|resident)"));
                }
            }
            "NET" => {
                let mut it = rest.split_whitespace();
                match it.next() {
                    Some(w) if eq_ascii_ci(w, "dhcp") => job.net = NetCfg::Dhcp,
                    Some(w) if eq_ascii_ci(w, "static") => match (it.next(), it.next()) {
                        (Some(cidr), Some(gw)) => {
                            job.net = NetCfg::Static {
                                cidr: cidr.to_string(),
                                gateway: gw.to_string(),
                            }
                        }
                        _ => {
                            understood = false;
                            job.unknown
                                .push(format!("NET static {rest:?} (want <cidr> <gateway>)"));
                        }
                    },
                    _ => {
                        understood = false;
                        job.unknown.push(format!("NET {rest:?} (want dhcp|static)"));
                    }
                }
            }
            "REPORT" => {
                if rest.is_empty() {
                    understood = false;
                    job.unknown.push(String::from("REPORT (empty url)"));
                } else {
                    job.report = Some(rest.to_string());
                }
            }
            "TOKEN" => {
                // A token is pasted into an HTTP header, so a control
                // character in it would be header injection: refuse it here,
                // where it becomes one BOOTLOG line, rather than on the wire.
                if rest.is_empty() {
                    understood = false;
                    job.unknown.push(String::from("TOKEN (empty)"));
                } else if rest.bytes().any(|b| b < 0x20 || b == 0x7F) {
                    understood = false;
                    job.unknown
                        .push(String::from("TOKEN (control character in value)"));
                } else {
                    job.token = Some(rest.to_string());
                }
            }
            "LISTEN" => match rest.parse::<u16>() {
                Ok(v) if v > 0 => job.listen = v,
                _ => {
                    understood = false;
                    job.unknown.push(format!("LISTEN {rest:?} (not a port)"));
                }
            },
            "TOKENS" => match rest.parse::<usize>() {
                Ok(v) if v > 0 => job.tokens = v.min(MAX_TOKENS),
                _ => {
                    understood = false;
                    job.unknown
                        .push(format!("TOKENS {rest:?} (not a positive integer)"));
                }
            },
            "PROMPT" => {
                if rest.is_empty() {
                    understood = false;
                    job.unknown.push(String::from("PROMPT (empty)"));
                } else {
                    let text = truncate_bytes(rest, PROMPT_MAX_BYTES);
                    job.steps.push(Step::Prompt {
                        text: text.to_string(),
                        tokens: job.tokens,
                    });
                }
            }
            "BENCH" => match rest.parse::<usize>() {
                Ok(v) if v > 0 => job.steps.push(Step::Bench {
                    tokens: v.min(MAX_TOKENS),
                }),
                _ => {
                    understood = false;
                    job.unknown
                        .push(format!("BENCH {rest:?} (not a positive integer)"));
                }
            },
            "AFTER" => {
                if eq_ascii_ci(rest, "reset") {
                    job.after = After::Reset;
                } else if eq_ascii_ci(rest, "halt") {
                    job.after = After::Halt;
                } else {
                    understood = false;
                    job.unknown
                        .push(format!("AFTER {rest:?} (want reset|halt)"));
                }
            }
            // ---- design §2 (phase 5): the fleet directives ---------------
            "RUNID" => {
                if valid_runid(rest) {
                    job.run_id = Some(rest.to_string());
                } else {
                    understood = false;
                    job.unknown
                        .push(format!("RUNID {rest:?} (want [A-Za-z0-9._-]{{1,64}})"));
                }
            }
            "TAG" => {
                if rest.is_empty() {
                    understood = false;
                    job.unknown.push(String::from("TAG (empty)"));
                } else {
                    job.tag = Some(truncate_bytes(rest, TAG_MAX_BYTES).to_string());
                }
            }
            "SHARD" => match parse_shard(rest) {
                Some(v) => job.shard = Some(v),
                None => {
                    understood = false;
                    job.unknown.push(format!(
                        "SHARD {rest:?} (want <i>/<n>, 1<=i<=n<={SHARD_MAX})"
                    ));
                }
            },
            "SEED" => match rest.parse::<u64>() {
                Ok(v) => job.seed = Some(v),
                Err(_) => {
                    understood = false;
                    job.unknown.push(format!("SEED {rest:?} (not a u64)"));
                }
            },
            "STRICT" => {
                if eq_ascii_ci(rest, "on") {
                    job.strict = true;
                } else if eq_ascii_ci(rest, "off") {
                    job.strict = false;
                } else {
                    understood = false;
                    job.unknown.push(format!("STRICT {rest:?} (want on|off)"));
                }
            }
            "CPUID" => {
                // Argument-free, like `MECH`.
                if rest.is_empty() {
                    job.steps.push(Step::Cpuid);
                } else {
                    understood = false;
                    job.unknown
                        .push(format!("CPUID {rest:?} (takes no argument)"));
                }
            }
            "VERIFY" => {
                // The `<NAME>` is validated by `files.rs`'s rules when the
                // step runs, not here: a parse that rejected it would turn a
                // typo into an ignored directive instead of a `job.N.err` the
                // controller can read.
                if rest.is_empty() {
                    understood = false;
                    job.unknown.push(String::from("VERIFY (empty name)"));
                } else {
                    job.steps.push(Step::Verify {
                        name: rest.to_string(),
                    });
                }
            }
            "MECH" => {
                // Argument-free. A trailing word is a typo, not a parameter,
                // and running a 24-minute diagnostic because someone misspelt
                // a directive is exactly what the unknown list is for.
                if rest.is_empty() {
                    job.steps.push(Step::Mech);
                } else {
                    understood = false;
                    job.unknown
                        .push(format!("MECH {rest:?} (takes no argument)"));
                }
            }
            "NETCHECK" => {
                if rest.is_empty() {
                    understood = false;
                    job.unknown.push(String::from("NETCHECK (empty target)"));
                } else {
                    job.steps.push(Step::NetCheck {
                        target: rest.to_string(),
                    });
                }
            }
            other => {
                understood = false;
                job.unknown.push(other.to_string());
            }
        }

        if understood {
            job.directives += 1;
        }
    }

    job
}

/// ASCII-uppercase `s` into `buf`, returning the borrowed view. Non-ASCII or
/// over-long keys come back unchanged, so they fall into the unknown-key arm
/// instead of being silently mangled into a directive.
fn upper_ascii<'b>(s: &'b str, buf: &'b mut [u8; 16]) -> &'b str {
    if s.len() > buf.len() || !s.is_ascii() {
        return s;
    }
    let n = s.len();
    buf[..n].copy_from_slice(s.as_bytes());
    buf[..n].make_ascii_uppercase();
    // SAFETY-free: ASCII-uppercasing ASCII bytes yields ASCII bytes, so this
    // is always valid UTF-8; the fallback keeps the function total anyway.
    core::str::from_utf8(&buf[..n]).unwrap_or(s)
}

fn eq_ascii_ci(a: &str, b: &str) -> bool {
    a.len() == b.len()
        && a.bytes()
            .zip(b.bytes())
            .all(|(x, y)| x.eq_ignore_ascii_case(&y))
}

/// `[A-Za-z0-9._-]{1,64}` — the `RUNID` grammar of design §1.2, shared by the
/// verb (`server.rs`) and the `JOB.TXT` directive so a controller cannot write
/// an id into a file that the socket would refuse.
#[must_use]
pub fn valid_runid(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= RUNID_MAX_BYTES
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

/// `SHARD <i>/<n>` with `1 <= i <= n <= SHARD_MAX` (design §2).
fn parse_shard(rest: &str) -> Option<(u32, u32)> {
    let (a, b) = rest.split_once('/')?;
    let i: u32 = a.trim().parse().ok()?;
    let n: u32 = b.trim().parse().ok()?;
    if i >= 1 && i <= n && n <= SHARD_MAX {
        Some((i, n))
    } else {
        None
    }
}

/// Truncate to at most `max` bytes on a UTF-8 character boundary.
fn truncate_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ---------------------------------------------------------------------------
// Escaping (spec §3)
// ---------------------------------------------------------------------------

/// Escape a value for a `key=value` line: `\\`, `\n`, `\r`, and any byte
/// outside printable ASCII as `\xNN`. The result is pure printable ASCII, so
/// a `RESULT.TXT` survives a serial console, an HTTP POST body and a text
/// editor without the collector having to guess an encoding.
pub fn escape(s: &str) -> String {
    escape_capped(s, usize::MAX)
}

/// [`escape`], stopping before the output would exceed `max` bytes. Escape
/// sequences are emitted whole or not at all, so the value never ends in a
/// half-written `\x`.
pub fn escape_capped(s: &str, max: usize) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        let mut unit = [0u8; 4];
        let piece: &str = match b {
            b'\\' => "\\\\",
            b'\n' => "\\n",
            b'\r' => "\\r",
            0x20..=0x7e => {
                unit[0] = b;
                core::str::from_utf8(&unit[..1]).unwrap_or("?")
            }
            _ => {
                const HEX: &[u8; 16] = b"0123456789abcdef";
                unit[0] = b'\\';
                unit[1] = b'x';
                unit[2] = HEX[(b >> 4) as usize];
                unit[3] = HEX[(b & 0x0f) as usize];
                core::str::from_utf8(&unit[..4]).unwrap_or("?")
            }
        };
        if out.len() + piece.len() > max {
            break;
        }
        out.push_str(piece);
    }
    out
}

// ---------------------------------------------------------------------------
// The result record (spec §3)
// ---------------------------------------------------------------------------

/// One `job.N.*` block.
#[derive(Clone, Debug)]
pub struct StepResult {
    /// `prompt` | `bench` | `netcheck`.
    pub kind: &'static str,
    /// The prompt actually conditioned on (the fixed bench prompt for BENCH).
    pub prompt: String,
    pub tokens: usize,
    pub wall_ms: u64,
    /// Structural only — see the module note on Rule A.
    pub tps: f64,
    /// RDTSC ticks per generated token. Ticks, not cycles.
    pub tsc_per_tok: u64,
    /// First 16 hex of sha256 over the generated ids as little-endian u32.
    pub digest: String,
    pub response: String,
    /// `job.N.ok` — emitted only for step kinds whose result is a pass/fail
    /// rather than a generation (`netcheck`, phase 1a). `None` leaves the
    /// `prompt`/`bench` blocks byte-for-byte what phase 0 wrote.
    pub ok: Option<bool>,
    // ---- design §3 (phase 5) --------------------------------------------
    /// `job.N.rate_valid` — **false whenever `env=vm`** (design §3, Rule A
    /// enforced by the artifact). Every step carries it, including the v0.1
    /// kinds, so a reader never has to know which kinds can produce a rate.
    pub rate_valid: bool,
    /// `job.N.nll_q16` — `EVAL` only (§2.1). Exact integer; comparable across
    /// boxes by construction, and never a performance figure.
    pub nll_q16: Option<u64>,
    /// `job.N.ntok` — `EVAL` only: scored positions, not tokens read.
    pub ntok: Option<u64>,
    /// `job.N.membw_mibs` — `MEMBW` only, and gated exactly like `tps`: `n/a`
    /// unless [`StepResult::rate_valid`] (design §3).
    pub membw_mibs: Option<u64>,
    /// `job.N.pass` — the pass/fail kinds (`cpuid`, `verify`).
    pub pass: Option<bool>,
    /// `job.N.items` — how many units of work the step actually did
    /// (`verify`: decode steps replayed).
    pub items: Option<u64>,
    /// `job.N.partial` — `0` when the step completed, `k` when a budget
    /// overrun stopped it after `k` of the requested units (design §5,
    /// "Partial results": prefix evidence, never a completed unit).
    pub partial: Option<u64>,
    /// `job.N.err` — `none` or one short slug (design §1.3).
    pub err: Option<String>,
    /// `job.N.detail` — escaped, capped at [`DETAIL_MAX`] bytes.
    pub detail: Option<String>,
    /// `<step-input>` for [`merge_key`] (§3.1), for the kinds whose input is
    /// not derivable from the v0.1 fields. `None` falls back to the phase-4
    /// derivation documented on [`merge_key`].
    pub merge_input: Option<String>,
}

impl StepResult {
    /// A step block with only the fields a lab kind fills, and every v0.1
    /// generation field left at its zero.
    ///
    /// The v0.1 kinds keep building [`StepResult`] literally, so their blocks
    /// stay byte-for-byte what phase 0 wrote (plus the one appended
    /// `rate_valid` line).
    #[must_use]
    pub fn lab(kind: &'static str, rate_valid: bool) -> StepResult {
        StepResult {
            kind,
            prompt: String::new(),
            tokens: 0,
            wall_ms: 0,
            tps: 0.0,
            tsc_per_tok: 0,
            digest: String::new(),
            response: String::new(),
            ok: None,
            rate_valid,
            nll_q16: None,
            ntok: None,
            membw_mibs: None,
            pass: None,
            items: None,
            partial: None,
            err: None,
            detail: None,
            merge_input: None,
        }
    }

    /// Is this one of the three kinds phase 0/1a shipped?
    ///
    /// Their blocks render exactly the v0.1 keys; a lab kind renders §3's
    /// keys instead, because `job.N.tps=0.00` on a `CPUID` step would be a
    /// performance number where no measurement was made (Rule A).
    #[must_use]
    fn is_v01(&self) -> bool {
        matches!(self.kind, "prompt" | "bench" | "netcheck")
    }

    /// Did this step fail? The question `STRICT on` asks after every step.
    #[must_use]
    pub fn failed(&self) -> bool {
        self.ok == Some(false)
            || self.pass == Some(false)
            || self.err.as_deref().is_some_and(|e| e != "none")
    }
}

/// The whole `RESULT.TXT`, in the key order of spec §3.
#[derive(Clone, Debug)]
pub struct ResultRecord {
    pub run_id: String,
    pub env: &'static str,
    pub cpu_brand: String,
    pub cpuid_sig: u32,
    pub mac: String,
    pub ip: String,
    /// `net` — how [`ip`](Self::ip) was obtained: `dhcp` | `static` | `none`.
    ///
    /// Beyond the key list of spec §3, deliberately. The address alone cannot
    /// say where it came from, and on QEMU's user network the static address
    /// a job asks for (10.0.2.15) is byte-identical to the first address
    /// slirp's DHCP server hands out — so a gate reading only `ip=` cannot
    /// tell a working DHCP client from one that silently fell back, and
    /// neither can a collector. This makes the provenance a fact in the
    /// record instead of something only a BOOTLOG line knew.
    pub net: String,
    pub budget_s: u64,
    pub steps: Vec<StepResult>,
    /// `ok` | `fail <reason>` | `none` (phase 1b fills the first two).
    pub report: String,
    /// `OK` | `FAIL <reason>`.
    pub verdict: String,
    /// Design §3's phase-4 additions, rendered after `verdict=`. `None` on a
    /// box that has no file plane in force (nothing does today; the field is
    /// an `Option` so a record can still be rendered before the slot is
    /// consulted).
    pub fleet: Option<FleetInfo>,
}

/// The phase-4 additions to the record (design §3), carried as a unit so the
/// v0.1 key order above is never disturbed: everything here renders **after**
/// `verdict=`.
///
/// Optional because a oneshot boot has no listener and therefore no honest
/// `uptime_s`/`served` to report — a `0` there would be a claim, not a
/// default.
#[derive(Clone, Debug)]
pub struct FleetInfo {
    /// `<model16>/<embed16>/<vocab16>` of the **resident** buffers.
    pub artifacts: String,
    /// Full sha256 of the resident `MODEL.SAF` buffer.
    pub model_sha: String,
    /// Full digests of all three, for [`merge_key`].
    pub full: (String, String, String),
    pub reloads: u64,
    /// `(uptime_s, served)` — resident mode only.
    pub resident: Option<(u64, u64)>,
    /// Regular files the boot volume root holds.
    pub files: usize,
    /// `true` when this record came out of the `RUNID` ring rather than being
    /// re-run (design §1.2/§1.4).
    pub replay: bool,
    /// `SEED` (design §2-ext). Serialised as the literal `none` inside
    /// [`merge_key`] when absent, so the key a phase-4 box computes is the key
    /// a phase-5 box computes for the same seedless job.
    pub seed: Option<u64>,
    /// `TAG <text>` (design §2). Echoed into the record and **excluded** from
    /// [`merge_key`] (§3.1).
    pub tag: Option<String>,
    /// `SHARD <i>/<n>` (design §2). Echoed and excluded, as `tag` is.
    pub shard: Option<(u32, u32)>,
}

/// `merge_key` — the first 16 lower-case hex of sha256 over design §3.1's
/// byte-exact input serialization.
///
/// NUL-terminated ASCII fields, in this order, **with no other separator**:
///
/// ```text
/// "v1"  
/// <model 64hex>   <embed 64hex>   <vocab 64hex>  
/// <env>  
/// <seed decimal, or "none">  
///   for each step, dispatch order, N ascending:
///     <N decimal>   <kind>   <step-input>  
/// "END"  
/// ```
///
/// It **excludes** `cpu_brand`, `shard`, `tag` and `run_id` — those are what
/// the comparison is about — and **includes `env`**, so an `iron` record and a
/// TCG record can never share a key and can never be presented as replicating
/// each other (Rule A, enforced by the artifact).
///
/// `<step-input>` by kind is §3.1's table for the lab kinds — `cpuid` and
/// `mech` empty, `membw` the MiB decimal, `verify` `<NAME>:<receipt 64hex>`,
/// `eval` `<NAME>:<corpus payload 64hex>:<lo>:<hi>:<W>` — carried on the step
/// as [`StepResult::merge_input`] because a receipt's or a corpus's digest is
/// not derivable from the v0.1 fields.
///
/// The three v0.1 kinds are **not** in §3.1's table, so this is the phase-4
/// extension of it, recorded here and in `docs/AEFINITY_OS_STATUS.md`:
/// `prompt` → the prompt text, `bench` → the token count, `netcheck` → the
/// target. Empty would have been simpler and wrong — two different prompts
/// would then share a merge key, which is the one thing the key exists to
/// prevent.
#[must_use]
pub fn merge_key(fleet: &FleetInfo, env: &str, steps: &[StepResult]) -> String {
    let mut h = aegis_core::witness::Sha256::new();
    let field = |h: &mut aegis_core::witness::Sha256, s: &str| {
        h.update(s.as_bytes());
        h.update(&[0u8]);
    };
    field(&mut h, "v1");
    field(&mut h, &fleet.full.0);
    field(&mut h, &fleet.full.1);
    field(&mut h, &fleet.full.2);
    field(&mut h, env);
    match fleet.seed {
        Some(v) => field(&mut h, &format!("{v}")),
        None => field(&mut h, "none"),
    }
    for (i, st) in steps.iter().enumerate() {
        field(&mut h, &format!("{}", i + 1));
        field(&mut h, st.kind);
        // §3.1's table for the lab kinds is carried on the step itself —
        // `verify` needs the receipt file's digest and `eval` the corpus
        // payload's, neither of which is derivable from the v0.1 fields.
        let input = match &st.merge_input {
            Some(v) => v.clone(),
            None => match st.kind {
                "prompt" => st.prompt.clone(),
                "bench" => format!("{}", st.tokens),
                _ => st.prompt.clone(),
            },
        };
        field(&mut h, &input);
    }
    field(&mut h, "END");
    let full = h.finalize();
    let mut hex = [0u8; 64];
    let n = aegis_core::witness::hex_lower(&full, &mut hex);
    String::from_utf8_lossy(&hex[..n.min(16)]).into_owned()
}

impl ResultRecord {
    /// A record pre-filled with this machine's identity and no jobs yet.
    pub fn new(budget_s: u64, run_id: String) -> Self {
        let mut brand_buf = [0u8; 48];
        ResultRecord {
            run_id,
            env: crate::sysinfo::env().as_str(),
            cpu_brand: crate::sysinfo::cpu_brand(&mut brand_buf).to_string(),
            cpuid_sig: crate::sysinfo::cpuid_sig(),
            mac: String::from("none"),
            ip: String::from("none"),
            net: String::from("none"),
            budget_s,
            steps: Vec::new(),
            report: String::from("none"),
            verdict: String::from("OK"),
            fleet: None,
        }
    }

    /// Record a failure once. The first reason wins: a budget overrun that
    /// then produces an empty response should be reported as the overrun.
    pub fn fail(&mut self, reason: &str) {
        if self.verdict == "OK" {
            self.verdict = format!("FAIL {reason}");
        }
    }

    /// Render the record exactly as spec §3 lays it out: `key=value`, LF,
    /// ASCII only.
    pub fn render(&self) -> String {
        let mut s = String::with_capacity(512 + self.steps.len() * 512);
        s.push_str("aefinity_os=0.1\n");
        s.push_str(&format!("run_id={}\n", escape(&self.run_id)));
        s.push_str(&format!("env={}\n", self.env));
        s.push_str(&format!("cpu_brand={}\n", escape(&self.cpu_brand)));
        s.push_str(&format!("cpuid_sig={:08x}\n", self.cpuid_sig));
        s.push_str(&format!("mac={}\n", self.mac));
        s.push_str(&format!("ip={}\n", self.ip));
        s.push_str(&format!("net={}\n", self.net));
        s.push_str(&format!("budget_s={}\n", self.budget_s));
        s.push_str(&format!("jobs={}\n", self.steps.len()));
        for (i, j) in self.steps.iter().enumerate() {
            let n = i + 1;
            s.push_str(&format!("job.{n}.kind={}\n", j.kind));
            if let Some(ok) = j.ok {
                s.push_str(&format!("job.{n}.ok={ok}\n"));
            }
            if j.is_v01() {
                // Spec §3 caps `response`, not `prompt`; a PROMPT line is
                // already bounded to 4 KiB at parse time.
                s.push_str(&format!("job.{n}.prompt={}\n", escape(&j.prompt)));
                s.push_str(&format!("job.{n}.tokens={}\n", j.tokens));
                s.push_str(&format!("job.{n}.wall_ms={}\n", j.wall_ms));
                s.push_str(&format!("job.{n}.tps={}\n", fixed2(j.tps)));
                s.push_str(&format!("job.{n}.tsc_per_tok={}\n", j.tsc_per_tok));
                s.push_str(&format!("job.{n}.digest={}\n", j.digest));
                s.push_str(&format!(
                    "job.{n}.response={}\n",
                    escape_capped(&j.response, RESPONSE_MAX)
                ));
            } else {
                // A lab block quotes no `tps` and no `tsc_per_tok`: no rate
                // was measured, and a `0.00` there would read as one (Rule A).
                s.push_str(&format!("job.{n}.wall_ms={}\n", j.wall_ms));
                if !j.digest.is_empty() {
                    s.push_str(&format!("job.{n}.digest={}\n", j.digest));
                }
            }
            // ---- design §3, appended after the v0.1 keys of the block ----
            s.push_str(&format!("job.{n}.rate_valid={}\n", j.rate_valid));
            if let Some(v) = j.nll_q16 {
                s.push_str(&format!("job.{n}.nll_q16={v}\n"));
            }
            if let Some(v) = j.ntok {
                s.push_str(&format!("job.{n}.ntok={v}\n"));
            }
            if j.membw_mibs.is_some() {
                // Gated exactly like `tps` (design §3): a bandwidth number
                // measured under TCG is meaningless, so the record says so
                // rather than printing it and relying on someone reading
                // `env=`.
                let v = match (j.rate_valid, j.membw_mibs) {
                    (true, Some(v)) => format!("{v}"),
                    _ => String::from("n/a"),
                };
                s.push_str(&format!("job.{n}.membw_mibs={v}\n"));
            }
            if let Some(v) = j.pass {
                s.push_str(&format!("job.{n}.pass={v}\n"));
            }
            if let Some(v) = j.items {
                s.push_str(&format!("job.{n}.items={v}\n"));
            }
            if let Some(v) = j.partial {
                s.push_str(&format!("job.{n}.partial={v}\n"));
            }
            if let Some(v) = &j.err {
                s.push_str(&format!("job.{n}.err={v}\n"));
            }
            if let Some(v) = &j.detail {
                s.push_str(&format!(
                    "job.{n}.detail={}\n",
                    escape_capped(v, DETAIL_MAX)
                ));
            }
        }
        s.push_str(&format!("report={}\n", self.report));
        s.push_str(&format!("verdict={}\n", self.verdict));
        // ---- design §3: appended after the existing keys, never inside them.
        // Every v0.1 assertion reads a key by name and the first match wins,
        // so appending cannot move anything a v0.1 client was reading.
        if let Some(f) = &self.fleet {
            // Design §3 renders `run_id`, `tag`, `shard`, `seed` together.
            // `run_id=` is a v0.1 key and keeps its v0.1 position above —
            // `RUNID` in `JOB.TXT` sets its *value*, and emitting it twice
            // would give a collector two answers to one question. The other
            // three are new and land here, ahead of `replay=`, in §3's order.
            if let Some(t) = &f.tag {
                s.push_str(&format!("tag={}\n", escape(t)));
            }
            if let Some((i, n)) = f.shard {
                s.push_str(&format!("shard={i}/{n}\n"));
            }
            if let Some(v) = f.seed {
                s.push_str(&format!("seed={v}\n"));
            }
            s.push_str(&format!("replay={}\n", f.replay));
            s.push_str(&format!("artifacts={}\n", f.artifacts));
            s.push_str(&format!("model_sha={}\n", f.model_sha));
            s.push_str(&format!("reloads={}\n", f.reloads));
            if let Some((up, served)) = f.resident {
                s.push_str(&format!("uptime_s={up}\n"));
                s.push_str(&format!("served={served}\n"));
            }
            s.push_str(&format!("files={}\n", f.files));
            s.push_str(&format!(
                "merge_key={}\n",
                merge_key(f, self.env, &self.steps)
            ));
        }
        s
    }
}

/// Two fixed decimals without `core::fmt`'s float formatter, which pulls in a
/// large formatting path this binary does not otherwise need. Negative and
/// non-finite inputs cannot occur here (both operands of the division are
/// non-negative and the divisor is checked), and are floored to `0.00`.
fn fixed2(v: f64) -> String {
    if v.is_nan() || v <= 0.0 {
        return String::from("0.00");
    }
    let hundredths = (v * 100.0) as u64;
    format!("{}.{:02}", hundredths / 100, hundredths % 100)
}

// ---------------------------------------------------------------------------
// Digest (spec §3)
// ---------------------------------------------------------------------------

/// sha256 over the generated token ids serialised as little-endian `u32`,
/// rendered as the first 16 hex characters. This is the CIS-style witness of
/// *what was generated*: two boxes running the same `JOB.TXT` against the
/// same artifacts must agree here, and that comparison — not a tok/s figure —
/// is the fleet check (spec §1/§3).
pub fn token_digest(ids: &[u32]) -> String {
    let mut h = aegis_core::witness::Sha256::new();
    for &id in ids {
        h.update(&id.to_le_bytes());
    }
    let full = h.finalize();
    let mut hex = [0u8; 64];
    let n = aegis_core::witness::hex_lower(&full, &mut hex);
    let take = n.min(16);
    core::str::from_utf8(&hex[..take]).unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// Firmware actions (spec §5)
// ---------------------------------------------------------------------------

/// Arm the firmware watchdog for `secs` seconds; `0` disables it.
///
/// This is the only thing standing between a hung job and a box that is
/// wedged until someone walks to the rack: the firmware, not the unikernel,
/// performs the reset, so it survives a hang anywhere in our own code.
/// Returns whether the firmware accepted it — a firmware that refuses is
/// logged, not fatal.
pub fn arm_watchdog(secs: u64) -> bool {
    uefi::boot::set_watchdog_timer(secs as usize, WATCHDOG_CODE, None).is_ok()
}

/// Perform the `AFTER` action. `Reset` never returns; `Halt` returns so the
/// caller falls through to whatever this build does next.
pub fn after(what: After, root: &mut Directory) {
    match what {
        After::Reset => {
            crate::boot_log(root, "RESET: job complete");
            // Disarm first: the reset path itself must not race a watchdog
            // expiry, and a firmware that ignores ResetSystem should leave a
            // clean state behind rather than a pending reset of its own.
            arm_watchdog(0);
            uefi::runtime::reset(uefi::runtime::ResetType::COLD, uefi::Status::SUCCESS, None);
        }
        After::Halt => {
            arm_watchdog(0);
            crate::boot_log(root, "HALT: job complete, watchdog disarmed");
        }
    }
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

/// Read a whole file from the boot volume root, capped at `max` bytes.
/// `None` for absent, empty, oversized or unreadable — the caller treats all
/// four the same way (no job).
fn read_small_file(root: &mut Directory, name: &str, max: usize) -> Option<Vec<u8>> {
    use uefi::proto::media::file::{File, FileAttribute, FileInfo, FileMode, FileType};
    let mut namebuf = [0u16; 32];
    let cstr = uefi::CStr16::from_str_with_buf(name, &mut namebuf).ok()?;
    let handle = root
        .open(cstr, FileMode::Read, FileAttribute::empty())
        .ok()?;
    let mut file = match handle.into_type().ok()? {
        FileType::Regular(f) => f,
        _ => return None,
    };
    let mut info_buf = [0u8; 256];
    let size = file.get_info::<FileInfo>(&mut info_buf).ok()?.file_size() as usize;
    if size == 0 || size > max {
        file.close();
        return None;
    }
    let mut buf = alloc::vec![0u8; size];
    let read = file.read(&mut buf).ok();
    file.close();
    let read = read?;
    buf.truncate(read);
    Some(buf)
}

/// Load and parse `JOB.TXT` from the boot volume root. `None` means "no job":
/// the caller must leave boot behaviour untouched.
pub fn load(root: &mut Directory) -> Option<Job> {
    let bytes = read_small_file(root, "JOB.TXT", JOB_MAX_BYTES)?;
    let text = match core::str::from_utf8(&bytes) {
        Ok(t) => t,
        Err(_) => {
            crate::boot_log(root, "JOB: JOB.TXT is not valid UTF-8 — ignored");
            return None;
        }
    };
    let job = parse(text);
    // Design §8, "Orphans": a `STAGE.PRT` on the volume at boot is the
    // wreckage of a `PUT` that a reset or a power loss interrupted. It is
    // deleted here and named in `BOOTLOG.TXT`, and `HEALTH parts=1` (before
    // this sweep runs, on the next boot) is what tells a scheduler the box is
    // `SUSPECT`.
    //
    // Deliberately after the `JOB.TXT` read, not before it: a box with no
    // `JOB.TXT` is not a fleet box, has no file plane, and its boot path is
    // untouched by phase 4 — which is the one thing this phase promised not
    // to change.
    if crate::files::sweep_parts(root) {
        crate::boot_log(
            root,
            "JOB: swept a stale STAGE.PRT left by an interrupted PUT",
        );
    }
    crate::boot_log(
        root,
        &format!(
            "JOB: parsed {} directives, {} steps, budget={}s mode={:?} after={:?}",
            job.directives,
            job.steps.len(),
            job.budget_s,
            job.mode,
            job.after
        ),
    );
    for k in &job.unknown {
        crate::boot_log(
            root,
            &format!("JOB: ignored unknown/invalid directive {k:?}"),
        );
    }
    Some(job)
}

/// Name of the in-progress record (see [`write_progress_record`]). A separate
/// file, deliberately: spec §3 says `RESULT.TXT` is written ONCE, at the end
/// of the job, and a collector that can find a half-finished record under that
/// name loses the guarantee that what it reads is a finished one.
pub const WIP_NAME: &str = "RESULT.WIP";

/// Write `RESULT.TXT` to the boot volume root: one create, one write, one
/// flush, one close, overwriting rather than appending (spec §3).
pub fn write_result_txt(root: &mut Directory, body: &str) -> bool {
    write_named(root, "RESULT.TXT", body)
}

/// Remove the in-progress marker from the boot volume.
///
/// [`dispatch`] does this itself for a oneshot job. The resident server
/// (`server.rs`) writes `RESULT.TXT` for each socket-delivered job without
/// going through `dispatch`, and the marker means "this box did not finish"
/// (spec §3) — a resident box that left one behind after a job it *did*
/// finish would say something false about itself on the next harvest.
pub fn clear_wip(root: &mut Directory) {
    delete_named(root, WIP_NAME);
}

/// Write `body` to `name` on the boot volume root.
///
/// The stale file is deleted first. Opening `CreateReadWrite` over a longer
/// previous record and writing a shorter one would leave the previous run's
/// tail on disk, and a collector reading that would see two records fused.
fn write_named(root: &mut Directory, name: &str, body: &str) -> bool {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
    let mut namebuf = [0u16; 32];
    let cstr = match uefi::CStr16::from_str_with_buf(name, &mut namebuf) {
        Ok(c) => c,
        Err(_) => return false,
    };
    if let Ok(h) = root.open(cstr, FileMode::ReadWrite, FileAttribute::empty())
        && let Ok(FileType::Regular(f)) = h.into_type()
    {
        let _ = f.delete();
    }
    let handle = match root.open(cstr, FileMode::CreateReadWrite, FileAttribute::empty()) {
        Ok(h) => h,
        Err(_) => return false,
    };
    let mut file = match handle.into_type() {
        Ok(FileType::Regular(f)) => f,
        _ => return false,
    };
    let wrote = file.write(body.as_bytes()).is_ok();
    let flushed = file.flush().is_ok();
    file.close();
    wrote && flushed
}

/// Remove `name` from the boot volume root if it is there. Absent, unopenable
/// and undeletable all mean the same thing to the caller: nothing to do here.
fn delete_named(root: &mut Directory, name: &str) {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
    let mut namebuf = [0u16; 32];
    let Ok(cstr) = uefi::CStr16::from_str_with_buf(name, &mut namebuf) else {
        return;
    };
    if let Ok(h) = root.open(cstr, FileMode::ReadWrite, FileAttribute::empty())
        && let Ok(FileType::Regular(f)) = h.into_type()
    {
        let _ = f.delete();
    }
}

/// What a probe of a name on the boot volume found.
///
/// Deliberately not `bool`, and deliberately not [`read_small_file`]: that
/// returns `None` for absent, empty, oversized *and* unreadable alike, which
/// is the right answer for "give me this file" and the wrong one for "is this
/// file there". An empty `RESULT.WIP` is present. A firmware error that is
/// not `NOT_FOUND` says nothing either way, and the marker's whole meaning is
/// its presence, so a probe that reports it may not guess.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Presence {
    Absent,
    Present,
    Unknown,
}

impl Presence {
    fn as_str(self) -> &'static str {
        match self {
            Presence::Absent => "absent",
            Presence::Present => "present",
            Presence::Unknown => "unknown",
        }
    }
}

/// Probe `name` by opening it. `NOT_FOUND` is the only answer that means gone.
fn presence(root: &mut Directory, name: &str) -> Presence {
    use uefi::proto::media::file::{File, FileAttribute, FileMode};
    let mut namebuf = [0u16; 32];
    let Ok(cstr) = uefi::CStr16::from_str_with_buf(name, &mut namebuf) else {
        return Presence::Unknown;
    };
    match root.open(cstr, FileMode::Read, FileAttribute::empty()) {
        Ok(h) => {
            h.close();
            Presence::Present
        }
        Err(e) if e.status() == uefi::Status::NOT_FOUND => Presence::Absent,
        Err(_) => Presence::Unknown,
    }
}

/// Second, independent probe: does `name` appear in the boot volume root's
/// directory listing?
///
/// `open` asks the firmware to resolve a name, which a driver may answer from
/// its own state; this walks the directory the medium reports. One probe
/// agreeing with itself is not evidence, so when the two disagree the BOOTLOG
/// line prints both rather than picking one.
fn dir_has(root: &mut Directory, name: &str) -> Presence {
    if root.reset_entry_readout().is_err() {
        return Presence::Unknown;
    }
    loop {
        match root.read_entry_boxed() {
            Ok(None) => return Presence::Absent,
            Ok(Some(info)) => {
                if cstr16_eq_ascii_ci(info.file_name(), name) {
                    return Presence::Present;
                }
            }
            Err(_) => return Presence::Unknown,
        }
    }
}

/// ASCII case-insensitive compare of a firmware-returned name against ours.
/// FAT short names come back upper case; a non-ASCII name is not one of ours.
fn cstr16_eq_ascii_ci(a: &uefi::CStr16, b: &str) -> bool {
    let mut want = b.bytes();
    for c in a.iter() {
        let got = u16::from(*c);
        if got > 0x7f {
            return false;
        }
        match want.next() {
            Some(w) if (got as u8).eq_ignore_ascii_case(&w) => {}
            _ => return false,
        }
    }
    want.next().is_none()
}

/// `run_id` for the record: the value of a `run_id=` line in `RUN.ID` on the
/// boot volume, or `none`. The controller writes that file when it stages a
/// stick, so a returned result can be tied to the run that asked for it.
fn run_id(root: &mut Directory) -> String {
    let Some(bytes) = read_small_file(root, "RUN.ID", 4096) else {
        return String::from("none");
    };
    let Ok(text) = core::str::from_utf8(&bytes) else {
        return String::from("none");
    };
    for line in text.lines() {
        if let Some(v) = line.trim().strip_prefix("run_id=") {
            let v = v.trim();
            if !v.is_empty() {
                return v.to_string();
            }
        }
    }
    String::from("none")
}

// ---------------------------------------------------------------------------
// Running (spec §5)
// ---------------------------------------------------------------------------

fn say(msg: &str) {
    crate::console::with_console(|st| {
        let _ = st.write_str(msg);
        core::fmt::Result::Ok(())
    });
}

/// Seconds elapsed since `start`, from UEFI `GetTime()`. `None` when the
/// firmware has no clock, or when the reading crossed midnight and would
/// otherwise produce a negative interval — an unusable budget clock is
/// reported as unusable rather than guessed at.
fn elapsed_since(start: f64) -> Option<f64> {
    let now = crate::wall_seconds()?;
    if now >= start {
        Some(now - start)
    } else {
        None
    }
}

/// Write the record so far as [`WIP_NAME`], marked as an unfinished run of
/// step `n`.
///
/// The firmware watchdog resets the box without running any of our code, so a
/// step that overruns between two engine checkpoints leaves nothing behind
/// unless something was written *first*. This is that something: after a
/// watchdog reset the volume holds a record whose `verdict` names the step
/// that did not come back, and `BOOTLOG.TXT` says whether it was still in
/// prefill or already decoding. Silence is the one outcome a headless box
/// must never produce; a stale-looking record that names its own step is
/// strictly better.
///
/// `dispatch` deletes it once the real `RESULT.TXT` is on the volume, so its
/// presence is itself the signal: a stick that comes home with a `RESULT.WIP`
/// on it is a box that did not finish.
///
/// A failure to write is not fatal and not logged here — the next line in
/// `BOOTLOG.TXT` after this one is the step starting either way, and
/// `dispatch` logs the outcome of the real write.
fn write_progress_record(rec: &ResultRecord, root: &mut Directory, n: usize) {
    let mut wip = rec.clone();
    wip.verdict = format!("FAIL incomplete: step {n} did not return");
    let _ = write_named(root, WIP_NAME, &wip.render());
}

/// Run every directive in file order and build the record (spec §3).
///
/// The budget is wall-clock, read from UEFI `GetTime()` at entry, and it is
/// enforced in three places, because a headless box that overruns silently is
/// the failure this module exists to prevent:
///
/// 1. **Before a step starts** — work that cannot finish is not begun.
/// 2. **Inside the engine**, at coarse checkpoints in *both* phases
///    (`process_intent_with_deadline`: twice per decoder layer during
///    prefill, once per token during decode). Checking only the per-token
///    decode callback is not enough: a spec-legal 4 KiB `PROMPT` can spend
///    the whole budget in prefill, before the first token exists, and the
///    software path then never runs at all.
/// 3. **The firmware watchdog** at `BUDGET + WATCHDOG_MARGIN_S`, which is the
///    backstop for a hang between checkpoints.
///
/// A job stopped by 1 or 2 still writes a complete `RESULT.TXT` — a truncated
/// answer plus an honest verdict is worth more to the collector than silence.
/// For 3 the unikernel is not running any more, so the record is written
/// *before* the step instead: see [`write_progress_record`].
///
/// Returns the [`crate::net::Net`] this run brought up, if any, so
/// [`dispatch`]'s `REPORT` step (phase 1b) can reuse an already-up NIC
/// instead of bringing up a second one; the caller owns dropping it.
pub fn run_job(
    job: &Job,
    root: &mut Directory,
    slot: &mut crate::reload::EngineSlot<'_>,
) -> (ResultRecord, Option<crate::net::Net>) {
    // Design §2: `RUNID` in the file names the record; `RUN.ID` on the volume
    // is the fallback the controller has used since phase 0.
    let rid = match &job.run_id {
        Some(v) => v.clone(),
        None => run_id(root),
    };
    let mut rec = ResultRecord::new(job.budget_s, rid);
    // Rule A, enforced by the artifact (design §3): no step made under
    // `env=vm` may carry a rate, and every step says so in its own block.
    let rate_valid = rec.env == "iron";

    if job.mode == Mode::Resident {
        crate::boot_log(
            root,
            "JOB: MODE resident is phase 2 — running the steps oneshot",
        );
    }

    // Brought up by the first directive that needs it (currently NETCHECK).
    // Returned to the caller rather than dropped here: phase 1b's REPORT step
    // runs after this function returns (once RESULT.TXT is on the volume) and
    // reuses this same NIC instead of bringing up a second one. Whichever
    // caller ends up holding it must drop it — which shuts the NIC down and
    // releases the exclusive SNP open (see `net::SnpDevice::drop`) — before
    // the machine resets.
    let mut net: Option<crate::net::Net> = None;

    let started = crate::wall_seconds();
    if started.is_none() {
        crate::boot_log(
            root,
            "JOB: firmware GetTime() unavailable — budget cannot be enforced",
        );
    }

    // The watchdog window this job is running under. Every lab step re-arms
    // it as it makes progress (design §8): a long `VERIFY` or a 2048-token
    // `EVAL` window can outlast the interval that was armed when the job
    // started, and a healthy box that is still working must not be reset for
    // taking a while.
    let wd_window = job.budget_s.saturating_add(WATCHDOG_MARGIN_S);

    // Design §2: MECH is moved **last by the dispatcher regardless of file
    // order**. `job.N` numbering follows dispatch order, which is also what
    // §3.1 serialises into `merge_key`, so the key describes what was run.
    let order = dispatch_order(&job.steps);

    for (i, &si) in order.iter().enumerate() {
        let n = i + 1;
        let step = &job.steps[si];

        // Out of budget before this step even starts: record it and stop
        // rather than begin work that cannot finish.
        if let Some(t0) = started
            && let Some(el) = elapsed_since(t0)
            && el >= job.budget_s as f64
        {
            rec.fail("budget");
            crate::boot_log(
                root,
                &format!("JOB: budget spent before step {n} — stopping"),
            );
            break;
        }

        let (kind, prompt, max_tokens) = match step {
            Step::Prompt { text, tokens } => ("prompt", text.clone(), *tokens),
            Step::Bench { tokens } => ("bench", String::from(BENCH_PROMPT), *tokens),
            Step::NetCheck { target } => {
                // Phase 1a. The network is brought up lazily, on the first
                // directive that needs it: a job of pure PROMPTs must not pay
                // for a NIC it never uses, and a boot with no JOB.TXT must
                // never reach this module at all.
                crate::boot_log(root, &format!("JOB: step {n} NETCHECK {target}"));
                let w0 = crate::wall_seconds();
                if net.is_none() {
                    net = crate::net::Net::bring_up(&job.net, root);
                }
                if let Some(nw) = net.as_ref() {
                    rec.mac = nw.mac_string();
                    rec.ip = nw.ip_string();
                    rec.net = nw.how().to_string();
                }
                let (ok, detail) = match net.as_mut() {
                    Some(nw) => netcheck(nw, target, root),
                    None => (false, String::from(crate::net::NetError::NoNic.as_str())),
                };
                let w1 = crate::wall_seconds();
                let wall_ms = match (w0, w1) {
                    (Some(a), Some(b)) if b >= a => ((b - a) * 1000.0) as u64,
                    _ => 0,
                };
                crate::boot_log(
                    root,
                    &if ok {
                        format!("NETCHECK: ok {detail}")
                    } else {
                        format!("NETCHECK: fail {detail}")
                    },
                );
                rec.steps.push(StepResult {
                    kind: "netcheck",
                    prompt: target.clone(),
                    tokens: 0,
                    wall_ms,
                    tps: 0.0,
                    tsc_per_tok: 0,
                    digest: token_digest(&[]),
                    response: detail,
                    ok: Some(ok),
                    rate_valid,
                    nll_q16: None,
                    ntok: None,
                    membw_mibs: None,
                    pass: None,
                    items: None,
                    partial: None,
                    err: None,
                    detail: None,
                    merge_input: None,
                });
                if !ok {
                    rec.fail("netcheck");
                }
                if strict_stop(job, &mut rec, root, n) {
                    break;
                }
                continue;
            }
            Step::Cpuid => {
                crate::boot_log(root, &format!("JOB: step {n} CPUID"));
                let sr = crate::lab::cpuid(rate_valid);
                crate::boot_log(
                    root,
                    &format!("JOB: step {n} cpuid {}", sr.detail.as_deref().unwrap_or("")),
                );
                rec.steps.push(sr);
                if strict_stop(job, &mut rec, root, n) {
                    break;
                }
                continue;
            }
            Step::Verify { name } => {
                crate::boot_log(root, &format!("JOB: step {n} VERIFY {name}"));
                write_progress_record(&rec, root, n);
                let w0 = crate::wall_seconds();
                let mut rearm = move || {
                    arm_watchdog(wd_window);
                };
                let mut sr = crate::lab::verify(root, &*slot, name, rate_valid, &mut rearm);
                sr.wall_ms = wall_ms_between(w0, crate::wall_seconds());
                crate::boot_log(
                    root,
                    &format!(
                        "JOB: step {n} verify pass={:?} items={:?} err={:?} digest={}",
                        sr.pass, sr.items, sr.err, sr.digest
                    ),
                );
                rec.steps.push(sr);
                if strict_stop(job, &mut rec, root, n) {
                    break;
                }
                continue;
            }
            Step::Mech => {
                crate::boot_log(root, &format!("JOB: step {n} MECH"));
                write_progress_record(&rec, root, n);
                let w0 = crate::wall_seconds();
                let Some(engine) = slot.engine_mut() else {
                    rec.fail("no engine");
                    crate::boot_log(root, "JOB: no engine in the slot — MECH skipped");
                    break;
                };
                crate::lab::mech(root, engine);
                let w1 = crate::wall_seconds();
                let mut sr = StepResult::lab("mech", rate_valid);
                sr.wall_ms = wall_ms_between(w0, w1);
                sr.pass = Some(true);
                sr.err = Some(String::from("none"));
                // §3.1: `mech` has an empty `<step-input>`.
                sr.merge_input = Some(String::new());
                rec.steps.push(sr);
                if strict_stop(job, &mut rec, root, n) {
                    break;
                }
                continue;
            }
        };

        say(&format!("[JOB] {n}/{} {kind}: ", job.steps.len()));
        crate::boot_log(
            root,
            &format!("JOB: step {n} {kind} tokens<={max_tokens}, prefill begin"),
        );

        // On-volume heartbeat before the work starts, so a step killed by the
        // firmware watchdog still leaves a record naming the step instead of
        // an empty volume. Overwritten by the real record at the end.
        write_progress_record(&rec, root, n);

        let budget_s = job.budget_s as f64;
        // The engine's deadline check. Copies only, so it borrows neither the
        // engine nor `root`, and it is monotone: once the budget is spent it
        // stays spent. A firmware with no usable clock (`started == None`)
        // yields `false` for ever — the budget cannot be enforced, which was
        // already logged, and the watchdog remains the guard.
        let over_budget = move || match started {
            Some(t0) => matches!(elapsed_since(t0), Some(el) if el >= budget_s),
            None => false,
        };
        let Some(engine) = slot.engine_mut() else {
            // Unreachable in practice: the slot is only ever empty inside
            // `EngineSlot::reload`, which cold-resets rather than returning.
            rec.fail("no engine");
            crate::boot_log(root, &format!("JOB: no engine in the slot at step {n}"));
            break;
        };
        let w0 = crate::wall_seconds();
        // SAFETY: RDTSC is unprivileged, takes no operands and has no side
        // effects; it is architecturally present on every x86_64 CPU this
        // unikernel can boot on, so there is no feature bit to test first.
        // Rule A: the delta it feeds is `tsc_per_tok` — invariant-rate ticks,
        // not cycles, and never a performance figure (see this file's header).
        let c0 = unsafe { core::arch::x86_64::_rdtsc() };
        aegis_core::inference::clear_generation_stop();
        {
            // One BOOTLOG line at the prefill→decode transition. It is what
            // distinguishes "the watchdog killed us mid-prefill" from "mid
            // decode" on a stick that comes home, and it costs one write per
            // step because `announced` latches it.
            let mut announced = false;
            let log_root = &mut *root;
            engine.process_intent_with_deadline(
                &prompt,
                max_tokens,
                |tok| {
                    if announced || tok.starts_with("[SYSTEM]") || tok.contains("[PERFORMANCE]") {
                        return;
                    }
                    announced = true;
                    crate::boot_log(log_root, &format!("JOB: step {n} prefill done, decoding"));
                },
                over_budget,
            );
        }
        // SAFETY: as at `c0` above — unprivileged, no operands, no side
        // effects, architecturally present.
        let c1 = unsafe { core::arch::x86_64::_rdtsc() };
        let w1 = crate::wall_seconds();
        let stopped = aegis_core::inference::generation_stop_requested();
        aegis_core::inference::clear_generation_stop();

        let ids = engine.last_generated_ids.clone();
        let response = engine.tokenizer.decode(&ids);
        let ntok = ids.len();
        let secs = match (w0, w1) {
            (Some(a), Some(b)) if b >= a => b - a,
            _ => 0.0,
        };
        let ticks = c1.saturating_sub(c0);

        say("\r\n");
        rec.steps.push(StepResult {
            kind,
            prompt,
            tokens: ntok,
            wall_ms: (secs * 1000.0) as u64,
            tps: if secs > 0.0 { ntok as f64 / secs } else { 0.0 },
            tsc_per_tok: if ntok > 0 { ticks / ntok as u64 } else { 0 },
            digest: token_digest(&ids),
            response,
            ok: None,
            rate_valid,
            nll_q16: None,
            ntok: None,
            membw_mibs: None,
            pass: None,
            items: None,
            partial: None,
            err: None,
            detail: None,
            merge_input: None,
        });

        // Budget first: a step the budget cut short in prefill also produced
        // no tokens, and "FAIL budget" is the reason a collector can act on.
        // `ResultRecord::fail` keeps the first reason, so the order is the
        // verdict.
        if stopped {
            rec.fail("budget");
            let phase = if ntok == 0 { "prefill" } else { "decode" };
            crate::boot_log(
                root,
                &format!("JOB: step {n} stopped by budget in {phase} after {ntok} tokens"),
            );
            break;
        }
        if ntok == 0 {
            rec.fail("no tokens generated");
        }
        let digest = match rec.steps.last() {
            Some(last) => last.digest.as_str(),
            None => "",
        };
        crate::boot_log(
            root,
            &format!("JOB: step {n} done, {ntok} tokens, digest={digest}"),
        );
        if strict_stop(job, &mut rec, root, n) {
            break;
        }
    }

    if let Some(nw) = net.as_ref() {
        let (tx_dropped, rx_errors) = nw.counters();
        crate::boot_log(
            root,
            &format!("NET: tx_dropped={tx_dropped} rx_errors={rx_errors}"),
        );
    }

    if rec.steps.is_empty() {
        rec.fail("no runnable directives");
    }
    (rec, net)
}

/// Dispatch order for a step list: file order, with every `MECH` moved to the
/// end (design §2), each group stable.
///
/// The box never reorders anything else. `MECH` is singled out because it runs
/// ~24 minutes under TCG and is a hands-off diagnostic — a `JOB.TXT` that put
/// it first would starve the work someone actually asked for, and the same
/// reasoning put v0.1's job hook before the MECH block in `main.rs`.
fn dispatch_order(steps: &[Step]) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::with_capacity(steps.len());
    for (i, st) in steps.iter().enumerate() {
        if !matches!(st, Step::Mech) {
            order.push(i);
        }
    }
    for (i, st) in steps.iter().enumerate() {
        if matches!(st, Step::Mech) {
            order.push(i);
        }
    }
    order
}

/// `STRICT on` (design §2): if the step just pushed failed, record
/// `verdict=FAIL <step-kind>` and tell the caller to stop.
///
/// Returns `false` — carry on — when `STRICT` is off, when there is no step to
/// judge, or when the step passed. `ResultRecord::fail` keeps the first
/// reason, so a budget overrun that also produced a failing step is still
/// reported as the overrun.
fn strict_stop(job: &Job, rec: &mut ResultRecord, root: &mut Directory, n: usize) -> bool {
    if !job.strict {
        return false;
    }
    let Some(last) = rec.steps.last() else {
        return false;
    };
    if !last.failed() {
        return false;
    }
    let kind = last.kind;
    rec.fail(kind);
    crate::boot_log(
        root,
        &format!("JOB: STRICT on — step {n} ({kind}) failed, stopping the job"),
    );
    true
}

/// Milliseconds between two `wall_seconds()` readings, or `0` when the
/// firmware has no usable clock. Never negative and never a guess.
pub(crate) fn wall_ms_between(w0: Option<f64>, w1: Option<f64>) -> u64 {
    match (w0, w1) {
        (Some(a), Some(b)) if b >= a => ((b - a) * 1000.0) as u64,
        _ => 0,
    }
}

/// One `NETCHECK host:port`: connect, send `HELLO <mac>\n`, wait for the peer
/// to close (spec §6 / phase 1a gate).
///
/// Returns `(ok, detail)`, where `detail` is what goes into `job.N.response`
/// and the `NETCHECK:` line in `BOOTLOG.TXT`. `ok` is decided by the connect
/// and the write, because those are what the gate asserts on the other end —
/// a listener that receives the line and then holds the socket open is not a
/// failure of this box.
fn netcheck(
    net: &mut crate::net::Net,
    target: &str,
    root: &mut Directory,
) -> (bool, alloc::string::String) {
    use crate::net::Until;

    let Some((ip, port)) = crate::net::parse_host_port(target) else {
        return (false, format!("bad target {target}"));
    };
    if net.ip().is_none() {
        return (false, alloc::string::String::from("no ip"));
    }

    let handle = match net.tcp_connect(ip, port, NETCHECK_TIMEOUT_MS) {
        Ok(h) => h,
        Err(e) => return (false, format!("connect {}", e.as_str())),
    };

    let hello = format!("HELLO {}\n", net.mac_string());
    if let Err(e) = net.tcp_send_all(&handle, hello.as_bytes(), NETCHECK_TIMEOUT_MS) {
        net.tcp_close(handle, NETCHECK_CLOSE_MS);
        return (false, format!("send {}", e.as_str()));
    }

    // Wait for the peer to close, or give up after the same bound. Either way
    // the line is already on the wire; this only decides what `detail` says.
    // `Ok` here means the peer actually sent FIN — a connection that ended
    // because this stack's own idle timeout fired reports `closed`, not `peer
    // closed`, because those are different facts about the other end.
    let (echo, waited) = net.tcp_recv_until(&handle, Until::Close, NETCHECK_TIMEOUT_MS);
    net.tcp_close(handle, NETCHECK_CLOSE_MS);

    let how = match waited {
        Ok(()) => "peer closed",
        // The wait ran out with the connection still up. That is not a fault:
        // a listener is free to hold the socket open, and the HELLO line was
        // already delivered. Saying so is better than reporting the bare
        // "timeout", which reads as a failure this was not.
        Err(crate::net::NetError::Timeout) => "peer held the connection open",
        Err(e) => e.as_str(),
    };
    crate::boot_log(
        root,
        &format!(
            "NETCHECK: sent {} bytes to {target}, {how}, {} bytes back",
            hello.len(),
            echo.len()
        ),
    );
    (true, format!("sent {} bytes, {how}", hello.len()))
}

/// The single entry point `main.rs` calls (spec §5).
///
/// Arms the watchdog at `BUDGET + 60`, runs the job, writes `RESULT.TXT`,
/// then performs `AFTER`. Does not return for `AFTER reset`; returns for
/// `AFTER halt`, and the caller carries on with its normal boot path.
///
/// `MODE resident` (spec §4, phase 2) is the third case: the network comes up,
/// [`crate::server::run`] takes the box over and never returns. It returns
/// here only when the box could not be made reachable — no NIC, or no address
/// — because a resident worker nobody can dial is not a resident worker, and
/// falling back to the normal boot path leaves someone at the console a way in.
pub fn dispatch(job: &Job, root: &mut Directory, slot: &mut crate::reload::EngineSlot<'_>) {
    if job.mode == Mode::Resident {
        dispatch_resident(job, root, slot);
        return;
    }
    let wd = job.budget_s.saturating_add(WATCHDOG_MARGIN_S);
    if arm_watchdog(wd) {
        crate::boot_log(root, &format!("JOB: watchdog armed at {wd}s"));
    } else {
        crate::boot_log(
            root,
            "JOB: firmware refused the watchdog — running unguarded",
        );
    }

    say("\r\n[AEFINITY OS] JOB.TXT found — running job\r\n");
    if slot.engine_mut().is_none() {
        // Unreachable in practice: the slot is handed over with the engine
        // `main.rs` built, and the only code that empties it is
        // `EngineSlot::reload`, which either refills it or cold-resets the box
        // (design §4.1). Saying so beats an `expect` on a firmware-adjacent
        // path where a panic is `panic = "abort"` — a dead box in a rack.
        crate::boot_log(root, "JOB: no engine in the slot — cannot run");
        return;
    }
    let (mut rec, mut net) = run_job(job, root, slot);
    rec.fleet = Some(fleet_info(slot, root, None, false, job));
    let body = rec.render();

    if write_result_txt(root, &body) {
        // The finished record is on the volume, so the in-progress marker is
        // no longer true. Removing it is what makes its presence meaningful.
        // The delete is only *issued* here; `settle_volume` confirms it after
        // the same stall the `RESULT.TXT` read-back gets. `delete()` returns
        // when the firmware has taken the request, not when the medium has,
        // so a probe fired immediately after it reports the hand-off and not
        // the volume. The claim this line used to carry was never established,
        // whether or not it happened to be true.
        delete_named(root, WIP_NAME);
        crate::boot_log(
            root,
            &format!("JOB: RESULT.TXT written, verdict={}", rec.verdict),
        );
    } else {
        crate::boot_log(root, "JOB: RESULT.TXT could not be written");
    }

    // Echo it: on a box with a serial console attached this is the result,
    // and phase 1b posts the identical bytes below.
    say("\r\n---- RESULT.TXT ----\r\n");
    for line in body.lines() {
        say(line);
        say("\r\n");
    }
    say("---- END ----\r\n");

    // ---- AEFINITY OS phase 1b: REPORT --------------------------------------
    // Spec §5: POST the exact bytes already on the volume to `REPORT <url>`,
    // after RESULT.TXT is written and before AFTER. Never rewrites
    // RESULT.TXT — spec §3/§6 make the on-disk record predate the POST, so
    // `report=` in it stays whatever `run_job` left there (`none`, unless a
    // future step sets it). A failed report is logged and nothing else: a
    // collector that is unreachable must never turn a completed job into a
    // box that will not reset.
    if let Some(url) = &job.report {
        if net.is_none() {
            net = crate::net::Net::bring_up(&job.net, root);
        }
        match net.as_mut() {
            Some(nw) => match crate::net::http::post(
                nw,
                url,
                body.as_bytes(),
                job.token.as_deref(),
                REPORT_TIMEOUT_MS,
            ) {
                Ok(status) => crate::boot_log(root, &format!("REPORT: ok {status}")),
                Err(e) => crate::boot_log(root, &format!("REPORT: fail {}", e.as_str())),
            },
            None => crate::boot_log(
                root,
                &format!("REPORT: fail {}", crate::net::NetError::NoNic.as_str()),
            ),
        }
    }
    // Whatever NIC this job used — for NETCHECK, for REPORT, or both — comes
    // down here, before the volume settle and the reset: `SnpDevice::drop`
    // releases the exclusive SNP open the firmware needs back.
    drop(net);

    settle_volume(root);
    after(job.after, root);
}

/// `MODE resident` (spec §4): bring the network up and hand the box to the
/// listener.
///
/// The watchdog is explicitly *disarmed* first. Spec §4: "Watchdog is re-armed
/// to `BUDGET+60` at `JOB` and to 0 (disabled) while idle — a resident box
/// idles indefinitely; a hung *job* is reset by firmware." Arming it here, as
/// the oneshot path does, would reset a perfectly healthy box that simply had
/// no work yet. `server::run` is what arms and disarms it around each
/// socket-delivered job.
///
/// The `NET` directive is applied as written. Spec §2 says `NET dhcp` "falls
/// back to static if given", and a `JOB.TXT` carries one `NET` line in one of
/// two forms — so a job that asked for DHCP has given no static address to
/// fall back to, and this says so in `BOOTLOG.TXT` rather than inventing one.
/// The fallback becomes reachable when the job format carries both.
fn dispatch_resident(job: &Job, root: &mut Directory, slot: &mut crate::reload::EngineSlot<'_>) {
    arm_watchdog(0);
    crate::boot_log(
        root,
        "RESIDENT: MODE resident — watchdog disarmed, bringing up the network",
    );
    say("\r\n[AEFINITY OS] MODE resident — bringing up the network\r\n");

    let Some(net) = crate::net::Net::bring_up(&job.net, root) else {
        crate::boot_log(
            root,
            "RESIDENT: no usable NIC — cannot listen, falling through to the boot path",
        );
        say("[AEFINITY OS] resident mode: no NIC, falling through\r\n");
        return;
    };
    if net.ip().is_none() {
        crate::boot_log(
            root,
            &format!(
                "RESIDENT: NIC {} came up with no address ({}) — cannot listen, \
                 falling through to the boot path",
                net.mac_string(),
                net.how()
            ),
        );
        say("[AEFINITY OS] resident mode: no address, falling through\r\n");
        // Dropping the Net shuts the NIC down and releases the exclusive SNP
        // open, so the boot path that follows is the one it would have had.
        drop(net);
        return;
    }

    crate::boot_log(
        root,
        &format!("RESIDENT: listening {}:{}", net.ip_string(), job.listen),
    );
    say(&format!(
        "[AEFINITY OS] RESIDENT: listening {}:{}\r\n",
        net.ip_string(),
        job.listen
    ));
    crate::server::run(net, root, slot, job)
}

/// Assemble design §3's phase-4 additions for one record.
///
/// `resident` is `(uptime_s, served)` in resident mode and `None` for a
/// oneshot boot, which has no listener and so no honest uptime to quote.
/// `files` is a live `LS` count — the cheapest way for a scheduler to see that
/// a box it filled still holds what it was given.
pub fn fleet_info(
    slot: &crate::reload::EngineSlot<'_>,
    root: &mut Directory,
    resident: Option<(u64, u64)>,
    replay: bool,
    job: &Job,
) -> FleetInfo {
    let d = slot.digests();
    let files = crate::files::ls(root).map(|(v, _)| v.len()).unwrap_or(0);
    FleetInfo {
        artifacts: d.artifacts_line(),
        model_sha: d.model.clone(),
        full: (d.model.clone(), d.embed.clone(), d.vocab.clone()),
        reloads: slot.reloads(),
        resident,
        files,
        replay,
        seed: job.seed,
        tag: job.tag.clone(),
        shard: job.shard,
    }
}

/// Read the record back off the volume before resetting the machine, confirm
/// the in-progress marker really went with it, and log what is actually
/// there.
///
/// `flush()` returns when the firmware has handed the write on, not when the
/// medium has taken it, and the next thing this code does is reset the
/// machine. The load path already stalls a second after each big read for the
/// same reason on USB. The read-back is the point: `write_result_txt`
/// returning true says the firmware accepted the write, and only reopening
/// the file says the record is on the volume — which is the difference
/// between a headless box that reported a job and one that only thinks it
/// did. The BOOTLOG line is what tells you which, on a stick that comes home.
///
/// Honesty note: an earlier reading of this failure blamed a lost write under
/// QEMU's `fat:rw:`. It was not — the record was on the volume and on the
/// host both times; the harness was looking for `RESULT.TXT` on a
/// case-sensitive filesystem where vvfat had written `result.txt`. The stall
/// and read-back stay because they are right on real hardware, not because
/// they fixed that.
fn settle_volume(root: &mut Directory) {
    uefi::boot::stall(core::time::Duration::from_secs(3));
    match read_small_file(root, "RESULT.TXT", JOB_MAX_BYTES) {
        Some(bytes) => crate::boot_log(
            root,
            &format!(
                "JOB: RESULT.TXT read-back OK, {} bytes on volume",
                bytes.len()
            ),
        ),
        None => crate::boot_log(
            root,
            "JOB: RESULT.TXT read-back FAILED — record not on volume",
        ),
    }
    confirm_wip_cleared(root);
    uefi::boot::stall(core::time::Duration::from_secs(1));
}

/// Confirm — after the settle stall — that [`WIP_NAME`] really is off the
/// volume, and say so in `BOOTLOG.TXT`.
///
/// Spec §3 defines the marker's presence to mean "this box did not finish",
/// so a `cleared=true` that was never checked against the medium is worse
/// than no line at all: it is the one claim on the stick that would be read
/// as evidence. The delete was issued in `dispatch`, before the echo and the
/// three-second stall, which is the same hand-off-versus-medium gap
/// [`settle_volume`] exists for. A marker still there after all of that gets
/// one more delete, one more stall and one more probe before it is reported
/// uncleared — `RESULT.TXT` is authoritative either way (§3), so this line is
/// a report, never a failure.
///
/// Both probes are printed because they answer different questions
/// ([`presence`], [`dir_has`]) and because a host mirror can disagree with
/// both: QEMU's `fat:rw:` block backend commits guest *writes* through to the
/// host directory but need not commit an unlink, so under the xtask gates the
/// staged ESP can still show a `RESULT.WIP` that the guest, the firmware and
/// the FAT directory all agree is gone. What the guest can honestly report is
/// what the volume it is holding says, and that is what this logs.
fn confirm_wip_cleared(root: &mut Directory) {
    let mut open_p = presence(root, WIP_NAME);
    let mut dir_p = dir_has(root, WIP_NAME);
    let mut retried = false;
    if open_p != Presence::Absent || dir_p != Presence::Absent {
        retried = true;
        delete_named(root, WIP_NAME);
        uefi::boot::stall(core::time::Duration::from_secs(1));
        open_p = presence(root, WIP_NAME);
        dir_p = dir_has(root, WIP_NAME);
    }
    let cleared = open_p == Presence::Absent && dir_p == Presence::Absent;
    crate::boot_log(
        root,
        &format!(
            "JOB: {WIP_NAME} cleared={cleared} (open={} dir={} retried={retried})",
            open_p.as_str(),
            dir_p.as_str()
        ),
    );
}
