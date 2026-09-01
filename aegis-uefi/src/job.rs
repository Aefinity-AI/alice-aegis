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

use aegis_core::inference::TernaryInferenceEngine;
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
/// Job bodies are capped (spec §4); the same cap guards the on-disk file.
const JOB_MAX_BYTES: usize = 64 * 1024;
/// A single `PROMPT` line is capped (spec §4).
const PROMPT_MAX_BYTES: usize = 4 * 1024;
/// Watchdog code passed to `set_watchdog_timer`; anything >= 0x1_0000 is the
/// UEFI-spec range reserved for the caller, so it cannot collide with a
/// firmware-defined code in a post-mortem.
const WATCHDOG_CODE: u64 = 0x1_0000;

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
}

/// A parsed `JOB.TXT`.
#[derive(Clone, Debug)]
pub struct Job {
    pub budget_s: u64,
    pub mode: Mode,
    pub net: NetCfg,
    /// `REPORT <url>` — phase 1b posts `RESULT.TXT` here.
    pub report: Option<String>,
    pub listen: u16,
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
            listen: DEFAULT_LISTEN,
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
    pub budget_s: u64,
    pub steps: Vec<StepResult>,
    /// `ok` | `fail <reason>` | `none` (phase 1b fills the first two).
    pub report: String,
    /// `OK` | `FAIL <reason>`.
    pub verdict: String,
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
            budget_s,
            steps: Vec::new(),
            report: String::from("none"),
            verdict: String::from("OK"),
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
        s.push_str(&format!("budget_s={}\n", self.budget_s));
        s.push_str(&format!("jobs={}\n", self.steps.len()));
        for (i, j) in self.steps.iter().enumerate() {
            let n = i + 1;
            s.push_str(&format!("job.{n}.kind={}\n", j.kind));
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
        }
        s.push_str(&format!("report={}\n", self.report));
        s.push_str(&format!("verdict={}\n", self.verdict));
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

/// Write `RESULT.TXT` to the boot volume root: one create, one write, one
/// flush, one close, overwriting rather than appending (spec §3).
///
/// The stale file is deleted first. Opening `CreateReadWrite` over a longer
/// previous record and writing a shorter one would leave the previous run's
/// tail on disk, and a collector reading that would see two records fused.
pub fn write_result_txt(root: &mut Directory, body: &str) -> bool {
    use uefi::proto::media::file::{File, FileAttribute, FileMode, FileType};
    let mut namebuf = [0u16; 32];
    let cstr = match uefi::CStr16::from_str_with_buf("RESULT.TXT", &mut namebuf) {
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

/// Run every directive in file order and build the record (spec §3).
///
/// The budget is wall-clock, read from UEFI `GetTime()` at entry. The token
/// callback checks it on every emitted token and asks the engine to stop
/// generating once it is spent; the record is then marked `FAIL budget`. A
/// job stopped this way still writes a complete `RESULT.TXT` — a truncated
/// answer plus an honest verdict is worth more to the collector than silence.
pub fn run_job(
    job: &Job,
    root: &mut Directory,
    engine: &mut TernaryInferenceEngine,
) -> ResultRecord {
    let rid = run_id(root);
    let mut rec = ResultRecord::new(job.budget_s, rid);

    if job.mode == Mode::Resident {
        crate::boot_log(
            root,
            "JOB: MODE resident is phase 2 — running the steps oneshot",
        );
    }

    let started = crate::wall_seconds();
    if started.is_none() {
        crate::boot_log(
            root,
            "JOB: firmware GetTime() unavailable — budget cannot be enforced",
        );
    }

    for (i, step) in job.steps.iter().enumerate() {
        let n = i + 1;

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
                // Phase 1a owns NETCHECK. Recording it as an unrun step with
                // an explicit verdict is honest; pretending it passed is not.
                crate::boot_log(
                    root,
                    &format!("JOB: step {n} NETCHECK {target} — phase 1a, not run"),
                );
                rec.steps.push(StepResult {
                    kind: "netcheck",
                    prompt: target.clone(),
                    tokens: 0,
                    wall_ms: 0,
                    tps: 0.0,
                    tsc_per_tok: 0,
                    digest: token_digest(&[]),
                    response: String::from("not implemented in phase 0"),
                });
                rec.fail("netcheck unimplemented");
                continue;
            }
        };

        say(&format!("[JOB] {n}/{} {kind}: ", job.steps.len()));
        crate::boot_log(root, &format!("JOB: step {n} {kind} tokens<={max_tokens}"));

        let budget_s = job.budget_s as f64;
        let w0 = crate::wall_seconds();
        let c0 = unsafe { core::arch::x86_64::_rdtsc() };
        aegis_core::inference::clear_generation_stop();
        engine.process_intent(&prompt, max_tokens, |tok| {
            if tok.starts_with("[SYSTEM]") || tok.contains("[PERFORMANCE]") {
                return;
            }
            if let Some(t0) = started
                && let Some(el) = elapsed_since(t0)
                && el >= budget_s
            {
                aegis_core::inference::request_generation_stop();
            }
        });
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
        });

        if ntok == 0 {
            rec.fail("no tokens generated");
        }
        if stopped {
            rec.fail("budget");
            crate::boot_log(
                root,
                &format!("JOB: step {n} stopped by budget after {ntok} tokens"),
            );
            break;
        }
        let digest = match rec.steps.last() {
            Some(last) => last.digest.as_str(),
            None => "",
        };
        crate::boot_log(
            root,
            &format!("JOB: step {n} done, {ntok} tokens, digest={digest}"),
        );
    }

    if rec.steps.is_empty() {
        rec.fail("no runnable directives");
    }
    rec
}

/// The single entry point `main.rs` calls (spec §5).
///
/// Arms the watchdog at `BUDGET + 60`, runs the job, writes `RESULT.TXT`,
/// then performs `AFTER`. Does not return for `AFTER reset`; returns for
/// `AFTER halt`, and the caller carries on with its normal boot path.
pub fn dispatch(job: &Job, root: &mut Directory, engine: &mut TernaryInferenceEngine) {
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
    let rec = run_job(job, root, engine);
    let body = rec.render();

    if write_result_txt(root, &body) {
        crate::boot_log(
            root,
            &format!("JOB: RESULT.TXT written, verdict={}", rec.verdict),
        );
    } else {
        crate::boot_log(root, "JOB: RESULT.TXT could not be written");
    }

    // Echo it: on a box with a serial console attached this is the result,
    // and phase 1b will POST the identical bytes.
    say("\r\n---- RESULT.TXT ----\r\n");
    for line in body.lines() {
        say(line);
        say("\r\n");
    }
    say("---- END ----\r\n");

    settle_volume(root);
    after(job.after, root);
}

/// Read the record back off the volume before resetting the machine, and log
/// what is actually there.
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
    let _ = uefi::boot::stall(core::time::Duration::from_secs(3));
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
    let _ = uefi::boot::stall(core::time::Duration::from_secs(1));
}
