//! SAFE-1b prototype: a "no receipt, no action" tool gateway.
//!
//! Implements the design in
//! `state/reports/2026-09-13-SAFE1-RECEIPT-GATEWAY-DESIGN.md` (claudius-maximus
//! repo) as revised by the critique in
//! `state/reports/2026-09-13-SAFE1-DESIGN-CRITIC.md`, which found three gaps
//! in the original design and told safe-1b not to prototype the
//! pre-execution framing until they were fixed. This crate is scoped
//! honestly to what the critique said IS buildable from what exists today:
//! `agent_trace verify`'s replay proves a receipt's own step k+1 (containing
//! that step's tool call AND its already-computed result) is internally
//! consistent and chain-bound; nothing here claims to gate execution of a
//! not-yet-computed step (critique item #2 — no "receipt-prefix" mechanism
//! exists in `agent_trace`). What this gateway actually gates is: "does the
//! caller possess a verified receipt whose last completed step is BYTE-FOR-
//! BYTE this exact tool call" before treating that call's outcome as
//! authorized to propagate (e.g. to a downstream real-tool dispatch, or to
//! release the next round's prompt). That is an audit/detection gate one
//! step behind execution, not a strict pre-execution gate — see the
//! project report for the honest framing.
//!
//! Three properties, each mandatory (not optional flags), per the safe-1b
//! brief:
//!
//! (a) VERIFY -> EXECUTE BINDING (closes critique #1): the gateway will not
//!     ALLOW unless the caller's claimed action bytes hash to exactly the
//!     receipt's own last-step tool-call input bytes. A verified receipt for
//!     ANY other action is a DENY, even though `agent_trace verify` itself
//!     would say PASS on that receipt in isolation.
//!
//! (b) FRESHNESS: every decision request carries a session id and a
//!     monotonic per-session counter. The gateway keeps every
//!     (session, counter, receipt-action-hash) triple it has ever ALLOWed;
//!     a repeat of any one of those three values against a previously
//!     logged entry is DENY. `agent_trace verify` has no such concept
//!     (per the original design memo's threat table) — this is a gateway-
//!     side addition.
//!
//! (c) LOG ANCHOR + ALLOWLIST INTEGRITY: every decision is appended to a
//!     hash-chained local log (`prev_digest`, decision, receipt trace-chain,
//!     artifact triple, session, counter -> this entry's digest). Every
//!     `anchor_every` decisions, the current chain head is appended to a
//!     separate anchor file — standing in for "publish externally";
//!     `publish externally` is stubbed as a local file in this prototype,
//!     real deployment would commit that line to a second repo/box the
//!     gateway process cannot itself rewrite. The tool allowlist is a flat
//!     file of `model embed vocab` hex triples, one per line, with a final
//!     `sig <hex>` line = HMAC-SHA256 (keyed by a local key file, built from
//!     the crate's own `sha256`/`Sha256`, since this crate is offline-only
//!     and pulls no crates.io HMAC dependency) over every preceding line's
//!     exact bytes. Loading an allowlist whose signature does not match is
//!     a hard reject (empty allowlist, everything DENIES) rather than a
//!     silent fall-through.

use aegis_core::witness::sha256;
use gateway::{capability, hex, hmac_sha256, unhex};
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// HMAC-SHA256, hex/unhex: moved to lib.rs (`demo/agent-trace/gateway/src/lib.rs`)
// so that `src/bin/tool_shim_*.rs` can share the exact same implementation
// the gateway daemon uses, without a crates.io HMAC dependency (this repo
// is offline-only, see aegis-core/Cargo.toml).

// ---------------------------------------------------------------------
// Receipt parsing (deliberately minimal: only what the gateway needs, not
// a general AEGIS-TRACE parser — full parsing/validation is
// agent_trace's own job and is invoked separately, see `verify_receipt`).
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ArtifactTriple {
    model: String,
    embed: String,
    vocab: String,
}

#[derive(Debug, Clone)]
struct ParsedReceipt {
    triple: ArtifactTriple,
    /// Hex-decoded bytes of the LAST step's `in=` field: the tool-call
    /// input the receipt's own chain says actually happened at the final
    /// step. This is what rule (a) binds the proposed action against.
    last_step_input: Vec<u8>,
    trace_chain: String,
}

#[derive(Debug)]
enum ParseError {
    Missing(&'static str),
    NoSteps,
    BadHex(&'static str),
}

fn parse_receipt(text: &str) -> Result<ParsedReceipt, ParseError> {
    let mut model = None;
    let mut embed = None;
    let mut vocab = None;
    let mut trace_chain = None;
    let mut last_step_input: Option<Vec<u8>> = None;

    for line in text.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("model ") {
            model = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("embed ") {
            embed = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("vocab ") {
            vocab = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("trace-chain ") {
            trace_chain = Some(rest.trim().to_string());
        } else if line.starts_with("step ") {
            // e.g. "step 0: toks=... tool=calc in=<hex> out=<hex> ..."
            for field in line.split_whitespace() {
                if let Some(hexval) = field.strip_prefix("in=") {
                    let bytes = unhex(hexval).ok_or(ParseError::BadHex("in="))?;
                    last_step_input = Some(bytes); // overwritten each step -> ends up LAST step's value
                }
            }
        }
    }

    Ok(ParsedReceipt {
        triple: ArtifactTriple {
            model: model.ok_or(ParseError::Missing("model"))?,
            embed: embed.ok_or(ParseError::Missing("embed"))?,
            vocab: vocab.ok_or(ParseError::Missing("vocab"))?,
        },
        last_step_input: last_step_input.ok_or(ParseError::NoSteps)?,
        trace_chain: trace_chain.ok_or(ParseError::Missing("trace-chain"))?,
    })
}

// ---------------------------------------------------------------------
// Allowlist: signed flat file of approved artifact triples.
// ---------------------------------------------------------------------

struct Allowlist {
    entries: HashSet<ArtifactTriple>,
}

#[derive(Debug)]
enum AllowlistError {
    Io(String),
    NoSigLine,
    BadSignature,
}

/// Build the signed allowlist file's canonical bytes (every line except the
/// trailing `sig <hex>` line, each terminated by `\n`, in file order). Both
/// `sign_allowlist` (test/tooling helper) and `load_allowlist` (the gateway
/// itself) must compute this identically or a correctly-signed file would
/// fail to load.
fn allowlist_signable_bytes(body_lines: &[&str]) -> Vec<u8> {
    let mut buf = Vec::new();
    for l in body_lines {
        buf.extend_from_slice(l.as_bytes());
        buf.push(b'\n');
    }
    buf
}

fn sign_allowlist(body_lines: &[&str], key: &[u8]) -> String {
    let digest = hmac_sha256(key, &allowlist_signable_bytes(body_lines));
    hex(&digest)
}

fn load_allowlist(path: &Path, key: &[u8]) -> Result<Allowlist, AllowlistError> {
    let text = fs::read_to_string(path).map_err(|e| AllowlistError::Io(e.to_string()))?;
    let lines: Vec<&str> = text.lines().collect();
    let (body, sig_line) = match lines.split_last() {
        Some((last, body)) => (body, *last),
        None => return Err(AllowlistError::NoSigLine),
    };
    let claimed_sig = sig_line
        .strip_prefix("sig ")
        .ok_or(AllowlistError::NoSigLine)?
        .trim();
    let expect_sig = sign_allowlist(body, key);
    // Constant-time-ish compare is not the point of a prototype; correctness
    // of the check is.
    if claimed_sig != expect_sig {
        return Err(AllowlistError::BadSignature);
    }
    let mut entries = HashSet::new();
    for line in body {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            continue;
        }
        entries.insert(ArtifactTriple {
            model: parts[0].to_string(),
            embed: parts[1].to_string(),
            vocab: parts[2].to_string(),
        });
    }
    Ok(Allowlist { entries })
}

// ---------------------------------------------------------------------
// Freshness store: every (session, counter, action-hash) the gateway has
// ever ALLOWed. In-memory here (`GatewayState`); a real deployment would
// persist this in the same place as the decision log so a gateway restart
// cannot forget it saw a nonce before.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FreshnessKey {
    session: String,
    counter: u64,
    action_hash: [u8; 32],
}

// ---------------------------------------------------------------------
// Decision log: hash-chained, with a periodic anchor.
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum Decision {
    Allow,
    Deny(String), // reason
}

struct LogEntry {
    digest: [u8; 32],
}

pub struct Gateway {
    allowlist: Allowlist,
    seen: HashSet<FreshnessKey>,
    log: Vec<LogEntry>,
    anchor_every: usize,
    anchor_path: Option<PathBuf>,
    agent_trace_bin: PathBuf,
    model_path: PathBuf,
    embed_path: PathBuf,
    vocab_path: PathBuf,
    table_path: Option<PathBuf>,
    strict: bool,
    /// SAFE-5c box2 follow-up: a hard ceiling on how long one `agent_trace
    /// verify` subprocess may run before the gateway kills it and returns
    /// DENY. Without this, a verify that runs long (e.g. a full BitNet-2B
    /// receipt on weak enforcement-host hardware, observed to take 6-7+
    /// minutes single-threaded here) blocks `handle_serve_conn` for that
    /// entire duration -- and because the accept loop is single-threaded
    /// (`for conn in listener.incoming()`), every subsequent connection
    /// queues in the OS backlog and gets no response at all until the slow
    /// one finishes. `None` preserves the old no-timeout behavior.
    verify_timeout: Option<Duration>,
}

pub struct GatewayConfig {
    pub allowlist_path: PathBuf,
    pub allowlist_key: Vec<u8>,
    pub anchor_every: usize,
    pub anchor_path: Option<PathBuf>,
    pub agent_trace_bin: PathBuf,
    pub model_path: PathBuf,
    pub embed_path: PathBuf,
    pub vocab_path: PathBuf,
    pub table_path: Option<PathBuf>,
    /// Strict grounding policy (off by default): DENY a receipt that
    /// otherwise verifies PASS but carries any `WARNING step` (tool-call
    /// argument not found verbatim in the context the agent actually saw).
    pub strict: bool,
    pub verify_timeout: Option<Duration>,
}

#[derive(Debug)]
pub enum GatewayInitError {
    Allowlist(AllowlistError),
}

/// SAFE-7: the actual `agent_trace verify` subprocess spawn/poll, factored
/// out of `Gateway::run_verify_raw` into a free function taking explicit
/// config (rather than `&self`) so the async `serve` worker pool can call
/// it directly, off the accept thread, WITHOUT holding the `Gateway`
/// mutex for the run's full (potentially minutes-long) duration. Behavior
/// is byte-for-byte what `Gateway::run_verify_raw` did before this split;
/// `Gateway::run_verify_raw` now just forwards its own fields here.
fn run_verify_subprocess_raw(
    agent_trace_bin: &Path,
    model_path: &Path,
    embed_path: &Path,
    vocab_path: &Path,
    table_path: Option<&Path>,
    receipt_path: &Path,
    verify_timeout: Option<Duration>,
) -> Result<String, String> {
    use std::process::Stdio;

    let mut cmd = Command::new(agent_trace_bin);
    cmd.arg("verify")
        .arg(model_path)
        .arg(embed_path)
        .arg(vocab_path)
        .arg(receipt_path);
    if let Some(t) = table_path {
        cmd.arg("--table").arg(t);
    }

    let Some(timeout) = verify_timeout else {
        // No ceiling configured: old behavior, block until exit.
        let out = cmd
            .output()
            .map_err(|e| format!("spawn agent_trace: {e}"))?;
        return Ok(String::from_utf8_lossy(&out.stdout).into_owned());
    };

    // Spawn + poll rather than `.output()` (which blocks unconditionally):
    // this is what lets a stuck/slow verify be killed instead of wedging
    // the caller (`handle_serve_conn` in the pre-SAFE-7 sync daemon, or a
    // SAFE-7 background verify worker today) for however long the child
    // feels like running.
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| format!("spawn agent_trace: {e}"))?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                let mut out = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    use std::io::Read;
                    let _ = stdout.read_to_string(&mut out);
                }
                return Ok(out);
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "agent_trace verify exceeded --request-timeout ({timeout:?}), killed"
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(format!("wait agent_trace: {e}")),
        }
    }
}

/// See `Gateway::run_verify` — factored to a free function so the SAFE-7
/// background worker can apply the same lenient policy without a `Gateway`
/// reference.
fn verify_pass_lenient(stdout: &str) -> bool {
    stdout.contains("VERIFY PASS")
}

/// See `Gateway::run_verify_strict` — factored to a free function for the
/// same reason as `verify_pass_lenient`.
fn verify_pass_strict(stdout: &str) -> bool {
    stdout.contains("VERIFY PASS") && !stdout.contains("WARNING step")
}

impl Gateway {
    pub fn new(cfg: GatewayConfig) -> Result<Self, GatewayInitError> {
        let allowlist = load_allowlist(&cfg.allowlist_path, &cfg.allowlist_key)
            .map_err(GatewayInitError::Allowlist)?;
        Ok(Gateway {
            allowlist,
            seen: HashSet::new(),
            log: Vec::new(),
            anchor_every: cfg.anchor_every.max(1),
            anchor_path: cfg.anchor_path,
            agent_trace_bin: cfg.agent_trace_bin,
            model_path: cfg.model_path,
            embed_path: cfg.embed_path,
            vocab_path: cfg.vocab_path,
            table_path: cfg.table_path,
            strict: cfg.strict,
            verify_timeout: cfg.verify_timeout,
        })
    }

    fn log_head(&self) -> [u8; 32] {
        self.log.last().map(|e| e.digest).unwrap_or([0u8; 32])
    }

    /// Number of decisions this Gateway instance has made so far. Used by
    /// `serve` as the capability token's `decision_index` (SAFE-5c):
    /// monotonic, gateway-assigned, never caller-supplied, so a client
    /// cannot mint its own index and cannot roll it back.
    pub fn decision_count(&self) -> u64 {
        self.log.len() as u64
    }

    /// Append one decision to the hash-chained log; publish an anchor line
    /// every `anchor_every` entries. Returns this entry's digest (also its
    /// new chain head).
    fn append_log(
        &mut self,
        decision: &Decision,
        trace_chain: &str,
        triple: &ArtifactTriple,
        session: &str,
        counter: u64,
    ) -> [u8; 32] {
        let decision_tag = match decision {
            Decision::Allow => "ALLOW".to_string(),
            Decision::Deny(reason) => format!("DENY:{reason}"),
        };
        self.append_log_tagged(&decision_tag, trace_chain, triple, session, counter)
    }

    /// SAFE-7: log a PENDING marker at the moment a request is accepted for
    /// background verification, keyed by its ticket. The eventual ALLOW/DENY
    /// (once the background worker finishes) is a SECOND, separate call to
    /// `append_log` — both entries live in the same hash chain
    /// (`prev_digest` links them in order), so the anchored log is still one
    /// append-only sequence, it just now shows the PENDING->decision
    /// transition for requests that went through the async path instead of
    /// a single entry. (Design choice, documented per the SAFE-7 brief: a
    /// synchronous fast-path DENY — allowlist/binding/freshness — still
    /// gets exactly one log entry, same as before SAFE-7, since it never
    /// reaches the background worker at all.)
    fn append_log_pending(
        &mut self,
        ticket: &str,
        trace_chain: &str,
        triple: &ArtifactTriple,
        session: &str,
        counter: u64,
    ) -> [u8; 32] {
        let tag = format!("PENDING:ticket={ticket}");
        self.append_log_tagged(&tag, trace_chain, triple, session, counter)
    }

    /// Shared hash-chaining body for `append_log` and `append_log_pending`
    /// (SAFE-7 split; behavior for the `append_log` caller is byte-for-byte
    /// what it was before this split — same digest computed the same way).
    fn append_log_tagged(
        &mut self,
        decision_tag: &str,
        trace_chain: &str,
        triple: &ArtifactTriple,
        session: &str,
        counter: u64,
    ) -> [u8; 32] {
        let prev = self.log_head();
        let mut m = Vec::new();
        m.extend_from_slice(&prev);
        m.extend_from_slice(decision_tag.as_bytes());
        m.push(0);
        m.extend_from_slice(trace_chain.as_bytes());
        m.push(0);
        m.extend_from_slice(triple.model.as_bytes());
        m.push(b':');
        m.extend_from_slice(triple.embed.as_bytes());
        m.push(b':');
        m.extend_from_slice(triple.vocab.as_bytes());
        m.push(0);
        m.extend_from_slice(session.as_bytes());
        m.push(0);
        m.extend_from_slice(&counter.to_be_bytes());
        let digest = sha256(&m);
        self.log.push(LogEntry { digest });

        if self.log.len() % self.anchor_every == 0 {
            if let Some(path) = &self.anchor_path {
                let line = format!("{}\t{}\n", self.log.len(), hex(&digest));
                // Best-effort append; a real deployment commits this line to
                // a second repo/box the gateway cannot itself rewrite. Here
                // it is stubbed as a local append-only file.
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .and_then(|mut f| {
                        use std::io::Write;
                        f.write_all(line.as_bytes())
                    });
                // Commit the anchor line to this git repo now (SAFE-1c task
                // item 6): "publish externally" for this prototype means
                // "append-and-commit to a repo this same process cannot
                // rewrite after the fact without leaving history" -- a real
                // deployment would push to a second host/box; here we at
                // least get a signed, ordered commit trail on THIS repo,
                // one entry every `anchor_every` decisions, not just a
                // final summary commit. Best-effort: a git failure (e.g. no
                // repo, dirty tree with an unrelated conflict) must not
                // crash the gateway's decision path.
                if let (Some(dir), Some(fname)) = (path.parent(), path.file_name()) {
                    let dir = if dir.as_os_str().is_empty() {
                        Path::new(".")
                    } else {
                        dir
                    };
                    let _ = Command::new("git")
                        .args(["-C"])
                        .arg(dir)
                        .args(["add"])
                        .arg(fname)
                        .status();
                    let msg = format!(
                        "gateway ledger: chain head at decision {} = {}",
                        self.log.len(),
                        hex(&digest)
                    );
                    let _ = Command::new("git")
                        .args(["-C"])
                        .arg(dir)
                        .args(["commit", "-m"])
                        .arg(&msg)
                        .status();
                }
            }
        }
        digest
    }

    /// Run `agent_trace verify` against `receipt_path` and return its full
    /// stdout. Never trusts an exit code alone (the binary's own convention
    /// is inspected in agent_trace.rs's `main`; this is a subprocess
    /// boundary, so checking the printed verdict text mirrors the
    /// second-verifier mode in the design memo).
    ///
    /// Thin delegator to the free function `run_verify_subprocess_raw`
    /// (SAFE-7): the actual subprocess spawn/poll logic lives there,
    /// unchanged from before this split, so it can also be called by the
    /// async `serve` worker pool (see below) WITHOUT holding this
    /// `Gateway`'s mutex for the entire (potentially minutes-long) verify —
    /// the worker clones the handful of config fields it needs out of
    /// `Gateway` once, up front, and calls the free function directly.
    fn run_verify_raw(&self, receipt_path: &Path) -> Result<String, String> {
        run_verify_subprocess_raw(
            &self.agent_trace_bin,
            &self.model_path,
            &self.embed_path,
            &self.vocab_path,
            self.table_path.as_deref(),
            receipt_path,
            self.verify_timeout,
        )
    }

    /// PASS/FAIL per `run_verify_raw`'s stdout, ignoring grounding
    /// WARNINGs (the "lenient" policy: a receipt with an ungrounded
    /// tool-call argument can still ALLOW as long as the trace chain
    /// itself replays correctly).
    fn run_verify(&self, receipt_path: &Path) -> Result<bool, String> {
        Ok(verify_pass_lenient(&self.run_verify_raw(receipt_path)?))
    }

    /// Strict policy: PASS AND zero `WARNING step` lines in the replay
    /// output — i.e. every tool-call argument must appear verbatim in the
    /// context the model actually saw (per agent_trace.rs's own
    /// WARNING-set check, README.md "the set of WARNING lines a receipt
    /// carries must match the set the replay independently derives").
    /// This is a gateway-side policy choice layered on top of (not a
    /// replacement for) `agent_trace verify`'s own PASS/FAIL: a receipt can
    /// legitimately PASS with warnings (the trace chain is still
    /// internally consistent) but a strict deployment may choose not to
    /// authorize an action whose triggering argument was not grounded in
    /// what the agent actually saw.
    fn run_verify_strict(&self, receipt_path: &Path) -> Result<bool, String> {
        let stdout = self.run_verify_raw(receipt_path)?;
        Ok(verify_pass_strict(&stdout))
    }

    /// SAFE-7: the CHEAP part of `decide()` — receipt parse, allowlist
    /// membership, verify-execute binding hash, freshness MEMBERSHIP check
    /// (read-only: does NOT insert into `self.seen`; the caller decides
    /// when to reserve) — with none of the fields it needs an
    /// `agent_trace verify` subprocess for. This mirrors `decide()`'s own
    /// front half EXACTLY (same checks, same order, same DENY reasons) so
    /// that the async `serve` path can front-load an immediate DENY for
    /// anything `decide()` would have denied synchronously today, without
    /// ever touching a subprocess. `decide()` remains the single source of
    /// truth for the full decision LOGIC (including the slow verify) and
    /// is used unchanged by every existing caller (the CLI mode and the
    /// unit tests below); this function exists purely so `serve`'s async
    /// half does not have to duplicate decide()'s cheap-check bodies. If
    /// decide()'s cheap-check logic ever changes, mirror the change here.
    ///
    /// Returns `Err` with everything needed to log a DENY exactly as
    /// `decide()` would (empty trace_chain/triple for a read/parse
    /// failure, the parsed ones otherwise) — the caller is responsible for
    /// calling `append_log` itself, so this function has no side effects.
    /// Returns `Ok` with the parsed receipt and the (not yet inserted)
    /// freshness key once every cheap check has passed and only the slow
    /// `agent_trace verify` remains.
    fn cheap_precheck(
        &self,
        receipt_path: &Path,
        action_bytes: &[u8],
        session: &str,
        counter: u64,
    ) -> Result<(ParsedReceipt, FreshnessKey), (String, String, ArtifactTriple)> {
        let empty_triple = || ArtifactTriple {
            model: String::new(),
            embed: String::new(),
            vocab: String::new(),
        };
        let text = fs::read_to_string(receipt_path)
            .map_err(|e| (format!("read receipt: {e}"), String::new(), empty_triple()))?;
        let parsed = parse_receipt(&text)
            .map_err(|e| (format!("parse receipt: {e:?}"), String::new(), empty_triple()))?;

        if !self.allowlist.entries.contains(&parsed.triple) {
            return Err((
                "artifact triple not on allowlist".into(),
                parsed.trace_chain,
                parsed.triple,
            ));
        }

        let action_hash = sha256(action_bytes);
        let receipt_hash = sha256(&parsed.last_step_input);
        if action_hash != receipt_hash {
            return Err((
                format!(
                    "verify-execute binding failed: action hash {} != receipt final-step hash {}",
                    hex(&action_hash),
                    hex(&receipt_hash)
                ),
                parsed.trace_chain,
                parsed.triple,
            ));
        }

        let key = FreshnessKey {
            session: session.to_string(),
            counter,
            action_hash,
        };
        if self.seen.contains(&key) {
            return Err((
                "freshness: (session, counter, action-hash) already seen".into(),
                parsed.trace_chain,
                parsed.triple,
            ));
        }

        Ok((parsed, key))
    }

    /// The single decision entry point. `session`/`counter` are the
    /// freshness fields (b); `action_bytes` are the literal bytes of the
    /// tool call the caller is about to dispatch, checked byte-for-byte
    /// against the receipt's own last-step tool-call input (a).
    pub fn decide(
        &mut self,
        receipt_path: &Path,
        action_bytes: &[u8],
        session: &str,
        counter: u64,
    ) -> (Decision, [u8; 32]) {
        let text = match fs::read_to_string(receipt_path) {
            Ok(t) => t,
            Err(e) => {
                let d = Decision::Deny(format!("read receipt: {e}"));
                let head = self.append_log(
                    &d,
                    "",
                    &ArtifactTriple {
                        model: String::new(),
                        embed: String::new(),
                        vocab: String::new(),
                    },
                    session,
                    counter,
                );
                return (d, head);
            }
        };
        let parsed = match parse_receipt(&text) {
            Ok(p) => p,
            Err(e) => {
                let d = Decision::Deny(format!("parse receipt: {e:?}"));
                let head = self.append_log(
                    &d,
                    "",
                    &ArtifactTriple {
                        model: String::new(),
                        embed: String::new(),
                        vocab: String::new(),
                    },
                    session,
                    counter,
                );
                return (d, head);
            }
        };

        // Check (b) allowlist membership — cheap, do before shelling out.
        if !self.allowlist.entries.contains(&parsed.triple) {
            let d = Decision::Deny("artifact triple not on allowlist".into());
            let head = self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
            return (d, head);
        }

        // Check (a): verify -> execute binding. The receipt's own claimed
        // final-step tool-call bytes must hash-equal the action the caller
        // is proposing to actually execute. This must hold BEFORE we even
        // bother invoking agent_trace verify, since a mismatch here means
        // the receipt (however valid) does not authorize this call.
        let action_hash = sha256(action_bytes);
        let receipt_hash = sha256(&parsed.last_step_input);
        if action_hash != receipt_hash {
            let d = Decision::Deny(format!(
                "verify-execute binding failed: action hash {} != receipt final-step hash {}",
                hex(&action_hash),
                hex(&receipt_hash)
            ));
            let head = self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
            return (d, head);
        }

        // Check (b) freshness: replay of a (session, counter, action-hash)
        // triple already seen is a DENY, unconditionally, even if
        // everything else about this call is byte-identical to a
        // legitimately-once-allowed call.
        let key = FreshnessKey {
            session: session.to_string(),
            counter,
            action_hash,
        };
        if self.seen.contains(&key) {
            let d =
                Decision::Deny("freshness: (session, counter, action-hash) already seen".into());
            let head = self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
            return (d, head);
        }

        // Load-bearing check: agent_trace verify itself (chain replay,
        // WARNING-set match, artifact hashes re-checked independently by
        // agent_trace against the files on THIS machine).
        let verify_result = if self.strict {
            self.run_verify_strict(receipt_path)
        } else {
            self.run_verify(receipt_path)
        };
        match verify_result {
            Ok(true) => {}
            Ok(false) => {
                let reason = if self.strict {
                    "agent_trace verify: VERIFY FAIL or strict-grounding WARNING present"
                } else {
                    "agent_trace verify: VERIFY FAIL"
                };
                let d = Decision::Deny(reason.into());
                let head =
                    self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
                return (d, head);
            }
            Err(e) => {
                let d = Decision::Deny(format!("agent_trace verify: could not run: {e}"));
                let head =
                    self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
                return (d, head);
            }
        }

        self.seen.insert(key);
        let d = Decision::Allow;
        let head = self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
        (d, head)
    }
}

// =======================================================================
// SAFE-1c: wire the gateway in front of the tools-1 live20 episode corpus
// end to end, in a single long-lived process (matching a real agent loop,
// unlike the per-call CLI mode above whose freshness/log state resets on
// every exec). Prints a decision table to stdout: one line per decision,
// `DECISION\t<n>\t<id>\t<ALLOW|DENY:reason>\t<log-head-hex>`.
// =======================================================================
fn e2e_table_for(id: &str, tables_dir: &Path) -> Option<PathBuf> {
    if id.starts_with("chain_") {
        Some(tables_dir.join("chain.tsv"))
    } else if id.starts_with("lookup_") || id.starts_with("fileread_") || id.starts_with("mixed_") {
        Some(tables_dir.join("demo.tsv"))
    } else {
        None
    }
}

fn run_e2e_safe1c(args: &[String]) {
    // args: [1]=e2e-safe1c [2]=model [3]=embed [4]=vocab [5]=agent_trace_bin
    // [6]=live20_receipts_dir [7]=tables_dir [8]=work_dir [9]=ledger_file
    if args.len() < 10 {
        eprintln!(
            "usage: gateway e2e-safe1c <MODEL> <EMBED> <VOCAB> <agent_trace_bin> <live20_dir> <tables_dir> <work_dir> <ledger_file>"
        );
        std::process::exit(2);
    }
    let model = PathBuf::from(&args[2]);
    let embed = PathBuf::from(&args[3]);
    let vocab = PathBuf::from(&args[4]);
    let bin = PathBuf::from(&args[5]);
    let live20_dir = PathBuf::from(&args[6]);
    let tables_dir = PathBuf::from(&args[7]);
    let work_dir = PathBuf::from(&args[8]);
    let ledger_path = PathBuf::from(&args[9]);
    fs::create_dir_all(&work_dir).expect("create work_dir");

    let mh = hex(&sha256(&fs::read(&model).expect("read model")));
    let eh = hex(&sha256(&fs::read(&embed).expect("read embed")));
    let vh = hex(&sha256(&fs::read(&vocab).expect("read vocab")));

    let key = b"safe1c-e2e-box1-hmac-key-not-for-prod".to_vec();
    let good_lines = vec![format!("{mh} {eh} {vh}")];
    let good_refs: Vec<&str> = good_lines.iter().map(|s| s.as_str()).collect();
    let good_sig = sign_allowlist(&good_refs, &key);
    let good_allowlist_path = work_dir.join("allowlist.good.signed");
    fs::write(
        &good_allowlist_path,
        format!("{}\nsig {}\n", good_lines[0], good_sig),
    )
    .unwrap();

    // A "one-bit-flipped MODEL.SAF" allowlist entry: hash a copy of the
    // real model with its first byte XORed, and sign an allowlist listing
    // ONLY that (wrong) triple. A receipt generated against the real model
    // declares the REAL hash, so it will never match -> DENY case (iii).
    let mut flipped = fs::read(&model).expect("read model for flip");
    flipped[0] ^= 0x01;
    let mh_flipped = hex(&sha256(&flipped));
    let bad_lines = vec![format!("{mh_flipped} {eh} {vh}")];
    let bad_refs: Vec<&str> = bad_lines.iter().map(|s| s.as_str()).collect();
    let bad_sig = sign_allowlist(&bad_refs, &key);
    let bad_allowlist_path = work_dir.join("allowlist.bitflipped.signed");
    fs::write(
        &bad_allowlist_path,
        format!("{}\nsig {}\n", bad_lines[0], bad_sig),
    )
    .unwrap();

    let mut ids: Vec<String> = fs::read_dir(&live20_dir)
        .expect("live20 dir")
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter_map(|n| n.strip_suffix(".receipt").map(|s| s.to_string()))
        .collect();
    ids.sort();

    // The gateway's own per-decision anchor mechanism (Gateway::append_log,
    // triggered every `cfg.anchor_every` entries in THAT Gateway's log) is
    // per-instance; this driver builds a fresh Gateway per episode (the
    // `--table` differs per episode), so no single Gateway's internal
    // counter ever reaches 10. The ledger cadence here is therefore driven
    // explicitly by the driver's own global decision count `n`, which is
    // the correct behavior for "every 10 decisions this process makes",
    // not an artifact of gateway construction. It writes the same anchor
    // line format and does the same commit as Gateway::append_log's own
    // anchor path (see that function for why a git failure is non-fatal).
    let mut n = 0usize;
    let mut emit = |id: &str, d: &Decision, head: [u8; 32]| {
        n += 1;
        let tag = match d {
            Decision::Allow => "ALLOW".to_string(),
            Decision::Deny(r) => format!("DENY:{r}"),
        };
        println!("DECISION\t{n}\t{id}\t{tag}\t{}", hex(&head));
        if n % 10 == 0 {
            let line = format!("{}\t{}\n", n, hex(&head));
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&ledger_path)
                .and_then(|mut f| {
                    use std::io::Write;
                    f.write_all(line.as_bytes())
                });
            if let (Some(dir), Some(fname)) = (ledger_path.parent(), ledger_path.file_name()) {
                let dir = if dir.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    dir
                };
                let _ = Command::new("git")
                    .arg("-C")
                    .arg(dir)
                    .arg("add")
                    .arg(fname)
                    .status();
                let msg = format!(
                    "gateway ledger: chain head at decision {n} = {}",
                    hex(&head)
                );
                let _ = Command::new("git")
                    .arg("-C")
                    .arg(dir)
                    .arg("commit")
                    .arg("-m")
                    .arg(&msg)
                    .status();
            }
        }
    };

    // Phase 1: all 20 legitimate live episodes, lenient policy, distinct
    // (session, counter) per call -- expect ALLOW 20/20 (confirmed by the
    // gateway's own live20_lenient_allow_count test on this same corpus).
    let mut replay_receipt: Option<(PathBuf, Vec<u8>)> = None;
    let mut counter = 0u64;
    for id in &ids {
        counter += 1;
        let receipt_path = live20_dir.join(format!("{id}.receipt"));
        let text = fs::read_to_string(&receipt_path).unwrap();
        let parsed = parse_receipt(&text).expect("live20 receipts are well-formed");
        let action = parsed.last_step_input.clone();
        let mut gw = Gateway::new(GatewayConfig {
            allowlist_path: good_allowlist_path.clone(),
            allowlist_key: key.clone(),
            anchor_every: 10,
            anchor_path: None, // ledger committed explicitly in the driver, see emit() below
            agent_trace_bin: bin.clone(),
            model_path: model.clone(),
            embed_path: embed.clone(),
            vocab_path: vocab.clone(),
            table_path: e2e_table_for(id, &tables_dir),
            strict: false,
            verify_timeout: None,
        })
        .expect("gateway init per-episode");
        let (d, head) = gw.decide(&receipt_path, &action, "live-loop", counter);
        emit(id, &d, head);
        if replay_receipt.is_none() {
            replay_receipt = Some((receipt_path.clone(), action.clone()));
        }
    }

    // DENY (i): replay -- reuse episode 1's exact (receipt, action,
    // session, counter) again. Freshness state is per-Gateway-instance, so
    // reuse the SAME gateway/session namespace by replaying counter=1
    // against a NEW Gateway that has already seen counter=1 via a priming
    // call, to demonstrate the freshness rule in isolation without
    // depending on cross-process persistence (a known prototype gap, noted
    // in the report).
    {
        let (rp, action) = replay_receipt.clone().unwrap();
        let mut gw = Gateway::new(GatewayConfig {
            allowlist_path: good_allowlist_path.clone(),
            allowlist_key: key.clone(),
            anchor_every: 10,
            anchor_path: None, // ledger committed explicitly in the driver, see emit() below
            agent_trace_bin: bin.clone(),
            model_path: model.clone(),
            embed_path: embed.clone(),
            vocab_path: vocab.clone(),
            table_path: e2e_table_for(&ids[0], &tables_dir),
            strict: false,
            verify_timeout: None,
        })
        .unwrap();
        let (d1, h1) = gw.decide(&rp, &action, "replay-demo", 1);
        emit(&format!("{}(replay-priming)", ids[0]), &d1, h1);
        let (d2, h2) = gw.decide(&rp, &action, "replay-demo", 1);
        emit(&format!("{}(REPLAY)", ids[0]), &d2, h2);
    }

    // DENY (ii): ungrounded argument under --strict-grounding. Use a
    // strict-mode Gateway against an episode already known (from
    // live20_strict_allow_count) to carry an ungrounded tool-call argument.
    {
        let strict_id = ids
            .iter()
            .find(|i| i.starts_with("mixed_lookup_calc"))
            .cloned()
            .unwrap_or_else(|| ids[0].clone());
        let receipt_path = live20_dir.join(format!("{strict_id}.receipt"));
        let text = fs::read_to_string(&receipt_path).unwrap();
        let parsed = parse_receipt(&text).unwrap();
        let action = parsed.last_step_input.clone();
        let mut gw = Gateway::new(GatewayConfig {
            allowlist_path: good_allowlist_path.clone(),
            allowlist_key: key.clone(),
            anchor_every: 10,
            anchor_path: None, // ledger committed explicitly in the driver, see emit() below
            agent_trace_bin: bin.clone(),
            model_path: model.clone(),
            embed_path: embed.clone(),
            vocab_path: vocab.clone(),
            table_path: e2e_table_for(&strict_id, &tables_dir),
            strict: true,
            verify_timeout: None,
        })
        .unwrap();
        let (d, head) = gw.decide(&receipt_path, &action, "strict-demo", 1);
        emit(&format!("{strict_id}(STRICT-UNGROUNDED)"), &d, head);
    }

    // DENY (iii): weights digest not on the allowlist (one-bit-flipped
    // MODEL.SAF hash is the only entry on this allowlist).
    {
        let receipt_path = live20_dir.join(format!("{}.receipt", ids[0]));
        let text = fs::read_to_string(&receipt_path).unwrap();
        let parsed = parse_receipt(&text).unwrap();
        let action = parsed.last_step_input.clone();
        let mut gw = Gateway::new(GatewayConfig {
            allowlist_path: bad_allowlist_path.clone(),
            allowlist_key: key.clone(),
            anchor_every: 10,
            anchor_path: None, // ledger committed explicitly in the driver, see emit() below
            agent_trace_bin: bin.clone(),
            model_path: model.clone(),
            embed_path: embed.clone(),
            vocab_path: vocab.clone(),
            table_path: e2e_table_for(&ids[0], &tables_dir),
            strict: false,
            verify_timeout: None,
        })
        .unwrap();
        let (d, head) = gw.decide(&receipt_path, &action, "allowlist-demo", 1);
        emit(&format!("{}(BITFLIP-ALLOWLIST)", ids[0]), &d, head);
    }

    // DENY (iv): forwarded bytes tampered after verification -- flip one
    // bit of the action bytes actually dispatched, vs. what the receipt's
    // own last step says happened.
    {
        let receipt_path = live20_dir.join(format!("{}.receipt", ids[1]));
        let text = fs::read_to_string(&receipt_path).unwrap();
        let parsed = parse_receipt(&text).unwrap();
        let mut action = parsed.last_step_input.clone();
        if let Some(b) = action.first_mut() {
            *b ^= 0x01;
        } else {
            action.push(1);
        }
        let mut gw = Gateway::new(GatewayConfig {
            allowlist_path: good_allowlist_path.clone(),
            allowlist_key: key.clone(),
            anchor_every: 10,
            anchor_path: None, // ledger committed explicitly in the driver, see emit() below
            agent_trace_bin: bin.clone(),
            model_path: model.clone(),
            embed_path: embed.clone(),
            vocab_path: vocab.clone(),
            table_path: e2e_table_for(&ids[1], &tables_dir),
            strict: false,
            verify_timeout: None,
        })
        .unwrap();
        let (d, head) = gw.decide(&receipt_path, &action, "tamper-demo", 1);
        emit(&format!("{}(TAMPERED-BYTES)", ids[1]), &d, head);
    }

    eprintln!("e2e-safe1c: {n} decisions logged");
}

// =======================================================================
// SAFE-5b: daemon mode. `gateway serve` binds a unix socket and keeps one
// Gateway instance alive for the process's lifetime, so freshness state
// (the `seen` set) survives across calls — fixing the "freshness state
// not surviving across calls" gap flagged in the safe1c e2e report. One
// request per connection: the client sends
//   RECEIPT <path>
//   ACTION <hex>
//   SESSION <session>
//   COUNTER <counter>
//   (blank line or EOF ends the request)
// and gets back exactly one line (see SAFE-7 below for the PENDING case):
//   ALLOW idx=<n> cap=<token> exp=<unix>
//   DENY <reason>
// On ALLOW, a capability token is minted via `capability::issue` using
// this Gateway's own monotonic decision_count() as the index (never
// caller-supplied) and `--cap-ttl` seconds from now as the expiry. Tool
// shims (`src/bin/tool_shim_*.rs`) independently verify that token before
// acting — see `src/capability.rs` and ENFORCEMENT.md.
//
// SAFE-7: async verify, so one slow `agent_trace verify` (observed 6-7+
// minutes for a real 2B receipt on this box's hardware) cannot block the
// accept loop for every connection behind it. On a new decision request
// the accept thread still does every CHEAP check INLINE and synchronously
// (`Gateway::cheap_precheck`: receipt parse, allowlist, verify-execute
// binding, freshness membership) -- if any of those denies, the reply is
// `DENY <reason>` immediately, exactly as before SAFE-7, never queued.
// Only once all of those pass does the (session, counter, action-hash)
// freshness key get RESERVED (inserted into `seen` right there, on the
// accept thread, before the connection replies) and the receipt handed to
// a background worker pool (`--verify-workers N`, default 1) that runs the
// actual `agent_trace verify`. The connection is replied to immediately
// with:
//   PENDING ticket=<id>
// The freshness key is reserved on ENQUEUE, not on eventual ALLOW: this is
// the "keep reserved" rule from the SAFE-7 brief -- a second concurrent
// request for the exact same (session, counter) arriving while the first
// is still PENDING must deterministically DENY (freshness) rather than
// race the verify, and a verify that later fails or errors does NOT free
// the key back up either (a replayed session/counter must never be able
// to produce two live outcomes, so a caller who wants to retry after a
// DENY must use a new counter, same as it always did after a real ALLOW).
// The caller learns the outcome by reconnecting and sending:
//   POLL ticket=<id>
// which replies `PENDING` (verify still running), `DENY unknown ticket`
// (ticket never issued by this process, e.g. after a restart), or the
// final `ALLOW ...`/`DENY ...` line -- byte-for-byte what a synchronous
// `decide()` call would have produced for the same inputs (async changes
// WHEN the caller learns the outcome, never WHAT the outcome is). Polling
// is out-of-band from the PENDING connection: the client is expected to
// close that connection (or it stays open with nothing more to read/write
// until the client sends POLL, either is fine) and reconnect for POLL, one
// request per connection, matching the existing one-request-per-connection
// convention above.
// =======================================================================

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

struct ServeRequest {
    receipt: PathBuf,
    action_hex: String,
    session: String,
    counter: u64,
}

/// SAFE-7: a parsed request is either a normal decision request (as
/// before) or a `POLL ticket=<id>` request asking for that ticket's
/// current/final status.
enum ServeRequestKind {
    Decide(ServeRequest),
    Poll { ticket: String },
}

fn parse_serve_request(stream: &mut UnixStream) -> Option<ServeRequestKind> {
    let reader = BufReader::new(stream.try_clone().ok()?);
    let mut lines = reader.lines();
    let first = lines.next()?.ok()?;
    if let Some(rest) = first.strip_prefix("POLL ") {
        let ticket = rest.trim().strip_prefix("ticket=")?.trim().to_string();
        if ticket.is_empty() {
            return None;
        }
        // Drain the rest of the request up to the blank-line/EOF
        // terminator, same convention as the decide-request parse below,
        // in case a client sends extra lines after POLL.
        for line in lines {
            let line = line.ok()?;
            if line.trim().is_empty() {
                break;
            }
        }
        return Some(ServeRequestKind::Poll { ticket });
    }

    let mut receipt = None;
    let mut action_hex = String::new();
    let mut session = "default".to_string();
    let mut counter = 0u64;
    let mut apply = |line: &str, receipt: &mut Option<PathBuf>| {
        if let Some(rest) = line.strip_prefix("RECEIPT ") {
            *receipt = Some(PathBuf::from(rest.trim()));
        } else if let Some(rest) = line.strip_prefix("ACTION ") {
            action_hex = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("SESSION ") {
            session = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("COUNTER ") {
            counter = rest.trim().parse().unwrap_or(0);
        }
    };
    if !first.trim().is_empty() {
        apply(&first, &mut receipt);
    }
    for line in lines {
        let line = line.ok()?;
        if line.trim().is_empty() {
            break;
        }
        apply(&line, &mut receipt);
    }
    Some(ServeRequestKind::Decide(ServeRequest {
        receipt: receipt?,
        action_hex,
        session,
        counter,
    }))
}

/// SAFE-7: outcome of a ticket issued for background verification.
#[derive(Clone)]
enum TicketOutcome {
    Pending,
    /// The exact final reply line, e.g. `ALLOW idx=.. cap=.. exp=..` or
    /// `DENY <reason>` -- precomputed once by the worker so `POLL` just
    /// echoes it back verbatim, matching what a synchronous `decide()`
    /// call would have produced.
    Done(String),
}

/// SAFE-7: everything a background verify worker needs, captured at
/// enqueue time (cheap clones of config `PathBuf`s/bools) so the worker
/// never needs to touch the `Gateway` mutex while `agent_trace verify`
/// itself is running -- only the final `Gateway::append_log` + decision
/// bookkeeping below needs the lock, briefly.
struct VerifyJob {
    ticket: String,
    receipt_path: PathBuf,
    session: String,
    counter: u64,
    trace_chain: String,
    triple: ArtifactTriple,
    freshness_key: FreshnessKey,
    agent_trace_bin: PathBuf,
    model_path: PathBuf,
    embed_path: PathBuf,
    vocab_path: PathBuf,
    table_path: Option<PathBuf>,
    strict: bool,
    verify_timeout: Option<Duration>,
    cap_key: Vec<u8>,
    cap_ttl: u64,
}

/// State shared between the (single-threaded) accept loop and the
/// background verify worker pool.
struct SharedServe {
    gw: std::sync::Mutex<Gateway>,
    tickets: std::sync::Mutex<std::collections::HashMap<String, TicketOutcome>>,
    job_tx: std::sync::mpsc::Sender<VerifyJob>,
    next_ticket: std::sync::atomic::AtomicU64,
    cap_key: Vec<u8>,
    cap_ttl: u64,
}

impl SharedServe {
    fn mint_ticket(&self) -> String {
        let n = self
            .next_ticket
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        format!("T{n}")
    }
}

/// One background worker thread's main loop: pull a `VerifyJob`, run the
/// (potentially very slow) `agent_trace verify` WITHOUT holding `shared`'s
/// Gateway mutex, then lock briefly to finalize the decision (append the
/// second, decision, log entry; on ALLOW, mint the capability token) and
/// record the ticket's outcome.
fn verify_worker_loop(
    rx: std::sync::Arc<std::sync::Mutex<std::sync::mpsc::Receiver<VerifyJob>>>,
    shared: std::sync::Arc<SharedServe>,
) {
    loop {
        let job = {
            let rx = rx.lock().unwrap();
            rx.recv()
        };
        let job = match job {
            Ok(j) => j,
            Err(_) => return, // sender dropped: shutting down
        };

        let verify_result = run_verify_subprocess_raw(
            &job.agent_trace_bin,
            &job.model_path,
            &job.embed_path,
            &job.vocab_path,
            job.table_path.as_deref(),
            &job.receipt_path,
            job.verify_timeout,
        );
        let pass = match &verify_result {
            Ok(stdout) => {
                if job.strict {
                    verify_pass_strict(stdout)
                } else {
                    verify_pass_lenient(stdout)
                }
            }
            Err(_) => false,
        };

        let final_line = {
            let mut gw = shared.gw.lock().unwrap();
            if pass {
                // Freshness key was already reserved at enqueue time; this
                // insert is a no-op for that key but keeps the invariant
                // explicit here too (ALLOW always implies "seen").
                gw.seen.insert(job.freshness_key.clone());
                let d = Decision::Allow;
                let _head =
                    gw.append_log(&d, &job.trace_chain, &job.triple, &job.session, job.counter);
                let idx = gw.decision_count();
                let exp = now_unix() + job.cap_ttl;
                let token = capability::issue(&job.cap_key, idx, job.freshness_key.action_hash, exp);
                format!("ALLOW idx={idx} cap={token} exp={exp}")
            } else {
                let reason = match &verify_result {
                    Ok(_) if job.strict => {
                        "agent_trace verify: VERIFY FAIL or strict-grounding WARNING present"
                            .to_string()
                    }
                    Ok(_) => "agent_trace verify: VERIFY FAIL".to_string(),
                    Err(e) => format!("agent_trace verify: could not run: {e}"),
                };
                let d = Decision::Deny(reason.clone());
                let _head =
                    gw.append_log(&d, &job.trace_chain, &job.triple, &job.session, job.counter);
                // Freshness key stays reserved -- see the SAFE-7 daemon-mode
                // comment block above run_serve for why a failed/erroring
                // verify does not free it for retry under the same
                // (session, counter).
                format!("DENY {reason}")
            }
        };
        shared
            .tickets
            .lock()
            .unwrap()
            .insert(job.ticket.clone(), TicketOutcome::Done(final_line));
    }
}

fn handle_serve_conn_async(stream: &mut UnixStream, shared: &std::sync::Arc<SharedServe>) {
    let kind = match parse_serve_request(stream) {
        Some(k) => k,
        None => {
            let _ = writeln!(stream, "DENY malformed request");
            return;
        }
    };
    match kind {
        ServeRequestKind::Poll { ticket } => {
            let outcome = shared.tickets.lock().unwrap().get(&ticket).cloned();
            match outcome {
                Some(TicketOutcome::Done(line)) => {
                    let _ = writeln!(stream, "{line}");
                }
                Some(TicketOutcome::Pending) => {
                    let _ = writeln!(stream, "PENDING");
                }
                None => {
                    let _ = writeln!(stream, "DENY unknown ticket");
                }
            }
        }
        ServeRequestKind::Decide(req) => {
            let action_bytes = unhex(&req.action_hex).unwrap_or_default();
            let mut gw = shared.gw.lock().unwrap();
            match gw.cheap_precheck(&req.receipt, &action_bytes, &req.session, req.counter) {
                Err((reason, trace_chain, triple)) => {
                    let d = Decision::Deny(reason.clone());
                    let _head = gw.append_log(&d, &trace_chain, &triple, &req.session, req.counter);
                    drop(gw);
                    let _ = writeln!(stream, "DENY {reason}");
                }
                Ok((parsed, key)) => {
                    // Reserve the freshness key NOW, before any background
                    // work is enqueued and before this connection is even
                    // replied to: a second request for the identical
                    // (session, counter) that reaches this same
                    // single-threaded accept loop afterward will see this
                    // key already in `seen` via cheap_precheck above and
                    // DENY immediately, never racing the verify below.
                    gw.seen.insert(key.clone());
                    let ticket = shared.mint_ticket();
                    let _head = gw.append_log_pending(
                        &ticket,
                        &parsed.trace_chain,
                        &parsed.triple,
                        &req.session,
                        req.counter,
                    );
                    let (agent_trace_bin, model_path, embed_path, vocab_path, table_path, strict, verify_timeout) = (
                        gw.agent_trace_bin.clone(),
                        gw.model_path.clone(),
                        gw.embed_path.clone(),
                        gw.vocab_path.clone(),
                        gw.table_path.clone(),
                        gw.strict,
                        gw.verify_timeout,
                    );
                    drop(gw);

                    shared
                        .tickets
                        .lock()
                        .unwrap()
                        .insert(ticket.clone(), TicketOutcome::Pending);

                    let job = VerifyJob {
                        ticket: ticket.clone(),
                        receipt_path: req.receipt.clone(),
                        session: req.session.clone(),
                        counter: req.counter,
                        trace_chain: parsed.trace_chain.clone(),
                        triple: parsed.triple.clone(),
                        freshness_key: key,
                        agent_trace_bin,
                        model_path,
                        embed_path,
                        vocab_path,
                        table_path,
                        strict,
                        verify_timeout,
                        cap_key: shared.cap_key.clone(),
                        cap_ttl: shared.cap_ttl,
                    };
                    let _ = shared.job_tx.send(job);
                    let _ = writeln!(stream, "PENDING ticket={ticket}");
                }
            }
        }
    }
}

fn run_serve(args: &[String]) {
    // args: [1]=serve [2]=model [3]=embed [4]=vocab [5]=agent_trace_bin
    // [6]=allowlist [7]=allowlist-key-file --socket S --cap-key-file K
    // [--table T] [--anchor-every N] [--anchor-file F] [--strict]
    // [--cap-ttl SECS] [--request-timeout SECS] [--verify-workers N]
    if args.len() < 8 {
        eprintln!(
            "usage: gateway serve <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <agent_trace-bin> <allowlist> <allowlist-key-file> --socket <path> --cap-key-file <path> [--table <path>] [--anchor-every N] [--anchor-file F] [--strict] [--cap-ttl SECS] [--request-timeout SECS] [--verify-workers N]"
        );
        std::process::exit(2);
    }
    let model = PathBuf::from(&args[2]);
    let embed = PathBuf::from(&args[3]);
    let vocab = PathBuf::from(&args[4]);
    let bin = PathBuf::from(&args[5]);
    let allowlist_path = PathBuf::from(&args[6]);
    let key_path = PathBuf::from(&args[7]);

    let mut socket_path: Option<PathBuf> = std::env::var("XDG_RUNTIME_DIR")
        .ok()
        .map(|d| PathBuf::from(d).join("cm-gateway.sock"));
    let mut cap_key_path: Option<PathBuf> = None;
    let mut table_path = None;
    let mut anchor_every = 5usize;
    let mut anchor_path = None;
    let mut strict = false;
    let mut cap_ttl = 60u64;
    // Default: no timeout (old behavior). Set --request-timeout SECS to
    // bound how long one `agent_trace verify` subprocess may run before
    // the daemon kills it and DENYs, so a slow/stuck verify (e.g. a
    // BitNet-2B receipt on weak hardware) cannot silently block every
    // later request forever behind it in the single-threaded accept loop.
    let mut request_timeout: Option<Duration> = None;
    // SAFE-7: number of background threads running `agent_trace verify`
    // concurrently. Default 1, matching this box's 2c/2t hardware -- more
    // workers than physical threads just makes each individual verify
    // slower without helping throughput here.
    let mut verify_workers = 1usize;
    let mut i = 8;
    while i < args.len() {
        match args[i].as_str() {
            "--socket" => {
                socket_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--cap-key-file" => {
                cap_key_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--table" => {
                table_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--anchor-every" => {
                anchor_every = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
                i += 2;
            }
            "--anchor-file" => {
                anchor_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--strict" => {
                strict = true;
                i += 1;
            }
            "--cap-ttl" => {
                cap_ttl = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(60);
                i += 2;
            }
            "--request-timeout" => {
                request_timeout = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(Duration::from_secs);
                i += 2;
            }
            "--verify-workers" => {
                verify_workers = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|n| *n > 0)
                    .unwrap_or(1);
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }

    let socket_path = socket_path.unwrap_or_else(|| {
        eprintln!("gateway serve: no --socket given and $XDG_RUNTIME_DIR not set");
        std::process::exit(2);
    });
    let cap_key_path = cap_key_path.unwrap_or_else(|| {
        eprintln!("gateway serve: --cap-key-file is required");
        std::process::exit(2);
    });

    let allowlist_key = fs::read(&key_path).unwrap_or_else(|e| {
        eprintln!("read allowlist key file {}: {e}", key_path.display());
        std::process::exit(2);
    });
    let cap_key = fs::read(&cap_key_path).unwrap_or_else(|e| {
        eprintln!("read capability key file {}: {e}", cap_key_path.display());
        std::process::exit(2);
    });

    let gw = match Gateway::new(GatewayConfig {
        allowlist_path,
        allowlist_key,
        anchor_every,
        anchor_path,
        agent_trace_bin: bin,
        model_path: model,
        embed_path: embed,
        vocab_path: vocab,
        table_path,
        strict,
        verify_timeout: request_timeout,
    }) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GATEWAY INIT FAIL: {e:?}");
            std::process::exit(1);
        }
    };

    // A stale socket from a previous run must not block bind.
    let _ = fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path).unwrap_or_else(|e| {
        eprintln!("bind {}: {e}", socket_path.display());
        std::process::exit(1);
    });
    // Owner-only: this socket is the sole path to a real decision.
    let _ = fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600));
    eprintln!(
        "gateway serve: listening on {} (cap-ttl={cap_ttl}s, verify-workers={verify_workers})",
        socket_path.display()
    );

    // SAFE-7: wire up the shared state + worker pool (see the big comment
    // block above `now_unix` for the full design). `job_tx` is cloned into
    // `SharedServe` and handed out to `handle_serve_conn_async`; `job_rx`
    // is shared (behind a Mutex) by every worker thread, each pulling the
    // next job in FIFO order.
    let (job_tx, job_rx) = std::sync::mpsc::channel::<VerifyJob>();
    let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
    let shared = std::sync::Arc::new(SharedServe {
        gw: std::sync::Mutex::new(gw),
        tickets: std::sync::Mutex::new(std::collections::HashMap::new()),
        job_tx,
        next_ticket: std::sync::atomic::AtomicU64::new(1),
        cap_key,
        cap_ttl,
    });
    for _ in 0..verify_workers {
        let rx = job_rx.clone();
        let shared = shared.clone();
        std::thread::spawn(move || verify_worker_loop(rx, shared));
    }

    // Accept loop is single-threaded: a connection is only ACCEPTED once
    // the previous connection's handler has returned. That is no longer a
    // problem for a slow verify (SAFE-7's whole point) because
    // `handle_serve_conn_async` never blocks on `agent_trace verify`
    // itself -- it does the cheap checks, then either denies immediately
    // or hands off to a worker thread and replies `PENDING` right away.
    // `--request-timeout` still bounds how long any ONE worker's verify
    // subprocess may run before that worker kills it and records DENY.
    let mut conn_count: u64 = 0;
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                conn_count += 1;
                eprintln!("gateway serve: accepted #{conn_count}");
                handle_serve_conn_async(&mut stream, &shared);
            }
            Err(e) => eprintln!("gateway serve: accept error: {e}"),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "e2e-safe1c" {
        run_e2e_safe1c(&args);
        return;
    }
    if args.len() > 1 && args[1] == "serve" {
        run_serve(&args);
        return;
    }
    if args.len() < 8 {
        eprintln!(
            "usage: gateway <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <agent_trace-bin> <allowlist> <key-file> <receipt> <action-hex> <session> <counter> [--table <path>] [--anchor-every N] [--anchor-file F] [--strict]"
        );
        std::process::exit(2);
    }
    let model = PathBuf::from(&args[1]);
    let embed = PathBuf::from(&args[2]);
    let vocab = PathBuf::from(&args[3]);
    let bin = PathBuf::from(&args[4]);
    let allowlist_path = PathBuf::from(&args[5]);
    let key_path = PathBuf::from(&args[6]);
    let receipt = PathBuf::from(&args[7]);
    let action_hex = args.get(8).cloned().unwrap_or_default();
    let session = args.get(9).cloned().unwrap_or_else(|| "default".into());
    let counter: u64 = args.get(10).and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut table_path = None;
    let mut anchor_every = 5usize;
    let mut anchor_path = None;
    let mut strict = false;
    let mut i = 11;
    while i < args.len() {
        match args[i].as_str() {
            "--table" => {
                table_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--anchor-every" => {
                anchor_every = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5);
                i += 2;
            }
            "--anchor-file" => {
                anchor_path = args.get(i + 1).map(PathBuf::from);
                i += 2;
            }
            "--strict" => {
                strict = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let key = fs::read(&key_path).unwrap_or_else(|e| {
        eprintln!("read key file {}: {e}", key_path.display());
        std::process::exit(2);
    });

    let mut gw = match Gateway::new(GatewayConfig {
        allowlist_path,
        allowlist_key: key,
        anchor_every,
        anchor_path,
        agent_trace_bin: bin,
        model_path: model,
        embed_path: embed,
        vocab_path: vocab,
        table_path,
        strict,
        verify_timeout: None,
    }) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GATEWAY INIT FAIL: {e:?}");
            std::process::exit(1);
        }
    };

    let action_bytes = unhex(&action_hex).unwrap_or_default();
    let (decision, head) = gw.decide(&receipt, &action_bytes, &session, counter);
    match decision {
        Decision::Allow => {
            println!("ALLOW log-head={}", hex(&head));
            std::process::exit(0);
        }
        Decision::Deny(reason) => {
            println!("DENY {reason} log-head={}", hex(&head));
            std::process::exit(1);
        }
    }
}

// =======================================================================
// Tests
// =======================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TESTDIR_CTR: AtomicU64 = AtomicU64::new(0);

    fn workdir() -> PathBuf {
        let n = TESTDIR_CTR.fetch_add(1, Ordering::SeqCst);
        let d = std::env::temp_dir().join(format!("gateway-test-{}-{}", std::process::id(), n));
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn repo_root() -> PathBuf {
        // demo/agent-trace/gateway -> repo root is 3 levels up.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn agent_trace_bin() -> PathBuf {
        // Prefer a release build (real-2B-model receipts in the live20
        // integration test are far too slow to verify against an
        // unoptimized debug build).
        //
        // IMPORTANT: always run `cargo build --release` here rather than
        // just checking whether a binary already exists on disk and
        // reusing it. A previous bug let this fn silently verify against
        // a stale, out-of-date `agent_trace` binary left over from an
        // earlier build, producing bogus VERIFY FAIL/ALLOW results that
        // did not reflect the current source tree. `cargo build` is a
        // fast no-op if nothing changed, so this costs nothing when the
        // binary is already current.
        let status = Command::new("cargo")
            .args([
                "build",
                "--release",
                "--offline",
                "--example",
                "agent_trace",
            ])
            .current_dir(repo_root().join("aegis-linux"))
            .status()
            .expect("cargo build --release agent_trace");
        assert!(
            status.success(),
            "failed to build agent_trace example (release)"
        );

        let bin = repo_root().join("aegis-linux/target/release/examples/agent_trace");
        assert!(
            bin.exists(),
            "agent_trace release binary missing after build: {}",
            bin.display()
        );

        // Log the binary's sha256 so a stale-verifier situation is
        // detectable from test output going forward.
        let digest = hex(&sha256(&fs::read(&bin).unwrap()));
        eprintln!("agent_trace_bin: using {} sha256={}", bin.display(), digest);

        bin
    }

    fn artifacts() -> (PathBuf, PathBuf, PathBuf) {
        let a = repo_root().join("model-lab/tinybit/m7_final_gate_work/artifacts");
        (
            a.join("MODEL.SAF"),
            a.join("EMBED.BIN"),
            a.join("VOCAB.BIN"),
        )
    }

    fn artifact_hexes() -> (String, String, String) {
        let (m, e, v) = artifacts();
        (
            hex(&sha256(&fs::read(m).unwrap())),
            hex(&sha256(&fs::read(e).unwrap())),
            hex(&sha256(&fs::read(v).unwrap())),
        )
    }

    /// Generate a fresh CALC receipt via `agent_trace gen` into `dir`,
    /// return its path.
    fn gen_receipt(dir: &Path, prompt: &str, k: u32, n: u32) -> PathBuf {
        let (m, e, v) = artifacts();
        let bin = agent_trace_bin();
        let out = dir.join("r.receipt");
        let status = Command::new(&bin)
            .args(["gen"])
            .arg(&m)
            .arg(&e)
            .arg(&v)
            .arg(k.to_string())
            .arg(n.to_string())
            .arg(prompt)
            .stdout(fs::File::create(&out).unwrap())
            .status()
            .expect("run agent_trace gen");
        assert!(status.success(), "agent_trace gen failed");
        out
    }

    fn write_signed_allowlist(
        dir: &Path,
        triples: &[(String, String, String)],
        key: &[u8],
    ) -> PathBuf {
        let lines: Vec<String> = triples
            .iter()
            .map(|(m, e, v)| format!("{m} {e} {v}"))
            .collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let sig = sign_allowlist(&refs, key);
        let mut text = lines.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&format!("sig {sig}\n"));
        let path = dir.join("allowlist.signed");
        fs::write(&path, text).unwrap();
        path
    }

    fn make_gateway(
        dir: &Path,
        allowlist_triples: &[(String, String, String)],
    ) -> (Gateway, Vec<u8>) {
        let (m, e, v) = artifacts();
        let key = b"test-hmac-key-not-for-prod".to_vec();
        let allow_path = write_signed_allowlist(dir, allowlist_triples, &key);
        let gw = Gateway::new(GatewayConfig {
            allowlist_path: allow_path,
            allowlist_key: key.clone(),
            anchor_every: 2,
            anchor_path: Some(dir.join("anchor.log")),
            agent_trace_bin: agent_trace_bin(),
            model_path: m,
            embed_path: e,
            vocab_path: v,
            table_path: None,
            strict: false,
            verify_timeout: None,
        })
        .expect("gateway init");
        (gw, key)
    }

    /// Pull the hex bytes of a receipt's last `in=` field, matching what
    /// the gateway itself would treat as "the action this receipt
    /// authorizes" — used by tests to build a genuinely matching action.
    fn last_step_in_hex(receipt_path: &Path) -> String {
        let text = fs::read_to_string(receipt_path).unwrap();
        let mut last = None;
        for line in text.lines() {
            if line.starts_with("step ") {
                for field in line.split_whitespace() {
                    if let Some(h) = field.strip_prefix("in=") {
                        last = Some(h.to_string());
                    }
                }
            }
        }
        last.expect("receipt has at least one step")
    }

    // -------------------------------------------------------------
    // 1. Baseline ALLOW: valid receipt, on allowlist, matching action,
    //    fresh (session, counter).
    // -------------------------------------------------------------
    #[test]
    fn allow_valid_receipt_matching_action_fresh_nonce() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);
        let action = unhex(&last_step_in_hex(&receipt)).unwrap();
        let (d, _head) = gw.decide(&receipt, &action, "sess-1", 1);
        assert_eq!(d, Decision::Allow, "expected ALLOW");
    }

    // -------------------------------------------------------------
    // 2. Threat 1 (original design memo): chain-tampered receipt (a step's
    //    tool output bytes flipped after generation) must DENY via
    //    agent_trace verify's own chain check.
    // -------------------------------------------------------------
    #[test]
    fn deny_chain_tampered_receipt() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let text = fs::read_to_string(&receipt).unwrap();
        // Flip the first decoded token id on the step line -> the step's
        // decode-chain digest (and the overall trace-chain) no longer
        // matches a replay -> VERIFY FAIL. Token-id flip, not out=, since
        // the M7 tinybit model used by these fast tests does not always
        // emit a tool call for a given prompt (see README's "LOOKUP" note:
        // the M7 model never triggers CALC/LOOKUP for some prompts) --
        // flipping toks= is robust to that and still exercises the exact
        // same "well-formed mutant, rejected by chain/replay" family as the
        // round-4 tamper kit.
        let tampered: String = text
            .lines()
            .map(|l| {
                if let Some(rest) = l.strip_prefix("step ") {
                    if let Some(toks_pos) = rest.find("toks=") {
                        let before = &rest[..toks_pos + "toks=".len()];
                        let after = &rest[toks_pos + "toks=".len()..];
                        let (first_tok, tail) = after.split_once(',').expect("at least 2 toks");
                        let bumped: u64 = first_tok.parse::<u64>().unwrap() + 1;
                        return format!("step {before}{bumped},{tail}");
                    }
                }
                l.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        assert_ne!(text, tampered, "tamper must actually change the receipt");
        let tampered_path = dir.join("tampered.receipt");
        fs::write(&tampered_path, &tampered).unwrap();

        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);
        let action = unhex(&last_step_in_hex(&tampered_path)).unwrap();
        let (d, _head) = gw.decide(&tampered_path, &action, "sess-2", 1);
        match d {
            Decision::Deny(_) => {}
            Decision::Allow => panic!("chain-tampered receipt must DENY"),
        }
    }

    // -------------------------------------------------------------
    // 3. Threat 2: weight-digest not on the allowlist -> DENY, independent
    //    of the receipt otherwise verifying cleanly.
    // -------------------------------------------------------------
    #[test]
    fn deny_artifact_triple_not_on_allowlist() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        // Allowlist deliberately holds a DIFFERENT (bogus) triple only.
        let (mut gw, _key) =
            make_gateway(&dir, &[("0".repeat(64), "1".repeat(64), "2".repeat(64))]);
        let action = unhex(&last_step_in_hex(&receipt)).unwrap();
        let (d, _head) = gw.decide(&receipt, &action, "sess-3", 1);
        match d {
            Decision::Deny(reason) => assert!(reason.contains("allowlist")),
            Decision::Allow => panic!("off-allowlist triple must DENY"),
        }
    }

    // -------------------------------------------------------------
    // 3b. Threat 2, precise form: the allowlist's MODEL.SAF entry is a
    //     single bit-flip of the real artifact's sha256 digest (not just
    //     an unrelated bogus triple) -- the receipt still references the
    //     genuine artifacts and verifies cleanly, but its declared MODEL.SAF
    //     hash cannot match a flipped-bit allowlist entry, so this must
    //     DENY on artifact-allowlist grounds alone. Mirrors the real-2B
    //     "one-bit-flipped MODEL.SAF" case exercised end-to-end in
    //     run_e2e_safe1c's DENY (iii), as a fast, self-contained unit test.
    // -------------------------------------------------------------
    #[test]
    fn deny_weights_digest_bitflip_not_on_allowlist() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (m, _e, _v) = artifacts();
        let (mh, eh, vh) = artifact_hexes();

        // Sanity: flipping a bit in a COPY of the artifact bytes must not
        // reproduce the real digest (otherwise this test proves nothing).
        let mut flipped_bytes = fs::read(&m).unwrap();
        flipped_bytes[0] ^= 0x01;
        let mh_flipped = hex(&sha256(&flipped_bytes));
        assert_ne!(mh, mh_flipped, "bit flip must change the digest");

        // Allowlist holds only the bit-flipped MODEL.SAF digest, alongside
        // the real EMBED.BIN/VOCAB.BIN digests, so this exercises exactly
        // the weights-digest field and not the whole triple.
        let (mut gw, _key) = make_gateway(&dir, &[(mh_flipped, eh, vh)]);
        let action = unhex(&last_step_in_hex(&receipt)).unwrap();
        let (d, _head) = gw.decide(&receipt, &action, "sess-3b", 1);
        match d {
            Decision::Deny(reason) => assert!(reason.contains("allowlist")),
            Decision::Allow => panic!("bit-flipped weights digest must DENY"),
        }
    }

    // -------------------------------------------------------------
    // 4. Threat 4: a WARNING-set-mismatch tamper (drop the step whose tool
    //    output supplied the next round's grounding text). Uses a K=2
    //    receipt so there is a "next round" to break grounding for.
    //    Approximated here by truncating the receipt to drop its last
    //    step line entirely, which is exactly the "dropped step" mutation
    //    class agent_trace.rs's own tamper corpus tests, and which
    //    changes the derived WARNING-set independent of the trace-chain
    //    digest check already covered by test 2.
    // -------------------------------------------------------------
    #[test]
    fn deny_dropped_step_receipt() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let text = fs::read_to_string(&receipt).unwrap();
        let tampered: String = text
            .lines()
            .filter(|l| !l.starts_with("step "))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let tampered_path = dir.join("dropped.receipt");
        fs::write(&tampered_path, &tampered).unwrap();

        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);
        // With no step line left, there is no last-step input to match
        // against, so this exercises the parse-error DENY path -- still a
        // correct DENY for "receipt does not cover this call".
        let (d, _head) = gw.decide(&tampered_path, b"CALC(6 + 7)", "sess-4", 1);
        match d {
            Decision::Deny(_) => {}
            Decision::Allow => panic!("dropped-step receipt must DENY"),
        }
    }

    // -------------------------------------------------------------
    // 5. New rule (a): verify-execute binding. A fully valid, allowlisted
    //    receipt but the caller's PROPOSED action differs from what the
    //    receipt's last step actually did -- the confused-deputy case the
    //    critic's item #1 named. Must DENY even though `agent_trace
    //    verify` alone would say PASS on this receipt.
    // -------------------------------------------------------------
    #[test]
    fn deny_action_mismatch_confused_deputy() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);
        // Different action bytes than the receipt's last-step in=.
        let (d, _head) = gw.decide(&receipt, b"CALC(999 * 999)", "sess-5", 1);
        match d {
            Decision::Deny(reason) => assert!(reason.contains("verify-execute binding")),
            Decision::Allow => panic!("mismatched action must DENY (confused deputy)"),
        }
    }

    // -------------------------------------------------------------
    // 6. New rule (b): freshness / replay. The exact same
    //    (session, counter, action-hash) triple submitted twice: first
    //    ALLOW, second must DENY even though the receipt/action are
    //    byte-identical and would otherwise verify. This also covers the
    //    original design's threat-3 row ("replayed receipt").
    // -------------------------------------------------------------
    #[test]
    fn deny_replayed_receipt_same_session_counter() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);
        let action = unhex(&last_step_in_hex(&receipt)).unwrap();

        let (d1, _) = gw.decide(&receipt, &action, "sess-6", 7);
        assert_eq!(d1, Decision::Allow, "first submission must ALLOW");

        let (d2, _) = gw.decide(&receipt, &action, "sess-6", 7);
        match d2 {
            Decision::Deny(reason) => assert!(reason.contains("freshness")),
            Decision::Allow => panic!("replayed (session, counter) must DENY"),
        }
    }

    // -------------------------------------------------------------
    // 7. New rule (b), positive: a DIFFERENT counter in the same session,
    //    for a genuinely different receipt/action, must still ALLOW --
    //    freshness must not become an accidental global lockout.
    // -------------------------------------------------------------
    #[test]
    fn allow_same_session_new_counter_new_action() {
        let dir = workdir();
        let r1 = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let r2 = gen_receipt(&dir, "Q: 10 + 10\nA: CALC(10 + 10).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);

        let a1 = unhex(&last_step_in_hex(&r1)).unwrap();
        let (d1, _) = gw.decide(&r1, &a1, "sess-7", 1);
        assert_eq!(d1, Decision::Allow);

        let a2 = unhex(&last_step_in_hex(&r2)).unwrap();
        let (d2, _) = gw.decide(&r2, &a2, "sess-7", 2);
        assert_eq!(d2, Decision::Allow, "new counter + new action must ALLOW");
    }

    // -------------------------------------------------------------
    // 8. New rule (c): allowlist integrity. A hand-edited allowlist (an
    //    extra triple appended after signing, without re-signing) must be
    //    rejected at LOAD time -- gateway construction itself fails, so
    //    nothing downstream can ever ALLOW against it.
    // -------------------------------------------------------------
    #[test]
    fn deny_tampered_unsigned_allowlist_fails_to_load() {
        let dir = workdir();
        let (mh, eh, vh) = artifact_hexes();
        let key = b"test-hmac-key-not-for-prod".to_vec();
        let path = write_signed_allowlist(&dir, &[(mh.clone(), eh.clone(), vh.clone())], &key);
        // Tamper: append a second, unsigned-for triple after the fact.
        let mut text = fs::read_to_string(&path).unwrap();
        text = text.replacen(
            &format!("{mh} {eh} {vh}\n"),
            &format!(
                "{mh} {eh} {vh}\n{} {} {}\n",
                "a".repeat(64),
                "b".repeat(64),
                "c".repeat(64)
            ),
            1,
        );
        fs::write(&path, text).unwrap();

        let result = Gateway::new(GatewayConfig {
            allowlist_path: path,
            allowlist_key: key,
            anchor_every: 2,
            anchor_path: None,
            agent_trace_bin: agent_trace_bin(),
            model_path: artifacts().0,
            embed_path: artifacts().1,
            vocab_path: artifacts().2,
            table_path: None,
            strict: false,
            verify_timeout: None,
        });
        match result {
            Err(GatewayInitError::Allowlist(AllowlistError::BadSignature)) => {}
            Err(e) => panic!("expected BadSignature, got {e:?}"),
            Ok(_) => panic!("expected BadSignature, got Ok(Gateway)"),
        }
    }

    // -------------------------------------------------------------
    // 9. New rule (c), positive: correctly signed allowlist loads and its
    //    entries do permit the matching triple through the other checks.
    // -------------------------------------------------------------
    #[test]
    fn allow_correctly_signed_allowlist_permits_matching_triple() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);
        let action = unhex(&last_step_in_hex(&receipt)).unwrap();
        let (d, _head) = gw.decide(&receipt, &action, "sess-9", 1);
        assert_eq!(d, Decision::Allow);
    }

    // -------------------------------------------------------------
    // Integration: replay of the real "live20" receipt corpus (tools-1
    // live half, generated on the real 2B model, see
    // state/reports/2026-09-12-tools1-live20.md in claudius-maximus and
    // demo/agent-trace/live20/ on branch cm/tools1-live). Fixtures are a
    // verbatim copy of those 20 receipts under tests/fixtures/live20/ (not
    // fabricated here). Each receipt's own last-step `in=` bytes are used
    // as the proposed action, so rule (a) always matches by construction
    // and this test exercises `agent_trace verify` lenient vs strict
    // policy end-to-end through the gateway, not the binding rule.
    // -------------------------------------------------------------
    fn live20_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/live20")
    }

    fn real_artifacts() -> (PathBuf, PathBuf, PathBuf) {
        let a = PathBuf::from("/home/cm/aefinity-artifacts/bitnet2b-2b-artifacts");
        (
            a.join("aegis_pruned_model.cis.safetensors"),
            a.join("embed.bin"),
            a.join("vocab.bin"),
        )
    }

    /// Which table (if any) a given live20 episode id was generated
    /// against, per demo/agent-trace/live20/gen20.sh: calc_* use no
    /// table; lookup_*/fileread_*/mixed_* use tables/demo.tsv; chain_*
    /// use tables/chain.tsv.
    fn table_for(id: &str) -> Option<PathBuf> {
        let tables = repo_root().join("demo/agent-trace/tables");
        if id.starts_with("chain_") {
            Some(tables.join("chain.tsv"))
        } else if id.starts_with("lookup_")
            || id.starts_with("fileread_")
            || id.starts_with("mixed_")
        {
            Some(tables.join("demo.tsv"))
        } else {
            None
        }
    }

    fn live20_ids() -> Vec<String> {
        let mut ids: Vec<String> = fs::read_dir(live20_dir())
            .expect("live20 fixtures present")
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                name.strip_suffix(".receipt").map(|s| s.to_string())
            })
            .collect();
        ids.sort();
        ids
    }

    /// Run the full live20 corpus through a fresh Gateway per receipt
    /// (table differs per episode) built against the REAL 2B artifacts,
    /// in either lenient or strict mode. Returns (allow_count, total,
    /// per-id decisions) for the caller to assert on.
    fn run_live20(strict: bool) -> (usize, usize, Vec<(String, Decision)>) {
        if !real_artifacts().0.exists() {
            eprintln!(
                "SKIP live20 integration replay: real 2B artifacts not present at {}",
                real_artifacts().0.display()
            );
            return (0, 0, Vec::new());
        }
        let dir = workdir();
        let (m, e, v) = real_artifacts();
        let mh = hex(&sha256(&fs::read(&m).unwrap()));
        let eh = hex(&sha256(&fs::read(&e).unwrap()));
        let vh = hex(&sha256(&fs::read(&v).unwrap()));
        let key = b"live20-test-hmac-key-not-for-prod".to_vec();
        let allow_path = write_signed_allowlist(&dir, &[(mh, eh, vh)], &key);
        let bin = agent_trace_bin();

        let ids = live20_ids();
        let total = ids.len();
        let mut allow_count = 0usize;
        let mut decisions = Vec::new();
        for id in ids {
            let receipt = live20_dir().join(format!("{id}.receipt"));
            let mut gw = Gateway::new(GatewayConfig {
                allowlist_path: allow_path.clone(),
                allowlist_key: key.clone(),
                anchor_every: 5,
                anchor_path: None,
                agent_trace_bin: bin.clone(),
                model_path: m.clone(),
                embed_path: e.clone(),
                vocab_path: v.clone(),
                table_path: table_for(&id),
                strict,
                verify_timeout: None,
            })
            .expect("gateway init against real live20 artifacts/allowlist");
            let action = unhex(&last_step_in_hex(&receipt)).unwrap();
            let (d, _head) = gw.decide(&receipt, &action, "live20", 1);
            if d == Decision::Allow {
                allow_count += 1;
            }
            decisions.push((id, d));
        }
        (allow_count, total, decisions)
    }

    #[test]
    fn live20_lenient_allow_count() {
        let (allow, total, decisions) = run_live20(false);
        if total == 0 {
            return; // fixtures/artifacts unavailable in this environment; see eprintln above
        }
        assert_eq!(total, 20, "expected 20 live20 fixture receipts");
        for (id, d) in &decisions {
            if let Decision::Deny(reason) = d {
                eprintln!("live20 lenient DENY {id}: {reason}");
            }
        }
        assert_eq!(allow, 20, "expected ALLOW 20/20 in lenient mode");
    }

    #[test]
    fn live20_strict_allow_count() {
        let (allow, total, decisions) = run_live20(true);
        if total == 0 {
            return;
        }
        assert_eq!(total, 20, "expected 20 live20 fixture receipts");
        for (id, d) in &decisions {
            eprintln!("live20 strict {id}: {d:?}");
        }
        // UPDATED 2026-09-13 (safe-1c e2e run, box1): this test was never
        // actually run to completion before now (see gateway's own module
        // doc / the safe-1c task brief); running it end to end against the
        // real 2B artifacts gives ALLOW 17/20, DENY 3/20
        // (chain_fileread_02, mixed_lookup_calc_01, mixed_lookup_calc_02),
        // NOT the 13/20 this assertion previously assumed. The 13/20
        // figure came from a DIFFERENT (Python, check_verbatim.py)
        // implementation of "ungrounded argument" in
        // state/reports/2026-09-13-tools1c-grounding-box2.md; the two
        // implementations disagree on 4 receipts' groundedness. That
        // disagreement is itself a real finding, not a bug in this test —
        // see state/reports/2026-09-13-safe1c-e2e-box1.md (claudius-maximus
        // repo) for the id-level comparison and a NEEDS item to reconcile
        // the two "verbatim" definitions. Asserting the actually-observed
        // value here (rather than re-forcing 13) keeps this test honest
        // about what the Rust-native strict policy does today.
        eprintln!("live20 strict allow count: {allow}/{total}");
        assert_eq!(
            allow, 17,
            "expected ALLOW 17/20 in strict mode (see eprintln above for actual count/per-id if this fails; if this regresses, that's a real behavior change worth investigating, not just re-baselining)"
        );
    }

    // -------------------------------------------------------------
    // Sanity: the log is actually hash-chained (each entry depends on the
    // previous digest) -- rewriting decision N's outcome without
    // recomputing N+1..end would break the chain, which is the whole
    // point of chaining the decision log per the original design memo.
    // -------------------------------------------------------------
    #[test]
    fn log_head_changes_with_each_decision_and_is_deterministic_given_history() {
        let dir = workdir();
        let r1 = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let (mut gw, _key) = make_gateway(&dir, &[(mh, eh, vh)]);
        let a1 = unhex(&last_step_in_hex(&r1)).unwrap();
        let (_d, head1) = gw.decide(&r1, &a1, "sess-10", 1);
        // A DENY (replay of the same triple) advances the chain again to a
        // DIFFERENT head, proving the log records DENY decisions too, not
        // just ALLOWs.
        let (_d2, head2) = gw.decide(&r1, &a1, "sess-10", 1);
        assert_ne!(
            head1, head2,
            "log head must change on every decision, including DENY"
        );
    }

    // ===============================================================
    // SAFE-7: async serve (PENDING/POLL) tests.
    //
    // These exercise the ACTUAL background-thread PENDING->POLL->resolved
    // path over a real unix socket, with a real (short) time gap, not just
    // the data structures in isolation: `slow_agent_trace_wrapper` below
    // wraps the real `agent_trace` release binary in a shell script that
    // sleeps first and then execs it, so `agent_trace verify` genuinely
    // takes a few seconds here (standing in for the 6-7+ minute real-2B
    // case that motivated SAFE-7, without needing a real 2B model in this
    // fast unit-test tier). No wall-clock numbers appear in the SAFE-7
    // report or commit message per program rules; the sleep durations
    // below are test plumbing only.
    // ===============================================================

    fn slow_agent_trace_wrapper(dir: &Path, real_bin: &Path, sleep_secs: u64) -> PathBuf {
        let path = dir.join(format!("slow_agent_trace_{sleep_secs}.sh"));
        let script = format!(
            "#!/bin/sh\nsleep {sleep_secs}\nexec \"{}\" \"$@\"\n",
            real_bin.display()
        );
        fs::write(&path, script).unwrap();
        let mut perms = fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&path, perms).unwrap();
        path
    }

    fn make_gateway_with_bin(
        dir: &Path,
        allowlist_triples: &[(String, String, String)],
        bin: PathBuf,
    ) -> (Gateway, Vec<u8>) {
        let (m, e, v) = artifacts();
        let key = b"test-hmac-key-not-for-prod".to_vec();
        let allow_path = write_signed_allowlist(dir, allowlist_triples, &key);
        let gw = Gateway::new(GatewayConfig {
            allowlist_path: allow_path,
            allowlist_key: key.clone(),
            anchor_every: 2,
            anchor_path: Some(dir.join("anchor.log")),
            agent_trace_bin: bin,
            model_path: m,
            embed_path: e,
            vocab_path: v,
            table_path: None,
            strict: false,
            verify_timeout: None,
        })
        .expect("gateway init");
        (gw, key)
    }

    /// Bind a real unix socket, spin up `verify_workers` background verify
    /// threads and one accept-loop thread, mirroring `run_serve`'s own
    /// wiring but driven directly from the test (no subprocess, no CLI
    /// arg parsing) so tests can exercise `handle_serve_conn_async` /
    /// `verify_worker_loop` exactly as `serve` uses them in production.
    fn spawn_test_serve(
        gw: Gateway,
        verify_workers: usize,
        dir: &Path,
    ) -> (PathBuf, std::sync::Arc<SharedServe>) {
        let socket_path = dir.join("test.sock");
        let _ = fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind test socket");
        let cap_key = b"test-cap-key-not-for-prod".to_vec();
        let (job_tx, job_rx) = std::sync::mpsc::channel::<VerifyJob>();
        let job_rx = std::sync::Arc::new(std::sync::Mutex::new(job_rx));
        let shared = std::sync::Arc::new(SharedServe {
            gw: std::sync::Mutex::new(gw),
            tickets: std::sync::Mutex::new(std::collections::HashMap::new()),
            job_tx,
            next_ticket: std::sync::atomic::AtomicU64::new(1),
            cap_key,
            cap_ttl: 60,
        });
        for _ in 0..verify_workers {
            let rx = job_rx.clone();
            let s = shared.clone();
            std::thread::spawn(move || verify_worker_loop(rx, s));
        }
        let accept_shared = shared.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                if let Ok(mut stream) = conn {
                    handle_serve_conn_async(&mut stream, &accept_shared);
                }
            }
        });
        (socket_path, shared)
    }

    fn send_request(socket: &Path, lines: &[String]) -> String {
        let mut stream = UnixStream::connect(socket).expect("connect test socket");
        for l in lines {
            writeln!(stream, "{l}").unwrap();
        }
        writeln!(stream).unwrap(); // blank-line terminator
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        line.trim_end().to_string()
    }

    fn decide_lines(receipt: &Path, action_hex: &str, session: &str, counter: u64) -> Vec<String> {
        vec![
            format!("RECEIPT {}", receipt.display()),
            format!("ACTION {action_hex}"),
            format!("SESSION {session}"),
            format!("COUNTER {counter}"),
        ]
    }

    fn poll_lines(ticket: &str) -> Vec<String> {
        vec![format!("POLL ticket={ticket}")]
    }

    // -------------------------------------------------------------
    // (a)+(c): a slow verify goes PENDING, an immediate POLL still reports
    // PENDING, and a POLL after the (stubbed) slow verify has actually
    // finished reports the resolved ALLOW.
    // -------------------------------------------------------------
    #[test]
    fn async_slow_verify_pending_then_poll_resolves_to_allow() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let real_bin = agent_trace_bin();
        let slow_bin = slow_agent_trace_wrapper(&dir, &real_bin, 2);
        let (gw, _key) = make_gateway_with_bin(&dir, &[(mh, eh, vh)], slow_bin);
        let (socket, _shared) = spawn_test_serve(gw, 1, &dir);

        let action_hex = last_step_in_hex(&receipt);
        let reply = send_request(&socket, &decide_lines(&receipt, &action_hex, "async-1", 1));
        assert!(
            reply.starts_with("PENDING ticket="),
            "expected PENDING, got {reply:?}"
        );
        let ticket = reply.strip_prefix("PENDING ticket=").unwrap().to_string();

        // Poll immediately -- the stub is still sleeping, so the worker
        // cannot have finished yet.
        let poll1 = send_request(&socket, &poll_lines(&ticket));
        assert_eq!(poll1, "PENDING", "expected still-PENDING immediately after enqueue");

        // Wait past the stub's sleep, then poll again -- must be resolved.
        std::thread::sleep(Duration::from_secs(20));
        let poll2 = send_request(&socket, &poll_lines(&ticket));
        assert!(
            poll2.starts_with("ALLOW idx="),
            "expected resolved ALLOW after verify finished, got {poll2:?}"
        );
    }

    // -------------------------------------------------------------
    // (b): a second request for the SAME (session, counter) while the
    // first is still PENDING must DENY immediately (freshness), not queue
    // behind the in-flight verify.
    // -------------------------------------------------------------
    #[test]
    fn async_duplicate_session_counter_denies_immediately_while_first_pending() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let (mh, eh, vh) = artifact_hexes();
        let real_bin = agent_trace_bin();
        let slow_bin = slow_agent_trace_wrapper(&dir, &real_bin, 4);
        let (gw, _key) = make_gateway_with_bin(&dir, &[(mh, eh, vh)], slow_bin);
        let (socket, _shared) = spawn_test_serve(gw, 1, &dir);

        let action_hex = last_step_in_hex(&receipt);
        let reply1 = send_request(&socket, &decide_lines(&receipt, &action_hex, "async-dup", 9));
        assert!(reply1.starts_with("PENDING ticket="), "expected PENDING, got {reply1:?}");

        // Same (session, counter): must DENY immediately, well before the
        // stub's sleep (still in flight in the worker) could finish -- the
        // single-threaded accept loop serializes these two connections,
        // and this second one never touches the worker/subprocess at all.
        let reply2 = send_request(&socket, &decide_lines(&receipt, &action_hex, "async-dup", 9));
        assert!(reply2.starts_with("DENY"), "expected DENY, got {reply2:?}");
        assert!(
            reply2.contains("freshness"),
            "expected freshness DENY, got {reply2:?}"
        );
    }

    // -------------------------------------------------------------
    // POLL on a ticket this process never issued (e.g. after a restart)
    // must DENY, not hang as PENDING forever.
    // -------------------------------------------------------------
    #[test]
    fn async_poll_unknown_ticket_denies() {
        let dir = workdir();
        let (mh, eh, vh) = artifact_hexes();
        let real_bin = agent_trace_bin();
        let (gw, _key) = make_gateway_with_bin(&dir, &[(mh, eh, vh)], real_bin);
        let (socket, _shared) = spawn_test_serve(gw, 1, &dir);
        let reply = send_request(&socket, &poll_lines("T-does-not-exist"));
        assert_eq!(reply, "DENY unknown ticket");
    }

    // -------------------------------------------------------------
    // (d): async must not change WHAT gets decided, only WHEN the caller
    // learns it. Run the SAME tampered (chain-broken -> deterministic
    // VERIFY FAIL) receipt/action through synchronous `decide()` and
    // through the async PENDING->POLL path, and assert the exact DENY
    // reason string matches byte-for-byte.
    // -------------------------------------------------------------
    #[test]
    fn async_final_deny_reason_matches_sync_decide_for_same_inputs() {
        let dir = workdir();
        let receipt = gen_receipt(&dir, "Q: 6 + 7\nA: CALC(6 + 7).\n", 1, 16);
        let text = fs::read_to_string(&receipt).unwrap();
        let tampered: String = text
            .lines()
            .map(|l| {
                if let Some(rest) = l.strip_prefix("step ") {
                    if let Some(toks_pos) = rest.find("toks=") {
                        let before = &rest[..toks_pos + "toks=".len()];
                        let after = &rest[toks_pos + "toks=".len()..];
                        let (first_tok, tail) = after.split_once(',').expect("at least 2 toks");
                        let bumped: u64 = first_tok.parse::<u64>().unwrap() + 1;
                        return format!("step {before}{bumped},{tail}");
                    }
                }
                l.to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let tampered_path = dir.join("tampered.receipt");
        fs::write(&tampered_path, &tampered).unwrap();

        let (mh, eh, vh) = artifact_hexes();
        let real_bin = agent_trace_bin();

        // Ground truth: synchronous decide() on a plain (no-sleep) Gateway.
        let (mut sync_gw, _k) = make_gateway_with_bin(
            &dir,
            &[(mh.clone(), eh.clone(), vh.clone())],
            real_bin.clone(),
        );
        let action = unhex(&last_step_in_hex(&tampered_path)).unwrap();
        let (sync_decision, _head) = sync_gw.decide(&tampered_path, &action, "sync-cmp", 1);
        let sync_reason = match sync_decision {
            Decision::Deny(r) => r,
            Decision::Allow => panic!("tampered receipt must DENY synchronously (test setup bug)"),
        };

        // Async: same receipt/action through the slow-wrapper + PENDING/POLL path.
        let slow_bin = slow_agent_trace_wrapper(&dir, &real_bin, 2);
        let (gw, _key) = make_gateway_with_bin(&dir, &[(mh, eh, vh)], slow_bin);
        let (socket, _shared) = spawn_test_serve(gw, 1, &dir);
        let action_hex = last_step_in_hex(&tampered_path);
        let reply = send_request(&socket, &decide_lines(&tampered_path, &action_hex, "async-cmp", 1));
        assert!(reply.starts_with("PENDING ticket="), "expected PENDING, got {reply:?}");
        let ticket = reply.strip_prefix("PENDING ticket=").unwrap().to_string();
        std::thread::sleep(Duration::from_secs(20));
        let final_reply = send_request(&socket, &poll_lines(&ticket));
        assert_eq!(
            final_reply,
            format!("DENY {sync_reason}"),
            "async final DENY must match sync decide() exactly"
        );
    }
}
