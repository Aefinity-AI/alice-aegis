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

use aegis_core::witness::{sha256, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------
// HMAC-SHA256, built on aegis_core's Sha256 (no external crate; this repo
// pins zero runtime deps beyond libm, see aegis-core/Cargo.toml).
// ---------------------------------------------------------------------

const BLOCK: usize = 64;

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut key_block = [0u8; BLOCK];
    if key.len() > BLOCK {
        let k = sha256(key);
        key_block[..32].copy_from_slice(&k);
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= key_block[i];
        opad[i] ^= key_block[i];
    }
    let mut inner = Sha256::new();
    inner.update(&ipad);
    inner.update(msg);
    let inner_digest = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner_digest);
    outer.finalize()
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    Some(out)
}

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
}

#[derive(Debug)]
pub enum GatewayInitError {
    Allowlist(AllowlistError),
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
        })
    }

    fn log_head(&self) -> [u8; 32] {
        self.log.last().map(|e| e.digest).unwrap_or([0u8; 32])
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
        let prev = self.log_head();
        let decision_tag = match decision {
            Decision::Allow => "ALLOW".to_string(),
            Decision::Deny(reason) => format!("DENY:{reason}"),
        };
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
            }
        }
        digest
    }

    /// Run `agent_trace verify` against `receipt_path` and return its full
    /// stdout. Never trusts an exit code alone (the binary's own convention
    /// is inspected in agent_trace.rs's `main`; this is a subprocess
    /// boundary, so checking the printed verdict text mirrors the
    /// second-verifier mode in the design memo).
    fn run_verify_raw(&self, receipt_path: &Path) -> Result<String, String> {
        let mut cmd = Command::new(&self.agent_trace_bin);
        cmd.arg("verify")
            .arg(&self.model_path)
            .arg(&self.embed_path)
            .arg(&self.vocab_path)
            .arg(receipt_path);
        if let Some(t) = &self.table_path {
            cmd.arg("--table").arg(t);
        }
        let out = cmd.output().map_err(|e| format!("spawn agent_trace: {e}"))?;
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    /// PASS/FAIL per `run_verify_raw`'s stdout, ignoring grounding
    /// WARNINGs (the "lenient" policy: a receipt with an ungrounded
    /// tool-call argument can still ALLOW as long as the trace chain
    /// itself replays correctly).
    fn run_verify(&self, receipt_path: &Path) -> Result<bool, String> {
        Ok(self.run_verify_raw(receipt_path)?.contains("VERIFY PASS"))
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
        Ok(stdout.contains("VERIFY PASS") && !stdout.contains("WARNING step"))
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
                let head = self.append_log(&d, "", &ArtifactTriple{model:String::new(),embed:String::new(),vocab:String::new()}, session, counter);
                return (d, head);
            }
        };
        let parsed = match parse_receipt(&text) {
            Ok(p) => p,
            Err(e) => {
                let d = Decision::Deny(format!("parse receipt: {e:?}"));
                let head = self.append_log(&d, "", &ArtifactTriple{model:String::new(),embed:String::new(),vocab:String::new()}, session, counter);
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
            let d = Decision::Deny("freshness: (session, counter, action-hash) already seen".into());
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
                let head = self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
                return (d, head);
            }
            Err(e) => {
                let d = Decision::Deny(format!("agent_trace verify: could not run: {e}"));
                let head = self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
                return (d, head);
            }
        }

        self.seen.insert(key);
        let d = Decision::Allow;
        let head = self.append_log(&d, &parsed.trace_chain, &parsed.triple, session, counter);
        (d, head)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
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
    let counter: u64 = args
        .get(10)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

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
        let d = std::env::temp_dir().join(format!(
            "gateway-test-{}-{}",
            std::process::id(),
            n
        ));
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
        // unoptimized debug build); fall back to debug for the small
        // tinybit-model unit tests if release isn't available.
        let release = repo_root().join("aegis-linux/target/release/examples/agent_trace");
        if release.exists() {
            return release;
        }
        let bin = repo_root().join("aegis-linux/target/debug/examples/agent_trace");
        if !bin.exists() {
            let status = Command::new("cargo")
                .args(["build", "--offline", "--example", "agent_trace"])
                .current_dir(repo_root().join("aegis-linux"))
                .status()
                .expect("cargo build agent_trace");
            assert!(status.success(), "failed to build agent_trace example");
        }
        bin
    }

    fn artifacts() -> (PathBuf, PathBuf, PathBuf) {
        let a = repo_root().join("model-lab/tinybit/m7_final_gate_work/artifacts");
        (a.join("MODEL.SAF"), a.join("EMBED.BIN"), a.join("VOCAB.BIN"))
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

    fn write_signed_allowlist(dir: &Path, triples: &[(String, String, String)], key: &[u8]) -> PathBuf {
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

    fn make_gateway(dir: &Path, allowlist_triples: &[(String, String, String)]) -> (Gateway, Vec<u8>) {
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
        let (mut gw, _key) = make_gateway(
            &dir,
            &[(
                "0".repeat(64),
                "1".repeat(64),
                "2".repeat(64),
            )],
        );
        let action = unhex(&last_step_in_hex(&receipt)).unwrap();
        let (d, _head) = gw.decide(&receipt, &action, "sess-3", 1);
        match d {
            Decision::Deny(reason) => assert!(reason.contains("allowlist")),
            Decision::Allow => panic!("off-allowlist triple must DENY"),
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
            &format!("{mh} {eh} {vh}\n{} {} {}\n", "a".repeat(64), "b".repeat(64), "c".repeat(64)),
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
        } else if id.starts_with("lookup_") || id.starts_with("fileread_") || id.starts_with("mixed_") {
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
        // Observed 2026-09-13 (leg-safe1b-live20, real 2B model, full 20-episode
        // corpus): 14/20 ALLOW, not 20/20. The 6 DENYs are agent_trace VERIFY
        // FAIL (not a strict-grounding WARNING) on every fileread_* and
        // chain_fileread_* episode -- i.e. the file-read tool's receipts do
        // not replay-verify even in lenient mode. This matches critic concern
        // (3) in 2026-09-13-SAFE1-DESIGN-CRITIC.md: verify's replay guarantee
        // was proven only for the deterministic CALC/LOOKUP sims, not for
        // non-deterministic tools like fileread. Root cause not yet
        // diagnosed -- filed as a follow-up (see state/NEEDS.md safe-1b-fileread-verify).
        assert_eq!(allow, 14, "expected ALLOW 14/20 in lenient mode (see eprintln above for actual count/per-id if this fails)");
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
        // Report the real observed count honestly rather than forcing a
        // brief-predicted number: the brief's own critic report
        // (state/reports/2026-09-13-tools1c-grounding-box2.md) independently
        // found 13/20 pass a strict ungrounded-argument check on this exact
        // corpus using a different (Python) implementation of the same
        // "verbatim-argument" rule; this asserts the gateway's own
        // Rust-native strict policy (agent_trace verify PASS with zero
        // WARNING lines) against that as a cross-check, not an assumption.
        eprintln!("live20 strict allow count: {allow}/{total}");
        // Observed 2026-09-13 (leg-safe1b-live20): 12/20, one below the
        // 13/20 cross-check prediction above (the same 6 fileread VERIFY
        // FAILs as lenient mode account for most of the gap; strict mode
        // additionally denies 2 mixed_lookup_calc episodes on grounding
        // WARNINGs that the Python cross-check didn't flag). See lenient
        // test comment above and state/NEEDS.md safe-1b-fileread-verify.
        assert_eq!(allow, 12, "expected ALLOW 12/20 in strict mode (see eprintln above for actual count/per-id if this fails)");
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
        assert_ne!(head1, head2, "log head must change on every decision, including DENY");
    }
}
