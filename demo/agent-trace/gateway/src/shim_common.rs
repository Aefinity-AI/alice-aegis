//! Shared plumbing for `src/bin/tool_shim_*.rs`: parse `--idx --cap --exp
//! --action-hash`, load/persist a per-shim consumed-index set (fsync'd
//! file, mirrors the gateway daemon's own freshness store per the SAFE-5
//! design), and print the standard `SHIM REFUSE: <reason>` line.
//!
//! Every shim independently recomputes the action hash from the actual
//! bytes it is about to execute and calls `capability::verify` with its
//! OWN consumed set (not the gateway's) — so a captured, still-valid
//! token cannot be replayed against the tool a second time even if it
//! somehow was not yet replayed against the gateway.

use crate::capability::{self, VerifyError};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct ShimArgs {
    pub idx: Option<u64>,
    pub cap: Option<String>,
    pub exp: Option<u64>,
    pub action_hash_hex: Option<String>,
    /// Everything else, in order, for the tool-specific payload.
    pub rest: Vec<String>,
}

/// Parse `--idx N --cap HEX --exp UNIX --action-hash HEX` out of argv,
/// leaving all other tokens (in order) in `rest`.
pub fn parse_shim_args(argv: &[String]) -> ShimArgs {
    let mut idx = None;
    let mut cap = None;
    let mut exp = None;
    let mut action_hash_hex = None;
    let mut rest = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--idx" => {
                idx = argv.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--cap" => {
                cap = argv.get(i + 1).cloned();
                i += 2;
            }
            "--exp" => {
                exp = argv.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "--action-hash" => {
                action_hash_hex = argv.get(i + 1).cloned();
                i += 2;
            }
            other => {
                rest.push(other.to_string());
                i += 1;
            }
        }
    }
    ShimArgs {
        idx,
        cap,
        exp,
        action_hash_hex,
        rest,
    }
}

pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Load this shim's own consumed-index set from `path` (one decimal index
/// per line; missing file = empty set).
pub fn load_consumed(path: &Path) -> HashSet<u64> {
    match fs::read_to_string(path) {
        Ok(text) => text.lines().filter_map(|l| l.trim().parse().ok()).collect(),
        Err(_) => HashSet::new(),
    }
}

/// Append `idx` to the consumed-index file and fsync it, so a crash right
/// after a successful verify cannot silently forget the index was spent.
pub fn persist_consumed(path: &Path, idx: u64) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            fs::create_dir_all(dir)?;
        }
    }
    let mut f: File = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{idx}")?;
    f.sync_all()
}

/// Refuse-and-exit helper: prints `SHIM REFUSE: <reason>` to stdout (so
/// escape-test harnesses can grep it without capturing stderr) and exits
/// with status 1. Never returns.
pub fn refuse(reason: &str) -> ! {
    println!("SHIM REFUSE: {reason}");
    std::process::exit(1);
}

/// Full capability check for one shim invocation: computes the action
/// hash of `action_bytes`, verifies it against `--idx/--cap/--exp`, checks
/// it against the on-disk consumed set at `consumed_path`, and on success
/// persists the newly-consumed index. Returns Ok(()) only if the caller
/// may proceed to execute the real action; otherwise calls `refuse` (which
/// exits the process) with the right reason string, covering: no token,
/// bad/forged token, expired token, replayed (already consumed) token,
/// and action-hash mismatch (the bytes about to be executed differ from
/// what the token was issued for).
///
/// `tool_tag` (SAFE-11) MUST be the caller's own FIXED identity (e.g.
/// `tool_shim_shell` always passes the literal `"shell"`) — never derived
/// from argv or any other caller-controlled input. This is what binds a
/// capability token to the specific tool it was approved for and closes
/// cross-shim token replay (a token minted for a
/// file-write ALLOW no longer verifies here if this shim's fixed tag
/// does not match the tag the gateway daemon actually issued the token
/// for).
pub fn check_and_consume(
    key: &[u8],
    args: &ShimArgs,
    action_bytes: &[u8],
    consumed_path: &Path,
    tool_tag: &str,
) {
    let (idx, cap, exp, claimed_hash_hex) =
        match (&args.idx, &args.cap, &args.exp, &args.action_hash_hex) {
            (Some(i), Some(c), Some(e), Some(h)) => (*i, c.clone(), *e, h.clone()),
            _ => refuse("no token"),
        };

    let real_hash = crate::action_hash(action_bytes);
    let real_hash_hex = crate::hex(&real_hash);
    if !claimed_hash_hex.eq_ignore_ascii_case(&real_hash_hex) {
        refuse("action hash mismatch");
    }

    let mut consumed = load_consumed(consumed_path);
    match capability::verify(
        key,
        idx,
        real_hash,
        exp,
        &cap,
        now_unix(),
        &mut consumed,
        tool_tag,
    ) {
        Ok(()) => {
            if let Err(e) = persist_consumed(consumed_path, idx) {
                refuse(&format!("could not persist consumed index: {e}"));
            }
        }
        Err(VerifyError::BadMac) => refuse("bad mac"),
        Err(VerifyError::Expired) => refuse("expired"),
        Err(VerifyError::AlreadyConsumed) => refuse("already consumed"),
        Err(VerifyError::NotYetValid) => refuse("not yet valid (clock skew out of bounds)"),
    }
}

/// Default consumed-set file path for a shim, namespaced by tool name, so
/// the three shims do not share (and cannot cross-invalidate) a token
/// index space. `$XDG_RUNTIME_DIR` if set, else a repo-relative fallback
/// for tests run outside a full user session.
pub fn default_consumed_path(tool: &str) -> PathBuf {
    let dir = std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    dir.join(format!("cm-gateway-shim-{tool}.consumed"))
}
