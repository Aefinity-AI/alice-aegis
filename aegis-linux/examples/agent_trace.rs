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
//! occurrence of `CALC(` or `LOOKUP(` in the step's newly decoded text
//! (never the prompt or earlier steps) and attempt to parse *only* that
//! occurrence per its own grammar. If it fails to parse, the step is
//! `no-tool` — the scanner never falls back to a later occurrence or the
//! other tool. Both tools may appear across one episode (different steps).
//!
//! Each step re-encodes its own growing prompt and decodes from position 0
//! with a fresh engine (no carried KV state across steps) — the simplest
//! thing that is unambiguously deterministic and cheap at this episode size
//! (K=3, N=16 by default).
//!
//!   agent_trace gen    <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <K> <N> ["prompt"] [--table <path>] > receipt
//!   agent_trace verify <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <receipt-file> [--table <path>]
//!
//! Rule A: prints no timing, ever. Rule B: the receipt carries a commit hash
//! and hostname (informational only — NOT folded into the trace chain, so a
//! receipt generated on one machine still verifies bit-for-bit on another).
//!
//! Table binding: when `--table` is given, the table file's sha256 and byte
//! length are folded into the trace genesis (see `trace_genesis`'s doc
//! comment for the exact fold order) and the receipt records a
//! `table-sha256 <64 hex>` header line. A table-less (v0) episode's genesis
//! fold is byte-identical to the pre-LOOKUP code path — no format bump.
//! `verify` recomputes the table's sha256 from the `--table` file it is
//! given and rejects a mismatch (or a missing `--table` when the receipt
//! declares one) as `VERIFY FAIL` before ever running the replay.

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;
use aegis_core::witness::{Sha256, WitnessChain, WitnessHeader, hex_lower, sha256};

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

/// Strict hex decode: even length, every byte pair a valid hex digit pair.
/// Used only on receipt-derived (hostile) fields in `verify`; `gen` never
/// parses hex, so its output path is unaffected.
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

/// Run the tool scanner over one step's decoded text. Deterministic: same
/// text and table in, same `ToolOutcome` out, always. `table` is `None` for
/// a table-less episode, in which case `LOOKUP(...)` is never recognized.
fn run_tool(decoded_text: &str, table: Option<&LookupTable>) -> ToolOutcome {
    match find_tool_call(decoded_text, table.is_some()) {
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
/// Fold order (exact): TRACE_DOMAIN, model_sha, embed_sha, vocab_sha,
/// k (BE u64), n (BE u64), prompt.len() (BE u64), prompt bytes, THEN — only
/// when `table` is `Some((table_sha, table_len))` — table_sha (32 raw
/// bytes) followed by table_len (BE u64). A table-less call folds none of
/// that trailing material, so its digest is byte-identical to the
/// pre-LOOKUP genesis fold (this is what keeps existing v0 receipts
/// verifying unchanged).
fn trace_genesis(
    model_sha: &[u8; 32],
    embed_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    k: u64,
    n: u64,
    prompt: &[u8],
    table: Option<(&[u8; 32], u64)>,
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
#[allow(clippy::too_many_arguments)]
fn decode_step(
    tensors: &SafeTensors,
    embed_bytes: &[u8],
    config: &ModelConfig,
    tokenizer: &AegisTokenizer,
    model_sha: &[u8; 32],
    embed_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    prompt: &str,
    n: usize,
) -> (Vec<u32>, [u8; 32]) {
    let pipeline = FullBitNetPipeline::new(tensors, embed_bytes, config).expect("build pipeline");
    let cis_model = CisModel::new(&pipeline, config).expect("CIS model conversion");
    let mut engine = CisEngine::new_with_mode(&cis_model, CisMode::FullInt);

    let prompt_ids = tokenizer.encode(prompt);
    assert!(!prompt_ids.is_empty(), "step prompt tokenized to nothing");
    assert!(
        prompt_ids.len() + n <= config.max_position_embeddings,
        "step prompt ({}) + N ({}) exceeds max_position_embeddings ({})",
        prompt_ids.len(),
        n,
        config.max_position_embeddings
    );

    let step_header = WitnessHeader {
        model_sha,
        embed_sha,
        vocab_sha,
        max_new: n as u64,
        prompt: prompt.as_bytes(),
    };
    let mut chain = WitnessChain::from_header(&step_header);

    let mut pos = 0usize;
    for &t in &prompt_ids {
        engine.forward_step_int(t, pos);
        pos += 1;
    }

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
    (generated, chain.digest())
}

/// Replay the whole K-step episode from the header inputs. Shared by gen
/// and verify — a verifier that calls this and gets the same
/// `EpisodeReplay` as the receipt claims has replayed the episode
/// bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn replay_episode(
    model_bytes: &[u8],
    embed_bytes: &[u8],
    vocab_bytes: &[u8],
    model_sha: &[u8; 32],
    embed_sha: &[u8; 32],
    vocab_sha: &[u8; 32],
    initial_prompt: &str,
    k: usize,
    n: usize,
    table: Option<&LookupTable>,
) -> EpisodeReplay {
    let tensors = SafeTensors::deserialize(model_bytes).expect("parse MODEL.SAF");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("read __metadata__")
        .expect("MODEL.SAF carries no aegis_config — repack in the forge");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
    let tokenizer = AegisTokenizer::new(vocab_bytes).expect("parse VOCAB.BIN");

    let mut prompt = initial_prompt.to_string();
    let mut trace_chain = trace_genesis(
        model_sha,
        embed_sha,
        vocab_sha,
        k as u64,
        n as u64,
        initial_prompt.as_bytes(),
        table.map(|t| (&t.sha256, t.len)),
    );
    let mut steps = Vec::with_capacity(k);

    for step_idx in 0..k {
        let (toks, decode_chain) = decode_step(
            &tensors,
            embed_bytes,
            &config,
            &tokenizer,
            model_sha,
            embed_sha,
            vocab_sha,
            &prompt,
            n,
        );
        let decoded_text = tokenizer.decode(&toks);
        let outcome = run_tool(&decoded_text, table);

        trace_chain = trace_fold_step(
            trace_chain,
            step_idx as u64,
            &decode_chain,
            outcome.name.as_bytes(),
            &outcome.input,
            &outcome.output,
        );

        prompt = prompt + &decoded_text + &tool_result_text(&outcome);

        steps.push(StepRecord {
            toks,
            tool_name: outcome.name,
            tool_input: outcome.input,
            tool_output: outcome.output,
            decode_chain,
        });
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
/// malformed receipt fails cleanly instead of panicking. Reads the config
/// the same way `replay_episode` does (same trusted MODEL.SAF/VOCAB.BIN).
fn validate_receipt_header(
    model_bytes: &[u8],
    vocab_bytes: &[u8],
    prompt: &str,
    k: usize,
    n: usize,
) -> Result<(), String> {
    let tensors = SafeTensors::deserialize(model_bytes)?;
    let cfg_json = tensors
        .metadata_field("aegis_config")?
        .ok_or_else(|| "MODEL.SAF carries no aegis_config".to_string())?;
    let config = ModelConfig::from_json(&cfg_json)?;
    let tokenizer = AegisTokenizer::new(vocab_bytes)?;
    let prompt_ids = tokenizer.encode(prompt);
    check_header_bounds(k, n, prompt_ids.len(), config.max_position_embeddings)
}

// ---------------------------------------------------------------------
// Receipt I/O.
// ---------------------------------------------------------------------

fn commit_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
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

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let table_path = extract_flag(&mut args, "--table");
    if args.len() < 6 {
        eprintln!(
            "usage: agent_trace gen    <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <K> <N> [prompt] [--table <path>]"
        );
        eprintln!(
            "       agent_trace verify <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <receipt-file> [--table <path>]"
        );
        std::process::exit(2);
    }
    let mode = args[1].as_str();
    let model_bytes = std::fs::read(&args[2]).expect("read MODEL.SAF");
    let embed_bytes = std::fs::read(&args[3]).expect("read EMBED.BIN");
    let vocab_bytes = std::fs::read(&args[4]).expect("read VOCAB.BIN");
    let model_sha = sha256(&model_bytes);
    let embed_sha = sha256(&embed_bytes);
    let vocab_sha = sha256(&vocab_bytes);

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

            let r = replay_episode(
                &model_bytes,
                &embed_bytes,
                &vocab_bytes,
                &model_sha,
                &embed_sha,
                &vocab_sha,
                prompt,
                k,
                n,
                table.as_ref(),
            );

            println!("AEGIS-TRACE v0");
            println!("model {}", hex(&model_sha));
            println!("embed {}", hex(&embed_sha));
            println!("vocab {}", hex(&vocab_sha));
            println!("K {k}");
            println!("N {n}");
            if let Some(t) = &table {
                println!("table-sha256 {}", hex(&t.sha256));
            }
            println!("prompt-hex {}", hex(prompt.as_bytes()));
            println!("commit {}", commit_hash());
            println!("host {}", host_name());
            for (i, s) in r.steps.iter().enumerate() {
                let ids: Vec<String> = s.toks.iter().map(|t| t.to_string()).collect();
                println!(
                    "step {i}: toks={} tool={} in={} out={} decode-chain={}",
                    ids.join(","),
                    s.tool_name,
                    hex(&s.tool_input),
                    hex(&s.tool_output),
                    hex(&s.decode_chain)
                );
            }
            println!("trace-chain {}", hex(&r.trace_chain));
        }
        "verify" => {
            let wtext = std::fs::read_to_string(&args[5]).expect("read receipt");
            let mut w_model = String::new();
            let mut w_embed = String::new();
            let mut w_vocab = String::new();
            let mut w_k = 0usize;
            let mut w_n = 0usize;
            let mut w_prompt = String::new();
            let mut w_steps: Vec<(Vec<u32>, String, String, String, String)> = Vec::new();
            let mut w_trace_chain = String::new();
            let mut w_table_sha: Option<String> = None;

            for line in wtext.lines() {
                if let Some(rest) = line.strip_prefix("step ") {
                    // "IDX: toks=.. tool=.. in=.. out=.. decode-chain=.."
                    let (label, body) = rest.split_once(':').unwrap_or((rest, ""));
                    let position = w_steps.len();
                    if let Err(reason) = check_step_label(label, position) {
                        println!("FAIL structure: {reason}");
                        std::process::exit(1);
                    }
                    let rest = body.trim();
                    let mut toks = Vec::new();
                    let mut tool = String::new();
                    let mut input = String::new();
                    let mut output = String::new();
                    let mut dchain = String::new();
                    for field in rest.split_whitespace() {
                        if let Some(v) = field.strip_prefix("toks=") {
                            toks = v
                                .split(',')
                                .filter(|s| !s.is_empty())
                                .map(|s| s.parse().expect("token id"))
                                .collect();
                        } else if let Some(v) = field.strip_prefix("tool=") {
                            tool = v.to_string();
                        } else if let Some(v) = field.strip_prefix("in=") {
                            input = v.to_string();
                        } else if let Some(v) = field.strip_prefix("out=") {
                            output = v.to_string();
                        } else if let Some(v) = field.strip_prefix("decode-chain=") {
                            dchain = v.to_string();
                        }
                    }
                    w_steps.push((toks, tool, input, output, dchain));
                    continue;
                }
                let mut it = line.splitn(2, ' ');
                let (key, v) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                match key {
                    "model" => w_model = v.into(),
                    "embed" => w_embed = v.into(),
                    "vocab" => w_vocab = v.into(),
                    "K" => w_k = v.parse().expect("K"),
                    "N" => w_n = v.parse().expect("N"),
                    "prompt-hex" => {
                        let bytes = match unhex(v) {
                            Ok(b) => b,
                            Err(()) => {
                                println!("FAIL structure: malformed hex in prompt-hex");
                                std::process::exit(1);
                            }
                        };
                        w_prompt = match String::from_utf8(bytes) {
                            Ok(s) => s,
                            Err(_) => {
                                println!("FAIL structure: malformed hex in prompt-hex");
                                std::process::exit(1);
                            }
                        };
                    }
                    "trace-chain" => w_trace_chain = v.into(),
                    "table-sha256" => w_table_sha = Some(v.into()),
                    _ => {}
                }
            }

            let mut fail = false;
            for (name, local, claimed) in [
                ("MODEL", hex(&model_sha), &w_model),
                ("EMBED", hex(&embed_sha), &w_embed),
                ("VOCAB", hex(&vocab_sha), &w_vocab),
            ] {
                if &local != claimed {
                    println!(
                        "FAIL artifact: {name} hash mismatch (receipt {} vs local {})",
                        &claimed[..16.min(claimed.len())],
                        &local[..16]
                    );
                    fail = true;
                }
            }
            if fail {
                std::process::exit(1);
            }

            if w_steps.len() != w_k {
                println!(
                    "FAIL structure: receipt claims K={} but has {} step lines",
                    w_k,
                    w_steps.len()
                );
                std::process::exit(1);
            }

            if let Err(reason) =
                validate_receipt_header(&model_bytes, &vocab_bytes, &w_prompt, w_k, w_n)
            {
                println!("FAIL structure: {reason}");
                std::process::exit(1);
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
                                &claimed[..16.min(claimed.len())]
                            );
                            std::process::exit(1);
                        }
                    };
                    let bytes = std::fs::read(path).unwrap_or_else(|e| {
                        println!("VERIFY FAIL — could not read --table {path}: {e}");
                        std::process::exit(1);
                    });
                    let local_sha = hex(&sha256(&bytes));
                    if &local_sha != claimed {
                        println!(
                            "FAIL artifact: TABLE hash mismatch (receipt {} vs local {})",
                            &claimed[..16.min(claimed.len())],
                            &local_sha[..16]
                        );
                        std::process::exit(1);
                    }
                    match parse_table(&bytes) {
                        Ok(t) => Some(t),
                        Err(reason) => {
                            println!("FAIL structure: bad --table: {reason}");
                            std::process::exit(1);
                        }
                    }
                }
                None => None,
            };

            let r = replay_episode(
                &model_bytes,
                &embed_bytes,
                &vocab_bytes,
                &model_sha,
                &embed_sha,
                &vocab_sha,
                &w_prompt,
                w_k,
                w_n,
                table.as_ref(),
            );

            let local_trace_chain = hex(&r.trace_chain);
            println!(
                "receipt trace-chain {}",
                &w_trace_chain[..16.min(w_trace_chain.len())]
            );
            println!(
                "local   trace-chain {}",
                &local_trace_chain[..16.min(local_trace_chain.len())]
            );

            if r.steps.len() != w_steps.len() {
                println!(
                    "VERIFY FAIL — replay produced {} steps, receipt has {}",
                    r.steps.len(),
                    w_steps.len()
                );
                std::process::exit(1);
            }

            let mut mismatch = false;
            for (i, (local, (w_toks, w_tool, w_in, w_out, w_dchain))) in
                r.steps.iter().zip(w_steps.iter()).enumerate()
            {
                let local_toks_match = &local.toks == w_toks;
                let local_tool_match = local.tool_name == w_tool;
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
            }

            if !mismatch && local_trace_chain == w_trace_chain {
                println!(
                    "VERIFY PASS — replay reproduced {} steps and the full trace chain bit-for-bit",
                    r.steps.len()
                );
            } else {
                println!("VERIFY FAIL — replay diverged from the receipt");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
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
        let o = run_tool("plain text", None);
        assert_eq!(o.name, "no-tool");
        assert!(o.input.is_empty());
        assert!(o.output.is_empty());
    }

    #[test]
    fn run_tool_success_is_calc() {
        let o = run_tool("prefix CALC(2 + 2) suffix", None);
        assert_eq!(o.name, "calc");
        assert_eq!(o.input, b"CALC(2 + 2)");
        assert_eq!(o.output, b"4");
    }

    #[test]
    fn run_tool_error_is_calc_error() {
        let o = run_tool("CALC(9 / 0)", None);
        assert_eq!(o.name, "calc-error");
        assert_eq!(o.output, b"div-by-zero");
    }

    // --- trace chain: deterministic, sensitive to every folded field ---

    fn base_digest() -> [u8; 32] {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
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
        let g = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 1, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_decode_chain() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[8u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_tool_name() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[9u8; 32], b"no-tool", b"CALC(1 + 1)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_tool_input() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 2)", b"2");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_fold_changes_with_tool_output() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
        let d0 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"2");
        let d1 = trace_fold_step(g, 0, &[9u8; 32], b"calc", b"CALC(1 + 1)", b"3");
        assert_ne!(d0, d1);
    }

    #[test]
    fn trace_genesis_changes_with_prompt_or_k_n() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g0 = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
        let g1 = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hellp", None);
        let g2 = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 4, 16, b"hello", None);
        let g3 = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 17, b"hello", None);
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
        let o = run_tool("CALC(2 + 2)", None);
        assert_eq!(tool_result_text(&o), "\nTOOL[calc]=4\n");
        let o2 = run_tool("no call", None);
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
        let o = run_tool("please LOOKUP(P-100) now", Some(&t));
        assert_eq!(o.name, "lookup");
        assert_eq!(o.input, b"LOOKUP(P-100)");
        assert_eq!(o.output, b"Gasket, O-ring, fuel line");
    }

    #[test]
    fn lookup_miss_returns_none_literal() {
        let t = parse_table(&demo_table_bytes()).unwrap();
        let o = run_tool("LOOKUP(P-999)", Some(&t));
        assert_eq!(o.name, "lookup");
        assert_eq!(o.output, b"NONE");
    }

    #[test]
    fn lookup_without_table_is_not_scanned() {
        // No table present: LOOKUP( text is inert, exactly like any other
        // plain text — this is what keeps a table-less episode identical to
        // the pre-LOOKUP behavior.
        let o = run_tool("LOOKUP(P-100)", None);
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
        let o = run_tool("first LOOKUP(P-100) then CALC(1 + 1)", Some(&t));
        assert_eq!(o.name, "lookup");
        let o2 = run_tool("first CALC(1 + 1) then LOOKUP(P-100)", Some(&t));
        assert_eq!(o2.name, "calc");
    }

    #[test]
    fn scanner_does_not_fall_back_when_earliest_call_fails_to_parse() {
        let t = parse_table(&demo_table_bytes()).unwrap();
        // Earliest is an invalid LOOKUP( — scanner must not fall back to the
        // later, valid CALC(...).
        let o = run_tool("LOOKUP(bad key) then CALC(1 + 1)", Some(&t));
        assert_eq!(o.name, "no-tool");
    }

    // --- genesis changes with the table ---

    #[test]
    fn trace_genesis_changes_with_table() {
        let model_sha = [1u8; 32];
        let embed_sha = [2u8; 32];
        let vocab_sha = [3u8; 32];
        let g_none = trace_genesis(&model_sha, &embed_sha, &vocab_sha, 3, 16, b"hello", None);
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
        );
        let g_b = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            Some((&table_sha_b, 42)),
        );
        let g_len = trace_genesis(
            &model_sha,
            &embed_sha,
            &vocab_sha,
            3,
            16,
            b"hello",
            Some((&table_sha_a, 43)),
        );
        assert_ne!(g_none, g_a, "table-less genesis must differ from table-bound genesis");
        assert_ne!(g_a, g_b, "genesis must be sensitive to table sha256");
        assert_ne!(g_a, g_len, "genesis must be sensitive to table length");
    }
}
