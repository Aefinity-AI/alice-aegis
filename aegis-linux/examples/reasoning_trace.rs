//! AEGIS-REASON v1 — a receipt-carrying self-check reasoning scaffold.
//!
//! `reasoning-1a` (OFFLINE half, box2). This is a **scaffold + synthetic-
//! transcript** deliverable only: no model runs here, no 2B generation.
//! `reasoning-1` (the live half, box1) is expected to produce real
//! draft/verify-step/final transcripts from the 2B model and feed them
//! through this same parser/verifier.
//!
//! ## Why a new format instead of reusing `AEGIS-TRACE`
//!
//! `agent_trace.rs`'s `AEGIS-TRACE` format binds a decode transcript (token
//! ids, tool in/out, per-step context/query digests) to a model + artifact
//! set. A self-check reasoning trace binds a DIFFERENT thing: a draft
//! answer, an independent recheck of that answer, and a final answer that
//! must be consistent with the recheck's verdict. Reusing `AEGIS-TRACE`'s
//! fields (`toks=`, `tool=`, `ctx=`, `q=`) would either be dishonest (there
//! is no tool call here) or force meaningless fields into every receipt.
//! Instead this format follows the SAME CONVENTIONS `agent_trace.rs`
//! established, deliberately:
//!
//! - one magic header line first (`AEGIS-REASON v1` here, `AEGIS-TRACE vN`
//!   there), a hash-chained header committing the episode's fixed inputs;
//! - one line per step, `step N: key=value key=value ...`, every
//!   whitespace-separated token a recognised `key=value` pair, no key
//!   repeated, no unknown key — the same strict-parse rule `agent_trace`'s
//!   `verify` enforces (a receipt that verifies is a canonical file, not
//!   one of many byte strings sharing a digest);
//! - a per-step hash chain folding the previous step's chain plus this
//!   step's fields, plus one final `trace-chain=` line folding everything —
//!   so any edited field, reordered step, or dropped step is detectable
//!   without re-running anything model-side;
//! - `verify` is the source of truth, independent of what a step CLAIMS
//!   about itself (see "verdict" below) — the whole point of a self-check
//!   scaffold is that the checker does not just trust the model's own
//!   "looks good to me".
//!
//! ## Trace shape: draft -> verify-step(+) -> final
//!
//! ```text
//! AEGIS-REASON v1
//! prompt-sha256=<sha256 of the prompt text, hex>
//! step 0: kind=draft answer=<token> text-sha256=<hex> chain=<hex>
//! step 1: kind=verify ref=0 recheck=<token> verdict=pass|contradict text-sha256=<hex> chain=<hex>
//! step 2: kind=final ref=1 answer=<token> text-sha256=<hex> chain=<hex>
//! trace-chain=<hex>
//! ```
//!
//! `answer=`/`recheck=` are bare tokens (`[A-Za-z0-9_+-]+`, no whitespace,
//! no `=`) — e.g. a numeric result or a short symbolic answer — never free
//! prose. Free-text rationale (the "because ..." the model actually wrote)
//! is represented ONLY by its `text-sha256=` digest, exactly the way
//! `agent_trace`'s `decode-chain=` stands in for the full token trajectory
//! without embedding it verbatim in the receipt line. A verifier that has
//! the full transcript (not just the receipt) can recompute `text-sha256`
//! over the exact prose bytes and compare; a receipt alone cannot be used
//! to forge rationale that reads differently from what was actually said.
//!
//! ### Prompt templates (informational — not parsed, not enforced)
//!
//! These are suggested prompts for whatever produces the transcript this
//! module parses (a model, in `reasoning-1`, or a human, in these tests).
//! The FORMAT does not depend on the exact wording — only on the model
//! (in the arithmetic sense) emitting a bare `answer=`/`recheck=` token
//! and, for a verify-step, a `verdict=` claim:
//!
//! - draft: "Answer the question. End your answer with a single line
//!   `ANSWER: <token>` with no other text on that line."
//! - verify-step: "Do not look at your previous answer's reasoning.
//!   Independently recompute the answer from scratch. End with
//!   `RECHECK: <token>` then a line `VERDICT: PASS` if RECHECK equals
//!   your original ANSWER, `VERDICT: CONTRADICT` if it does not."
//! - final: "Given the verify-step's verdict, if PASS restate the
//!   original answer; if CONTRADICT, revise your answer to trust the
//!   recheck. End with `ANSWER: <token>`."
//!
//! ### Step-marker representation
//!
//! In this module a "step marker" is the parsed `step N: kind=... ...`
//! line, not raw prose. The design above assumes an upstream extraction
//! step (regex over `ANSWER:`/`RECHECK:`/`VERDICT:` lines, out of scope
//! here) turns a model's free-text response into `answer=`/`recheck=`/
//! `verdict=` receipt fields plus a `text-sha256=` over the full response.
//! `reasoning-1` (box1, live) owns that extraction; this module owns
//! everything from "well-formed receipt text" onward.
//!
//! ### Parser rules
//!
//! - First non-blank line must be exactly `AEGIS-REASON v1`; a second such
//!   line anywhere is a `FAIL structure` (duplicate header).
//! - Second line must be `prompt-sha256=<64 lowercase hex>`.
//! - Every following line up to (not including) the trailer is a
//!   `step N:` line, N starting at 0 and increasing by exactly 1 each
//!   line — no gaps, no reordering, no duplicate index.
//! - A `step` line's fields, after `kind=`, depend on `kind`:
//!     - `draft`:  requires `answer=`, `text-sha256=`, `chain=`.
//!     - `verify`: requires `ref=`, `recheck=`, `verdict=`, `text-sha256=`, `chain=`.
//!     - `final`:  requires `ref=`, `answer=`, `text-sha256=`, `chain=`.
//!   Any other key, any missing required key, or any repeated key is a
//!   parse error (`FAIL structure`), not silently ignored.
//! - Last line must be `trace-chain=<64 lowercase hex>`.
//! - Sequencing (checked after parsing, before verifying chains):
//!     - step 0 must be `kind=draft`.
//!     - exactly one `kind=final` step, and it must be the LAST step.
//!     - at least one `kind=verify` step must exist before the final step.
//!     - every `ref=` must point at a strictly earlier step index.
//!     - a `verify` step's `ref=` must point at a `draft` step.
//!     - the `final` step's `ref=` must point at the LAST `verify` step.
//!
//! ### Verifier rules: pass vs contradict
//!
//! The receipt's own `verdict=` field is a CLAIM, never trusted directly.
//! `verify` recomputes it independently from structured fields and treats
//! a mismatch as a finding in its own right (a lying verify-step is worse
//! than a contradicting one — it defeats the entire point of the scaffold):
//!
//! - `computed_verdict(verify step) = pass`   if `recheck` == the referenced
//!   draft step's `answer` (byte-exact token comparison);
//!   `= contradict` otherwise.
//! - If `computed_verdict != verdict` (the claimed field): `VERDICT LIE
//!   step N` — the step's self-report does not match its own recheck vs.
//!   draft comparison.
//! - If `computed_verdict == contradict`: `CONTRADICTION step N` is always
//!   printed (whether or not the step honestly said so) — this is the
//!   "must be caught/flagged" case from the design brief. It is a FINDING,
//!   not by itself a chain-integrity FAIL: a trace that honestly reports
//!   and then correctly resolves a contradiction is a working self-check,
//!   not a broken receipt.
//! - Final-step consistency, using the LAST verify step's
//!   `computed_verdict` and the draft step it (transitively) checked:
//!     - `computed_verdict == pass`: final's `answer` MUST equal that
//!       draft's `answer`, else `FINAL DRIFT step N` (a clean pass-through
//!       must not silently change the answer).
//!     - `computed_verdict == contradict`: final's `answer` MUST NOT equal
//!       that draft's `answer`, else `FINAL IGNORES CONTRADICTION step N`
//!       (a caught contradiction that the final step then ignores is a
//!       silent self-check failure — this is the case the whole scaffold
//!       exists to prevent).
//! - Chain integrity, independent of all of the above: recompute every
//!   step's `chain=` via `fold_step` from that step's own fields and the
//!   previous step's chain (genesis for step 0), and the trailer's
//!   `trace-chain=` as the last step's chain. Any mismatch is
//!   `STEP N CHAIN MISMATCH` / `TRACE CHAIN MISMATCH`, exactly the
//!   tamper-evidence `agent_trace`'s `verify` provides for decode steps.
//!
//! `verify` is OK (no findings beginning `FAIL`/`VERDICT LIE`/`FINAL`) iff
//! the receipt is structurally well-formed, every chain matches, no
//! verify-step lied about its own verdict, and the final step is
//! consistent with the last verify-step's (real) verdict. A `CONTRADICTION`
//! finding does NOT by itself make the trace fail — see above.
//!
//! ### Open gaps left for `reasoning-1` (live) or a future tick
//!
//! - Multi-round traces (draft -> verify -> revised draft -> verify ->
//!   final) are not modelled; only one draft and one linear run of
//!   verify-steps culminating in one final step. Extending `ref=` chains
//!   to support a second draft is a straightforward generalisation of the
//!   sequencing rules above, deferred because no live transcript needs it
//!   yet.
//! - `text-sha256=` is never checked against actual prose here (no prose
//!   is transported in these synthetic tests) — `reasoning-1` (live) needs
//!   to decide whether the full rationale text ships alongside the
//!   receipt (as `agent_trace`'s bundle/pack does for tool tables) or stays
//!   off-receipt entirely.
//! - `answer=`/`recheck=` are bare tokens, not domain-typed (no int
//!   parsing/canonicalisation) — "42" and "042" are different tokens here.
//!   Fine for hand-written synthetic transcripts; a live numeric-answer
//!   scaffold may want to canonicalise before folding.
//!
//! ## `reasoning-2`: CALC-augmented verify-step (additive)
//!
//! A `kind=verify` step may OPTIONALLY carry a tool call, recording that
//! the recheck was produced by invoking the same `CALC` tool
//! `agent_trace.rs` defines (`CALC(<int> <op> <int>)`, checked i64
//! arithmetic, same five ops `+ - * / %`), instead of purely re-deriving
//! the answer from the model's own tokens:
//!
//! ```text
//! step 1: kind=verify ref=0 recheck=<token> verdict=pass|contradict tool=calc tool-in=CALC(<a><op><b>) tool-out=<token> text-sha256=<hex> chain=<hex>
//! ```
//!
//! `tool=`/`tool-in=`/`tool-out=` are either ALL present or ALL absent on a
//! `verify` step — a partial tool record is a parse error. This is
//! deliberately additive: a `verify` step with none of these three fields
//! parses and verifies EXACTLY as before (`reasoning-1`'s no-tool
//! verify-step is untouched). `draft`/`final` steps never carry tool
//! fields (same "non-draft field"/"verify-only field" rejection as every
//! other kind-specific field).
//!
//! `tool-in=` is a COMPACT form of `agent_trace`'s `CALC(...)` grammar —
//! no internal whitespace, since every `step` line field is a single
//! whitespace-delimited token (`CALC(3+4)`, not `CALC(3 + 4)`) — otherwise
//! the same `<int> <op> <int>` grammar, re-parsed here (`calc::parse`)
//! rather than importing `agent_trace`'s private `find_calc`/`eval_calc`,
//! per this module's existing "reimplement, don't import across examples"
//! convention. `tool-out=` is a bare token: the decimal `i64` result, or
//! one of the three fixed error strings `agent_trace::eval_calc` already
//! uses (`overflow`, `div-by-zero`, `bad-op`).
//!
//! `verify` recomputes `tool-out=` from `tool-in=` independently
//! (`calc::eval`) and treats any mismatch as `TOOL RESULT MISMATCH step N`
//! — a `Fail` finding, catching a tampered/hallucinated tool result even
//! if every other field and chain link is internally consistent. It also
//! checks that `recheck=` actually equals the (claimed) `tool-out=` value:
//! a verify-step that carries a genuine, correctly-recomputed tool result
//! but then writes a DIFFERENT `recheck=` token gets the tool's cryptographic
//! backing for nothing, so that mismatch is its own `Fail`,
//! `RECHECK IGNORES TOOL OUTPUT step N`. Everything downstream of
//! `recheck=` (verdict-lie / contradiction / final-drift checks) is
//! UNCHANGED — a tool-backed `recheck=` is just a `recheck=` value with
//! stronger provenance, not a different code path.

use aegis_core::witness::{Sha256, hex_lower, sha256};

const REASON_DOMAIN: &str = "AEGIS-REASON v1\n";

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

fn unhex(s: &str) -> Result<[u8; 32], ()> {
    if s.len() != 64 || !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return Err(());
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| ())?;
    }
    Ok(out)
}

/// Bare-token grammar for `answer=`/`recheck=` values: no whitespace, no
/// `=`, non-empty — enough to keep every whitespace-separated slice of a
/// `step` line an unambiguous single `key=value` pair.
fn is_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'+' | b'-'))
}

/// `reasoning-2`: a compact, whitespace-free reimplementation of
/// `agent_trace.rs`'s `CALC(<int> <op> <int>)` grammar and checked i64
/// arithmetic (same five ops, same three fixed error strings), scoped to
/// this module rather than imported across examples (see module doc
/// comment).
mod calc {
    /// Parse a compact `CALC(<int><op><int>)` call (no internal
    /// whitespace — every `step` line field is one whitespace-delimited
    /// token). Returns `None` if `s` is not exactly this shape.
    pub fn parse(s: &str) -> Option<(i64, u8, i64)> {
        let inner = s.strip_prefix("CALC(")?.strip_suffix(')')?;
        if inner.is_empty() {
            return None;
        }
        let b = inner.as_bytes();
        let mut i = 0usize;
        if b[i] == b'-' {
            i += 1;
        }
        let digits_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == digits_start {
            return None; // no digits for `a`
        }
        let op_pos = i;
        if op_pos >= b.len() {
            return None;
        }
        let op = b[op_pos];
        if !matches!(op, b'+' | b'-' | b'*' | b'/' | b'%') {
            return None;
        }
        let a: i64 = inner[..op_pos].parse().ok()?;
        let b_str = &inner[op_pos + 1..];
        if b_str.is_empty() {
            return None;
        }
        let b_val: i64 = b_str.parse().ok()?;
        Some((a, op, b_val))
    }

    /// Checked i64 arithmetic, identical semantics to
    /// `agent_trace::eval_calc`: `Err` carries a fixed, deterministic
    /// error string, never a crash.
    pub fn eval(a: i64, op: u8, b: i64) -> Result<i64, &'static str> {
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

    /// `parse` then `eval`, rendered as the bare token that goes in
    /// `tool-out=` (a decimal `i64`, or one of `eval`'s fixed error
    /// strings). `None` iff `s` is not a well-formed `CALC(...)` call.
    pub fn run(s: &str) -> Option<String> {
        let (a, op, b) = parse(s)?;
        Some(match eval(a, op, b) {
            Ok(v) => v.to_string(),
            Err(e) => e.to_string(),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Draft,
    Verify,
    Final,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Draft => "draft",
            Kind::Verify => "verify",
            Kind::Final => "final",
        }
    }
}

#[derive(Clone, Debug)]
struct Step {
    idx: usize,
    kind: Kind,
    answer: Option<String>,
    ref_idx: Option<usize>,
    recheck: Option<String>,
    verdict: Option<String>,
    tool: Option<String>,
    tool_in: Option<String>,
    tool_out: Option<String>,
    text_sha256: [u8; 32],
    chain: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct ReasoningTrace {
    prompt_sha256: [u8; 32],
    steps: Vec<Step>,
    trace_chain: [u8; 32],
}

/// Fold one step into the running chain. `chain` is the genesis digest for
/// step 0, the previous step's `chain=` for every later step. Fields are
/// folded in a fixed order, each length-prefixed (LE u32) so no
/// concatenation ambiguity exists between adjacent variable-length fields
/// (same rationale as `agent_trace::trace_fold_step`, new domain/algorithm
/// since this is a new format, not required to bit-match it).
fn fold_step(chain: &[u8; 32], step: &Step) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(chain);
    s.update(b"RSTEP");
    s.update(&(step.idx as u64).to_be_bytes());
    s.update(&[match step.kind {
        Kind::Draft => 0u8,
        Kind::Verify => 1u8,
        Kind::Final => 2u8,
    }]);
    let mut field = |present: bool, bytes: &[u8]| {
        s.update(&[present as u8]);
        s.update(&(bytes.len() as u32).to_le_bytes());
        s.update(bytes);
    };
    field(
        step.ref_idx.is_some(),
        &step.ref_idx.unwrap_or(0).to_be_bytes(),
    );
    field(
        step.answer.is_some(),
        step.answer.as_deref().unwrap_or("").as_bytes(),
    );
    field(
        step.recheck.is_some(),
        step.recheck.as_deref().unwrap_or("").as_bytes(),
    );
    field(
        step.verdict.is_some(),
        step.verdict.as_deref().unwrap_or("").as_bytes(),
    );
    field(
        step.tool.is_some(),
        step.tool.as_deref().unwrap_or("").as_bytes(),
    );
    field(
        step.tool_in.is_some(),
        step.tool_in.as_deref().unwrap_or("").as_bytes(),
    );
    field(
        step.tool_out.is_some(),
        step.tool_out.as_deref().unwrap_or("").as_bytes(),
    );
    s.update(&step.text_sha256);
    s.finalize()
}

fn genesis(prompt_sha256: &[u8; 32]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(REASON_DOMAIN.as_bytes());
    s.update(prompt_sha256);
    s.finalize()
}

/// Split a `step` line's body (after `"step N:"`) into `key=value` tokens,
/// rejecting anything that is not exactly one `=` with a non-empty key —
/// same strict rule `agent_trace::verify_one` applies to its step lines.
fn split_fields(body: &str) -> Result<Vec<(&str, &str)>, String> {
    body.split_whitespace()
        .map(|tok| {
            let mut parts = tok.splitn(2, '=');
            let k = parts.next().unwrap_or("");
            let v = parts.next();
            match v {
                Some(v) if !k.is_empty() => Ok((k, v)),
                _ => Err(format!("malformed field {tok:?}")),
            }
        })
        .collect()
}

/// Parse receipt text into a `ReasoningTrace`. Pure structural validation
/// only (header shape, step numbering, known/required/non-duplicate keys,
/// well-formed hex/tokens, sequencing rules from the module doc comment).
/// Chain integrity and verdict semantics are `verify`'s job, not the
/// parser's — a malformed receipt never reaches `verify`.
pub fn parse(text: &str) -> Result<ReasoningTrace, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    match lines.next() {
        Some(REASON_HEADER) => {}
        Some(other) => return Err(format!("expected {REASON_HEADER:?} first, got {other:?}")),
        None => return Err("empty receipt".to_string()),
    }
    let prompt_sha256 = match lines.next() {
        Some(l) => match l.strip_prefix("prompt-sha256=") {
            Some(hexs) => unhex(hexs).map_err(|()| "malformed prompt-sha256".to_string())?,
            None => return Err(format!("expected prompt-sha256= line, got {l:?}")),
        },
        None => return Err("missing prompt-sha256 line".to_string()),
    };

    let rest: Vec<&str> = lines.collect();
    if rest.is_empty() {
        return Err("no step lines".to_string());
    }
    let (step_lines, trailer) = rest.split_at(rest.len() - 1);
    let trailer = trailer[0];
    if step_lines.is_empty() {
        return Err("no step lines".to_string());
    }

    let mut steps = Vec::with_capacity(step_lines.len());
    for (expected_idx, line) in step_lines.iter().enumerate() {
        if line.starts_with(REASON_HEADER.trim_end()) {
            return Err("duplicate AEGIS-REASON header line".to_string());
        }
        let rest = line
            .strip_prefix("step ")
            .ok_or_else(|| format!("expected step line, got {line:?}"))?;
        let (label, body) = rest
            .split_once(':')
            .ok_or_else(|| format!("malformed step line {line:?}"))?;
        let n: usize = label
            .trim()
            .parse()
            .map_err(|_| format!("malformed step label {label:?}"))?;
        if n != expected_idx {
            return Err(format!("step label {n} at position {expected_idx}"));
        }
        let fields = split_fields(body.trim())?;
        let mut kind: Option<Kind> = None;
        let mut answer: Option<String> = None;
        let mut ref_idx: Option<usize> = None;
        let mut recheck: Option<String> = None;
        let mut verdict: Option<String> = None;
        let mut tool: Option<String> = None;
        let mut tool_in: Option<String> = None;
        let mut tool_out: Option<String> = None;
        let mut text_sha256: Option<[u8; 32]> = None;
        let mut chain: Option<[u8; 32]> = None;
        for (k, v) in fields {
            let dup = |slot_is_some: bool| -> Result<(), String> {
                if slot_is_some {
                    Err(format!("duplicate key {k:?} on step {n}"))
                } else {
                    Ok(())
                }
            };
            match k {
                "kind" => {
                    dup(kind.is_some())?;
                    kind = Some(match v {
                        "draft" => Kind::Draft,
                        "verify" => Kind::Verify,
                        "final" => Kind::Final,
                        other => return Err(format!("step {n}: unknown kind {other:?}")),
                    });
                }
                "answer" => {
                    dup(answer.is_some())?;
                    if !is_token(v) {
                        return Err(format!("step {n}: malformed answer token {v:?}"));
                    }
                    answer = Some(v.to_string());
                }
                "ref" => {
                    dup(ref_idx.is_some())?;
                    ref_idx = Some(
                        v.parse()
                            .map_err(|_| format!("step {n}: malformed ref {v:?}"))?,
                    );
                }
                "recheck" => {
                    dup(recheck.is_some())?;
                    if !is_token(v) {
                        return Err(format!("step {n}: malformed recheck token {v:?}"));
                    }
                    recheck = Some(v.to_string());
                }
                "verdict" => {
                    dup(verdict.is_some())?;
                    if v != "pass" && v != "contradict" {
                        return Err(format!("step {n}: unknown verdict {v:?}"));
                    }
                    verdict = Some(v.to_string());
                }
                "tool" => {
                    dup(tool.is_some())?;
                    if v != "calc" {
                        return Err(format!("step {n}: unknown tool {v:?}"));
                    }
                    tool = Some(v.to_string());
                }
                "tool-in" => {
                    dup(tool_in.is_some())?;
                    if calc::parse(v).is_none() {
                        return Err(format!("step {n}: malformed tool-in {v:?}"));
                    }
                    tool_in = Some(v.to_string());
                }
                "tool-out" => {
                    dup(tool_out.is_some())?;
                    if !is_token(v) {
                        return Err(format!("step {n}: malformed tool-out token {v:?}"));
                    }
                    tool_out = Some(v.to_string());
                }
                "text-sha256" => {
                    dup(text_sha256.is_some())?;
                    text_sha256 =
                        Some(unhex(v).map_err(|()| format!("step {n}: malformed text-sha256"))?);
                }
                "chain" => {
                    dup(chain.is_some())?;
                    chain = Some(unhex(v).map_err(|()| format!("step {n}: malformed chain"))?);
                }
                other => return Err(format!("step {n}: unknown key {other:?}")),
            }
        }
        let kind = kind.ok_or_else(|| format!("step {n}: missing kind"))?;
        let text_sha256 = text_sha256.ok_or_else(|| format!("step {n}: missing text-sha256"))?;
        let chain = chain.ok_or_else(|| format!("step {n}: missing chain"))?;
        match kind {
            Kind::Draft => {
                if answer.is_none() {
                    return Err(format!("step {n}: draft missing answer"));
                }
                if ref_idx.is_some() || recheck.is_some() || verdict.is_some() {
                    return Err(format!("step {n}: draft has non-draft field"));
                }
                if tool.is_some() || tool_in.is_some() || tool_out.is_some() {
                    return Err(format!("step {n}: draft has verify-only tool field"));
                }
            }
            Kind::Verify => {
                if ref_idx.is_none() || recheck.is_none() || verdict.is_none() {
                    return Err(format!("step {n}: verify missing ref/recheck/verdict"));
                }
                if answer.is_some() {
                    return Err(format!("step {n}: verify has answer field"));
                }
                // tool/tool-in/tool-out: additive, all-or-nothing (reasoning-2).
                let tool_field_count =
                    tool.is_some() as u8 + tool_in.is_some() as u8 + tool_out.is_some() as u8;
                if tool_field_count != 0 && tool_field_count != 3 {
                    return Err(format!(
                        "step {n}: partial tool record (need all of tool/tool-in/tool-out or none)"
                    ));
                }
            }
            Kind::Final => {
                if ref_idx.is_none() || answer.is_none() {
                    return Err(format!("step {n}: final missing ref/answer"));
                }
                if recheck.is_some() || verdict.is_some() {
                    return Err(format!("step {n}: final has verify-only field"));
                }
                if tool.is_some() || tool_in.is_some() || tool_out.is_some() {
                    return Err(format!("step {n}: final has verify-only tool field"));
                }
            }
        }
        if let Some(r) = ref_idx {
            if r >= n {
                return Err(format!("step {n}: ref={r} does not precede this step"));
            }
        }
        steps.push(Step {
            idx: n,
            kind,
            answer,
            ref_idx,
            recheck,
            verdict,
            tool,
            tool_in,
            tool_out,
            text_sha256,
            chain,
        });
    }

    let trace_chain = trailer
        .strip_prefix("trace-chain=")
        .ok_or_else(|| format!("expected trace-chain= trailer, got {trailer:?}"))
        .and_then(|hexs| unhex(hexs).map_err(|()| "malformed trace-chain".to_string()))?;

    // Sequencing rules.
    if steps[0].kind != Kind::Draft {
        return Err("step 0 must be kind=draft".to_string());
    }
    let final_positions: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind == Kind::Final)
        .map(|(i, _)| i)
        .collect();
    if final_positions.len() != 1 {
        return Err(format!(
            "exactly one kind=final step required, found {}",
            final_positions.len()
        ));
    }
    if final_positions[0] != steps.len() - 1 {
        return Err("kind=final step must be the last step".to_string());
    }
    let verify_positions: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind == Kind::Verify)
        .map(|(i, _)| i)
        .collect();
    if verify_positions.is_empty() {
        return Err("at least one kind=verify step required before final".to_string());
    }
    for s in &steps {
        if s.kind == Kind::Verify {
            let r = s.ref_idx.unwrap();
            if steps[r].kind != Kind::Draft {
                return Err(format!(
                    "step {}: verify ref={r} is not a draft step",
                    s.idx
                ));
            }
        }
    }
    let last_verify = *verify_positions.last().unwrap();
    let final_step = &steps[final_positions[0]];
    if final_step.ref_idx.unwrap() != last_verify {
        return Err(format!(
            "final step must ref the last verify step ({last_verify}), got ref={}",
            final_step.ref_idx.unwrap()
        ));
    }

    Ok(ReasoningTrace {
        prompt_sha256,
        steps,
        trace_chain,
    })
}

const REASON_HEADER: &str = "AEGIS-REASON v1";

/// One verifier finding. `Fail` findings mean the receipt does not verify;
/// `Note`/`Contradiction` are informational (the second is the "a
/// verify-step caught a contradiction" signal the design brief asks for,
/// and by itself does NOT fail the trace — see the module doc comment).
#[derive(Debug, PartialEq, Eq)]
pub enum Finding {
    Fail(String),
    Contradiction(String),
    Note(String),
}

impl Finding {
    pub fn is_fail(&self) -> bool {
        matches!(self, Finding::Fail(_))
    }
    pub fn message(&self) -> &str {
        match self {
            Finding::Fail(m) | Finding::Contradiction(m) | Finding::Note(m) => m,
        }
    }
}

/// Independently recompute every step's chain, every verify-step's
/// verdict, and the final step's consistency with the (real) last
/// verdict. Returns every finding in step order; `ok()` (no `Fail`
/// findings) is the pass/fail bit a caller should act on. `verify` never
/// trusts a `verdict=` field's own claim — see the module doc comment.
pub fn verify(trace: &ReasoningTrace) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut chain = genesis(&trace.prompt_sha256);
    let mut computed_verdicts: Vec<Option<bool>> = vec![None; trace.steps.len()]; // true=pass

    for step in &trace.steps {
        let expected = fold_step(&chain, step);
        if expected != step.chain {
            findings.push(Finding::Fail(format!("STEP {} CHAIN MISMATCH", step.idx)));
        }
        chain = expected;

        if step.kind == Kind::Verify {
            let draft = &trace.steps[step.ref_idx.unwrap()];
            let recheck = step.recheck.as_deref().unwrap();
            let draft_answer = draft.answer.as_deref().unwrap();

            // reasoning-2: if this verify-step carries a CALC tool call,
            // independently recompute its result and never trust the
            // claimed `tool-out=` field, same "verify recomputes, never
            // trusts a claim" discipline as `verdict=` below.
            if let Some(tool_in) = step.tool_in.as_deref() {
                let claimed_out = step.tool_out.as_deref().unwrap();
                match calc::run(tool_in) {
                    Some(recomputed) if recomputed == claimed_out => {}
                    Some(recomputed) => {
                        findings.push(Finding::Fail(format!(
                            "TOOL RESULT MISMATCH step {}: tool-in={tool_in:?} claims out={claimed_out:?} recomputed={recomputed:?}",
                            step.idx
                        )));
                    }
                    None => {
                        // Parser already validated tool-in against the
                        // same calc::parse grammar, so this is
                        // unreachable in practice; treat defensively as
                        // a mismatch rather than panicking.
                        findings.push(Finding::Fail(format!(
                            "TOOL RESULT MISMATCH step {}: tool-in={tool_in:?} does not parse",
                            step.idx
                        )));
                    }
                }
                if recheck != claimed_out {
                    findings.push(Finding::Fail(format!(
                        "RECHECK IGNORES TOOL OUTPUT step {}: tool-out={claimed_out:?} recheck={recheck:?}",
                        step.idx
                    )));
                }
            }

            let computed_pass = recheck == draft_answer;
            computed_verdicts[step.idx] = Some(computed_pass);
            let claimed_pass = step.verdict.as_deref() == Some("pass");
            if computed_pass != claimed_pass {
                findings.push(Finding::Fail(format!(
                    "VERDICT LIE step {}: claims {} but recheck={recheck:?} draft-answer={draft_answer:?} -> {}",
                    step.idx,
                    step.verdict.as_deref().unwrap_or(""),
                    if computed_pass { "pass" } else { "contradict" }
                )));
            }
            if !computed_pass {
                findings.push(Finding::Contradiction(format!(
                    "CONTRADICTION step {}: draft(step {})={draft_answer:?} recheck={recheck:?}",
                    step.idx, draft.idx
                )));
            }
        }
    }

    if chain != trace.trace_chain {
        findings.push(Finding::Fail("TRACE CHAIN MISMATCH".to_string()));
    }

    let final_step = trace.steps.last().unwrap();
    let last_verify_idx = final_step.ref_idx.unwrap();
    let last_verify = &trace.steps[last_verify_idx];
    let draft_idx = last_verify.ref_idx.unwrap();
    let draft = &trace.steps[draft_idx];
    let computed_pass = computed_verdicts[last_verify_idx].unwrap_or(true);
    let final_answer = final_step.answer.as_deref().unwrap();
    let draft_answer = draft.answer.as_deref().unwrap();
    if computed_pass && final_answer != draft_answer {
        findings.push(Finding::Fail(format!(
            "FINAL DRIFT step {}: draft(step {draft_idx})={draft_answer:?} final={final_answer:?}",
            final_step.idx
        )));
    }
    if !computed_pass && final_answer == draft_answer {
        findings.push(Finding::Fail(format!(
            "FINAL IGNORES CONTRADICTION step {}: final repeats rejected draft(step {draft_idx})={draft_answer:?}",
            final_step.idx
        )));
    }

    findings
}

/// `true` iff `verify` found nothing that should stop this trace from
/// being trusted (no `Finding::Fail`). A `Finding::Contradiction` alone
/// does not fail a trace — a caught-and-resolved contradiction is the
/// scaffold working as intended.
pub fn ok(findings: &[Finding]) -> bool {
    !findings.iter().any(Finding::is_fail)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: reasoning_trace <receipt-file>");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[1]).unwrap_or_else(|e| {
        eprintln!("read {}: {e}", args[1]);
        std::process::exit(2);
    });
    let trace = match parse(&text) {
        Ok(t) => t,
        Err(e) => {
            println!("FAIL structure: {e}");
            std::process::exit(1);
        }
    };
    let findings = verify(&trace);
    for f in &findings {
        let tag = match f {
            Finding::Fail(_) => "FAIL",
            Finding::Contradiction(_) => "CONTRADICTION",
            Finding::Note(_) => "NOTE",
        };
        println!("{tag}: {}", f.message());
    }
    if ok(&findings) {
        println!("VERIFY PASS");
    } else {
        println!("VERIFY FAIL");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------
// Tests: hand-written synthetic transcripts, not model-generated. Every
// receipt here is built with `Builder` (below), which computes real
// chains/digests, so a test that mutates a field is exercising the exact
// same chain math `verify` uses, not a hand-typed hex string.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a well-formed draft -> verify -> final receipt string,
    /// computing every `text-sha256=`/`chain=`/`trace-chain=` field from
    /// the caller's structural choices, so tests describe INTENT (what
    /// the draft answered, what the recheck found, what verdict is
    /// claimed, what the final says) rather than raw bytes.
    struct Builder {
        prompt: &'static str,
        draft_answer: &'static str,
        recheck: &'static str,
        claimed_verdict: &'static str,
        final_answer: &'static str,
        /// reasoning-2: `Some((tool_in, tool_out))` puts a CALC tool
        /// record on the verify step; `None` renders the plain
        /// reasoning-1 verify step (no tool fields at all).
        tool_call: Option<(&'static str, &'static str)>,
    }

    impl Builder {
        fn render(&self) -> String {
            let prompt_sha256 = sha256(self.prompt.as_bytes());
            let mut chain = genesis(&prompt_sha256);
            let mut out = String::new();
            out.push_str(REASON_HEADER);
            out.push('\n');
            out.push_str(&format!("prompt-sha256={}\n", hex(&prompt_sha256)));

            #[allow(clippy::too_many_arguments)]
            let push_step = |idx: usize,
                             kind: Kind,
                             answer: Option<&str>,
                             ref_idx: Option<usize>,
                             recheck: Option<&str>,
                             verdict: Option<&str>,
                             tool_call: Option<(&str, &str)>,
                             text: &str,
                             out: &mut String,
                             chain: &mut [u8; 32]| {
                let text_sha256 = sha256(text.as_bytes());
                let (tool, tool_in, tool_out) = match tool_call {
                    Some((tin, tout)) => (
                        Some("calc".to_string()),
                        Some(tin.to_string()),
                        Some(tout.to_string()),
                    ),
                    None => (None, None, None),
                };
                let step = Step {
                    idx,
                    kind,
                    answer: answer.map(|s| s.to_string()),
                    ref_idx,
                    recheck: recheck.map(|s| s.to_string()),
                    verdict: verdict.map(|s| s.to_string()),
                    tool,
                    tool_in,
                    tool_out,
                    text_sha256,
                    chain: [0u8; 32], // placeholder, computed below
                };
                let new_chain = fold_step(chain, &step);
                let mut line = format!("step {idx}: kind={}", kind.as_str());
                if let Some(r) = ref_idx {
                    line.push_str(&format!(" ref={r}"));
                }
                if let Some(a) = answer {
                    line.push_str(&format!(" answer={a}"));
                }
                if let Some(r) = recheck {
                    line.push_str(&format!(" recheck={r}"));
                }
                if let Some(v) = verdict {
                    line.push_str(&format!(" verdict={v}"));
                }
                if let Some((tin, tout)) = tool_call {
                    line.push_str(&format!(" tool=calc tool-in={tin} tool-out={tout}"));
                }
                line.push_str(&format!(
                    " text-sha256={} chain={}\n",
                    hex(&text_sha256),
                    hex(&new_chain)
                ));
                out.push_str(&line);
                *chain = new_chain;
            };

            push_step(
                0,
                Kind::Draft,
                Some(self.draft_answer),
                None,
                None,
                None,
                None,
                "draft rationale",
                &mut out,
                &mut chain,
            );
            push_step(
                1,
                Kind::Verify,
                None,
                Some(0),
                Some(self.recheck),
                Some(self.claimed_verdict),
                self.tool_call,
                "verify rationale",
                &mut out,
                &mut chain,
            );
            push_step(
                2,
                Kind::Final,
                Some(self.final_answer),
                Some(1),
                None,
                None,
                None,
                "final rationale",
                &mut out,
                &mut chain,
            );

            out.push_str(&format!("trace-chain={}\n", hex(&chain)));
            out
        }
    }

    fn clean_pass() -> Builder {
        Builder {
            prompt: "What is 2 + 2?",
            draft_answer: "4",
            recheck: "4",
            claimed_verdict: "pass",
            final_answer: "4",
            tool_call: None,
        }
    }

    fn caught_and_resolved_contradiction() -> Builder {
        Builder {
            prompt: "What is 17 * 3?",
            draft_answer: "41", // wrong
            recheck: "51",      // correct
            claimed_verdict: "contradict",
            final_answer: "51", // final trusts the recheck
            tool_call: None,
        }
    }

    #[test]
    fn clean_pass_verifies_ok() {
        let text = clean_pass().render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(ok(&findings), "findings: {findings:?}");
        assert!(
            !findings
                .iter()
                .any(|f| matches!(f, Finding::Contradiction(_))),
            "clean pass-through must not flag a contradiction: {findings:?}"
        );
    }

    #[test]
    fn caught_contradiction_is_flagged_and_still_verifies() {
        let text = caught_and_resolved_contradiction().render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(
            ok(&findings),
            "resolved contradiction must still verify: {findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|f| matches!(f, Finding::Contradiction(_))),
            "a verify-step that contradicts its draft must be caught/flagged: {findings:?}"
        );
    }

    #[test]
    fn verify_step_lying_about_pass_is_a_fail() {
        let mut b = caught_and_resolved_contradiction();
        b.claimed_verdict = "pass"; // recheck != draft answer, but claims pass
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(!ok(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.message().starts_with("VERDICT LIE"))
        );
    }

    #[test]
    fn verify_step_lying_about_contradict_is_a_fail() {
        let mut b = clean_pass();
        b.claimed_verdict = "contradict"; // recheck == draft answer, but claims contradict
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(!ok(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.message().starts_with("VERDICT LIE"))
        );
    }

    #[test]
    fn final_ignoring_contradiction_is_a_fail() {
        let mut b = caught_and_resolved_contradiction();
        b.final_answer = "41"; // repeats the rejected draft answer
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(!ok(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.message().starts_with("FINAL IGNORES CONTRADICTION"))
        );
    }

    #[test]
    fn final_drift_on_clean_pass_is_a_fail() {
        let mut b = clean_pass();
        b.final_answer = "5"; // drifted from the (correctly verified) draft
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(!ok(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.message().starts_with("FINAL DRIFT"))
        );
    }

    #[test]
    fn tampered_chain_hex_fails_chain_check() {
        let text = clean_pass().render();
        // Flip one hex nibble in step 0's chain= field.
        let tampered = text.replacen("step 0:", "step 0:", 1);
        let idx = tampered.find("chain=").expect("chain field present");
        let mut bytes = tampered.into_bytes();
        let target = idx + "chain=".len();
        bytes[target] = if bytes[target] == b'0' { b'1' } else { b'0' };
        let tampered = String::from_utf8(bytes).unwrap();
        let trace = parse(&tampered).expect("still structurally well-formed");
        let findings = verify(&trace);
        assert!(!ok(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.message().contains("CHAIN MISMATCH"))
        );
    }

    #[test]
    fn tampered_answer_after_signing_fails_chain_check() {
        // Editing the answer without recomputing chain/trace-chain must be
        // caught even though every field is individually well-formed.
        let text = clean_pass().render();
        let tampered = text.replace("answer=4 text-sha256", "answer=9 text-sha256");
        let trace = parse(&tampered).expect("still structurally well-formed");
        let findings = verify(&trace);
        assert!(!ok(&findings));
    }

    #[test]
    fn missing_verify_step_is_a_parse_error() {
        let prompt_sha256 = sha256(b"prompt");
        let mut chain = genesis(&prompt_sha256);
        let draft = Step {
            idx: 0,
            kind: Kind::Draft,
            answer: Some("4".to_string()),
            ref_idx: None,
            recheck: None,
            verdict: None,
            tool: None,
            tool_in: None,
            tool_out: None,
            text_sha256: sha256(b"draft"),
            chain: [0u8; 32],
        };
        chain = fold_step(&chain, &draft);
        let text = format!(
            "AEGIS-REASON v1\nprompt-sha256={}\nstep 0: kind=draft answer=4 text-sha256={} chain={}\ntrace-chain={}\n",
            hex(&prompt_sha256),
            hex(&draft.text_sha256),
            hex(&chain),
            hex(&chain),
        );
        let err = parse(&text).unwrap_err();
        assert!(err.contains("final"), "err: {err}");
    }

    #[test]
    fn out_of_order_step_label_is_rejected() {
        let text = clean_pass().render();
        let tampered = text.replace("step 1:", "step 5:");
        let err = parse(&tampered).unwrap_err();
        assert_eq!(err, "step label 5 at position 1");
    }

    #[test]
    fn duplicate_key_is_rejected() {
        let text = clean_pass().render();
        let tampered = text.replace(
            "step 0: kind=draft answer=4",
            "step 0: kind=draft answer=4 answer=9",
        );
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("duplicate key"), "err: {err}");
    }

    #[test]
    fn unknown_key_is_rejected() {
        let text = clean_pass().render();
        let tampered = text.replace(
            "step 0: kind=draft answer=4",
            "step 0: kind=draft answer=4 bogus=1",
        );
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("unknown key"), "err: {err}");
    }

    #[test]
    fn ref_pointing_forward_or_to_self_is_rejected() {
        let text = clean_pass().render();
        let tampered = text.replace("ref=0", "ref=1"); // verify step 1 refs itself
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("does not precede"), "err: {err}");
    }

    #[test]
    fn malformed_answer_token_is_rejected() {
        let text = clean_pass().render();
        let tampered = text.replace("answer=4 text-sha256", "answer=\"4\" text-sha256");
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("malformed answer token") || err.contains("malformed field"));
    }

    #[test]
    fn duplicate_header_is_rejected() {
        let text = clean_pass().render();
        let tampered = format!("AEGIS-REASON v1\n{text}");
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("duplicate AEGIS-REASON header line") || err.contains("expected"));
    }

    // -----------------------------------------------------------------
    // reasoning-2: CALC-augmented verify-step.
    // -----------------------------------------------------------------

    #[test]
    fn calc_module_matches_agent_trace_semantics() {
        assert_eq!(calc::run("CALC(3+4)"), Some("7".to_string()));
        assert_eq!(calc::run("CALC(-5*2)"), Some("-10".to_string()));
        assert_eq!(calc::run("CALC(9/0)"), Some("div-by-zero".to_string()));
        assert_eq!(
            calc::run("CALC(9223372036854775807+1)"),
            Some("overflow".to_string())
        );
        assert_eq!(calc::run("CALC(3 + 4)"), None); // no internal whitespace allowed
        assert_eq!(calc::run("CALC(3&4)"), None); // bad op
    }

    #[test]
    fn calc_verify_step_catches_wrong_draft_answer() {
        // Draft claims 41 (wrong); verify-step calls CALC(17*3)=51, which
        // becomes the recheck; 51 != 41 so this is a genuine
        // contradiction, correctly caught, and final must trust the tool.
        let mut b = caught_and_resolved_contradiction();
        b.tool_call = Some(("CALC(17*3)", "51"));
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(ok(&findings), "findings: {findings:?}");
        assert!(
            findings
                .iter()
                .any(|f| matches!(f, Finding::Contradiction(_))),
            "CALC disagreeing with the draft must be flagged as a contradiction: {findings:?}"
        );
        assert!(
            !findings.iter().any(|f| f.message().starts_with("TOOL")),
            "no tool-integrity finding expected on a correctly-recomputed tool call: {findings:?}"
        );
    }

    #[test]
    fn calc_verify_step_confirms_correct_draft() {
        // Draft already says 4; verify-step calls CALC(2+2)=4, confirming
        // it — a clean pass-through, no contradiction.
        let mut b = clean_pass();
        b.tool_call = Some(("CALC(2+2)", "4"));
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(ok(&findings), "findings: {findings:?}");
        assert!(
            !findings
                .iter()
                .any(|f| matches!(f, Finding::Contradiction(_))),
            "CALC confirming the draft must not be flagged as a contradiction: {findings:?}"
        );
    }

    #[test]
    fn tampered_calc_result_is_caught() {
        // The tool actually computes CALC(2+2)=4 (matches recheck=4,
        // draft=4, so the receipt would otherwise look like a clean
        // pass), but the receipt CLAIMS tool-out=9 — a tampered/
        // hallucinated tool result that disagrees with what CALC(2+2)
        // actually evaluates to. `verify` must recompute independently
        // and catch this even though every chain link is otherwise
        // internally consistent (the tamper is baked into the signed
        // receipt via the Builder, not a post-hoc edit).
        let mut b = clean_pass();
        b.tool_call = Some(("CALC(2+2)", "9"));
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(!ok(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.message().starts_with("TOOL RESULT MISMATCH")),
            "findings: {findings:?}"
        );
    }

    #[test]
    fn recheck_ignoring_tool_output_is_caught() {
        // tool-out is correct (CALC(2+2)=4) but recheck=5 doesn't match
        // it: the tool's cryptographic backing is being claimed for a
        // recheck value the tool never actually produced.
        let mut b = clean_pass();
        b.recheck = "5";
        b.claimed_verdict = "contradict";
        b.final_answer = "5";
        b.tool_call = Some(("CALC(2+2)", "4"));
        let text = b.render();
        let trace = parse(&text).expect("parses");
        let findings = verify(&trace);
        assert!(!ok(&findings));
        assert!(
            findings
                .iter()
                .any(|f| f.message().starts_with("RECHECK IGNORES TOOL OUTPUT")),
            "findings: {findings:?}"
        );
    }

    #[test]
    fn tool_partial_record_is_a_parse_error() {
        let text = clean_pass().render();
        // Splice in a lone tool-out= with no tool=/tool-in= alongside it.
        let tampered = text.replace(
            "step 1: kind=verify ref=0 recheck=4",
            "step 1: kind=verify ref=0 recheck=4 tool-out=4",
        );
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("partial tool record"), "err: {err}");
    }

    #[test]
    fn tool_field_on_draft_step_is_rejected() {
        let text = clean_pass().render();
        let tampered = text.replace(
            "step 0: kind=draft answer=4",
            "step 0: kind=draft answer=4 tool=calc tool-in=CALC(2+2) tool-out=4",
        );
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("verify-only tool field"), "err: {err}");
    }

    #[test]
    fn malformed_tool_in_is_rejected() {
        let text = clean_pass().render();
        let tampered = text.replace(
            "step 1: kind=verify ref=0 recheck=4 verdict=pass",
            "step 1: kind=verify ref=0 recheck=4 verdict=pass tool=calc tool-in=CALC(2&2) tool-out=4",
        );
        let err = parse(&tampered).unwrap_err();
        assert!(err.contains("malformed tool-in"), "err: {err}");
    }
}
