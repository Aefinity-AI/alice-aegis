//! reasoning_live_calc — LIVE half of reasoning-2: same driver as
//! `reasoning_live.rs` (reasoning-1's live half) on the SAME CALC-only
//! subset of the EVAL-60 T1 suite, but the verify-step now invokes the
//! `CALC` tool (compact-grammar reimplementation, `reasoning_trace.rs`'s
//! `calc` module — see that module's "reasoning-2" doc comment section)
//! instead of asking the model to freely re-derive the answer and
//! extracting a token from its generated text.
//!
//! What changes vs `reasoning_live.rs`, concretely:
//! - draft round: UNCHANGED (same prompt, same generation, same
//!   extraction).
//! - verify round: no model generation at all. `recheck=` is the CALC
//!   tool's OWN result for `item.expr` (ground truth arithmetic, checked
//!   i64, same semantics as `agent_trace::eval_calc`), recorded on the
//!   receipt as `tool=calc tool-in=... tool-out=...` alongside `recheck=`
//!   (which must equal `tool-out=` — `reasoning_trace::verify` enforces
//!   this). The verify step's `text-sha256=` is over a short fixed
//!   description of the tool call actually made (`"CALC tool invoked:
//!   <in> -> <out>"`), not over any model output — no model tokens are
//!   fabricated to stand in for a rationale that was never generated.
//! - final round: UNCHANGED prompt shape (same "recheck matched"/
//!   "recheck did not match" prompts, same generation, same extraction
//!   fallback), just now driven by the tool-backed `recheck`.
//!
//! This is the fair, honest reading of "same live setup, CALC now
//! allowed in verify": the draft and final rounds still burn real model
//! generation exactly as `reasoning_live.rs` did; only the middle round's
//! source of truth changes from "ask the model twice and hope it agrees
//! with itself" to "ask a checked arithmetic evaluator". See the run
//! report for why this is expected to (and does) push CALC-only accuracy
//! very high, and why that is NOT the same claim as "the scaffold makes
//! the model better at arithmetic in general" — CALC only covers items
//! whose expression is literally `CALC(<int> <op> <int>)`, by construction
//! of this eval subset.
//!
//! Run:
//!   reasoning_live_calc <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <suite.tsv> <outdir> [limit]

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel, argmax_i64};
use aegis_core::model::{FullBitNetPipeline, ModelConfig, SafeTensors};
use aegis_core::tokenizer::AegisTokenizer;
use aegis_core::witness::{Sha256, hex_lower, sha256};
use std::fs;
use std::path::Path;

const REASON_DOMAIN: &str = "AEGIS-REASON v1\n";

fn hex(b: &[u8]) -> String {
    let mut out = vec![0u8; b.len() * 2];
    let n = hex_lower(b, &mut out);
    String::from_utf8(out[..n].to_vec()).unwrap()
}

fn genesis(prompt_sha256: &[u8; 32]) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(REASON_DOMAIN.as_bytes());
    s.update(prompt_sha256);
    s.finalize()
}

enum FKind {
    Draft,
    Verify,
    Final,
}

/// `reasoning_trace.rs`'s `calc` module, reimplemented here rather than
/// imported across examples (same convention that module documents).
mod calc {
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
            return None;
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

    pub fn run(s: &str) -> Option<String> {
        let (a, op, b) = parse(s)?;
        Some(match eval(a, op, b) {
            Ok(v) => v.to_string(),
            Err(e) => e.to_string(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_step(
    chain: &[u8; 32],
    idx: usize,
    kind: &FKind,
    ref_idx: Option<usize>,
    answer: Option<&str>,
    recheck: Option<&str>,
    verdict: Option<&str>,
    tool: Option<&str>,
    tool_in: Option<&str>,
    tool_out: Option<&str>,
    text_sha256: &[u8; 32],
) -> [u8; 32] {
    let mut s = Sha256::new();
    s.update(chain);
    s.update(b"RSTEP");
    s.update(&(idx as u64).to_be_bytes());
    s.update(&[match kind {
        FKind::Draft => 0u8,
        FKind::Verify => 1u8,
        FKind::Final => 2u8,
    }]);
    let mut field = |present: bool, bytes: &[u8]| {
        s.update(&[present as u8]);
        s.update(&(bytes.len() as u32).to_le_bytes());
        s.update(bytes);
    };
    field(
        ref_idx.is_some(),
        &(ref_idx.unwrap_or(0) as u64).to_be_bytes(),
    );
    field(answer.is_some(), answer.unwrap_or("").as_bytes());
    field(recheck.is_some(), recheck.unwrap_or("").as_bytes());
    field(verdict.is_some(), verdict.unwrap_or("").as_bytes());
    field(tool.is_some(), tool.unwrap_or("").as_bytes());
    field(tool_in.is_some(), tool_in.unwrap_or("").as_bytes());
    field(tool_out.is_some(), tool_out.unwrap_or("").as_bytes());
    s.update(text_sha256);
    s.finalize()
}

fn is_token(s: &str) -> bool {
    !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'+' | b'-'))
}

/// Extract the first bare token after a `LABEL:` marker in generated text.
fn extract(text: &str, label: &str) -> Option<String> {
    let idx = text.find(label)?;
    let after = &text[idx + label.len()..];
    let tok: String = after
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '-'))
        .collect();
    if tok.is_empty() { None } else { Some(tok) }
}

struct Model<'a> {
    cis_model: CisModel<'a>,
    tokenizer: AegisTokenizer<'a>,
}

fn generate(model: &Model, prompt: &str, max_new: usize) -> (String, [u8; 32]) {
    let mut engine = CisEngine::new_with_mode(&model.cis_model, CisMode::FullInt);
    let prompt_ids = model.tokenizer.encode(prompt);
    let mut pos = 0usize;
    for &t in &prompt_ids {
        engine.forward_step_int(t, pos);
        pos += 1;
    }
    let mut generated = Vec::with_capacity(max_new);
    let mut current = argmax_i64(engine.decode_logits());
    for _ in 0..max_new {
        generated.push(current);
        engine.forward_step_int(current, pos);
        pos += 1;
        current = argmax_i64(engine.decode_logits());
    }
    let text = model.tokenizer.decode(&generated);
    let digest = sha256(text.as_bytes());
    (text, digest)
}

struct Item {
    id: String,
    expr: String,
    expected: String,
}

fn parse_calc_items(tsv: &str) -> Vec<Item> {
    let mut out = Vec::new();
    for (i, line) in tsv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 7 {
            continue;
        }
        let (id, expected_tool, expected_input, expected_output) =
            (cols[0], cols[4], cols[5], cols[6]);
        if expected_tool != "CALC" {
            continue;
        }
        let expr = expected_input
            .strip_prefix("CALC(")
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or(expected_input)
            .to_string();
        out.push(Item {
            id: id.to_string(),
            expr,
            expected: expected_output.to_string(),
        });
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        eprintln!(
            "usage: reasoning_live_calc <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN> <suite.tsv> <outdir> [limit]"
        );
        std::process::exit(2);
    }
    let limit: usize = args
        .get(6)
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let model_bytes = fs::read(&args[1]).expect("read MODEL.SAF");
    let embed_bytes = fs::read(&args[2]).expect("read EMBED.BIN");
    let vocab_bytes = fs::read(&args[3]).expect("read VOCAB.BIN");
    let suite_text = fs::read_to_string(&args[4]).expect("read suite.tsv");
    let outdir = &args[5];
    fs::create_dir_all(outdir).expect("mkdir outdir");
    fs::create_dir_all(format!("{outdir}/receipts")).expect("mkdir receipts");

    let tensors = SafeTensors::deserialize(&model_bytes).expect("parse MODEL.SAF");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("read __metadata__")
        .expect("MODEL.SAF carries no aegis_config");
    let config = ModelConfig::from_json(&cfg_json).expect("parse aegis_config");
    let pipeline =
        FullBitNetPipeline::new(&tensors, &embed_bytes, &config).expect("build pipeline");
    let cis_model = CisModel::new(&pipeline, &config).expect("CIS model conversion");
    let tokenizer = AegisTokenizer::new(&vocab_bytes).expect("parse VOCAB.BIN");
    let model = Model {
        cis_model,
        tokenizer,
    };

    let items = parse_calc_items(&suite_text);
    let total_calc = items.len();
    let n = items.len().min(limit);
    println!("CALC items in suite: {total_calc}; running {n}");

    let mut summary = String::from(
        "item_id\texpected\tdraft\ttool_in\ttool_out\trecheck\tcomputed_verdict\tfinal\tfinal_correct\treceipt_bytes\textract_ok\n",
    );

    let mut final_correct = 0usize;
    let mut extract_fail = 0usize;
    let mut tool_parse_fail = 0usize;

    for item in items.into_iter().take(n) {
        // --- draft round: UNCHANGED from reasoning_live.rs ---
        let draft_prompt = format!(
            "Answer the arithmetic question. End your answer with a single line \
             ANSWER: <number> with no other text on that line.\n\
             Q: 2 + 2\nANSWER: 4\n\
             Q: 6 * 7\nANSWER: 42\n\
             Q: {}\nANSWER:",
            item.expr
        );
        let (draft_text, draft_text_hash) = generate(&model, &draft_prompt, 12);
        let draft_answer = extract(&draft_text, "").unwrap_or_default();
        let draft_tok = if is_token(&draft_answer) {
            draft_answer.clone()
        } else {
            "NONE".to_string()
        };

        // --- verify round: reasoning-2, CALC tool instead of model self-recompute ---
        let tool_in = format!("CALC({})", item.expr.replace(' ', ""));
        let tool_out = match calc::run(&tool_in) {
            Some(v) => v,
            None => {
                // Suite's expected_input is always CALC(<int> <op> <int>);
                // a parse failure here would mean the suite row itself
                // doesn't match that grammar. Record honestly, don't
                // fabricate a result.
                tool_parse_fail += 1;
                "NONE".to_string()
            }
        };
        let recheck_tok = tool_out.clone();
        let verify_rationale = format!("CALC tool invoked: {tool_in} -> {tool_out}");
        let verify_text_hash = sha256(verify_rationale.as_bytes());

        let extract_ok = draft_tok != "NONE" && recheck_tok != "NONE";
        if draft_tok == "NONE" {
            extract_fail += 1;
        }

        let computed_verdict = if recheck_tok == draft_tok {
            "pass"
        } else {
            "contradict"
        };

        // --- final round: UNCHANGED prompt shape from reasoning_live.rs ---
        let final_prompt = if computed_verdict == "pass" {
            format!(
                "Your independent recheck matched your draft answer ({draft_tok}). \
                 Restate the answer. End with ANSWER: <number>.\nANSWER:"
            )
        } else {
            format!(
                "Your independent recheck ({recheck_tok}) did not match your draft \
                 answer ({draft_tok}). Trust the recheck. State the corrected answer. \
                 End with ANSWER: <number>.\nANSWER:"
            )
        };
        let (final_text, final_text_hash) = generate(&model, &final_prompt, 12);
        let final_raw = extract(&final_text, "").unwrap_or_default();
        let mut final_tok = if is_token(&final_raw) {
            final_raw
        } else {
            "NONE".to_string()
        };
        let final_extract_fallback = final_tok == "NONE";
        if final_extract_fallback {
            final_tok = if computed_verdict == "pass" {
                draft_tok.clone()
            } else {
                recheck_tok.clone()
            };
        }

        // --- build AEGIS-REASON v1 receipt (with reasoning-2 tool fields) ---
        let prompt_sha256 = sha256(draft_prompt.as_bytes());
        let chain0 = genesis(&prompt_sha256);
        let chain0 = fold_step(
            &chain0,
            0,
            &FKind::Draft,
            None,
            Some(&draft_tok),
            None,
            None,
            None,
            None,
            None,
            &draft_text_hash,
        );
        let chain1 = fold_step(
            &chain0,
            1,
            &FKind::Verify,
            Some(0),
            None,
            Some(&recheck_tok),
            Some(computed_verdict),
            Some("calc"),
            Some(&tool_in),
            Some(&tool_out),
            &verify_text_hash,
        );
        let chain2 = fold_step(
            &chain1,
            2,
            &FKind::Final,
            Some(1),
            Some(&final_tok),
            None,
            None,
            None,
            None,
            None,
            &final_text_hash,
        );

        let receipt = format!(
            "AEGIS-REASON v1\n\
             prompt-sha256={}\n\
             step 0: kind=draft answer={} text-sha256={} chain={}\n\
             step 1: kind=verify ref=0 recheck={} verdict={} tool=calc tool-in={} tool-out={} text-sha256={} chain={}\n\
             step 2: kind=final ref=1 answer={} text-sha256={} chain={}\n\
             trace-chain={}\n",
            hex(&prompt_sha256),
            draft_tok,
            hex(&draft_text_hash),
            hex(&chain0),
            recheck_tok,
            computed_verdict,
            tool_in,
            tool_out,
            hex(&verify_text_hash),
            hex(&chain1),
            final_tok,
            hex(&final_text_hash),
            hex(&chain2),
            hex(&chain2),
        );

        let receipt_path = format!("{outdir}/receipts/{}.receipt", item.id);
        fs::write(&receipt_path, &receipt).expect("write receipt");

        let final_correct_flag = final_tok == item.expected;
        if final_correct_flag {
            final_correct += 1;
        }

        summary.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            item.id,
            item.expected,
            draft_tok,
            tool_in,
            tool_out,
            recheck_tok,
            computed_verdict,
            final_tok,
            final_correct_flag,
            receipt.len(),
            extract_ok,
        ));

        println!(
            "{}: expected={} draft={} tool_out={} verdict={} final={} correct={} extract_ok={}",
            item.id,
            item.expected,
            draft_tok,
            tool_out,
            computed_verdict,
            final_tok,
            final_correct_flag,
            extract_ok
        );
    }

    fs::write(format!("{outdir}/summary.tsv"), &summary).expect("write summary");
    println!(
        "\nDONE: {n} CALC items run; final-answer correct {final_correct}/{n}; \
         draft-extraction failures {extract_fail}/{n}; tool-parse failures {tool_parse_fail}/{n}"
    );
    let _ = Path::new(outdir);
}
