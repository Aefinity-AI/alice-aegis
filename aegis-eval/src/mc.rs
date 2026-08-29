//! Multiple-choice eval mode (`--mc`): lm-eval-harness style scoring on
//! explicit token ids, so the engine and the transformers reference consume
//! byte-identical inputs.
//!
//! Per item, per choice k: teacher-forced NLL over `ctx_ids + choice_ids[k]`,
//! summed over ONLY the continuation positions. Implementation: two
//! `calculate_perplexity` runs are differenced —
//!
//!   cont_nll[k] = totalNLL(ctx + choice_k) - totalNLL(ctx)
//!
//! Both runs restart the engine at position 0 over the same ctx prefix, so
//! the prefix NLL terms are bitwise identical and the subtraction isolates
//! the continuation sum exactly (up to the exp/ln round-trip inside
//! `calculate_perplexity`, ~1e-15 relative — far below model numerics).
//! This keeps aegis-core untouched: no new engine API, no second code path
//! that could drift from the one G4a validated.
//!
//! Scores (stated in the results header):
//!   acc      : predict argmin_k cont_nll[k]                (raw sum)
//!   acc_norm : predict argmax_k (-cont_nll[k] / utf8_byte_len(choice_text_k))
//!              (lm-eval byte-length normalization; byte lengths are
//!              precomputed in the items file so both sides use identical
//!              denominators)
//! Ties break toward the lower choice index on both sides.

use aegis_core::cis_infer::{CisEngine, CisMode, CisModel};
use aegis_core::inference::TernaryInferenceEngine;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

pub struct McItem {
    pub id: String,
    pub answer_idx: usize,
    pub ctx_ids: Vec<u32>,
    pub choice_ids: Vec<Vec<u32>>,
    pub choice_byte_lens: Vec<usize>,
}

/// Split one JSON object line into `(key, raw value span)` pairs, respecting
/// strings/escapes and bracket depth. Purpose-built for the mc_prep.py items
/// schema; anything structurally unexpected is a hard error, never a guess.
fn top_level_fields(line: &str) -> Result<Vec<(String, &str)>, String> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    let skip_ws = |i: &mut usize| {
        while *i < bytes.len() && (bytes[*i] as char).is_ascii_whitespace() {
            *i += 1;
        }
    };
    skip_ws(&mut i);
    if i >= bytes.len() || bytes[i] != b'{' {
        return Err("line does not start with '{'".into());
    }
    i += 1;
    let mut fields = Vec::new();
    loop {
        skip_ws(&mut i);
        if i < bytes.len() && bytes[i] == b'}' {
            return Ok(fields);
        }
        // key string
        if i >= bytes.len() || bytes[i] != b'"' {
            return Err(format!("expected key quote at byte {i}"));
        }
        let (key, after) = parse_json_string(line, i)?;
        i = after;
        skip_ws(&mut i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err(format!("expected ':' at byte {i}"));
        }
        i += 1;
        skip_ws(&mut i);
        // value span: scan to ',' or '}' at depth 0, skipping strings
        let start = i;
        let mut depth = 0i32;
        while i < bytes.len() {
            match bytes[i] {
                b'"' => {
                    let (_, after) = parse_json_string(line, i)?;
                    i = after;
                    continue;
                }
                b'[' | b'{' => depth += 1,
                b']' | b'}' if depth > 0 => depth -= 1,
                b',' | b'}' if depth == 0 => break,
                _ => {}
            }
            i += 1;
        }
        if i >= bytes.len() {
            return Err("unterminated value".into());
        }
        fields.push((key, line[start..i].trim_end()));
        if bytes[i] == b',' {
            i += 1;
        }
    }
}

/// Parse the JSON string starting at byte `start` (which must be `"`).
/// Returns (decoded string, index one past the closing quote).
fn parse_json_string(line: &str, start: usize) -> Result<(String, usize), String> {
    let bytes = line.as_bytes();
    debug_assert_eq!(bytes[start], b'"');
    let mut out = String::new();
    let mut i = start + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Ok((out, i + 1)),
            b'\\' => {
                i += 1;
                let esc = *bytes.get(i).ok_or("dangling escape")?;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'u' => {
                        let hex = line.get(i + 1..i + 5).ok_or("truncated \\u escape")?;
                        let cp = u32::from_str_radix(hex, 16)
                            .map_err(|e| format!("bad \\u escape {hex:?}: {e}"))?;
                        // Surrogate pairs are not expected in item ids; refuse
                        // rather than silently mangle.
                        let ch = char::from_u32(cp)
                            .ok_or(format!("\\u{hex} is not a scalar value (surrogate?)"))?;
                        out.push(ch);
                        i += 4;
                    }
                    other => return Err(format!("unsupported escape \\{}", other as char)),
                }
                i += 1;
            }
            _ => {
                // multi-byte UTF-8: copy the whole char
                let ch = line[i..].chars().next().ok_or("bad utf-8 boundary")?;
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    Err("unterminated string".into())
}

fn parse_usize_array(raw: &str) -> Result<Vec<usize>, String> {
    let inner = raw
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("not an array: {raw:.40}"))?;
    let inner = inner.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("bad integer {:?}: {e}", s.trim()))
        })
        .collect()
}

fn parse_u32_array(raw: &str) -> Result<Vec<u32>, String> {
    Ok(parse_usize_array(raw)?
        .into_iter()
        .map(|v| v as u32)
        .collect())
}

fn parse_nested_u32_arrays(raw: &str) -> Result<Vec<Vec<u32>>, String> {
    let inner = raw
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("not an array: {raw:.40}"))?;
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, b) in inner.bytes().enumerate() {
        match b {
            b'[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b']' => {
                depth -= 1;
                if depth == 0 {
                    out.push(parse_u32_array(&inner[start..=i])?);
                }
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err("unbalanced nested array".into());
    }
    Ok(out)
}

impl McItem {
    pub fn parse(line: &str) -> Result<Self, String> {
        let fields = top_level_fields(line)?;
        let get = |name: &str| -> Result<&str, String> {
            fields
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| *v)
                .ok_or_else(|| format!("missing field {name:?}"))
        };
        let (id, _) = parse_json_string(get("id")?, 0)?;
        let item = McItem {
            id,
            answer_idx: get("answer_idx")?
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("answer_idx: {e}"))?,
            ctx_ids: parse_u32_array(get("ctx_ids")?)?,
            choice_ids: parse_nested_u32_arrays(get("choice_ids")?)?,
            choice_byte_lens: parse_usize_array(get("choice_byte_lens")?)?,
        };
        if item.choice_ids.is_empty()
            || item.choice_ids.len() != item.choice_byte_lens.len()
            || item.answer_idx >= item.choice_ids.len()
        {
            return Err(format!(
                "inconsistent item {}: {} choices, {} byte lens, answer_idx {}",
                item.id,
                item.choice_ids.len(),
                item.choice_byte_lens.len(),
                item.answer_idx
            ));
        }
        Ok(item)
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Total teacher-forced NLL (nats) over predictions 1..len-1, recovered from
/// `calculate_perplexity` (which returns exp(total/(len-1))). Exact because
/// `ids.len() <= window` is pre-checked by the caller, so the count inside
/// matches `len-1`.
fn total_nll(engine: &mut TernaryInferenceEngine<'_>, ids: &[u32]) -> Result<f64, String> {
    let ppl = engine.calculate_perplexity(ids);
    if !ppl.is_finite() || ppl <= 0.0 {
        return Err(format!(
            "calculate_perplexity returned {ppl} on a {}-token sequence \
             (NaN = out-of-vocab target id; 0 = sequence too short)",
            ids.len()
        ));
    }
    Ok(ppl.ln() * (ids.len() - 1) as f64)
}

/// Shared MC scoring loop: everything except the "how do I get total NLL for
/// a token sequence" step is identical between the float path (`run_mc`) and
/// the CIS-1 full-integer path (`run_mc_cis_full`) — same items file format,
/// same acc/acc_norm definitions, same JSONL schema. `total_nll_fn` is the
/// only thing that differs; it is handed a token sequence and returns the
/// summed teacher-forced NLL (nats) over predictions 1..len-1.
fn run_mc_generic(
    items_path: &str,
    out_path: &str,
    window: usize,
    vocab: u32,
    vocab_len_for_caveat: usize,
    mode_label: &str,
    mut total_nll_fn: impl FnMut(&[u32]) -> Result<f64, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read_to_string(items_path)?;
    let mut out = BufWriter::new(File::create(out_path)?);

    let norm_def = "acc: pred = argmin_k cont_nll[k] (raw sum over continuation \
                    tokens); acc_norm: pred = argmax_k(-cont_nll[k] / \
                    utf8_byte_len(choice_text_k)); ties -> lower index";
    println!("MC mode ({mode_label}): items={items_path} out={out_path}");
    println!("Scoring: {norm_def}");
    writeln!(
        out,
        "{{\"header\":\"aegis-eval --mc\",\"mode\":\"{}\",\"items\":\"{}\",\"normalization\":\"{}\"}}",
        json_escape(mode_label),
        json_escape(items_path),
        json_escape(norm_def)
    )?;

    let t0 = Instant::now();
    let mut n = 0usize;
    let mut n_correct_raw = 0usize;
    let mut n_correct_norm = 0usize;
    let mut tokens_forwarded = 0usize;

    for (line_no, line) in data.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let item =
            McItem::parse(line).map_err(|e| format!("{items_path} line {}: {e}", line_no + 1))?;

        if item.ctx_ids.len() < 2 {
            return Err(format!(
                "item {}: ctx_ids too short ({})",
                item.id,
                item.ctx_ids.len()
            )
            .into());
        }
        for (k, cont) in item.choice_ids.iter().enumerate() {
            if cont.is_empty() {
                return Err(format!("item {} choice {k}: empty continuation", item.id).into());
            }
            let seq_len = item.ctx_ids.len() + cont.len();
            if seq_len > window {
                return Err(format!(
                    "item {} choice {k}: sequence {} tokens exceeds KV window {} \
                     (calculate_perplexity would silently clamp — refusing)",
                    item.id, seq_len, window
                )
                .into());
            }
            if item.choice_byte_lens[k] == 0 {
                return Err(format!("item {} choice {k}: zero byte length", item.id).into());
            }
        }
        if let Some(&bad) = item
            .ctx_ids
            .iter()
            .chain(item.choice_ids.iter().flatten())
            .find(|&&t| t >= vocab)
        {
            return Err(format!("item {}: token id {bad} >= vocab {vocab}", item.id).into());
        }

        let ctx_nll =
            total_nll_fn(&item.ctx_ids).map_err(|e| format!("item {} ctx: {e}", item.id))?;
        tokens_forwarded += item.ctx_ids.len();

        let mut cont_nll = Vec::with_capacity(item.choice_ids.len());
        let mut cont_per_tok = Vec::with_capacity(item.choice_ids.len());
        for (k, cont) in item.choice_ids.iter().enumerate() {
            let mut seq = item.ctx_ids.clone();
            seq.extend_from_slice(cont);
            let total =
                total_nll_fn(&seq).map_err(|e| format!("item {} choice {k}: {e}", item.id))?;
            tokens_forwarded += seq.len();
            let nll = total - ctx_nll;
            cont_nll.push(nll);
            cont_per_tok.push(nll / cont.len() as f64);
        }

        let argbest = |score: &dyn Fn(usize) -> f64| -> usize {
            let mut best = 0usize;
            for k in 1..cont_nll.len() {
                if score(k) > score(best) {
                    best = k;
                }
            }
            best
        };
        let pred_raw = argbest(&|k| -cont_nll[k]);
        let pred_norm = argbest(&|k| -cont_nll[k] / item.choice_byte_lens[k] as f64);
        let correct_raw = pred_raw == item.answer_idx;
        let correct_norm = pred_norm == item.answer_idx;

        n += 1;
        n_correct_raw += correct_raw as usize;
        n_correct_norm += correct_norm as usize;

        let fmt_vec = |v: &[f64]| {
            v.iter()
                .map(|x| format!("{x:.6}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        writeln!(
            out,
            "{{\"id\":\"{}\",\"answer_idx\":{},\"choice_nll\":[{}],\"choice_nll_per_token\":[{}],\
             \"choice_cont_tokens\":[{}],\"choice_byte_lens\":[{}],\"pred_raw\":{},\"pred_norm\":{},\
             \"correct_raw\":{},\"correct_norm\":{}}}",
            json_escape(&item.id),
            item.answer_idx,
            fmt_vec(&cont_nll),
            fmt_vec(&cont_per_tok),
            item.choice_ids
                .iter()
                .map(|c| c.len().to_string())
                .collect::<Vec<_>>()
                .join(","),
            item.choice_byte_lens
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(","),
            pred_raw,
            pred_norm,
            correct_raw,
            correct_norm
        )?;
        out.flush()?;
        println!(
            "[{n}] {} pred_raw={pred_raw} pred_norm={pred_norm} gold={} nll=[{}] | running acc {:.3} acc_norm {:.3} | {:.0}s",
            item.id,
            item.answer_idx,
            fmt_vec(&cont_nll),
            n_correct_raw as f64 / n as f64,
            n_correct_norm as f64 / n as f64,
            t0.elapsed().as_secs_f64()
        );
    }

    if n == 0 {
        return Err("no items found in items file".into());
    }
    let acc = n_correct_raw as f64 / n as f64;
    let acc_norm = n_correct_norm as f64 / n as f64;
    let dt = t0.elapsed().as_secs_f64();

    writeln!(
        out,
        "{{\"summary\":true,\"mode\":\"{}\",\"n\":{n},\"acc\":{acc:.6},\"acc_norm\":{acc_norm:.6},\
         \"tokens_forwarded\":{tokens_forwarded},\"wall_s\":{dt:.1}}}",
        json_escape(mode_label)
    )?;
    out.flush()?;

    println!("--------------------------------------------------");
    println!("MC summary ({mode_label}): n={n} acc={acc:.4} ({n_correct_raw}/{n}) acc_norm={acc_norm:.4} ({n_correct_norm}/{n})");
    println!(
        "tokens forwarded: {tokens_forwarded} | wall {dt:.1}s ({:.2} tok/s)",
        tokens_forwarded as f64 / dt
    );
    println!("--------------------------------------------------");
    crate::print_caveat(vocab_len_for_caveat);
    Ok(())
}

/// Float-path MC (unchanged behavior/output vs the pre-C2a harness): scores
/// via `TernaryInferenceEngine::calculate_perplexity`.
pub fn run_mc(
    engine: &mut TernaryInferenceEngine<'_>,
    items_path: &str,
    out_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let window = engine.config.max_position_embeddings;
    let vocab = engine.config.vocab_size as u32;
    let vocab_len = engine.tokenizer.vocab_len();
    run_mc_generic(
        items_path,
        out_path,
        window,
        vocab,
        vocab_len,
        "float",
        |ids| total_nll(engine, ids),
    )
}

/// CIS-1 FULL-INTEGER path MC: same items file, same acc/acc_norm scoring,
/// but continuation NLL is recovered from `CisEngine::calculate_perplexity_int`
/// (mode `FullInt`) instead of the float engine — the same all-integer
/// ROPE-I/SOFTMAX-I/ACT-I forward pass `run_cis_full`'s PPL number uses.
/// `CisPplResult` only reports `ppl` (=exp(total_nll/scored)); `total_nll` is
/// recovered exactly the same way `total_nll()` above recovers it from the
/// float engine's `calculate_perplexity`.
pub fn run_mc_cis_full(
    engine: &TernaryInferenceEngine<'_>,
    items_path: &str,
    out_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let window = engine.config.max_position_embeddings;
    let vocab = engine.config.vocab_size as u32;
    let vocab_len = engine.tokenizer.vocab_len();

    let model = CisModel::new(engine.pipeline(), &engine.config)
        .map_err(|e| format!("CIS model conversion: {e}"))?;
    let mut cis = CisEngine::new_with_mode(&model, CisMode::FullInt);

    run_mc_generic(
        items_path,
        out_path,
        window,
        vocab,
        vocab_len,
        "full-int",
        |ids| {
            let r = cis.calculate_perplexity_int(ids);
            if !r.ppl.is_finite() || r.ppl <= 0.0 || r.scored == 0 {
                return Err(format!(
                    "calculate_perplexity_int returned ppl={} scored={} on a {}-token sequence \
                 (NaN = out-of-vocab target id; 0 = sequence too short)",
                    r.ppl,
                    r.scored,
                    ids.len()
                ));
            }
            Ok(r.ppl.ln() * r.scored as f64)
        },
    )
}
