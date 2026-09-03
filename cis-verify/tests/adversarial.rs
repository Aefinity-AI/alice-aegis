//! E20 — adversarial receipt corpus against the standalone `cis-verify`.
//!
//! Pre-registered (lab plan `2026-09-02-LAB-PLAN-four-axes.md` §2.1):
//!
//! > PASS = `cis-verify` rejects **30/30** crafted-invalid cases with a
//! > specific, correct error, **and** still accepts the known-good
//! > receipts unchanged. FAIL = any accepted forgery.
//!
//! Five attack families, six cases each, all built against the real in-repo
//! M7 artifacts (`model-lab/tinybit/m7_final_gate_work/artifacts/`) and
//! either the vendored golden receipt or receipts minted here by driving
//! `cis_verify`'s own engine:
//!
//! - **(a) replay** — splice a different prompt / artifact hash / foreign
//!   token stream into an otherwise-valid receipt.
//! - **(b) forgery** — flip digest bits, and (harder) re-fold the whole
//!   witness chain over falsified logits so the receipt is *internally*
//!   perfectly self-consistent.
//! - **(c) truncation** — drop trailing witness-chain entries, with and
//!   without repairing the dependent fields.
//! - **(d) wrong-artifact** — substitute a structurally identical but
//!   different MODEL.SAF / EMBED.BIN / VOCAB.BIN, including the strong form
//!   where the attacker also rewrites the receipt's artifact hashes and
//!   re-derives the chain header so the cheap hash gate cannot fire.
//! - **(e) mode-downgrade** — claim the canonical FullInt logit stream while
//!   committing to values a cheaper numeric path would have produced
//!   (coarser grid, 32-bit accumulator, rescaled, top-k-only, short-fold).
//!
//! Every case is *semantically plausible*: nothing here is random fuzz. The
//! E family in particular keeps the real token ids and the real `cis-digest`,
//! so only the full-logit chain can catch it — which is the whole point of
//! committing to logits rather than tokens.
//!
//! Rule C: reads only the already-vendored fixture under `tests/fixtures/`
//! and the in-repo M7 artifacts; writes nothing under `tests/golden/` or
//! `docs/hardware_logs/`. Rule A: this file produces counts and pass/fail
//! only — no timing of any kind.

use cis_verify::config::ModelConfig;
use cis_verify::forward::{CisEngine, CisModel};
use cis_verify::ops::argmax_i64;
use cis_verify::receipt::{Receipt, cis_digest_of};
use cis_verify::safetensors::SafeTensors;
use cis_verify::sha256::sha256;
use cis_verify::verify::{VerifyOutcome, verify};
use cis_verify::vocab::Tokenizer;
use cis_verify::witness::{WitnessChain, WitnessHeader};

const GOLDEN: &str = include_str!("fixtures/witness_v1_m7_once64.receipt");

/// `model` line of `tests/golden/witness_v1_bitnet2b_once64.receipt` (A39) —
/// a real, foreign, well-formed artifact hash, so the value the replay family
/// splices in is one an adversary could actually have in hand rather than an
/// invented string. Pinned rather than `include_str!`d because this crate
/// packages standalone (`Cargo.toml` `include`) and must build without the
/// rest of the repo; `foreign_model_sha_matches_the_2b_golden_receipt` below
/// re-checks it against the file whenever the checkout does have it.
const FOREIGN_MODEL_SHA: &str = "facb3597665603ba45730cc1f70ba6d82f53473d97f04fd039ca4296a45868db";

const BASE_PROMPT: &str = "Once upon a time";
/// Kept small on purpose: the light lane budgets RSS, and every case below
/// costs one full replay. 12 steps is enough for prefix/suffix/step-order
/// attacks to be distinguishable.
const BASE_MAXTOK: usize = 12;

// ---------------------------------------------------------------------------
// artifacts
// ---------------------------------------------------------------------------

fn artifact_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../model-lab/tinybit/m7_final_gate_work/artifacts")
        .join(name)
}

fn load_m7() -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let m = artifact_path("MODEL.SAF");
    let e = artifact_path("EMBED.BIN");
    let v = artifact_path("VOCAB.BIN");
    if !m.exists() || !e.exists() || !v.exists() {
        return None;
    }
    Some((
        std::fs::read(m).unwrap(),
        std::fs::read(e).unwrap(),
        std::fs::read(v).unwrap(),
    ))
}

/// First byte offset of the tensor-data region of a safetensors buffer:
/// 8-byte little-endian header length, then that many JSON bytes. Mutating
/// at or after this offset leaves the container structurally valid (same
/// tensor names, shapes and offsets) and changes only weights — i.e. a
/// *different model in the same shape*, which is the artifact-substitution
/// an adversary would actually attempt.
fn safetensors_data_start(buf: &[u8]) -> usize {
    let n = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    8 + n
}

/// Return a VOCAB.BIN whose string-table record for `id` has been rewritten
/// to `new` (same byte length, so every later record keeps its offset — the
/// container stays structurally valid and only the tokenizer's *meaning*
/// changes). This is the vocab substitution that actually matters: a
/// mutation in an unused merges entry is not an attack, it is a different
/// file that computes the same thing, and the verifier is right to accept it
/// (see the E20 evidence note on case D22).
fn vocab_with_token_string_replaced(v: &[u8], id: u32, new: &str) -> Vec<u8> {
    let num = u32::from_le_bytes(v[4..8].try_into().unwrap());
    assert!(id < num, "token id out of range");
    let mut off = 8usize;
    for cur in 0..num {
        let len = u16::from_le_bytes([v[off], v[off + 1]]) as usize;
        off += 2;
        if cur == id {
            assert_eq!(
                len,
                new.len(),
                "replacement string must keep the record length"
            );
            let mut out = v.to_vec();
            out[off..off + len].copy_from_slice(new.as_bytes());
            return out;
        }
        off += len;
    }
    unreachable!()
}

// ---------------------------------------------------------------------------
// minting: drive cis-verify's own engine and keep every step's full logits
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Step {
    token: u32,
    logits: Vec<i64>,
}

struct Run {
    prompt: String,
    prompt_ids: Vec<u32>,
    steps: Vec<Step>,
    model_sha: [u8; 32],
    embed_sha: [u8; 32],
    vocab_sha: [u8; 32],
}

/// Reproduces `forward::run_decode`'s step order exactly (prefill on prompt
/// ids, then per step: `decode_logits` → argmax → fold → `forward_step_int`),
/// but retains each step's complete i64 logit vector so the tests below can
/// re-fold a *self-consistent* chain over falsified values.
fn mint(m: &[u8], e: &[u8], v: &[u8], prompt: &str, maxtok: usize) -> Run {
    let tensors = SafeTensors::deserialize(m).expect("MODEL.SAF parses");
    let cfg_json = tensors
        .metadata_field("aegis_config")
        .expect("metadata readable")
        .expect("aegis_config present");
    let config = ModelConfig::from_json(&cfg_json).expect("aegis_config parses");
    let model = CisModel::new(&tensors, e, &config).expect("CIS model builds");
    let tokenizer = Tokenizer::new(v).expect("VOCAB.BIN parses");

    let prompt_ids = tokenizer.encode(prompt);
    let mut engine = CisEngine::new(&model);
    let mut pos = 0usize;
    for &t in &prompt_ids {
        engine.forward_step_int(t, pos);
        pos += 1;
    }
    let mut steps = Vec::with_capacity(maxtok);
    for _ in 0..maxtok {
        let (token, logits) = {
            let l = engine.decode_logits();
            (argmax_i64(l), l.to_vec())
        };
        steps.push(Step { token, logits });
        engine.forward_step_int(token, pos);
        pos += 1;
    }

    Run {
        prompt: prompt.to_string(),
        prompt_ids,
        steps,
        model_sha: sha256(m),
        embed_sha: sha256(e),
        vocab_sha: sha256(v),
    }
}

/// Build receipt text from arbitrary (possibly falsified) parts, always
/// internally self-consistent: the chain is folded from the given header
/// fields over the given steps, and `cis-digest` is the FNV fold over the
/// prompt ids then the given token ids. An attacker with the model can do
/// exactly this much; the question the corpus asks is whether that is enough.
fn forge(
    run: &Run,
    maxtok: u64,
    steps: &[Step],
    fold_logits: impl Fn(usize, &[i64]) -> Vec<i64>,
    fold_steps: usize,
    shas: ([u8; 32], [u8; 32], [u8; 32]),
) -> String {
    let (model_sha, embed_sha, vocab_sha) = shas;
    let header = WitnessHeader {
        model_sha: &model_sha,
        embed_sha: &embed_sha,
        vocab_sha: &vocab_sha,
        max_new: maxtok,
        prompt: run.prompt.as_bytes(),
    };
    let mut chain = WitnessChain::from_header(&header);
    for (i, s) in steps.iter().enumerate().take(fold_steps) {
        chain.fold_step(s.token, &fold_logits(i, &s.logits));
    }
    let ids: Vec<u32> = steps.iter().map(|s| s.token).collect();
    let cis_digest = cis_digest_of(&run.prompt_ids, &ids);
    Receipt {
        model_sha,
        embed_sha,
        vocab_sha,
        maxtok,
        prompt: run.prompt.as_bytes().to_vec(),
        prompt_toks: run.prompt_ids.len() as u64,
        gen_toks: ids.len() as u64,
        token_ids: ids,
        cis_digest,
        chain: chain.digest(),
    }
    .format()
}

/// The honest case: a receipt for exactly what the engine did.
fn genuine(run: &Run, maxtok: u64, n_steps: usize) -> String {
    let steps = &run.steps[..n_steps];
    forge(
        run,
        maxtok,
        steps,
        |_, l| l.to_vec(),
        n_steps,
        (run.model_sha, run.embed_sha, run.vocab_sha),
    )
}

fn shas_of(m: &[u8], e: &[u8], v: &[u8]) -> ([u8; 32], [u8; 32], [u8; 32]) {
    (sha256(m), sha256(e), sha256(v))
}

fn alloc_zeroed_like(l: &[i64]) -> Vec<i64> {
    vec![0i64; l.len()]
}

/// Guard against a *degenerate* artifact-substitution case: if the crafted
/// receipt happens to be exactly the receipt those substituted artifacts
/// would genuinely produce, then accepting it is correct behaviour and the
/// case proves nothing. (This guard is here because two first-draft cases —
/// an unused-merges-table mutation and an i32 clamp that fit inside the real
/// logit range — were degenerate in exactly this way and read as bypasses.)
fn assert_substitution_is_material(sm: &[u8], se: &[u8], sv: &[u8], crafted: &str, id: &str) {
    let real = mint(sm, se, sv, BASE_PROMPT, BASE_MAXTOK);
    let honest = genuine(&real, BASE_MAXTOK as u64, BASE_MAXTOK);
    assert_ne!(
        honest, crafted,
        "{id} is degenerate: the crafted receipt IS the honest receipt for the \
         substituted artifacts, so those artifacts compute the same thing"
    );
}

// ---------------------------------------------------------------------------
// case table
// ---------------------------------------------------------------------------

/// Which artifact bytes the verifier is handed for a case.
enum Art {
    /// The pristine M7 triple.
    Pristine,
    /// A substituted triple (model, embed, vocab), owned by the case.
    Sub(Vec<u8>, Vec<u8>, Vec<u8>),
}

struct Case {
    id: &'static str,
    family: &'static str,
    what: &'static str,
    receipt: String,
    art: Art,
    /// Pre-registered acceptable `FailedField::name()` values. More than one
    /// only where the true first divergence is genuinely data-dependent; the
    /// harness records which one actually fired.
    expect: &'static [&'static str],
}

// ---------------------------------------------------------------------------
// the corpus
// ---------------------------------------------------------------------------

fn build_corpus(m: &[u8], e: &[u8], v: &[u8]) -> Vec<Case> {
    let run = mint(m, e, v, BASE_PROMPT, BASE_MAXTOK);
    let good = genuine(&run, BASE_MAXTOK as u64, BASE_MAXTOK);
    let pristine_shas = (run.model_sha, run.embed_sha, run.vocab_sha);
    let mut cases: Vec<Case> = Vec::new();

    // --- (a) REPLAY -------------------------------------------------------

    // A01 — a different prompt that tokenizes to the SAME number of ids, so
    // the cheap prompt-toks pre-check cannot fire; everything else is the
    // real run's output.
    let alt_same_len = {
        let tokenizer = Tokenizer::new(v).unwrap();
        let want = run.prompt_ids.len();
        let cands = [
            "Twice upon a time",
            "Once upon a cake",
            "Once about a time",
            "Once upon a hill",
            "Once upon a星",
            "Deep in the wood",
        ];
        cands
            .iter()
            .copied()
            .find(|p| *p != BASE_PROMPT && tokenizer.encode(p).len() == want)
            .expect("need an alternative prompt with identical token count")
    };
    let mut alt_run = Run {
        prompt: alt_same_len.to_string(),
        prompt_ids: run.prompt_ids.clone(),
        steps: run.steps.clone(),
        model_sha: run.model_sha,
        embed_sha: run.embed_sha,
        vocab_sha: run.vocab_sha,
    };
    // Keep the ORIGINAL prompt's ids in the cis-digest fold: this is a replay
    // of the real run's outputs under a different claimed prompt.
    cases.push(Case {
        id: "A01",
        family: "replay",
        what: "prompt swapped for one with an identical token count; real token-ids/digest/chain kept",
        receipt: genuine(&alt_run, BASE_MAXTOK as u64, BASE_MAXTOK),
        art: Art::Pristine,
        // A different prompt drives a different decode; token-id is the first
        // field that can see it. cis-digest is the fallback if the greedy
        // stream happened to coincide.
        expect: &["token-id", "cis-digest"],
    });

    // A02 — prompt swapped for one with a DIFFERENT token count.
    alt_run.prompt = "A".to_string();
    cases.push(Case {
        id: "A02",
        family: "replay",
        what: "prompt swapped for one with a different token count (prompt-toks left stale)",
        receipt: genuine(&alt_run, BASE_MAXTOK as u64, BASE_MAXTOK),
        art: Art::Pristine,
        expect: &["prompt-tokenize"],
    });

    // A03 — cross-receipt splice: header/prompt from run 1, token stream and
    // both digests from run 2 (a real, valid receipt for a different prompt).
    let run2 = mint(m, e, v, "In the beginning", BASE_MAXTOK);
    let spliced = {
        let r1 = Receipt::parse(&good).unwrap();
        let r2 = Receipt::parse(&genuine(&run2, BASE_MAXTOK as u64, BASE_MAXTOK)).unwrap();
        Receipt {
            token_ids: r2.token_ids.clone(),
            cis_digest: r2.cis_digest,
            chain: r2.chain,
            ..r1
        }
        .format()
    };
    cases.push(Case {
        id: "A03",
        family: "replay",
        what: "token-ids + cis-digest + chain spliced in from a valid receipt for another prompt",
        receipt: spliced,
        art: Art::Pristine,
        expect: &["token-id"],
    });

    // A04 — a real foreign artifact hash (the 2B receipt's model line)
    // spliced into the M7 golden receipt.
    cases.push(Case {
        id: "A04",
        family: "replay",
        what: "foreign (BitNet-2B) model hash spliced into the golden M7 receipt",
        receipt: GOLDEN.replacen(
            "model 23cfad0a3c7de2b676e176755f0848cd2cad8e71e2cdffea3d903a232c96e973",
            &format!("model {FOREIGN_MODEL_SHA}"),
            1,
        ),
        art: Art::Pristine,
        expect: &["model"],
    });

    // A05 — a genuine 12-step receipt relabelled as a 24-step run: the
    // attacker claims more work than was witnessed.
    cases.push(Case {
        id: "A05",
        family: "replay",
        what: "genuine 12-step receipt relabelled maxtok=24 (gen-toks/token-ids untouched)",
        receipt: good.replacen(
            &format!("maxtok {BASE_MAXTOK}"),
            &format!("maxtok {}", BASE_MAXTOK * 2),
            1,
        ),
        art: Art::Pristine,
        expect: &["token-id"],
    });

    // A06 — the same valid receipt, artifacts presented in the wrong slots.
    cases.push(Case {
        id: "A06",
        family: "replay",
        what: "valid golden receipt verified with EMBED.BIN and VOCAB.BIN swapped",
        receipt: GOLDEN.to_string(),
        art: Art::Sub(m.to_vec(), v.to_vec(), e.to_vec()),
        expect: &["embed"],
    });

    // --- (b) FORGERY ------------------------------------------------------

    // B07 — flip the low bit of the final cis-digest.
    let b07 = {
        let mut r = Receipt::parse(&good).unwrap();
        r.cis_digest ^= 1;
        r.format()
    };
    cases.push(Case {
        id: "B07",
        family: "forgery",
        what: "low bit of cis-digest flipped, everything else genuine",
        receipt: b07,
        art: Art::Pristine,
        expect: &["cis-digest"],
    });

    // B08 — flip the low bit of the final chain digest.
    let b08 = {
        let mut r = Receipt::parse(&good).unwrap();
        r.chain[31] ^= 1;
        r.format()
    };
    cases.push(Case {
        id: "B08",
        family: "forgery",
        what: "low bit of the final chain digest flipped, everything else genuine",
        receipt: b08,
        art: Art::Pristine,
        expect: &["chain"],
    });

    // B09 — one logit bit flipped at one step, then the chain and digest
    // re-folded: the receipt is internally flawless, it just does not
    // describe what the model computes.
    cases.push(Case {
        id: "B09",
        family: "forgery",
        what: "one non-winning logit bit flipped at step 5; chain + digest re-folded (self-consistent)",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |i, l| {
                let mut l = l.to_vec();
                if i == 5 {
                    let win = argmax_i64(&l) as usize;
                    let victim = if win == 0 { l.len() - 1 } else { 0 };
                    l[victim] ^= 1;
                }
                l
            },
            BASE_MAXTOK,
            pristine_shas,
        ),
        art: Art::Pristine,
        expect: &["chain"],
    });

    // B10 — steer the argmax at step 3 to a different token, then re-fold
    // both the chain and the digest over the steered stream.
    let b10 = {
        let mut steps = run.steps.clone();
        let s = &mut steps[3];
        let win = argmax_i64(&s.logits) as usize;
        let target = if win == 0 { 1 } else { 0 };
        s.logits[target] = s.logits[win].saturating_add(1);
        s.token = argmax_i64(&s.logits);
        assert_ne!(s.token, run.steps[3].token, "steering did not move argmax");
        forge(
            &run,
            BASE_MAXTOK as u64,
            &steps,
            |_, l| l.to_vec(),
            BASE_MAXTOK,
            pristine_shas,
        )
    };
    cases.push(Case {
        id: "B10",
        family: "forgery",
        what: "argmax steered to a chosen token at step 3; chain + digest re-folded (self-consistent)",
        receipt: b10,
        art: Art::Pristine,
        expect: &["token-id"],
    });

    // B11 — a wholly fabricated logit stream of the right shape.
    let b11 = {
        let width = run.steps[0].logits.len();
        let steps: Vec<Step> = (0..BASE_MAXTOK)
            .map(|i| {
                let logits: Vec<i64> = (0..width)
                    .map(|j| ((i * 7919 + j * 104729) % 1_000_003) as i64 - 500_000)
                    .collect();
                Step {
                    token: argmax_i64(&logits),
                    logits,
                }
            })
            .collect();
        forge(
            &run,
            BASE_MAXTOK as u64,
            &steps,
            |_, l| l.to_vec(),
            BASE_MAXTOK,
            pristine_shas,
        )
    };
    cases.push(Case {
        id: "B11",
        family: "forgery",
        what: "entirely synthetic logit stream of the correct width; chain + digest re-folded",
        receipt: b11,
        art: Art::Pristine,
        expect: &["token-id"],
    });

    // B12 — the real steps, in reverse order; chain and digest re-folded so
    // the receipt is again internally consistent.
    let b12 = {
        let mut steps = run.steps.clone();
        steps.reverse();
        forge(
            &run,
            BASE_MAXTOK as u64,
            &steps,
            |_, l| l.to_vec(),
            BASE_MAXTOK,
            pristine_shas,
        )
    };
    cases.push(Case {
        id: "B12",
        family: "forgery",
        what: "real steps re-ordered (reversed); chain + digest re-folded (self-consistent)",
        receipt: b12,
        art: Art::Pristine,
        expect: &["token-id"],
    });

    // --- (c) TRUNCATION ---------------------------------------------------

    // C13 — drop trailing ids, leave gen-toks stale.
    let c13 = {
        let r = Receipt::parse(&good).unwrap();
        let kept: Vec<String> = r.token_ids[..BASE_MAXTOK - 3]
            .iter()
            .map(|t| t.to_string())
            .collect();
        let all: Vec<String> = r.token_ids.iter().map(|t| t.to_string()).collect();
        good.replacen(
            &format!("token-ids {}", all.join(",")),
            &format!("token-ids {}", kept.join(",")),
            1,
        )
    };
    assert_ne!(c13, good, "C13 rewrite did not apply");
    cases.push(Case {
        id: "C13",
        family: "truncation",
        what: "trailing token-ids dropped, gen-toks left stale",
        receipt: c13,
        art: Art::Pristine,
        expect: &["receipt-parse"],
    });

    // C14 — drop trailing ids AND repair gen-toks, but leave the two digests
    // describing the full run.
    let c14 = {
        let mut r = Receipt::parse(&good).unwrap();
        r.token_ids.truncate(BASE_MAXTOK - 3);
        r.gen_toks = r.token_ids.len() as u64;
        r.format()
    };
    cases.push(Case {
        id: "C14",
        family: "truncation",
        what: "trailing token-ids dropped and gen-toks repaired; cis-digest/chain left stale",
        receipt: c14,
        art: Art::Pristine,
        expect: &["token-id"],
    });

    // C15 — the hard one: a fully self-consistent 9-step prefix that still
    // claims maxtok=12, i.e. three witnessed steps silently discarded.
    cases.push(Case {
        id: "C15",
        family: "truncation",
        what: "self-consistent 9-step prefix (digest + chain re-folded) still claiming maxtok=12",
        receipt: genuine(&run, BASE_MAXTOK as u64, BASE_MAXTOK - 3),
        art: Art::Pristine,
        expect: &["token-id"],
    });

    // C16 — the file itself cut mid-line.
    cases.push(Case {
        id: "C16",
        family: "truncation",
        what: "receipt file cut mid-line at 60% of its length",
        receipt: good[..good.len() * 3 / 5].to_string(),
        art: Art::Pristine,
        expect: &["receipt-parse"],
    });

    // C17 — the chain line removed entirely.
    let c17: String = good
        .lines()
        .filter(|l| !l.starts_with("chain "))
        .map(|l| format!("{l}\n"))
        .collect();
    cases.push(Case {
        id: "C17",
        family: "truncation",
        what: "chain line removed entirely",
        receipt: c17,
        art: Art::Pristine,
        expect: &["receipt-parse"],
    });

    // C18 — the token-ids line removed entirely, gen-toks left claiming 12.
    let c18: String = good
        .lines()
        .filter(|l| !l.starts_with("token-ids "))
        .map(|l| format!("{l}\n"))
        .collect();
    cases.push(Case {
        id: "C18",
        family: "truncation",
        what: "token-ids line removed entirely, gen-toks still claims the full run",
        receipt: c18,
        art: Art::Pristine,
        expect: &["receipt-parse"],
    });

    // --- (d) WRONG-ARTIFACT -----------------------------------------------

    // A structurally identical MODEL.SAF whose weights differ: same header,
    // same tensor names/shapes/offsets, 64 mutated bytes deep in the data.
    let mut model_sub = m.to_vec();
    {
        let start = safetensors_data_start(&model_sub);
        let at = start + (model_sub.len() - start) / 3;
        for b in &mut model_sub[at..at + 64] {
            *b ^= 0xFF;
        }
    }
    // Same, but a single bit — enough for the hash gate, not necessarily
    // enough to change the forward pass (which is why D19 keeps the receipt
    // hash stale and expects the hash gate, not a replay divergence).
    let mut model_one_bit = m.to_vec();
    {
        let start = safetensors_data_start(&model_one_bit);
        model_one_bit[start + 7] ^= 0x01;
    }
    let mut embed_sub = e.to_vec();
    for b in &mut embed_sub[..65_536.min(e.len())] {
        *b ^= 0xA5;
    }
    // A VOCAB.BIN that tokenizes differently: the single-character record the
    // prompt's first byte maps through is rewritten, so `encode(prompt)` no
    // longer produces the witnessed ids.
    let vocab_sub = {
        let tokenizer = Tokenizer::new(v).unwrap();
        let id = tokenizer
            .get_token_id("O")
            .expect("M7 VOCAB.BIN carries a single-char 'O' record");
        vocab_with_token_string_replaced(v, id, "Q")
    };
    let model_trunc = m[..m.len() / 2].to_vec();
    let model_empty: Vec<u8> = Vec::new();

    cases.push(Case {
        id: "D19",
        family: "wrong-artifact",
        what: "single weight bit flipped in MODEL.SAF, receipt hash left stale",
        receipt: good.clone(),
        art: Art::Sub(model_one_bit, e.to_vec(), v.to_vec()),
        expect: &["model"],
    });

    // D20-D24: the strong form — the attacker also rewrites the receipt's
    // artifact hash and re-derives the chain header, so the cheap hash gate
    // cannot fire and only the replay can catch the substitution.
    let d20 = forge(
        &run,
        BASE_MAXTOK as u64,
        &run.steps,
        |_, l| l.to_vec(),
        BASE_MAXTOK,
        shas_of(&model_sub, e, v),
    );
    assert_substitution_is_material(&model_sub, e, v, &d20, "D20");
    cases.push(Case {
        id: "D20",
        family: "wrong-artifact",
        what: "64 weight bytes mutated in MODEL.SAF, receipt hash rewritten + chain header re-derived",
        receipt: d20,
        art: Art::Sub(model_sub, e.to_vec(), v.to_vec()),
        expect: &["token-id", "cis-digest", "chain"],
    });

    let d21 = forge(
        &run,
        BASE_MAXTOK as u64,
        &run.steps,
        |_, l| l.to_vec(),
        BASE_MAXTOK,
        shas_of(m, &embed_sub, v),
    );
    assert_substitution_is_material(m, &embed_sub, v, &d21, "D21");
    cases.push(Case {
        id: "D21",
        family: "wrong-artifact",
        what: "64 KiB of EMBED.BIN mutated, receipt hash rewritten + chain header re-derived",
        receipt: d21,
        art: Art::Sub(m.to_vec(), embed_sub, v.to_vec()),
        expect: &["token-id", "cis-digest", "chain"],
    });

    let d22 = forge(
        &run,
        BASE_MAXTOK as u64,
        &run.steps,
        |_, l| l.to_vec(),
        BASE_MAXTOK,
        shas_of(m, e, &vocab_sub),
    );
    assert_substitution_is_material(m, e, &vocab_sub, &d22, "D22");
    cases.push(Case {
        id: "D22",
        family: "wrong-artifact",
        what: "VOCAB.BIN record for a prompt character rewritten, receipt hash rewritten + chain header re-derived",
        receipt: d22,
        art: Art::Sub(m.to_vec(), e.to_vec(), vocab_sub),
        expect: &["prompt-tokenize", "token-id", "cis-digest", "chain"],
    });

    cases.push(Case {
        id: "D23",
        family: "wrong-artifact",
        what: "MODEL.SAF truncated to half, receipt hash rewritten (must not panic)",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| l.to_vec(),
            BASE_MAXTOK,
            shas_of(&model_trunc, e, v),
        ),
        art: Art::Sub(model_trunc, e.to_vec(), v.to_vec()),
        expect: &["artifact-load"],
    });

    cases.push(Case {
        id: "D24",
        family: "wrong-artifact",
        what: "MODEL.SAF empty, receipt hash rewritten (must not panic)",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| l.to_vec(),
            BASE_MAXTOK,
            shas_of(&model_empty, e, v),
        ),
        art: Art::Sub(model_empty, e.to_vec(), v.to_vec()),
        expect: &["artifact-load"],
    });

    // --- (e) MODE-DOWNGRADE -----------------------------------------------
    //
    // Every case here keeps the REAL token ids and therefore the real
    // cis-digest: the attacker's claim is "this is the canonical FullInt
    // run", and the tokens agree. Only the full-logit chain can tell that a
    // cheaper numeric path produced the committed values — so each case
    // pre-registers `chain` exactly, with no alternatives.

    cases.push(Case {
        id: "E25",
        family: "mode-downgrade",
        what: "logits committed on a 2^4-coarser grid (low 4 bits cleared); real tokens kept",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| l.iter().map(|x| x & !0xF).collect(),
            BASE_MAXTOK,
            pristine_shas,
        ),
        art: Art::Pristine,
        expect: &["chain"],
    });

    cases.push(Case {
        id: "E26",
        family: "mode-downgrade",
        what: "logits committed on a 2^20-coarser grid (low 20 bits cleared); real tokens kept",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| l.iter().map(|x| x & !0xF_FFFF).collect(),
            BASE_MAXTOK,
            pristine_shas,
        ),
        art: Art::Pristine,
        expect: &["chain"],
    });

    cases.push(Case {
        id: "E27",
        family: "mode-downgrade",
        what: "chain folded over a one-hot vector (argmax-only commitment posing as a full-logit one); real tokens kept",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| {
                let win = argmax_i64(l) as usize;
                let mut out = alloc_zeroed_like(l);
                out[win] = l[win];
                out
            },
            BASE_MAXTOK,
            pristine_shas,
        ),
        art: Art::Pristine,
        expect: &["chain"],
    });

    cases.push(Case {
        id: "E28",
        family: "mode-downgrade",
        what: "logits rescaled by an argmax-preserving arithmetic shift (>>1); real tokens kept",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| l.iter().map(|x| x >> 1).collect(),
            BASE_MAXTOK,
            pristine_shas,
        ),
        art: Art::Pristine,
        expect: &["chain"],
    });

    cases.push(Case {
        id: "E29",
        family: "mode-downgrade",
        what: "only a 256-entry prefix of each logit vector committed (top-k commitment); real tokens kept",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| l[..256.min(l.len())].to_vec(),
            BASE_MAXTOK,
            pristine_shas,
        ),
        art: Art::Pristine,
        expect: &["chain"],
    });

    cases.push(Case {
        id: "E30",
        family: "mode-downgrade",
        what: "final step omitted from the chain fold while all 12 token-ids are claimed",
        receipt: forge(
            &run,
            BASE_MAXTOK as u64,
            &run.steps,
            |_, l| l.to_vec(),
            BASE_MAXTOK - 1,
            pristine_shas,
        ),
        art: Art::Pristine,
        expect: &["chain"],
    });

    // Every pristine-artifact case must differ from a receipt the artifacts
    // would honestly produce — otherwise "the verifier accepted it" would
    // mean "the claim was true", not "the forgery worked".
    for c in &cases {
        if matches!(c.art, Art::Pristine) {
            assert_ne!(
                c.receipt, good,
                "{} is degenerate (== honest receipt)",
                c.id
            );
            assert_ne!(
                c.receipt, GOLDEN,
                "{} is degenerate (== golden receipt)",
                c.id
            );
        }
    }

    cases
}

// ---------------------------------------------------------------------------
// the pre-registered gate
// ---------------------------------------------------------------------------

#[test]
fn e20_thirty_crafted_invalid_receipts_are_all_rejected() {
    let Some((m, e, v)) = load_m7() else {
        panic!(
            "M7 artifacts absent — E20 cannot run vacuously. Expected \
             model-lab/tinybit/m7_final_gate_work/artifacts/{{MODEL.SAF,EMBED.BIN,VOCAB.BIN}}"
        );
    };

    let cases = build_corpus(&m, &e, &v);
    assert_eq!(cases.len(), 30, "corpus size is pre-registered at 30");

    let mut rejected = 0usize;
    let mut accepted: Vec<&str> = Vec::new();
    let mut wrong_field: Vec<String> = Vec::new();

    eprintln!("E20 adversarial receipt corpus — cis-verify");
    eprintln!("{:<5} {:<15} {:<16} detail", "id", "family", "fired");
    for c in &cases {
        let (mm, ee, vv): (&[u8], &[u8], &[u8]) = match &c.art {
            Art::Pristine => (&m, &e, &v),
            Art::Sub(a, b, d) => (a, b, d),
        };
        match verify(&c.receipt, mm, ee, vv) {
            VerifyOutcome::Pass { steps } => {
                eprintln!(
                    "{:<5} {:<15} {:<16} ACCEPTED ({steps} steps) — {}",
                    c.id, c.family, "**BYPASS**", c.what
                );
                accepted.push(c.id);
            }
            VerifyOutcome::Fail { field, detail } => {
                let name = field.name();
                let ok = c.expect.contains(&name);
                if ok {
                    rejected += 1;
                } else {
                    wrong_field.push(format!(
                        "{}: fired '{name}' (detail: {detail}), pre-registered {:?}",
                        c.id, c.expect
                    ));
                }
                let d: String = detail.chars().take(72).collect();
                eprintln!(
                    "{:<5} {:<15} {:<16} {}{d}",
                    c.id,
                    c.family,
                    name,
                    if ok { "" } else { "UNEXPECTED " }
                );
            }
        }
    }
    eprintln!("E20 result: {rejected}/30 rejected with a pre-registered field");

    assert!(
        accepted.is_empty(),
        "CRITICAL — cis-verify ACCEPTED forged receipt(s): {accepted:?}"
    );
    assert!(
        wrong_field.is_empty(),
        "rejected, but not on the pre-registered field:\n  {}",
        wrong_field.join("\n  ")
    );
    assert_eq!(rejected, 30, "pre-registered PASS line is 30/30");
}

// ---------------------------------------------------------------------------
// must NOT over-reject: the known-good side of the pre-registered line
// ---------------------------------------------------------------------------

#[test]
fn e20_known_good_receipts_still_verify() {
    let Some((m, e, v)) = load_m7() else {
        panic!("M7 artifacts absent — E20 cannot run vacuously");
    };

    // 1. The vendored golden M7 receipt (A32 lineage), unchanged.
    assert!(
        matches!(
            verify(GOLDEN, &m, &e, &v),
            VerifyOutcome::Pass { steps: 64 }
        ),
        "golden M7 receipt must still verify"
    );

    // 2. A receipt minted here for the same artifacts.
    let run = mint(&m, &e, &v, BASE_PROMPT, BASE_MAXTOK);
    let good = genuine(&run, BASE_MAXTOK as u64, BASE_MAXTOK);
    assert!(
        verify(&good, &m, &e, &v).is_pass(),
        "self-minted {BASE_MAXTOK}-step receipt must verify"
    );

    // 3. A legitimately SHORTER run is not a truncation attack: when maxtok
    //    is reduced with the chain header re-derived, the receipt describes a
    //    real, complete 9-step decode and must be accepted. (Contrast C15,
    //    which keeps maxtok=12 and is rejected.)
    let short = genuine(&run, (BASE_MAXTOK - 3) as u64, BASE_MAXTOK - 3);
    assert!(
        verify(&short, &m, &e, &v).is_pass(),
        "a genuine 9-step receipt must not be mistaken for a truncated 12-step one"
    );

    // 4. Round-tripping a receipt through parse/format must not change its
    //    verdict.
    let round = Receipt::parse(&good).unwrap().format();
    assert_eq!(round, good, "parse/format must be byte-identical");
}

/// DOCUMENTED GAP (not a bypass of any numeric claim): the v1-CIS receipt
/// carries no `mode` field, and unknown keys are ignored by design
/// (`receipt.rs`, mirroring `cis_witness.rs:191`). Nothing outside the ten
/// known fields is bound by the witness chain, so an attacker can attach
/// arbitrary *metadata* lines to a receipt that verifies. Every computational
/// claim — artifacts, prompt, token stream, full logit chain — remains bound
/// (families B and E above prove it), so this changes no verified quantity;
/// it is recorded here because a reader who sees `mode fullint` on a receipt
/// might believe the verifier checked it. It did not.
#[test]
fn e20_unknown_receipt_lines_are_not_bound_by_the_chain() {
    let Some((m, e, v)) = load_m7() else {
        panic!("M7 artifacts absent — E20 cannot run vacuously");
    };
    let mut misleading = GOLDEN.to_string();
    misleading.push_str("mode hybrid\n");
    misleading.push_str("attested-by nobody\n");
    assert!(
        verify(&misleading, &m, &e, &v).is_pass(),
        "unknown keys are ignored by design; if this ever fails the format gained a check"
    );
}

/// Provenance check for [`FOREIGN_MODEL_SHA`]: when the full repo is present,
/// the pinned constant must still be the 2B golden receipt's `model` line.
/// Skips (does not fail) in a standalone crate checkout, matching the
/// artifact-skip convention in `tests/artifact_hash_golden.rs`.
#[test]
fn foreign_model_sha_matches_the_2b_golden_receipt() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/golden/witness_v1_bitnet2b_once64.receipt");
    if !path.exists() {
        eprintln!("2B golden receipt not present — skipping provenance check");
        return;
    }
    let text = std::fs::read_to_string(&path).unwrap();
    let line = text
        .lines()
        .find(|l| l.starts_with("model "))
        .expect("2B golden receipt has a model line");
    assert_eq!(
        line,
        format!("model {FOREIGN_MODEL_SHA}"),
        "FOREIGN_MODEL_SHA drifted from tests/golden/witness_v1_bitnet2b_once64.receipt"
    );
}

/// Provenance for the two measurements cited in `docs/ledger/E20.md`'s
/// "Corrections" section — the reason each first-draft case was degenerate.
/// Both are counts/ranges, not timings (Rule A).
#[test]
fn e20_degenerate_case_measurements() {
    let Some((m, e, v)) = load_m7() else {
        panic!("M7 artifacts absent — E20 cannot run vacuously");
    };

    // (1) Why the first-draft E27 (clamp into i32) was a no-op: the M7 logit
    //     range over the witnessed steps lies entirely inside i32.
    let run = mint(&m, &e, &v, BASE_PROMPT, BASE_MAXTOK);
    let (mut lo, mut hi) = (i64::MAX, i64::MIN);
    for s in &run.steps {
        for &x in &s.logits {
            lo = lo.min(x);
            hi = hi.max(x);
        }
    }
    eprintln!(
        "M7 logits: width {} min {lo} max {hi} (i32::MAX {}) fits_in_i32 {}",
        run.steps[0].logits.len(),
        i32::MAX,
        lo >= i32::MIN as i64 && hi <= i32::MAX as i64
    );
    assert!(
        lo >= i32::MIN as i64 && hi <= i32::MAX as i64,
        "if this ever fails, an i32 clamp becomes a real mode-downgrade case"
    );

    // (2) Why the first-draft D22 (mutate VOCAB.BIN at the midpoint) hit the
    //     merges table rather than the string table: the string table ends
    //     well before the midpoint.
    let num = u32::from_le_bytes(v[4..8].try_into().unwrap());
    let mut off = 8usize;
    for _ in 0..num {
        let len = u16::from_le_bytes([v[off], v[off + 1]]) as usize;
        off += 2 + len;
    }
    eprintln!(
        "M7 VOCAB.BIN: num_tokens {num}, string table ends at byte {off}, file len {}",
        v.len()
    );
    assert!(
        off < v.len() / 2,
        "string table now spans the midpoint; the D22 note in docs/ledger/E20.md needs revising"
    );
}
