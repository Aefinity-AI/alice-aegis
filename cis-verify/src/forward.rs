//! The forward-pass glue (builder task 5, `docs/design/CIS_VERIFY_DESIGN.md`
//! §3.3/§6.2 item 5): embedding lookup → N decoder layers → final norm → LM
//! head → argmax, `FullInt` only (spec v0.3 / `docs/CIS-1_SPEC_v1.0.md` §5).
//! Mirrors `aegis-core/src/cis_infer.rs`'s `CisModel`/`CisEngine`, but only
//! the `CisMode::FullInt` branch — there is no Hybrid mode in this crate at
//! all (design doc §2.2: "Receipts under discussion are `FullInt`-only...
//! a verifier can refuse any receipt whose header/config implies Hybrid").
//!
//! Zero float, zero `libm`, anywhere in this file — the FullInt decode path
//! needs neither (design doc §2.2, §3.2). Every op comes from `crate::ops`
//! (§5.1-§5.6, §5.11) or `crate::attn` (§5.7-§5.10); this module is only the
//! per-layer plumbing between them (§5.12's grid assignments) plus container
//! parsing (MODEL.SAF tensor lookup, BF16 embedding rows).
//!
//! `core`-only integer arithmetic — no floats, no `libm`, no `unsafe`.

use alloc::{format, string::String, vec, vec::Vec};

use crate::attn::{ExpLut, RopeTableI, inv_sqrt_q30, relu2_q20, rope_apply_i, silu_q20, softmax_i};
use crate::config::{Activation, ModelConfig};
use crate::ops::{
    ActScale, F, FRatio, GQ, QScale64, argmax_i64, bf16_to_fixed, f32_to_fixed, f32_to_ratio,
    fix_q_vec, normq, quantq, rne_div, ternary_matvec_i8,
};
use crate::safetensors::SafeTensors;
use crate::vocab::Tokenizer;
use crate::witness::WitnessChain;

/// Full-integer attention: q/k/v fixed point, Q.16 (spec v0.3, identical to
/// `cis_infer.rs:51`).
pub const QK_F: u32 = 16;
/// Full-integer attention: score fixed point, Q.24 (spec v0.3, identical to
/// `cis_infer.rs:53`).
pub const SCORE_F: u32 = 24;
/// Full-integer attention: probability fixed point, Q0.15 (spec §2,
/// identical to `cis_infer.rs:55`).
pub const PROB_F: u32 = 15;

// ---------------------------------------------------------------------------
// Scale plumbing: weight-scale rational x activation scale -> a QScale64
// that lands a matvec accumulator exactly on a Q.`frac` fixed-point grid.
// Identical arithmetic to `cis_infer.rs:513-545`'s `fixed_qscale`/
// `residual_qscale`, minus the Hybrid-only f64 sibling (`dequant_scale_f64`).
// ---------------------------------------------------------------------------

/// OUT = ±rne(acc . wm.2^we . num/den . 2^frac). Returns the sign
/// separately; the magnitude ratio feeds `QScale64` exactly.
fn fixed_qscale(w: &FRatio, s: &ActScale, frac: u32) -> (bool, QScale64) {
    if w.m == 0 || s.num == 0 {
        return (false, QScale64::from_ratio(0, 1));
    }
    let mut num = (w.m as u128)
        .checked_mul(s.num)
        .expect("fixed_qscale: numerator overflow");
    let mut den = s.den;
    let sh = frac as i32 + w.e;
    if sh >= 0 {
        num = num
            .checked_shl(sh as u32)
            .filter(|v| v >> sh == (w.m as u128) * s.num)
            .expect("fixed_qscale: numerator shift overflow");
    } else {
        den = den
            .checked_shl((-sh) as u32)
            .filter(|v| v >> -sh == s.den)
            .expect("fixed_qscale: denominator shift overflow");
    }
    (w.neg, QScale64::from_ratio(num, den))
}

/// The residual-grid case (`frac = F`).
fn residual_qscale(w: &FRatio, s: &ActScale) -> (bool, QScale64) {
    fixed_qscale(w, s, F)
}

// ---------------------------------------------------------------------------
// Container-boundary helpers: BF16 table rows, gains, and the LM-head dot.
// Identical arithmetic to `cis_infer.rs:591-672`.
// ---------------------------------------------------------------------------

/// Norm-gain bytes (BF16 or F32, dtype derived from the length ratio) → Q.GQ
/// i32 gains. Identical to `cis_infer.rs:591-626`.
fn gains_to_q(bytes: &[u8], n: usize, what: &str) -> Result<Vec<i32>, String> {
    if n == 0 || bytes.len() < n {
        return Err(format!("{what}: {} bytes for {} gains", bytes.len(), n));
    }
    let elem = bytes.len() / n;
    let mut out = Vec::with_capacity(n);
    match elem {
        2 => {
            for i in 0..n {
                let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                let v = bf16_to_fixed(bits, GQ);
                if v.unsigned_abs() >= 1 << 31 {
                    return Err(format!("{what}: gain {i} out of Q.{GQ} i32 range"));
                }
                out.push(v as i32);
            }
        }
        4 => {
            for i in 0..n {
                let bits = u32::from_le_bytes([
                    bytes[i * 4],
                    bytes[i * 4 + 1],
                    bytes[i * 4 + 2],
                    bytes[i * 4 + 3],
                ]);
                let v = f32_to_fixed(bits, GQ);
                if v.unsigned_abs() >= 1 << 31 {
                    return Err(format!("{what}: gain {i} out of Q.{GQ} i32 range"));
                }
                out.push(v as i32);
            }
        }
        _ => return Err(format!("{what}: unsupported element size {elem}")),
    }
    Ok(out)
}

/// Identical to `cis_infer.rs:630-638`.
fn check_bf16_table(bytes: &[u8], count: usize, what: &str) -> Result<(), String> {
    if bytes.len() != count * 2 {
        return Err(format!(
            "{what}: {} bytes, expected {} BF16 values",
            bytes.len(),
            count
        ));
    }
    Ok(())
}

/// One BF16 table row → Q.F i64 values, exact RNE. Identical to
/// `cis_infer.rs:641-654`.
fn bf16_row_to_q(row: &[u8], out: &mut [i64]) {
    debug_assert_eq!(row.len(), out.len() * 2);
    let (chunks, _) = row.as_chunks::<2>();
    for (o, b) in out.iter_mut().zip(chunks) {
        let v = bf16_to_fixed(u16::from_le_bytes(*b), F);
        assert!(
            v.unsigned_abs() < 1 << 31,
            "bf16_row_to_q: value out of Q.{F} i32 range"
        );
        *o = v;
    }
}

/// Integer dot of i8 codes against one BF16 table row, converting each
/// element on the fly. Identical to `cis_infer.rs:660-672`.
fn dot_i8_bf16q(a: &[i8], row: &[u8]) -> i64 {
    debug_assert_eq!(a.len() * 2, row.len());
    let (chunks, _) = row.as_chunks::<2>();
    let mut acc: i64 = 0;
    for (&x, b) in a.iter().zip(chunks) {
        let w = bf16_to_fixed(u16::from_le_bytes(*b), F);
        assert!(
            w.unsigned_abs() < 1 << 31,
            "dot_i8_bf16q: value out of Q.{F} i32 range"
        );
        acc += x as i64 * w;
    }
    acc
}

// ---------------------------------------------------------------------------
// Model: checkpoint bytes -> integer tables, once at load.
// ---------------------------------------------------------------------------

struct CisLayer<'a> {
    ln1: Vec<i32>,
    ln2: Vec<i32>,
    attn_sub: Option<Vec<i32>>,
    ffn_sub: Option<Vec<i32>>,
    q_w: &'a [u8],
    k_w: &'a [u8],
    v_w: &'a [u8],
    o_w: &'a [u8],
    gate_w: &'a [u8],
    up_w: &'a [u8],
    down_w: &'a [u8],
    q_s: FRatio,
    k_s: FRatio,
    v_s: FRatio,
    o_s: FRatio,
    gate_s: FRatio,
    up_s: FRatio,
    down_s: FRatio,
}

pub struct CisModel<'a> {
    layers: Vec<CisLayer<'a>>,
    final_g: Vec<i32>,
    /// BF16 embedding table bytes (vocab x hidden). Rows converted to Q.F
    /// on the fly at each lookup/dot (identical rule to `cis_infer.rs:575-
    /// 581`: avoids materializing a ~vocab*hidden*4-byte table).
    emb: &'a [u8],
    /// LM-head BF16 bytes: the untied `lm_head.weight` when present,
    /// otherwise the tied embedding table.
    head: &'a [u8],
    pub config: ModelConfig,
}

fn read_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<f32, String> {
    let view = tensors.tensor(name)?;
    let data = view.data();
    if data.len() < 4 {
        return Err(format!("{} too small for f32", name));
    }
    let mut bytes = [0u8; 4];
    bytes.copy_from_slice(&data[0..4]);
    Ok(f32::from_le_bytes(bytes))
}

fn read_optional<'a>(tensors: &SafeTensors<'a>, name: &str) -> Result<Option<&'a [u8]>, String> {
    if tensors.has_tensor(name) {
        Ok(Some(tensors.tensor(name)?.data()))
    } else {
        Ok(None)
    }
}

impl<'a> CisModel<'a> {
    /// Build the integer model directly from a parsed MODEL.SAF plus the
    /// EMBED.BIN bytes and the already-parsed config. Reads exactly the
    /// tensor names `aegis-core/src/model.rs`'s `DecoderLayer`/
    /// `FullBitNetPipeline` read (design doc §3.3: `safetensors.rs` mirrors
    /// only the `SafeTensors` container itself; this function is the
    /// forward-pass-specific tensor lookup, kept here rather than as a
    /// separate pipeline type this crate has no other use for).
    pub fn new(
        tensors: &SafeTensors<'a>,
        embed_bytes: &'a [u8],
        config: &ModelConfig,
    ) -> Result<CisModel<'a>, String> {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        if !hidden.is_multiple_of(4) || !inter.is_multiple_of(4) {
            return Err(String::from("CIS packing requires dims divisible by 4"));
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let ln1 = gains_to_q(
                tensors
                    .tensor(&format!("{prefix}.input_layernorm.weight"))?
                    .data(),
                hidden,
                &format!("layer {i} input_layernorm"),
            )?;
            let ln2 = gains_to_q(
                tensors
                    .tensor(&format!("{prefix}.post_attention_layernorm.weight"))?
                    .data(),
                hidden,
                &format!("layer {i} post_attention_layernorm"),
            )?;
            let attn_sub = match read_optional(
                tensors,
                &format!("{prefix}.self_attn.attn_sub_norm.weight"),
            )? {
                Some(b) => Some(gains_to_q(b, hidden, &format!("layer {i} attn_sub_norm"))?),
                None => None,
            };
            let ffn_sub =
                match read_optional(tensors, &format!("{prefix}.mlp.ffn_sub_norm.weight"))? {
                    Some(b) => Some(gains_to_q(b, inter, &format!("layer {i} ffn_sub_norm"))?),
                    None => None,
                };
            layers.push(CisLayer {
                ln1,
                ln2,
                attn_sub,
                ffn_sub,
                q_w: tensors
                    .tensor(&format!("{prefix}.self_attn.q_proj.weight"))?
                    .data(),
                k_w: tensors
                    .tensor(&format!("{prefix}.self_attn.k_proj.weight"))?
                    .data(),
                v_w: tensors
                    .tensor(&format!("{prefix}.self_attn.v_proj.weight"))?
                    .data(),
                o_w: tensors
                    .tensor(&format!("{prefix}.self_attn.o_proj.weight"))?
                    .data(),
                gate_w: tensors
                    .tensor(&format!("{prefix}.mlp.gate_proj.weight"))?
                    .data(),
                up_w: tensors
                    .tensor(&format!("{prefix}.mlp.up_proj.weight"))?
                    .data(),
                down_w: tensors
                    .tensor(&format!("{prefix}.mlp.down_proj.weight"))?
                    .data(),
                q_s: f32_to_ratio(
                    read_f32(tensors, &format!("{prefix}.self_attn.q_proj.weight_scale"))?
                        .to_bits(),
                ),
                k_s: f32_to_ratio(
                    read_f32(tensors, &format!("{prefix}.self_attn.k_proj.weight_scale"))?
                        .to_bits(),
                ),
                v_s: f32_to_ratio(
                    read_f32(tensors, &format!("{prefix}.self_attn.v_proj.weight_scale"))?
                        .to_bits(),
                ),
                o_s: f32_to_ratio(
                    read_f32(tensors, &format!("{prefix}.self_attn.o_proj.weight_scale"))?
                        .to_bits(),
                ),
                gate_s: f32_to_ratio(
                    read_f32(tensors, &format!("{prefix}.mlp.gate_proj.weight_scale"))?.to_bits(),
                ),
                up_s: f32_to_ratio(
                    read_f32(tensors, &format!("{prefix}.mlp.up_proj.weight_scale"))?.to_bits(),
                ),
                down_s: f32_to_ratio(
                    read_f32(tensors, &format!("{prefix}.mlp.down_proj.weight_scale"))?.to_bits(),
                ),
            });
        }

        let final_g = gains_to_q(
            tensors.tensor("model.norm.weight")?.data(),
            hidden,
            "final norm",
        )?;
        check_bf16_table(embed_bytes, vocab * hidden, "EMBED.BIN")?;
        let head = if config.tie_word_embeddings {
            embed_bytes
        } else {
            let t = tensors.tensor("lm_head.weight")?;
            check_bf16_table(t.data(), vocab * hidden, "lm_head")?;
            t.data()
        };

        Ok(CisModel {
            layers,
            final_g,
            emb: embed_bytes,
            head,
            config: config.clone(),
        })
    }

    fn emb_row_to_q(&self, tok: u32, out: &mut [i64]) {
        let hidden = out.len();
        let start = tok as usize * hidden * 2;
        if start + hidden * 2 <= self.emb.len() {
            bf16_row_to_q(&self.emb[start..start + hidden * 2], out);
        } else {
            out.fill(0);
        }
    }

    fn head_row(&self, j: usize, hidden: usize) -> &[u8] {
        &self.head[j * hidden * 2..(j + 1) * hidden * 2]
    }
}

// ---------------------------------------------------------------------------
// Engine: FullInt only. Owns its own integer KV cache; no float state
// anywhere (design doc §2.2: Hybrid is entirely out of scope).
// ---------------------------------------------------------------------------

pub struct CisEngine<'m, 'a> {
    model: &'m CisModel<'a>,
    h: Vec<i64>,
    codes: Vec<i8>,
    acc_a: Vec<i32>,
    acc_b: Vec<i32>,
    fixed: Vec<i64>,
    logits: Vec<i64>,
    qi: Vec<i32>,
    ki: Vec<i32>,
    vi: Vec<i32>,
    k_icache: Vec<i32>,
    v_icache: Vec<i32>,
    iscores: Vec<i64>,
    iprobs: Vec<i32>,
    exp_lut: ExpLut,
    rope_i: RopeTableI,
    isq_q30: i64,
}

impl<'m, 'a> CisEngine<'m, 'a> {
    pub fn new(model: &'m CisModel<'a>) -> Self {
        let c = &model.config;
        let hidden = c.hidden_size;
        let inter = c.intermediate_size;
        let head_dim = hidden / c.num_attention_heads;
        let kv_dim = c.num_key_value_heads * head_dim;
        let max_pos = c.max_position_embeddings;
        let widest = hidden.max(inter);
        CisEngine {
            h: vec![0i64; hidden],
            codes: vec![0i8; widest],
            acc_a: vec![0i32; widest],
            acc_b: vec![0i32; widest],
            fixed: vec![0i64; widest],
            logits: vec![0i64; c.vocab_size],
            qi: vec![0i32; hidden],
            ki: vec![0i32; kv_dim],
            vi: vec![0i32; kv_dim],
            k_icache: vec![0i32; model.layers.len() * max_pos * kv_dim],
            v_icache: vec![0i32; model.layers.len() * max_pos * kv_dim],
            iscores: vec![0i64; max_pos],
            iprobs: vec![0i32; max_pos],
            exp_lut: ExpLut::new(),
            rope_i: RopeTableI::new(max_pos, head_dim, c.rope_theta.to_bits()),
            isq_q30: inv_sqrt_q30(head_dim as u64),
            model,
        }
    }

    /// Materialize the integer logits for the current residual state (final
    /// norm + integer LM head), vocab-sized, exact i64. Pair with
    /// `crate::ops::argmax_i64`. Identical arithmetic to
    /// `cis_infer.rs:1227-1244` minus the (Hybrid/NLL-only) f64 return.
    pub fn decode_logits(&mut self) -> &[i64] {
        let c = &self.model.config;
        let hidden = c.hidden_size;
        let vocab = c.vocab_size;
        let _s = normq(
            &mut self.codes[..hidden],
            &self.h[..hidden],
            &self.model.final_g,
        );
        for (j, l) in self.logits[..vocab].iter_mut().enumerate() {
            *l = dot_i8_bf16q(&self.codes[..hidden], self.model.head_row(j, hidden));
        }
        &self.logits[..vocab]
    }

    /// One decode step: token embedding through every layer, integer
    /// residual stream in `self.h`. Identical arithmetic to the `FullInt`
    /// arm of `cis_infer.rs:881-1220` (the `Hybrid` arm does not exist in
    /// this crate at all).
    pub fn forward_step_int(&mut self, current_tok: u32, seq_pos: usize) {
        let c = &self.model.config;
        let hidden = c.hidden_size;
        let inter = c.intermediate_size;
        let num_heads = c.num_attention_heads;
        let num_kv_heads = c.num_key_value_heads;
        let head_dim = hidden / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let max_pos = c.max_position_embeddings;

        // Embedding lookup: BF16 row converted to Q.F on the fly (exact).
        self.model.emb_row_to_q(current_tok, &mut self.h[..hidden]);

        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            // --- attention block ----------------------------------------
            let s_in = normq(&mut self.codes[..hidden], &self.h[..hidden], &layer.ln1);

            ternary_matvec_i8(
                &mut self.acc_a[..hidden],
                &self.codes[..hidden],
                layer.q_w,
                hidden,
                hidden,
            );
            let (neg, qs) = fixed_qscale(&layer.q_s, &s_in, QK_F);
            for (o, &a) in self.qi[..hidden].iter_mut().zip(&self.acc_a[..hidden]) {
                let v = qs.rescale(a as i64);
                let v = if neg { -v } else { v };
                assert!(v.unsigned_abs() < 1 << 29, "FullInt: q exceeds Q.16 range");
                *o = v as i32;
            }
            ternary_matvec_i8(
                &mut self.acc_a[..kv_dim],
                &self.codes[..hidden],
                layer.k_w,
                kv_dim,
                hidden,
            );
            let (neg, qs) = fixed_qscale(&layer.k_s, &s_in, QK_F);
            for (o, &a) in self.ki[..kv_dim].iter_mut().zip(&self.acc_a[..kv_dim]) {
                let v = qs.rescale(a as i64);
                let v = if neg { -v } else { v };
                assert!(v.unsigned_abs() < 1 << 29, "FullInt: k exceeds Q.16 range");
                *o = v as i32;
            }
            ternary_matvec_i8(
                &mut self.acc_a[..kv_dim],
                &self.codes[..hidden],
                layer.v_w,
                kv_dim,
                hidden,
            );
            let (neg, qs) = fixed_qscale(&layer.v_s, &s_in, QK_F);
            for (o, &a) in self.vi[..kv_dim].iter_mut().zip(&self.acc_a[..kv_dim]) {
                let v = qs.rescale(a as i64);
                let v = if neg { -v } else { v };
                assert!(v.unsigned_abs() < 1 << 29, "FullInt: v exceeds Q.16 range");
                *o = v as i32;
            }

            rope_apply_i(
                &mut self.qi[..hidden],
                &mut self.ki[..kv_dim],
                seq_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                &self.rope_i,
            );
            let slot = (layer_idx * max_pos + seq_pos) * kv_dim;
            self.k_icache[slot..slot + kv_dim].copy_from_slice(&self.ki[..kv_dim]);
            self.v_icache[slot..slot + kv_dim].copy_from_slice(&self.vi[..kv_dim]);

            for h_idx in 0..num_heads {
                let kv_h = h_idx / (num_heads / num_kv_heads);
                let qh = &self.qi[h_idx * head_dim..(h_idx + 1) * head_dim];
                for t in 0..=seq_pos {
                    let kb = (layer_idx * max_pos + t) * kv_dim + kv_h * head_dim;
                    let mut acc: i128 = 0;
                    for (qv, kv) in qh.iter().zip(&self.k_icache[kb..kb + head_dim]) {
                        acc += *qv as i128 * *kv as i128;
                    }
                    let sc = rne_div(acc * self.isq_q30 as i128, 1 << (2 * QK_F + 30 - SCORE_F));
                    assert!(
                        sc >= i64::MIN as i128 && sc <= i64::MAX as i128,
                        "FullInt: score exceeds i64"
                    );
                    self.iscores[t] = sc as i64;
                }
                softmax_i(
                    &mut self.iscores[..=seq_pos],
                    &mut self.iprobs[..=seq_pos],
                    &self.exp_lut,
                );
                for d in 0..head_dim {
                    let mut mix: i64 = 0;
                    for t in 0..=seq_pos {
                        let vb = (layer_idx * max_pos + t) * kv_dim + kv_h * head_dim;
                        mix += self.iprobs[t] as i64 * self.v_icache[vb + d] as i64;
                    }
                    self.fixed[h_idx * head_dim + d] =
                        rne_div(mix as i128, 1 << (QK_F + PROB_F - F)) as i64;
                }
            }
            let g_attn = F; // FullInt lands exactly on the Q.F grid.

            let s_o = match &layer.attn_sub {
                Some(g) => normq(&mut self.codes[..hidden], &self.fixed[..hidden], g),
                None => quantq(&mut self.codes[..hidden], &self.fixed[..hidden], g_attn),
            };
            ternary_matvec_i8(
                &mut self.acc_a[..hidden],
                &self.codes[..hidden],
                layer.o_w,
                hidden,
                hidden,
            );
            let (neg, qs) = residual_qscale(&layer.o_s, &s_o);
            for (hi, &a) in self.h[..hidden].iter_mut().zip(&self.acc_a[..hidden]) {
                let d = qs.rescale(a as i64);
                *hi += if neg { -d } else { d };
            }

            // --- MLP block -----------------------------------------------
            let s_mlp = normq(&mut self.codes[..hidden], &self.h[..hidden], &layer.ln2);
            ternary_matvec_i8(
                &mut self.acc_a[..inter],
                &self.codes[..hidden],
                layer.up_w,
                inter,
                hidden,
            );
            ternary_matvec_i8(
                &mut self.acc_b[..inter],
                &self.codes[..hidden],
                layer.gate_w,
                inter,
                hidden,
            );

            let (ng, qg) = fixed_qscale(&layer.gate_s, &s_mlp, F);
            let (nu, qu) = fixed_qscale(&layer.up_s, &s_mlp, F);
            for i in 0..inter {
                let g = qg.rescale(self.acc_b[i] as i64);
                let g = if ng { -g } else { g };
                let u = qu.rescale(self.acc_a[i] as i64);
                let u = if nu { -u } else { u };
                assert!(
                    g.unsigned_abs() < 1 << 40 && u.unsigned_abs() < 1 << 40,
                    "FullInt: MLP value exceeds Q.20 headroom"
                );
                self.fixed[i] = match c.hidden_act {
                    Activation::Relu2 => relu2_q20(g, u),
                    Activation::Silu => silu_q20(g, u, &self.exp_lut),
                };
            }
            // FullInt escape valve (spec §5.10 gap): re-fix onto a
            // per-vector block exponent so BitNet-2B-scale ACT-I products
            // stay inside the normq/quantq 2^50 residual headroom.
            // M7-scale products never trip it (degenerates to G = F).
            let g_mlp = fix_q_vec(&mut self.fixed[..inter]);

            let s_down = match &layer.ffn_sub {
                Some(g) => normq(&mut self.codes[..inter], &self.fixed[..inter], g),
                None => quantq(&mut self.codes[..inter], &self.fixed[..inter], g_mlp),
            };
            ternary_matvec_i8(
                &mut self.acc_a[..hidden],
                &self.codes[..inter],
                layer.down_w,
                hidden,
                inter,
            );
            let (neg, qs) = residual_qscale(&layer.down_s, &s_down);
            for (hi, &a) in self.h[..hidden].iter_mut().zip(&self.acc_a[..hidden]) {
                let d = qs.rescale(a as i64);
                *hi += if neg { -d } else { d };
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Decode orchestration: greedy argmax, EOS ignored — the shape both
// `cis_decode.rs` and `cis_witness.rs`'s `replay()` share.
// ---------------------------------------------------------------------------

/// Report of one greedy decode run: prompt token ids, generated token ids
/// (exactly `max_new` of them, EOS ignored by design), and the FNV-1a 64
/// `cis-digest` fold (prompt ids then generated ids, each LE32 — identical
/// fold to `crate::receipt::cis_digest_of` / `cis_witness.rs`'s `replay()`).
pub struct DecodeReport {
    pub prompt_ids: Vec<u32>,
    pub generated_ids: Vec<u32>,
    pub fnv_digest: u64,
}

/// Run the deterministic FullInt greedy decode: tokenize `prompt`, prefill,
/// then generate exactly `max_new` tokens by argmax, EOS ignored. When
/// `chain` is `Some`, every generated step's full i64 logit vector is
/// absorbed into it via [`WitnessChain::fold_step`] BEFORE the next
/// `forward_step_int` call — the exact order `cis_witness.rs:88-98`
/// requires (design doc §1.2). Identical step order/digest fold to
/// `aegis-linux/examples/cis_decode.rs` and `cis_witness.rs`'s `replay()`.
pub fn run_decode(
    model: &CisModel<'_>,
    tokenizer: &Tokenizer<'_>,
    prompt: &str,
    max_new: usize,
    mut chain: Option<&mut WitnessChain>,
) -> DecodeReport {
    use crate::fnv::{FNV1A64_OFFSET, fnv1a64};

    let mut engine = CisEngine::new(model);
    let prompt_ids = tokenizer.encode(prompt);

    let mut digest = FNV1A64_OFFSET;
    let mut pos = 0usize;
    for &t in &prompt_ids {
        digest = fnv1a64(digest, &t.to_le_bytes());
        engine.forward_step_int(t, pos);
        pos += 1;
    }

    let mut generated = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let tok = {
            let logits = engine.decode_logits();
            let t = argmax_i64(logits);
            if let Some(c) = chain.as_deref_mut() {
                c.fold_step(t, logits);
            }
            t
        };
        digest = fnv1a64(digest, &tok.to_le_bytes());
        generated.push(tok);
        engine.forward_step_int(tok, pos);
        pos += 1;
    }

    DecodeReport {
        prompt_ids,
        generated_ids: generated,
        fnv_digest: digest,
    }
}
