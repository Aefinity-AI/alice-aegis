# CA1 — PTQ collapse-point at production scale (Falcon-E-1B)

Host: penguin (i5-10210U). Rule A: perplexity only, no wall-clock/tok-s figures below (the raw log has incidental wall-clock; it is not a reported result).

## Scope reduction (found before any heavy compute ran)

The plan's target artifact, `falcon_e_1b_model.safetensors` (530 MB), is **not** a
dense checkpoint. It is already the engine-forged artifact: all 168
attention/MLP linears across 24 layers are 2-bit packed U8 codes
`{0,1,2} = {0,+1,-1}` with a per-tensor F32 absmax scale (QAT-trained
native-ternary, matching the E13 CIS-1 result). Row-wise absmax PTQ at any
bit width >= 2 bits (int8/int4/int3/int2/ternary all have >= 4 representable
levels) reconstructs a 3-level `{-scale,0,+scale}` source **exactly** — this
is algebraically provable (the quantization scale cancels) and was verified
numerically in `scripts/ptq_ladder.py`'s sanity check before any model was
scored. **There is no PTQ ladder to observe on the transformer body**: it is
already sitting at the floor of the ladder by construction.

The one dense (non-ternary) linear layer at production scale in this
artifact is `lm_head.weight` (BF16, `[32768, 2048]`, 67,108,864 params,
untied). This report applies the same row-wise absmax weight-only PTQ
recipe as the 30M dense-body ladder
(`2026-08-29-ALICE-RUST-AND-BARRIERS.md`) to **lm_head only**, fake-quantized
back to BF16 in place, all other tensors byte-identical to the source file.
This is a scope reduction from "every linear layer" to "the one dense linear
layer this native-ternary architecture actually has" — recorded honestly,
not hidden.

## Method

`scripts/ptq_ladder.py` reads `lm_head.weight`, applies symmetric row-wise
absmax PTQ (N signed levels, round-to-nearest, clip, dequantize back to
BF16) at int8_w / int4_w / int3_w / int2_w / ternary_w (int2_w and
ternary_w use the identical 2-bit formula — ternary IS 2-bit representation
of `{-1,0,+1}`), writing 5 copies of the model with only that one tensor's
bytes patched. Scored with `aegis-eval <M> <E> <V> test.txt 200 --sample`
(float path, same 199-token contiguous window methodology as A35/A41),
same `test.txt`/`falcon_e_1b_embed.bin`/`falcon_e_1b_vocab.bin` triple for
every rung.

## Results

| numerics | lm_head PPL (199 tok) | delta vs float |
|---|---|---|
| float (baseline) | 11.736 | — |
| int8_w | 11.734 | -0.02% |
| int4_w | 11.818 | +0.70% |
| int3_w | 13.251 | +12.9% |
| **int2_w** | **611.649** | **+5111%** |
| **ternary_w** | **611.649** | **+5111%** |

int2_w and ternary_w are bit-identical (same 2-bit formula, confirmed:
33,325/67,108,864 elements unchanged by quantization in both cases,
identical PPL to 3 decimals) — an internal consistency check, not a
re-run-the-outlier case (both numbers are reproductions of the same
computation, not independent measurements of an outlier).

**Collapse point: between int3_w and int2_w**, matching the 30M-model
table's collapse location (also between int3_w and int2_w: 21.40 -> 8364).
Monotone except int8/int4/int3 are a smooth degrading ramp and int2/ternary
jump together — consistent with the 30M table's shape (smooth ramp through
int3, cliff at int2/ternary).

## PASS/FAIL vs the plan's criteria (2.4, CA1)

Plan's criterion (a): "collapse point matches 30M (ternary_w ppl >10x
float, int3 clearly worse than int4) -> scale-invariance confirmed."

- ternary_w ppl >10x float: 611.649 / 11.736 = 52.1x float. **YES.**
- int3 clearly worse than int4: 13.251 vs 11.818 (+12.2 percentage points
  relative to each other). **YES.**

**Qualified PASS** on the ONE dense linear layer available at production
scale in this artifact (lm_head, 67M params) — the collapse-point location
(between int3 and int2, cliff magnitude >10x at ternary) reproduces at a
much larger dense-layer size than the 30M reference. This is **not** the
"every linear layer of a 1B model" experiment the plan specified, because
that experiment is not executable against this artifact (see Scope
reduction above) — it would require a genuinely dense (pre-QAT) Falcon-E-1B
checkpoint, which is not on this box.

## What this does and does not prove

- Does NOT show PTQ collapse generalizes to the transformer body at 1B
  scale (untestable with the available artifact — the body is already
  QAT-ternary, not dense).
- DOES show, on a 67M-parameter dense matrix (2.2x the entire 30M
  reference model's parameter count), that the int3-to-int2 PTQ cliff
  reproduces in both location and severity (>10x) at a meaningfully larger
  scale than previously measured. This is a genuine, if narrower, positive
  data point for scale-invariance of the collapse point, not a toy-scale
  artifact.
- Any future "PTQ tolerance improves/degrades with scale" wording should
  cite this as "dense-layer, not whole-model" evidence, and should note
  the whole-model experiment remains unrun.

Raw log: `docs/ca1-ladder.log` (not committed — regenerate via
`scripts/ca1_driver.sh`; wall-clock lines present are incidental per
Rule A and are not reported here).
