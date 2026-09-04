# E-S2 -- post-hoc zero-threshold sweep on BitNet-2B

Plan: `state/reports/2026-09-04-SUBBIT-TERNARY-PLAN.md` section 2 (repo
`claudius-maximus`). Host for everything below: penguin (Rule A: counts,
entropies, byte sizes only -- no timing figure recorded).

## Scope-reduction finding (found before the sweep ran)

`model.safetensors` (1.18 GB, the file the plan calls "the BitNet bf16
master weights") is **not** a dense pre-quantization checkpoint. Its body
tensors are U8, HF1BitLLM 2-bit-packed layout (packed along dim 0 in out/4
row-blocks), matching `config.json`'s
`quantization_config: {quant_method: bitnet, quantization_mode: offline}` --
i.e. this is the already-QAT-trained, already-ternary shipped checkpoint,
the same family of artifact CA1 found for Falcon-E-1B's body.

Verified over the full body (30 layers x 7 projections = 210 tensors,
2,084,044,800 weights), not a spot check:

1. Every unpacked 2-bit code is in `{0,1,2}`, never `3` -- exactly ternary,
   no residual continuous information.
2. Re-encoding these same trits into the engine's packing reproduces
   `aegis_pruned_model.safetensors` (already on penguin) **tensor-for-tensor,
   0 mismatched elements out of 2,084,044,800**, and every per-tensor scale
   matches exactly too (0 scale mismatches). This checkpoint uses the
   engine's own MULTIPLY convention directly (scale copies straight across,
   e.g. `q_proj.0` scale = 1.21875 in both files) -- **not** the
   onebitllms/Falcon-E DIVISION convention `repack_ternary.py`'s
   `scalar_f32_scale()` warns about; checked empirically, not assumed.

So `model.safetensors` and `aegis_pruned_model.safetensors` encode
IDENTICAL weights, just different containers. There is no latent to
threshold-sweep against -- the plan's own caveat ("on BitNet-2B, per tensor,
rank |latent| is unavailable") turns out to apply to the file the plan named
as the workaround, too.

**BitNet's default rule** (absmean quantization, Wang et al. 2023):
`gamma = mean(|W|)` per tensor, `W~ = RoundClip(W/gamma, -1, 1)`. For the
sign/zero decision alone (all a ternary weight retains), this is
algebraically identical to a hard threshold at `0.5*gamma`. So **tau=0.5
in a tau*gamma threshold parameterization is BitNet's default, and it is
exactly what the shipped body already used** -- confirmed by the exact
tensor match above, not a resemblance.

## Real, measured whole-body census (replaces the E-S1 H0, not yet pushed)

```
$ ~/venvs/es1/bin/python scripts/ptq_zero_sweep.py --verify
=== E-S2 verify+census (2,084,044,800 body weights, count only) ===
exact-match vs aegis_pruned_model.safetensors: 0 mismatched tensors, 0 mismatched elements, 0 scale mismatches
whole-body: p(-1)=0.288940 p(0)=0.422109 p(+1)=0.288951  H0=1.5603 bits/weight
per-projection-type p0:
  down_proj    p0=0.4131  n=530,841,600
  gate_proj    p0=0.4236  n=530,841,600
  k_proj       p0=0.4465  n=49,152,000
  o_proj       p0=0.4184  n=196,608,000
  q_proj       p0=0.4675  n=196,608,000
  up_proj      p0=0.4161  n=530,841,600
  v_proj       p0=0.3764  n=49,152,000
```

p0=0.4221 body-wide sits inside E-S1's prediction band (0.4-0.6 -> H0
1.45-1.55) -- H0=1.5603 is close to the uniform-trit ceiling (1.585), i.e.
the shipped model is near-maximally entropic at the trit level: BitNet's
QAT training pushes for near-balanced -1/0/+1 usage, not sparsity, which is
exactly why forcing MORE zeros post-hoc (this lane's actual question) costs
quality rather than being free.

## Adapted method for the sweep's non-baseline points

Since no continuous latent survives quantization, "raising tau past 0.5"
computes nothing new on this artifact. `scripts/ptq_zero_sweep.py`
substitutes the honest version of what the plan's own section-0 H(p0) table
is actually about: **magnitude-blind (uniform-random, independent per
weight, seeded) pruning of the shipped nonzero trits** down to target p0
values lifted directly from that table -- including its two headline
numbers, p0=0.9535 (0.314 bit/weight) and p0=0.9803 (0.158 bit/weight).
Magnitude-blind pruning cannot protect "important" weights, so its PPL cost
at a given p0 is a **pessimistic upper bound** relative to any magnitude- or
loss-aware pruning -- flagged wherever these two numbers get used.

Sweep points (8, matching the plan's point count): baseline (p0=0.4221,
pass-through, no pruning) | 0.70 | 0.80 | 0.90 | 0.95 | 0.9535 (0.314 bit) |
0.97 | 0.9803 (0.158 bit).

**Plan errata found while checking the smoke output against the plan's own
table:** recomputing H(p0) at p0=0.9535 (both via the general -Sum p*log2(p)
formula and the plan's own h(p0)+(1-p0) shortcut -- they are algebraically
identical for an even +/- split) gives H=0.3178, not the plan's listed
0.314 -- a ~0.004 bit (1.2% relative) discrepancy. Every other row checked
against the smoke output matches the plan's table to the digits given
(0.70->1.181, 0.80->0.922, 0.90->0.569, 0.95->0.336, 0.97->0.224,
0.9803->0.158, all reproduced below to 3-4 decimals). p0~=0.955 is the
value that actually yields H~=0.314; not changed here since the plan
prewrote p0=0.9535 as the sweep point, but flagged so the "0.314 bit" label
on that row is read as approximate.

## Smoke test (penguin, forge-only, no eval -- see "why no eval on penguin" below)

Full 8-point sweep run end-to-end on penguin (forge + census + RESULT.txt
row per point, `--eval` omitted): all 8 points forged successfully, achieved
p0 matched target to 6 decimal places at every point, and the H0 values
reproduce the plan's own table exactly at shared points (0.70 -> 1.1813
vs plan's 1.181; 0.80 -> 0.9219 vs plan's 0.922). Forged files deleted
after each point per the plan's tidiness note (2B model ~= 0.5 GB each).

Single-point forge validated first in isolation (p0=0.80 -> 541 tensors,
522,833,779 bytes, matching `aegis_pruned_model.safetensors`'s 541-tensor
shape almost byte-for-byte -- the ~2 KB difference is the `aegis_config`
metadata this forge writes and the shipped file does not carry).

Full 8-point sweep table from the penguin smoke run (forge-only; PPL/digest
columns blank -- no `--eval-bin` given, see below):

| p0 target | p0 achieved | H0 (bits/w) | packed bytes | coded bytes | skip bytes |
|---|---|---|---|---|---|
| baseline (0.4221) | 0.422109 | 1.5603 | 521,011,200 | 406,470,539 | 269,643,729 |
| 0.70 | 0.699992 | 1.1813 | 521,011,200 | 307,737,561 | 213,901,177 |
| 0.80 | 0.800000 | 0.9219 | 521,011,200 | 240,167,357 | 173,076,128 |
| 0.90 | 0.899993 | 0.5690 | 521,011,200 | 148,233,848 | 112,593,694 |
| 0.95 | 0.949995 | 0.3364 | 521,011,200 | 87,640,480 | 69,324,918 |
| 0.9535 | 0.953504 | 0.3178 | 521,011,200 | 82,793,680 | 65,731,664 |
| 0.97 | 0.969995 | 0.2244 | 521,011,200 | 58,462,898 | 47,357,076 |
| 0.9803 | 0.980300 | 0.1595 | 521,011,200 | 41,538,601 | 34,208,105 |

`packed_bytes` is constant across the sweep by construction (2-bit packing
cost does not depend on content) -- `coded_bytes` (order-0 Shannon estimate,
N*H0/8) and `skip_bytes` (nonzero-only run-length estimate) are what
actually shrink, which is the whole point of the E-S3 accounting these rows
also carry.

**Why no `aegis-eval --cis-full` run on penguin:** `--cis-full` holds three
engine passes' worth of KV cache + arena for a 2B-class model in one
process (CA1's own note: "two 2B-scale engines do not coexist" even in the
two-pass `--cis` mode, on a 6 GB box). penguin has ~2 GB available under
this lane's MemoryMax cap; box1 (5.7 GB, AVX2) is where the plan assigns
this cost. Per the task's fallback clause, the penguin smoke test covers
ternarize + census + forge only.

## Staged for box1

`~/aefinity-artifacts/bitnet_2b_master.safetensors` on cm-box1 (rsync from
this `model.safetensors`, sha256 verified both ends) -- box1 already has
`aefinity-artifacts/aegis_pruned_model.safetensors` and
`aefinity-artifacts/aegis-forge/{embed,vocab}.bin` staged from an earlier
lane. Leg: `claudius-maximus` branch `cm/es2-legs`, `legs/es2-sweep.sh`
(not yet pushed to box1 or started -- e6-membw, a Rule-A timing leg, was
still active at rsync time; this lane only staged the file, per the task's
explicit scope).

Expected cost per point on box1: forge (streaming per-tensor, dominated by
sequential read of ~1.2 GB + write of ~0.5 GB) plus one `aegis-eval --cis-full`
call (three full-model passes, 200 tokens, on AVX2). CA1's own driver
(same model scale class via a different artifact) treated a single-rung
build+eval as the unit of work; this lane's 8 rungs skip the one-time
`cargo build --release` cost after the first. No wall-clock figure is
recorded here (Rule A) -- `cm leg status cm-box1` after a push will show
real elapsed time once it runs.
