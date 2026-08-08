# Verified Whole-Number Inference on Bare Metal

**Integer-exact language models with cross-implementation determinism, measured
on commodity hardware — Aefinity AI**

> **STATUS: PUBLIC PREPRINT DRAFT (snapshot 2026-08-02).** All gates below are satisfied for this snapshot; two pre-existing RANGER figure debts remain flagged by `verify-figures.sh` and are excluded from the claims. Publication gates, in order: (1) HP
> Stream cross-ISA boot ✅ DONE 2026-08-02 (`hp_L_BOOTLOG_2026-08-02.txt`); (2) `scripts/verify-figures.sh`
> clean on every number below; (3) independent fact-check pass against the raw
> logs; (4) repository secrets/credentials audit; (5) explicit push approval.
> Every figure in this document carries its evidence path. A figure without a
> log does not ship — that rule cost this program a retraction once and will
> not be violated again.

## Abstract

We present a ternary-quantized transformer inference engine with three
properties we believe are jointly unique in the edge-AI space, each backed by
raw instrument logs, preregistered predictions, and adversarial re-verification:

1. **Whole-number inference costs almost nothing.** A fully integer forward
   pass — attention, softmax, rotary embeddings, and activations included; no
   floating-point value exists anywhere — scores within **+0.064%** perplexity
   of the float engine on our 14M-parameter model, and the integer/float gap is
   **+0.74%** at 2B-class production scale (pruned-vocab BitNet b1.58-2B-4T,
   hybrid attention). These pass a pre-declared 5% kill line with **78×** and
   **6.7×** margins respectively.
2. **What the model says is provable.** Because the integer path is exact and
   associative, its outputs are bit-identical across compilers, code
   generation, machines, and boots. We observed byte-identical model responses
   across three different binaries (including one with zero SIMD instructions)
   and three bare-metal boots, and an identical integer self-test digest
   (`76985613c965f643`) on every implementation tested so far: two Dell
   bare-iron cold boots, the native dev host, and a fully emulated CPU —
   each leg instrument-logged — including the fourth implementation, an
   SSE2-class Celeron (HP Stream N4020, bare iron, 2026-08-02).
3. **The operating system is optional — and removing it is now a measured
   win.** In a preregistered, paired comparison (n=10 per prompt per arm,
   decision rule committed before boot), our ring-0 UEFI unikernel decodes
   faster than a minimal Linux running the same engine on the same machine on
   3/3 prompts (+3.7%, +10.4%, +5.4% median decode throughput), after the
   unikernel takes over the two OS jobs that mattered: idle-core parking and
   keeping console I/O out of the timed path.

We additionally report a kernel-level result: exploiting ReLU² activation
sparsity (78.9% exact zeros at the `down_proj` input of BitNet-2B) with a
column-skip kernel yields **2.88–2.89×** on the real activation distribution on
AVX2 bare metal — via a variant that is *byte-identical* to the incumbent kernel,
so adoption carries zero quality risk.

## 1. The claims, with their evidence chains

### C1 — Integer quality (the E2 gates)

| Model | Path | PPL (float) | PPL (integer) | Delta | Kill line | Evidence |
|---|---|---|---|---|---|---|
| M7 (14.17M) | hybrid integer | 5.639491 | 5.657126 | **+0.31%** | 5% | ledger A19; `docs/hardware_logs/cis1_e2_int_vs_float_ppl_m7_i5-10210U_crosvm_2026-08-01.log` |
| M7 (14.17M) | **full integer** | 5.639491 | 5.643085 | **+0.064%** | 5% | ledger A20; `docs/hardware_logs/cis1_fullint_attention_ppl_m7_i5-10210U_crosvm_2026-08-01.log` |
| BitNet b1.58-2B-4T (pruned vocab) | hybrid integer | 30.706665 | 30.934140 | **+0.74%** | 5% | ledger A21; `docs/hardware_logs/cis1_e2_bitnet2b_int_vs_float_ppl_i5-10210U_crosvm_2026-08-01.log` |

Notable: the full-integer path is *closer* to float than the hybrid — removing
the two float re-entry quantizations outweighs the fixed-point table
quantization it adds. All runs bit-identical across repeats (digests in logs);
teacher-forced, same run, same binary, same token window; float base computed
in-run, never quoted from history. Caveats declared in each log: pruned-vocab
PPLs are not cross-tokenizer comparable; the BitNet window is 200 tokens on an
`<unk>`-dense region.

The integer semantics (CIS-1) are specified with normative constants — a
Q0.31 exponential LUT, Q1.30 rotary tables generated at load by an
integer-only procedure, RNE as the single rounding mode (arbitrated by
measurement, +0.31% whole-path) — every constant generated and cross-checked
by an independent big-integer implementation
(`scripts/cis_e2_golden_gen.py`), never by the code under test.

### C2 — Determinism and verifiability

- **Across code generation:** a soft-float build (zero vector instructions;
  census `docs/hardware_logs/mecha1_softfloat_census_2026-08-01_v2.log`) and a
  hardfloat AVX2+FMA build produced **byte-identical responses** on all test
  prompts on Dell bare metal (ledger A18/P5).
- **Across boots and binaries:** MECH v1.1 and MECH v2 boots (different
  binaries) produced byte-identical responses; within MECH v2, 27/27
  repeat-run comparisons were exact
  (`docs/hardware_logs/mech2_U_BOOTLOG_2026-08-01.txt`).
- **Across machines (integer):** the CIS integer self-test digest
  `76985613c965f643` is identical on Dell bare iron (two cold boots:
  `mech2colskip_L_dell_BOOTLOG_2026-08-01.txt`, `_dell2_`), the crosvm dev
  host (`cis_selftest_crosvm_devhost_2026-08-01.log`), and a TCG-emulated CPU
  (`cis_selftest_witness_tcg_qemu_2026-08-01.log`), and is pinned as a
  regression constant (`aegis-linux/tests/cis_selftest_digest.rs`), and on HP
  Stream Celeron N4020 bare iron (SSE2-class, `hp_L_BOOTLOG_2026-08-02.txt`).
  The four-implementation jury is complete: same digest on every leg (A25).
- **By construction:** the full-integer forward pass contains no SIMD dispatch
  and enters no float kernel; cross-kernel-path identity is structural, not
  empirical (ledger A20). A hash-chained generation transcript (witness v0)
  reproduced its chain hash across two Dell boots and an emulated boot, all
  logged (the two Dell logs above + `cis_selftest_witness_tcg_qemu_2026-08-01.log`).

### C3 — The OS-cost result (preregistered)

Prehistory, reported in full because negative results are deliverables: our
first OS-comparison (Band 3, 2026-07-31) found minimal Linux *faster* than the
unikernel by a pooled −39.3%. Mechanism decomposition (MECH v1, ledger A13)
found a turbo-bin effect (+7.4%, idle cores parked by firmware cost the 1-core
bin) and a console cost, and exposed that the MECH measurement binary was
accidentally soft-float (ledger A14; the 119–250× anomaly this explained is
itself instrument-logged). With the build fixed, AP-PARK (MWAIT-C6 idle-core
parking) adopted, and console output buffered during timing, the preregistered
paired redo (MECH v2, prereg
`docs/hardware_logs/mech2_PREREGISTRATION_2026-08-01.md`, committed before
boot) found:

| Prompt | Unikernel median (n=10) | Minimal Linux median (n=10) | Unikernel advantage |
|---|---|---|---|
| "hello alice" | 386.4 tok/s | 372.6 tok/s | **+3.7%** (+3.6%) |
| "how are you today?" | 342.4 tok/s | 310.1 tok/s | **+10.4%** (+9.4%) |
| "continue" | 352.7 tok/s | 334.7 tok/s | **+5.4%** (+5.1%) |

Advantage stated in throughput form; parenthesized values are the
preregistration's declared form, Δ=(t_L−t_U)/t_L in seconds/token. The 3/3
verdict holds under either metric; both forms are in ledger A22.

Decision rule (3/3 median wins + clean bit-exactness): **met**. Declared,
unresolved caveats carried with the claim: the arms time different code paths
(intent loop vs harness), token counts differ by ±1 at EOS, and the unikernel
metric amortizes prefill (conservative against the unikernel). One
preregistered prediction failed and is reported as such: run-to-run timing
spread was 4.7–9.6%, not the predicted <3% (P-V2-2, ledger A22). Evidence:
`mech2_U_BOOTLOG_2026-08-01.txt`, `mech2colskip_L_dell_BOOTLOG_2026-08-01.txt`
(+ `_dell2_` replicate). Machine: Dell Inspiron 15, i5-5200U Broadwell-U.
TSC constant 2.1975 GHz from the A12 calibration (2026-07-29); the same-day
L-boot in-log calibration read 2.1949 GHz (−0.12% — worst-case shifts +3.7%
to +3.55%, verdict unaffected); effective clock ratio printed in-log per
measurement.

### C4 — Activation-sparsity column-skip (kernel-level)

ReLU² models produce exact zeros: 78.9% of BitNet-2B's `down_proj` input at
the point of kernel consumption (ledger A15; re-derived bit-for-bit by the archived adversarial review —
see `docs/ADVERSARIAL_REVIEWS_2026-08-01.md`). A column-major skip kernel measured on real captured activation vectors on
the Dell (94 of the 750 captured vectors loaded by the bench; subset mean
z=0.7927, slightly sparser than the 750-vector pooled 0.7886): **2.88–2.89×
vs the incumbent** (ordered variant,
byte-identical to the incumbent by construction and by test), replicated
across two cold boots, 3 interleaved captures each; z=0 overhead honestly
measured at 0.69× (skip machinery costs ~45% when nothing is skippable).
Synthetic z=0.9: 5.55–5.94× across the six captures. Not yet wired into end-to-end inference — kernel-level
claim only. Evidence: `mech2colskip_L_dell_BOOTLOG_2026-08-01.txt` (+ dell2),
capture provenance `relu2_act_capture_bitnet2b_2026-08-01.log`, exactness
proofs in `aegis-core/tests/colskip_exactness.rs` (including the
fma(±0,w,s)=s identity with its −0.0 boundary counterexample).

## 2. Methodology — why these numbers can be trusted

- **Rule A:** no performance number from emulation, ever; every measurement
  names its physical machine and carries the effective/nominal clock ratio
  when tick-derived.
- **Rule B:** no number enters the research ledger without a raw instrument
  log; `scripts/verify-figures.sh` enforces the mapping mechanically.
- **Rule C:** logs and golden files are append-only; mislabeled or superseded
  files are marked and kept, never rewritten.
- **Rule D:** bit-exactness before benchmarks — byte-identity assertions
  caught every real bug this program has had; benchmarks caught none.
- **Preregistration:** MECH v2's decision rules and predictions were
  committed to the repository before its boots (verified by commit
  timestamps). The earlier oscost preregistration was written pre-boot but
  entered git post-hoc with its boot logs — disclosed here; only MECH v2
  carries the full pre-commit provenance.
- **Adversarial review:** every result lane in this document was re-executed
  and attacked by an independent reviewer agent before merge; the verbatim
  verdicts are archived in `docs/ADVERSARIAL_REVIEWS_2026-08-01.md`.
- **Negative results ship:** the CTZ kernel (A6), LUT-mpGEMM (A7), fused QKV
  matvec (A16, 1.43–1.91× *slower*), and bitplane matvec (A17, 0.39–0.41×)
  are settled negatives with the same evidentiary standard as the wins.

## 3. Reproduction

Single-machine quality gates: `aegis-eval <artifacts> <heldout> 512 --cis`
and `--cis-full` print float/integer PPLs, deltas, and digests in one run.
Cross-ISA identity: `cis_selftest` prints a 64-bit digest that must match on
any conforming machine. Bare-metal protocol: stage per
`docs/RUNCARD_SUNDOWN_2026-08-01.md` (gated by `check-efi-simd.sh` and QEMU
correctness boots), boot hands-off, extract past the pinned log offsets.

## 4. Limitations and open work

- BitNet-2B full-integer attention pending (KV-cache footprint on 6 GB dev
  hardware); its hybrid gate is passed.
- The four-implementation jury is complete (ledger A25); next is a full E4
  transcript replay with witness v1 hash chains (spec'd, not built).
- CIS-1 spec v0.3 constants need ratification before a v1.0 freeze.
- The colskip kernel needs end-to-end wiring and a system-level A/B before
  any decode-throughput claim.
- The OS-cost claim is scoped to this workload/machine class and carries its
  declared code-path caveats; it is not a general "kernels are useless" claim.

## 5. Why this matters

Edge AI's trust problem is not model quality — it is that nobody can prove
what a model did after the fact. A whole-number model whose every token is
reproducible bit-for-bit by a $150 laptop, or by a bootable USB verifier with
no operating system underneath it, turns "trust me" into "check me." That is
the product: not the fastest inference (though ring 0 now measures fastest on
our iron), but the only inference in its class that comes with receipts.

---
*Draft prepared 2026-08-01. All ledger references: `program/RESEARCH_LEDGER.md`
rows A13–A24 (A22 MECH v2, A23 colskip, A24 G1). Contact: Aefinity AI.*
