# 6. Cost and quality

Every figure in this section is a physical-hardware measurement or a same-run,
same-binary comparison logged under `docs/hardware_logs/`; none derives from
QEMU/TCG (Rule A), and each number below carries its ledger row and machine
name so it can be checked against the raw log rather than this prose (Rule B).

## Quality

The all-integer forward pass costs **+0.0637% perplexity** against float on
the M7 reference model (5.643085 vs. 5.639491, pinned 471-token teacher-forced
heldout, i5-10210U crosvm, digest `0xBED4A17A1A5EE296`, A20) — comfortably
inside the preregistered +5% kill line (78× headroom). The earlier
integer-dominant *hybrid* configuration on the same model, in which attention
and activations were still float, cost more: **+0.3127%** (5.657126 vs.
5.639491, digest `0x42E820C2A8A59CD6`, A19). The all-integer path is *closer*
to float than the hybrid, not farther: dropping the two f32→fixed re-entry
quantizations that the hybrid path requires outweighs the Q0.15/Q1.30 table
quantization that the full-integer path adds.

On the production-scale model, BitNet-2B, the measured figure is
**+0.7408% perplexity** (integer 30.934140 vs. float 30.706665,
i5-10210U crosvm, digest `0x24C4E510A86659D6`, A21). **This is the HYBRID
path — attention still runs in f32.** Do not read +0.7408% as an all-integer
number. The figure also carries caveats that make it non-comparable outside
its own run: it uses a vocabulary pruned from 128,256 to 50,256 tokens
(ASCII-oriented; never compared cross-tokenizer), and a 200-token evaluation
window in an `<unk>`-dense region, which is not comparable either to
published full-vocabulary perplexities or to this project's own
longer-window anchor (1,898 tokens, 10.758). The only claim it supports is
the *relative* integer-vs-float cost, measured in the same run with the same
binary and window.

The complete all-integer forward pass on BitNet-2B — every op, including
attention, in integer — is now measured: **+0.1239% perplexity** (full-integer
30.744724 vs. float 30.706665, teacher-forced, 199 scored tokens, same
sha-identical artifacts and window as A21, i5-10210U crosvm, argmax digest
`0xB274DE03F5862DB7`, A35), *closer* to float than the hybrid figure above —
40× inside the preregistered +5% kill line. Two sequential runs agreed on
every computed value. Reaching that number required a fix: the unfixed
binary panicked 2/2 runs (`normq: residual out of range`, `cis_infer.rs:313`).
An env-gated trace (branch `cm/e1b-normq-trace`) localized the fault to the
MLP, not attention — attention re-entry stayed exact at ≤38 bits on all 30
layers, while the ACT-I (relu²) MLP output landed on Q.20 at 52–55 bits, a
genuine spec gap: §5.10 lacked the §5.6 per-vector block exponent that the
hybrid boundary already carried. The fix — an RNE-rounded block exponent on
the ACT-I output, degenerating to the identity at M7 ranges — is ratified as
spec erratum v1.0.3 (§11). It changes only that one op: `CIS_SELFTEST
76985613c965f643` and `CIS_DECODE 67e8c0a96abc04e1`, the M7 conformance
digests reported throughout this paper, are unchanged. The A21 caveats
(pruned vocabulary, 200-token window, not cross-comparable, no timing) apply
identically to this figure.

## Cost

CIS-1's throughput cost was measured directly rather than assumed, on two
physical machines, one binary, three interleaved captures across three
shapes each, 9/9 bit-exact (A26). On the **Dell i5-5200U (Broadwell-U)**,
scalar integer costs 25% against scalar float: C/B median **1.248×**
(range 1.239–1.260). On the **HP N4020 (Gemini Lake / Goldmont Plus)**,
integer is **4–14% faster**: C/B median **0.961×** (range 0.856–0.963).
The two machines disagree in sign because the cost is a property of the
microarchitecture, not of the semantics: Broadwell's two FMA units favor the
float LUT+FMA path, while on Goldmont Plus, where floating point is weak,
integer wins. The Dell result decomposes as **4.61× absent-SIMD × 1.248×
semantics** — the earlier, larger "cost of determinism" figures this project
had previously circulated (including its own) were measuring the cost of
missing SIMD, not of integer semantics. A methodology control on the same
run — A/B on the HP, where AVX2 is absent and arm A must fall back to arm B,
so the true ratio is known in advance — measured 1.000× (0.999–1.001, n=9),
which is what licenses trusting the Dell's decomposition.

At parity SIMD width, the picture sharpens further (A27). On the Dell,
`cis_avx2::ternary_matvec_i8_avx2` measured **D/A median 0.340× — 2.94×
faster** than the hand-written float AVX2 kernel it replaces (range
0.331–0.439), bit-identical to the scalar reference 9/9 with 0 void; D/C
median 0.061× is **16.4× over the scalar reference**. This result carries a
method note that is part of its provenance, not incidental: the *first*
arm-D run measured D/A 0.276× but was **rejected** — arm A's own repeated
measurements drifted +58% across the run where clock arithmetic predicted
+28%, while arms B and C tracked the clock to 0.1%, indicating arm D had
perturbed arm A's cache state and flattered the ratio. The bench was then
changed to re-measure arm A immediately after arm D on every repetition
(A′); on the accepted run, A′/A median = 1.000× (D/A equals D/A′ to three
decimals), which is what licenses treating 0.340× as measured rather than
assumed.

## Engine context (not CIS-1 claims)

These figures characterize the surrounding unikernel/kernel-candidate work
on the Dell i5-5200U. They are not part of the CIS-1 conformance or
cross-ISA claims and are reported here only as supporting context.

The ring-0 unikernel was compared against a minimal-Linux decode path under
a preregistered, paired, hands-off-boot protocol (A22): the preregistered
throughput form gave **+3.6% / +9.4% / +5.1%** across three prompts (3/3),
with 27/27 within-boot bit-exactness and byte-identical responses. The same
protocol's precision prediction, however, **failed**: **P-V2-2 FAIL** —
measured spread 4.7–9.6% against a predicted <3%. The bit-exactness held;
the wall-time repeatability did not match the prereg's own expectation, and
that failure is reported alongside the pass.

A column-skip kernel candidate measured **2.88–2.89×** faster than the
incumbent on the real BitNet-2B `down_proj` activation distribution (ordered
variant, byte-identical to the incumbent by construction and test — zero
quality risk on adoption); a chain variant measured 2.80×. Its GMAC/s figures
are stated **NOMINAL** (skipped work is counted as done; speedup is a time
ratio), and it is not wired end-to-end — this is a kernel-level result only
(A23).

A memory-bandwidth ceiling was measured on the same machine: peak sequential
read 11.19 / 10.95 GB/s at one thread, 11.63 / 11.70 GB/s at four threads,
against a ternary weight-stream pattern of 0.62 GB/s (A24). The stream
figure is a scalar LUT-walk **lower bound** on the engine's streaming rate,
not the engine's actual rate — the bench's own caveat — and its roughly 18×
gap to peak sequential bandwidth is what motivates the column-skip work
above.

## Table 4 — every number in this section

| Number | Metric | Machine | Row |
|---|---|---|---|
| +0.3127% | Hybrid-path PPL cost, M7 (5.657126 vs 5.639491) | i5-10210U crosvm | A19 |
| +0.0637% | Full-integer PPL cost, M7 (5.643085 vs 5.639491) | i5-10210U crosvm | A20 |
| +0.7408% | **HYBRID**-path (f32 attention) PPL cost, BitNet-2B (30.934140 vs 30.706665) | i5-10210U crosvm | A21 |
| +0.1239% | **FULL-INTEGER**-path PPL cost, BitNet-2B, after v1.0.3 erratum (30.744724 vs 30.706665) | i5-10210U crosvm | A35 |
| 1.248× | Scalar integer/float throughput ratio, C/B | Dell i5-5200U (Broadwell-U) | A26 |
| 0.961× | Scalar integer/float throughput ratio, C/B | HP N4020 (Gemini Lake) | A26 |
| 4.61× × 1.248× | Absent-SIMD × semantics decomposition | Dell i5-5200U | A26 |
| 0.340× (2.94×) | AVX2 integer/float kernel ratio, D/A, parity SIMD | Dell i5-5200U | A27 |
| 0.061× (16.4×) | AVX2 integer kernel vs. scalar reference, D/C | Dell i5-5200U | A27 |
| +3.6% / +9.4% / +5.1% | Ring-0 vs. minimal-Linux decode throughput, preregistered form | Dell i5-5200U | A22 |
| 4.7–9.6% (predicted <3%, FAIL) | P-V2-2 timing-precision spread | Dell i5-5200U | A22 |
| 2.88–2.89× / 2.80× | Column-skip kernel vs. incumbent, ordered/chain (NOMINAL GMAC/s) | Dell i5-5200U | A23 |
| 11.19–11.70 GB/s vs. 0.62 GB/s | Peak sequential read vs. ternary weight-stream (LUT-walk lower bound) | Dell i5-5200U | A24 |
