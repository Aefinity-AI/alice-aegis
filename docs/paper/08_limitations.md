# 8. Limitations and what is not claimed

We state these as precisely as the ledger does, because each is where a careful reviewer should push.

**Scope of the semantics (spec §10).** CIS-1 v1.0 covers the BitNet-b1.58-style decoder block
(ternary linear layers, RMSNorm, GQA attention with RoPE, squared-ReLU MLP) with hidden size
384–2560, head dimension ≤128, and sequence length ≤512; headroom bounds are derived for these
ranges only. Bit-identity across kernel paths is claimed for the full-integer pipeline (v0.3 and
later); the earlier hybrid path, which kept attention and activations in floating point, is not
claimed to be cross-path identical and is reported only for the quality comparison in §6.

**The clean-room implementers were language-model agents.** The two from-scratch implementations
that reproduced the op-level digest (A31) were written by LLM subagents given the specification
text as their sole source, with reference-source access forbidden and, as far as the harness can
establish, not used. This demonstrates implementability from text; it is not an independent
third-party audit, and we do not describe it as one.

**Attribution of the bare-metal boot logs.** The unikernel's verifier prints no CPU identifier.
The Dell log entry is attributable to that machine because its firmware memory map differs from
the emulator's; the HP log entry's memory map is identical to the Dell's, so in-log evidence does
not discriminate the two machines and attribution of that entry rests on the operator's physical
presence (A33, A34). The scalar-path corollary — the HP processor lacks AVX2, so a pass on it
implies the SSE2 path executed — depends on the same attribution.

**Perplexity figures are not cross-comparable.** The 2B-parameter integer-vs-float comparison
(A21) uses a vocabulary pruned from 128,256 to 50,256 tokens and a 200-token evaluation window in
an out-of-vocabulary-dense region; the resulting absolute perplexity must not be compared with
published full-vocabulary numbers or with this project's own longer-window anchors. Only the
*relative* integer-vs-float cost, measured in the same run with the same binary and window, is the
claim.

**Cost figures are microarchitecture-specific.** The 25% cost on Broadwell-U and the 4–14%
advantage on Goldmont Plus (A26) are measurements of two machines, not a model of CPUs in
general; the 2.94× AVX2 result (A27) is one machine, three shapes, three captures, with its
rejected first run and the order control recorded. No token-level throughput number for the
integer path exists yet, and the README says so.

**Cloud-runner legs are identity evidence only.** The aarch64 results (A28, A29, A30, A32) were
obtained on hosted CI runners; the machine is named as precisely as the platform allows, and no
timing is quoted or quotable from them.

**What the receipt does not do.** A receipt proves that a conforming computation over the bound
artifacts produced the bound outputs. It does not prove which physical machine ran it (that is
platform attestation's job), does not hide the model or the prompt (it is not a zero-knowledge
proof), and does not protect against an adversary who controls the artifacts and the verifier
together.

**Retractions.** This project has previously published figures it later retracted — an unlogged
throughput number and two kernel-comparison claims among them. They remain in the ledger, marked,
with the reasons. Every figure in this paper carries a ledger row and a raw log path (Table 3),
and the reader should treat any number without one as a description, not a measurement.
