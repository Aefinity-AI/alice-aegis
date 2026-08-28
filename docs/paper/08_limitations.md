# 8. Limitations and what is not claimed

We state these as precisely as the ledger does, because each is where a careful reviewer should push.

**Scope of the semantics.** The frozen v1.0 spec fixes the data types and grids (spec §3) and lists what is not claimed (spec §10); the numeric operating envelope below is the one the headroom derivations were carried out for, recorded in ledger rows A20–A21 and the v0.3 working notes (CIS-1_SPEC_DRAFT_v0.1.md), not in the frozen text itself. CIS-1 v1.0 covers the BitNet-b1.58-style decoder block
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

**The `cis-verify` reimplementation is an LM-agent transcription, not a clean-room audit.** Like
the two implementers behind A31, `cis-verify` (§5) was built by a language-model agent — given the
specification and the receipt format as reference material, in three phases — not by an
independent third party working blind to the source. This demonstrates that the spec and receipt
format are re-implementable without `aegis-core`'s SIMD/dispatch code and data structures; it is
not the same claim as an external auditor's clean-room verification, and we do not describe it as
one (A37).

**Attribution of the bare-metal boot logs.** The unikernel's verifier prints no CPU identifier.
The Dell log entry is attributable to that machine because its firmware memory map differs from
the emulator's; the HP log entry's memory map is identical to the Dell's, so in-log evidence does
not discriminate the two machines and attribution of that entry rests on the operator's physical
presence (A33, A34). The scalar-path corollary — the HP processor lacks AVX2, so a pass on it
implies the SSE2 path executed — depends on the same attribution.

**Perplexity figures are not cross-comparable.** The 2B-parameter integer-vs-float comparisons —
the hybrid path (A21) and the complete all-integer path after the v1.0.3 erratum (A35) — use a
vocabulary pruned from 128,256 to 50,256 tokens and a 200-token evaluation window in an
out-of-vocabulary-dense region; the resulting absolute perplexities must not be compared with
published full-vocabulary numbers, with this project's own longer-window anchors, or with each
other across runs beyond what the shared, sha-identical artifacts and window support. Only the
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

**BitNet-2B's token-level identity leg is x86-only.** A36 establishes the FullInt decode digest
for BitNet-2B on x86-64, identical across two runs; the aarch64 leg that A29 provides for the
smaller M7 model has not yet been run at 2B scale, so the cross-ISA claim in §4 does not yet
extend to BitNet-2B.

**What the receipt does not do.** A receipt proves that a conforming computation over the bound
artifacts produced the bound outputs. It does not prove which physical machine ran it (that is
platform attestation's job), does not hide the model or the prompt (it is not a zero-knowledge
proof), and does not protect against an adversary who controls the artifacts and the verifier
together.

**Retractions.** This project has previously published figures it later retracted — an unlogged
throughput number and two kernel-comparison claims among them. They remain in the ledger, marked,
with the reasons. Every figure in this paper carries a ledger row and a raw log path (Table 3),
and the reader should treat any number without one as a description, not a measurement.
