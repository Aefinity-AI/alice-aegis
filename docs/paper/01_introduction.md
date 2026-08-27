# 1. Introduction

Ask two machines to run "the same" quantized language model on the same prompt and you will
usually get two different sets of logits. The weights are identical; the arithmetic is not.
Parallel reductions sum in different orders, dequantization paths differ between kernels,
normalization layers stay in floating point, and vendor libraries choose algorithms at runtime.
The disagreements are small — a few units in the last place — and they are also decisive: greedy
decoding is a chain of argmax operations, and one flipped comparison changes every token that
follows. In practice, "reproducible" means "statistically similar," and no one can say exactly
what a deployed model computed on a given input.

This paper makes reproducibility exact and then makes it *checkable*. We define **CIS-1**, a
canonical integer semantics for transformer inference: a frozen specification in which every
operation of the forward pass — the ternary matrix–vector product, activation quantization,
RMS normalization, softmax, rotary position embedding, the MLP nonlinearity, and the final argmax —
is an integer computation with pinned constants and a single normative rounding rule. A
conforming implementation is free to use any instruction set, any vectorization, any
parallelization; it is *not* free to produce a different bit. Conformance is not a matter of
opinion or tolerance: it is a digest. An op-level self-test prints one 64-bit value, and a
complete 64-token greedy decode of a reference model prints another. Either your implementation
prints those values or it does not conform.

Because the semantics are exact, they support something floating-point inference cannot: a
**decode receipt**. A receipt binds cryptographic hashes of the three model artifacts, the
generated token ids, a digest over those tokens, and a hash chain over the complete integer logit
vector at every decode step. Any conforming machine can replay the decode from source and
verify the receipt bit-for-bit. The receipt is not a claim about a trusted platform; it is a
claim about a computation, and it is checkable by anyone with the artifacts and the spec.

We demonstrate the following, each backed by an entry in the project's public research ledger
with the raw log it was taken from (Table 3):

1. **The specification is implementable from its text.** Two independent implementers, given
   only the specification and denied access to the reference source, each wrote a from-scratch
   implementation that reproduced the op-level digest on first execution (ledger A31).
2. **One semantics, two instruction sets.** The op-level digest is identical on five x86
   execution paths — two bare-metal machines spanning AVX2 and SSE2-class hardware, a virtualized
   development host, and full-system emulation — and on an aarch64 Neoverse system (A25, A28).
   A NEON kernel is bit-identical to the reference on real ARM silicon (A30), and a complete
   greedy decode produces the same token digest on x86-64 and aarch64 (A29). Both digests are
   standing continuous-integration gates: a future divergence fails the build.
3. **The receipt crosses the ISA boundary and the operating-system boundary.** A receipt minted
   on x86 verifies on aarch64 in public CI (A32). The same receipt is re-derived by a unikernel
   that boots from UEFI firmware with no operating system present, on two physical machines —
   one executing the AVX2 path, the other (which lacks AVX2) the scalar path (A33, A34).
4. **Determinism is not the expensive part.** Measured on physical hardware, the integer semantics
   cost 25% against scalar floating point on a Broadwell-U core and are 4–14% *faster* on a
   Goldmont Plus core (A26); at parity vector width the integer AVX2 kernel is 2.94× faster than
   the hand-written floating-point AVX2 kernel it replaces (A27). Earlier and larger "cost of
   determinism" figures, including this project's own, decompose into the cost of missing SIMD,
   not of integer semantics.
5. **The quality cost is small and was gated in advance.** Against a preregistered +5% perplexity
   kill line, the all-integer forward pass costs +0.06% on a 384-hidden reference model and
   +0.74% on a 2-billion-parameter ternary model (A19–A21).

We are equally explicit about what is *not* shown (§8): the clean-room implementers were
language-model agents rather than third-party engineers; the physical-machine attribution of one
boot log rests on operator witness because the verifier prints no CPU identifier; the
2B-parameter perplexity figure uses a pruned vocabulary and a short window and is not comparable
to published numbers; and no token-level throughput figure for the integer path has yet been
measured. The project publishes negative results and retractions in the same ledger as its
claims, and a standing falsification bounty invites anyone to find a machine on which the digests
diverge.

The contribution, then, is not a faster kernel or a smaller model. It is a way to say, and to
prove, exactly what a model computed — on hardware you do not have to trust, without an operating
system you would otherwise have to audit.
