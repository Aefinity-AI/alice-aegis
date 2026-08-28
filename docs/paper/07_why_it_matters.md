# 7. Why this matters

**For deployment without trust.** Edge, air-gapped, and sovereign deployments run models on
hardware and firmware the operator cannot fully audit and networks they cannot rely on. Today the
operator's assurance that "the model we validated is the model that ran" rests on platform
attestation — a measured boot chain that vouches for the software stack — plus the hope that the
stack is deterministic. CIS-1 removes the hope. A conforming engine's output is a mathematical
function of the artifacts and the input; the decode receipt is the evidence, and any conforming
machine, including one booted from a USB stick with no operating system, can check it. Platform
attestation and computational verification are complementary: one says which binary ran, the
other says what that binary computed.

**For audit and regulation.** Regulators and incident investigators increasingly need to answer
"what did the system output for this input, and can you prove it?" With floating-point inference
the honest answer is a distribution. With CIS-1 it is a receipt: hashes of the exact artifacts,
the exact tokens, and a commitment to every intermediate logit vector, replayable by the auditor
on their own hardware. The receipt format is small, the verifier is the reference implementation,
and the specification is public.

**For research reproducibility.** Every result in this paper is either a digest reproduced in
public continuous integration or a measurement logged on a named physical machine, and Table 3 says
which. A reader who obtains the artifacts does not have to trust our
numbers; they can print the same 64-bit values. We believe this is the correct standard for
inference-engine claims and that the industry's tolerance-based "matches within epsilon"
reporting has hidden real divergence for years.

**For the economics of edge AI.** The engine that produces these receipts boots from firmware with
no operating system on decade-old laptops and re-derives the receipt for the reference model there
(A33, A34); the same engine runs a 2-billion-parameter ternary model in its Linux harness on the same
class of hardware (A21). The complete all-integer forward pass on that 2B model now passes its
preregistered quality gate (+0.1239% perplexity against float, 40× inside the +5% kill line, A35) —
though the 2B model has not yet been booted on the unikernel itself; no ledger row yet establishes
that. The cost of the integer semantics is a property of the
microarchitecture, not of the semantics: 25% against scalar floating point on one core, 4–14% faster
on another (A26), and at parity vector width the integer AVX2 kernel is 2.94× faster than the
floating-point AVX2 kernel it replaces (A27). On the commodity CPUs where it was measured,
verifiability did not require specialized or trusted hardware, and did not have to cost performance.

**A standing invitation.** The project maintains a public falsification bounty: find any machine
on which a conforming build fails to reproduce the digests and the author will pay and record the
finding in the research ledger as a deliverable. We would rather buy a counterexample than defend
a claim. Reviewers are invited to try before publication.
