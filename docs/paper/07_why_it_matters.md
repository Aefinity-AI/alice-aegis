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

**For research reproducibility.** Every result in this paper is a digest or a logged measurement
on a named physical machine. A reader who obtains the artifacts does not have to trust our
numbers; they can print the same 64-bit values. We believe this is the correct standard for
inference-engine claims and that the industry's tolerance-based "matches within epsilon"
reporting has hidden real divergence for years.

**For the economics of edge AI.** The engine that produces these receipts runs a 2-billion-
parameter ternary model on a decade-old laptop with no operating system, and the integer path is
faster than the floating-point path on the hardware where it was measured. Verifiability did not
cost performance; on commodity CPUs it improved it. That changes the calculus for organizations
that assumed provable inference required specialized or trusted hardware.

**A standing invitation.** The project maintains a public falsification bounty: find any machine
on which a conforming build fails to reproduce the digests and the author will pay and record the
finding in the research ledger as a deliverable. We would rather buy a counterexample than defend
a claim. Reviewers are invited to try before publication.
