# The $50 Falsification Challenge

I'm Justin Thompson. I build AI inference engines in Orange, Texas, and I
claim something most of the AI industry can't: **my engine's output is
bit-for-bit identical on any hardware you run it on.** Not "close." Not
"statistically similar." Identical. Every logit, every token, every time.

That claim is either true or it isn't. So here's the deal:

**Find any machine where it isn't, and I'll pay you $50 and put your name
in my research ledger as the person who broke it.**

As of 2026-09-09 the bounty covers three tracks: the frozen **integer**
semantics (CIS-1, this repo), the **floating-point** semantics (CIS-2,
`github.com/Aefinity-AI/cis2-spec`), and the **agent-episode receipt**
(`demo/agent-trace/`, this repo). $50 per distinct root cause, first
finder, any track.

I'm a one-man shop. Fifty bucks is real money to me — and I'm offering it
because paying to be proven wrong is the cheapest research there is. This
program publishes its negative results as deliverables. If you falsify my
core claim, that's the most valuable negative result I could buy.

## Track A — CIS-1: the integer claim, precisely

CIS-1 (`docs/CIS-1_SPEC_v1.0.md`) is a frozen integer semantics for
transformer inference. Any conforming build of this repo, on any hardware,
must reproduce:

1. **Op-level:** `cis_selftest` prints
   `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true`
2. **Token-level:** `cis_decode` on the in-repo M7 model
   (64 tokens, prompt "Once upon a time") prints
   `CIS_DECODE digest=67e8c0a96abc04e1`
3. **Witness:** `cis_witness verify` on
   `tests/golden/witness_v1_m7_once64.receipt` — a receipt minted on my
   x86 dev box — prints `VERIFY PASS`, meaning your machine reproduced
   every one of the 64 decode steps' full logit vectors, hash-chained.

These already hold on: bare-metal AVX2 (Dell i5-5200U), bare-metal
SSE2-class (HP Celeron N4020), a virtualized i5-10210U, QEMU emulation, and
aarch64 Neoverse — the CI in this repo re-proves both digests on x86-64 and
ARM on every commit. I'm claiming they hold on *your* machine too.

## Track B — CIS-2: the floating-point claim, precisely

The harder one. CIS-2 (`github.com/Aefinity-AI/cis2-spec`,
`docs/CIS2_SPEC_v0.3b.md`) is a normative spec for the **fp32** decode path
of an ordinary transformer — not ternary, not quantized. It pins the things
IEEE-754 leaves to the implementer: reduction order (strictly left-to-right),
FMA contraction (forbidden), denormals (FTZ/DAZ on, pinned via MXCSR and
FPCR.FZ), the sin/cos and exp polynomials (pinned as f32 hex literals), and
the RoPE `inv_freq` table. It folds the **complete fp32 logit vector** at
every decode step — not the argmax, the whole vector — into a SHA-256
witness chain.

The claim: any conforming implementation, written from that document, on any
CPU, reproduces

```
CIS2_REF, SmolLM2-135M, prompt "Once upon a time", 16 greedy tokens
  d82743059d1db929e710236fe4ec37f89e6f932524801345a006980f7c3cc9df
```

That value currently holds for a Rust reference and two clean-room
implementations (Rust and C11) written from the spec text alone, on native
x86_64 and native aarch64, under gcc and clang — CI in that repo re-proves
all of it on every push — and, as of 2026-09-08, for a CUDA implementation on
an NVIDIA Tesla P100 (`docs/GPU_RESULT.md`), where the per-step trace is
byte-identical to the CPU one, not merely equal at the digest.

**Two ways to win Track B.** Run the published reference or either
clean-room on hardware where it disagrees; *or* — the one I most want —
write your **own** implementation from `docs/CIS2_SPEC_v0.3b.md`, in any
language, for any device, and get a different digest. The GPU port I ran is
deliberately not published, so a second GPU implementation is a genuine
independent check rather than a re-run of my code. If your implementation
disagrees because the spec text permits two readings, that is the finding I
am paying for: it means the document is not yet sufficient, which is the
entire thing CIS-2 claims to be.

Read `docs/GPU_RESULT.md`'s "Scope" section before you start — it says
plainly what the GPU result does not cover (one Pascal device, one
toolchain, no tensor cores, no batching, correctness only, no timing). Those
are open ground, not claims to be broken.

## Track C — the agent receipt

`demo/agent-trace/` mints an **AEGIS-TRACE v1** receipt over a whole
multi-step agent episode: K rounds of integer-only decode, tool calls, and
tool results, hash-chained, with each step binding both the exact context it
saw and the query text it was given. `run.sh verify` replays the episode
from the receipt header on your machine and compares every step.

The claim: on any machine, `demo/agent-trace/run.sh all` reproduces the
episode and verifies it, and `run.sh tamper` fails all four mutations. Win
by producing a **false PASS** — a mutated receipt that `verify` accepts —
or a **false FAIL** — an honest replay on real hardware that `verify`
rejects. A false PASS is worth the most to me and is the one I would attack
first: find a change to a token id, tool input, tool output, context, or
step ordering that the chain does not notice.

## What wins the $50

- A **reproducible** run of an unmodified conforming build (stable Rust,
  build commands from the README) that produces a **different digest, a
  witness FAIL, or a false PASS** on real hardware, with the full log
  attached.
- On Track B only, an implementation you wrote yourself from the spec text
  counts as conforming for this purpose even though it is not my code —
  that is the point of the track. Say which spec clauses you implemented
  from and include the source, so the disagreement can be localized.
- First finder per **distinct root cause** gets paid, across all three
  tracks. If your finding exposes a spec hole (the spec text permits two
  readings that produce different bits), that counts — that's the most
  interesting kind, and on Track B it is the primary target.

## What doesn't

- Modified source, non-conforming toolchains, or builds that skip the
  documented gates.
- Failing hardware (bad RAM, overclocks past stability). If it doesn't
  reproduce, it's not a finding.
- Bugs that don't change the digests (report them anyway — I'll credit
  you in the ledger, they're just not what the bounty is for).

## How to report

Open an issue titled `FALSIFICATION:` with your hardware, OS/toolchain
versions, exact commands, and complete output — in this repo for Track A or
Track C, in `Aefinity-AI/cis2-spec` for Track B (either is fine if you're
not sure which). I'll reproduce it, publish
the finding under your name in `program/RESEARCH_LEDGER.md` with the raw
log — same treatment as every other result in this program — and pay you.

Every number in this repo traces to an instrument log from a named machine.
If that discipline sounds extreme, read the ledger: it's the reason I'll
take this bet and the reason you should want to break it.

— Justin B. Thompson, Aefinity AI
