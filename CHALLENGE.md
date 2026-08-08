# The $50 Falsification Challenge

I'm Justin Thompson. I build AI inference engines in Orange, Texas, and I
claim something most of the AI industry can't: **my engine's output is
bit-for-bit identical on any hardware you run it on.** Not "close." Not
"statistically similar." Identical. Every logit, every token, every time.

That claim is either true or it isn't. So here's the deal:

**Find any machine where it isn't, and I'll pay you $50 and put your name
in my research ledger as the person who broke it.**

I'm a one-man shop. Fifty bucks is real money to me — and I'm offering it
because paying to be proven wrong is the cheapest research there is. This
program publishes its negative results as deliverables. If you falsify my
core claim, that's the most valuable negative result I could buy.

## The claim, precisely

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

## What wins the $50

- A **reproducible** run of an unmodified conforming build (stable Rust,
  build commands from the README) that produces a **different digest or a
  witness FAIL** on real hardware, with the full log attached.
- First finder per **distinct root cause** gets paid. If your finding
  exposes a spec hole (the spec text permits two readings that produce
  different bits), that counts — that's the most interesting kind.

## What doesn't

- Modified source, non-conforming toolchains, or builds that skip the
  documented gates.
- Failing hardware (bad RAM, overclocks past stability). If it doesn't
  reproduce, it's not a finding.
- Bugs that don't change the digests (report them anyway — I'll credit
  you in the ledger, they're just not what the bounty is for).

## How to report

Open an issue titled `FALSIFICATION:` with your hardware, OS/toolchain
versions, exact commands, and complete output. I'll reproduce it, publish
the finding under your name in `program/RESEARCH_LEDGER.md` with the raw
log — same treatment as every other result in this program — and pay you.

Every number in this repo traces to an instrument log from a named machine.
If that discipline sounds extreme, read the ledger: it's the reason I'll
take this bet and the reason you should want to break it.

— Justin B. Thompson, Aefinity AI
