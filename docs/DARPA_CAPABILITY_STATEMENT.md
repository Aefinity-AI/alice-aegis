# Capability Statement — Assured On-Device Autonomy for Disconnected Life-Safety

**Aefinity AI** (Texas, incorporating 2026) · Point of contact: Justin B. Thompson
Prepared 2026-07-10 · For a DARPA SBIR/STTR Direct-to-Phase-II white paper (DSIP, dodsbirsttr.mil)

> **Reading note (integrity policy).** This statement distinguishes what is
> **DEMONSTRATED** (measured, logged, reproducible today) from what is
> **PROPOSED** (the objective of the requested work). No number in the
> Demonstrated section is estimated; each is produced by a command in a public
> repository and gated against fabrication. Nothing in the Proposed section is
> claimed as built.

---

## 1. The operational gap

A person can be incapacitated — a fall, a blast, a medical collapse, an assault —
in a setting where no network is reachable and no human is in range, with
capable computing *already on their body* (phone, wearable), and that computing
does nothing, because it cannot decide and act without a data center in the
loop. The window in which intervention saves a life is minutes. Current
on-device AI cannot run a capable model continuously, offline, within a
wearable's power and trust budget, and autonomously summon aid.

This gap is not academic. It is present in dismounted operations, isolated
outposts, contested/EW-denied environments, and every disconnected setting where
the warfighter — or the civilian — is beyond reach at the moment of injury.

## 2. The proposed capability (the objective of this work)

**An assured, self-contained inference kernel that runs continuously on
commodity edge hardware with no operating system, fuses on-body sensor streams,
recognizes a life-threatening event, and initiates a summons-for-aid action —
deterministically, offline, and within a wearable-class power budget.**

The revolutionary element is not efficiency. It is *autonomy under
disconnection with a trusted, auditable decision path* — a capability that does
not exist in fielded systems and that a cloud-dependent or general-OS stack
structurally cannot provide.

## 3. Feasibility already DEMONSTRATED (the Direct-to-Phase-II basis)

The hard part of §2 — running a capable neural model on constrained commodity
hardware, with no operating system, deterministically, on a tiny auditable code
base — is **already working and measured today.** The following are reproducible
from a committed repository:

- **A 2-billion-parameter language model performs coherent inference with NO
  operating system**, booting from USB directly into firmware on a commodity
  x86 laptop. Verified on physical hardware (Dell Inspiron 15), with a machine-
  written transcript recovered from the boot medium: prompted for the capital of
  France, it answered correctly and terminated on its own.
- **The entire trusted code base is 11,391 lines of Rust** (`aegis-core/src`
  8,226 + `aegis-uefi/src` 3,165; 8,708 excluding blanks and comments — count
  reproduces with `cat aegis-core/src/*.rs aegis-uefi/src/*.rs | wc -l`),
  **compiling to a 343 KB executable**
  (`artifacts/BOOTX64_GATE1_2026-07-31_afa1dd18.EFI`), with two runtime
  dependencies. No kernel, no network stack, no
  package manager — an attack and audit surface orders of magnitude smaller than
  any OS-hosted stack.
- **It runs across a decade of hardware from one binary**, detecting CPU
  capability at runtime and falling back from vector to scalar execution —
  verified on emulated 2008-era CPUs through current silicon.
- **It measures and manages its own processor power state from firmware**
  (reads APERF/MPERF, requests P-states via HWP/SpeedStep) — the function an
  operating system's governor normally performs, here performed by the
  application itself. This is the mechanism a continuous wearable guardian
  requires to meet a power budget without an OS.
- **Every performance claim is measured, not asserted.** WikiText-2 perplexity
  and per-token energy are computed by the running engine; a repository gate
  rejects any source that prints a metric it did not compute. This directly
  answers DARPA's documented preference for *verifiable* AI: no number in this
  program can be fabricated, by construction.

*(Measured values — perplexity, tokens/second, joules/token, and the P-state
findings — are provided with their producing commands in the accompanying
technical report. They are omitted from this one-pager only to keep it to a
page, not because they are soft.)*

## 4. Why this is infeasible elsewhere (the constrained-hardware argument)

On the oldest, most resource-limited target hardware, mainstream inference
stacks **cannot run at all** — they require an operating system, more memory,
or CPU features the hardware lacks. This engine runs there anyway. That is a
capability-that-shouldn't-be-possible on that hardware, demonstrated with a log
— precisely the bar DARPA's Low Resource Computing line sets ("*capabilities
otherwise believed not feasible on that hardware*"). The seven-machine
measurement campaign now in progress is designed to produce exactly this
evidence: the set of real, legacy, donated machines on which every alternative
stack fails and this one succeeds.

## 5. Metrics — test-driven, all measurable

Phase objectives are stated as measurements a government evaluator can
reproduce, not as adjectives:

| Objective | Measured how | Status |
|---|---|---|
| Runs with no OS on constrained hardware | logged boot + generation transcript on the target | **demonstrated** |
| Continuous operation within wearable power budget | joules/token and sustained watts, battery-discharge method | partial (whole-system energy measured; wearable-class port proposed) |
| Deterministic, auditable decision path | bit-identical output across runs; trusted-base line count | demonstrated for inference; sensor-fusion path proposed |
| Event recognition true/false-positive rates | labeled event corpus, held-out test | **proposed** — the core Phase II research |
| Assured no-fabrication of reported metrics | repository integrity gate, CI-enforced | **demonstrated** |

## 6. Transition and mechanism

Direct-to-Phase-II via the DoW 2026 BAA through DSIP; Aefinity AI is a US-owned
for-profit small entity (incorporating in Texas, SAM.gov registration in
progress). Dual-use transition is immediate and civilian: the same guardian
capability applies to eldercare, lone-worker safety, and disability support —
any person who may be unable to summon the help that on-body technology could
already provide.

## 7. Honest risk statement

The inference substrate is proven. **The event-recognition capability is the
research** — sensor fusion, false-positive control, and an assured decision
path are unbuilt and are the reason the funding is requested, not a claim of
prior accomplishment. The proposal succeeds or fails on whether that capability
can be measured to a fieldable true/false-positive rate on the demonstrated
substrate. That is a question this team proposes to answer with data, in the
open, the same way every claim in the substrate was answered.

---

## Appendix A — Heilmeier Catechism (the reviewer's checklist)

DARPA program managers judge proposals against George Heilmeier's questions.
Answered here in plain language, one paragraph each.

**1. What are you trying to do?**
Build a self-contained AI that runs on a phone or wearable with no operating
system, recognizes when its wearer has suffered a life-threatening event, and
summons help — working offline, when no network and no person can be reached.

**2. How is it done today, and what are the limits of current practice?**
On-device AI today needs an operating system, a network, or a data center in the
loop. It cannot run a capable model continuously within a wearable's power and
trust budget, and it cannot decide and act autonomously when disconnected. So at
the moment a person is incapacitated beyond reach, the capable computing already
on their body does nothing.

**3. What is new, and why do you think it will succeed?**
An inference engine that runs a 2-billion-parameter model with no operating
system on commodity hardware, on an 11,391-line auditable code base, managing its
own power state from firmware. It will succeed because the hard part — capable,
deterministic, OS-free inference on constrained hardware — is already built and
measured on real machines. The remaining work is event recognition on that
proven substrate.

**4. Who cares? If you succeed, what difference will it make?**
Anyone who can be injured beyond reach: the dismounted or isolated warfighter in
a disconnected or EW-denied environment, and — in immediate dual-use — the
eldercare patient, the lone worker, the person living with a disability. The
difference is the minutes between injury and aid, which is the difference
between recoverable and fatal.

**5. What are the risks?**
The inference substrate is proven; the event-recognition capability is not built
and is the research. The risk is whether detection can be measured to a fieldable
true/false-positive rate on wearable-class hardware. That is the question the
funding is requested to answer — stated as a risk, not disguised as a result.

**6. How much will it cost and how long?**
Direct-to-Phase-II under the DoW 2026 BAA: a ~24-month effort at the SBIR Phase II
scale (≈$1.8M cost ceiling), with a defined base period for the detection
research and an option period for the wearable-class hardware port and field
evaluation. *(Exact cost volume and schedule to be built to the specific topic;
figures here are the program's nominal Phase II envelope, not a quote.)*

**7. What are the mid-term and final exams?**
Mid-term: on the demonstrated substrate, a labeled event corpus yields a measured
detection true/false-positive rate, and the engine sustains continuous operation
within a stated power budget on wearable-class hardware — both reproducible by a
government evaluator. Final: an end-to-end demonstration on real hardware in which
a simulated life-threatening event, offline, triggers a correct summons-for-aid
action, with the full decision path logged and auditable.
