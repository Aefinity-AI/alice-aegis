# Pre-registered: does the absence of an operating system throttle the CPU?

**Written and committed BEFORE the test.** The previous pre-registration
(`PREREGISTERED_HARDWARE_TEST.md`) falsified a prediction of mine and produced the
hypothesis below. This one tests it.

## The hypothesis (H1)

A Dell Inspiron 15 running the A.L.I.C.E. unikernel measured **~3.65 B TSC ticks/token**,
against 0.44–0.60 B on an i5-10210U Chromebook — roughly **7× slower**. Microarchitecture
explains perhaps 1.2–1.3× of that.

> **H1: There is no operating system to raise the processor's P-state.** The firmware
> leaves the core near its minimum clock, nothing ever asks it to go faster, and the
> engine runs there for the entire session.

`rdtsc` cannot see this. The TSC is *invariant* — it ticks at nominal frequency however
slowly the core is actually running — so a throttled core merely appears to need more
"cycles" per token.

## The competing hypothesis (H2), which must be ruled out

> **H2: The Dell is simply a slower machine** — older microarchitecture, slower or
> single-channel memory, smaller caches — and its clock is already near nominal.

H1 and H2 make **opposite, directly measurable predictions**, so this is decidable.

## The instrument

`IA32_APERF` (MSR 0xE8) counts at the **actual** core clock.
`IA32_MPERF` (MSR 0xE7) counts at the **nominal** clock.
Their ratio across an interval **is** the P-state — measured, not inferred.

Two new REPL commands read it, both gated on `CPUID` *and* on the absence of a hypervisor,
so an unsupported `rdmsr` can never fault:

- `/cpuinfo` — CPU brand, CPUID leaf 0x16 base/max/bus MHz, HWP support, the current /
  base / 1-core-turbo P-state ratios, and the actual clock as a percentage of nominal
  measured over a 1-second idle window.
- `/turbo` — asks the processor for maximum performance. On Skylake and later via HWP
  (Speed Shift: `IA32_PM_ENABLE`, `IA32_HWP_REQUEST`); on Sandy Bridge through Broadwell
  via legacy Enhanced SpeedStep (`IA32_PERF_CTL[15:8]`). **This is precisely the job a
  Linux `cpufreq` governor performs, and which nothing performs here.**

Generation and `/benchmark` now also record **wall-clock seconds** from UEFI Runtime
Services `GetTime()`, giving true tokens/second independent of any TSC assumption, and the
**actual clock percentage sustained during the run**.

## Protocol — one command per machine

```
/autotest     ->  1. identify CPU, measure idle clock
                  2. warmup generation (not recorded)
                  3. RUN1  baseline benchmark
                  4. RUN2  repeat, NOTHING changed        <- measures drift
                  5. /turbo, raise the P-state
                  6. RUN3  benchmark after turbo
/exit
```

### Why RUN2 exists (a flaw caught in a QEMU dry run, before any real data)

A first version ran one benchmark before `/turbo` and one after. Under QEMU — where
`/turbo` **fails outright**, changing nothing — the second benchmark still came out
**~6% faster** (2.94 → 3.11 tok/s). Frequency ramp, TLB warmth, branch predictors.

Had that shipped, any turbo effect under ~10% would have been indistinguishable from an
artifact of running second. So the protocol now runs two identical benchmarks back to
back, changes nothing between them, and only then raises the P-state:

```
drift  = RUN2 / RUN1     (the noise floor: nothing changed)
effect = RUN3 / RUN2     (only the P-state changed)
```

**An effect that does not clearly exceed the drift is not an effect.**

## PREDICTIONS

### If H1 is true (no OS ⇒ nothing raises the P-state)

- **Q1.** RUN1 and RUN2 report an actual clock **below 60% of nominal** under load.
- **Q2.** `/turbo` succeeds (via HWP or legacy SpeedStep).
- **Q3.** RUN3 reports an actual clock **above 90% of nominal.**
- **Q4.** `effect ≥ 1.5×`, and `effect` exceeds `3 × |drift − 1|`.

### If H2 is true (the machine is simply slower)

- **Q1′.** The clock is already **near or above 100% of nominal** before `/turbo`.
- **Q4′.** `|effect − 1| ≤ 10%`, comparable to drift — no headroom existed.

### Either way

- **Q5.** The CPU brand string and CPUID leaf 0x16 identify the exact processor, settling
  the nominal frequency without anyone reading a BIOS screen.
- **Q6.** `drift` is small (≤ ~6%) and roughly consistent across machines. If drift is
  large or erratic, the instrument is not trustworthy and no conclusion may be drawn.

## The multi-machine study

Seven laptops, each running `/autotest` once. **Each machine is its own control** —
RUN1/RUN2/RUN3 happen on the same silicon, memory, and firmware, so cross-machine
confounds (microarchitecture, RAM speed, cache size) cancel inside each row.

`collect_autotest.sh <device> <nickname>` appends one row per machine to
`docs/hardware_logs/pstate_dataset.tsv` and prints the drift/effect analysis and a
verdict. Cross-machine comparison of *absolute* tok/s is secondary and confounded;
the *within-machine* effect is the result.

**Prediction for the study (Q7):** if H1 holds, the effect will be large on machines whose
idle clock reads far below nominal, and small on machines whose firmware already boots at
a high P-state. The correlation between `idle_clock` and `effect` is itself the finding.

## What each outcome means

| Outcome | Consequence |
|---|---|
| **H1 confirmed** | §5.6 of the technical report inverts. Running without an operating system is *substantially slower*, because the OS was doing something valuable nobody noticed — asking the CPU to go fast. The unikernel's case rests entirely on auditability and determinism, with a performance penalty to declare honestly. **And the unikernel can fix it itself, in ~40 lines.** |
| **H2 confirmed** | §5.6 survives. The Dell is simply an older machine, and bare-metal inference costs nothing in throughput. The 7× is microarchitecture and memory. |
| **H1 confirmed AND `/turbo` closes the gap** | The strongest result available: a bare-metal LLM that manages its own P-states, plus a measured demonstration of exactly what the operating system was contributing. |

## Known limitations, stated in advance

1. **`GetTime()` resolution is 1 second on OVMF** (the nanosecond field reads zero). Over a
   ~2-minute `/benchmark` that is <1% error. It is useless for short generations.
2. **If the Dell is Haswell it has no HWP** (Speed Shift arrived with Skylake). The legacy
   SpeedStep path handles it, but if firmware has EIST disabled (`IA32_MISC_ENABLE[16]`),
   `/turbo` will fail and say so rather than pretend.
3. `MSR_PLATFORM_INFO` and `MSR_TURBO_RATIO_LIMIT` are model-specific. They are read only
   when the CPU is Intel and no hypervisor bit is set. Under QEMU all of it is skipped —
   verified, no fault.
4. A single machine, one workload. Whatever we learn is about this Dell.

## Result

```
STATUS: NOT YET RUN
```

---

*The last pre-registration caught the author's error rather than the operator's. That is
what they are for.*
