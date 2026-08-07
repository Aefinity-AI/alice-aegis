# The Gauntlet — one boot, every approach raced against itself

Design goal: each borrowed/donated laptop boot is expensive (~15 min). So one
command must extract the **maximum measured comparison** from that boot, in an
ordered sequence where **each segment is its own control** on the same silicon —
the same principle as the turbo-vs-baseline A/B already in `/autotest`.

Ordering rule (inherited from `/autotest`): cheapest/most-robust first, so an
interrupted run still yields the early rows. Warmup is always unrecorded.

## The segments (each writes a structured `GAUNTLET:` line to BOOTLOG.TXT)

| # | Segment | What it races | Why it matters | Cost |
|---|---|---|---|---|
| 0 | Identify | — | CPU, freqs, RAM, feature bits, idle clock | instant |
| 1 | Warmup | — | prime caches/predictors, not recorded | 1 gen |
| 2 | **SIMD value** | scalar path vs AVX2 path, same chip | what vectorization buys on THIS microarch — narrows toward 1.0 on weak-SIMD chips, which is itself the finding | 2 bench |
| 3 | **Batching value** | per-token prefill vs batched GEMM | what the GEMM buys on THIS memory hierarchy | 2 prefill |
| 4 | **P-state** | base clock vs /turbo (with RUN1/RUN2 drift control) | the headline: does removing the OS throttle the core, and can the app fix it | 3 bench |
| 5 | **Context slope** | decode tok/s at 20 / 100 / 400 tokens | KV-cache growth cost per machine — connects to the "where do the joules go" thesis | 3 gen |

Each segment reports prefill and decode separately: prefill tokens/ticks/seconds,
decode ticks/token, and **decode-only** wall-clock tok/s. The segment's wall time
(UEFI `GetTime()`) is split across the two phases in proportion to their TSC
ticks (invariant-rate), because a whole-run average lets a fixed prefill cost
amortize over longer outputs and fake a speedup with generation length — the
first QEMU run reported a bogus 1.44x "speedup" at 400 tokens exactly this way.
Where available, the actual clock % from APERF/MPERF is appended, so a throttled
or turbo'd run is never mistaken for a fast or slow kernel.

## Why this needs RUNTIME toggles, not compile-time features

`legacy_matmul`, `int8_act`, and forced-scalar are compile-time features today.
You cannot rebuild on a borrowed laptop. To race them in one boot, the switches
must be **runtime** and **default-off** (so default behavior stays byte-identical
and the coherence gate still guards it). That is the enabling change; everything
else is orchestration.

## Staged implementation — ordered by (value / risk)

**Tier 1 — safe, lands now (exposes already-tested paths at runtime):**
- runtime `force_scalar` flag → enables segment 2
- runtime `force_legacy_prefill` flag → enables segment 3
- `/gauntlet` REPL command orchestrating segments 0–5
- Guarded by the coherence gate + `gemm_equivalence` + `thread_safety` tests;
  defaults off, so a normal boot is unchanged.

**Tier 2 — after fleet v1 (real new instrumentation):**
- per-phase forward-pass timing (prefill / attention / FFN / LM-head), to show
  *where* the time goes on each machine, not just the total.
- int8-vs-f32 activation race — deferred because perplexity is too slow to
  measure per boot and the speed delta is ~2%; better done once, offline.

**Tier 3 — the far-flung, proposal-only until reviewed:**
- **CTZ-vs-SIMD across hardware.** We proved CTZ loses 6.3x at 42% sparsity on
  *one* chip. The open micro-question: does the crossover shift on old,
  weak-SIMD CPUs where an FMA is relatively less dominant? Porting the CTZ kernel
  into the live engine behind a runtime flag would let the fleet answer it. This
  is genuinely novel and genuinely the most work; it touches the hot path, so it
  does not land unreviewed.
- **Second model size.** Racing the approach-ratios on a smaller model would show
  how they scale with parameter count — but needs another model prepared. Big
  lift, deferred.

## Pre-registered predictions (before the fleet runs)

- **Segment 2:** the scalar/AVX2 ratio *shrinks* on older CPUs (weaker/narrower
  SIMD). On a pre-AVX2 machine it is exactly 1.0 (both take the scalar path).
- **Segment 3:** the batching benefit is *larger* on machines with less memory
  bandwidth, because that is where re-streaming weights per token hurts most.
- **Segment 4:** on true bare metal, base-clock runs read <60% of nominal and
  `/turbo` lifts them >90% with a ≥1.5x throughput gain — unless the firmware
  already boots at a high P-state (then the effect is small, and *that* machine
  becomes the control that proves the mechanism).
- **Segment 5:** tok/s falls monotonically with generation length on every
  machine; the *slope* correlates with memory bandwidth, not clock.

If a prediction fails, that is the finding. The last two pre-registrations each
caught the author's error, not the operator's.
