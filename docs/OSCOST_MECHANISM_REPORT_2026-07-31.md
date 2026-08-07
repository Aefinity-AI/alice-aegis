# Why Minimal Linux Beat the Bare-Metal Unikernel — Mechanism Report

**Date:** 2026-07-31
**Question answered:** the 2026-07-31 two-stick test (Dell Inspiron 15, i5-5200U
Broadwell-U, M7 model, 6 boots/arm) measured minimal Linux **faster** than the
no-OS unikernel at decode: pooled Δ_OS median **−39.3%**, CI [−45.4%, −10.0%],
Band 3. What, exactly, does the OS provide that bare metal lacks — down to the
lines of code?

**Provenance:** every figure in this report traces to
`docs/hardware_logs/oscost_U_BOOTLOG_2026-07-31.txt`,
`docs/hardware_logs/oscost_L_BOOTLOG_2026-07-31.txt`, and the banked derivation
`docs/hardware_logs/oscost_ANALYSIS_2026-07-31.md`. Nothing here is a new
measurement. Result is Band 3 under the locked preregistration
(`docs/hardware_logs/oscost_PREREGISTRATION_2026-07-30.md`) — **not publishable
until the mechanism experiment below is run.**

---

## The headline, stated precisely

It is **not** the scheduler, not the page cache, not multi-core compute, and
not anything mystical about "an OS." The engine is the same single-threaded
`aegis-core` binary on both arms; Linux never gave it a second core to compute
on. The gap decomposes into two concrete, code-level mechanisms:

| # | Mechanism | Status | Est. share |
|---|---|---|---|
| H1 | Synchronous firmware console writes **inside the timed decode loop** | prime suspect, untested | large (gradient evidence) |
| H2 | Idle-core management: Linux parks the idle core in C6, unlocking the 1-core turbo bin | **measured** | ~8 percentage points of ceiling |
| H3 | UEFI memory caching attributes (MTRR/PAT) | untested, only if H1+H2 don't close | unknown |

---

## H1 — the unikernel pays for its own console, per token, inside the stopwatch

### The exact lines

The unikernel's timed decode path (`aegis-uefi/src/main.rs`):

- `main.rs:1521` — `let t0 = unsafe { core::arch::x86_64::_rdtsc() };` opens the timing window.
- `main.rs:1525` — `engine.process_intent(cmd, max_new_tokens, |token_str| { ... })` runs decode.
- `main.rs:1529–1532` — **inside the per-token callback, inside the timed window:**
  ```rust
  let _ = uefi::system::with_stdout(|st| {
      let _ = st.write_str(token_str);
      core::fmt::Result::Ok(())
  });
  ```
- `main.rs:1536` — `let dt = ... _rdtsc() - t0;` closes the window.

`with_stdout` is the UEFI `EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL`. On real firmware
that call is **synchronous**: it returns only after the firmware's ConSplitter
has rendered glyphs through the GOP framebuffer, and once the screen is full,
**every line of output forces a full-screen scroll redraw** executed by
firmware code that was never meant to be on anyone's fast path. The decode
loop cannot retire token *n+1* until the firmware finishes painting token *n*.

The Linux arm runs the same engine (`aegis-linux/src/main.rs`):

- `main.rs:93` — `let t_start = std::time::Instant::now();`
- `main.rs:101–102` — per-token: `print!("{}", token); std::io::stdout().flush();`

but its stdout is **a pipe, not a screen**: the boot script
(`scripts/linux-arm/init.sh:63`) runs the engine as `"$@" 2>&1 | tee -a "$LOG"`.
So each `flush()` is a `write(2)` into a kernel pipe buffer — a bounded memcpy,
microseconds — and `tee` drains it asynchronously on the *other* core. The
Linux arm never waits for rendering inside its timed region.

**This is not "Linux is faster." This is "the two arms bought different things
inside the stopwatch."**

### Why H1 is the prime suspect (banked evidence)

1. **Cost tracks emitted text volume.** Per-prompt Δ_OS: −9.6% on the short
   85-token response vs −39.3% and −49.3% on the long ones
   (`oscost_ANALYSIS_2026-07-31.md`). A KV-cache or bandwidth mechanism scales
   with context; a console mechanism scales with characters printed.
2. **The unikernel's own CTX_20/100/400 probes are flat** (GAUNTLET seg5,
   `oscost_U_BOOTLOG_2026-07-31.txt`) — those probes count tokens but do
   **not** print them (`main.rs:986–988`, callback counts only). Same engine,
   no per-token console, no length-dependent slowdown. That is H1's signature.

## H2 — nobody in firmware-land manages the idle core, so turbo is capped (measured)

### The exact lines and MSRs

The unikernel *does* ask for maximum performance:
`aegis-uefi/src/cpu.rs:369–439` (`request_max_performance`). On Broadwell
(no HWP) it takes the legacy Enhanced SpeedStep path and writes the requested
P-state ratio into `IA32_PERF_CTL[15:8]` (`cpu.rs:431–433`).

The ask was granted only partially, **every boot**:

- Requested: ratio 27 (2.7 GHz — the i5-5200U's **1-core** turbo bin,
  `MSR_TURBO_RATIO_LIMIT` (0x1AD) byte 0).
- Granted: ratio 25 (2.5 GHz — the **2-core** bin, byte 1). U logs read
  `clock 113%` of the 2.2 GHz nominal on all six boots
  (`oscost_U_BOOTLOG_2026-07-31.txt`); 25/22 ≈ 113.6%.

Turbo bin arithmetic counts every core in **C0/C1** as active; a core retires
from the count only in C3 or deeper. UEFI firmware parks application
processors in HLT (C1) or a spin loop — so the idle sibling core *counts*, and
the busy core is capped at the 2-core bin forever. `IA32_PERF_CTL` is a
request to an arbiter; the parked-but-awake core outranks it.

Linux ships a driver whose whole job is this: **`intel_idle`** (cpuidle
subsystem). Its Broadwell table enters C6 via `MWAIT` with hint `0x20`. With
the idle core in C6, the turbo arbiter grants the 1-core bin — and the L arm
hit the full 2.7 GHz in 3 of 6 boots (`oscost_L_BOOTLOG_2026-07-31.txt`).

Ceiling difference: 27/25 → ~8%. Real, measured, and **much smaller than
−39.3%** — which is why H1 carries the burden of proof.

### The fix is ~85 lines, no OS required

Now implemented in the MECH binary:

- `aegis-uefi/src/cpu.rs:475–501` — `ap_park_mwait_c6`: the AP procedure.
  `CLI`, then `MONITOR`/`MWAIT(eax=0x20)` in a loop — the exact mechanism of
  Linux's `intel_idle` Broadwell entry, in a dozen instructions.
- `aegis-uefi/src/main.rs:94–178` — `park_aps_for_turbo`: dispatches it to
  every AP via UEFI's own `EFI_MP_SERVICES.StartupAllAPs` (non-blocking).

## On "access to all cores"

UEFI already exposes every core: `EFI_MP_SERVICES` (used at `main.rs:94–178`)
can start code on all APs — there is no OS gatekeeping core access. Linux's
SMP advantage **for this workload** was never about computing on more cores
(the engine is single-threaded on both arms); it was about *managing the idle
ones* so the busy one clocks higher. Multi-core GEMM is a separate, real
roadmap item — and it is equally available to the unikernel
(`StartupAllAPs`) and Linux (`pthreads`).

## What parts of Linux were actually load-bearing (Aefinity OS design input)

| Linux subsystem | What it bought here | Unikernel-side equivalent | Cost |
|---|---|---|---|
| pipe/tty write path | O(bytes) buffered `write(2)`, rendering off the timed path | buffer tokens, print after timing (MECH QUIET mode); or serial console; or double-buffered GOP blit | ~20 lines |
| `cpuidle` / `intel_idle` | idle core to C6 → 1-core turbo bin (2.7 vs 2.5 GHz) | `ap_park_mwait_c6` + MP dispatch (done) | ~85 lines |
| `cpufreq` governor | P-state raised at boot | `request_max_performance()` (already had it) | done |
| scheduler | nothing — one runnable thread | n/a | — |
| page cache / VFS | nothing — weights preloaded into RAM once | FAT32 loader (already had it) | — |
| virtual memory | nothing measured | n/a | — |

The honest summary for the Aefinity OS thesis: **an OS is a bundle of
services; this experiment caught the unikernel missing two small ones, not
losing to the bundle.** Whether the remaining gap after H1+H2 is zero is
exactly what the MECH experiment measures — if it is, "AI on bare metal" is
fully rehabilitated at equal or better throughput, with the boot-time,
attack-surface, and determinism advantages intact.

## The experiment that settles it (built, pending hardware run)

MECH binary (`aegis-uefi` @ MECH block, `main.rs:767–854`), one hands-off boot
on the Dell:

1. **LOUD ×3** — per-token console inside the timed region (replicates protocol) →
2. **QUIET ×3** — tokens buffered, printed after the window → H1 share = (LOUD−QUIET)/LOUD.
3. **AP park** → **QUIET2 ×3** → H2 share via `clock %` delta (113% → ~123% expected if the bin unlocks).

Greedy decode makes every pass token-identical, so the run doubles as its own
bit-exactness gate (Rule D). Publishable Δ_OS still requires the properly
interleaved 10-boot redo afterward (preregistration §8 discipline).

---

*Report only. No new numbers. Ledger rows remain gated on
`scripts/verify-figures.sh` and raw logs per Rule B.*
