# Field protocol — running `/autotest` on the test fleet

Read this before the first machine. It exists because the code writes to
model-specific registers, and a machine killed by a #GP is a data point lost.

The laptops are donated, not borrowed, which lowers the social stakes but not the
engineering ones: a bricked boot is still a missing row in the table, and the
NVMe-equipped machine is reserved for experiments that need it intact.

## What this does to the host machine

| | |
|---|---|
| Writes to the host's disk | **Never.** The engine reads only from the USB stick and writes only to it. |
| Writes to firmware / NVRAM | **No.** UEFI may create its own `NvVars` on the *stick*, not the machine. |
| Writes to CPU registers | **Yes** — `/turbo` sets the P-state via `IA32_HWP_REQUEST` or `IA32_PERF_CTL`. |
| Survives a power cycle | **No.** P-state requests are volatile. Power off and the CPU forgets. |
| Can it damage the CPU | **No.** It requests a frequency the silicon already advertises as valid. It cannot exceed the vendor's own turbo ratio, and thermal/power limits remain enforced by hardware. |

The one setting you change on the *host* is **Secure Boot**, which must be off to load
an unsigned EFI binary. **Turn it back on before you return the laptop.**

## Before you touch someone's machine

1. **If a machine is not yours, ask.** It boots from your USB, changes nothing on the
   disk, and you will restore the Secure Boot setting.
2. **Charge or plug in.** A full `/autotest` takes ~10–20 minutes depending on the
   machine's clock, and a mid-run power loss wastes the trip.
3. **Note the make, model, and rough year.** The CPU brand string identifies the chip;
   only you can record that it was a 2013 ThinkPad rather than a 2021 XPS. Firmware
   P-state behavior is the variable under study, and it tracks the *laptop*, not the CPU.

## The run

```
F2 (or F12/Del) at the vendor logo
  → Secure Boot: OFF          ← the only host setting you change
  → Boot mode: UEFI (not Legacy/CSM)
  → USB boot: enabled
F12 → pick the USB device

wait for:  A.L.I.C.E.>          (780 MB loads through firmware; minutes, with dots)

type:      /autotest            ← then leave it alone, ~10-20 min, six phases
type:      /exit
```

Then **power off** and take the stick. Restore Secure Boot on any machine you are
returning.

## Collecting

```
./collect_autotest.sh /dev/sda  <nickname>
```

Nickname it something you will recognise: `thinkpad-t440p-2013`,
`xps13-2021`, `dell-inspiron15`. One row lands in
`docs/hardware_logs/pstate_dataset.tsv`; the raw log is filed verbatim beside it.

**Collect before you re-flash. Ever.** The stick is the only copy.

## What can go wrong, and what it means

| Symptom | Meaning |
|---|---|
| Firmware refuses the stick | Secure Boot still on, or Legacy/CSM boot mode |
| Hangs before the A.L.I.C.E. banner | Firmware/USB issue. Check `BOOTLOG.TXT` — if it is absent, it never reached STAGE 1 |
| `FATAL: Could not allocate…` at STAGE 3 | Under ~2 GB usable RAM. Not a bug. Record it and move on |
| `SIMD level: SSE2 (scalar fallback)` | Pre-2013 CPU, no AVX2. **This is a pass.** It will be slow — possibly 20+ s/token |
| `Cannot raise P-state: …` | The CPU or firmware refused. The reason is logged. **Still a data point** — record it |
| `AUTOTEST DONE` never appears | The run was interrupted. `RUN1`/`RUN2` may still be in the log and are still usable |

**A machine where `/turbo` fails is not a failed experiment.** It is a row in the table
with `effect = n/a` and a stated reason, and it constrains the hypothesis just as much as
a machine where it succeeds.

## Safety of the MSR writes, stated precisely

Every model-specific register is read or written **only after CPUID says it exists**:

```
IA32_APERF / IA32_MPERF (0xE7/0xE8)  ← CPUID.06H:ECX[0]   and not under a hypervisor
IA32_PERF_STATUS / PERF_CTL (0x198/9) ← CPUID.01H:ECX[7]  (EIST)
MSR_PLATFORM_INFO (0xCE)              ← Nehalem+ (proxied by APERF/MPERF)
MSR_TURBO_RATIO_LIMIT (0x1AD)         ← CPUID.06H:EAX[1]  (Turbo Boost)
IA32_PM_ENABLE / HWP_* (0x770-0x774)  ← CPUID.06H:EAX[7]  (HWP)
```

Reading a nonexistent MSR raises `#GP`. A UEFI application installs no exception
handler, so `#GP` is a dead machine. Two such faults were found by audit **before** any
borrowed hardware was touched: `MSR_TURBO_RATIO_LIMIT` on parts without Turbo Boost, and
`MSR_PLATFORM_INFO` on pre-Nehalem parts. Both are now gated.

If a machine still dies at `/turbo`, **stop the study, power-cycle the laptop (it will be
fine), and tell me the CPU brand string.** That is a bug in the gating, and it is worth
more than the remaining data points.

## The NVMe machine is special. Do not spend it on `/autotest`.

One donated laptop reportedly has an NVMe drive, unlocked and clean. That machine is the
apparatus for the two experiments this project most needs, and both require installing an
operating system on it:

1. **The definitive OS-vs-no-OS comparison.** Same engine, same binary lineage, same
   silicon: run `aegis-linux` under Linux, then boot the unikernel, and compare
   tokens/second *and* the APERF/MPERF clock in both. Nobody has this. It settles §5.6
   outright rather than by inference.
2. **The no-OS energy delta.** Measure joules/token on battery under Linux, then bare
   metal, on one machine. If removing the operating system changes energy per token —
   in either direction — that is a publishable result and only this apparatus can produce
   it.

Run `/autotest` on it once for the dataset, then keep it clean and install Linux.
