# Your benchmark laptop may be lying to you

*How we found out our test machine had been running at one-fifth speed the
whole time — and why this post is a retraction, not a speedup.*

**TL;DR.** A Dell Inspiron 15 (i5-5200U, Broadwell-U) we were using as a
bare-metal benchmark target was clamped to core ratio 5 out of a requested 27.
It was not thermal throttling. The machine's embedded controller was asserting
`PROCHOT` externally, permanently, from cold boot, on a cool machine. Every
performance number we took on that box before 2026-07-29 was measured on a CPU
running at roughly a fifth of its nominal clock. Clearing one MSR bit fixed it.

The fix is not novel — it is about fifteen years old. The *measurement error*
is the point of this post.

---

## The symptom

Our engine prints a diagnostic line at boot before it does anything else. On
this machine it said:

```
TURBO_DIAG boot-pre: cur_ratio=5 req_ratio=27 eist=1 turbo_dis=0 clkmod=0x00 \
                     prochot_now=1 prochot_log=1 hot=0 temp=Tj-63C bdprochot_en=1
```

Read that line carefully, because the whole diagnosis is in it:

- `cur_ratio=5` against `req_ratio=27`. We asked for 2.7 GHz. We were getting
  roughly 500 MHz.
- `eist=1`, `turbo_dis=0`, `clkmod=0x00`. SpeedStep is on, turbo is not
  disabled, and clock modulation (duty-cycle throttling) is off. None of the
  usual suspects.
- `prochot_now=1` — the processor is being told it is hot **right now**.
- `hot=0` and `temp=Tj-63C` — but the thermal sensor says it is 63 degrees
  *below* its junction limit.

`PROCHOT` asserted on a cold CPU. That contradiction is the finding.

## What is actually happening

`PROCHOT#` is a bidirectional pin. The CPU drives it *out* when it is genuinely
overheating. But the board can also drive it *in* — that is the "bd" in
bd-PROCHOT — to force the CPU to its minimum ratio for reasons that have nothing
to do with the die temperature: a failing battery, a charger the EC does not
recognise, a VRM the firmware distrusts, or an EC that simply latched the
condition once and never cleared it.

When the board asserts it, the CPU obeys unconditionally. No thermal event, no
error, no log entry anywhere in the OS. The machine simply runs at its lowest
ratio forever and reports itself perfectly healthy.

This is common in second-hand and cheaply-acquired hardware, which is exactly
the kind of machine you use for a "does it run on weak silicon?" project.

## The fix

Bit 0 of `MSR_POWER_CTL` (`0x1FC`) enables the board's ability to assert
PROCHOT. Clear it, then re-request the P-state:

```
STAGE 7 bd-prochot: CLEARED (read-back confirms 0); prochot_was_asserted=1
STAGE 7 pstate:     legacy SpeedStep ratio=27 (~2700MHz)
TURBO_DIAG boot-post: cur_ratio=25 req_ratio=27 eist=1 turbo_dis=0 clkmod=0x00 \
                      prochot_now=0 prochot_log=1 hot=0 temp=Tj-56C bdprochot_en=0
```

`cur_ratio` 5 → 25. Reproduced on 3/3 cold boots.

Two details worth copying if you implement this yourself. **Read the MSR back**
after writing it; some ECs re-assert. And note that `prochot_log=1` stays set —
that is the sticky log bit recording that it *was* asserted, which is useful
evidence and is not an error.

## What it cost us

Here is the honest part.

We had a clean A/B available: the same 14.17M-parameter model, two builds made
the same day differing only by the STAGE 7 fix, generating the same 214 tokens.

| | ticks/token | logged clock |
|---|---|---|
| Before | 53,250,965 | 22% of nominal |
| After | 13,791,915 | 113% of nominal |

**3.861× wall-time improvement**, and the 214-token output was byte-identical at
both clock speeds — which is the check that proves we changed the clock and not
the computation.

Note that 3.861× is *less* than the 5.000× the ratio change implies. That
difference is real and expected: ratio 5 → 25 is exactly 5× by construction, but
only the compute-bound part of the work scales with core clock. Memory-bound
work does not. The gap between 5.000× and 3.861× is a crude measure of how much
of this workload is waiting on memory.

We are deliberately **not** reporting this as a speedup. We did not make
anything faster. We stopped measuring a crippled machine. Every figure taken on
that box before the fix was wrong, and we retracted them.

## Prior art — this is not our discovery

`MSR 0x1FC` bit 0 has been common knowledge for roughly fifteen years.
[ThrottleStop](https://www.techpowerup.com/download/techpowerup-throttlestop/)
has exposed it on Windows for most of that time, and
[arter97/DisablePROCHOT](https://github.com/arter97/DisablePROCHOT) does exactly
this from a UEFI application with no OS present. If you search for your
symptoms you will find them. We did not, for weeks, because we were not looking
for a platform fault — we were looking for a bug in our own kernels.

That is the transferable lesson, and it is the only original thing here.

## What we have not established

- **No clear-only arm.** We changed two things in STAGE 7 — cleared bd-PROCHOT
  *and* requested the maximum P-state. We never ran clear-without-boost, so
  "you must clear before boosting" remains an untested hypothesis on our part.
  Prior art ([erpalma/throttled #163](https://github.com/erpalma/throttled/issues/163))
  suggests the ordering does not matter, which would mean our framing is wrong.
- **Durability is untested.** We do not know whether the clear survives S3
  suspend, or whether the EC re-asserts after a firmware event.
- **One machine.** n=1 laptop, 3 cold boots.

## Check your own machine

Before you trust any benchmark from a laptop — especially a used one:

1. Read `cur_ratio` and compare it to what you requested.
2. If they differ, check whether `PROCHOT` is asserted **and** whether the
   thermal sensor agrees. `prochot_now=1` with `hot=0` is the signature.
3. On Linux, `rdmsr 0x1fc` will show you bit 0.

If your machine has been quietly running at minimum ratio, every number you have
published from it is wrong by a factor you have not measured. We would rather
you learn that from this post than from a reviewer.

---

**Evidence.** All figures above come from these logs, which are append-only in
our repository:

- `docs/hardware_logs/m7_BAREMETAL_bdprochot_FIXED_2026-07-29.log` — the
  diagnostic and the fix
- `docs/hardware_logs/m7_BAREMETAL_BOOT_2026-07-29.log:70` — pre-fix
  53,250,965 ticks/token
- `docs/hardware_logs/m7_baremetal_prompts_postfix_2026-07-29.log:102` —
  post-fix 13,791,915 ticks/token
- Fix commit `f337137`

The `tok/s` column in those logs is computed against a wall clock with
one-second resolution and is too coarse to quote; the tick counts are the
instrument. Ledger row **A12**.

*Machine: Dell Inspiron 15, Intel i5-5200U (Broadwell-U), bare metal, no
operating system.*

---

**Authorship.** The diagnosis, the fix, and every measurement in this post are
mine (Justin B. Thompson, Aefinity AI). The prose was drafted by Claude
(Anthropic) working from the primary logs cited above, at my direction, and
published under my name. Disclosed because a post about not fooling yourself
should not start by fooling you.
