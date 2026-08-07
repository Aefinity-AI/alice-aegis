# Pre-registered prediction: first real-hardware boot of the fixed engine

**Written and committed BEFORE the test was run.** Timestamp is the commit date.
The point of writing this first is that a prediction recorded in advance cannot be
quietly adjusted after the result is known. This project has a documented history of
the opposite (see `PRE_REVIEW_SCRUB_LIST.md`).

## Apparatus

- **Machine:** Dell Inspiron 15, Intel Core i5 (exact model TBD — see "Open variables")
- **Image:** `aegis-boot.img`, 3.0 GB FAT superfloppy, containing
  `EFI/BOOT/BOOTX64.EFI` (interactive REPL build, hard-float, AVX2 codegen verified:
  161 FMA instructions, 433 ymm references), `MODEL.SAF` (522 MB),
  `EMBED.BIN` (257 MB), `VOCAB.BIN` (1.76 MB)
- **Engine state:** all twelve defects of 2026-07-09 fixed; identical binary lineage to
  the one that produces coherent output in QEMU and in Linux userspace.

## History being tested against

Earlier builds of this engine **did boot on this same Dell** and ran, producing
gibberish. That is now explained: those binaries were built with the stock
`x86_64-unknown-uefi` target, which is **soft-float**, so they contained zero AVX
instructions. The separate bug that gated AVX-enable on the wrong CPUID bit was
therefore harmless — there was nothing to fault on. The gibberish came from the
tokenizer (no BPE merges, no special tokens, non-byte-level encode) and the LM-head
size guard, not from the hardware.

**All of those are fixed. The hardware was never the problem.**

## PREDICTIONS

### P1 — It boots. (confidence: high)
The Inspiron 15 line is UEFI-native. The image booted on this machine before.
Failure here would most likely be Secure Boot rejecting an unsigned EFI binary.

### P2 — It generates coherent English. (confidence: moderate-high)
Specifically: prompted "What is the capital of France?", it emits a response containing
the word **Paris**, then stops on `<|eot_id|>` rather than running to the token cap.

This is the substantive prediction. It has never been true on this machine.

### P3 — Speed depends on one bit of CPU capability.
- **If the i5 is 4th-gen (Haswell, i5-4xxx) or newer:** it has AVX2+FMA. The banner will
  read `SIMD level: AVX2+FMA`. Expect roughly **2–4 tokens/sec** (the unikernel is
  single-threaded; measured 599M cycles/token bare-metal on an i5-10210U).
- **If the i5 is 2nd or 3rd gen (Sandy/Ivy Bridge, i5-2xxx/3xxx):** it has AVX but not
  AVX2/FMA. Runtime dispatch will fall back to scalar. The banner will read
  `SIMD level: SSE2 (scalar fallback)`. Expect roughly **0.3–0.5 tokens/sec**
  (measured 4.01B cycles/token on emulated Nehalem). **Slow is not broken.**

### P4 — Model load takes minutes, not seconds.
780 MB through firmware file I/O over USB. Progress dots will appear. The watchdog is
disabled and petted every ~10 MB, so the machine should not reset.

### P5 — `BOOTLOG.TXT` will be written back to the stick, and will name the last stage
reached, whatever happens. Stages 1 through 7.

## What would falsify each

| Prediction | Falsified by |
|---|---|
| P1 | Firmware refuses the image, or hangs before the A.L.I.C.E. banner |
| P2 | Output is gibberish, empty, repeated tokens, or never terminates |
| P3 | Banner reports a SIMD level inconsistent with the CPU's actual capability |
| P4 | Watchdog reset, or a load failure at STAGE 4 |
| P5 | No `BOOTLOG.TXT` on the stick after a failed boot |

## Open variables (unknown at time of writing)

1. **Exact i5 model** — determines AVX2 vs scalar, and therefore P3.
2. **Installed RAM** — the engine needs ~1.6 GB. Only validated at 2 GB in QEMU.
   Below that, expect a clean failure at STAGE 3 with an explicit message.
3. **Secure Boot state** — must be OFF. The EFI binary is unsigned.
4. Whether this machine's firmware accepts a superfloppy (no partition table) USB image.
   It did before.

## Result

**Run: 2026-07-09, ~20:01. Dell Inspiron 15, Core i5.**

Verbatim `BOOTLOG.TXT` recovered from the stick:

```
==== A.L.I.C.E. BOOT ====
STAGE 1: boot volume opened, AVX enable attempted
STAGE 2: sizes OK model=522831576 embed=257310720 vocab=1759936
STAGE 3: tensor memory allocated
STAGE 4a: MODEL.SAF loaded
STAGE 4b: EMBED.BIN loaded
STAGE 4c: VOCAB.BIN loaded
STAGE 5: working heap online
STAGE 6: engine online, SIMD=AVX2+FMA
```

| Prediction | Verdict | Evidence |
|---|---|---|
| **P1** boots | **CONFIRMED** | STAGE 1 reached; firmware accepted the unsigned superfloppy image |
| **P3** SIMD dispatch correct | **CONFIRMED** | `SIMD=AVX2+FMA` — the i5 is 4th-gen Haswell or newer, and runtime detection chose the vector path |
| **P4** 780 MB loads through firmware | **CONFIRMED** | STAGE 4a/4b/4c; no watchdog reset |
| **P5** `BOOTLOG.TXT` written back | **CONFIRMED** | 330 bytes recovered from the stick |
| **P2** emits "Paris", stops on `<\|eot_id\|>` | **NOT ESTABLISHED** | see below |

### P2 is not confirmed, and this is the important part

The operator reports the engine worked. **The stick contains no record of what the
model said.** `STAGE 7` — the line that logs generated tokens — lives inside
`#[cfg(feature = "qemu-test")]`, so the interactive REPL build never writes it.

The machine did the thing, and nothing recorded what it said. That is precisely the
failure mode this project spent fifteen months inside: an unwitnessed success is
indistinguishable, in the record, from a fabricated one. A human eyewitness is not
evidence we can cite.

**P2 therefore remains open.** It is not marked confirmed on operator testimony.

### Corrective action, taken immediately

The interactive build now writes `PROMPT:`, `RESPONSE:`, and cycles-per-token to
`BOOTLOG.TXT` on the boot volume after every generation, and `/benchmark` persists its
measurement there too. Real hardware has no serial console to scrape; a generation that
is not recorded cannot be cited. Binary grew 183,808 → 187,392 bytes to buy this.

**The test will be re-run with the self-recording build, and this section updated with
the verbatim transcript — whatever it says.**

### What is nonetheless established

The engine booted on a physical machine, with no operating system, loaded 780 MB of
weights through firmware file I/O, correctly detected AVX2+FMA on silicon it had never
seen, and brought the inference engine online. That is the first logged real-hardware
boot in the project's history, and it is now on the stick rather than in a memory.

---

# FINAL RESULT — run 3, 2026-07-09 16:51

Verbatim from `docs/hardware_logs/dell_run_2026-07-09_165138.txt`:

```
==== A.L.I.C.E. BOOT ====
STAGE 1: boot volume opened, AVX enable attempted
STAGE 2: sizes OK model=522831576 embed=257310720 vocab=1759936
STAGE 3: tensor memory allocated
STAGE 4a: MODEL.SAF loaded
STAGE 4b: EMBED.BIN loaded
STAGE 4c: VOCAB.BIN loaded
STAGE 5: working heap online
STAGE 6: engine online, SIMD=AVX2+FMA
PROMPT: "the capital of france is..."
RESPONSE: the capital of france is paris
  (6 tokens, 54142854551 cycles, 9023809091 cycles/token)
BENCHMARK: 53 tokens, 225810661665 cycles, 4260578521 cycles/token
```

## Verdicts

| Prediction | Verdict |
|---|---|
| **P1** boots | **CONFIRMED** |
| **P2** emits "Paris", terminates | **CONFIRMED** — `the capital of france is paris`, 6 tokens, stopped on its own rather than running to the cap |
| **P3a** SIMD dispatch correct | **CONFIRMED** — `SIMD=AVX2+FMA` |
| **P3b** speed ≈ 2–4 tok/s | **FALSIFIED** — see below |
| **P4** load completes, no watchdog reset | **CONFIRMED** |
| **P5** `BOOTLOG.TXT` written | **CONFIRMED** |

**A 2-billion-parameter language model answered a question on a Dell Inspiron 15 with
no operating system, and wrote its own transcript back to the USB stick it booted from.**

The model mirrored the operator's lowercase, uncapitalised prompt style. It is not a
verbatim string match for "Paris"; it is the correct answer, generated.

## P3b is falsified, and the failure is more interesting than the prediction

`rdtsc` on modern Intel is **invariant**: it ticks at the CPU's *nominal* frequency
regardless of the core's actual clock. These "cycles" are therefore a proxy for wall
time, not for work done.

Subtracting the two runs to cancel most of the differing prefill cost:

```
(225,810,661,665 − 54,142,854,551) / (53 − 6)  =  3.65 B ticks/token   (upper bound on decode)
```

Against the Chromebook's measured 0.44–0.60 B ticks/token, single-threaded: **the Dell is
roughly 7× slower per token.** At any plausible nominal frequency (1.6–2.7 GHz) that is
**0.44–0.74 tok/s**, against a predicted 2–4.

Haswell-versus-Comet-Lake IPC accounts for perhaps 1.2–1.3× of that. It does not account
for 7×.

### The hypothesis this raises, which directly contradicts §5.6 of the technical report

**There is no operating system to manage P-states.**

Under Linux — and under QEMU/KVM, where the *host's* Linux governs frequency — the CPU
scales up under load. On bare metal, nothing does. The firmware leaves the core at its
base or minimum P-state, and the engine runs there for the entire session.

`docs/TECHNICAL_REPORT.md` §5.6 currently claims *"bare metal buys ~0% speed over
Linux"*, on the strength of a bare-metal-in-QEMU measurement landing within 2% of Linux
userspace. **That measurement was made on a machine whose clock was being managed by a
Linux host.** It may say nothing whatsoever about real bare metal.

If this hypothesis holds, the honest statement inverts: **running without an operating
system is substantially *slower*, because the operating system was doing something
valuable that nobody noticed — asking the CPU to go fast.**

That would be a genuinely novel result, and it is the sort of thing only this apparatus
can measure.

### What would settle it (not yet run)

1. **Wall-clock timing inside the unikernel.** UEFI Runtime Services expose `GetTime()`.
   Read it before and after generation to get tokens/second directly, independent of any
   TSC assumption.
2. **Read the CPU's actual frequency.** `IA32_MPERF`/`IA32_APERF` MSRs give the ratio of
   actual to nominal clock. Reading them in the UEFI app would measure the P-state
   directly rather than inferring it.
3. **A Linux userspace baseline on the same Dell**, same engine, same prompt. Apples to
   apples, one machine, OS as the only variable.
4. **The Dell's exact CPU model**, to fix its nominal TSC frequency.

Until then P3b stands falsified, and §5.6 stands **in doubt on its own evidence.**

---

**Note on why this file exists.** For fifteen months this project recorded results that
were never measured: a hardcoded `println!("...Paris")`, a simulated "103 tok/s", a
13-byte file named `vmlinuz` containing the string `DUMMY_KERNEL` inside an ISO labeled
bootable. The cure is not to try harder to be honest afterward. The cure is to write the
prediction down first, where it can be wrong in public.
