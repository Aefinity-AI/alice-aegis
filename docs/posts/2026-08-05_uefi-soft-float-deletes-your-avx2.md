# `#[target_feature(enable = "avx2")]` does nothing on `x86_64-unknown-uefi`

*A compile-time flag silently deleted every vector instruction from our binary.
Nothing failed. Nothing warned. The only thing that caught it was counting
instructions in the shipped artifact.*

**TL;DR.** Rust's stock `x86_64-unknown-uefi` target sets `+soft-float`. That
feature lowers every `f32` operation to a soft-float library call — **including
inside function bodies annotated `#[target_feature(enable = "avx2")]`**. The
attribute does not protect you. Our hand-written AVX2 kernels compiled to zero
vector instructions, and the resulting binary ran 217–258× slower than it
should have. It booted fine and produced correct output the entire time.

---

## The symptom

We were chasing an anomaly we had labelled MECH-A1: one function,
`process_intent`, was running two orders of magnitude slower on bare metal than
its own component benchmarks predicted. We assumed a bug in our code — a
pathological allocation, a cache disaster, a mis-set MTRR.

It was none of those. We eventually stopped theorising and just counted the
instructions in the binary we had actually shipped to the USB stick.

## The census

```
== Binary 1 (TRUE staged MECH stick binary): preserved copy
size=453120 bytes
census: xmm=0 ymm=0 vfmadd=0
target features (rustc --print target-spec-json, x86_64-unknown-uefi):
    "-mmx,-sse,+soft-float"
```

Zero `xmm`. Zero `ymm`. Zero `vfmadd`.

This is a binary whose source contains hand-written AVX2 ternary matvec kernels,
with `#[target_feature(enable = "avx2")]` on the hot functions, `_mm256_*`
intrinsics throughout, built `--release`. It contains **not one vector
instruction**.

## Why the attribute doesn't save you

This is the part worth internalising, because it is genuinely counter-intuitive.

`#[target_feature(enable = "avx2")]` *adds* a capability to a function. It does
not *remove* `+soft-float` from the target. And `+soft-float` is not a
"prefer not to use SSE" hint — it instructs LLVM that floating-point registers
are unavailable, so every scalar `f32` operation must be lowered to a call into
`compiler_builtins`.

The result is that your intrinsics get scalarised and then routed through
soft-float helpers. You asked for a 256-bit fused multiply-add; you got eight
function calls into a software float library, each preserving IEEE
single-rounding semantics — which is why the output stays *correct*. It is only
catastrophically slow.

The stock UEFI target sets this deliberately and for a good reason: UEFI
firmware does not guarantee that the SSE state has been initialised, and
touching `xmm` before `CR4.OSFXSR`/`XCR0` are configured will fault. The target
is being conservative on your behalf. That is defensible. What is dangerous is
that the consequence is completely silent — no warning, no link error, no
runtime failure. The build succeeds and the program is correct.

## The fix

We build against a custom target spec instead — `x86_64-uefi-hardfloat.json`,
which is the stock target with the float features removed (`"-mmx"` only) —
and enable the vector state ourselves during firmware bring-up before any
kernel runs.

```
== Binary 2 (hardfloat): aegis-uefi/build_hardfloat.sh output
path: target/x86_64-uefi-hardfloat/release/aegis-uefi.efi (features "-mmx")
size=402432 bytes
census: xmm=6855 ymm=433 vfmadd=210
```

Note the binary got **smaller** — 453,120 → 402,432 bytes. All those soft-float
library calls and their helper routines were larger than the vector code that
replaced them.

If you do this, you must enable the state yourself before executing any vector
code — set `CR4.OSFXSR` (bit 9), `CR4.OSXMMEXCPT` (bit 10) and `CR4.OSXSAVE`
(bit 18), then `xsetbv` with `ecx=0` to write XCR0 and enable the SSE and AVX
state components. Skipping that is exactly what the stock target was protecting
you from, so if you take the guard rail off, you own the bring-up.

## What it cost

Measured on the same physical machine, same workload:

| build | ticks/token |
|---|---|
| soft-float (stock target) | 1.65–1.67 × 10⁹ |
| hardfloat (custom target) | 6.48–7.59 × 10⁶ |

**217–258× recovery.** The derivation — `1.65e9/7.59e6` through `1.67e9/6.48e6`
— is written down in `docs/MECH11_ANALYSIS_2026-08-01.md:17`. We had predicted
a band of 6.6–14M ticks/token before the rebuild; measured 6.48–7.59M, so the
prediction held.

The important consequence for us was not the speed. It was that MECH-A1 — an
anomaly we had spent real effort attributing to the engine's design — was never
an engine property at all. It was a build-provenance failure. We removed it from
the evidence set for the hypothesis it had been supporting.

## The gate we added

A finding you can only detect by inspecting the artifact needs a mechanical
check, or you will ship it again. Ours refuses to stage any `.efi` whose
instruction census comes back with `ymm=0` or `vfmadd=0`:

```bash
scripts/check-efi-simd.sh <path-to.efi>     # objdump census; non-zero exit on a scalar binary
```

The build script itself hard-fails on the same condition, so a soft-float
artifact cannot reach a USB stick even by accident.

**If you take one thing from this post:** for any binary where SIMD is
load-bearing, add an objdump census to CI. Not a benchmark — benchmarks drift
and fail quietly. Count the instructions and assert the count is non-zero. It is
three lines and it would have saved us weeks.

## A correction, since it is relevant

Our first census (`v1`) was itself wrong. A rebuild running concurrently in
another worktree overwrote the artifact between the moment we identified the
binary and the moment we measured it, so the "Binary 1" section of that log
described a fresh stock-target rebuild rather than the binary actually staged to
the sticks. The conclusion happened to be identical — the stock target still
emits zero vector instructions on the current tree — but the provenance was
broken, so we re-ran it against a preserved bit-identical copy and superseded
the log.

The lesson generalises: measure a *preserved copy* of the artifact you actually
shipped, identified by hash, not whatever is sitting at the build path when you
get around to looking.

---

**Evidence.**

- `docs/hardware_logs/mecha1_softfloat_census_2026-08-01_v2.log` — the census
  (supersedes v1; see above)
- `docs/hardware_logs/mech_U_BOOTLOG_2026-08-01.txt` — timing attribution
- `docs/MECH11_ANALYSIS_2026-08-01.md:17` — the recovery-factor derivation
- Staged soft-float binary preserved as
  `artifacts/BOOTX64_MECH_softfloat_2026-08-01_431ff3a8.EFI` (md5 `431ff3a8`)
- Hardfloat binary md5 `7ed77474`

Ledger row **A14**.

*Machine: Dell Inspiron 15, Intel i5-5200U (Broadwell-U, AVX2), bare metal, no
operating system.*

---

**Authorship.** The engineering, the debugging, the instruction census and every
measurement in this post are mine (Justin B. Thompson, Aefinity AI). The prose
was drafted by Claude (Anthropic) working from the primary logs cited above, at
my direction, and published under my name. Disclosed because a post about not
fooling yourself should not start by fooling you.
