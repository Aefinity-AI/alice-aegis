# A.L.I.C.E. / Aegis — Full Session Record, 2026-07-09
## Adversarial handoff document

**From:** Justin B. Thompson ("Spark") and Claude (Anthropic), working session of 2026-07-09
**To:** Gemini Deep Think ("Gems"), and any other reasoner asked to check this work
**Purpose:** A complete, self-contained record of what was audited, what was measured,
what was refuted, and where we may still be wrong.

---

### HOW TO READ THIS DOCUMENT

This is not a summary written to persuade you. It is written so you can **attack it.**

Several claims in here contradict work that you (Gemini/"Gems") helped author in earlier
sessions of this project — specifically the "Infinity O/S," "Patentable IP," and
"Technical Audit Report" documents. Some of those documents contain numbers that were
never measured. That is stated below plainly, without blame: they were produced
collaboratively by a human and one or more LLMs, in a mode where nobody was running the
benchmark. This session ran the benchmarks.

**We also got things wrong today.** Section 8 lists the errors Claude made during this
session, several of which were caught only because a measurement contradicted them. Do
not assume this document is correct. Section 9 lists the specific attacks we think are
most likely to succeed against it.

Everything below is either (a) a number produced by a command you can re-run, or
(b) explicitly labeled as inference, estimate, or opinion. Where those are mixed, the
mixing is called out.

---

## 1. WHAT THE PROJECT IS

**A.L.I.C.E.** (Aegis Lightweight Inference Core Engine) is a from-scratch `no_std` Rust
implementation of the BitNet b1.58 ternary transformer that performs LLM inference:

- as a **UEFI application booting directly from firmware, with no operating system**, and
- as an ordinary Linux userspace binary, from the same engine source.

The entire inference stack — model loader, ternary kernels, attention, tokenizer,
sampler — is ~1,600 lines of Rust with `libm` and `serde` as the only runtime
dependencies.

### 1.1 Provenance (important, and previously misstated)

| Component | Origin |
|---|---|
| Model weights | **Microsoft BitNet b1.58-2B-4T** (MIT license). Downloaded, not trained. |
| Ternary quantization | **Microsoft's.** The weights arrive already quantized. |
| Inference engine, kernels, unikernel, tooling | Original work in this repository. |

`aegis-forge` performs **vocabulary pruning** (128,256 → 50,256 tokens) and embedding
slicing. It does **not** quantize. Any document in the archive describing it as a
"Ternary Quantization Pipeline" is factually wrong.

### 1.2 Test machine (every number below is from this machine unless stated)

- Intel Core i5-10210U (Comet Lake, 14nm), 4 physical cores / 8 logical
- **AVX2 + FMA + BMI2 + POPCNT. No AVX-512, no VNNI.** (This matters — see §5.2)
- 6.4 GB RAM, running inside a ChromeOS Crostini VM (Debian)
- Measured single-thread memory bandwidth ceiling: **17.3 GB/s**
- Measured idle system power: **6.30 W**

### 1.3 Model architecture (verified against config, not assumed)

30 layers · hidden 2560 · GQA 20 query / 5 KV heads · head_dim 128 ·
intermediate 6912 · squared-ReLU FFN · SubLN · RoPE θ=500,000 · RMSNorm ε=1e-5 ·
tied BF16 embeddings · ternary weights packed 4 per byte (00=0, 01=+1, 10=−1)
with a per-tensor f32 scale.

---

## 2. WHAT THE AUDIT FOUND (before any new work)

Four independent audit agents examined the local repository and a 1 TB archive drive
containing ~15 months of prior work (March 2026 – July 2026). Findings, stated without
euphemism:

### 2.1 Real and non-trivial
- A working `no_std` BitNet forward pass with hand-written AVX2 ternary kernels.
- A UEFI application that enables AVX itself at the firmware level (CR4/XCR0 via
  `xsetbv` — firmware hands off with AVX disabled), scans the raw UEFI memory map to
  allocate contiguous physical pages, and streams weights through a 64 KB-aligned DMA
  bounce buffer to respect XHCI transfer-boundary rules.
- Vocabulary pruning to fit a 6 GB device.

### 2.2 Not real — claims unsupported by any code
- **"Patentable IP"** (V5 manual): three claimed inventions — a 5:8 structured ternary
  block, a "Power-of-2 Neuromorphic Ring Allocator," and a "Neural RPC" — **none are
  implemented anywhere.** The shipped kernel is a conventional 2-bit LUT + FMA matvec.
- **"Fibonacci Survival Rings," "Golden-Ratio 61.8% sparsity," "Kaiming Variance
  Restorer"** — numerology. §5.1 shows the real weights are 42.21% zeros, so the 61.8%
  premise was never a property of this model.
- **Every quantitative figure in the pre-2026-07-09 documents** was fabricated,
  simulated, or copied:
  - The perplexity chart "14.12 → 14.58" came from a hardcoded mock. No perplexity was
    ever computed by that code.
  - "103.5 tok/s, BEATS GPT-4" (May 2026) was a `DualMatVecKernel` **simulation**, as
    that era's own development journal admits.
  - `TTFT: 14ms / TPS: 84.6 / Peak RSS: 412 MB` in `antigravity-aegis/src/main.rs` are
    string literals, not measurements.
  - The README was Microsoft's model card verbatim, including their **0.028 J/token**
    energy figure, presented as if it were this project's result.
  - The Infinity OS dashboard prints hardcoded "0.9 mJ per token" and
    "Multiplications: ZERO"; its ISO staging tree contains a **13-byte file named
    `vmlinuz` whose entire contents are the string `DUMMY_KERNEL`**. It never booted.
- **The Gemma-4 era "success" was faked.** In `FAILED GEMMA4 PROJECT/AEGIS SOURCE.txt`
  (~line 512), the engine generates tokens and then executes
  `println!("Decoded text: The capital of France is Paris");` — a hardcoded literal,
  printed regardless of the (garbage) tokens actually produced.
- Across ~15 months, **no transmuted dense model ever produced one logged coherent
  sentence.** Coherence appeared only after switching to Microsoft's pre-quantized
  BitNet weights.

### 2.3 Note for Gemini specifically
Documents titled `2026-07-04-Infinity-O-S-The-Gems-Spark-Edition-sources.txt`,
`Aegis_Model_Technical_Audit_Report.md.docx`, and the "PATENTABLE IP" files appear to be
LLM-assisted and grade the work against the author's own specifications — a circular
validation. The technical reasoning in them is often *locally* sound (the SparseGPT
error-compensation discussion is correct; the endianness hazard is real). The failure is
that **none of it was ever tested**, and the numbers attached to it were invented.

This is offered as data about a failure mode, not as an accusation. The failure mode is:
*a human and an LLM can generate internally consistent, technically literate,
confidently worded engineering documents about a system that does not work, indefinitely,
as long as neither of them runs the program.*

---

## 3. DEFECTS FIXED TO REACH COHERENCE (all in this session)

The engine did not work when this session began. Twelve defects, in the order they were
found. Items 1–5 were blocking; the rest are correctness or safety.

1. **LM-head guard sized for F32** (`ops.rs`): embeddings had been migrated to BF16, but
   the size guard still required 4 bytes/element, so `f32_dot_argmax` returned early and
   argmax was always token 0.
2. **`vocab.bin` contained no BPE merges.** The vocabulary stripper never wrote the
   merges section, so tokenization degraded to character-by-character.
3. **Llama-3 special tokens were entirely absent.** They live in `tokenizer.json`'s
   `added_tokens` array, not `model.vocab`, which the stripper never read. Without
   `<|eot_id|>` the model could never emit EOS. Regenerated `vocab.bin` + `embed.bin` as
   a matched 50,256-token pair, rows sliced losslessly from Microsoft's BF16 table.
4. **The stock `x86_64-unknown-uefi` Rust target is soft-float.** Every unikernel binary
   ever built by this project contained **zero vector instructions** — all AVX2
   intrinsics were scalarized to software library calls. This is the root cause of every
   historical "hang" on bare metal. Fixed with a custom hard-float target spec +
   `-Zbuild-std`. **(This is a known upstream issue: rust-lang/rust#136540. It was not a
   discovery, though we initially treated it as one. See §8.)**
5. **The AVX-enable block never executed.** It gated on CPUID leaf 1, ECX **bit 27**
   (OSXSAVE = "the OS has already enabled this"), which is always clear at firmware
   handoff, instead of **bit 26** (XSAVE = "the CPU supports this"). The moment real AVX
   code existed, the first instruction raised #UD. The banner had been printing
   "Hardware AVX-2 Active" over disabled hardware for the project's entire life.
6. **`calculate_perplexity` advanced state with `tokens[i]` instead of `tokens[i+1]`.**
   Any perplexity this code could ever have produced was meaningless.
7. **The tokenizer's `encode()` was not byte-level** while `decode()` implemented the
   full GPT-2 byte↔unicode table. Newlines, tabs, and all non-ASCII collapsed to
   `<|reserved_special_token_0|>` — which is *why* the chat template had to hand-inject
   `"ĊĊ"` tokens; the encoder physically could not produce them. BPE merges also ran
   across word boundaries. **Fixing this alone moved perplexity 14.488 → 12.801 with no
   change to the model weights.**
8. **Console-reachable stack overflow** in the UEFI REPL: the buffer guard checked the
   current length, then wrote up to 3 UTF-8 bytes for a `Char16`.
9. **Truncated USB reads were reported as success**, silently loading corrupted weights.
   (QEMU's XHCI model does not reproduce short reads; real hardware does.)
10. **`SafeTensors::tensor()` sliced the buffer with no upper bound** — a malformed
    offset panicked, which on bare metal is an unrecoverable fault, not a `Result`.
11. **`measure_energy.sh` could not have produced a correct number.** Its token count was
    hardcoded to 128 by a no-op command substitution, and it divided decode-only power by
    a wall time that included model loading.
12. **The `static mut` LUT and SIMD flag were UB under threads**, blocking multicore.

---

## 4. MEASURED RESULTS

Every number is from a command in §11. Machine verified idle (no process >25% CPU)
before each measurement. Where a number was later found to be contaminated, both the
contaminated and corrected values are shown.

### 4.1 Correctness

| Claim | Result |
|---|---|
| Coherent generation, Linux userspace | "The capital of France is **Paris**." with correct EOS; fluent multi-sentence answers |
| Coherent generation, **bare-metal (no OS)** | Same output, booted via OVMF UEFI under QEMU-KVM, exits with hardware success signal (isa-debug-exit code 33) |
| Hardware compatibility | **6/6** QEMU matrix: Nehalem (2008-class, no AVX → scalar fallback), SandyBridge, Haswell, host × q35/legacy-pc firmware × SATA and XHCI-USB boot |

**Honesty note on the matrix:** `test_matrix.sh` was subsequently extended with two more
cases (an 8 GB run to exercise >4 GB physical allocation, and a 1400 MB run to force the
heap's chunk-halving path). **Those two were never executed.** A reproducer will therefore
see 8 cases, of which we have validated 6. The script also documents what QEMU *cannot*
reproduce: real-USB short reads, 64 KB DMA boundary enforcement, BOT stalls, and the
fragmented memory map of real firmware. A green matrix is necessary, not sufficient.

### 4.2 Language-model quality

WikiText-2 test split, teacher-forced NLL, 1,842-token contiguous sample, greedy.
**Deterministic**: the f32 baseline reproduced to the digit on a repeat run.

| Configuration | Perplexity |
|---|---|
| Original (char-level tokenizer bug) | 14.488 |
| + byte-level tokenizer, word-bounded merges | **12.801** |
| + per-token int8 activation quantization | **12.738** |

**Caveat, load-bearing:** this is the **vocabulary-pruned** model (50,256 of 128,256
tokens, ASCII-oriented). These numbers are **not comparable** to published full-vocab
BitNet perplexities. The sample is a single 1,842-token window; we have **no error bars**
(see §9).

### 4.3 Throughput (clean machine)

| threads | decode tok/s |
|---|---|
| 1 | 3.64 |
| 2 | 6.00 |
| 4 | 7.91 |
| **8** | **8.25** |

Decode by generation length: 6.53 tok/s @20 tokens, 6.98 @100, 6.41 @400, 5.37 sustained
over a 4-minute run. The final drop is *consistent with* thermal throttling; this VM does
not expose CPU temperature, so that is inference, not measurement.

Prefill, batched GEMM vs per-token matvec (single thread, 351-token prompt, A/B run in
**both orders** to defeat thermal drift): 572M & 596M cycles/token → 352M & 341M.
**1.63–1.75×.**

**Bare metal vs Linux userspace: 599M vs 571M cycles/token — within 2%** *(in QEMU/KVM; a
real-hardware run has since put this in doubt — see §5.6)*.

### 4.4 Energy (battery-discharge method, on battery, machine idle)

| | |
|---|---|
| Idle power | **6.30 W** (independent coulomb-counter cross-check: 6.35 W — agree to 0.8%) |
| Power during decode | **24.04 W** |
| Incremental | **17.74 W** |
| **Energy per token, incremental** | **3.31 J** |
| **Energy per token, total system draw** | **4.48 J** |

Method: 240 s idle baseline, 261 s decode window, 1,401 tokens generated.

**Method caveat, load-bearing:** `current_now` on this platform is **latched** — it holds
a value for 15–30 s and lags load changes. `charge_counter` steps in coarse 40 mAh
increments. Short windows resolve neither. An earlier attempt with 60 s windows produced
two estimates disagreeing by 40% and was **discarded**. The 0.8% agreement at idle in the
final run is what makes the number reportable.

### 4.5 DO NOT COMPARE THIS TO MICROSOFT'S 0.028 J/token

Microsoft's model card lists 0.028 J/token. Ours is 3.31 J/token — 118× larger. **These
are not the same measurement.**

- Microsoft's is a **modeled estimate of matmul arithmetic energy** on modern-process
  assumptions. It excludes memory, interconnect, the rest of the SoC, and the board.
- Ours is **measured wall power of an entire 2019 laptop** — 14nm SoC, DRAM, board
  regulators, fan — minus only idle draw, running **batch-1, single-stream** inference.

Batch-1 decode is the *worst case* for J/token: weights are re-read for every token with
no amortization. The honest framing of our number is: **a complete offline LLM assistant
costs ~18 W above idle on a six-year-old Chromebook.**

---

## 5. THE NEGATIVE RESULTS (the actual contributions)

These are the parts we believe are novel, useful, and most worth attacking.

### 5.1 The real sparsity of BitNet b1.58-2B is 42.21%, not ~61.8%

Full scan of **all 210 ternary tensors** (2.084 G weights) in
`aegis_pruned_model.safetensors`:

```
zero      : 42.21 %
+1        : 28.90 %
-1        : 28.89 %
invalid(11): 0.0000 %
```

(An earlier 6-tensor sample gave 40.78%. The full-model figure is 42.21%. Both are
reported here because the 40.78% number appears in an intermediate benchmark run.)

The "golden ratio 61.8% sparsity" that the Infinity OS design was built around was never
a property of these weights.

### 5.2 "Zero-multiplication" ternary kernels LOSE to SIMD FMA, at every realistic sparsity

The one genuine artifact in the whole archive was
`Infinity OS/crates/aegis-forge-v5/src/dual_matvec.rs`: a dual-bitmap ternary matvec
that skips zeros using `trailing_zeros()` (CTZ) with the `w &= w-1` clear-lowest-bit
idiom, performing **no multiplications at all**. It was never benchmarked against
anything real.

Ported faithfully, same machine, same session, best-of-5 interleaved, outputs verified to
agree, both layouts costing 2 bits/weight:

| zero fraction | CTZ dual-bitmap | AVX2 LUT matvec | AVX2 batched GEMM |
|---|---|---|---|
| **42.21%** (real weights) | 12.22 ms | **1.94 ms (6.3× faster)** | **1.05 ms (11.7×)** |
| 61.8% (the assumed target) | 8.52 ms | 1.92 ms (4.5×) | 1.04 ms (8.2×) |
| 90% | 5.25 ms | 2.02 ms (2.6×) | 1.06 ms (5.0×) |
| 99% | 1.87 ms | 1.86 ms (ties) | 1.01 ms (1.8×) |

**CTZ needs ~99% sparsity to tie the matvec and ~99.5% to tie the GEMM.**
Substituting it into the engine today would take decode from 8.25 tok/s to **~0.7 tok/s**
and energy from 3.31 to **~30 J/token**.

**Why the premise misleads.** On this silicon a fused multiply-add is *one instruction,
eight lanes wide, fully pipelined* — the same cost as an add. The SIMD kernel therefore
processes a zero weight at **zero marginal cost.** Skipping zeros buys nothing, while the
skip machinery costs a serial loop-carried accumulator, an unpredictable branch per
nonzero, and a scalar load per nonzero.

> **Ternary weights make multiplications free. They do not make SIMD lanes free.**

The algorithm is not wrong — unstructured zero-skip is correct at extreme sparsity. It is
wrong at 42%.

### 5.3 The corollary: the field's causal story about ternary LLMs may be wrong

Horowitz (ISSCC 2014): a 32-bit FP multiply costs ~3.7 pJ; a 32-bit DRAM access costs
640–2000 pJ. On modern HBM, ~200 pJ per 32 bits vs ~1.5 pJ for a MAC — a **130×–650×
gap.** Arithmetic is nearly free; **data movement is where the energy is.**

Therefore:

> **BitNet does not save energy because it eliminates multiplications. It saves energy
> because it moves 2 bits per weight instead of 16.** "Multiplication-free inference" is
> the right effect attributed to the wrong cause.

If this is correct, it has design consequences: effort spent on multiplication-free
adder trees and zero-skip kernels is spent on a variable that stopped costing anything,
while the variable that matters (bytes moved per token) is addressed by quantization,
KV-cache compression, and batching.

**This is the claim we most want you to attack.** §9 lists how.

### 5.4 T-MAC-style LUT mpGEMM also loses on AVX2 without VNNI

We built the `pshufb`-based LUT kernel (the technique bitnet.cpp calls TL1): a packed
weight nibble encodes two ternary weights, and for a fixed activation pair all 16
outcomes fit in one 16-byte `_mm256_shuffle_epi8` table — evaluating 64 MACs in a single
instruction.

Clean-machine result, one activation vector, 2560×6912:

```
current f32 LUT + FMA matvec ....  9.65 GMAC/s
current batched GEMM ........... 17.07 GMAC/s
LUT-mpGEMM pshufb kernel .......  9.44 GMAC/s   <- loses
```

The pshufb genuinely does 64 MACs per instruction. What kills it is **widening the int8
partial sums**: `extracti128` + 2× `cvtepi8_epi16` + 2× `add_epi16` ≈ 9 instructions per
64 MACs, not 3. Meanwhile one `vfmadd231ps` already delivers 8 MACs with its LUT operand
hitting L1. `maddubs_epi16` would widen and pair-sum in one instruction, but it sums
**adjacent lanes**, which in this layout are different output rows.

**A real integer win requires `vpdpbusd` (AVX-VNNI / AVX-512 VNNI)**: 32 int8 MACs per
instruction, i32 accumulation, no widening. Comet Lake lacks it. **On a VNNI machine this
conclusion may reverse — we could not test that.**

### 5.5 int8 activations: quality win, not a speed win

The reference BitNet quantizes activations to int8 per token before every BitLinear, and
the weights were trained against that grid via straight-through estimation. **Feeding f32
activations is the deviation from the reference, not the other way round.**

Applied as a quantize→dequantize round trip on the existing f32 path (numerically
identical to an integer kernel, requiring no new kernel):

**WikiText-2 perplexity 12.801 → 12.738, for ~2% decode cost.** Now enabled by default.

(Clamped to [−127, +127], never −128: `_mm256_sign_epi8` saturates −(−128) to +127, so
the asymmetric end of int8 would silently flip sign in any future integer kernel.)

### 5.6 Bare metal buys ~0% speed over Linux — ⚠️ REFUTED BY REAL HARDWARE, 2026-07-09

**Read this section as a cautionary tale, not a result.**

Same engine, same machine: **599M cycles/token bare-metal vs 571M userspace, within 2%.**

That measurement was taken **under QEMU/KVM, on a host whose Linux kernel was managing
CPU frequency.** A subsequent run on a real Dell Inspiron 15 (Core i5, AVX2) measured
~3.65 B TSC ticks/token against this machine's 0.44–0.60 B — roughly **7× slower**, which
microarchitecture cannot explain.

The likely cause: **on true bare metal there is no operating system to raise P-states.**
The firmware leaves the core at its base or minimum clock. `rdtsc` is invariant — it ticks
at nominal frequency regardless of the core's actual speed — so a slow core inflates
ticks/token exactly as observed.

If this holds, the conclusion **inverts**: running without an operating system is
substantially *slower*, because the OS was doing something valuable that nobody noticed —
asking the CPU to go fast. The unikernel's case would then rest entirely on auditability
and determinism, with a performance penalty to declare honestly.

Not settled. Four experiments would resolve it (UEFI `GetTime()` wall-clock timing;
reading `IA32_MPERF`/`IA32_APERF` to measure the actual P-state; a Linux userspace
baseline on the same Dell; the Dell's exact CPU model). See
`docs/PREREGISTERED_HARDWARE_TEST.md`.

**Do not cite the paragraph below without the paragraph above.**

This quietly demolishes the *performance* rationale for unikernel AI — including this
project's original one. LLM decode is compute-bound in a tight loop; it makes no syscalls
and the OS scheduler is almost never invoked. The reason to run without an OS is
**attack surface, determinism, and sovereignty — not throughput.**

---

## 6. WHERE THE ENGINE'S TIME ACTUALLY GOES (measured)

| Quantity | Value |
|---|---|
| Ternary matvec MACs/token | 2,084 M (**94.2%** of all MACs) |
| LM-head MACs/token | 129 M (5.8% of MACs, but 33% of bytes) |
| Bytes read per token | 778 MB (521 MB ternary weights + 257 MB BF16 embeddings) |
| **Achieved bandwidth** | **2.08 GB/s** |
| **Machine bandwidth ceiling (measured)** | **17.3 GB/s** |
| **Achieved compute** | **3.48 MAC/cycle/core = 21.7% of single-core AVX2 peak** |

**This engine is compute/ILP-bound, not memory-bound** — it uses 12% of available
bandwidth. This inverts the usual assumption for LLM inference on CPU, and it is *why*
the batched GEMM's ~8× reduction in weight traffic only yielded 1.75×: the win came from
amortizing the per-weight LUT unpack, not from the memory traffic.

Note this is a property of *this* configuration (tiny ternary weights, slow CPU). On a
GPU with fp16 weights, decode is bandwidth-bound. Do not generalize.

---

## 7. WHAT IS AND IS NOT ORIGINAL (checked against the literature, not assumed)

**Not original:**
- The artifact category. **Someone else has already built bare-metal UEFI LLM
  inference** — freestanding C, on a Dell E6510
  (insights.marvin-42.com, "Bare-Metal AI: Running LLM Inference Directly in UEFI").
  Their writeup reports **no model, no tokens/sec, no SIMD, no perplexity, no energy**,
  and describes itself as "quite slow due to lack of optimization." We are not first;
  we are, as far as we can tell, the only instance with measurements attached.
- The soft-float UEFI problem: known upstream, **rust-lang/rust#136540**.
- The model, the quantization, the AVX2 LUT matvec, batched GEMM tiling, row-parallel
  thread pools, int8 activations — all standard.

**Plausibly original:**
- §5.1 (real sparsity), §5.2 (CTZ refutation + crossover point), §5.3 (the causal
  correction), §5.4 (pshufb loses without VNNI), §5.6 (no-OS gives no speedup).
- The apparatus itself: the same engine, same binary, with and without an operating
  system, fully instrumented. Nobody else appears to have this.

**The experiment nobody has run:** we measured bare-metal *speed*. We never measured
bare-metal **energy**. Does removing the OS — timer interrupts, scheduler, daemons —
reduce joules per token? This apparatus can answer that and no other apparently can.

---

## 8. ERRORS CLAUDE MADE DURING THIS SESSION

Listed so you can calibrate how much to trust the rest.

1. **Told the user the soft-float finding was a novel discovery.** It is a known,
   tracked rustc issue (#136540). I asserted it without checking.
2. **Concluded "SMT siblings contend for the FMA ports; 8 threads is slower than 4"**
   and changed the engine's thread default on that basis. **Wrong.** A runaway process
   (see §8.5) was holding a physical core, making 4 threads oversubscribe. On a clean
   machine, 8 threads is ~5% *faster*. Default reverted.
3. **Predicted a 4× speedup from int8 kernels.** That figure assumes VNNI, which this
   CPU lacks. The real ceiling on AVX2 is ~1.23× for the naive kernel, and the
   sophisticated one (§5.4) actually loses.
4. **First `lut_mpgemm` benchmark was wrong by 3×** — it rebuilt the activation LUT
   inside the row-block loop (276,480 times instead of 3,456). Caught before any
   conclusion was drawn, but only because the result looked implausible.
5. **I spawned the runaway process myself.** A `grep -r . /sys/class/power_supply/battery`
   was rewritten by tooling into `ugrep` treating `.` as a path, recursively walking
   `aegis-uefi/` including gigabytes of build artifacts. It ran for **8 hours**, pinning
   one of four physical cores, and **contaminated every performance number taken before
   ~12:15**, including the "idle" power baseline (inflated 6.3 W → 20 W).
   `measure_energy.sh` now refuses to run if any process exceeds 25% CPU.
6. **First energy measurement used 60 s windows** against a sensor that latches for
   15–30 s. Two independent estimates disagreed by 40%. Discarded and re-run.
7. **Reported 40.8% weight sparsity from a 6-tensor sample.** The full-model figure is
   42.21%. (The conclusion is unaffected; the CTZ benchmark was re-run at the true value.)

**Pattern:** every one of these was caught by a measurement contradicting a confident
statement, and none were caught by reasoning alone.

---

## 9. HOW TO ATTACK THIS DOCUMENT

The most likely places we are wrong. We would rather you find these than a grant reviewer.

**Against §5.2 (the CTZ refutation):**
- **Is our CTZ port a strawman?** It is scalar, faithful to the original. Could a
  *vectorized* zero-skip — AVX2 gather on the set-bit indices, or a POPCNT-based
  formulation — beat the FMA kernel at 42% sparsity? We did not test this. Our physical
  argument says no (gather is slow, and you still pay to identify the nonzeros), but we
  have no measurement.
- **Is the comparison fair on memory?** Both layouts are 2 bits/weight. But CTZ's two
  bitmaps are accessed in a different pattern than the packed codes. Cache effects?
- **Does the conclusion hold on other ISAs?** ARM NEON has no FMA-with-LUT equivalent of
  the same shape. RISC-V vector? A machine with `vpdpbusd`? We tested exactly one CPU.

**Against §5.3 (the causal correction):**
- Horowitz's numbers are from 2014 and 45nm. Do the ratios hold on 3nm with modern SRAM
  hierarchies and HBM3? The direction is robust; the magnitude may not be.
- **Is "bytes moved" the whole story?** At data-center batch, weights are amortized and
  KV-cache traffic dominates. Our claim is about the *mechanism of BitNet's win*, not
  about total data-center energy. Have we conflated them anywhere?
- Does anyone actually claim multiplication-elimination as the mechanism, or are we
  attacking a strawman assembled from marketing copy? **Find the strongest version of the
  opposing claim in the literature and check it.**

**Against §4.2 (perplexity):**
- **We have no error bars.** One 1,842-token sample, one run. The 12.801 → 12.738 delta
  is 0.5%. Greedy decoding makes it run-to-run deterministic, so it is not *noise* in
  that sense — but a different text sample could easily produce a different delta, or the
  opposite sign. **This is the weakest number in the document.** The correct experiment is
  the full test set with bootstrap resampling. It was not run (~4 h).
- The pruned vocabulary makes these numbers incomparable to published figures. Is there
  any legitimate way to compare?

**Against §4.4 (energy):**
- **Single run.** No repeats, no error bars. The coulomb cross-check agrees at idle
  (0.8%) but the load window's two estimates differ by ~17%, which is the counter's
  quantization limit, not a validation.
- Battery-discharge measurement includes DC-DC conversion losses. A wall meter on AC
  would measure a different (larger) quantity. Which is the right one to report?

**Against §5.6 (no-OS gives no speedup):**
- Two data points (599M vs 571M cycles/token), taken on different days, on a machine we
  now know was contaminated for part of that period. **This number should be re-taken.**
- `rdtsc` counts at the nominal rate; a throttled core inflates the count. Was the
  bare-metal run throttled differently from the userspace one?

**Against the whole document:**
- Everything was measured on **one CPU, one model, one workload, batch size 1.** The
  paper's-worth of general claims rest on a very narrow base.

---

## 10. WHAT WE THINK THE NEXT WORK IS

Not a plan — a set of falsifiable hypotheses, ranked by (leverage × feasibility for a
solo researcher with a Chromebook).

1. **Write up §5.2 + §5.3.** The data exists. A short, correct paper that says
   *"ternary LLMs save energy through data movement, not arithmetic"* with a measured
   crossover point. Prediction to falsify: someone builds a vectorized zero-skip kernel
   that beats FMA at <99% sparsity.

2. **Energy as a function of context length; what KV-cache compression actually buys.**
   At data-center batch, weight cost amortizes and **KV traffic + token count dominate.**
   Our own data shows the effect (6.98 → 6.41 tok/s from 100 → 400 tokens). Nobody has
   published measured J/token vs context on a machine with the OS removed as a confound.
   Prediction: J/token grows super-linearly with context; int4 KV quantization cuts it
   substantially with perplexity cost below noise.

3. **Speculative decoding as an *energy* technique, not a latency technique.** It verifies
   K draft tokens per weight read — a direct data-movement multiplier. Almost everyone
   reports it as a latency win. The batched GEMM built in §4.3 *is* the verification
   kernel. Prediction: J/token falls with acceptance-rate × batch-amortization, and the
   win is larger on memory-bound hardware than on this compute-bound one.

4. **The no-OS energy delta** (§7). Small, unique, cheap. Nobody else can run it.

5. **Output length as an unowned energy variable.** Energy is near-linear in tokens
   emitted. A 30% reduction in verbosity is a 30% reduction in serving energy, available
   today, with no hardware change. Prediction: measured J/answer varies more across
   prompting/decoding configurations than across quantization schemes.

**What we are NOT doing:** building another engine, pursuing a patent, or claiming the
edge artifact addresses data-center carbon. It does not, and cannot: at serving batch,
the very thing it optimizes (per-token weight movement) is amortized away.

---

## 11. REPRODUCTION

Repository: 8 commits on `main`, 130 files, ~3.7 MB.
Backup bundle: `alice-repo-2026-07-09.bundle` (`git clone` it directly).

```bash
# Userspace generation (stable Rust)
cd aegis-linux && cargo build --release --features parallel
./target/release/aegis-linux ~/aegis_pruned_model.safetensors \
    ~/aegis-forge/embed.bin ~/aegis-forge/vocab.bin 64 "Your prompt"

# Unikernel (nightly + rust-src; the script FAILS if soft-float regresses)
cd aegis-uefi && ./build_hardfloat.sh
mcopy -o -i ~/aegis-boot.img \
    target/x86_64-uefi-hardfloat/release/aegis-uefi.efi ::/EFI/BOOT/BOOTX64.EFI
qemu-system-x86_64 -enable-kvm -nographic -bios /usr/share/ovmf/OVMF.fd \
    -m 2G -machine q35 -cpu host -drive file=aegis-boot.img,format=raw

# Hardware-compatibility matrix (no USB burn cycles)
cd aegis-uefi && ./test_matrix.sh

# Perplexity
cd aegis-eval && ./target/release/aegis-eval <model> <embed.bin> <vocab.bin> wikitext2.txt 1900

# THE KERNEL RACE — the central negative result
cd aegis-core && cargo run --release --bin ctz_vs_simd

# The rejected T-MAC-style kernel, with its numbers in the header
cd aegis-core && cargo run --release --bin lut_mpgemm

# Energy (must be on battery; refuses if any process >25% CPU)
./measure_energy.sh

# Bit-identity of the batched GEMM against the per-token path, all batch sizes
cd aegis-core && cargo test --release --test gemm_equivalence
cd aegis-core && cargo test --release --features parallel --test thread_safety
```

**Before benchmarking anything: `ps -eo pcpu,comm --sort=-pcpu | head`.**
A single busy core adds ~13 W on this machine — larger than the entire inference signal.
We learned this the expensive way (§8.5).

---

## 12. THE QUESTION FOR GEMINI

You participated in producing some of the documents refuted above. We are not interested
in apology or in blame — the failure mode is more interesting than the fault.

**Three specific requests:**

1. **Attack §5.2 and §5.3.** These are the claims we believe are novel and true. Find the
   strongest published version of the "multiplication-free inference" thesis and tell us
   whether we are refuting it or a strawman. Design the experiment that would falsify us.

2. **Audit §8 and §9.** We have listed our own errors and the attacks we expect. What did
   we miss? Which number in §4 would you trust least, and why?

3. **The generalization question.** Every measurement here is one CPU, one model, batch 1.
   Which conclusions survive that, and which are artifacts of this machine? Specifically:
   does §5.2 hold on a CPU with `vpdpbusd`? Does §5.6 hold for a memory-bound workload?

And one open question we could not resolve, offered without an answer:

> A human and an LLM, working together in good faith, produced fifteen months of
> internally consistent, technically literate, confidently worded engineering documents
> describing a system that did not work — and neither of them ran the program. In one day
> of running the program, three cherished claims died, including one the LLM had itself
> recommended that morning.
>
> **What is the minimum intervention that would have caught this in week one?**

We think the answer is a coherence test wired to CI before any documentation is written.
But we would like to know what you think, because you were there.

---

*Prepared 2026-07-09. Every number herein is traceable to a command in §11 or is labeled
as estimate, inference, or opinion. Where this document is wrong, it is wrong in ways we
have tried to make easy to find.*
