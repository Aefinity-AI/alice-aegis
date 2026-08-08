# A.L.I.C.E. Technical Report

**Aegis Lightweight Inference Core Engine — bare-metal ternary LLM inference**
Justin B. Thompson · 2026-07-09
Status: working prototype, all results below measured on commodity hardware

---

## 1. Summary

A.L.I.C.E. is a from-scratch, `no_std` Rust implementation of the BitNet b1.58
ternary transformer that runs **coherent large-language-model inference with no
operating system**, booting directly from UEFI firmware off a USB stick. The
identical engine also runs as a Linux userspace binary. The complete inference
stack — model loader, 2-bit ternary kernels, attention, tokenizer, sampler — is
**3,865 lines** of auditable Rust in `aegis-core/src` (3,119 excluding blanks and
comments), with `libm` and `serde` as the only runtime dependencies. Earlier
drafts of this report said "approximately 1,600 lines"; that figure was never
derived from a count and is 2.4× low.

The capability this demonstrates: a **sovereign, air-gapped AI appliance**. No
kernel, no network stack, no filesystem daemons, no package manager — the
attack surface is the firmware plus the **5,443 lines** of Rust that constitute
the deployed unikernel (`aegis-core/src` 3,865 + `aegis-uefi/src` 1,578; 4,318
excluding blanks and comments). The entire deployed
artifact (engine + 2B-parameter model + tokenizer) fits in under 800 MB and
boots on machines from ~2012 onward (64-bit UEFI firmware).

## 2. What was measured (Intel i5-10210U, 6 GB RAM Chromebook, 2026-07-09 unless otherwise noted)

| # | Claim | Measurement | Evidence |
|---|---|---|---|
| 1 | Coherent generation, Linux userspace | "The capital of France is **Paris**." with correct chat-template EOS; fluent multi-sentence answers to open prompts | `aegis-linux` session logs |
| 2 | Coherent generation, bare-metal (no OS) | Same prompt, same correct answer, booted via OVMF UEFI under QEMU-KVM; run terminates with hardware success signal (isa-debug-exit code 33) | `aegis-uefi/qemu_success_2026-07-09.log` |
| 3 | Decode throughput, AVX2+FMA path | **598,921,319 rdtsc ticks/token**, bare-metal unikernel under QEMU/KVM on the i5-10210U host. A matching 571 M userspace figure was quoted in earlier drafts, was never written to any log, and is **withdrawn** — along with the "within 2%" claim it supported (571 vs 599 is 4.7% apart, and the runs were days apart). On **physical** hardware the same binary measures **726,238,201 ticks/token (2.85 tok/s)** on a Dell i5-5200U once the platform's externally-asserted bd-PROCHOT clamp is cleared: **1.21× the virtualized figure, not 7×**. The "~7× slower on real hardware" this report carried from 2026-07-09 to 2026-07-28 was that machine held at core ratio 5 (~500 MHz against 2200 nominal) with `hot=0` and the die 60 °C below Tj — a firmware clamp, not a property of running without an OS. `PREREGISTERED_HARDWARE_TEST.md` records the original prediction; its "no OS to manage P-states" hypothesis is superseded by this measurement | `aegis-uefi/qemu_success_2026-07-09.log:22`, `docs/hardware_logs/bitnet_baremetal_postfix_2026-07-29.log:17`, `docs/hardware_logs/gauntlet_dell_i5-5200U_2026-07-12_005500.txt:13` |
| 3a | Decode throughput, multicore | **2.80 tok/s single-thread → 4.94 tok/s multicore = 1.76×** — both on battery, both from committed logs taken the same day by the same script (single-thread build is f32-activation; multicore is `parallel`+`int8_act`). The only per-thread sweep on record — 2.62 / 4.77 / 5.61 / 5.40 tok/s at 1/2/4/8 threads, **2.14× at 4 threads, with 8 threads measuring SLOWER than 4** — exists solely in the message of commit 154f00a; no harness output was captured, so it is recorded here as **commit-only**. A later "clean re-measure" claiming 3.64 → 8.25 tok/s (2.27×, SMT positive) has no log anywhere on this disk and is **withdrawn** | `docs/hardware_logs/energy_run_i5-10210U_2026-07-14.log:12`, `docs/hardware_logs/energy_run_i5-10210U_multicore_2026-07-14.log:12`; commit 154f00a (commit-only) |
| 3b | Prefill throughput | batched GEMM 1.63–1.75× over per-token (single thread, A/B in both orders); 417 M → 176 M cycles/token at 4 threads (2.37×, commit 154f00a). An 8-thread prefill pair appeared in an earlier draft with no surviving log and is **withdrawn** | same |
| 4 | Decode throughput, scalar fallback | 3.60 B cycles/token on emulated Nehalem (2008-class CPU, no AVX); 3.71 B on the 2G-RAM variant. An earlier 4.01 B figure appeared only in a commit message with no surviving log and is superseded by these logged runs | `aegis-uefi/matrix_logs/Nehalem_q35_sata.log` (3,596,808,670), `Nehalem_q35_sata_2G.log` (3,707,228,193) |
| 5 | CPU compatibility (QEMU-emulated) | **6/6 matrix pass** under QEMU/KVM CPU feature masking — emulated CPUs, not physical hardware: Nehalem, SandyBridge (scalar dispatch), Haswell, modern (AVX2 dispatch) × q35/legacy-pc firmware × SATA and **XHCI USB** boot | `aegis-uefi/test_matrix.sh`, `matrix_logs/` |
| 6 | Language-model quality | WikiText-2 (`wiki.test.tokens`) **full test set** perplexity **16.124** (teacher-forced NLL, 312,119 scored tokens, chunked 5,000-char cold-KV mode — which biases upward vs continuous context; 30.6 h run, 2026-07-17 → 07-19) on the vocabulary-pruned model with the current byte-level tokenizer. Continuous-context sample anchor, first 7,401 ASCII characters: **10.758** (1,898 tokens, 2026-07-16). The previous anchors — full set **15.825** (313,479 tokens, 2026-07-12) and sample **10.348** — are **superseded**: both predate commit 5989c32 (2026-07-16), which made whitespace runs a single pre-token and restored 117 Llama-3 merges an id-0 guard had been discarding. Tokenization is denser after that fix, so per-token perplexity rose **by construction**; the +1.9% and +4.0% moves are artifacts of a changed denominator, not quality regressions, and **cross-tokenizer perplexities are not comparable in either direction**. Earlier figures 14.488 / 12.801 / 12.738 are superseded and **unlogged** — see §4 item 8 | `docs/hardware_logs/wikitext2_full_ppl_2026-07-17_newtokenizer_run.log`, `docs/hardware_logs/wikitext2_sample1900_2026-07-16_repin.log` |
| 7 | Energy per token | **3.99 J incremental** (19.69 W over idle at 4.94 tok/s), **4.99 J total system** (24.65 W). Idle 4.96 W. Battery-discharge method, 385 s decode window, 1,901 tokens, 2026-07-14 logged run (parallel+int8_act build). Single-thread build measures 4.63 J incremental. The earlier 3.31 J figure (2026-07-09, 261 s window) is superseded — its raw readings were never logged, so it fails this report's own traceability standard, though it is consistent within ~20% | `docs/hardware_logs/energy_run_i5-10210U_multicore_2026-07-14.log`, `docs/hardware_logs/energy_run_i5-10210U_2026-07-14.log`, `measure_energy.sh` |

**Comparability caveats, stated plainly:** (a) the perplexity is for the
50,256-token ASCII-pruned vocabulary, not directly comparable to full-vocab
figures; (b) the full-test-set run is complete — 16.124 over 312,119 scored
tokens in chunked cold-KV mode, which biases upward vs the continuous-context
sample anchor of ~10.76; (c) throughput is greedy decoding, single stream, batch
size 1; (d) perplexities measured under different tokenizers are not comparable
in either direction — see row 6.

### 2.1 On the energy number, before anyone compares it to Microsoft's

Microsoft's model card lists **0.028 J/token** for BitNet b1.58 2B. Our
measured 3.99 J/token is ~142× that. **The two quantities are not the same
measurement and should never be compared directly.**

- Microsoft's figure is a *modeled* estimate of the **arithmetic energy** of
  the matmuls, on assumptions about a modern process node. It excludes memory,
  interconnect, the rest of the SoC, and the board.
- Ours is *measured battery-discharge power, whole-system* (on battery, under
  Linux) of an **entire 2019 laptop** — a 14 nm
  i5-10210U, its DRAM, board regulators, and fan — minus only the machine's
  idle draw, running a single-stream, batch-1, f32-activation engine.

The honest framings of our number are:

1. **A full offline LLM assistant costs ~20 W above idle on a six-year-old
   Chromebook**, or ~25 W total system draw. That is a real, verifiable
   deployment fact and it is the number that matters for edge/SWaP claims.
2. On-battery decode sustains **4.94 tok/s** over the 385 s logged window — and
   that is the highest *logged* decode rate this project has on file for the
   multicore build. Earlier drafts explained it as being held below an
   "8.25 tok/s short-burst peak"; that peak has no log and is withdrawn, so
   there is no measured on-AC figure to contrast against. Quote the logged,
   sustained 3.99 J/token; it is the conservative, traceable figure.
3. Batch-1 single-stream inference is the *worst case* for J/token — the
   weights are re-read for every token with no amortization across a batch.
   Data-center serving amortizes exactly this cost, which is why comparing an
   edge appliance to a data-center J/token figure flatters neither.

**Method note.** `current_now` on this platform is latched (holds for 15–30 s)
and `charge_counter` steps in coarse 40 mAh increments, so short windows cannot
resolve either. The 2026-07-14 production run used a **300 s idle baseline and a
385 s decode window** — long enough for the latch to update many times.

The two sensors do **not** agree closely, and that disagreement is the honest
uncertainty band: the primary `power_now` sampling read 4.957 W idle and
24.650 W under decode, while independent coulomb integration over the same
windows read 4.42 W and 29.41 W — **11% and 19% apart**. Taking the coulomb
estimate instead of the primary puts the same run at 5.06 J/token incremental
and 5.96 J total. `power_now` is the primary sensor and the 40 mAh counter step
is coarse relative to these windows, so 3.99 J is the reported figure;
**3.99–5.06 J incremental is the band.**

An earlier attempt with 60 s windows disagreed by 40% between the two sensors
and was discarded (recorded in commit 254ba43's message; no readings log
survives). ⚠️ The "240 s / 261 s windows" and the "0.8% idle agreement (6.30 W
sampled vs 6.35 W coulomb)" quoted in earlier drafts of this note belong to the
**superseded 2026-07-09 run**, whose raw readings were never logged and whose
decode-window cross-check was never recorded at all. They were never evidence
for the number in row 7.

**A caution that cost us a day of measurements.** Every performance figure in
an earlier draft of this report was taken while a runaway process held one of
four physical cores (a recursive `ugrep` left running for eight hours). It
inflated "idle" power from 6.3 W to 20 W and made 4 threads appear to beat 8.
All numbers here were re-taken on a verified-quiet machine, and
`measure_energy.sh` now refuses to run if any process exceeds 25% CPU. If you
reproduce this work, check `ps` first.

## 3. Architecture

```
USB stick (FAT32)                         RAM (locked physical pages)
├── EFI/BOOT/BOOTX64.EFI  ── UEFI boot ─▶ engine (~220 KB)
├── MODEL.SAF   522 MB    ── DMA bounce ▶ 2-bit ternary weights (zero-copy views)
├── EMBED.BIN   257 MB    ── 64KB-aligned▶ BF16 embeddings (tied LM head)
├── VOCAB.BIN   1.7 MB    ──  chunks   ─▶ BPE vocab + 110k merges
└── BOOTLOG.TXT ◀── stage checkpoints written back for post-crash forensics
```

- **Model:** Microsoft BitNet b1.58-2B-4T (MIT): 30 layers, hidden 2560, GQA
  20q/5kv, squared-ReLU FFN, SubLN, RoPE θ=500k. Weights ternary {-1,0,+1},
  packed 4 per byte. Quantization is Microsoft's; everything below is original.
- **Kernels:** hand-written AVX2 ternary GEMV (1024-entry LUT unpack +
  FMA, 4-row unrolled) with **runtime CPUID dispatch** to portable scalar
  fallbacks — one binary serves 2008-era to current CPUs (the 2008-era claim
  is verified under QEMU CPU emulation only, not on physical hardware).
- **Firmware bring-up:** the UEFI app enables AVX itself (CR4.OSXSAVE + XCR0
  via `xsetbv` — firmware boots with it off), allocates weights by scanning
  the UEFI memory map for contiguous conventional regions, and streams files
  through a 64 KB-aligned DMA bounce buffer (XHCI transfer-boundary rule).
- **Memory:** zero-allocation arena for the entire hot path; KV cache sized
  for a 2,048-token window. Measured footprint (Dell i5-5200U bare-metal
  gauntlet): 781,902,232 bytes (~782 MB) read-only weights + 571,912,597 bytes
  (~572 MB decimal / 545.41 MiB) peak working memory ≈ 1.35 GB total.
  Compatibility floor: **2 GB RAM** for the 2B pruned model (QEMU: 2 GB boots
  and generates; 1.4 GB fails with a clean OOM panic).
- **Custom hard-float UEFI target:** the stock Rust UEFI target is soft-float
  (zero SIMD instructions emitted). `x86_64-uefi-hardfloat.json` + build-std
  produces real vector code; `build_hardfloat.sh` verifies FMA presence in the
  binary at every build.

## 4. Defects found and fixed to reach this state (2026-07-09)

Documented because reviewers value failure analysis, and because several were
silent for the project's entire history:

1. **LM-head size guard** expected F32 embeddings after the BF16 migration —
   argmax silently returned token 0 (`aegis-core/src/ops.rs`).
2. **BPE merges missing from vocab.bin** — the vocab stripper never wrote the
   merges section, degrading tokenization to characters.
3. **Chat-template special tokens absent** — Llama-3 specials live in
   tokenizer.json `added_tokens`, which the stripper never read; regenerated
   vocab + embeddings as a matched 50,256-token pair.
4. **Soft-float UEFI codegen** — every prior unikernel build contained zero
   vector instructions; all AVX2 intrinsics were scalarized to library calls
   (~100× slowdown). Root cause of every historical "hang" on bare metal.
5. **AVX enable gated on the wrong CPUID bit** (27 = already-enabled, vs 26 =
   capability) — the enable block never executed; first real AVX instruction
   faulted #UD.
6. **Perplexity state advanced with the wrong token** (`tokens[i]` vs
   `tokens[i+1]`) — any perplexity ever computed before this fix was invalid.
7. **Quadratic BPE encode** — O(n²) merge loop; fine for prompts, days for
   megabyte inputs. Mitigated in the eval harness; heap-based encoder queued.
8. **Tokenizer was not byte-level** — `encode()` looked up Unicode characters
   directly while `decode()` implemented the full GPT-2 byte↔unicode table, so
   every newline, tab, and non-ASCII character became
   `<|reserved_special_token_0|>`, and BPE merges ran across word boundaries.
   Fixing both (byte mapping + word-bounded merges) moved WikiText-2 perplexity
   from **14.488 → 12.801** with no change to the model weights. (Both figures
   are since superseded by the re-baselined anchors in §2 row 6; the delta
   documents the fix's effect at the time.)
9. **Console-reachable stack overflow** in the UEFI REPL: the buffer guard
   checked the current length, then wrote up to 3 UTF-8 bytes for a `Char16`.
10. **Truncated reads reported as success** in the UEFI loader — a short USB
    read silently loaded corrupted weights.
11. **Unbounded tensor slicing** — a malformed safetensors offset panicked,
    which on bare metal is an unrecoverable fault, not a `Result`.
12. **The energy script could not have produced a correct number** — its token
    count was hardcoded by a no-op command substitution, and it divided
    decode-only power by a wall time that included model loading.

## 5. Known limitations

- Activations are f32; the BitNet W1.58A8 int8 path (the larger half of the
  published efficiency win) is not yet implemented.
- Greedy argmax + repetition penalty only; temperature/top-p unimplemented.
- Multicore in userspace only (`parallel` feature, needs std). The UEFI
  unikernel remains single-threaded; bringing up the other cores requires the
  firmware's MP Services protocol.
- Pruned ASCII vocabulary: multilingual/emoji text degrades.
- 2,048-token context window.
- 2B-parameter model quality: fluent but factually unreliable.

## 6. Where the performance actually goes (measured, 2026-07-09)

Profiling the decode step on this machine:

| Quantity | Value |
|---|---|
| Ternary matvec MACs per token | 2,084 M (94.2% of all MACs) |
| LM-head MACs per token | 129 M (5.8% of MACs, but 33% of memory traffic) |
| Bytes read per token | 778 MB (521 MB ternary weights + 257 MB BF16 embeddings) |
| Achieved DRAM read bandwidth, single thread | **2.04 GB/s** — 778 MB/token at the logged single-thread rate of 2.62 tok/s (commit 154f00a). Derived instead from the logged decode timing (598,921,319 rdtsc ticks/token at this machine's **measured** 2.1119 GHz TSC rate) it is **2.74 GB/s**. The 2.08 GB/s carried in earlier drafts assumed the core ticked at the marketed 1.60 GHz base clock; it does not |
| Machine bandwidth ceiling, single thread | **Withdrawn — never logged.** The 17.3 GB/s figure carried since commit 2ef5956 has no benchmark, no script and no log anywhere in this repository. A re-measurement on the same machine (192 MB AVX2 read stream, eight accumulators, best of nine, machine **not** idle — load average 7.6) reached **11.97 GB/s** read-only. Until a STREAM-style triad is run on a verified-idle machine and its output committed, take the single-thread read ceiling as **~12 GB/s, provisional** |
| Achieved compute | **≈2.3 MAC/cycle/core ≈ 14% of single-core AVX2 f32 peak** (peak = 2 FMA units × 8 f32 lanes = 16 MAC/cycle). ⚠️ The 3.48 MAC/cycle / 21.7% carried in earlier drafts divided MACs by **rdtsc ticks**, which are not core cycles — RDTSC advances at the invariant nominal rate regardless of DVFS, and this machine's measured effective/nominal ratio under load is **1.53×** (3.226 GHz core against a 2.1119 GHz TSC). Every MAC/cycle and %-of-peak figure derived from `_rdtsc()` (`aegis-core/src/inference.rs:784`) is low by that factor. The corrected values above are **derived, not measured** — the engine has no PMU access, so `CPU_CLK_UNHALTED` has never been read |

**Conclusion: the engine is bound by neither wall — it is latency/ILP-bound.**
It uses roughly 17–23% of a provisional ~12 GB/s read ceiling and ~14% of peak
FMA throughput. Both utilisations are low, which is the actual finding: the
headroom is real and reachable, and it does not require winning a bandwidth
fight. Note this is a *weaker* claim than the "12% of bandwidth" in earlier
drafts, and it rests on a withdrawn ceiling — treat the ratios as provisional
until the triad and a PMU-based cycle count are both logged.

Within the FFN/attention split, the FFN accounts for 76% of per-layer MACs.

## 7. Roadmap, ordered by measured payoff

**Tier 1 — large, well-understood wins**

1. ✅ **Batched prefill GEMM** (done 2026-07-09). Weights now stream once per
   8-token tile instead of once per token. Measured **1.63–1.75×** single
   threaded. Notably *not* the 8× the traffic reduction implies — because the
   engine is compute-bound, the real win is amortizing the per-weight LUT
   unpack, not the memory traffic. Tile widths 4–8 are equivalent; ≥10 regress
   on register spills.
2. ✅ **Multicore decode** (done 2026-07-09). Row-parallel matvec and LM head
   over a persistent worker pool (per-call `thread::scope` cost 52 ms/token in
   spawns and had to be replaced). The gain on record is **2.14× at 4 threads**
   (2.62 → 5.61 tok/s) — but it is **commit-only**: it exists solely in the
   message of commit 154f00a, no harness output was ever captured, and the
   ledger (A4.sweep2026) bans it from external documents until `thread_sweep`
   emits a log. Cite the logged multicore pair (2.80 → 4.94 tok/s, 1.76×)
   instead. Recorded in commit 154f00a with its full Amdahl
   accounting: the parallel fraction is 94.2% (Amdahl ceiling 3.41×) and all-core
   turbo runs at ~0.71× of single-core, giving an achievable ~2.43×, of which
   2.14× is 88%. **In that same sweep 8 threads measured 5.40 tok/s — slower
   than 4 threads**, SMT siblings contending for the FMA ports. A later claim
   that SMT is worth ~5% and that 8 threads reach 8.25 tok/s has no log and is
   **withdrawn**; the shipped default of logical processors
   (`available_parallelism()`, `aegis-core/src/ops.rs:642`) therefore rests on an
   unsupported measurement and **must be re-benchmarked before it is defended**.
   Prerequisite completed: the `static mut` LUT is now a compile-time `const`
   and the SIMD flag an atomic.
3. ⚠️ **W1.58A8 int8 activations** (investigated 2026-07-09). **Quality: adopted.
   Speed: rejected on this ISA — with measurements.**

   *Quality.* Per-token absmax int8 activation quantization is what the
   reference does, and the weights were trained against it via straight-through
   estimation, so f32 activations are the deviation. Measured WikiText-2
   perplexity **12.801 → 12.738** at ~2% decode cost (both figures since
   superseded by the re-baselined anchors in §2 row 6; the A/B delta is the
   point here). Now on by default
   (`int8_act`); simulated as a quantize→dequantize round trip on the f32 path,
   which is numerically identical to an integer kernel and needed no new kernel.

   *Speed.* Candidate kernels versus the existing ones. The `pshufb` row's
   original "8.28 GMAC/s (16% slower)" verdict is **retracted** — the bench it
   came from never measured the f32 arm it claimed to lose to (the baseline was
   a number in a comment). Re-measured 2026-07-30 with every arm in ONE binary,
   interleaved, same weights (dev box i5-10210U; logs
   `docs/hardware_logs/lut_mpgemm_sameproc_ab*_2026-07-30.log`, three runs;
   derivation in `lut_mpgemm_ab_findings_2026-07-30.md`):

   | kernel | measured, same-binary A/B (3 runs) |
   |---|---|
   | current f32 LUT + FMA (matvec) | 8.94–9.36 GMAC/s (stable, ±4.7%) |
   | current batched GEMM (batch 8, per token) | 20.68–23.67 GMAC/s |
   | LUT-mpGEMM / T-MAC style (`pshufb`), incl. table build | 8.24–11.29 GMAC/s (bimodal, 36% swing) |
   | naive int8 (`sign_epi8`+`maddubs`) | not built — 1.23× ceiling |
   | pre-unpacked int8 weights | rejected: 2.1 GB of weights, won't fit |

   In its modal state the `pshufb` kernel is ~20% *faster* than the f32 matvec
   kernel-side (the ordering flipped in one of three runs). **The rejection
   stands anyway, on memory-traffic grounds:** the nibble layout spends 4
   bits/weight instead of 2, which takes decode traffic from 780,142,296 to
   1,302,973,872 B/token (1.670×). At post-throttle-fix clocks the decode path
   is bandwidth-bound (crossover 1.42–1.78 GHz, roof 10.3–12.8 tok/s), so the
   roof under the 4-bit layout falls to ~6.2–7.7 tok/s — pshufb's best measured
   kernel gain (1.21×) nets out to a ~28% decode *loss* (1.21/1.670 = 0.72×).
   The batched GEMM also beats it ~2× for prefill in every run, and the
   prototype kernel's output is numerically incorrect as benchmarked (madd
   pairing artifact, flagged in-source); repairing it costs widening
   instructions. There is no operating point where the swap wins. The prototype
   is kept at `aegis-core/benches/lut_mpgemm.rs`.

   **The existing f32 kernels remain the right answer on AVX2.** A real integer
   win needs `vpdpbusd` (AVX-VNNI / AVX-512 VNNI): 32 int8 MACs per instruction
   with i32 accumulation and no widening — and on such hardware the 4-bit
   layout's traffic penalty would still have to be beaten. Comet Lake lacks it;
   Ice Lake and later have it. Revisit there, not here.

(An earlier draft here called the 2.62 → 5.61 tok/s (2.14×) multicore result
"superseded" by a 3.64 → 8.25 tok/s re-benchmark. **That inversion is
retracted.** The 2.14× figure carries a full four-point thread sweep, an Amdahl
derivation and a named mechanism in commit 154f00a; the 8.25 tok/s re-benchmark
exists as four lines in the message of commit 254ba43 and nowhere else — no
harness output, no log file, no artifact. A number with no primary source cannot
supersede a number that has one. **2.14× at 4 threads stands.** Note that
254ba43 is the same run that produced the 3.31 J/token energy figure this report
already retracted for being unlogged; its other outputs have no better
provenance.) The remaining large speedups are not kernel-level on this CPU.

### 7.1 Settled: the CTZ "zero-multiplication" kernel is slower, at every realistic sparsity

An earlier project era (Infinity OS, `aegis-forge-v5/src/dual_matvec.rs`) built a
dual-bitmap ternary matvec that skips zero weights using `trailing_zeros()` and
performs **no multiplications at all** — only adds, subtracts, and bit scans.
It was the one genuine "zero-multiplication" artifact in the archives, and it
was never benchmarked against anything real; the era's headline figures were
simulated.

Raced against the current kernels, same machine, same session, best-of-5,
outputs verified to agree:

Re-measured to an instrument log 2026-07-30 (dev box, i5-10210U Comet Lake under
crosvm, `git_head` a249b2c); every figure below traces to
`docs/hardware_logs/ctz_vs_simd_2026-07-29.log`. 2560×6912 ternary matvec.

| zero fraction | CTZ dual-bitmap | AVX2 LUT matvec | batched GEMM |
|---|---|---|---|
| **42.21%** (measured, real BitNet weights) | 12.19 ms | **1.85 ms (6.59× faster)** | **1.04 ms (11.74×)** |
| 61.8% (the "golden ratio" target) | 8.73 ms | 1.85 ms (4.73×) | 1.05 ms (8.33×) |
| 90% | 5.52 ms | 1.83 ms (3.02×) | 1.01 ms (5.49×) |
| 99% | 2.02 ms | 1.83 ms (1.10×) | 1.01 ms (1.99×) |
| 100% (pure loop floor, no real weights) | 0.35 ms | 1.83 ms (0.19× — CTZ wins) | 0.99 ms (0.36×) |

Outputs agree bit-for-bit in every row. Memory is a wash: both layouts are
2 bits/weight (CTZ splits it across two bitmaps; the current packing uses one
2-bit code), so there is no footprint advantage either way.

CTZ needs ~99% sparsity to tie the matvec and ~99.5% to tie the GEMM, and only
reaches the crossover at literally-all-zeros. **The real BitNet weights are
42.21% zeros** — a full scan of all 210 ternary tensors (2.084 G weights). An
earlier 6-tensor sample gave 40.8% and that sample figure was what this section
and the ledger quoted; the conclusion is unaffected. The 61.8% figure the old
design assumed was never a property of these weights. Substituting CTZ into the
engine today would land on the decode path, which runs `ternary_matvec`
(`aegis-core/src/inference.rs:575`–`704`), so the applicable gap is the **6.59×
matvec column**, not the 11.74× GEMM column. Measured against the logged
on-battery baseline — 4.94 tok/s at 3.99 J/token incremental — that is roughly
**0.75 tok/s** and about **26 J/token incremental** (33 J/token total system).
Earlier drafts extrapolated from 8.25 tok/s and 3.31 J/token, both withdrawn as
unlogged.

**Why the premise misleads.** On this silicon a fused multiply-add is *one
instruction, eight lanes wide, fully pipelined* — the same cost as an add. The
SIMD kernel therefore processes a zero weight at **zero marginal cost**.
Skipping zeros buys nothing, while the skip machinery costs a serial
loop-carried accumulator, an unpredictable branch per nonzero, and a scalar
load per nonzero. Ternary weights make multiplies free; they do not make SIMD
lanes free. "Multiplications: ZERO" was optimizing a quantity that had already
stopped costing anything.

The algorithm is not wrong — unstructured zero-skip is correct when sparsity is
extreme. It is wrong at 41%. Benchmark preserved at
`aegis-core/benches/ctz_vs_simd.rs`.

**Tier 2 — quality and correctness (now the highest-value remaining work)**

4. **Full-regex pre-tokenization.** The current word-splitting approximates the
   reference GPT-2/Llama regex. With the int8 activation path now closed, this
   is the leading remaining suspect for the residual gap to reference
   perplexity (~11–12 vs our earlier 12.738 sample figure — superseded, see §2
   row 6), together with the pruned vocabulary.
5. ✅ **Full WikiText-2 test-set perplexity** (re-run 2026-07-17 → 07-19 on the
   fixed byte-level tokenizer: **16.124**, 312,119 scored tokens, chunked cold-KV
   mode, 30.6 h; log
   `docs/hardware_logs/wikitext2_full_ppl_2026-07-17_newtokenizer_run.log`.
   Supersedes the 2026-07-12 old-tokenizer run, 15.825 over 313,479 tokens, kept
   at `wikitext2_full_ppl_2026-07-12_run.log`). Still open: a like-for-like
   benchmark against `bitnet.cpp`/`llama.cpp` on identical hardware, the
   identical file, **and** the identical pruned vocabulary — without all three
   held constant the comparison measures the harness, not the engine.
6. **Heap-based BPE encoder** (O(n log n) instead of O(n²)).
7. **Repetition penalty over the prompt context**, not just generated tokens;
   revisit the combined divide-and-subtract penalty constants.
8. **f64 RoPE frequency computation** to remove long-context rotation drift.

**Tier 3 — the measurements that make the case**

9. **Energy per token** — `measure_energy.sh` on battery, then the same
   measurement on the bare-metal boot. The **no-OS energy delta** is the
   headline experiment: it is the one number nobody else can report, and the
   engine is already instrumented for it.
10. **Real-hardware validation** across machines (requires 64-bit UEFI;
    `BOOTLOG.TXT` gives post-crash forensics without a serial cable).

**Tier 4 — capability expansion**

11. **Sampling** (temperature/top-p are stubs; greedy only today).
12. **Larger context** via KV-cache paging from disk — the arena design
    already makes the working set explicit.
13. **A second model target.** The salvageable dense→ternary quantizer from
    the project's Gemma-era archives could, once validated against a reference
    forward pass, remove the dependency on Microsoft shipping pre-quantized
    weights. Validate the quantizer in PyTorch *before* it touches Rust.

## 8. Reproducibility

```bash
# userspace generation
cd aegis-linux && cargo build --release && ./target/release/aegis-linux \
  ~/aegis_pruned_model.safetensors ~/aegis-forge/embed.bin ~/aegis-forge/vocab.bin 64 "prompt"

# unikernel build + boot (nightly + rust-src)
cd aegis-uefi && ./build_hardfloat.sh && \
mcopy -o -i ~/aegis-boot.img target/x86_64-uefi-hardfloat/release/aegis-uefi.efi ::/EFI/BOOT/BOOTX64.EFI && \
qemu-system-x86_64 -enable-kvm -nographic -bios /usr/share/ovmf/OVMF.fd \
  -m 2G -machine q35 -cpu host -drive file=$HOME/aegis-boot.img,format=raw

# compatibility matrix
cd aegis-uefi && ./test_matrix.sh

# perplexity
cd aegis-eval && cargo build --release && ./target/release/aegis-eval \
  ~/aegis_pruned_model.safetensors ~/aegis-forge/embed.bin ~/aegis-forge/vocab.bin wikitext2_test.txt 1900

# energy (on battery)
./measure_energy.sh
```

§2 rows 1–5 and §6 were produced by the commands above on 2026-07-09 on the
i5-10210U. Row 6 (perplexity) was measured 2026-07-17 → 07-19 (full test set) and
2026-07-16 (sample anchor); row 7 (energy) was measured 2026-07-14. Row 3's
real-hardware comparison is a separate bare-metal USB boot of a Dell Inspiron 15
documented in `PREREGISTERED_HARDWARE_TEST.md` and is **not** reproducible from
this command list. The sample anchor requires the `--sample` flag — without it
`aegis-eval` runs chunked full-text mode over the whole file and reports a
different quantity:

```bash
# sample anchor (continuous 1,900-token prefix)
./target/release/aegis-eval ~/aegis_pruned_model.safetensors \
  ~/aegis-forge/embed.bin ~/aegis-forge/vocab.bin test.txt 1900 --sample

# full test set (chunked, cold KV per 5,000-char chunk)
./target/release/aegis-eval ~/aegis_pruned_model.safetensors \
  ~/aegis-forge/embed.bin ~/aegis-forge/vocab.bin test.txt 1900
```

Raw logs are preserved in `docs/hardware_logs/`. Historical documents predating
2026-07-09 contain simulated or unverified figures and are explicitly superseded
(see `PRE_REVIEW_SCRUB_LIST.md`).
