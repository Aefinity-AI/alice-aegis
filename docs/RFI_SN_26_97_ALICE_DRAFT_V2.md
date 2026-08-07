# COVER PAGE
**Response to DARPA-SN-26-97 (Low Resource Computing)**
**Submission Date:** 14 JUL 2026

**Project Title:** A.L.I.C.E. (Aegis Lightweight Inference Core Engine) — Sovereign Bare-Metal AI for DDIL Operations
**Categories Addressed:**
- Section 1(b): Low Memory
- Section 2(d): Low Complexity of User Experience

**Organization:** Aefinity AI (Orange, TX)
**Technical POC:** Justin Brian Thompson — full contact details supplied in the
submitted response; redacted in this public copy.
**Admin POC:** Justin Brian Thompson — full contact details supplied in the
submitted response; redacted in this public copy.
**Proprietary Information:** None.

---

# TECHNICAL SECTION (Max 8 Pages)

## Thesis Statement

In Denied, Degraded, Intermittent, and Limited (DDIL) environments, US forces require localized decision support that operates below the threshold of traditional power, network, and logistics chains. A.L.I.C.E. (hereafter ALICE) is a bare-metal, OS-less large-language-model unikernel — a `no_std` Rust application that boots directly through UEFI (Unified Extensible Firmware Interface) firmware on commodity x86-64 hardware, loads ternary (1.58-bit) model weights from a USB stick, and runs inference with no operating system, no network stack, no telemetry, and no persistence on the host machine. Its entire physical footprint is under 1.5 GB of RAM on a decade-old commodity laptop. ALICE demonstrates, with runtime-computed measurements on real legacy hardware, that the minimum compute floor for useful on-device language-model inference is far lower than current deployment paradigms assume — and that the floor is set by firmware generation and memory size, not by CPU capability.

## 1. Capabilities and Challenges Addressed (Respondent's Perspective)

The reality of a DDIL environment is not a degraded Wi-Fi signal; it is complete operational isolation. When a small unit or isolated soldier is cut off, every watt of power and every RF emission becomes a potential targeting solution. In these moments an operator does not need a cloud dashboard, and a 400 W server rack requiring a secure uplink is worse than useless — it is a liability.

The capability gap DARPA-SN-26-97 addresses is fundamentally a cognitive-load and decision-support problem under isolation. The modern soldier's judgment degrades with fatigue exactly when the stakes rise; a tireless, local, network-independent assistant addresses that failure mode directly — but only if it runs on what is actually available in theater: scavenged, commodity, legacy computers and battery power.

ALICE's approach is to delete the operating system rather than optimize around it. Traditional OS stacks require patching, emit telemetry, present large attack surfaces, and can lock the operator out at the worst possible moment (TPM/BitLocker trips, forced updates, license checks). ALICE boots from UEFI firmware directly into a single-purpose inference engine: there is no network stack to attack, no background process competing for memory, and the host disk is never mounted or written. The zero-persistence-on-host property rests on the design — the engine touches only its own boot USB stick, where it may append an optional flight-record log. The host platform's own measured-boot attestation independently detects the session: returning a test laptop to its resident Windows installation after an ALICE session triggers the TPM's boot-path attestation exactly as it should for any foreign boot. ALICE contains no networking or radio code of any kind; it performs no fire control, no targeting, no transmission — it is a reference and reasoning tool that informs a human who decides.

## 2. Technical Approach

- **The unikernel paradigm.** Eliminating the OS removes the OS tax on memory, compute, and trust. ALICE is a `no_std` Rust UEFI application executing at Ring 0 with a custom physical-memory allocator, its own FAT32 loader with DMA bounce buffers, and hand-written SIMD kernels. Working memory is a single arena sized at initialization; the inference hot loop performs zero heap allocation.
- **Ternary weights (BitNet b1.58).** The engine runs Microsoft's openly published BitNet b1.58-2B-4T, whose weights take values {-1, 0, +1} and pack four to a byte. The primary systems win is footprint and bandwidth: 2-bit weights cut model storage and memory traffic 8× versus bf16, which is what makes a 2-billion-parameter model fit — weights plus working memory — in under 1.5 GB. This is state-minimization at the representation level: the model's entire parameter state is compressed to two bits per weight and decompressed on the fly by a lookup-table unpack feeding dense AVX2 fused-multiply-add kernels. Runtime CPU dispatch falls back to portable scalar code on silicon without AVX2/FMA: the same binary is verified on Broadwell (2015) bare metal, on **physical no-AVX silicon** — a fanless Celeron N4020 laptop running the SSE2 scalar-fallback path bare-metal (Section 4) — and, via a QEMU/KVM feature-masking matrix, on emulated pre-AVX2 cores (Nehalem, 2008-class; Sandy Bridge, 2011-class).
- **Legacy-silicon execution.** Directly countering the RFI's exclusion of conventional high-tier CPUs: ALICE requires no GPU, no NPU, no AVX-512, no advanced-node or HBM supply chain. Assessed compatibility floor: any 64-bit UEFI machine (~2012 onward) with 2 GB of RAM and Secure Boot disabled — a one-time firmware-menu toggle, since the binary is unsigned today (signed/shim boot is roadmap work). Within that floor, CPU features determine speed, not boot success. The RAM boundary is bracketed empirically under QEMU/OVMF memory-limit runs (physical-floor validation is in progress): 2 GB boots and generates; 1.4 GB halts during weight loading with an explicit out-of-memory diagnostic.

## 3. Development Strategy and Metrics

- **Mission-model distillation (funded phase).** The published BitDistill pipeline (Microsoft, arXiv:2510.13998 — SubLN insertion, continual pre-training, attention distillation) converts small dense checkpoints to ternary at task parity. From the compute budgets reported in that paper, we estimate a rear-security / DDIL doctrine Q&A mission model at the 0.5–1B scale is a conversion on the order of days on a single GPU — an estimate to be validated in the funded phase. Until then, ALICE runs published natively ternary checkpoints; a port of TII's Falcon-E-1B (665 MB, benchmark parity with its bf16 counterpart per TII's model card) is in progress to extend the portfolio downward, below the current 2 GB floor toward sub-1 GB machines.
- **Curated on-device corpus with provenance (funded phase).** The model does language; a provenance-tagged reference corpus will do facts. In the planned design, substantive survival, medical, and cultural claims will be retrieved and cited on-screen, never free-generated, and "I don't know" will be a first-class output. Calibration will be reported as a measured metric on an adversarial evaluation set, alongside throughput.
- **Power envelope.** Measured whole-system energy: 3.99 J/token incremental (24.7 W under load vs 5.0 W idle, sustained multicore decode at 4.94 tok/s), obtained by battery-discharge measurement on the development laptop running the same engine under Linux — whole-system draw, not modeled kernel arithmetic, with both raw discharge logs committed beside the result, including their sensor caveats (single-threaded configuration: 4.63 J/token). At ~25 W whole-system, the envelope is battery-scale — hours on a laptop battery — not a standard commercial power level. The bare-metal (no-OS) energy figure is not yet measured; the OS-versus-no-OS energy delta on identical hardware is precisely the kind of measurement this program should demand, and it is next on our bench.
- **Verification discipline as a deliverable.** Every number in this document is computed at runtime from real work and traceable to a raw log in the project repository: bare-metal runs are logged by the engine to a boot-volume flight record (BOOTLOG.TXT) and harvested into a version-controlled dataset; quality and energy evaluations are logged by the host-side harness, with each measurement's environment stated where it is reported. The measurement harness aborts on artifact mismatches rather than reporting plausible garbage. We consider this discipline part of the low-resource story: a device trusted by an isolated operator must never bluff — and neither may its benchmark reports.
- **Implications for LRC program measurement.** The discipline above suggests candidate metrics for the program itself: whole-system joules per token by battery discharge, boot-to-first-token seconds, peak working-memory high-water mark from allocator instrumentation, and speedup ratios derived from raw cycle counts rather than rounded display values — each reported with an explicit environment label and a raw log. We would welcome the workshop conversation on standardizing low-resource metrics of this kind.

## 3a. Section 2(d) — Low Complexity of User Experience

The no-OS architecture is itself the human-interface simplification:

- **Deployment is two actions:** insert the USB stick, power on. There is no installation, no account, no network enrollment, no configuration, no driver, and no update — and therefore no step at which deployment can be done wrong. On firmware that ignores the standard removable-boot path, a fallback loader script chain-boots the engine from the EFI shell without user action.
- **Operation is one skill:** type a question, read the answer. The entire human-machine interface is a text prompt on a text console (currently English, greedy decoding). If an operator can type, they are fully trained.
- **Nothing to misconfigure under stress:** there are no menus, modes, settings, or files. The system is stateless by design — a power cycle returns it to a known-good state, and there is no state on the host to corrupt or recover.
- **Field diagnostics without tooling:** the engine appends a stage-by-stage boot flight record to its own USB stick; if a machine fails, plugging the stick into any other computer shows exactly how far boot progressed. Failure triage requires no debugger, no serial cable, and no expertise beyond reading seven numbered lines.

These properties were exercised, not assumed: the same stick, untouched, booted and ran the instrumented test sequence across the machines reported in Section 4.

## 4. Identification of Current Data (Empirical Metrics)

All numbers below are runtime-computed from real work and preserved with raw logs in the project repository; the specific log file behind each row is named in its Notes cell. The environment differs by row and is stated explicitly: bare-metal rows come from the engine's built-in instrumented test sequence (`/gauntlet`), logged to the boot volume; the cross-architecture row is a virtualized run; the quality row comes from the host-side evaluation harness running the same engine code. Speed ratios are derived from raw time-stamp-counter (TSC) tick counts recorded in the logs, not from rounded display values.

**Primary (bare-metal) environment:** Dell Inspiron 15, Intel Core i5-5200U (Broadwell, 2015), booted bare-metal from a 940 MiB USB image. No operating system present. CPU identity taken from CPUID at runtime. **Secondary (bare-metal) environment:** HP Stream, Intel Celeron N4020 (Gemini Lake, fanless), 4 GB RAM — a machine with no AVX of any kind — booted from the same USB image.

| Metric | Measured value | Notes |
|---|---|---|
| Total physical footprint | **< 1.5 GB** (782 MB read-only weights + **572 MB** peak working memory ≈ 1.35 GB) | Bare-metal Dell run. Working-memory high-water mark measured by allocator instrumentation at runtime: 571,912,597 bytes (the log prints it as 545.41 MiB = 572 MB); OS footprint = 0 bytes. Log: gauntlet_dell_i5-5200U_2026-07-12_005500.txt |
| Decode throughput | **0.61 tok/s** (single-threaded, AVX2+FMA) | Bare-metal Dell run, measured **while firmware held the core at 22% of nominal clock (~480 MHz)** — with no OS present, no software layer owns power management; the engine reads the ratio from the CPU's APERF/MPERF activity counters. Scaling by nominal clock suggests ≈4.5× headroom (a linear extrapolation, not a measurement). Log: same file |
| SIMD speedup (same binary) | **4.84×** (decode cost fell from 17.16 to 3.55 billion ticks/token, scalar vs AVX2+FMA) | Bare-metal Dell run; runtime CPU dispatch, ratio from raw TSC ticks. Log: same file |
| Batched prefill | **1.98×** over per-token prefill | Bare-metal Dell run; weight-reuse matrix-multiply (GEMM) tiling, prefill tick counts. Log: same file |
| Context-depth resilience | **0.968×** decode rate at 400 vs 20 tokens of context | Bare-metal Dell run; near-flat decode as the attention key-value (KV) cache grows (tick-derived). Log: same file |
| Cross-architecture consistency | SIMD **5.82×**, batching **2.40×** on a 2019 i5-10210U | Same experiment, second microarchitecture, run under QEMU/KVM virtualization on that host — reported as same-binary ratios, not absolute bare-metal throughput. Log: gauntlet_qemu_i5-10210U_host_v2_2026-07-10_113923.txt |
| Pre-AVX2 hardware execution | **0.17 tok/s** scalar at the firmware-default 1.1 GHz clock; **0.42 tok/s (2.43×)** after the engine's own Ring-0 P-state request raised the clock to ~245% of nominal | Bare-metal HP Celeron N4020 run — same binary, SSE2 scalar-fallback path on silicon with no AVX at all. First machine where the engine's direct clock intervention takes effect (the Dell's firmware ignores it). Peak working memory byte-identical to the Dell run: 571,912,597 bytes. Context slope 0.988×. Log: gauntlet_hp_stream_n4020_2026-07-14_141531.txt |
| Language-model quality | WikiText-2 perplexity **15.8 on the full test set** (313,479 tokens teacher-forced, 16.1 h eval run); 10.35 on a 1,899-token continuous sample | Host-side evaluation harness (aegis-eval, same engine code, Linux); pruned-vocabulary model (50,256 of 128,256 tokens, English-oriented). Full-set figure uses chunked evaluation with a cold KV cache per ~5,000-character chunk, which biases perplexity upward vs continuous context — both caveats are recorded in the raw logs. Logs: wikitext2_full_ppl_2026-07-12_run.log; wikitext2_sample1900_2026-07-14_run.log |

**Compatibility floor (assessed):** 64-bit UEFI firmware (~2012 onward, required by specification — BIOS-only machines are below the floor by design), Secure Boot disabled (one-time firmware toggle; the binary is unsigned today), and 2 GB RAM. The RAM boundary is bracketed empirically under QEMU/OVMF memory-limit runs: 2 GB boots and generates; 1.4 GB halts during weight loading with an explicit out-of-memory diagnostic — expected, since firmware, boot services, and load-time staging sit on top of the 1.35 GB steady-state payload. Physical validation of the 2 GB floor is in progress. The engine's scalar path executes on any x86-64.

## 5. Estimated Time to Availability and Risk Assessment

- **Availability:** the core engine is operational today on commodity bare metal; the evidence trail above is reproducible from public components — Microsoft's published model plus this engine. Distilled mission model, curated corpus integration, and the sub-1 GB model portfolio: estimated 6–9 months.
- **Technical risks:** (1) Firmware P-state control without an OS — the largest performance factor on legacy laptops, and a finding in its own right: with the OS deleted, no software layer owns power management, and some firmware hands off the core at its minimum clock (the 22% case above). The mitigation is already demonstrated on one machine: on the Celeron N4020, the engine's direct Ring-0 P-state request raised the clock ~2.4× (measured, Section 4); the Dell's firmware ignores the same request, so per-family handling remains engineering work. (2) Firmware diversity across OEM UEFI implementations — empirically our dominant deployment-failure class (four distinct per-machine firmware quirks required targeted fixes to date); mitigated by a QEMU firmware/media/memory test matrix, a per-machine boot-stage flight recorder for field triage, and the EFI-shell fallback loader. (3) Context-window scaling without OS paging; mitigated by measured near-flat KV-cache behavior and arena budgeting. (4) Model-quality floor for safety-relevant answers; mitigated by the planned corpus-with-provenance architecture and measured calibration rather than fluency.

---

# REFERENCES
1. Microsoft Research, *BitNet b1.58-2B-4T* (arXiv:2504.12285) and *The Era of 1-bit LLMs* (arXiv:2402.17764)
2. Microsoft Research, *BitNet Distillation* (arXiv:2510.13998)
3. TII, *Falcon-Edge: 1.58-bit language models* (falcon-lm.github.io/blog/falcon-edge)
4. Aefinity AI, *A.L.I.C.E. Technical Report & raw hardware logs*, July 2026 (git-versioned; commit hashes available on request). No prior or ongoing Government sponsorship.

---

# SUMMARY SLIDE (docs/RFI_summary_slide_v3.pdf — merged as the final page of the response PDF)
Visual strip: USB stick → UEFI boot (no OS) → ternary engine → operator Q&A, with the three strongest numbers:
1. **< 1.5 GB total footprint** (572 MB peak working memory, measured)
2. **Runs on a 2015 commodity laptop** — 0.61 tok/s while firmware held 22% clock; 4.84× SIMD speedup, tick-derived
3. **No OS, no network stack, no host persistence** — no OS to patch, no telemetry, nothing left on the host
