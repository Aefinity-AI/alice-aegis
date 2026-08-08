# A.L.I.C.E. — Aegis Lightweight Inference Core Engine

A **bare-metal large-language-model inference engine**: a from-scratch, `no_std`
Rust implementation of the BitNet b1.58 ternary transformer that boots directly
from UEFI firmware with **no operating system**, plus a matching Linux userspace
harness. Single-purpose, offline, auditable — the entire inference stack is
~1,600 lines of Rust with `libm` and `serde` as the only runtime dependencies.

**Author:** Justin B. Thompson · **Model weights:** [Microsoft BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) (MIT license — see `MICROSOFT_MODEL_CARD.md`; the ternary quantization is Microsoft's work, the inference engine and unikernel are original)

## Verified results (Intel i5-10210U; 2026-07-09 unless noted)

All numbers below are measured, with the producing command and log noted.
Nothing in this table is estimated or simulated.

| Measurement | Result | Evidence |
|---|---|---|
| Coherent generation, Linux userspace | "The capital of France is **Paris**." + fluent paragraphs | `aegis-linux` runs |
| Coherent generation, **bare-metal (no OS)** | Same output, booted via OVMF/QEMU-KVM | `aegis-uefi/qemu_success_2026-07-09.log` |
| Decode speed, single thread | **2.80 tok/s** | `docs/hardware_logs/energy_run_i5-10210U_2026-07-14.log:12` (on battery, f32-activation build) |
| Decode speed, multicore | **4.94 tok/s (1.76×)** | `docs/hardware_logs/energy_run_i5-10210U_multicore_2026-07-14.log:12` (on battery, `parallel`+`int8_act`) |
| Prefill, batched GEMM vs per-token | 1.63–1.75× (single thread, 351-token prompt) | A/B in both orders; recorded in the working history, not reproducible from this snapshot |
| Decode speed, SSE2 scalar fallback | ~3.6 B cycles/token | `aegis-uefi/matrix_logs/Nehalem_q35_sata.log` (QEMU-emulated CPU — correctness only) |
| **Energy per token, incremental** | **3.99 J** (19.69 W over idle), band 3.99–5.06 J | `docs/hardware_logs/energy_run_i5-10210U_multicore_2026-07-14.log`, 385 s window |
| **Energy per token, total system** | **4.99 J** (24.65 W draw) | same run |
| Idle system power | 4.96 W sampled (coulomb cross-check 4.42 W — 11% apart) | same run |
| CPU compatibility matrix (QEMU-emulated) | Nehalem ✅ SandyBridge ✅ Haswell ✅ modern ✅ (auto SIMD dispatch; QEMU/KVM CPU feature masking — emulated, not physical hardware) | `aegis-uefi/test_matrix.sh`, logs in `aegis-uefi/matrix_logs/` |
| USB (XHCI) boot path | ✅ boots and generates | matrix case `host_q35_usb` |
| WikiText-2 perplexity (pruned-vocab model) | **15.825** full test set (teacher-forced, 313,479 tokens, chunked 5,000-char cold-KV mode — biases upward vs continuous context; 16.1 h); continuous 1,900-token sample anchor **10.35** (10.348 logged 2026-07-14; 10.394 on the 2026-07-12 multicore build — delta is FP summation order). Earlier 12.738 / 12.801 figures superseded (predate tokenizer fixes; the 12.8-era sample file was deleted, so they are unreproducible) | `docs/hardware_logs/wikitext2_full_ppl_2026-07-12_run.log`, `docs/hardware_logs/wikitext2_sample1900_2026-07-14_run.log` |

## CIS-1: bit-identical inference on any hardware

**[`docs/CIS-1_SPEC_v1.0.md`](docs/CIS-1_SPEC_v1.0.md)** — a frozen,
normative integer semantics for transformer inference whose output is
bit-identical on any conforming implementation, any ISA. Conformance is
machine-checkable: two pinned digests (operation-level and a complete
64-token decode) that this repo's CI re-proves on x86-64 and aarch64 on
every commit. The op-level digest has been reproduced by this one
implementation, unmodified, on two ISAs across six execution environments —
including physical AVX2 and SSE2-class machines booted into a minimal-Linux
initramfs; the token-level digest on x86-64 and aarch64 CI hosts. Speed,
honestly: at kernel level, at parity SIMD, the integer path measured 2.94×
*faster* than the hand-tuned float AVX2 kernel on the Dell i5-5200U (ledger
A27); scalar-vs-scalar it measured 1.248× slower on that same machine and
0.961× — faster — on an HP Celeron N4020 (A26). Token-level throughput is
not yet measured.

Think the bit-identical claim can't survive contact with your hardware?
**[CHALLENGE.md](CHALLENGE.md)** — find a machine where it diverges and get
paid for the finding.

## Components

| Directory | What it is |
|---|---|
| `aegis-core/` | The engine: `no_std` BitNet forward pass — 2-bit-packed ternary matvec (AVX2 LUT kernel + portable scalar fallback, runtime CPUID dispatch), GQA attention with KV cache, RoPE, squared-ReLU FFN, SubLN, BF16 tied-embedding LM head, BPE tokenizer, teacher-forced perplexity |
| `aegis-uefi/` | The unikernel: UEFI app that enables AVX at firmware level (CR4/XCR0), loads weights from FAT32 through a 64 KB DMA bounce buffer, runs an interactive REPL. Writes `BOOTLOG.TXT` stage checkpoints to the boot volume for post-crash forensics |
| `aegis-forge/` | Model prep: vocabulary pruning (128,256 → 50,256 tokens incl. all specials) + BF16 embedding slicing |
| `aegis-linux/` | Linux userspace harness (same engine, fast dev loop) |
| `aegis-eval/` | Perplexity evaluator (real measurement; no mock values) |

## Build & run

```bash
# Userspace (any Linux, stable Rust)
cd aegis-linux && cargo build --release
./target/release/aegis-linux ~/aegis_pruned_model.safetensors \
    ~/aegis-forge/embed.bin ~/aegis-forge/vocab.bin 64 "Your prompt here"

# Unikernel (requires nightly + rust-src; see script header for why)
cd aegis-uefi && ./build_hardfloat.sh
mcopy -o -i ~/aegis-boot.img target/x86_64-uefi-hardfloat/release/aegis-uefi.efi ::/EFI/BOOT/BOOTX64.EFI

# Boot in QEMU
qemu-system-x86_64 -enable-kvm -nographic -bios /usr/share/ovmf/OVMF.fd \
    -m 2G -machine q35 -cpu host -drive file=aegis-boot.img,format=raw

# Hardware-compatibility matrix (no USB burn cycles needed)
cd aegis-uefi && ./test_matrix.sh

# Real hardware: dd aegis-boot.img to a USB stick, boot via UEFI.
# If it fails, read BOOTLOG.TXT off the stick to see the last completed stage.
```

**Important build note:** the stock `x86_64-unknown-uefi` Rust target is
*soft-float* — built that way, the binary contains zero SIMD instructions and
runs ~100× slower. Always build the unikernel via `build_hardfloat.sh`, which
uses the custom hard-float target spec and verifies FMA instructions are
present in the output.

## Honest limitations

- **The quantization is not ours.** Weights arrive pre-ternarized from
  Microsoft. `aegis-forge` prunes vocabulary; it does not quantize.
- **Activations follow the reference (int8 grid) but the kernel is still f32.**
  Quantization is applied as a quantize→dequantize round trip, which is
  numerically identical to an integer kernel and gives the quality benefit. An
  actual integer kernel would need `vpdpbusd` (AVX-VNNI) to beat the current
  f32 LUT+FMA path — three alternatives were built and measured, and all lost.
  See `aegis-core/benches/lut_mpgemm.rs`.
- **Greedy decoding only** (argmax + repetition penalty); temperature/top-p are
  stubs.
- **Pruned ASCII-oriented vocabulary** (50,256 of 128,256 tokens): non-English
  and emoji-heavy text will tokenize and score worse than the full model.
  Perplexity numbers must carry this caveat.
- **2,048-token context.** Measured footprint (Dell i5-5200U bare-metal):
  ~782 MB read-only weights + ~572 MB peak working memory ≈ 1.35 GB total;
  compatibility floor **2 GB RAM** for the 2B pruned model (QEMU: 2 GB boots
  and generates, 1.4 GB fails with a clean OOM panic).
- **Multicore only in userspace.** The `parallel` feature (row-parallel kernels
  over a persistent worker pool) requires std; the UEFI unikernel stays
  single-threaded until it drives the firmware's MP Services protocol.
  Threads default to physical cores — SMT siblings contend for the FMA ports
  and measure slightly *slower*. Override with `AEGIS_THREADS`.
- A 2B-parameter model is factually unreliable; treat outputs accordingly.

## Provenance

Model: BitNet b1.58-2B-4T (Microsoft Research, MIT). Boot/runtime, kernels,
unikernel, and evaluation harness: original work in this repository. The
tokenizer *implementation* is original too — but its *data* is not: the
vocabulary and all 280,147 merges are the Meta Llama 3 tokenizer, carried into
this repository through Microsoft's model files. Microsoft's own model card
states it: *"Tokenizer: LLaMA 3 Tokenizer (vocab size: 128,256)."*

Historical documents in older backups contain unverified or simulated
performance figures; **only the numbers in the table above, each traceable to
a log in this repository, should be cited.**

## License

This project's original work is licensed under the **Apache License, Version
2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Copyright 2026 Justin B. Thompson / Aefinity AI.

Third-party material redistributed here — Microsoft's BitNet model files and
the artifacts derived from them, HuggingFace's `modeling_bitnet.py`, ARIS's
`evidence_check.py`, and the ARC and WikiText-2 evaluation data — retains its
own licenses and is **not** covered by the Apache-2.0 grant. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete
file-by-file accounting, including the source-availability pointer required for
the MPL-2.0 `ucs2` crate linked into the published `.efi` binaries.

## Acknowledgments

A special thank you to **Linda Blanchard**, **Charles Seaman** and the **Seaman family**, as well as **Bonnie Power** for their unwavering support in making this mission possible.
