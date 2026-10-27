# RELATED_WORK (deliverable D2): fetched-source sweep, 2026-10-27

Method: every scored entry below carries a URL fetched in this session (tag "fetched 2026-10-27"); statements come from the fetched page text, not from recall. Where a fetch failed or nothing was found the entry says NOT FETCHED; two attestation standards were fetched only as far as HTTP 200 (TCG: response headers, content-type PDF; NIST: page title) and are listed but not scored. Axes: A1 = no operating system at run time; A2 = integer-exact / bit-identical results across ISAs; A3 = machine-checkable conformance + receipts. No numbers about ALICE or CIS-1 appear here (the claims ledger program/PAPER_CLAIMS_LEDGER.md owns them).

Alert status: no ALERT. No fetched source covers all three axes, and none contradicts the gap statement below (see coverage table). Closest on A1: Djiby Diop's UEFI project. Closest on A2+A3: EigenAI and Cankaya (GPU-only).

## Ternary/BitNet inference

- **BitNet b1.58 (Ma, 2024)** — https://arxiv.org/abs/2402.17764 (fetched 2026-10-27). What it does: introduces an LLM with weights in {-1, 0, 1} and reports parity with full-precision Transformers in perplexity and tasks with lower latency, memory and energy. Relation: A1 no (model, not a runtime); A2 no (no bit-exact contract); A3 no.
- **BitNet b1.58 2B4T Technical Report (Ma, 2025)** — https://arxiv.org/abs/2504.12285 (fetched 2026-10-27). What it does: describes a 2B-parameter ternary model trained natively on 4T tokens. Relation: A1 no; A2 no; A3 no.
- **BitNet b1.58 2B4T model card (Microsoft, HF)** — https://huggingface.co/microsoft/bitnet-b1.58-2B-4T (fetched 2026-10-27). What it does: hosts the packed 1.58-bit weights under the MIT licence (card lists ~2B parameters, 4T training tokens) and directs users to the bitnet.cpp runtime. Relation: A1 no; A2 no (card makes no determinism statement); A3 no.
- **1-bit AI Infra Part 1.1 (Wang, 2024)** — https://arxiv.org/abs/2410.16144 (fetched 2026-10-27). What it does: presents bitnet.cpp CPU kernels for lossless-by-the-authors'-definition BitNet b1.58 inference with reported x86 and ARM speedups. Relation: A1 no (runs under a host OS); A2 no (speed claim, no cross-ISA bit-identity contract); A3 no.
- **Bitnet.cpp: Efficient Edge Inference for Ternary LLMs (Wang, 2025)** — https://arxiv.org/abs/2502.11880 (fetched 2026-10-27). What it does: presents an edge ternary-LLM inference system with lookup-table and 2-bit-scaled matmul kernels. Relation: A1 no; A2 no; A3 no.
- **microsoft/BitNet (bitnet.cpp repo)** — https://github.com/microsoft/BitNet (fetched 2026-10-27). What it does: MIT-licensed inference framework for 1-bit LLMs with CPU and GPU kernels, built on llama.cpp. Relation: A1 no (needs Python, CMake, clang, a host OS); A2 no (ships x86 and ARM kernels; the README defines "lossless" without claiming bit-exact or cross-ISA-identical output); A3 no.

## Deterministic and bit-exact inference

- **Impacts of floating-point non-associativity on reproducibility (Shanmugavelu, 2024)** — https://arxiv.org/abs/2408.05148 (fetched 2026-10-27). What it does: measures run-to-run variability from non-associativity in HPC and deep learning and evaluates deterministic GPU alternatives and a deterministic accelerator. Relation: A1 no; A2 partial (analyses and mitigations, no cross-ISA identity contract); A3 no.
- **Defeating Nondeterminism in LLM Inference (He, Thinking Machines Lab, Sep 2025 per the page)** — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ (fetched 2026-10-27). What it does: attributes run-to-run nondeterminism in LLM serving to batch-size-dependent kernel reductions and presents batch-invariant GPU kernels; the page reports 1000 temperature-0 completions of one prompt becoming identical once those kernels are enabled, within one serving setup. Relation: A1 no; A2 partial (bitwise-identical results within one GPU setup; the page itself notes that CPU kernels mixing vectorized and non-vectorized intrinsics need not agree bitwise; no cross-ISA claim); A3 no.
- **I-BERT (Kim, 2021)** — https://arxiv.org/abs/2101.01321 (fetched 2026-10-27). What it does: integer-only BERT inference with integer approximations of nonlinear layers. Relation: A1 no; A2 partial (integer arithmetic, no stated cross-ISA identity or frozen rounding spec); A3 no.
- **I-ViT (Li, 2022)** — https://arxiv.org/abs/2207.01405 (fetched 2026-10-27). What it does: integer-and-shift-only Vision Transformer inference with integer Softmax, GELU and LayerNorm. Relation: A1 no; A2 partial (integer arithmetic only); A3 no.
- **Taming the Exponential (Danopoulos, 2026)** — https://arxiv.org/abs/2604.02292 (fetched 2026-10-27; abstract page only). What it does: clipped-linear softmax surrogate for int8 hardware (AMD Versal AI Engines). Relation: A1 no; A2 partial (integer-native op, single target); A3 no.
- **Bare-Metal Tensor Virtualization (Kilictas, 2026)** — https://arxiv.org/abs/2601.03324 (fetched 2026-10-27; abstract page). What it does: software-only "virtual tensor core" on Apple Silicon using memory mapping and SIMD kernels; abstract describes the implementation as "deterministic". Relation: A1 no (uses mmap, so a host OS, despite "bare-metal" in the title); A2 partial (determinism claimed on one ARM64 platform; no cross-ISA claim in the abstract); A3 no.
- **Bit-Exact AI Inference Verification Without Performance Tradeoffs (Cankaya, 2026)** — https://arxiv.org/abs/2606.00279 (fetched 2026-10-27). What it does: argues vLLM and HF Transformers can be deterministic without costly flags if information is recorded and no atomics are used, and shows by software emulation that bitwise re-computation does not need identical hardware across NVIDIA GPU variants. Relation: A1 no (not mentioned); A2 partial (several NVIDIA GPUs, not different ISAs); A3 partial (verification by recompute and sampling; no receipt or conformance digest on the abstract page).
- **EigenAI (Alves, 2026)** — https://arxiv.org/abs/2602.00182 and full text https://arxiv.org/html/2602.00182v1 (fetched 2026-10-27). What it does: bit-exact LLM inference on fixed GPU architectures with operator-signed receipts posted to EigenDA and optimistic re-execution in a TEE. Relation: A1 no (containerized runtime on a host OS, TEE enclaves); A2 partial (the text states determinism "holds only within fixed GPU families"; CPU ISAs not addressed); A3 partial (signed receipts and byte-equality checks; no cross-implementation conformance digest identified in the fetched text).

## Verifiable inference (proofs, receipts, attestations)

- **DiFR (Karvonen, 2025)** — https://arxiv.org/abs/2511.20621 (fetched 2026-10-27). What it does: verifies LLM inference against a trusted reference run via token checks and compressed activation fingerprints despite nondeterminism. Relation: A1 no; A2 no (tolerance-based); A3 partial (verification procedure, statistical rather than exact).
- **zkLLM (Sun, 2024)** — https://arxiv.org/abs/2404.16109 (fetched 2026-10-27). What it does: zero-knowledge proof system for LLM inference with a parallel lookup argument and an attention-specific proof. Relation: A1 no; A2 no (not addressed on the abstract page); A3 partial (cryptographic proof of computation, heavy prover cost, no conformance digest).
- **ZKML survey (Peng, 2025)** — https://arxiv.org/abs/2502.18535 (fetched 2026-10-27). What it does: surveys zero-knowledge verifiable ML across training, testing and inference. Relation: A1 no; A2 no; A3 partial (survey of proof approaches).
- **Orion (Ebel, 2023)** — https://arxiv.org/abs/2311.03470 (fetched 2026-10-27). What it does: compiles PyTorch networks into fully-homomorphic-encryption programs. Relation: A1 no; A2 no; A3 no (privacy of inputs, not a receipt).
- **zkRNN (Zarinjouei, IACR ePrint 2026/073)** — https://eprint.iacr.org/2026/073 (fetched 2026-10-27). What it does: zero-knowledge proofs of RNN inference by a recursive sum-check extension of GKR. Relation: A1 no; A2 no; A3 partial (proof of computation for RNNs, not Transformers or a conformance digest).
- **CommitLLM (LambdaClass, repo)** — https://github.com/lambdaclass/CommitLLM (fetched 2026-10-27). What it does: MIT-licensed commit-and-audit protocol where a provider returns a compact receipt and a CPU verifier checks opened traces with Freivalds-style randomized checks and canonical re-execution. Relation: A1 no (no OS statement; tested on Modal A100 GPUs); A2 no (README does not say whether verification works across hardware; attention is audit-only, not exact); A3 partial (receipt plus verifier, but README lists a known residual forgery hole and no frozen conformance digest was found).
- **Seal, Then Sample (Lim, 2026)** — https://arxiv.org/abs/2609.27367 (fetched 2026-10-27). What it does: sampled layerwise proofs committing to activations per chunk and proving a verifier-chosen subset. Relation: A1 no; A2 no (reports argmax agreement with a floating-point reference, i.e. not exact); A3 partial (commitments and sampled proofs).
- **MPROOF** — NOT FETCHED: two web searches ("MPROOF ChunkFP verifiable LLM inference arXiv"; "MPROOF verifiable inference merkle proof model; ChunkFP ...") returned no document with that name. No entry is made from memory; the closest hits (Seal, Then Sample; TensorCommitments arXiv 2602.12630; TOPLOC arXiv 2501.16007) were not individually fetched, except Seal, Then Sample.
- **ChunkFP** — NOT FETCHED: same searches, no document with that name found.
- **TCG attestation overview (v1.0 rev. 1.37, Feb 2024 per the file name) and NIST SP 800-155 (BIOS Integrity Measurement Guidelines, initial public draft)** — https://trustedcomputinggroup.org/wp-content/uploads/Overview-of-TCG-Technologies-for-Device-Identification-and-Attestation-Version-1.0-Revision-1.37_5Feb24-2.pdf and https://csrc.nist.gov/pubs/sp/800/155/ipd (fetched 2026-10-27 only as far as HTTP 200: the TCG PDF by response headers, content-type application/pdf, no body read; the NIST page by its title "SP 800-155, BIOS Integrity Measurement Guidelines"; neither body read this session). What they are, per docs/paper/02_related_work.md and not per a body read here: platform attestation of boot stages (measured boot, TPM), not verification of model output. Relation: not scored; carried over since that prior draft cites them.

## Bare-metal and unikernel execution

- **OO / llm-baremetal (Diop, HF model card)** — https://huggingface.co/djibydiop/llm-baremey (fetched 2026-10-27). What it does: freestanding C (no libc) x86_64 UEFI project running Mamba-2.8B and LLaMA2 (GGUF Q4/Q8) inference with a self-test command and a seed command; page carries a self-declared priority claim that this entry does not assess, no licence field in the card metadata, and a "tok/s" sample run. Relation: A1 covers (UEFI, no OS, per the page); A2 partial (a seed command implies reproducible sampling; no integer-exact or cross-ISA claim, x86_64 only); A3 partial (a self-test command exists; no digest, spec or receipt described). The GitHub repo links https://github.com/Djiby-diop/llm-baremetal and https://github.com/LLM-Baremetal both returned HTTP 404 (NOT FETCHED); claims rest on the HF page only.
- **Bare-Metal Tensor Virtualization (Kilictas, 2026)** — see "Deterministic and bit-exact inference" above; A1 no.
- **unikraft/app-llama2-c** — https://github.com/unikraft/app-llama2-c (fetched 2026-10-27). What it does: MIT-licensed bootable Llama 2 on the Unikraft unikernel, served over HTTP, built on lib-musl. Relation: A1 partial (a unikernel is a library OS linked into the image, with musl libc and a virtual machine monitor; not UEFI firmware-only); A2 no (no determinism claim); A3 no.
- **Unikraft (Kuenzer, EuroSys 2021)** — https://arxiv.org/abs/2104.12721 (fetched 2026-10-27); code https://github.com/unikraft/unikraft (fetched 2026-10-27, BSD-3-Clause, x86 and ARM). What it does: modular micro-library OS for building specialized unikernels. Relation: A1 partial (specialised OS linked into the image, not an absence of OS code); A2 no; A3 no.
- **Unikernels: Library Operating Systems for the Cloud (Madhavapeddy, ASPLOS 2013)** — https://mirage.io/papers (listing) and the PDF it links, http://anil.recoil.org/papers/2013-asplos-mirage.pdf (both fetched 2026-10-27; the PDF front page reads "ASPLOS'13, March 16-20, 2013, Houston, Texas, USA"). What it does: introduces single-purpose, compile-time-specialised unikernels (MirageOS). Relation: A1 partial (library OS); A2 no; A3 no.
- **OSv (Kivity, USENIX ATC 2014)** — https://www.usenix.org/conference/atc14/technical-sessions/presentation/kivity (fetched 2026-10-27). What it does: guest OS for single-application cloud VMs that runs existing Linux applications. Relation: A1 partial (a guest OS remains present); A2 no; A3 no.
- **IncludeOS** — https://github.com/includeos/IncludeOS (fetched 2026-10-27). What it does: Apache-2.0 (LICENSE file) C++ library OS linked into a service; the README names x86_64 targets only and bundles musl libc and virtio drivers. Relation: A1 partial (library OS under a hypervisor); A2 no (x86_64 only); A3 no.
- MirageOS note: the Madhavapeddy entry is the MirageOS paper. Nanos was not searched.

## Edge inference stacks

- **llama.cpp** — https://github.com/ggml-org/llama.cpp (fetched 2026-10-27). What it does: MIT-licensed C/C++ LLM inference across x86, ARM, RISC-V CPUs and many GPU/NPU backends. Relation: A1 no (host OS); A2 no (README makes no determinism statement); A3 no.
- **ExecuTorch** — https://github.com/pytorch/executorch (fetched 2026-10-27). What it does: PyTorch on-device runtime with CPU, GPU, NPU and DSP delegates (Qualcomm, MediaTek, Samsung, Arm Ethos-U, NXP). Relation: A1 partial (lists embedded and microcontroller targets, no OS-free statement); A2 no (README says to validate numerics on the target); A3 no.
- **TensorFlow Lite Micro** — https://github.com/tensorflow/tflite-micro (fetched 2026-10-27). What it does: Apache-2.0 runtime for bare-metal systems, DSPs and microcontrollers. Relation: A1 partial (README names bare-metal targets; no explicit OS-free claim; no LLM-scale claim); A2 no (no bit-exact statement on the page); A3 no.
- **OpenVINO** — https://github.com/openvinotoolkit/openvino (fetched 2026-10-27). What it does: Apache-2.0 toolkit for deploying models on Intel CPU, GPU and NPU. Relation: A1 no; A2 no; A3 no.
- **AMD Ryzen AI Software** — https://ryzenai.docs.amd.com/en/latest/ (fetched 2026-10-27). What it does: runtime for running models on the Ryzen AI NPU via ONNX Runtime. Relation: A1 no; A2 no (no determinism statement on the page); A3 no.
- **Qualcomm AI Engine Direct (QNN) SDK** — NOT FETCHED: fetch of https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/introduction.html returned only the title "Qualcomm Documentation". ExecuTorch's README (above) is the only fetched source mentioning Qualcomm NPU delegation.

## Gap statement

No system in our sweep combines (1) execution with no operating system at run time, (2) integer-exact results that are bit-identical across instruction-set architectures, and (3) machine-checkable conformance plus a verifiable decode receipt. The sweep is limited to the sources listed above; MPROOF and ChunkFP were not found, the Qualcomm SDK page did not load, and the UEFI project's source repository was not reachable.

| Entry | A1 no-OS | A2 bit-identical across ISAs | A3 conformance + receipt |
|---|---|---|---|
| BitNet b1.58 / 2B4T report / model card | no | no | no |
| 1-bit AI Infra, bitnet.cpp paper, microsoft/BitNet | no | no | no |
| Shanmugavelu 2024, He (Thinking Machines) | no | partial: analysis, mitigations, batch-invariant GPU kernels; no cross-ISA contract | no |
| I-BERT, I-ViT, Taming the Exponential | no | partial: integer ops, no cross-ISA contract | no |
| Bare-Metal Tensor Virtualization | no | partial: "deterministic" claim, one ARM64 platform | no |
| Cankaya, Bit-Exact AI Inference Verification | no | partial: across NVIDIA GPUs, not ISAs | partial: recompute audit, no receipt on abstract page |
| EigenAI | no | partial: fixed GPU families only | partial: signed receipts, no conformance digest found |
| DiFR | no | no | partial: statistical verification |
| zkLLM, ZKML survey, zkRNN, Seal Then Sample | no | no | partial: proofs of computation, no conformance digest |
| Orion | no | no | no |
| CommitLLM | no | no | partial: receipt + verifier, known residual hole, no frozen digest |
| MPROOF, ChunkFP | not fetched | not fetched | not fetched |
| TCG attestation overview, NIST SP 800-155 | not scored: boot-chain attestation, bodies not read | not scored | not scored |
| OO / llm-baremetal (UEFI) | covers | partial: seed command, x86_64 only | partial: self-test only |
| unikraft/app-llama2-c, Unikraft, MirageOS, OSv, IncludeOS | partial: library or guest OS present | no | no |
| llama.cpp, OpenVINO, AMD Ryzen AI | no | no | no |
| ExecuTorch, TFLite Micro | partial: embedded targets | no | no |
| Qualcomm QNN | not fetched | not fetched | not fetched |

## Sources

All URLs above were fetched 2026-10-27 unless the entry says NOT FETCHED. TCG (headers only) and NIST (page title only) were fetched as far as HTTP 200 and are not scored. MPROOF and ChunkFP have no URL, as no document was found.
