# A.L.I.C.E. Unikernel — Full Technical Audit

**Scope:** `aegis-core` (inference), `aegis-forge` (vocab/embedding pipeline), `aegis-uefi` (bare-metal boot), `aegis-eval` (perplexity), plus the vendored `rusty-loader` bootloader.
**Target model:** BitNet b1.58-2B-4T (native ternary), verified against Microsoft's reference config.
**Goals assessed:** post-quantization coherence · inference speed · boot compatibility · grant-review readiness · honest overall assessment.

---

## 0. Verdict up front

This is a genuinely impressive piece of systems work: a real, `no_std`, bare-metal LLM that boots on UEFI with hand-written AVX2 ternary kernels and a custom physical-memory allocator. The architecture is faithfully implemented and the UEFI hardware debugging is real engineering.

But two things need to be fixed before this is either *coherent* or *review-ready*:

1. **A correctness bug that corrupts every prompt: the prefill path reads embeddings as BF16 (2-byte stride) while the actual `EMBED.BIN` and the decode path are F32 (4-byte stride).** This alone explains any "the output is subtly wrong / drifts after the prompt" behavior.
2. **The headline benchmark and perplexity numbers are hard-coded mock values, explicitly labeled "for DARPA review."** Shipping those to a review board is the single biggest risk in this project and must be replaced with real measurements.

Details, severity, and fixes below.

---

## 1. Coherence audit (post-quantization correctness)

### 1.1 — CRITICAL: prefill vs decode embedding dtype mismatch

The same `self.pipeline.embeddings` buffer is read with two different element sizes:

- **Prefill** (`forward_batch`, inference.rs ~line 282):
  `let start = current_tok as usize * emb_dim * 2; // BF16 is 2 bytes` → reads 2-byte stride, reconstructs BF16.
- **Decode** (`forward_step`, ~line 408):
  `let start = current_tok as usize * emb_dim * 4;` → reads 4-byte stride as F32.
- **LM head** (`f32_dot_avx2`, ops.rs ~line 1094):
  `row * emb_dim * 4`, dereferenced as `*const f32` → F32.

Your shipped `EMBED.BIN` is **513,935,360 bytes = 50,189 × 2560 × 4**, i.e. **F32**. So decode and the LM head are correct; **prefill is wrong**. In prefill, element *i* is read from byte offset `token*5120 + i*2`, which lands in the *middle* of real 4-byte floats. The entire prompt is therefore embedded from garbage, propagated through all 30 layers, written into the KV cache, and the first generated token is conditioned on a corrupted context.

Because only the **last** token's hidden state survives prefill (inference.rs ~line 526) and decode then uses the correct F32 path, the model won't crash — it will just produce **incoherent or prompt-ignoring output**, which is exactly the failure mode you'd describe as "not coherent after quantization."

**Fix (make prefill match decode/LM-head):**
```rust
// forward_batch, replace the BF16 block with F32:
let start = current_tok as usize * emb_dim * 4;
for i in 0..emb_dim {
    let o = start + i * 4;
    let f = f32::from_le_bytes([
        self.pipeline.embeddings[o],   self.pipeline.embeddings[o+1],
        self.pipeline.embeddings[o+2], self.pipeline.embeddings[o+3],
    ]);
    self.arena.batch_hidden_state[hidden_offset + i] = f;
}
```
Then add a **one-time invariant check** at engine init so this can never silently diverge again:
```rust
debug_assert_eq!(embeddings.len(), vocab_size * hidden_size * 4,
    "EMBED.BIN must be F32: got {} bytes", embeddings.len());
```
This is the highest-value change in the entire audit.

### 1.2 — "Aegis quantization" is vocabulary pruning, not quantization

`aegis-forge` prints `"Ternary Quantization (1.58-bit) Pipeline Completed Successfully"` but performs **no quantization**. It (a) keeps the first 50k tokens + specials (`vocab_stripper.rs`), (b) slices matching embedding rows, (c) writes `vocab.bin` / `embed.bin` / pruned config. The ternary weights come **pre-quantized from Microsoft**; you never quantize anything.

This matters for two reasons: it's a factual misstatement in code that a reviewer will read, and it means your "coherence after quantization" question is really "coherence after **vocabulary pruning** + faithful re-implementation." Rename the stage to what it is (e.g. "Aegis Vocabulary Pruning & Embedding Slicer") and drop the "1.58-bit quantization" claim from the forge output.

### 1.3 — Vocabulary pruning: correct, but fragile and lossy

The row-alignment logic is actually **sound**: `keep_indices` is built in old-id-sorted order, embeddings are sliced in that same order, and `vocab.bin` is emitted sorted by new id (which equals old-id order for kept tokens). Because embeddings are **tied** (`tie_word_embeddings: true`), the same pruned matrix correctly serves both input embedding and the LM head. No off-by-one there.

Caveats:
- **Special-token IDs are remapped.** LLaMA-3 specials live at 128000–128255; after pruning they move to ~50000+. That's internally consistent *here*, but anything comparing against the stock tokenizer (your perplexity harness, external eval) must use the pruned mapping or numbers will be meaningless.
- **Non-ASCII / multilingual capability is destroyed** by construction. Fine for an English demo; state it explicitly so it's a scoping decision, not a surprise regression.
- The keep-rule `token.starts_with("<")` is broad and will retain any ordinary token beginning with `<`. Harmless, but not what the comment implies.

### 1.4 — Architecture fidelity (verified against reference): good

Confirmed correct vs Microsoft's config: `rope_theta = 500000`, ReLU² FFN gate (`relu2`), SubLN (`attn_sub_norm` / `ffn_sub_norm`), GQA 20 query / 5 KV heads, head_dim 128, intermediate 6912, 30 layers, per-tensor `weight_scale` dequant, tied embeddings for the LM head. This is a faithful re-implementation — credit where due.

One deliberate deviation, **not a bug**: you keep activations in **f32** rather than the reference's int8 (W1.58**A8**). That's *higher* precision than reference, so per-token fidelity should be ≥ reference for retained tokens. It does forfeit the int8 speed path (see §2).

### 1.5 — Minor coherence / correctness nits

- **Dead double-argmax** (inference.rs ~lines 557 & 574): you argmax, then apply the repetition penalty, then argmax again. The first result is overwritten and wasted. Delete the first call.
- **Repetition penalty is nonstandard**: it both scales (`/1.2` or `*1.2`) *and* subtracts a flat `2.0`, applied to already-generated tokens only (not the prompt). It works, but it's an idiosyncratic formula — document it or move to standard HF-style penalty so reviewers recognize it.
- **`Sampler` is argmax-only**; `temperature`/`top_p` are stored but unused. Greedy decoding is fine for a deterministic demo — just don't describe it as top-p.
- **`BitNetModel::new`** hard-codes `max_position_embeddings: 1024` and other dims, but the live path uses `ModelConfig` from JSON, so this struct is dead. Remove it to avoid a reviewer thinking 1024 is your real context.

---

## 2. Speed audit

The kernels that exist are good; the losses are structural.

### 2.1 — Dominant cost: full-vocab F32 LM head **every** token
`f32_dot` computes `vocab × emb_dim` MACs per token (≈ 50,189 × 2560 ≈ 128M MACs/token) in **F32**, and it runs on every decode step. On a ternary 2B model this is very likely your single largest per-token cost.
- Quick win: the LM head is tied to embeddings — you already pay to store them. Consider **BF16 embeddings + on-the-fly widen** in the dot product to halve memory bandwidth (bandwidth, not flops, is your limiter here).
- Bigger win: you rarely need all 50k logits. For greedy decode you only need the argmax; you can't skip the matmul entirely, but you *can* fuse the argmax into the dot loop (no `-inf` fill, no second pass) and drop the separate `logits.fill(-INF)`.

### 2.2 — Scalar attention inner loops
The QK° and °V loops (`for d in 0..head_dim { ... }`, inference.rs ~lines 338–352 and 445–458) are scalar. head_dim is 128 → trivially vectorizable with the same `_mm256_fmadd_ps` + horizontal-sum pattern you already use in `ternary_matvec`. At long context this is O(seq²·head_dim) scalar work.

### 2.3 — No int8 activation path
You multiply f32 activations by ternary weights. A true BitNet fast path quantizes activations to int8 per-token and uses integer accumulation (the whole point of ternary). You've chosen accuracy over speed — that's defensible, but if "fastest it can be" is a hard requirement, the int8 GEMM is where the order-of-magnitude lives. This is a large change; scope it as a separate milestone.

### 2.4 — Smaller wins
- **F32 embeddings double the file** (514MB vs 257MB BF16). On a RAM-constrained bare-metal box, that's real; BF16 storage + widen-on-load recovers 257MB.
- **Per-element scalar BF16→F32** embedding conversion in the (now-fixed) lookup — once it's F32 you can `copy_from_slice` / bulk-load instead of element-by-element.
- **`init_unpack_lut` uses `static mut` + bool flag** — fine single-threaded, but it re-checks the flag on every matvec call. Initialize once at engine construction.
- The **4-row-unrolled, LUT-based ternary matvec is genuinely good** — keep it. Main headroom there is prefetching weight rows and widening to 8-row unroll if register pressure allows.

### 2.5 — Benchmark methodology
The UEFI `/benchmark` (main.rs ~line 3741) is *real* (rdtsc-based), but it assumes a fixed **2.5 GHz** to convert cycles→seconds. TSC frequency ≠ core clock and isn't constant. Read the actual TSC frequency (CPUID leaf 0x15) or report cycles/token only. The `aegis-eval` perplexity and the `antigravity` CLI `/status` + `/benchmark` are **fabricated** — see §4.

---

## 3. Boot compatibility audit

This is the strongest part of the project. The UEFI work reflects real hardware bring-up.

### 3.1 — Done right
- **Boot-device handle via `LoadedImage.device()`** (main.rs ~line 3610) instead of blindly taking the first `SimpleFileSystem` — the correct fix for the NVMe-enumerates-first problem. Good.
- **64KB DMA bounce buffer, over-allocated to 128KB then aligned** so a single transfer never crosses a 64KB boundary (XHCI bulk limit) — correct, and the alignment math is sound (worst case still fits the 64KB slice inside 128KB).
- **`MaxAddress(0xFFFFFFFF)`** for the bounce buffer to keep DMA under 4GB — correct instinct.
- **Custom physical allocator** scanning `EfiConventionalMemory` and locking via `AllocateType::Address` to dodge a broken firmware `AnyPages` — pragmatic and effective.
- **Watchdog disabled + petted during load**, uppercase FAT32 8.3 names, explicit `file.close()` — all sensible bare-metal hygiene.

### 3.2 — Boot robustness risks
- **Hard-coded exact file sizes** (`model_size = 522831576`, etc., main.rs ~line 3625). Any re-forge changes these and boot dies with `size mismatch`. Read `FileInfo.file_size()` (you already fetch it in `load_file_into`) and allocate from that. This is the #1 boot-fragility item.
- **`AllocateType::Address(desc.phys_start)`** grabs a conventional region at its exact base. UEFI may itself be using low conventional pages; blindly locking the first fitting region can collide. Prefer scanning for the *largest* region and, ideally, skipping the lowest 1–2MB. Add a fallback loop over multiple candidate regions before panicking.
- **No CPUID guard before `xsetbv`** (main.rs ~line 3567). If firmware ever runs you on a CPU without XSAVE/AVX, `xgetbv`/`xsetbv` `#UD`s at boot. Check CPUID.1:ECX.OSXSAVE/AVX and CPUID.7 before enabling. Cheap insurance for "highly boot compatible."
- **Three 1-second `stall`s** between files (~3s of pure delay). If they're load-bearing for USB state-machine flush, keep them but comment *why* with the specific controller; otherwise they're just slow boot.
- **Single global allocator lock across 16 heaps** iterated linearly on every `alloc`/`dealloc` — fine for single-core, but `dealloc` walks all heaps doing range checks. Not a correctness issue; note it if you ever go multicore.

### 3.3 — Portability
Boot is x86_64/UEFI-only in practice. The vendored `rusty-loader` carries aarch64/riscv64 paths, but your `aegis-uefi` (AVX2 enable, `_rdtsc`, x86 intrinsics throughout `ops.rs`) is hard x86_64. That's fine — just scope "boot compatible" as "modern x86_64 UEFI," which is what the debugging notes actually demonstrate.

---

## 4. Grant-review readiness — the honest part

The engineering underneath is real. The *presentation layer* around it currently undercuts it, and some of it would be a serious problem in front of a review board.

### 4.1 — Fabricated metrics (must fix before any external review)
- `aegis-eval/perplexity.rs` returns **hard-coded** `baseline_ppl = 14.12`, `pruned_ppl = 14.58`, with the comment `// Mock computation of cross-entropy loss for DARPA review`. It never loads WikiText-2, never runs the model.
- The `antigravity` CLI `/benchmark` and `/status` print **fixed** "TTFT 14ms / TPS 84.6 / Peak RSS 412MB."
- `/grant-review` runs one generation and then unconditionally prints **"Phase I Grant Viable."**

Presenting invented numbers as measurements to a federal review board is the kind of thing that ends programs and careers. Even framed as "placeholder," it should not live in code paths named for a review. **Replace with a real harness**: load WikiText-2, run teacher-forced forward passes, accumulate NLL, report perplexity with the exact eval config. Report real cycles/token from `/benchmark` with a correctly measured clock.

### 4.2 — What a real Phase I package needs
Reviewers will want, roughly in order:
1. **A crisp novelty claim.** "Bare-metal, OS-less ternary LLM inference with a custom UEFI memory path" is a legitimate systems contribution. Lead with the systems angle, not the model (the model is Microsoft's).
2. **Reproducible benchmarks vs. real baselines** — `bitnet.cpp` and `llama.cpp` on the same hardware, same prompts: tokens/sec, TTFT, peak RSS, energy if you can. Honest even where you lose.
3. **A perplexity/accuracy delta** from vocabulary pruning (real WikiText-2), plus at least one downstream task, so the accuracy cost of the 50k-vocab cut is quantified.
4. **A threat/failure model** for the "no OS" claim: what happens on ECC errors, thermal events, malformed input; why bare-metal is a feature not a liability for the target mission.
5. **A clear scope statement**: x86_64 UEFI, English-only pruned vocab, greedy decode, 2B params.

### 4.3 — Professional framing
Directory/product naming (`killboxincorporated`, `aegis_lobotomized_*`, "ANTIGRAVITY: AEGIS EDITION") and ANSI-art banners read as hobby-project theater. None of it is disqualifying, but for a review package, rename to neutral, descriptive identifiers and let the systems work speak. Reserve the personality for the README, not the eval harness.

---

## 5. Genuine strengths (keep these)

- A working `no_std` transformer that boots on real UEFI hardware — very few people get this working end-to-end.
- Faithful BitNet re-implementation (verified against the reference config).
- Real, non-trivial AVX2 ternary kernels with LUT unpacking and 4-row unrolling.
- Zero-allocation `WorkingMemoryArena` — the right pattern for OOM-free bare-metal inference.
- Thoughtful, correctly-diagnosed UEFI hardware fixes (device handle, DMA alignment, AnyPages workaround).
- Clean separation: `forge` (offline prep) / `core` (inference) / `uefi` (boot) / `eval`.

---

## 6. Prioritized punch list

**P0 — correctness / integrity (do first)**
1. Fix prefill embedding stride to F32; add the length invariant assert. (§1.1)
2. Replace fabricated perplexity + CLI benchmark/status numbers with real measurements. (§4.1)

**P1 — robustness / honesty**
3. Read file sizes from `FileInfo` instead of hard-coding them. (§3.2)
4. Rename the "quantization" stage to "vocabulary pruning"; drop the 1.58-bit claim from forge output. (§1.2)
5. CPUID guard before `xsetbv`; multi-region fallback in the physical allocator. (§3.2)
6. Fix the TSC→seconds clock assumption in `/benchmark`. (§2.5)
7. Delete the dead first `argmax`; remove dead `BitNetModel`. (§1.5)

**P2 — speed**
8. Vectorize the attention QK°/°V inner loops. (§2.2)
9. Fuse argmax into the LM-head dot; drop the `-inf` fill + second pass. (§2.1)
10. BF16 embeddings + widen-on-load to halve embedding memory/bandwidth. (§2.4)
11. (Milestone) int8 activation path for the true BitNet fast lane. (§2.3)

**P2 — review package**
12. Real baseline comparison vs bitnet.cpp / llama.cpp; scope statement; novelty framing; neutral naming. (§4.2–4.3)

---

*Note on `/exit`, `/clear`, `.gemini/...scratch` files, and the `_ALICE_1_0_BACKUP` trees: these are dev scaffolding duplicated in the dump. Strip them from any artifact you submit — reviewers shouldn't be reading backup copies and scratch LUT experiments.*
