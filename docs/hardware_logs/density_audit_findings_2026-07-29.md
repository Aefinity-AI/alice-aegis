# Intelligence-density audit — 2026-07-29

Workflow wf_23e4ee46-325: 10 agents, 225 tool calls, 752k tokens.

## Recoverable bytes
```
DEPLOYED SET = 9,253,042 B. Two independent audits reproduce the byte accounting exactly (MODEL.SAF 2,797,632 = 13,692 hdr + 2,752,512 U8 [49 tensors] + 31,232 BF16 + 196 F32; EMBED.BIN 6,291,456; VOCAB.BIN 163,954). I re-derived the EMBED.BIN numbers on-box today; they land within 1 B of the adversarial review.

=== LOSSLESS — bit-exact, zero quality change ===
A. Reaches the RAM/decode path (i.e. genuinely shrinks the running system):
   * MODEL.SAF safetensors JSON header -> fixed binary header: 13,692 B (0.15% of set).
   * THAT IS THE ENTIRE LIST. Nothing else lossless survives contact with the loader.
B. Storage / boot-read only (does NOT reduce runtime RAM — see engine section):
   * MODEL.SAF ternary container: 549,132 B with stock zlib -9 TODAY; 576,005 B at the
     order-0 entropy floor (needs a custom static-3-symbol rANS/arithmetic coder).
     [MEASURED_TODAY, reproduced twice: H0 = 1.581470 b/w vs log2(3) = 1.584963, N = 11,010,048]
   * EMBED.BIN: 1,328,079 B with stock zlib -9 TODAY; 2,055,198 B at the 16-bit symbol
     entropy floor (H16 = 10.7734 bits; BF16 exponent field uses 27 of 256 values,
     H(exp) = 2.8731 bits of 8 stored). [MEASURED_TODAY by me: verify_embed.py]
   * Stock-tools total available today: 1,877,211 B = 20.29% of the deployed set, zero new code.
   * Custom-coder ceiling: 2,631,203 B = 28.44%. I do not recommend pursuing it (see what_to_cut).

=== NEAR-LOSSLESS — lossless on the measured corpus, unbounded OOD risk ===
   * Prune the 60 safely-removable never-used vocab rows (NOT 138 — 78 of those are
     base-256 ByteLevel alphabet tokens and the tokenizer has byte_fallback=False,
     unk_token=None, so removing them makes '^' unencodable): 46,080 B standalone,
     23,280 B if stacked after INT8. Bonus: 8192-60 = 8132 = 4 x 2033, so the kernel's
     4-row blocking survives the prune with no tail path. [verified today]

=== LOSSY — quality cost measured against loss, not against a byte count ===
   * Per-row INT8 embedding: 6,291,456 -> 3,178,496 B = 3,112,960 B saved = 33.64% of the
     deployed set. COST: dNLL +0.00191 nats, ppl 4.1704 -> 4.1791 = +0.13%, paired t = +4.20,
     n = 64 disjoint windows [adversarial review, MEASURED_TODAY]. My independent kernel
     check today on the real bytes: logit relMSE 9.67e-5, max abs logit error 0.0549,
     argmax preserved.
   * INT6 embedding: ~3.90 MB (42.1% of set) for +3.7% ppl. Park — no byte-aligned SIMD path.
   * INT4 / INT3 / ternary embedding: DEAD. ppl 4.17 -> 9.08 / 6,527 / 832,311.

=== WHAT I ACTUALLY RECOMMEND SHIPPING (stacked, no double-counting) ===
   INT8 embed (3,112,960) + binary header (13,692) + 60-row vocab prune (23,280)
   = 3,149,932 B = 34.04% of the deployed set.
   9,253,042 B -> 6,103,110 B, on disk AND in RAM AND in the decode path.
   Total quality cost: +0.13% validation perplexity, plus an OOD-token caveat on the prune.
   Optionally zlib the shipped files at rest for OTA/storage only — that is a distribution
   decision, not an engine decision.
```

## Ranked actions

### 1. Per-row INT8 requantization of EMBED.BIN + a matching int8 lm_head kernel in aegis-core/src/ops.rs. This is the only action in the entire audit that wins on all three axes at once (disk, RAM, decode). Prototype kernel written and validated TODAY at /tmp/claude-1000/-home-killboxincorporated/d75b4760-6465-4d8e-bf5f-32d865609e80/scratchpad/int8_lmhead_bench.rs (4 rows x 8 cols, _mm_loadl_epi64 -> _m
- bytes: 3,112,960 B (33.64% of deployed set). ~389,000 B/hour at 8 h.
- cost: MEASURED: dNLL +0.00191 nats, ppl 4.1704 -> 4.1791 (+0.13%), paired t=+4.20 n=64. Independently corroborated today: logit relMSE 9.67e-5, argmax preserved. Also MEASURED TODAY: kernel goes 487.9 us -> 195.2 us per token = 2.500x (12.89 -> 16.29 GB/s effective) on the real EMBED.BIN bytes, i5-10210U. DERIVED end-to-end: -292.7 us of a ~2.00 ms token budget => ~1.17x decode, ~470 -> ~550 tok/s. That end-to-end figure is UNBENCHMARKED.
- effort: 6-10 h: export path (export_hf.py), int8 kernel (prototype exists), inference.rs plumbing + a lm_head_bytes()/scale-pointer signature change, re-run the round-trip parity gate. Requires re-passing the torch->engine gate (currently 0.19%).
- novel: False

### 2. zlib -9 the shipped artifacts at rest, decompress at boot. Highest raw bytes-per-hour in the audit — and I am ranking it #2 only because the metric is misleading here. Read the engine section before doing this: it buys nothing ALICE cares about and it makes the one boot failure mode the loader already guards against (contiguous huge-page fragmentation, aegis-uefi/src/main.rs:369) strictly more lik
- bytes: 1,877,211 B on disk (20.29% of set); 549,132 from MODEL.SAF + 1,328,079 from EMBED.BIN. ~469,000 B/hour at 4 h. ZERO bytes of runtime RAM. ZERO us of decode.
- cost: None — bit-exact. The cost is architectural, not statistical: peak boot RAM goes UP (compressed + decompressed buffers co-resident), and boot-read savings are swamped by 3.0 s of unconditional uefi::boot::stall(1s) calls at main.rs:398, 406, 414.
- effort: 3-5 h (a no_std inflate in the UEFI loader is the real work, not the compression).
- novel: False

### 3. Prune the 60 genuinely-unused vocabulary rows (never-used-in-training MINUS the base-256 ByteLevel alphabet), remap ids, shrink EMBED.BIN and VOCAB.BIN.
- bytes: 46,080 B standalone; 23,280 B if applied after INT8. ~11,500 B/hour at 4 h.
- cost: Zero on train_8k.bin (536,203,430 tokens) and valid_8k.bin by construction. Non-zero and unbounded on OOD input that would have produced a pruned id — needs an explicit remap-or-reject path. 8132 % 4 == 0 so the kernel blocking is unaffected.
- effort: 4 h including a re-tokenize round-trip test over both corpora.
- novel: False

### 4. Replace the MODEL.SAF safetensors JSON header with a fixed binary header.
- bytes: 13,692 B (0.15%). ~6,800 B/hour at 2 h. Small, but it is the ONLY lossless saving in the audit that reaches runtime RAM, because aegis-uefi/src/main.rs:365-383 allocates huge pages sized to the FILE and loads it verbatim.
- cost: None. Costs you safetensors tooling compatibility on the artifact.
- effort: 2 h, and it is nearly free if you are already in the loader for action 1.
- novel: False

### 5. MEASURE, don't build: instrument real boot wall-clock on the target USB/FAT32 iron and delete or shrink the three unconditional 1-second stalls at aegis-uefi/src/main.rs:398, 406, 414 if they are not load-bearing. NOVEL FINDING — neither probe nor the adversarial review looked at the loader, and every boot-I/O argument in this audit is conditioned on it.
- bytes: 0 B. Potentially ~3.0 s of boot time — plausibly an order of magnitude more than compressing the entire 9.25 MB artifact set buys. Infinite bytes/hour is the wrong metric; this is the highest-value hour in the list.
- cost: None. Risk: the comments say the stalls flush USB hardware state; they may be protecting against a real short-read bug (there is already short-read handling at main.rs:211). Measure before cutting.
- effort: 1-2 h to instrument, unknown to fix if the stalls turn out to be necessary.
- novel: True

### 6. Custom static-3-symbol rANS / arithmetic coder for the ternary payload, and a BF16-field-split coder for the embedding. Listed for completeness. DO NOT BUILD — see what_to_cut.
- bytes: 2,631,203 B on disk (28.44%). ~65,800 B/hour at 40 h. Zero RAM, zero decode.
- cost: None statistically. Architecturally fatal in the inner loop: variable-length codes make row offsets data-dependent, and ternary_matvec_avx2 (ops.rs:401+) computes row * packed_dim_in + col_packed and does read_unaligned::<u64> on a fixed stride. Confirmed against source, not assumed.
- effort: 30-50 h for two coders plus no_std decoders.
- novel: False

### 7. Exploit the 20.523% of (token, unit) activations that the int8 per-token absmax quantizer annihilates before down_proj (L0 32.45% down to L6 15.44%). NOT a byte saving — a COMPUTE saving. If the thesis is 'no wasted work' rather than 'no wasted bytes', this is the largest untouched number in the audit.
- bytes: 0 B of storage. Up to ~20% of down_proj MACs multiply by a hard zero. Bytes/hour is undefined; this is the wrong axis for it.
- cost: None if done exactly (skipping a zero input is arithmetically free). But the ternary kernel runs at only 2.47 GB/s effective weight-read bandwidth — it is latency/compute bound, not bandwidth bound — so branchy sparsity skipping in AVX2 will most likely LOSE. UNMEASURED in either direction.
- effort: 8-16 h to prototype and benchmark, with a real chance the answer is 'no gain'. Negative result is publishable.
- novel: True

## ALICE engine implications
```
Grounded in source I read today, not inferred. Loader: /home/killboxincorporated/aegis-uefi/src/main.rs:356-414. Kernels: /home/killboxincorporated/aegis-core/src/ops.rs (ternary_matvec_avx2 ~:401, f32_dot_avx2 :757).

THE STRUCTURAL FACT THAT DECIDES EVERYTHING: main.rs:365-383 reads each file's size, calls allocator::allocate_huge_pages(size/4096) for CONTIGUOUS physical pages, and loads the file bytes VERBATIM into that allocation. The kernels then index those bytes in place. Therefore for A.L.I.C.E., file size == runtime RAM, one to one. Any format that must be expanded before the kernel can read it saves disk and NOTHING ELSE.

1) REDUCES BOOT I/O (FAT32 USB read) — every byte removed from any file, including compression.
   BUT: main.rs:398, 406, 414 each call uefi::boot::stall(Duration::from_secs(1)) between file loads,
   "to flush USB hardware state machine", plus another at :158. That is >=3.0 s of UNCONDITIONAL,
   size-independent boot cost. The full 9.25 MB set at plausible USB2/FAT32 rates reads in a fraction
   of a second. Removing 3.1 MB therefore changes boot time by a small fraction of a second inside a
   boot dominated by deliberate stalls. Boot-I/O is the WEAKEST justification in this audit and
   every argument that leans on it should be discounted until someone measures real boot wall-clock.
   [Stalls: read from source today. USB throughput: NOT MEASURED — do not quote a ratio.]

2) REDUCES RUNTIME RAM — only formats the kernel reads directly:
   * Per-row INT8 embedding: -3,112,960 B resident.
   * Binary header: -13,692 B resident.
   * Vocab prune: -23,280 B resident (post-INT8).
   Total -3,149,932 B of the huge-page allocation. This matters more than it looks: main.rs:369 has an
   explicit fatal-error path for "contiguous alloc for MODEL.SAF (memory map too fragmented / not enough
   RAM)", and the stated deployment target is 2 GB pre-AVX2 iron (ledger M13/M21). Shrinking the
   contiguous request is a BOOT-RELIABILITY win, not just a footprint win.

3) REDUCES DECODE TIME — exactly one action, and it is now measured, not argued:
   * Per-row INT8 embedding. f32_dot_avx2 streams ALL 8192 x 384 BF16 = 6,291,456 B every single decode
     step to produce logits (tied head). MEASURED TODAY on the real EMBED.BIN, i5-10210U:
       bf16 arm 487.9 us/call, 12.89 GB/s   (independently replicates the logged 12.58 GB/s / ~0.50 ms)
       int8 arm 195.2 us/call, 16.29 GB/s   speedup 2.500x, bytes -49.48%
     Saves 292.7 us of a ~2.00 ms token budget (M22: 470-507 tok/s on aegis-linux) => DERIVED ~1.17x
     end-to-end, roughly 470 -> 550 tok/s. UNBENCHMARKED end-to-end; do not ship the tok/s number
     without an engine run.
     HONEST CAVEAT: this box has 6 MiB L3 and EMBED.BIN is exactly 6.0 MiB. The bf16 arm sits on the
     L3 cliff; the int8 arm (3.03 MiB) sits under it. That is why the speedup is 2.50x and not 2.0x.
     On hardware where neither fits, expect ~2.0x. On hardware where both fit, less. Quote 2.0x.

4) SHRINKS THE FILE AND DOES NOTHING ELSE — the trap:
   * zlib/lzma/rANS on MODEL.SAF or EMBED.BIN: storage and OTA only. Runtime RAM UNCHANGED (must
     decompress to the exact layout the kernel indexes). Peak boot RAM WORSE. Fragmentation risk UP.
     Decode unchanged. On the current loader this is close to zero practical value.

5) ACTIVELY NEGATIVE — base-243 5-trit packing, and this is the single most important new fact:
   the adversarial review recommended it as "the engineering answer". It was BENCHMARKED TODAY at
   /tmp/claude-1000/-home-killboxincorporated/d75b4760-6465-4d8e-bf5f-32d865609e80/scratchpad/base243_bench.rs
   (4-trit arm copied verbatim in structure from ops.rs, both arms bit-accurate to a scalar reference,
   max abs err 0.00003). Result, real m7 layer shapes, whole model one decode step:
       4-trit  1115.5 us, payload 2,752,512 B, 2.47 GB/s effective
       base243 3717.0 us, payload 2,206,848 B, 0.59 GB/s effective
       time +233.2% for bytes -19.8%; per-matvec ratio 3.24x-3.56x across all 7 shapes.
   Base-243 is DEAD as a runtime format. It trades 545,664 B for 2.6 ms per token on a 2.0 ms budget.
   As a storage-only format it is strictly worse than zlib. Kill it in both roles.

BONUS DECODE ACCOUNTING (composed from two measured microbenches + M22): ternary matvecs 1115.5 us
+ lm_head 487.9 us = 1603 us of a ~2000 us token. The lm_head is 24.4% of decode and is the only
bandwidth-bound kernel. That is the whole reason action 1 outranks everything else.
```

## The defensible story
```
A claim survives, but it is NOT the one the thesis started with. Exact wording, every number traceable to a file or a command run today:

--- USE THIS ---
"A.L.I.C.E. runs a 14,171,392-parameter from-scratch ternary language model on bare metal from a 9,253,042-byte artifact set. We audited that artifact byte by byte.

The ternary weights are already at their information-theoretic floor: measured order-0 entropy is 1.581470 bits per weight against a log2(3) = 1.584963 ceiling over 11,010,048 weights — 99.78% of maximum. There is no redundancy left to compress out of the weights; the only slack is 20.93% of container overhead from storing a 1.585-bit symbol in a 2-bit field, and an i.i.d.-shuffle control shows no exploitable structure beyond the marginal (0.229% of payload).

The waste was not in the weights. It was in the embedding table, which holds 22.20% of the parameters in 67.99% of the bytes — a 7.88x density asymmetry. Requantizing it to per-row INT8 removes 3,112,960 bytes, 33.6% of the entire deployed artifact, for a measured +0.13% validation perplexity (4.1704 -> 4.1791; delta-NLL +0.00191 nats, paired t = +4.20, n = 64 disjoint windows), and makes the only bandwidth-bound kernel in the decode path 2.5x faster (487.9 -> 195.2 microseconds per token on the real bytes, Intel i5-10210U, AVX2).

We also ran the experiments that could have embarrassed us. There are no dead units to prune: ablating the quietest 1 unit per layer out of 7,168 already costs +0.00073 nats (t = +2.20). Ranking units by activation magnitude is indistinguishable from ranking them at random (bottom-10% ablation +0.09070 vs random-10% +0.09330) — exactly what Elhage et al.'s superposition account predicts, and it invalidates magnitude-based pruning for this architecture. The one truly free unit in the model is free because the int8 activation quantizer zeroes it, not because training left it idle."
--- END ---

TWO SENTENCES THAT MUST NEVER BE SAID, and why:
1. "Every byte carries actual intelligence / provable information density." DEAD. Maximum entropy
   means maximum INCOMPRESSIBILITY, not maximum information content. The audit's own i.i.d. control
   proves it against itself: a random ternary stream with the same marginal compresses to within
   6,307 B (0.229%) of the trained weights. The measurement cannot distinguish a trained model from
   noise, so it can support a compression claim and nothing else. A reviewer kills this in one line.
2. "It is 26%, and not one byte more." DEAD. Conditioning on row identity is worth 15,700 B and on
   column identity 16,644 B gross (8,740-14,078 B net of side information) — 2.5x the sequential
   structure the probe tested. The conclusion (not worth building) survives; the bound does not.

WHAT THE SURVIVING CLAIM IS, HONESTLY LABELLED: it is a CREDIBILITY claim — "we can prove what our
bytes are doing, including where we were wrong" — not a capability claim. That is genuinely
differentiating against data-centre players, who publish neither byte-level artifact audits nor
their failed ablations. It is not the same thing as "provable information density at the edge,"
which this audit does not support.
```

## Cut

- CUT base-243 5-trit packing, in both the runtime and the storage role. BENCHMARKED TODAY: 3.24x-3.56x slower per matvec, whole-model decode 1115.5 -> 3717.0 us (+233.2%) for -19.8% bytes. It trades 545,664 B for 2.6 ms per token on a 2.0 ms budget. The adversarial review recommended it; the benchmark kills it.
- CUT the custom rANS/arithmetic entropy coder for MODEL.SAF (576,005 B). 40 h of work for boot-I/O and OTA only: zero runtime RAM, zero decode benefit, worse peak boot RAM, higher contiguous-allocation fragmentation risk, and swamped by 3.0 s of unconditional boot stalls. If you want the disk bytes, zlib -9 gets 95.3% of them (549,132 B) in an afternoon.
- CUT all context-modelled / order-N compression research. Order-1 conditioning is worth 838 B, order-2 1,568 B, row/column conditioning 8,740-14,078 B net — all under 0.6% of payload. lzma lands 1.73% ABOVE the order-0 bound. There is nothing there.
- CUT magnitude-ranked pruning from the roadmap entirely. MEASURED: bottom-5% ablation +0.04292 vs random-5% +0.03706 (bottom is WORSE); bottom-10% +0.09070 vs random-10% +0.09330; bottom-25% +0.29787 vs random-25% +0.30294. All indistinguishable within SE. Any future pruning that ranks by activation magnitude is ranking by noise. Superposition predicted this.
- CUT the 105,984 B vocab-pruning figure from every document. It is 2.3x too high. 78 of the 138 never-used rows are base-256 ByteLevel alphabet tokens and the tokenizer has byte_fallback=False and unk_token=None, so they ARE the encoding-totality guarantee — id 62 is '^'. Prune them and the model cannot encode a caret. The real number is 60 rows = 46,080 B standalone, 23,280 B post-INT8.
- CUT INT4 and everything below it for the embedding. ppl 4.1704 -> 9.08 at INT4, 6,527 at INT3, 832,311 at ternary. And CUT the reasoning that produced that prediction: relMSE is NOT comparable across tensor roles. Embedding error enters the logits directly through the tied head and is never compensated; mid-network error is absorbed by layers that were QAT-trained against it. Anyone porting the 'relMSE 0.33-0.46 is acceptable' intuition to the embedding destroys the model.
- CUT Probe 2's primary dead-unit test (tau = 1e-3 ABSOLUTE on post-RMSNorm activations, freq_mean 0.9887-0.9958). On unit-RMS activations that measures the density of a distribution near zero and nothing else. It is not corroboration of the relative-threshold result; it is zero evidence, and reporting them side by side as agreement is the kind of thing a reviewer notices.
- CUT the factual errors before anything ships: Probe 1's 'ternary payload, 98 U8 tensors' (there are 49 — its own shape list sums to 49), and 'it is 26%, and not one byte more' (false by ~14 KB, and arrived at by testing the wrong hypothesis).
- CUT the phrase 'provable information density' and the sentence 'every byte carries actual intelligence' from all external material. See the_defensible_story for why. This is the single most dangerous sentence in the package.
- CUT any quotation of 'paired SE 0.0016 nats' as a fixed harness noise floor. Paired sd is INTERVENTION-dependent (0.02695 nats/window for a 10% ablation, ~10x smaller for a 1-unit-per-layer ablation). Quote the SE for the specific intervention or quote nothing.
- DO NOT CUT, but stop quoting as precise: Probe 2's per-head share table (min 11.01%) and residual-norm table. They come from 4,096 contiguous tokens whose effective sample size is nearer the document count than the token count, and they carry no error bars. The dead-unit NULL is safe (a narrow corpus makes more units look dead, not fewer, and the causal ablation confirmed it) — the per-head numbers are not.

## Honest verdict
```
No. That is overreach, and it is the wrong axis entirely.

Nothing in this audit measured a capability. Every number here is about the CONTAINER — how efficiently bytes encode weights that already existed. The single best action, per-row INT8 embedding, makes the model 0.13% WORSE while making it 34% smaller. You cannot get from "smaller container" to "more capable model." A 34% smaller artifact that is 0.13% worse is a deployment win and an engineering credibility win. It is not evidence of capability, and a reviewer will say so within one paragraph.

The honest strategic reading, in three parts:

1. THE USER'S THESIS IS HALF-CONFIRMED AND HALF-INVERTED. "No wasted bytes" — the ternary weights are at 99.78% of their entropy ceiling and there are no dead units, no negligible heads, no removable layers. That half is confirmed, causally, by an ablation neither probe originally ran. But the framing "every byte carries actual intelligence" is a category error: maximum entropy is maximum incompressibility, and the i.i.d. control shows this analysis cannot tell the trained model from random ternary noise. The thesis survives as an ENGINEERING discipline. It dies as an INFORMATION-THEORETIC claim.

2. THE AUDIT SPENT ITS EFFORT ON THE WRONG 30%. Two probes optimised MODEL.SAF, which was already near-optimal, and left EMBED.BIN — 68% of the artifact — unmeasured until the adversarial pass. The lesson is generalisable and worth writing down: measure the biggest object first, not the most interesting one.

3. WHERE THE ACTUAL CAPABILITY STORY LIVES, AND IT IS NOT HERE. Ledger M21: the ternary model beat its own fp32 twin, 5.140 vs 5.513 final val PPL, at an identical 500,002,816-token budget on identical data in identical order. Ledger M22: it generates coherent TinyStories-class English on bare metal at 470-507 tok/s. Those are the capability claims, they are already logged, and they are stronger than anything in this density audit. Note the disclosure that already sits on M21 — recipe-vs-recipe, not single-variable, 2.17x param handicap. Keep it there.

So: "small, honestly trained, and fully accounted for" — defensible today. "Small and capable" — you need domain evals (M6), not byte counts, and this audit does not advance it by one point. What this audit actually buys is the thing data-centre players will not do: publish the byte-level teardown of your own artifact including the parts where you were wrong, the recommendation you benchmarked and killed (base-243, +233% decode), and the pruning heuristic you proved was noise. That is a real moat. It is a moat made of credibility, not of capability. Sell it as that.

MY OWN FAVOURITE NUMBER, KILLED AS INSTRUCTED: I expected the 20.93% ternary-container win to be the headline. It is not. It is boot-I/O-only, it is 5.8% of the deployed set, and on the current loader it is dominated by 3.0 s of unconditional uefi::boot::stall calls at aegis-uefi/src/main.rs:398/406/414 that nobody in this audit had looked at. The unglamorous action — measure real boot wall-clock before optimising a single byte for boot — is worth more than the entire compression programme.
```

