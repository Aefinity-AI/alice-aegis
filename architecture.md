# Architecture ground truth

Facts below were verified against the source and against Microsoft's BitNet
b1.58-2B-4T reference config. When the code and this file disagree, the code has
drifted — investigate before proceeding, and update this file once resolved.

## Model configuration (verified vs reference)

| Parameter | Value |
|---|---|
| Layers | 30 |
| Hidden size (`emb_dim`) | 2560 |
| Attention | GQA — 20 query heads / 5 KV heads, head_dim 128 |
| FFN | intermediate 6912, ReLU² gate (`relu2`) |
| Norms | RMSNorm + SubLN (`attn_sub_norm`, `ffn_sub_norm`) |
| RoPE | `rope_theta = 500000` |
| Weights | Ternary (pre-quantized by Microsoft), per-tensor `weight_scale` dequant |
| Embeddings | Tied (`tie_word_embeddings: true`) — same matrix feeds input lookup and LM head |
| Activations | **f32 by deliberate choice** (reference is int8/A8) — higher fidelity, forfeits the int8 speed path |
| Sampling | Greedy argmax only; `temperature`/`top_p` fields exist but are unused — never describe the demo as top-p |
| Pruned vocab | 50,189 tokens (first 50k by original id + specials) |

The live config is parsed from JSON at init. Any hardcoded-dimension struct (a dead
`BitNetModel::new` with `max_position_embeddings: 1024` existed historically) is
dead code — remove on sight so nobody mistakes 1024 for the real context length.

## On-disk artifacts (produced by aegis-forge, loaded by aegis-uefi)

FAT32 8.3 uppercase names on the USB volume: `MODEL.SAF`, `EMBED.BIN`, `VOCAB.BIN`.

**MODEL.SAF** — the pruned safetensors file (ternary weights + scales).

**EMBED.BIN** — raw byte slice of `model.embed_tokens.weight`, rows in
keep-order. Critical: forge slices **raw bytes at the source tensor's dtype**
(`bytes_per_element = byte_len / (rows × hidden)`) — it does not convert. So the
dtype of EMBED.BIN is whatever the input safetensors had. The shipped F32 artifact
is 513,935,360 bytes = 50,189 × 2560 × 4. A BF16 re-forge would be 256,967,680
bytes. **Consumers must derive element size from byte length at init** (see prime
directive 3) — never assume.

**VOCAB.BIN** — flat binary, little-endian:

```
u32 magic  = 0x564F4341 ("VOCA")
u32 count  = number of tokens
repeat count times, sorted by new token id ascending:
    u16 byte_len
    [byte_len] raw UTF-8 token bytes
```

New id = position in this sorted order. Check `aegis-core/src/tokenizer.rs` for the
exact reader before changing the writer, and vice versa.

## Vocabulary pruning — what it is and its sharp edges

`aegis-forge` performs **vocabulary pruning + embedding slicing**, not
quantization. The ternary weights arrive pre-quantized from Microsoft. Historical
log lines claiming "Ternary Quantization (1.58-bit) Pipeline" are factually wrong
and read badly to reviewers — rename to "Vocabulary Pruning & Embedding Slicer"
wherever found.

The row-alignment logic is sound: `keep_indices` is built in old-id-sorted order,
embeddings are sliced in that order, VOCAB.BIN is emitted in that order, and tied
embeddings mean the same pruned matrix correctly serves input lookup and LM head.
Sharp edges to preserve in any change:

- **Special-token remapping.** LLaMA-3 specials live at 128000–128255 originally;
  after pruning they sit at ~50000+. Anything comparing against the stock
  tokenizer (perplexity harness, external evals, HF tooling) must translate through
  the pruned mapping or its numbers are meaningless.
- **English/ASCII-only by construction.** Non-ASCII capability is destroyed. That
  is a scope decision — state it explicitly in review materials, never let it
  surface as a surprise.
- The keep rule `token.starts_with("<")` retains any ordinary token beginning with
  `<`, not only specials. Harmless, but don't let comments claim otherwise.

## Embedding readers — the parity checklist

Every place that indexes into the embeddings buffer must use the single derived
element size. Inventory of readers (re-verify locations when editing):

1. **Prefill** — `forward_batch` in `aegis-core/src/inference.rs` (historically the
   buggy 2-byte reader).
2. **Decode** — `forward_step` in the same file.
3. **LM head** — `f32_dot_avx2` (and any fused-argmax successor) in
   `aegis-core/src/ops.rs`, tied to the same matrix.

Only the **last** token's hidden state survives prefill into decode, so a corrupted
prefill does not crash — it produces coherent-looking-but-prompt-ignoring output.
That failure signature ("drifts after the prompt", "ignores what I asked") should
send you straight to a stride/parity audit. The parity test: run one token through
both `forward_batch` and `forward_step` and compare hidden states exactly.

## Known cleanup targets (remove/fix on sight if still present)

- Dead double-argmax in the decode loop (argmax → penalty → argmax again; first
  call wasted).
- Dead `BitNetModel` struct with hardcoded dims.
- Nonstandard repetition penalty: scales by 1.2 **and** subtracts a flat 2.0,
  applied only to generated tokens. It works, but either document the formula
  where reviewers will read it or move to the standard HF-style penalty.
- `init_unpack_lut` guarded by `static mut` + bool checked on every matvec call —
  initialize once at engine construction instead.
- `_ALICE_1_0_BACKUP` trees and `.gemini/**/scratch` files: dev scaffolding. Never
  edit them, never let them into a review artifact, and don't let greps against
  them masquerade as findings about the live code (always check the path).
