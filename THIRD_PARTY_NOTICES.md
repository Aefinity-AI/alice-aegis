# Third-Party Notices — A.L.I.C.E. / Aegis

The root `LICENSE` (Apache-2.0, Copyright 2026 Justin B. Thompson / Aefinity AI)
covers this project's original work. **It does not apply to any material listed
below.** Each item here retains its own license and its own copyright holder.

This file must ship with any redistribution of this repository *or of any binary
built from it* — several obligations below (notice retention, and the MPL-2.0
source-availability pointer in §5) attach to binary distribution, and the `.efi`
artifacts under `artifacts/` are tracked and published.

Third-party files are left in the working locations the tooling expects; they are
identified by path here rather than relocated, because scripts including
`build_vocab.py`, `generate_vocab.py`, `aegis-forge/regen_vocab_embed.py`, and
`sync_linux.sh` resolve some of them by absolute path.

---

## 1. Microsoft BitNet b1.58-2B-4T — MIT

    Upstream:  https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
    Revision:  04c3b9ad9361b824064a1f25ea60a8be9599b127
    License:   MIT — Copyright (c) Microsoft Corporation.
    Full text: third_party/microsoft-bitnet-b1.58-2B-4T/LICENSE

The file `third_party/microsoft-bitnet-b1.58-2B-4T/LICENSE` is Microsoft's MIT
license text, retained byte-for-byte (sha256 `646f8936…`). Until 2026-08-05 this
file stood at the repository root, where it was mistaken for the license of this
project. It is not, and never was: it is the license file of the *model*, swept
in together with the model snapshot in the first commit (`c9981f2`, 2026-07-09).

### 1.1 Redistributed verbatim

Each byte-identical to the upstream Hugging Face revision above:

| Path | Upstream name | Size (bytes) |
|---|---|---|
| `MICROSOFT_MODEL_CARD.md` | `README.md` | 9,647 |
| `data_summary_card.md` | `data_summary_card.md` (EU AI Act data summary) | 3,861 |
| `config.json` | `config.json` | 844 |
| `generation_config.json` | `generation_config.json` | 199 |
| `special_tokens_map.json` | `special_tokens_map.json` | 73 |
| `tokenizer.json` | `tokenizer.json` | 9,085,698 |
| `tokenizer_config.json` | `tokenizer_config.json` | 50,834 |
| `aegis-linux/vocab.json` | `tokenizer.json` → `model.vocab`, extracted verbatim | 3,431,412 |

`aegis-linux/vocab.json` is a verbatim extraction, not a derivation: it is
`json.dumps(upstream["model"]["vocab"])` with all 128,000 ids preserved and no
author-authored content.

### 1.2 Derived from the above

The *transformation* in each case is this project's original work (see
`aegis-forge/src/vocab_stripper.rs`, `generate_vocab.py`, `build_vocab.py`,
`aegis-forge/repack_ternary.py`, `aegis-forge/regen_vocab_embed.py` — those
generators are the author's and are licensed under the root LICENSE). The
*content* is Microsoft's.

| Path | Derivation |
|---|---|
| `aegis-forge/aegis_pruned_vocab.json` | vocabulary pruned from upstream `tokenizer.json`; all 280,147 merges, 256 added tokens, pre-tokenizer, normalizer, decoder and post-processor retained verbatim |
| `aegis-forge/aegis_lobotomized_vocab.json` | byte-identical to the above |
| `aegis-forge/aegis_pruned_config.json` | upstream `config.json` with `vocab_size` and `max_position_embeddings` changed |
| `aegis-forge/aegis_lobotomized_config.json` | upstream `config.json` with `vocab_size` changed |

The same derivations appear under other paths and receive the same treatment:
`aegis-forge_ALICE_1_0_BACKUP/aegis_lobotomized_{vocab,config}.json` and
`aegis-uefi-usb-payload/{tokenizer.json,aegis_lobotomized_vocab.json}`.

### 1.3 Model output

| Path | Nature |
|---|---|
| `artifacts/relu2_down_in_bitnet2b_2026-08-01.av1` | The `AEGISAV1` container format and the capture harness are this project's original work. The numeric values it holds are inference output of Microsoft's weights. |

### 1.4 Tokenizer lineage — Meta Llama 3

`tokenizer.json`, `aegis-linux/vocab.json`, `VOCAB.BIN`, and the derived
vocabulary artifacts in §1.2 contain the **Meta Llama 3 tokenizer**. Microsoft's
own model card states: *"Tokenizer: LLaMA 3 Tokenizer (vocab size: 128,256)."*
The file is byte-identical to the tokenizer shipped with
`meta-llama/Meta-Llama-3-8B-Instruct`.

Microsoft redistributes it under MIT. The originating artifact is Meta's, whose
upstream terms are the **Llama 3 Community License**, which carries a "Built with
Meta Llama 3" naming obligation for derivatives. **Whether those terms reach
through Microsoft's MIT redistribution to this repository is a question for
counsel and is not resolved here.** It is recorded rather than assumed away.

Note that `tokenizer_config.json` is *not* Llama-identical — it carries BitNet's
own chat template and is genuinely Microsoft's derivative.

---

## 2. HuggingFace Transformers — Apache-2.0

    File:      modeling_bitnet.py
    Upstream:  https://github.com/huggingface/transformers
               src/transformers/models/bitnet/modeling_bitnet.py
               @ b262680af446ce012f43a44f20d97f8a2abba3bd
    Notice:    Copyright 2025 The BitNet Team and The HuggingFace Inc. team.
               All rights reserved.
               Licensed under the Apache License, Version 2.0.
               http://www.apache.org/licenses/LICENSE-2.0
    Status:    Unmodified, byte-identical (37,453 bytes / 823 lines,
               sha256 e3965a8a9768519fdd22f376d216e665fe5f6d74a0a40c87da097b119c5d6e38)

The file retains its own upstream Apache-2.0 header. It is reference material and
is not executed by this project — its three-dot relative imports
(`from ...activations import ACT2FN`) resolve only inside the `transformers`
package.

---

## 3. ARIS — MIT

    File:      program/loop/tools/evidence_check.py
    Upstream:  https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep
               tools/evidence_check.py
    Notice:    MIT License. Copyright (c) 2026 wanshuiyin.
    Status:    Vendored unmodified (9,970 bytes / 212 lines,
               sha256 2edbd901d760d22e7e43a14e8e88ab0e089303f489055cd9b5e9d1f4dce50382)

`program/loop/README.md` already acknowledges this file as *"(vendored from ARIS,
MIT, unmodified)"*. That acknowledgment is good faith but is not an MIT notice;
this entry supplies the required copyright and permission notice retention, as
the vendored file itself carries no header.

The file's own docstring credits a further influence: the reconcile pattern is
described there as *"adapted from NousResearch/hermes-agent's curator"*.

**The other five tools in `program/loop/tools/` — `claimlint.py`, `ledger.py`,
`prereg.py`, `runcard.py`, `runq.py` — are this project's original work and are
NOT covered by this entry.**

---

## 4. Evaluation data

| Path | Source | License |
|---|---|---|
| `model-lab/data/evals/arc_easy/ARC-Easy/validation-00000-of-00001.parquet` | `allenai/ai2_arc`, ARC-Easy validation split | **CC BY-SA 4.0** |
| `model-lab/data/evals/arc_easy/arc_easy_val_n570_seed42.jsonl` | the same, complete split (570 records, verbatim) | CC BY-SA 4.0 (adaptation) |
| `model-lab/data/evals/arc_easy/arc_easy_val_n100_seed42.jsonl` | the same, 100-item seeded subsample | CC BY-SA 4.0 (adaptation) |
| `aegis-core/tests/fixtures/falcon-e-1b/t2d_sample.txt` | **WikiText-2** test split (Merity et al., Salesforce) — a 7,325-byte excerpt | CC BY-SA (dataset card YAML says 3.0; card body says 4.0) |
| `aegis-core/tests/fixtures/falcon-e-1b/ref_tokens.txt` | tokenization of that same excerpt | CC BY-SA (derived) |

`model-lab/tinybit/**` weights were trained by the author on **TinyStories V2**
(`roneneldan/TinyStories`, **CDLA-Sharing-1.0**). The weights are the author's
work; the training corpus is named here for completeness.

CC BY-SA's Collection provision means these files do **not** virally relicense
this repository's Apache-2.0 code. The live obligations are license notice and
attribution — which is what this section discharges.

---

## 5. Rust dependencies linked into distributed binaries

The `.efi` artifacts under `artifacts/` statically link the crates resolved in
`aegis-uefi/Cargo.lock`, plus `library/core` and `library/alloc` from
`rust-lang/rust` (via `-Zbuild-std=core,alloc`, see `aegis-uefi/build_hardfloat.sh`).
These are permissively licensed (predominantly `MIT OR Apache-2.0`) and their
notices travel with the crates themselves, with one exception that requires
explicit action:

### 5.1 `ucs2` 0.3.3 — MPL-2.0 (weak copyleft)

    Upstream: https://github.com/rust-osdev/ucs2-rs
    License:  Mozilla Public License, Version 2.0

`ucs2` is a **non-optional** dependency of `uefi` 0.38.0 and its code is actually
reached at runtime: `uefi`'s `impl fmt::Write for Output` calls
`ucs2::encode_with`, which this project drives through `aegis-uefi/src/main.rs`.

MPL-2.0 §3.2–3.3 require that recipients of a binary form be informed how to
obtain the Source Code Form of the covered files. **This project uses `ucs2`
unmodified**, so that obligation is discharged by this pointer:

> The source of `ucs2` 0.3.3 is available from crates.io at
> <https://crates.io/crates/ucs2/0.3.3> and upstream at
> <https://github.com/rust-osdev/ucs2-rs>.

### 5.2 Note on `uefi`

`uefi` 0.38.0 is **`MIT OR Apache-2.0`**, not MPL-2.0. uefi-rs relicensed from
MPL-2.0 at version 0.34.0, four minor versions before the one used here. Earlier
project notes that recorded it as MPL-2.0 were mistaken; the crate's own
`src/lib.rs` carries `// SPDX-License-Identifier: MIT OR Apache-2.0`.

### 5.3 Note on `safetensors`

`safetensors` 0.4.5 (used by `aegis-forge`) is **Apache-2.0 with no MIT
alternative**. This does not constrain this project's own licensing, but
downstream consumers must honor Apache-2.0 for that dependency.

---

## 6. What is NOT third-party

Recorded explicitly so it is not surrendered by mistake in a future pass.

The inference engine and unikernel — all tracked `.rs` across `aegis-core`,
`aegis-uefi`, `aegis-linux`, `aegis-eval`, `aegis-forge`, and `xtask` — are the
author's original work. This includes the hand-written AVX2 ternary kernels,
which are structurally unlike Microsoft's `bitnet.cpp` (no `pshufb`, no TL1/TL2,
no `maddubs`; instead a compile-time `[f32; 1024]` unpack LUT with
`_mm256_insertf128_ps` and FMA over a bespoke 2-bit code).

Also the author's: this project's container formats (`ACOV`/`VOCAB.BIN`,
`MODEL.SAF`, `EMBED.BIN`, `AEGISAV1`); all Python tooling **except**
`modeling_bitnet.py`; `fable-hand`; the `tinybit` trainer and its
from-scratch-trained weights; and all documentation and hardware logs.

Two acknowledged influences were examined and found to be idea, not expression:
`aegis-core/src/cis.rs` credits a "TFLite/gemmlowp lineage" but uses a different
algorithm and different rounding (exact i128 multiply, round-to-nearest-even);
and `mtrr_decode.rs` implements Intel SDM Vol 3A §12.11, a specification, and is
structurally unlike Linux's `mtrr_type_lookup`. Neither triggers an obligation.

---

*Prepared 2026-08-05 from a file-by-file provenance audit. Corrections are
welcome as issues — this file is meant to be accurate, not flattering.*
