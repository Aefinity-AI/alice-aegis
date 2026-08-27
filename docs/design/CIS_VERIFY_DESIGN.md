# `cis-verify` — standalone third-party verifier for CIS-1 decode receipts

Design doc for RESEARCH_LOOP.md E4. Status: design only, no code beyond
illustrative signatures. Target: a crate a third party can build and run
without trusting, or depending on, this repo's engine code.

**Sources read:** `docs/CIS-1_SPEC_v1.0.md` §5–§8, `aegis-core/src/witness.rs`,
`aegis-linux/examples/cis_witness.rs`, `aegis-core/src/{cis,cis_attn,cis_infer,
model,tokenizer,json,kvcache}.rs`, `aegis-core/tests/witness_contract.rs`,
`tests/golden/witness_v1_m7_once64.receipt` (read-only), `program/
RESEARCH_LEDGER.md` row A32, `docs/paper/{05_receipt,08_limitations}.md`.

---

## 1. The receipt format, as implemented

The receipt is a **plain-text, line-oriented, hex-encoded KV file** — not a
binary format. It is produced by `cis_witness gen` and consumed by
`cis_witness verify` (`aegis-linux/examples/cis_witness.rs:149-159,
171-193`), built on the chain primitive in `aegis-core/src/witness.rs`.

### 1.1 Byte layout (as written by `cis_witness gen`, lines 149-159)

```
AEGIS-WITNESS v1-CIS                    literal, no key
model <64 hex chars>                    sha256(MODEL.SAF bytes)
embed <64 hex chars>                    sha256(EMBED.BIN bytes)
vocab <64 hex chars>                    sha256(VOCAB.BIN bytes)
maxtok <decimal>                        max_new (u64) as ASCII decimal
prompt-hex <hex(len(prompt)*2) chars>   prompt bytes, hex-encoded (UTF-8 in)
prompt-toks <decimal>                   tokenizer.encode(prompt).len()
gen-toks <decimal>                      number of generated tokens (== maxtok
                                         unless a future early-stop exists;
                                         v1 has none)
token-ids <csv of decimal u32>          every generated token id, in order
cis-digest <16 hex chars>               FNV-1a 64, LE fold over prompt ids
                                         then generated ids (spec §8 Tier 3)
chain <64 hex chars>                    final WitnessChain digest (SHA-256)
```

Verified against `tests/golden/witness_v1_m7_once64.receipt` (11 lines,
read-only): hash/chain fields are exactly 64 hex chars (32-byte SHA-256),
`cis-digest` is exactly 16 hex chars (8-byte FNV-1a 64), `prompt-hex` is
32 hex chars (16-byte UTF-8 prompt "Once upon a time"), `token-ids` has 64
comma-separated entries matching `gen-toks 64`. Field values on that golden
file: `model=23cfad0a…`, `embed=3752cd5c…`, `vocab=5a1d79ca…`,
`cis-digest=67e8c0a96abc04e1` (matches spec §8 Tier 3's pinned constant,
`docs/CIS-1_SPEC_v1.0.md:343`), `chain=aee25b770bd7b22e…` (matches ledger
row A32, `program/RESEARCH_LEDGER.md:46`).

**Parsing is line-splitting on the first space** (`cis_witness.rs:171-193`):
`let mut it = line.splitn(2, ' ')`. No escaping, no length framing beyond
what's implicit in fixed hex widths; unknown keys are silently ignored
(`_ => {}`, line 191). This is fragile as a wire format (a value containing
a literal newline would break it) but every field here is either a fixed-hex
digest or a decimal integer or hex-of-bytes, so it's safe in practice. A
verifier crate should treat this as v1's actual contract, not aspire to a
stricter format it doesn't have.

### 1.2 What is committed, and how (the actual cryptographic construction)

`aegis_core::witness` (`aegis-core/src/witness.rs`):

- **Header hash** (`WitnessHeader::hash`, lines 154-169) — the chain's
  genesis value:
  `SHA256(WITNESS_DOMAIN_V1 ‖ model_sha ‖ embed_sha ‖ vocab_sha ‖
  BE64(max_new) ‖ BE64(len(prompt)) ‖ prompt)`
  where `WITNESS_DOMAIN_V1 = b"AEGIS-WITNESS v1-CIS\n"` (line 143, a version
  domain-separator — bumping the format means changing this literal, which
  changes every downstream digest by design).
- **Per-step fold** (`WitnessChain::fold_step`, lines 189-201):
  `chain' = SHA256(chain ‖ b"STEP" ‖ BE64(step_index) ‖ LE32(token_id) ‖
  BE64(len(logits)) ‖ LE64(logits[0]) ‖ … ‖ LE64(logits[n-1]))`
  — note the mixed endianness is real and load-bearing: multi-byte *lengths
  and step indices* are big-endian, `token_id` and each `i64` logit are
  little-endian. A verifier must reproduce this exactly, not "some
  consistent" encoding.
- **`token_id` folded is the argmax winner**, and **`logits` is the
  complete i64 LM-head output vector for that step** (all `vocab_size`
  entries), absorbed *before* the next `forward_step_int` call
  (`cis_witness.rs:88-98`). This is the receipt's real payload: 64 steps ×
  a full logit vector each, not a summary (ledger A32, paper §5).
- SHA-256 itself: a from-scratch FIPS 180-4 streaming implementation
  (`witness.rs:22-132`), `no_std`, no-alloc, pinned against the FIPS
  test vectors in `aegis-core/tests/witness_contract.rs:16-73` (empty
  string, `"abc"`, the two-block vector, the million-`'a'` long-message
  vector, and a streaming-vs-one-shot chunk-size fuzz).

**Critical fact for verifier design: the per-step logit vectors are NOT
serialized into the receipt file.** Only their SHA-256 fold (the `chain`
field) and the argmax winners (`token-ids`) survive to disk. A verifier
therefore cannot check the chain against stored logits — it must recompute
every logit vector at every step from the three artifacts, which means
**running the entire decode**, not spot-checking a hash.

---

## 2. What a verifier must recompute, and what it can skip

### 2.1 Must recompute (spec ops, normative, §5)

To reproduce one receipt, a verifier must run the identical `CisMode::FullInt`
forward pass `cis_infer.rs:841-…` performs, step by step, which touches
every op in spec §5:

1. **Embedding lookup** — BF16 row → Q.20 by exact RNE (§5.6, `bf16_to_fixed`,
   `cis_infer.rs:97`); CIS mode requires EMBED.BIN and the LM-head table to
   be BF16, exactly `vocab_size*hidden_size*2` bytes (`check_bf16_table`,
   `cis_infer.rs:588-597`; enforced in `CisModel::new`, `cis_infer.rs:693-698`).
2. **RMSNORM-I**, per §5.4's four-step normative procedure (exact
   `isqrt`, Q2.30 inverse-RMS, the 2¹⁵ downshift) — reference `cis.rs:298`
   (`rmsnorm_i`), used at every norm site through `normq`/`quantq`
   (`cis_infer.rs:306,346`).
3. **TMV** (ternary matvec), all seven per layer (q/k/v/o/gate/up/down) —
   reference `cis.rs:202` (`ternary_matvec_i8`), spec §5.1 including its
   five-condition rejection surface (§5.1, "the rejection surface, itself
   normative").
4. **QUANT-ACT / REQUANT** — per-token dynamic absmax quantization
   (`cis.rs:247`, spec §5.2) and the `(M,S)` fixed-point rescale (`cis.rs:123,
   133`, spec §5.3) plus the `QScale64` i64/Q.20 variant (`cis_infer.rs:194-284`,
   spec §5.3's "QScale64" paragraph).
5. **ROPE-I** — load-generated Q1.30 tables from `(max_seq, head_dim,
   rope_theta)` alone (`cis_attn.rs:292,341`, spec §5.9), digest-pinned per
   shape (`0xD8345EBF01E990FA` for M7 shape, spec §5.9).
6. **SOFTMAX-I** — exact-RNE-division softmax on the Q.24 score grid
   (`cis_attn.rs:173`, spec §5.8), sharing the exp-LUT machinery of §5.7
   (`cis_attn.rs:78-171`, LUT digest `0x66C2A0EEB8C2DC43`, normative).
7. **ACT-I** — relu² or silu MLP elementwise on the Q.20 grid
   (`cis_attn.rs:391,413`, spec §5.10) — which one depends on
   `ModelConfig.hidden_act` read from MODEL.SAF's `aegis_config` metadata.
8. **Pipeline grid assignments** (§5.12) — the Q.16 q/k/v grid, Q.24 score
   grid, Q.20 residual/V-mix grid; these are not a separate op but the glue
   between the ops above and must match exactly (getting a shift by one bit
   wrong reproduces nothing).
9. **ARGMAX** (§5.11) — exact i64 equality, ties break to lowest index
   (`cis_infer.rs:419`, `argmax_i64`) — this selects `token_id` for both the
   `token-ids` field and the chain fold.
10. **The chain and header hash construction** itself (§1.2 above) — SHA-256,
    the exact byte layout, endianness, and domain string.
11. **Artifact hashing** — plain `sha256(MODEL.SAF bytes)` /
    `sha256(EMBED.BIN bytes)` / `sha256(VOCAB.BIN bytes)`, whole-file, no
    normative parsing needed for this specific check (compare against the
    receipt's `model`/`embed`/`vocab` fields before anything else —
    `cis_witness.rs:195-212` fails fast here).
12. **Tokenizer encode** — the BPE merge loop over the receipt's declared
    prompt bytes (`tokenizer.rs:169`, VOCAB.BIN parse at `tokenizer.rs:9-97`)
    to get `prompt_ids`, which seed the chain and the FNV digest before any
    generated token.
13. **Container-boundary conversions** (§5.6) as needed by 1–9: `bf16_to_fixed`,
    `f32_to_fixed` (norm gains and quant scales stored as F32 or BF16
    depending on checkpoint), `fix_f32_vec`.

### 2.2 Can skip

- **No SIMD/vector kernels.** `cis_avx2.rs`, `cis_neon.rs`, and the f32
  `ops*.rs`/`inference.rs` families are explicitly out of scope (lib.rs
  gates them behind `target_arch`, `docs/CIS-1_SPEC_v1.0.md:275-296` §6
  states they conform "by matching bits, not by copying loops" — a verifier
  only needs the reference loops, never the fast ones).
- **`CisMode::Hybrid`.** Receipts under discussion are `FullInt`-only (the
  M7 golden and the paper's claim, §7's conformance boundary excludes
  Hybrid). A verifier can refuse (not silently accept) any receipt whose
  header/config implies Hybrid was used, since Hybrid carries no cross-ISA
  claim (spec §7, `docs/CIS-1_SPEC_v1.0.md:297-304`).
- **KV-cache performance structure.** `kvcache.rs`'s actual cache is an
  optimization for the *engine*; a verifier only needs *a* correct
  attention-context accumulation (it can keep the full sequence in a plain
  buffer sized for `max_new + prompt_toks` — no need to replicate ring-buffer
  or eviction logic, since none exists at these context lengths).
- **`arena.rs`, `sampler.rs`, `pool.rs`.** Zero-allocation working-memory
  arena, temperature/top-k sampling, thread pool — engine-only. The receipt
  is always greedy argmax (spec §8, §10 "seeded sampling… not claimed"), so
  no sampler is needed at all.
- **Perplexity/NLL code** (`cis_infer.rs:1178-1251`, `libm::exp`/`libm::log`
  usage) — unrelated to decode/witnessing; this is also where the engine's
  *only* `libm` calls in the FullInt-relevant code live outside Hybrid mode
  (verified by `grep -n libm aegis-core/src/cis_infer.rs`: line 933 is
  Hybrid-only float score scaling under `#[cfg(target_arch = "x86_64")]`;
  lines 1241-1251 are perplexity, not decode). **The `FullInt` decode path
  itself needs zero float, zero `libm`.**
- **`json.rs`'s general-purpose surface.** The verifier only needs the two
  things `model.rs` actually parses: SafeTensors's flat header JSON
  (tensor name → `{dtype, shape, data_offsets}`) and one string field inside
  `__metadata__` (`aegis_config`). It does not need arbitrary JSON.
- **Multi-threading (`parallel` feature, `pool.rs`).** A verifier is
  correctness-only; single-threaded scalar execution is sufficient and
  simpler to audit.

---

## 3. `cis-verify`: dependency set, module layout, CLI

### 3.1 The independence question this design must resolve

RESEARCH_LOOP.md's E4 description says "no engine dependency beyond the spec
ops" — a real constraint, not a phrasing accident. If `cis-verify` simply
`path`-depends on `aegis-core` and calls `CisEngine::forward_step_int`, it
inherits any bug **shared** between the minting engine and the "verifier,"
which defeats exactly the claim §10 flags as open ("independent third-party
audit" — `docs/paper/08_limitations.md`, "clean-room implementers" paragraph)
and that ledger A31 demonstrates is achievable (two from-scratch clean-room
implementations reproduced the Tier-2 digest from spec text alone, no source
access). Two honest options:

- **Option A (fast, weaker claim): vendor, don't depend.** Copy the pure
  spec-op functions (§2.1 items 1–10, roughly `cis.rs` + `cis_attn.rs` +
  the FullInt-only slice of `cis_infer.rs` + `witness.rs`) into `cis-verify`
  as its own source, with no `path = "../aegis-core"` dependency and no
  shared `Cargo.lock` entry. This still shares *code provenance* (same
  author, same commit history) even though it doesn't share a compiled
  crate — good enough to catch a shipped-binary regression or a kernel-path
  bug, not good enough to catch a spec-vs-reference bug baked into both
  copies.
- **Option B (slow, real claim): clean-room from spec text.** A separate
  implementer (ideally an agent session with `docs/CIS-1_SPEC_v1.0.md` as
  its only input, source access forbidden — repeating the A31 methodology)
  writes `cis-verify` from the spec prose alone, checked only against Tier 1
  golden vectors and the Tier 2/3 digest constants, never against
  `aegis-core` source.

**Recommendation:** ship Option A first (it's buildable now, in scope for
this repo's own CI, and immediately gives E4's "someone else's *machine*
verifies" claim); track Option B as a follow-on published as a *separate*
repository/crate by a deliberately source-blind builder, since a
same-repo, same-history crate can never fully carry the "verification
without trust in us" claim E4 states as its purpose. This design's module
layout below supports either path — it's Option A's file boundaries with
the property that every file could be handed as-is to an Option B
implementer as "the module to reimplement," one at a time.

### 3.2 Dependency set

```toml
[package]
name = "cis-verify"
edition = "2024"

[dependencies]
# none, if the CLI itself is written as std but the verification core
# (below) is dependency-free. No serde, no sha2 crate, no clap.
```

Justification, each a deliberate subtraction:

- **No `serde`/`serde_json`.** `aegis-core` already proved a ~150-line
  hand-rolled parser (`aegis-core/src/json.rs`, 414 lines total including
  comments/tests) beats pulling in `serde + serde_json + serde_core +
  memchr` (~27,700 lines, per the comment at `aegis-core/Cargo.toml:9-10`)
  for a format this narrow (flat header JSON, one metadata string). Same
  argument applies harder here: `cis-verify` reads exactly the same two
  shapes and nothing else.
- **No `sha2` crate.** `witness.rs`'s from-scratch SHA-256 is ~110 lines,
  `no_std`, no-alloc, and *is itself part of what's being verified* — the
  chain construction is bespoke (custom domain string, mixed-endian step
  folding), so a generic `sha2` crate would only cover the primitive, not
  the construction, and pulling in a crate whose internals aren't part of
  the audited surface adds an unaudited dependency for no simplification.
- **No `clap`.** Four positional arguments, no subcommands beyond
  `verify`/`gen` (optional — see §3.4), hand-parsed `std::env::args()` as
  `cis_witness.rs` already does (`cis_witness.rs:110-121`).
- **`libm`: not required for the FullInt decode/verify path** (§2.2) — but
  *is* required if `cis-verify` also wants to reproduce the ROPE-I table
  generator's `sincos_q62`/`log2_q32_f32` from scratch without hand-copying
  the isqrt-chain code, since those are pure integer already (§5.7, §5.9 are
  parameter-free integer procedures, no `libm` in `cis_attn.rs` at all per
  `grep -n libm aegis-core/src/cis_attn.rs` returning only the doc-comment
  line, not code). **Net: zero runtime dependencies for the core crate.**
- **`no_std`-capable: yes, for the verification core.** Every function in
  §2.1 items 1–10 is either `core`-only (`cis.rs`, `cis_attn.rs` per their
  own doc comments — "core-only integer arithmetic — no floats, no libm, no
  unsafe") or `alloc`-only (`Vec` for per-layer weight views, tensor offset
  maps). The CLI binary (file I/O, `std::env::args`, stdout) is
  necessarily `std`; the crate should be split so the `no_std+alloc` core
  is a library and `std` lives only in a thin `bin/` wrapper — this is
  what buys "the unikernel verifies receipts with the same bytes of code
  as the Linux host" (`witness.rs:15-17`'s stated design goal) for
  `cis-verify` too, and keeps the door open for a future bare-metal
  verifier appliance (spec §8, "planned, not built").

### 3.3 Module layout

```
cis-verify/
├── Cargo.toml
├── src/
│   ├── lib.rs            # #![no_std] + alloc; re-exports below; CisVerifyError
│   ├── safetensors.rs     # MODEL.SAF header parse: 8-byte LE len, JSON header,
│   │                      # {name -> (dtype, shape, data_offsets)}, __metadata__
│   │                      # lookup. Mirrors model.rs:130-228's shape, independent
│   │                      # code (Option A: transcribed, not imported).
│   ├── json_min.rs        # the narrow JSON reader safetensors.rs needs:
│   │                      # object members, string unescape, u64 pairs.
│   │                      # Mirrors json.rs's scope, ~150 lines.
│   ├── config.rs          # ModelConfig::from_json equivalent — the
│   │                      # aegis_config metadata string -> struct.
│   ├── vocab.rs           # VOCAB.BIN parse (magic 0x564F4341, id_to_string,
│   │                      # merges) + BPE encode(). Mirrors tokenizer.rs.
│   ├── ops.rs             # §2.1 items 2-4, 9, 13: rmsnorm_i, ternary_matvec_i8,
│   │                      # quantize_activations_i32, requant_i32/64, QScale,
│   │                      # QScale64, argmax_i64, bf16_to_fixed, f32_to_fixed.
│   │                      # Mirrors cis.rs + the non-attention half of
│   │                      # cis_infer.rs.
│   ├── attn.rs            # §2.1 items 5-7: exp2_chain/ExpLut, softmax_i,
│   │                      # RopeTableI/rope_apply_i, sincos_q62, relu2_q20,
│   │                      # silu_q20. Mirrors cis_attn.rs.
│   ├── forward.rs         # the glue: one decode step, FullInt only —
│   │                      # embedding lookup -> N decoder layers -> final
│   │                      # norm -> LM head dot -> argmax. Mirrors
│   │                      # cis_infer.rs's CisModel/CisEngine but FullInt-
│   │                      # only (no Hybrid branch to skip auditing).
│   ├── sha256.rs           # transcribed from witness.rs:22-132, byte-identical
│   │                      # algorithm, pinned to the same FIPS vectors.
│   ├── witness.rs          # WitnessHeader/WitnessChain — transcribed from
│   │                      # witness.rs:141-210, must reproduce the exact
│   │                      # domain string, field order, and endianness of
│   │                      # §1.2.
│   ├── receipt.rs          # receipt text parse/format — transcribed from
│   │                      # cis_witness.rs:149-193's KV grammar.
│   └── verify.rs           # orchestration: load 3 artifacts -> hash check ->
│                          # parse receipt -> replay -> compare fields ->
│                          # VerifyOutcome.
└── src/bin/
    └── cis-verify.rs      # std: argv, file reads, prints, exit code.
```

Each module above is a candidate unit for Option B's clean-room
substitution — a builder can be handed `docs/CIS-1_SPEC_v1.0.md` §5.4 alone
and asked to produce `ops.rs`'s `rmsnorm_i`, checked only against the
Tier-1 golden vectors, with zero visibility into `aegis-core/src/cis.rs`.

### 3.4 CLI

```
cis-verify <receipt> <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN>
```

Illustrative signature only (no implementation):

```rust
enum VerifyOutcome {
    Pass { steps: u64 },
    Fail { field: FailedField, detail: String },
}

enum FailedField {
    ArtifactHash { which: &'static str },   // "MODEL" | "EMBED" | "VOCAB"
    PromptTokenize,                          // encode(prompt) != prompt-toks
    TokenId { step: usize },                 // first divergent generated id
    CisDigest,                               // FNV fold mismatch
    Chain,                                    // SHA-256 chain mismatch
    ReceiptParse { line: usize },            // malformed receipt line
}

fn verify(
    receipt_text: &str,
    model_bytes: &[u8],
    embed_bytes: &[u8],
    vocab_bytes: &[u8],
) -> VerifyOutcome;
```

Output contract (mirrors `cis_witness.rs`'s existing PASS/FAIL prose so
existing logs/tooling that grep for it keep working):

```
$ cis-verify witness_v1_m7_once64.receipt MODEL.SAF EMBED.BIN VOCAB.BIN
VERIFY PASS — replay reproduced 64 tokens, the token digest, and the full
logit chain bit-for-bit
$ echo $?
0
```

```
$ cis-verify tampered.receipt MODEL.SAF EMBED.BIN VOCAB.BIN
FAIL field=chain receipt=aee25b77… local=9f2c1a08…
token divergence at generated index None
VERIFY FAIL — replay diverged from the receipt
$ echo $?
1
```

Field order of checks (fail fast, cheapest first — same order
`cis_witness.rs:195-238` already uses): artifact hashes → full replay →
token-id sequence → FNV digest → chain digest. Exit code 0 iff PASS; every
failure path names the first field that diverged (per-field naming is new
relative to `cis_witness.rs`, which currently reports `local_fnv`/
`local_chain` together — worth the small addition since "which field
failed" is explicitly asked for by a third party debugging a mismatch).

No `--gen` mode is required for E4 (verification, not minting) but the same
`forward.rs`/`witness.rs` core trivially supports one, matching
`cis_witness.rs gen`'s shape, if a follow-on wants `cis-verify` to also
independently mint receipts for cross-checking against this repo's engine.

---

## 4. Threat model — what a receipt proves and doesn't

Consistent with `docs/paper/08_limitations.md`'s "What the receipt does not
do" paragraph and `docs/paper/05_receipt.md`. `cis-verify` is designed to
make exactly this claim mechanically checkable, no more:

**A PASS proves:**
- A CIS-1-conforming computation, run over *these exact bytes* of
  MODEL.SAF/EMBED.BIN/VOCAB.BIN and *this exact prompt*, deterministically
  produces *these exact 64 token ids* and *this exact sequence of full
  logit vectors* — because the verifier independently recomputed all of it
  from the spec, not from trusting the receipt's claims.
- The claim is portable: any machine, any ISA, any (conforming) kernel path
  reproduces the same PASS, because CIS-1's ops are defined to be
  order-invariant integer arithmetic (spec §1 axiom) — this is what A32
  already demonstrated for the reference engine across x86-64/aarch64 CI,
  and physical iron (A33/A34); `cis-verify` extends the *verifying* side of
  that claim to a party who never ran the minting engine at all.
- Tampering with any single field is detected: the chain test suite
  (`aegis-core/tests/witness_contract.rs:104-140`) demonstrates a one-bit
  logit flip, a token-id substitution, or step reordering each changes the
  final digest; a verifier reproducing the same construction inherits this
  sensitivity for free.

**A PASS does NOT prove (matches `08_limitations.md` verbatim):**
- **Which physical machine ran it.** No CPUID, no attestation, no
  timestamp is in the receipt (`witness.rs`'s header commits only artifact
  hashes, `max_new`, and the prompt — nothing hardware-identifying). This
  is `08_limitations.md`'s explicit statement and is unaffected by adding
  `cis-verify`: verification proves *what was computed*, not *where*.
  Platform attestation (TPM/measured boot) is spec §10, "separate layer."
- **Model confidentiality or prompt privacy.** The receipt is not a
  zero-knowledge proof; both MODEL.SAF and the prompt must be handed to the
  verifier in full to replay. A receipt is a disclosure mechanism, not a
  privacy one.
- **Anything if the artifacts and the verifier are both controlled by the
  same adversary.** If someone hands `cis-verify` a MODEL.SAF that isn't
  the model they claim it is (but is internally self-consistent with a
  receipt they also fabricated by actually running that substituted
  model), `cis-verify` will PASS — correctly, because it verified exactly
  what it was given. Binding "this SHA-256" to "the model claimed in
  marketing" is an out-of-band problem (the paper's honesty; not one a
  hash-chain can solve).
- **Non-greedy generation, or any decoding beyond argmax.** Spec §10:
  seeded sampling has no reference implementation and is out of scope; a
  receipt for temperature/top-k sampling does not exist in v1.
  `cis-verify` should reject (not silently accept) a receipt whose header
  claims are inconsistent with greedy-only decode.
- **Logit-level transcript beyond what's chained.** Spec §8 states plainly
  that Tier 3 (and by extension the witness chain, which absorbs the same
  logits) is "not... per-step logits" as a *published, human-inspectable*
  transcript — the logits are folded into a hash, never revealed. A
  verifier can confirm a chain matches; it cannot show a third party *what
  the logits were* without replaying itself. (This differs from spec §8's
  Tier 3 digest, which the witness chain generalizes — the witness receipt
  is strictly the stronger artifact of the two already, so this is a
  clarification, not a gap.)
- **2B/production-scale conformance** (until E1/E3 land) — the golden
  receipt this design tests against is the M7 (small) model; `cis-verify`
  itself is model-size-agnostic (nothing in §2.1 is model-scale-specific),
  but *evidence* that it works at 2B scale is a separate, later claim
  (spec §10, "2B full-integer conformance artifacts" — not yet measured).
- **The −128 kernel hazard, the SIMD-kernel equality suites, or anything
  about vector-kernel correctness** — spec §8 states outright that no
  conformance tier exercises the −128 activation case and that kernel
  equality is "a test bar, deliberately not called proof" (spec §6.3). A
  verifier that only ever runs scalar reference ops (§2.2) says nothing
  about whether some other machine's AVX2/NEON kernel would have agreed —
  it only confirms what the *reference semantics* compute, which is what a
  receipt claims (§7's scope) but is a narrower statement than "this
  specific optimized binary is bug-free."

---

## 5. Test plan

Reuses the golden receipt and the two pinned digests; adds no code, no
tests, and no files under `tests/golden/` or `docs/hardware_logs/` (both
read-only for this task per Rule C).

1. **Golden-receipt round-trip (primary conformance test).**
   `cis-verify tests/golden/witness_v1_m7_once64.receipt <M7 MODEL.SAF>
   <M7 EMBED.BIN> <M7 VOCAB.BIN>` against the in-repo M7 artifacts
   (`model-lab/tinybit/m7_final_gate_work/artifacts/`, per spec §8 Tier 3's
   citation) must print `VERIFY PASS` and exit 0. This is the crate's one
   must-pass acceptance test; everything else is regression coverage around
   it.
2. **Tier 2 digest cross-check.** Because `cis-verify`'s `ops.rs`/`attn.rs`
   reimplement the same reference ops spec §8 Tier 2 exercises
   (`cis_selftest`'s A1–A8 goldens, B1–B6 sweeps), a `cis-verify`-internal
   `selftest` build target should independently fold the same 14 sections
   into one FNV-1a 64 digest and assert it equals `76985613c965f643` (spec
   §8, `docs/CIS-1_SPEC_v1.0.md:333`). This is the cheapest possible signal
   that the reimplementation (Option A transcription or Option B clean-room)
   didn't silently drift from the reference on any individual op *before*
   spending a full decode replay to find out.
3. **Tier 3 digest cross-check.** Independent of the witness chain, replay
   the M7 artifacts' 64-token greedy decode and assert the FNV-1a 64 token
   digest equals `67e8c0a96abc04e1` (spec §8, line 343) — this exercises
   §2.1 items 1–9, 12–13 without needing the receipt file or the SHA-256
   chain at all, isolating "did the forward pass reproduce" from "did the
   chain construction match."
4. **Negative tests — one flip per field, expect the named failure.**
   Using the golden receipt as a base, construct (in a scratch/test-only
   copy, never touching `tests/golden/`) receipts with: (a) one changed hex
   digit in `model`/`embed`/`vocab` → expect `FailedField::ArtifactHash`;
   (b) one substituted `token-ids` entry → expect `FailedField::TokenId` at
   the correct index, chain mismatch too; (c) `cis-digest` corrupted but
   everything else untouched → expect `FailedField::CisDigest`; (d) `chain`
   corrupted alone → expect `FailedField::Chain`; (e) `maxtok` changed
   without regenerating → expect a header-hash-driven full chain mismatch
   from step 0 (header binds `max_new`, `witness.rs:150,164`, so this is
   the "genesis changes" case, distinct from (b)'s "one step changes"
   case). Mirrors the sensitivity properties already pinned in
   `witness_contract.rs:104-140`, but exercised through the file format
   and the standalone crate rather than through direct API calls.
5. **Malformed-receipt robustness.** Truncated file, extra unknown key
   (must be ignored, matching `cis_witness.rs:191`'s `_ => {}`), a
   `token-ids` list shorter than `gen-toks` claims, non-hex characters in a
   hash field — each should produce a clean `ReceiptParse` failure, never a
   panic (bare-metal verifier future target has no unwind).
6. **Cross-implementation parity with `cis_witness verify`.** For every
   case in (1)-(5), run both `cis-verify` and the existing
   `aegis-linux/examples/cis_witness.rs verify` and assert they agree on
   PASS/FAIL (not necessarily on the exact prose). This is the test that
   actually exercises "two independent implementations of the same spec
   agree," which is the entire point of E4; a disagreement here is a
   finding (either implementation could be the one that's wrong) and
   should block release, not be quietly resolved by copying the other's
   answer.
7. **SHA-256 primitive.** Transcribe `witness_contract.rs`'s FIPS vector
   tests (lines 16-73) unchanged into `cis-verify`'s own test suite —
   catches a transcription bug in the algorithm itself, independent of the
   witness-chain construction around it.
8. **`no_std` build check.** `cargo build -p cis-verify --no-default-features
   --target x86_64-unknown-none` (or equivalent) for the library crate,
   confirming §3.2's `no_std+alloc` claim holds as code is added, not just
   at design time.

None of these require new files under `tests/golden/` — (1)-(3) consume the
existing golden receipt and the spec's own pinned constants (already public
in `docs/CIS-1_SPEC_v1.0.md`); (4)-(5)'s tampered fixtures live under
`cis-verify/tests/fixtures/`, a new, ordinary (non-append-only) test
directory scoped to the new crate.

---

## 6. Code size estimate and work breakdown

### 6.1 Size estimate

Based on the mirrored originals' actual line counts (Option A transcription
scope — Option B clean-room work is comparable in scope but not
directly estimable from existing files):

| module | mirrors | source lines | estimate |
|---|---|---|---|
| `sha256.rs` | `witness.rs:22-132` | 111 | ~120 |
| `witness.rs` | `witness.rs:141-223` | 83 | ~90 |
| `ops.rs` | `cis.rs` (628 total, minus doc comments/tests) | ~450 | ~350 |
| `attn.rs` | `cis_attn.rs` (707 total) | ~550 | ~450 |
| `forward.rs` | FullInt-only slice of `cis_infer.rs` (1568 total, Hybrid ~40% of it) | ~600 | ~500 |
| `safetensors.rs` + `json_min.rs` | `model.rs:130-235` + `json.rs` (414) | ~520 | ~450 |
| `config.rs` | `model.rs:35-127` | ~90 | ~100 |
| `vocab.rs` | `tokenizer.rs` (258 total) | 258 | ~250 |
| `receipt.rs` | `cis_witness.rs:149-193` | ~45 | ~150 (parser needs to be more defensive than the reference's happy-path `unwrap`s, per §5's malformed-input tests) |
| `verify.rs` | `cis_witness.rs:195-256` (verify-mode logic) | ~60 | ~150 (per-field `FailedField` reporting is new) |
| `bin/cis-verify.rs` | `cis_witness.rs:109-121` (arg handling) | ~15 | ~60 |
| tests (fixtures + cases from §5) | — | — | ~400 |
| **total** | | | **~2,900 lines** |

For scale: this is smaller than `aegis-core/src/json.rs` (414) +
`cis_infer.rs` (1568) + `model.rs` (438) + `tokenizer.rs` (258) +
`cis.rs` (628) + `cis_attn.rs` (707) + `witness.rs` (223) combined
(~4,236 lines) — expected, since `cis-verify` sheds Hybrid mode, KV-cache
tuning, `act_stats`/`legacy_matmul` feature variants, and the
production-engine ergonomics (`std` file-loading conveniences,
multi-threading) that `aegis-core` carries for reasons unrelated to
verification.

### 6.2 Work breakdown (≤6 tasks for a builder agent)

Ordered so each task is independently testable against a spec section or
existing golden constant before the next depends on it — a builder can stop
after any task and have something that passes its own listed check.

1. **SHA-256 + witness chain construction** (`sha256.rs`, `witness.rs`,
   `receipt.rs` parse/format). No model/tensor code at all yet.
   *Check:* reproduce `witness_contract.rs`'s FIPS vectors, and reproduce
   `header.hash()`/`chain.digest()` byte-for-byte on the test vectors
   already in `witness_contract.rs:92-152` (same inputs, same expected
   relationships — chain determinism, single-logit-flip sensitivity,
   token-id-change sensitivity, step-reordering sensitivity, header binds
   every field). This task alone validates the entire cryptographic
   construction (§1.2) with zero dependency on model parsing.
2. **Artifact parsing: SafeTensors + minimal JSON + VOCAB.BIN + BPE encode**
   (`safetensors.rs`, `json_min.rs`, `config.rs`, `vocab.rs`).
   *Check:* parse the actual M7 `MODEL.SAF`/`VOCAB.BIN` in-repo artifacts,
   extract `aegis_config`, confirm `hidden_size`/`num_hidden_layers`/etc.
   match what `ModelConfig::from_json` would report (cross-check against
   `aegis-core`'s own parse as a one-time bootstrap sanity check, not a
   runtime dependency); tokenize `"Once upon a time"` and confirm 4 token
   ids matching the golden receipt's `prompt-toks 4`.
3. **Reference integer ops** (`ops.rs`: RMSNORM-I, TMV, QUANT-ACT, REQUANT,
   QScale/QScale64, ARGMAX, container-boundary conversions).
   *Check:* Tier 1 — reproduce the `cis.rs` unit golden vectors (spec §8
   Tier 1's citation, `scripts/cis_e2_golden_gen.py`-generated) for each op
   in isolation.
4. **Integer attention/activation ops** (`attn.rs`: exp-LUT, SOFTMAX-I,
   ROPE-I table generation + rotation, ACT-I).
   *Check:* reproduce the two normative digests spec §5.7/§5.9 pin
   (`0x66C2A0EEB8C2DC43` exp-LUT, `0xD8345EBF01E990FA` M7-shape RoPE table)
   — both are self-contained (no model artifacts needed, pure constant
   generation), making this the cheapest possible high-confidence check on
   the hardest arithmetic in the crate.
5. **Forward pass glue + Tier 2/3 digests** (`forward.rs`: embedding lookup
   through LM head + argmax, one decode step, `FullInt` only).
   *Check:* independently reproduce `CIS_SELFTEST digest=76985613c965f643`
   (Tier 2) and `CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4
   gen_toks=64 mode=fullint` (Tier 3) against the M7 artifacts — this is
   the task that proves tasks 2–4 compose correctly into a real decode,
   before the receipt file enters the picture at all.
6. **`verify.rs` orchestration + CLI + negative-test suite** (artifact-hash
   fast-fail, full replay, field-by-field comparison against the parsed
   receipt, `FailedField` reporting; `bin/cis-verify.rs`; §5's tampered
   fixtures 4(a)-(e) and malformed-input cases).
   *Check:* the golden-receipt round-trip (§5 item 1) PASSes; all of §5's
   negative tests report the correct `FailedField`; `cis-verify` and
   `cis_witness verify` agree on every case (§5 item 6) — this is the
   task that closes the loop and makes E4's "no engine dependency beyond
   the spec ops... someone else's machine verifies without us" claim
   checkable end to end.

Tasks 1–4 have no dependency ordering among themselves beyond "before 5";
a builder agent (or several in parallel) could take 1, 2, 3, and 4
concurrently, with 5 and 6 as the two integration passes.
