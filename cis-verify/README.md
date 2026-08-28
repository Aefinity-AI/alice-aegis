# cis-verify

Standalone, third-party verifier for **CIS-1 decode receipts** — the
`AEGIS-WITNESS v1-CIS` format produced by the A.L.I.C.E. project's ternary
transformer inference engine. This crate reimplements the entire check —
SHA-256, FNV-1a 64, the witness hash-chain, receipt parsing, artifact
hashing, `MODEL.SAF`/`VOCAB.BIN` parsing, the tokenizer, and the reference
integer forward pass — from the [CIS-1 spec](../docs/CIS-1_SPEC_v1.0.md)
alone, with **zero runtime dependencies** and **zero shared code** with the
engine it checks (`aegis-core`). `no_std` + `alloc` for the verification
core; a small `std`-gated CLI binary adds file I/O.

Design background: [`docs/design/CIS_VERIFY_DESIGN.md`](../docs/design/CIS_VERIFY_DESIGN.md).

## What a receipt is

A receipt is a plain-text, line-oriented file that binds one decode run to
its exact inputs and outputs. It carries:

- SHA-256 hashes of the three model artifacts (`MODEL.SAF`, `EMBED.BIN`,
  `VOCAB.BIN`) it was run against,
- the prompt (hex-encoded) and its token count,
- every generated token id, in order,
- a `cis-digest` — one FNV-1a 64 fold over the prompt and generated token
  ids (the same constant CIS-1 §8 Tier 3 pins), and
- a `chain` — a SHA-256 hash-chain digest over the whole run.

It is produced once, by the reference implementation, at decode time. This
crate never produces one — it only re-derives everything in it from the
receipt's own artifact hashes and prompt, and checks that the result
matches, field by field.

## What `VERIFY PASS` proves — and does not

**What it proves.** A receipt that verifies proves that a conforming
computation over the bound artifacts produced the bound outputs: given the
exact `MODEL.SAF` / `EMBED.BIN` / `VOCAB.BIN` bytes hashed in the receipt
and the exact prompt, re-running the CIS-1 §5 reference integer ops
end-to-end reproduces every one of the 64 generated token ids and the two
folded digests, bit-for-bit. Because this crate shares no code, no data
structures, and no SIMD/dispatch path with `aegis-core`, a pass is evidence
from a second, independent implementation of the same spec — not a replay
of the engine's own arithmetic.

**What it does not prove.**

- It does not prove *which physical machine* ran the original decode —
  that is platform attestation's job, a separate layer this format does
  not attempt.
- It does not hide the model or the prompt — this is not a zero-knowledge
  proof; the artifacts and prompt are hashed, not concealed.
- It does not protect against an adversary who controls both the artifacts
  and the verifier together — verification establishes that inputs and
  outputs match under CIS-1 semantics, not that the inputs are themselves
  trustworthy.
- This crate itself is a spec transcription, not a clean-room audit: it was
  built with the CIS-1 spec and the receipt format as reference material,
  not by a party working blind to the reference implementation's source.
  That makes it evidence that the spec is independently re-implementable —
  it is not the same claim as an external auditor's clean-room review, and
  it should not be described as one.

(This mirrors the "What the receipt does not do" paragraph of
[`docs/paper/08_limitations.md`](../docs/paper/08_limitations.md).)

## Install

Once published to crates.io:

```bash
cargo install cis-verify --features std
```

For now, run it straight out of this checkout:

```bash
cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- \
    <receipt> <MODEL.SAF> <EMBED.BIN> <VOCAB.BIN>
```

Exit code is `0` on `VERIFY PASS`, `1` on `VERIFY FAIL (<field>)`, `2` on a
usage/IO error.

## The two pinned digests

CIS-1 §8 freezes two conformance digests that any correct implementation
must reproduce exactly. `cis-verify` reproduces both, independently:

| Tier | What it covers | Pinned digest |
|---|---|---|
| Tier 2 — operation-level | 14 sections of golden vectors and deterministic sweeps over the §5 integer ops (`rne_div`, `REQUANT`, `TMV`, `QUANT-ACT`, `RMSNORM-I`, `ARGMAX`), folded into one FNV-1a 64 digest | `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true` |
| Tier 3 — token-level | Greedy decode of the prompt `"Once upon a time"` for 64 new tokens on the in-repo M7 artifacts, `FullInt` mode; one FNV-1a 64 digest over every token id (prompt included) | `CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4 gen_toks=64 mode=fullint` |

Both digests are re-proven on every commit in CI, on x86-64 and aarch64
(`arm-digest.yml`).

## Golden-receipt example

The golden receipt at
[`tests/fixtures/witness_v1_m7_once64.receipt`](tests/fixtures/witness_v1_m7_once64.receipt)
(1 KB) is the M7 receipt referenced by CIS-1 §8 Tier 3 and ledger row A32.
Verifying it against the in-repo M7 artifacts:

```bash
$ cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- \
    cis-verify/tests/fixtures/witness_v1_m7_once64.receipt \
    model-lab/tinybit/m7_final_gate_work/artifacts/MODEL.SAF \
    model-lab/tinybit/m7_final_gate_work/artifacts/EMBED.BIN \
    model-lab/tinybit/m7_final_gate_work/artifacts/VOCAB.BIN

cis-verify: receipt=cis-verify/tests/fixtures/witness_v1_m7_once64.receipt
cis-verify: MODEL.SAF=model-lab/tinybit/m7_final_gate_work/artifacts/MODEL.SAF (2797632 bytes)
cis-verify: EMBED.BIN=model-lab/tinybit/m7_final_gate_work/artifacts/EMBED.BIN (6291456 bytes)
cis-verify: VOCAB.BIN=model-lab/tinybit/m7_final_gate_work/artifacts/VOCAB.BIN (163954 bytes)
check: receipt parse ......... ok
check: artifact hashes ........ ok
check: prompt tokenization ..... ok
check: token-id sequence (64 steps) ok
check: cis-digest (FNV-1a 64) .. ok
check: witness chain (SHA-256) . ok
VERIFY PASS
```

## Tamper examples

A single flipped byte in an artifact fails naming the artifact, not a
generic checksum error:

```bash
$ cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- \
    cis-verify/tests/fixtures/witness_v1_m7_once64.receipt \
    model-lab/tinybit/m7_final_gate_work/artifacts/MODEL.SAF \
    model-lab/tinybit/m7_final_gate_work/artifacts/EMBED.BIN \
    /tmp/VOCAB_TAMPERED.BIN
...
check failed: vocab (receipt 5a1d79caf517084a73de5e5379d2995e61f23afaada91822da12ecb4ad7fcd8a local ae269eaedf3bc65f216642781108a1acac01fe36ee51931351d432e348358401)
VERIFY FAIL (vocab)
```

A single tampered generated token id fails naming the step, not a downstream
digest mismatch:

```bash
$ # receipt with "token-ids 12," rewritten to "token-ids 13,"
$ cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- \
    /tmp/tampered_token.receipt \
    model-lab/tinybit/m7_final_gate_work/artifacts/MODEL.SAF \
    model-lab/tinybit/m7_final_gate_work/artifacts/EMBED.BIN \
    model-lab/tinybit/m7_final_gate_work/artifacts/VOCAB.BIN
...
check failed: token-id (local[0]=12 receipt[0]=13)
VERIFY FAIL (token-id)
```

`tests/verify_golden.rs` runs these (and a tampered-chain, and a
truncated-receipt case) as negative tests whenever the in-repo M7 artifacts
are present; it skips them, rather than failing, in a checkout that doesn't
have `model-lab/` (such as a standalone `cargo install` of this crate).

## `no_std`

The verification core (`sha256`, `fnv`, `witness`, `receipt`, `artifact`,
`safetensors`, `json_min`, `config`, `vocab`, `forward`, `ops`, `attn`,
`verify`) is `#![no_std]` + `alloc` unconditionally — it has no dependency
on an OS, an allocator choice, or the `std` feature. The `std` feature gates
only `src/bin/cis-verify.rs` (argv, file I/O, `println!`) and
`examples/cis_decode.rs`; everything else builds and runs identically on
bare metal. (The A.L.I.C.E. unikernel itself calls the same verification
core with no OS underneath it — see ledger row A40.)

## Provenance

This crate's own correctness claims are logged the same way every other
number in this project is — with a ledger row and a raw log, not just
prose:

- **A37** — first standalone reproduction: both pinned digests (Tier 2
  `CIS_SELFTEST`, Tier 3 `CIS_DECODE`), `VERIFY PASS` on the M7 golden
  receipt (all six checks), and tamper tests failing by naming the correct
  field.
- **A38** — the same reproduction crosses the ISA boundary in public CI
  (aarch64 and x86-64 GitHub runners), now a standing CI gate.
- **A39** — a production-scale (2B-parameter, BitNet-2B) decode receipt
  verifies bit-for-bit on both ISAs, by both the reference verifier and
  this crate.
- **A40** — the 2B receipt is re-derived by the A.L.I.C.E. unikernel itself,
  booted with no OS underneath, under QEMU/TCG (identity evidence only —
  no performance claim; see Rule A in the repo's `CLAUDE.md`).

Full rows: [`program/RESEARCH_LEDGER.md`](../program/RESEARCH_LEDGER.md)
(A37–A40). Normative spec: [`docs/CIS-1_SPEC_v1.0.md`](../docs/CIS-1_SPEC_v1.0.md)
(v1.0.3 — frozen 2026-08-07, v1.0.1/v1.0.2 errata; both conformance digests
unchanged since freeze).
