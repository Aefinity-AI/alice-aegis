# 9. Reproducibility statement

**Repository.** `https://github.com/Aefinity-AI/alice-aegis`. The spec is
`docs/CIS-1_SPEC_v1.0.md`; the research ledger with every measurement's raw
log path is `program/RESEARCH_LEDGER.md`.

**CI workflows.** `.github/workflows/arm-digest.yml` re-proves both
conformance digests and the decode receipt on x86-64 and aarch64 on every
push — the standing gate behind every identity claim in §4–5. This is a
correctness/identity gate only (Rule A): no timing figure is ever recorded
from it. `.github/workflows/aefinity-ci.yml` is the general build/format/lint
gate (host job) plus an OVMF boot-correctness job for the unikernel.

**Golden receipt.** `tests/golden/witness_v1_m7_once64.receipt`, minted on
the i5-10210U crosvm dev host and verified bit-for-bit on aarch64 in public
CI (run 31249589879, snapshot `ce93bbb`, A32).

**The two conformance digests.**
- Op-level: `CIS_SELFTEST digest=76985613c965f643 ALL_PASS=true`
- Token-level decode: `CIS_DECODE digest=67e8c0a96abc04e1 prompt_toks=4 gen_toks=64 mode=fullint`

**Commands** (the invocation cores of `.github/workflows/arm-digest.yml`; the workflow additionally pipes each output through `grep '^CIS_…' | tee /dev/stderr | grep -q <digest> || exit 1` so a mismatch fails the job, and re-declares the artifact directory `M` before the witness step — see the file for the exact wrappers — which
runs them on every push on both `ubuntu-24.04` and `ubuntu-24.04-arm`):

```
cargo build --release --example cis_selftest --manifest-path aegis-linux/Cargo.toml
./aegis-linux/target/release/examples/cis_selftest

cargo build --release --example cis_decode --manifest-path aegis-linux/Cargo.toml
M=model-lab/tinybit/m7_final_gate_work/artifacts
./aegis-linux/target/release/examples/cis_decode "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" 64 "Once upon a time"

cargo build --release --example cis_witness --manifest-path aegis-linux/Cargo.toml
./aegis-linux/target/release/examples/cis_witness verify \
  "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN" \
  tests/golden/witness_v1_m7_once64.receipt
```

The first two lines reproduce the op-level digest; the next two, the
token-level digest against the in-repo M7 model; the last two replay and
verify the x86-minted golden receipt. A mismatch in any digest is a
falsification of the corresponding claim in §4–5, not a bug to be quietly
fixed — the workflow says so at the point it would fail.

**Standalone verifier (`cis-verify`).** A separate crate at `cis-verify/`, with zero external
runtime dependencies and no dependency on `aegis-core` (§5, A37):

```
cargo test --features std --manifest-path cis-verify/Cargo.toml -- --nocapture

cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- \
  tests/golden/witness_v1_m7_once64.receipt \
  "$M/MODEL.SAF" "$M/EMBED.BIN" "$M/VOCAB.BIN"
```

The first command runs the crate's test suite, including the pinned op-level digest, both pinned
table digests, the token-level decode digest, and the golden-receipt verification and tamper
tests. The second runs the CLI directly against the golden receipt and prints `VERIFY PASS` (or
`VERIFY FAIL (<field>)`) in about 1.4 seconds.

**BitNet-2B cross-ISA CI (`bitnet2b-receipt.yml`, A38, A39).** The BitNet-2B artifacts (~745 MB
combined) are too large to commit; they are attached as release assets on tag
`artifacts-bitnet2b-2026-08-27` and downloaded fresh by every job with
`gh release download artifacts-bitnet2b-2026-08-27`, then hash-checked against the pinned
SHA-256s before use. `.github/workflows/bitnet2b-receipt.yml` runs on every push to `main`, on
both `ubuntu-24.04` and `ubuntu-24.04-arm`:

```
cargo build --release --example cis_decode  --manifest-path aegis-linux/Cargo.toml
cargo build --release --example cis_witness --manifest-path aegis-linux/Cargo.toml
M=2b-artifacts
./aegis-linux/target/release/examples/cis_decode "$M/aegis_pruned_model.cis.safetensors" "$M/embed.bin" "$M/vocab.bin" 64 "Once upon a time"

./aegis-linux/target/release/examples/cis_witness verify \
  "$M/aegis_pruned_model.cis.safetensors" "$M/embed.bin" "$M/vocab.bin" \
  tests/golden/witness_v1_bitnet2b_once64.receipt

cargo run --release --features std --manifest-path cis-verify/Cargo.toml --bin cis-verify -- \
  tests/golden/witness_v1_bitnet2b_once64.receipt \
  "$M/aegis_pruned_model.cis.safetensors" "$M/embed.bin" "$M/vocab.bin"
```

The first two lines build the reference drivers; `cis_decode` reproduces the BitNet-2B token-level
digest (`cab11400d737ac4a`); `cis_witness verify` and the standalone `cis-verify` CLI each
independently verify the same golden receipt. The aarch64 job running the same four commands is
the source for A38 (the standalone verifier's aarch64 leg) and A39 (the receipt's cross-ISA
verification at 2B scale); the x86-64 job in the same run prints the identical lines.

**Append-only evidence (Rule C).** `tests/golden/` and `docs/hardware_logs/`
are append-only: every figure in this paper traces to a file under one of
these two paths, and no existing file under either is ever edited, only
added to. Both directories are shipped with the paper's release artifact so
a reader can check a claim against its raw log without re-running anything,
and can re-run the commands above to check it against live hardware anyway.
