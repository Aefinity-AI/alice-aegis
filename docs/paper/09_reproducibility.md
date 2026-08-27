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

**Commands** (quoted verbatim from `.github/workflows/arm-digest.yml`, which
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

**Append-only evidence (Rule C).** `tests/golden/` and `docs/hardware_logs/`
are append-only: every figure in this paper traces to a file under one of
these two paths, and no existing file under either is ever edited, only
added to. Both directories are shipped with the paper's release artifact so
a reader can check a claim against its raw log without re-running anything,
and can re-run the commands above to check it against live hardware anyway.
